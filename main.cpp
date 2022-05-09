#include "vulkan/vulkan.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <malloc.h>
#include <chrono>

#define VKB_VALIDATION_LAYERS
// Generates predetermined random 32 bit numbers
#define znew   (z=36969*(z&65535)+(z>>16))
#define wnew   (w=18000*(w&65535)+(w>>16))
#define MWC    ((znew<<16)+wnew )
static unsigned long z = 362436069, w = 521288629;

#define BAIL_ON_BAD_RESULT(result) \
  if (VK_SUCCESS != (result)) { fprintf(stderr, "Failure at %u %s\n", __LINE__, __FILE__); exit(result); }

#define INPUT_LENGTH 10000000
#define MAX_WG_NUMBER 65535

// Remember when changing this to also change it in the shader.
// This is dependent on VkPhysicalDeviceLimits::maxComputeWorkGroupInvocations
#define WG_SIZE 1024

#define RADIX_BITS 4

typedef struct PushConsts {
    uint32_t inputLength;
    uint32_t sumArrLength;
    uint32_t startBit;
    uint32_t elementsPerWI;
} PushConsts;

bool readShaderFile(const std::string &filename, std::vector<char> &fileContent) {
  std::ifstream shaderFile(filename, std::ios::ate | std::ios::binary);

  if (!shaderFile.is_open()) {
    std::cerr << "Failed to open shader file " << filename << std::endl;
    return false;
  }
  size_t fileSize = static_cast<size_t>(shaderFile.tellg());
  fileContent.resize(fileSize);
  shaderFile.seekg(0);
  shaderFile.read(fileContent.data(), fileSize);
  shaderFile.close();

  return true;
}

bool checkSorted(uint32_t *array, size_t count) {
  size_t previous = 0;
  for (size_t i = 0; i < count; i++) {
    if (previous == i) {
      continue;
    }
    if (array[previous] > array[i]) {
      return false;
    }
    previous = i;
  }
  return true;
}

VkResult vkGetBestTransferQueueNPH(VkPhysicalDevice physicalDevice, uint32_t *queueFamilyIndex) {
  uint32_t queueFamilyPropertiesCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, 0);

  VkQueueFamilyProperties *const queueFamilyProperties = (VkQueueFamilyProperties *) _malloca(
      sizeof(VkQueueFamilyProperties) * queueFamilyPropertiesCount);

  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, queueFamilyProperties);

  // first try and find a queue that has just the transfer bit set
  for (uint32_t i = 0; i < queueFamilyPropertiesCount; i++) {
    // mask out the sparse binding bit that we aren't caring about (yet!)
    const VkQueueFlags maskedFlags = (~VK_QUEUE_SPARSE_BINDING_BIT & queueFamilyProperties[i].queueFlags);

    if (!((VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT) & maskedFlags) &&
        (VK_QUEUE_TRANSFER_BIT & maskedFlags)) {
      *queueFamilyIndex = i;
      return VK_SUCCESS;
    }
  }

  // otherwise we'll prefer using a compute-only queue,
  // remember that having compute on the queue implicitly enables transfer!
  for (uint32_t i = 0; i < queueFamilyPropertiesCount; i++) {
    // mask out the sparse binding bit that we aren't caring about (yet!)
    const VkQueueFlags maskedFlags = (~VK_QUEUE_SPARSE_BINDING_BIT & queueFamilyProperties[i].queueFlags);

    if (!(VK_QUEUE_GRAPHICS_BIT & maskedFlags) && (VK_QUEUE_COMPUTE_BIT & maskedFlags)) {
      *queueFamilyIndex = i;
      return VK_SUCCESS;
    }
  }

  // lastly get any queue that'll work for us (graphics, compute or transfer bit set)
  for (uint32_t i = 0; i < queueFamilyPropertiesCount; i++) {
    // mask out the sparse binding bit that we aren't caring about (yet!)
    const VkQueueFlags maskedFlags = (~VK_QUEUE_SPARSE_BINDING_BIT & queueFamilyProperties[i].queueFlags);

    if ((VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT) & maskedFlags) {
      *queueFamilyIndex = i;
      return VK_SUCCESS;
    }
  }

  return VK_ERROR_INITIALIZATION_FAILED;
}

VkResult vkGetBestComputeQueueNPH(VkPhysicalDevice physicalDevice, uint32_t *queueFamilyIndex) {
  uint32_t queueFamilyPropertiesCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, 0);

  VkQueueFamilyProperties *const queueFamilyProperties = (VkQueueFamilyProperties *) _malloca(
      sizeof(VkQueueFamilyProperties) * queueFamilyPropertiesCount);

  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, queueFamilyProperties);

  // first try and find a queue that has just the compute bit set
  for (uint32_t i = 0; i < queueFamilyPropertiesCount; i++) {
    // mask out the sparse binding bit that we aren't caring about (yet!) and the transfer bit
    const VkQueueFlags maskedFlags = (~(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT) &
                                      queueFamilyProperties[i].queueFlags);

    if (!(VK_QUEUE_GRAPHICS_BIT & maskedFlags) && (VK_QUEUE_COMPUTE_BIT & maskedFlags)) {
      *queueFamilyIndex = i;
      return VK_SUCCESS;
    }
  }

  // lastly get any queue that'll work for us
  for (uint32_t i = 0; i < queueFamilyPropertiesCount; i++) {
    // mask out the sparse binding bit that we aren't caring about (yet!) and the transfer bit
    const VkQueueFlags maskedFlags = (~(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT) &
                                      queueFamilyProperties[i].queueFlags);

    if (VK_QUEUE_COMPUTE_BIT & maskedFlags) {
      *queueFamilyIndex = i;
      return VK_SUCCESS;
    }
  }

  return VK_ERROR_INITIALIZATION_FAILED;
}

const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

bool checkValidationLayerSupport() {
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

  for (const char *layerName: validationLayers) {
    bool layerFound = false;

    for (const auto &layerProperties: availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }

    if (!layerFound) {
      return false;
    }
  }

  return true;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
    void *pUserData) {

  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

  return VK_FALSE;
}

VkResult
CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo, const VkAllocationCallbacks *pAllocator,
                             VkDebugUtilsMessengerEXT *pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  }
  else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

// TODO add proper Descriptor set management for optimal binding

int main(int argc, const char *const argv[]) {
  (void) argc;
  (void) argv;

  if (INPUT_LENGTH > 5e8) { // due to memory constraints on GPU
    std::cout << "Can only support up to 500,000,000 elements" << std::endl;
    exit(-1);
  }

  // VK APP SETUP
  const VkApplicationInfo applicationInfo = {
      VK_STRUCTURE_TYPE_APPLICATION_INFO,
      0,
      "VKComputeSample",
      0,
      "",
      0,
      VK_MAKE_VERSION(1, 2, 0)
  };

#ifndef NDEBUG
  if (!checkValidationLayerSupport()) {
    std::cerr << "No validation layer support" << std::endl;
    exit(-1);
  }
#endif

  VkValidationFeatureEnableEXT enables[] = {VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT};
  VkValidationFeaturesEXT features = {};
  features.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
  features.enabledValidationFeatureCount = 1;
  features.pEnabledValidationFeatures = enables;
  const char *validationFeatures[] = {"VK_EXT_validation_features", "VK_EXT_debug_utils"};

  const VkInstanceCreateInfo instanceCreateInfo = {
      VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, &applicationInfo,
      static_cast<uint32_t>(validationLayers.size()), validationLayers.data(),
      2, validationFeatures
  };

  VkInstance instance;
  BAIL_ON_BAD_RESULT(vkCreateInstance(&instanceCreateInfo, 0, &instance));
  // VK APP SETUP - END

  // VK DEBUG MESSENGER
//  VkDebugUtilsMessengerEXT debugMessenger;
//  VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = {
//      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
//      .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT/* | VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT*/,
//      .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
//      .pfnUserCallback = debugCallback,
//      .pUserData = nullptr
//  };
//  BAIL_ON_BAD_RESULT(CreateDebugUtilsMessengerEXT(instance, &debugCreateInfo, nullptr, &debugMessenger));
  // VK DEBUG MESSENGER - END

  // SETUP PHYSICAL DEVICES
  uint32_t physicalDeviceCount = 0;
  BAIL_ON_BAD_RESULT(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, 0));

  VkPhysicalDevice *const physicalDevices = (VkPhysicalDevice *) malloc(
      sizeof(VkPhysicalDevice) * physicalDeviceCount);

  BAIL_ON_BAD_RESULT(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices));

  VkPhysicalDevice physicalDevice;
  if (physicalDeviceCount == 0) {
    std::cout << "No physical devices that support Vulkan applications" << std::endl;
    vkDestroyInstance(instance, 0);
    return 1; // TODO could possibly add cleanup for instance?
  }
  else if (physicalDeviceCount > 1) {
    if (argc == 0) {
      std::cout << "More than one capable physical devices present but no device index was provided in program arguments." << std::endl;
      vkDestroyInstance(instance, 0);
      return 1;
    }
    physicalDevice = physicalDevices[std::strtol(argv[0], nullptr, 10)];
  }
  else {
    physicalDevice = physicalDevices[0];

  }
  // SETUP PHYSICAL DEVICES - END

  // CREATE VkDevice
  uint32_t queueFamilyIndex = 0;
  BAIL_ON_BAD_RESULT(vkGetBestComputeQueueNPH(physicalDevice, &queueFamilyIndex));

  const float queuePrioritory = 1.0f;
  const VkDeviceQueueCreateInfo deviceQueueCreateInfo = {
      VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      0,
      0,
      queueFamilyIndex,
      1,
      &queuePrioritory
  };

  const VkDeviceCreateInfo deviceCreateInfo = {
      VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      0,
      0,
      1,
      &deviceQueueCreateInfo,
      0,
      0,
      0,
      0,
      0
  };

  VkDevice device;
  BAIL_ON_BAD_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, 0, &device));
  // CREATE VkDevice - END

  // INIT MEMORY
  VkPhysicalDeviceMemoryProperties properties;

  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &properties);

  // DECLARE INPUT AND HISTOGRAM MEMORY/BUF SIZES
  const uint32_t inputLength = INPUT_LENGTH;
  const VkDeviceSize inputMemSize = sizeof(uint32_t) * inputLength;

  uint32_t wgCount = ceil((double) inputLength / (double) WG_SIZE);
  uint32_t elementsPerWI = 1;
  if (wgCount > MAX_WG_NUMBER) {
    wgCount = MAX_WG_NUMBER;
    elementsPerWI = ceil(inputLength / (WG_SIZE * wgCount));
  }
  const uint8_t radixElements = 1 << RADIX_BITS;
  if (WG_SIZE < radixElements) {
    std::cout << "Work group size must be equal or greater than the radix elements" << std::endl;
    exit(-1);
  }
  const uint32_t histogramLength = radixElements * wgCount;
  const VkDeviceSize histogramMemSize = sizeof(uint32_t) * histogramLength;
  const VkDeviceSize globalPrefixSumsBufSz = sizeof(uint32_t) * wgCount;
  // DECLARE INPUT AND HISTOGRAM MEMORY/BUF SIZES - END

  // set memoryTypeIndex to an invalid entry in the properties.memoryTypes array
  uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;

  for (uint32_t k = 0; k < properties.memoryTypeCount; k++) {
    if ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & properties.memoryTypes[k].propertyFlags) &&
        (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT & properties.memoryTypes[k].propertyFlags) &&
        (inputMemSize < properties.memoryHeaps[properties.memoryTypes[k].heapIndex].size)) {
      memoryTypeIndex = k;
      break;
    }
  }

  BAIL_ON_BAD_RESULT(memoryTypeIndex == VK_MAX_MEMORY_TYPES ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS);

  VkMemoryAllocateInfo memAllocateInfo = {
      VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      0,
      inputMemSize,
      memoryTypeIndex
  };
  VkDeviceMemory inputDeviceMem;
  BAIL_ON_BAD_RESULT(vkAllocateMemory(device, &memAllocateInfo, 0, &inputDeviceMem));

  VkDeviceMemory outputDeviceMem;
  BAIL_ON_BAD_RESULT(vkAllocateMemory(device, &memAllocateInfo, 0, &outputDeviceMem));

  memAllocateInfo.allocationSize = histogramMemSize;
  VkDeviceMemory histogramDeviceMem;
  BAIL_ON_BAD_RESULT(vkAllocateMemory(device, &memAllocateInfo, 0, &histogramDeviceMem));

  memAllocateInfo.allocationSize = globalPrefixSumsBufSz;
  VkDeviceMemory globalPSumTotalsDeviceMem;
  BAIL_ON_BAD_RESULT(vkAllocateMemory(device, &memAllocateInfo, 0, &globalPSumTotalsDeviceMem));
  // INIT MEMORY - END

  // INIT MEMORY BUFFERS
  VkBufferCreateInfo bufferCreateInfo = {
      VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      0,
      0,
      inputMemSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_SHARING_MODE_EXCLUSIVE,
      1,
      &queueFamilyIndex
  };

  VkBuffer inputBuffer;
  BAIL_ON_BAD_RESULT(vkCreateBuffer(device, &bufferCreateInfo, 0, &inputBuffer));
  BAIL_ON_BAD_RESULT(vkBindBufferMemory(device, inputBuffer, inputDeviceMem, 0));

  VkBuffer outputBuffer;
  BAIL_ON_BAD_RESULT(vkCreateBuffer(device, &bufferCreateInfo, 0, &outputBuffer));
  BAIL_ON_BAD_RESULT(vkBindBufferMemory(device, outputBuffer, outputDeviceMem, 0));

  VkBuffer histogramBuffer;
  bufferCreateInfo.size = histogramMemSize;
  BAIL_ON_BAD_RESULT(vkCreateBuffer(device, &bufferCreateInfo, 0, &histogramBuffer));
  BAIL_ON_BAD_RESULT(vkBindBufferMemory(device, histogramBuffer, histogramDeviceMem, 0));

  VkBuffer globalPSumTotalsBuffer;
  bufferCreateInfo.size = globalPrefixSumsBufSz;
  BAIL_ON_BAD_RESULT(vkCreateBuffer(device, &bufferCreateInfo, 0, &globalPSumTotalsBuffer));
  BAIL_ON_BAD_RESULT(vkBindBufferMemory(device, globalPSumTotalsBuffer, globalPSumTotalsDeviceMem, 0));
  // INIT MEMORY BUFFERS - END

  // INITIALIZE SORTING ARRAY
  uint32_t *hostInput;
  BAIL_ON_BAD_RESULT(vkMapMemory(device, inputDeviceMem, 0, inputMemSize, 0, (void **) &hostInput));

  for (uint32_t k = 0; k < inputLength; k++) {
    hostInput[k] = (uint32_t) abs((int) MWC);
//    hostInput[k] = k % 16;
  }

  vkUnmapMemory(device, inputDeviceMem);
  // INITIALIZE SORTING ARRAY - END

  // CREATE SHADER MODULES

  // TODO create 3 shader modules, one for each stage: count, scan, reorder

  std::vector<char> computeShader{};
  readShaderFile("../shaders/radix_histogram.spv", computeShader);

  VkShaderModuleCreateInfo shaderModuleCreateInfo = {
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      0,
      0,
      computeShader.size(),
      reinterpret_cast<const uint32_t *>(computeShader.data())
  };
  VkShaderModule radixHistShaderModule;
  BAIL_ON_BAD_RESULT(vkCreateShaderModule(device, &shaderModuleCreateInfo, 0, &radixHistShaderModule));

  readShaderFile("../shaders/radix_scan.spv", computeShader);
  shaderModuleCreateInfo.codeSize = computeShader.size();
  shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t *>(computeShader.data());
  VkShaderModule radixScanShaderModule;
  BAIL_ON_BAD_RESULT(vkCreateShaderModule(device, &shaderModuleCreateInfo, 0, &radixScanShaderModule));

  readShaderFile("../shaders/radix_globalsums.spv", computeShader);
  shaderModuleCreateInfo.codeSize = computeShader.size();
  shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t *>(computeShader.data());
  VkShaderModule radixGlobalSumShaderModule;
  BAIL_ON_BAD_RESULT(vkCreateShaderModule(device, &shaderModuleCreateInfo, 0, &radixGlobalSumShaderModule));

  readShaderFile("../shaders/radix_reorder.spv", computeShader);
  shaderModuleCreateInfo.codeSize = computeShader.size();
  shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t *>(computeShader.data());
  VkShaderModule radixReorderShaderModule;
  BAIL_ON_BAD_RESULT(vkCreateShaderModule(device, &shaderModuleCreateInfo, 0, &radixReorderShaderModule));

  // CREATE SHADER MODULES - END

  // DESCRIPTOR SET LAYOUTS AND BINDINGS
  VkDescriptorSetLayoutBinding histogramDescSetLayoutBindings[2] = {
      {
          0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0
      },
      {
          1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0
      }
  };
  VkDescriptorSetLayoutCreateInfo histogramDescSetLayoutCreateInfo = {
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, 0, 0, 2, histogramDescSetLayoutBindings
  };

  VkDescriptorSetLayout histogramDescSetLayout;
  BAIL_ON_BAD_RESULT(vkCreateDescriptorSetLayout(device, &histogramDescSetLayoutCreateInfo, 0, &histogramDescSetLayout));

  VkDescriptorSetLayoutBinding scanDescSetLayoutBindings[2] = {
      {
          0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0
      },
      {
          1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0
      }
  };
  VkDescriptorSetLayoutCreateInfo scanDescSetLayoutCreateInfo = {
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, 0, 0, 2, scanDescSetLayoutBindings
  };

  VkDescriptorSetLayout scanDescSetLayout;
  BAIL_ON_BAD_RESULT(vkCreateDescriptorSetLayout(device, &scanDescSetLayoutCreateInfo, 0, &scanDescSetLayout));

  VkDescriptorSetLayoutBinding reorderDescSetLayoutBindings[4] = {
      {
          0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0
      },
      {
          1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0
      },
      {
          2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0
      },
      {
          3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0
      }
  };
  VkDescriptorSetLayoutCreateInfo reorderDescSetLayoutCreateInfo = {
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, 0, 0, 4, reorderDescSetLayoutBindings
  };

  VkDescriptorSetLayout reorderDescSetLayout;
  BAIL_ON_BAD_RESULT(vkCreateDescriptorSetLayout(device, &reorderDescSetLayoutCreateInfo, 0, &reorderDescSetLayout));
  // DESCRIPTOR SET LAYOUTS AND BINDINGS - END

  // HISTOGRAM PUSH CONSTANTS
  VkPushConstantRange histogramPushConsts = {
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(PushConsts)
  };
  // HISTOGRAM PUSH CONSTANTS - END

  // PIPELINE LAYOUTS CREATION
  VkPipelineLayoutCreateInfo histogramPipelineLayoutCreationInfo = {
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0, 0, 1, &histogramDescSetLayout, 1, &histogramPushConsts
  };
  VkPipelineLayout histogramPipelineLayout;
  BAIL_ON_BAD_RESULT(vkCreatePipelineLayout(device, &histogramPipelineLayoutCreationInfo, 0, &histogramPipelineLayout));

  VkPipelineLayoutCreateInfo scanPipelineLayoutCreationInfo = {
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0, 0, 1, &scanDescSetLayout, 1, &histogramPushConsts
  };
  VkPipelineLayout scanPipelineLayout;
  BAIL_ON_BAD_RESULT(vkCreatePipelineLayout(device, &scanPipelineLayoutCreationInfo, 0, &scanPipelineLayout));

  VkPipelineLayoutCreateInfo reorderPipelineLayoutCreationInfo = {
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0, 0, 1, &reorderDescSetLayout, 1, &histogramPushConsts
  };
  VkPipelineLayout reorderPipelineLayout;
  BAIL_ON_BAD_RESULT(vkCreatePipelineLayout(device, &reorderPipelineLayoutCreationInfo, 0, &reorderPipelineLayout));

  // TODO add pipeline layout for reorder

  // PIPELINE LAYOUTS CREATION - END

  // PIPELINES CREATION
  VkComputePipelineCreateInfo histogramPipelineCreateInfo = {
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, 0, 0,
      {
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_COMPUTE_BIT, radixHistShaderModule, "main", 0
      },
      histogramPipelineLayout, 0, 0
  };

  VkComputePipelineCreateInfo scanPipelineCreateInfo = {
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, 0, 0,
      {
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_COMPUTE_BIT, radixScanShaderModule, "main", 0
      },
      scanPipelineLayout, 0, 0
  };

  VkComputePipelineCreateInfo globalSumPipelineCreateInfo = {
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, 0, 0,
      {
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_COMPUTE_BIT, radixGlobalSumShaderModule, "main", 0
      },
      scanPipelineLayout, 0, 0
  };

  VkComputePipelineCreateInfo reorderPipelineCreateInfo = {
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, 0, 0,
      {
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_COMPUTE_BIT, radixReorderShaderModule, "main", 0
      },
      reorderPipelineLayout, 0, 0
  };

  VkComputePipelineCreateInfo pipelineInfos[4] = {histogramPipelineCreateInfo, scanPipelineCreateInfo,
                                                  globalSumPipelineCreateInfo,reorderPipelineCreateInfo};
  VkPipeline pipelines[4];
  BAIL_ON_BAD_RESULT(vkCreateComputePipelines(device, 0, 4, pipelineInfos, 0, pipelines));
  // COMPUTE PIPELINE CREATION - END

  // ALLOCATE DESCRIPTOR SET
  VkDescriptorPoolSize descriptorPoolSize = {
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8
  };

  VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, 0, 0, 3, 1, &descriptorPoolSize
  };

  VkDescriptorPool descriptorPool;
  BAIL_ON_BAD_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, 0, &descriptorPool));

  VkDescriptorSetAllocateInfo descSetAllocateInfo = {
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, 0, descriptorPool, 1, &histogramDescSetLayout
  };
  VkDescriptorSet histDescSet;
  BAIL_ON_BAD_RESULT(vkAllocateDescriptorSets(device, &descSetAllocateInfo, &histDescSet));

  VkDescriptorSet scanDescSet;
  descSetAllocateInfo.pSetLayouts = &scanDescSetLayout;
  BAIL_ON_BAD_RESULT(vkAllocateDescriptorSets(device, &descSetAllocateInfo, &scanDescSet));

  VkDescriptorSet reorderDescSet;
  descSetAllocateInfo.pSetLayouts = &reorderDescSetLayout;
  BAIL_ON_BAD_RESULT(vkAllocateDescriptorSets(device, &descSetAllocateInfo, &reorderDescSet));

  VkDescriptorBufferInfo inputDescrBufInfo = {
      inputBuffer,
      0,
      VK_WHOLE_SIZE
  };

  VkDescriptorBufferInfo outputDescrBufInfo = {
      outputBuffer,
      0,
      VK_WHOLE_SIZE
  };

  VkDescriptorBufferInfo histogramDescrBufInfo = {
      histogramBuffer,
      0,
      VK_WHOLE_SIZE
  };

  VkDescriptorBufferInfo globalPSumTotalsDescrBufInfo = {
      globalPSumTotalsBuffer,
      0,
      VK_WHOLE_SIZE
  };

  // interesting thing here is that for the vkMemMap changes to become visible to the compute shader (i.e. provide data to the buffer)
  // the descriptor set for that binding must become writable... One would expect that this would be a requirement only for the shader
  // to write to the buffer.
  VkWriteDescriptorSet writeDescriptorSets[2] = {
      {
          VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, histDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &inputDescrBufInfo,     0
      },
      {
          VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, histDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &histogramDescrBufInfo, 0
      }
  };
  vkUpdateDescriptorSets(device, 2, writeDescriptorSets, 0, 0);

  VkWriteDescriptorSet scanWrite[2] = {
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, scanDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &histogramDescrBufInfo,        0},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, scanDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &globalPSumTotalsDescrBufInfo, 0}
  };
  vkUpdateDescriptorSets(device, 2, scanWrite, 0, 0);

  VkWriteDescriptorSet reorderWrite[4] = {
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, reorderDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &inputDescrBufInfo,     0},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, reorderDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &outputDescrBufInfo,    0},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, reorderDescSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &histogramDescrBufInfo, 0},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, reorderDescSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &globalPSumTotalsDescrBufInfo, 0}
  };
  vkUpdateDescriptorSets(device, 4, reorderWrite, 0, 0);
  // ALLOCATE DESCRIPTOR SET - END

  // COMMAND BUFFERS
  VkCommandPoolCreateInfo commandPoolCreateInfo = {
      VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, 0, 0, queueFamilyIndex
  };

  VkCommandPool commandPool;
  BAIL_ON_BAD_RESULT(vkCreateCommandPool(device, &commandPoolCreateInfo, 0, &commandPool));

  VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0, commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1
  };

  VkCommandBuffer commandBuffer;
  BAIL_ON_BAD_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer));

  VkCommandBufferBeginInfo commandBufferBeginInfo = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, 0, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, 0
  };

  auto totalTime = 0;

  for (int i = 0; i < (sizeof(uint32_t) * 8 / RADIX_BITS); i++) {
    uint32_t startBit = (RADIX_BITS * i);
    PushConsts pushConsts = {inputLength, histogramLength, startBit, elementsPerWI}; // TODO update the startBit here for every new pass

    BAIL_ON_BAD_RESULT(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo));

    // RECORD HISTOGRAM PIPELINE
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[0]);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, histogramPipelineLayout, 0, 1, &histDescSet, 0, 0);
    vkCmdPushConstants(commandBuffer, histogramPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 4, &pushConsts);
    vkCmdDispatch(commandBuffer, wgCount, 1, 1);
    // RECORD HISTOGRAM PIPELINE - END

    // ADD BUFFER BARRIER
    // GPUs are free to reschedule ordering of commands in command buffers which means that we must put a barrier
    // if one command is dependent on another command's results
    VkBufferMemoryBarrier barrier = {
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        queueFamilyIndex, queueFamilyIndex, histogramBuffer, 0, histogramMemSize
    };
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 1, &barrier, 0, nullptr);
    // ADD BUFFER BARRIER - END

    // RECORD SCAN PIPELINE
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[1]);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scanPipelineLayout, 0, 1, &scanDescSet, 0, 0);
    vkCmdPushConstants(commandBuffer, scanPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 4, &pushConsts);
    vkCmdDispatch(commandBuffer, wgCount, 1, 1); // TODO probably need less wgs
    // RECORD SCAN PIPELINE - END

    // BUFFER BARRIER
    barrier = {
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        queueFamilyIndex, queueFamilyIndex, globalPSumTotalsBuffer, 0, globalPrefixSumsBufSz
    };
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 1, &barrier, 0, nullptr);
    // BUFFER BARRIER - END

    // RECORD GLOBAL SUM PIPELINE
    pushConsts.sumArrLength = wgCount;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[2]);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scanPipelineLayout, 0, 1, &scanDescSet, 0, 0);
    vkCmdPushConstants(commandBuffer, scanPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 4, &pushConsts);
    vkCmdDispatch(commandBuffer, 1, 1, 1); // TODO probably need less wgs
    // RECORD GLOBAL SUM PIPELINE - END

    // BUFFER BARRIER
    barrier = {
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        queueFamilyIndex, queueFamilyIndex, globalPSumTotalsBuffer, 0, globalPrefixSumsBufSz
    };
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 1, &barrier, 0, nullptr);
    // BUFFER BARRIER - END

    // RECORD REORDER PIPELINE
    pushConsts.sumArrLength = histogramLength;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[3]);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, reorderPipelineLayout, 0, 1, &reorderDescSet, 0, 0);
    vkCmdPushConstants(commandBuffer, reorderPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 4, &pushConsts);
    vkCmdDispatch(commandBuffer, wgCount, 1, 1);
    // RECORD REORDER PIPELINE - END

    // BUFFER BARRIER
    barrier = {
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        queueFamilyIndex, queueFamilyIndex, outputBuffer, 0, inputMemSize
    };
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 1, &barrier, 0, nullptr);
    // BUFFER BARRIER - END

    VkBufferCopy bufferCopy = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = inputMemSize
    };
    vkCmdCopyBuffer(commandBuffer, outputBuffer, inputBuffer, 1, &bufferCopy);

    BAIL_ON_BAD_RESULT(vkEndCommandBuffer(commandBuffer));
    // COMMAND BUFFERS - END

    VkQueue queue;
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

    VkSubmitInfo submitInfo = {
        VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &commandBuffer, 0, 0
    };

    const VkFenceCreateInfo fenceCI = {
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        0,
        0
    };
    VkFence fence;
    vkCreateFence(device, &fenceCI, nullptr, &fence);

    auto start = std::chrono::high_resolution_clock::now();

    BAIL_ON_BAD_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));

    uint64_t waitTimeNanos = 1e10;
    BAIL_ON_BAD_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, waitTimeNanos));
    auto stop = std::chrono::high_resolution_clock::now();
    totalTime += std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    vkResetCommandBuffer(commandBuffer, 0);

//    // TODO REMOVE THIS. RESETS THE OUTPUT BUFFER. ONLY FOR TEST, NO REASON IN PROD
//    BAIL_ON_BAD_RESULT(vkMapMemory(device, outputDeviceMem, 0, inputMemSize, 0, (void **) &hostInput));
//    for (size_t i = 0; i < inputLength; i++) {
//      hostInput[i] = 0;
//    }
//    vkUnmapMemory(device, outputDeviceMem);
//    // TODO REMOVE THIS. RESETS THE OUTPUT BUFFER. ONLY FOR TEST, NO REASON IN PROD

  }

  printf("Sort in %d millis\n", totalTime);

#if 1
  BAIL_ON_BAD_RESULT(vkMapMemory(device, outputDeviceMem, 0, inputMemSize, 0, (void **) &hostInput));

//  std::cout << "Histogram: [" << std::endl;
//  for (size_t i = 0; i < inputLength; i++) {
//    if (i != 0 && i % 16 == 0) {
//      std::cout << std::endl;
//    }
//    std::cout << hostInput[i] << ", ";
//  }
//  std::cout << std::endl << "]" << std::endl;
  std::cout << "The array is " << (checkSorted(hostInput, inputLength) ? "sorted" : "unsorted") << std::endl;

  vkUnmapMemory(device, outputDeviceMem);
#endif

}
