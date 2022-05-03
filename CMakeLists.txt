cmake_minimum_required(VERSION 3.21)
project(GPURadix)

set(CMAKE_CXX_STANDARD 20)

#set(CMAKE_CXX_FLAGS /Wall)
#set(CMAKE_CXX_FLAGS_RELEASE /O2)

include_directories(
        C:\\VulkanSDK\\1.3.204.0\\Include
)

link_directories(
        C:\\VulkanSDK\\1.3.204.0\\Lib
)

link_libraries(vulkan-1.lib)

add_executable(GPURadix main.cpp)
#add_executable(GPURadix cpu_radix.cpp)
