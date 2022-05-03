//
// Created by KPentaris on 29/03/2022.
//

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <chrono>
#include <memory>
#include <algorithm>
#include <iostream>

// Generates predetermined random 32 bit numbers
#define znew   (z=36969*(z&65535)+(z>>16))
#define wnew   (w=18000*(w&65535)+(w>>16))
#define MWC    ((znew<<16)+wnew )
static unsigned long z = 362436069, w = 521288629;

bool checkSorted(int *array, size_t count) {
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

void inPlacePrefixSum(int *buckets, size_t elements) {
  // prefix sum
  int previous = 0;
  for (size_t i = 0; i < elements; ++i) {
    int temp = buckets[i];
    buckets[i] = previous;
    previous += temp;
  }
}

int comp(const void *a, const void *b) {
  int diff = *(int *) a - *(int *) b;
  if (diff < 0) return -1;
  if (diff > 0) return 1;
  return 0;
}

int main(int argc, char **argv) {
  short maxBits = 32;
  size_t elements = 678;
  short radixBits = 4;

  int *unsorted = (int *) _malloca(elements * sizeof(int));
  for (size_t i = 0; i < elements; ++i) {
    unsorted[i] = abs((int) MWC);
//    unsorted[i] = i;
  }
  int *sorted = (int *) _malloca(elements * sizeof(int));
  size_t counters = 1 << radixBits;

  int *buckets = new int[counters];

  int passes = ceil((float) maxBits / (float) radixBits);

  auto start = std::chrono::high_resolution_clock::now();

  int mask = (1 << radixBits) - 1; // mask for each pass
  for (size_t pass = 0; pass < passes; ++pass) {
    memset(buckets, 0, counters * sizeof(int)); // reset buckets
    // count radix occurrences
    for (size_t i = 0; i < elements; ++i) {
      int bucketIdx = (unsorted[i] & mask) >> (pass * radixBits);
      ++buckets[bucketIdx];
    }

    inPlacePrefixSum(buckets, counters);

    if(pass == 0) {
      std::cout << "Histogram: [" << std::endl;
      for (size_t i = 0; i < counters; ++i) {
        std::cout << buckets[i] << ", ";
      }
      std::cout << std::endl << "]" << std::endl;
    }

    for (size_t i = 0; i < elements; ++i) {
      int bitChunk = (unsorted[i] & mask) >> (pass * radixBits);
      int chunkSortPosition = buckets[bitChunk]++; // increase the position of the same bitchunk if we encounter it again
      sorted[chunkSortPosition] = unsorted[i];
    }

    memcpy(unsorted, sorted, elements * sizeof(int));
    mask = mask << radixBits; // next pass mask
  }


  auto stop = std::chrono::high_resolution_clock::now();

  printf(checkSorted(sorted, elements) ? "SORTED\n" : "UNSORTED\n");
  printf("In %d millis", std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
  _freea(unsorted);
  _freea(sorted);

  return 0;
}
