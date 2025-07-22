/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"

__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  // Basic implementation - each thread processes one element
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid < numVals) {
    unsigned int bin = vals[tid];
    // Note: Assuming bin values are valid, but add safety check if needed
    atomicAdd(&histo[bin], 1);
  }
}

// Optimized version using shared memory (uncomment to use)

__global__
void yourHistoOptimized(const unsigned int* const vals,
                        unsigned int* const histo,
                        int numVals,
                        int numBins)
{
  // Shared memory for per-block histogram
  extern __shared__ unsigned int localHisto[];
  
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int localTid = threadIdx.x;
  
  // Initialize shared memory
  for (int i = localTid; i < numBins; i += blockDim.x) {
    localHisto[i] = 0;
  }
  __syncthreads();
  
  // Each thread processes multiple elements to reduce overhead
  for (int i = tid; i < numVals; i += blockDim.x * gridDim.x) {
    unsigned int bin = vals[i];
    if (bin < numBins) { // Safety check for bin bounds
      atomicAdd(&localHisto[bin], 1);
    }
  }
  __syncthreads();
  
  // Merge local histogram to global histogram
  for (int i = localTid; i < numBins; i += blockDim.x) {
    if (localHisto[i] > 0) {
      atomicAdd(&histo[i], localHisto[i]);
    }
  }
}


void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  // Define block size (number of threads per block)
  const int blockSize = 512;
  
  // Choose approach:
  // 1 = Basic atomic operations (simple, always works)
  // 2 = Shared memory optimization (faster for reasonable numBins)
  const int approach = 2;

  switch (approach) {
    case 1: {
      // Basic approach - each thread processes one element
      const int gridSize = (numElems + blockSize - 1) / blockSize;
      yourHisto<<<gridSize, blockSize>>>(d_vals, d_histo, numElems);
      break;
    }
    
    case 2: {
      // Optimized approach with shared memory
      const int maxGridSize = 65535; // Maximum grid size for older GPUs
      const int gridSizeOpt = (numElems + blockSize - 1) / blockSize;
      const int finalGridSize = (gridSizeOpt < maxGridSize) ? gridSizeOpt : maxGridSize;
      const int sharedMemSize = numBins * sizeof(unsigned int);
      
      // Check if shared memory size is reasonable (< 48KB per block)
      if (sharedMemSize <= 48 * 1024) {
        yourHistoOptimized<<<finalGridSize, blockSize, sharedMemSize>>>(
          d_vals, d_histo, numElems, numBins);
      } else {
        // Fall back to basic approach if shared memory requirement is too large
        printf("Warning: Too many bins for shared memory optimization, using basic approach\n");
        const int gridSize = (numElems + blockSize - 1) / blockSize;
        yourHisto<<<gridSize, blockSize>>>(d_vals, d_histo, numElems);
      }
      break;
    }
    
    default:
      printf("Error: Invalid approach selected\n");
      return;
  }

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
