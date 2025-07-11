//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <algorithm>
#include <cstring>
#include <stdio.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__ void generate_histogram(unsigned int* d_bin, const size_t numElems, const unsigned int mask, const unsigned int bit, const unsigned int* d_input)
{
   int threadId = threadIdx.x + blockDim.x * blockIdx.x;
   if (threadId >= numElems) return;
   int bin = (d_input[threadId] & mask) >> bit;
   atomicAdd(&(d_bin[bin]), 1);
}

// CUDA kernel for the up-sweep (reduce) phase
__global__ void up_sweep(unsigned int *data, int n, int d) {
    int k = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = 1 << (d + 1);
    if (k < n / stride) {
        int ai = stride * k + (1 << d) - 1;
        int bi = stride * k + stride - 1;
        data[bi] += data[ai];
    }
}

// CUDA kernel for the down-sweep phase
__global__ void down_sweep(unsigned int *data, int n, int d) {
    int k = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = 1 << (d + 1);
    if (k < n / stride) {
        int ai = stride * k + (1 << d) - 1;
        int bi = stride * k + stride - 1;
        int t = data[ai];
        data[ai] = data[bi];
        data[bi] += t;
    }
}

void blelloch_scan(unsigned int *d_scan, const unsigned int* d_bin, int numBins) {
    unsigned int *d_data;
    cudaMalloc(&d_data, numBins * sizeof(unsigned int));
    cudaMemcpy(d_data, d_bin, numBins * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

    int threads_per_block = 512;
    int blocks = (numBins + threads_per_block - 1) / threads_per_block;

    // Up-sweep (reduce) phase
    for (int d = 0; d < log2(numBins); ++d) {
        up_sweep<<<blocks, threads_per_block>>>(d_data, numBins, d);
        cudaDeviceSynchronize();
    }

    // Set last element to 0
    cudaMemset(d_data + numBins - 1, 0, sizeof(unsigned int));

    // Down-sweep phase
    for (int d = log2(numBins) - 1; d >= 0; --d) {
        down_sweep<<<blocks, threads_per_block>>>(d_data, numBins, d);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(d_scan, d_data, numBins * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaFree(d_data);
}

// CUDA kernel for determining the final position and moving elements
__global__ void scatter_kernel(unsigned int *d_in_vals, unsigned int *d_in_pos,
                               unsigned int *d_out_vals, unsigned int *d_out_pos,
                               unsigned int *d_scan,
                               size_t n, unsigned int bit, unsigned int mask) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= n) return;
    unsigned int bin = (d_in_vals[idx] & mask) >> bit;
    unsigned int pos = d_scan[bin];
    for (int i = 0; i < idx; ++i) {
        unsigned int temp_bin = (d_in_vals[i] & mask) >> bit;
        if (temp_bin == bin) {
            pos++;
        }
    }
    d_out_vals[pos] = d_in_vals[idx];
    d_out_pos[pos] = d_in_pos[idx];
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  unsigned int* d_bin;
  unsigned int* d_scan;

  const int numBits = 1;
  const int numBins = 1 << numBits;

  checkCudaErrors(cudaMalloc(&d_bin, numBins * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_scan, numBins * sizeof(unsigned int)));

  const int THREADS_PER_BLOCK = 1024;
  const size_t BLOCK_PER_GRID = numElems / THREADS_PER_BLOCK + 1;
  const dim3 blockSize(THREADS_PER_BLOCK, 1, 1);
  const dim3 gridSize(BLOCK_PER_GRID, 1, 1);

  for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits) {
    unsigned int mask = (numBins - 1) << i;

    checkCudaErrors(cudaMemset(d_bin, 0, numBins * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_scan, 0, numBins * sizeof(unsigned int)));

    generate_histogram<<<gridSize, blockSize>>>(d_bin, numElems, mask, i, d_inputVals);

    blelloch_scan(d_scan, d_bin, numBins);

    scatter_kernel<<<gridSize, blockSize>>>(
        d_inputVals, d_inputPos, d_outputVals, d_outputPos, d_scan, 
        numElems, i, mask);
        cudaDeviceSynchronize();

    cudaMemcpy(d_inputVals, d_outputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_inputPos, d_outputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
  }
  cudaMemcpy(d_outputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_outputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

  cudaFree(d_bin);
  cudaFree(d_scan);
}
