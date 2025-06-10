/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include<iostream>
#include <cmath>

__global__ void min_reduce(float* d_out, const float* const d_in, int num)
{
   int threadId = threadIdx.x + blockDim.x * blockIdx.x;
   if (threadId > num - 1) return;

   if (threadId % 2 == 1) return;

   if (threadId == num - 1) {
      d_out[threadId/2] = d_in[threadId];
   } else {
      d_out[threadId/2] = d_in[threadId] < d_in[threadId+1]? d_in[threadId] : d_in[threadId+1];
   }
}

__global__ void max_reduce(float* d_out, const float* const d_in, const int num)
{
   int threadId = threadIdx.x + blockDim.x * blockIdx.x;
   if (threadId > num - 1) return;

   if (threadId % 2 == 1)
      return;

   if (threadId == num - 1) {
      d_out[threadId/2] = d_in[threadId];
   } else {
      d_out[threadId/2] = d_in[threadId] > d_in[threadId+1]? d_in[threadId] : d_in[threadId+1];
   }
}

void reduce(float& out_val, const float* const d_logLuminance, int num, bool find_min)
{
   const size_t THREADS_PER_BLOCK = 1024;
   float* d_in = nullptr;
   float* d_out = nullptr;
   while (num > 1) {
      const size_t BLOCK_PER_GRID = std::ceil(1.0f * num / THREADS_PER_BLOCK);
      const dim3 blockSize(THREADS_PER_BLOCK, 1, 1);
      const dim3 gridSize(BLOCK_PER_GRID, 1, 1);
      checkCudaErrors(cudaMalloc(&d_out, BLOCK_PER_GRID * sizeof(float)));
      // const size_t sharedMemSize = sizeof(float) * ((num+1)/2);
      if (!d_in) {
         if (find_min)
            min_reduce<<<gridSize, blockSize>>>(d_out, d_logLuminance, num);
         else
            max_reduce<<<gridSize, blockSize>>>(d_out, d_logLuminance, num);
      } else {
         if (find_min)
            min_reduce<<<gridSize, blockSize>>>(d_out, d_in, num);
         else
            max_reduce<<<gridSize, blockSize>>>(d_out, d_in, num);
      }
      checkCudaErrors(cudaGetLastError());
      num = (num + 1) / 2;
      d_in = d_out;
   }
   checkCudaErrors(cudaMemcpy(&out_val, d_out, sizeof(float), cudaMemcpyDeviceToHost));
}

__global__ void generate_histogram(unsigned int* d_bin, const float* const d_logLuminance, size_t numBins, float range, float min_logLum)
{
   int threadId = threadIdx.x + blockDim.x * blockIdx.x;
   if (threadId >= numBins) return;
   int bin = (d_logLuminance[threadId] - min_logLum) / range * numBins;
   bin = bin == numBins ? numBins - 1 : bin;
   atomicAdd(&(d_bin[bin]), 1);
}

// Hillis Steele Scan
__global__ void compute_cdf(unsigned int* d_cdf, const unsigned int* const d_bin, size_t numBins)
{
   int threadId = threadIdx.x + blockDim.x * blockIdx.x;
   if (threadId >= numBins) return;
   int prev_value = 0;
   int prev_prev_value = 0;
   int dist = 1;
   for (int i = 0; i < numBins; i++) {
      if (dist > numBins / 2) break;
      if (i == 0) {
         if (threadId == 0) {
            d_cdf[threadId] = d_bin[threadId];
         } else {
            d_cdf[threadId] = d_bin[threadId] + d_bin[threadId-dist];
         }
      } else {
         if (threadId >= dist) {
            d_cdf[threadId] = prev_value + prev_prev_value;
         }
      }
      __syncthreads();
      prev_value = d_cdf[threadId];
      dist = pow(2, i+1);
      if (threadId >= dist) {
         prev_prev_value = d_cdf[threadId - dist];
      }
      __syncthreads();
   }
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
   // std::cout << "row: " << numRows << ", col: " << numCols << std::endl;
   int num = numRows * numCols;
   reduce(min_logLum, d_logLuminance, num, true);
   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

   reduce(max_logLum, d_logLuminance, num, false);
   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

   printf("%f %f\n", min_logLum, max_logLum);
   float range = max_logLum - min_logLum;
   printf("%f\n", range);

   const int THREADS_PER_BLOCK = 1024;
   const size_t BLOCK_PER_GRID = num / THREADS_PER_BLOCK + 1;
   const dim3 blockSize(THREADS_PER_BLOCK, 1, 1);
   const dim3 gridSize(BLOCK_PER_GRID, 1, 1);

   unsigned int* d_bin;
   checkCudaErrors(cudaMalloc(&d_bin, numBins * sizeof(unsigned int)));
   checkCudaErrors(cudaMemset(d_bin, 0, numBins * sizeof(unsigned int)));
   generate_histogram<<<gridSize, blockSize>>>(d_bin, d_logLuminance, numBins, range, min_logLum);
   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
   compute_cdf<<<gridSize, blockSize>>>(d_cdf, d_bin, numBins);
   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
