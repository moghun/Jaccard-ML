#ifndef _GPU_UTIL
#define _GPU_UTIL
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <utility>
#include <tuple>
#include <cmath>
#include <omp.h>

inline cudaError_t verboseCudaMalloc(void ** ptr, size_t size, const char*file, int line);
template <typename EN, typename VID>
__device__ inline EN bst_spec(const VID* array, EN right, VID target){
  EN match   = (EN)(-1); 
  right++;
  EN left  = 1; 
  EN middle;
  VID curr;
  while (left <= right) { 
    middle = (left + right) >> 1;
    curr  = array[middle-1];
    if (curr > target) {
      right = middle - 1;
    } else if (curr < target) {
      left = middle + 1;
    } else {
      match = middle-1;
      break;
    }
  }
  return match;
}

template <typename EN, typename VID>
__device__ inline EN bst(const EN* __restrict__ xadj, const VID* __restrict__ adj, VID neighbor, VID target){
  EN match   = (EN)(-1); 
  EN left  = xadj[neighbor]+1; 
  EN right = xadj[neighbor + 1]; 
  VID curr;
  EN middle;
  while (left <= right) { 
    middle = ((unsigned long long)left + (unsigned long long)right) >> 1;
    curr  = adj[middle-1];
    if (curr > target) {
      right = middle - 1;
    } else if (curr < target) {
      left = middle + 1;
    } else {
      match = middle-1;
      break;
    }
  }
  return match;
}
#define FULL_MASK 0xffffffff
__inline__ __device__ 
int warpReduce(int val, unsigned int length, unsigned mask) {
  for (int delta =length/2; delta>0; delta/=2){
    val+=__shfl_down_sync(mask, val,delta, length); 
  }
  return val;
}
__inline__ __device__ 
unsigned calculateMask(char length, unsigned long long thread_id){
#define WARP_SIZE 32
  if (length >= 32) return FULL_MASK;
  //unsigned mask = 0x80000000;
  unsigned mask = 1;
  for (char i =1; i<length; i++){
    mask = (mask<<1)|1; 
  //  mask = (mask>>1)|0x80000000; 
  }
  int group_in_warp = (thread_id%WARP_SIZE)/length;
  mask = mask << (group_in_warp*length);
  //mask = mask >> (group_in_warp*length);
  return mask;
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
#define vcudaMalloc(ptr, size) { verboseCudaMalloc(ptr, size, __FILE__, __LINE__);}
inline cudaError_t verboseCudaMalloc(void ** ptr, size_t size, const char*file, int line){
  size_t gpu_free_memory, gpu_total_memory;
  gpuErrchk(cudaMemGetInfo(&gpu_free_memory,&gpu_total_memory));
  //printf("Attempting to allocate %zu bytes. Free memory %zu/%zu\n", size, gpu_free_memory, gpu_total_memory);
  cudaError_t error = cudaMalloc(ptr, size);
  gpuErrchk(cudaMemGetInfo(&gpu_free_memory,&gpu_total_memory));
  //printf("Free memory after allocation: %zu/%zu\n", gpu_free_memory, gpu_total_memory);
  return error; 
}
#endif
