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

#include "gpu_utils.cu"

#define MINDEG 32
#define MAXDEG 96
#define MAX_GRID_DIM 65535
#define THREADS_PER_BLOCK 64
using namespace std;

typedef unsigned long long ull;

template <typename EN, typename VID>
void mod_4(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, vector<EN*> &bins, vector<pair<unsigned long long, unsigned long long>>& bin_sizes, vector<EN> ranges){
  for (EN i = 0; i < xadj[n]; i++){
    int selected_bin = i%4;
    bins[selected_bin][bin_sizes[selected_bin].first++] = i;
  }
}

template <typename EN, typename VID>
void small_large(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, vector<EN*> &bins, vector<EN>& bin_sizes, vector<EN> ranges){
  for (EN i = 0; i < n; i++){
    if (xadj[i+1]-xadj[i]<MINDEG ||xadj[i+1]-xadj[i]>MAXDEG)
      for (EN j = xadj[i]; j<xadj[i+1]; j++)
        bins[0][bin_sizes[0].first++] = j;
    else
      bins[1][bin_sizes[1].first++] = i;
  }
}
#undef MINDEG
#undef MAXDEG

template <typename EN, typename VID>
void split_vertices_by_ranges_cugraph_heur(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, vector<EN*> &bins, vector<pair<unsigned long long, unsigned long long>>& bin_sizes, vector<EN> ranges){
  for (EN i = 0; i < n; i++){
    EN degu = xadj[i+1]-xadj[i];
      bool added = false;
      for (int r = 0; r < ranges.size() && !added; r++){
        if (degu < ranges[r]) {
            bins[r][bin_sizes[r].first++] = i;
            ///// the value of bin_sizes[r].second is ignored
            //bin_sizes[r].second += degu;
            added = true;
        }
      }
      if (!added){
        bins[ranges.size()][bin_sizes[ranges.size()].first++] = i;
        ///// the value of bin_sizes[r].second is ignored
        //bin_sizes[ranges.size()].second += degu;
      }
  }
}
template <typename EN, typename VID>
void split_vertices_by_ranges_xadj_start(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, vector<EN*> &bins, vector<pair<unsigned long long, unsigned long long>>& bin_sizes, vector<EN> ranges){
  for (EN i = 0; i < n; i++){
    EN degu = xadj[i+1]-xadj[i];
    if (xadj_start[i] != degu){
      bool added = false;
      for (int r = 0; r < ranges.size() && !added; r++){
        if (degu < ranges[r]) {
            bins[r][bin_sizes[r].first++] = i;
            bin_sizes[r].second += degu - xadj_start[i];
            added = true;
        }
      }
      if (!added){
        bins[ranges.size()][bin_sizes[ranges.size()].first++] = i;
        bin_sizes[ranges.size()].second += degu - xadj_start[i];
      }
    }
  }
}
template <typename EN, typename VID>
void split_vertices_by_ranges_pseudo_sort(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, vector<EN*> &bins, vector<pair<unsigned long long, unsigned long long>>& bin_sizes, vector<EN> ranges){
  for (EN i = 0; i < n; i++){
    EN degu = xadj[i+1]-xadj[i];
    bool added = false;
    for (int r = 0; r < ranges.size() && !added; r++){
      if (degu < ranges[r]) {
        unsigned long long num_added = 0;
        EN lower_limit = (r == 0) ? 0: ranges[r-1]; // the edge count after which to ignore an edge (don't calculate its jaccard)
        for (EN j = xadj[i]; j < xadj[i+1]; j++){
          if (xadj[adj[j]+1]-xadj[adj[j]] >= lower_limit) num_added++;
        }
        if (num_added>0)
          bin_sizes[r].second+=num_added;
          bins[r][bin_sizes[r].first++] = i;
          added = true;
      }
    }
    if (!added){
      unsigned long long num_added =0;
      EN lower_limit = ranges[ranges.size()-1]; // the edge count after which to ignore an edge (don't calculate its jaccard)
      for (EN j = xadj[i]; j < xadj[i+1]; j++){
        if (xadj[adj[j]+1]-xadj[adj[j]] > ranges[ranges.size()-1]) num_added++;
      }
      if (num_added>0){
        bins[ranges.size()][bin_sizes[ranges.size()].first++] = i;
        bin_sizes[ranges.size()].second+=num_added;
      }
    }
  }
}
template <typename EN, typename VID>
void split_vertices_by_ranges(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, vector<EN*> &bins, vector<pair<unsigned long long, unsigned long long>>& bin_sizes, vector<EN> ranges){
  for (EN i = 0; i < n; i++){
    EN degu = xadj[i+1]-xadj[i];
    bool added = false;
    for (int r = 0; r < ranges.size() && !added; r++){
      if (degu < ranges[r]) {
        unsigned long long num_added = 0;
        for (EN j = xadj[i]; j < xadj[i+1]; j++){
          if (adj[j] >= i) num_added++;
        }
        if (num_added>0)
          bin_sizes[r].second+=num_added;
          bins[r][bin_sizes[r].first++] = i;
          added = true;
      }
    }
    if (!added){
      unsigned long long num_added =0;
      for (EN j = xadj[i]; j < xadj[i+1]; j++){
        if (adj[j] >= i) num_added++;
      }
      if (num_added>0){
        bins[ranges.size()][bin_sizes[ranges.size()].first++] = i;
        bin_sizes[ranges.size()].second+=num_added;
      }
    }
  }
}

template <typename EN, typename VID>
void split_vertices_by_ranges_edges_onebin(VID * is, EN* xadj, VID* adj, EN * tadj, EN* xadj_start, VID n, vector<EN*> &bins, vector<pair<unsigned long long, unsigned long long>>& bin_sizes, vector<EN> ranges){
  for (EN i = 0; i < n; i++){
    EN degu = xadj[i+1]-xadj[i];
    for (EN start = xadj[i]; start < xadj[i+1]; start++){ 
      EN degv = xadj[adj[start]+1]-xadj[adj[start]];
      if (degu < degv || degu == degv && i < adj[start])
        bins[0][bin_sizes[0].first++] = start;//adj[start];
    }
  }
}
template <typename EN, typename VID>
void split_vertices_by_ranges_edges_nofilter(VID * is, EN* xadj, VID* adj, EN * tadj, EN* xadj_start, VID n, vector<EN*> &bins, vector<pair<unsigned long long, unsigned long long>>& bin_sizes, vector<EN> ranges){
  for (EN i = 0; i < n; i++){
    EN degu = xadj[i+1]-xadj[i];
    bool added = false;
    for (int r = 0; r < ranges.size() && !added; r++){
      if (degu < ranges[r]) {
        bin_sizes[r].second++;
        for (EN start = xadj[i]; start < xadj[i+1]; start++){ 
          bins[r][bin_sizes[r].first++] = start;//adj[start];
        }
        added = true;
      }
    }
    if (!added){
      bin_sizes[ranges.size()].second++;
      for (EN start = xadj[i]; start < xadj[i+1]; start++){ 
        bins[ranges.size()][bin_sizes[ranges.size()].first++] = start; //adj[start];
      }
    }
  }
}
template <typename EN, typename VID>
void split_vertices_by_ranges_edges(VID * is, EN* xadj, VID* adj, EN * tadj, EN* xadj_start, VID n, vector<EN*> &bins, vector<pair<unsigned long long, unsigned long long>>& bin_sizes, vector<EN> ranges){
for (EN i = 0; i < n; i++){
  EN degu = xadj[i+1]-xadj[i];
  bool added = false;
  for (int r = 0; r < ranges.size() && !added; r++){
    if (degu < ranges[r]) {
      for (EN start = xadj[i]; start < xadj[i+1]; start++){ 
        EN degv = xadj[adj[start]+1]-xadj[adj[start]];
        if (degu < degv || degu == degv && i < adj[start])
          bins[r][bin_sizes[r].first++] = start;//adj[start];
      }
      added = true;
    }
  }
  if (!added){
    for (EN start = xadj[i]; start < xadj[i+1]; start++){ 
      EN degv = xadj[adj[start]+1]-xadj[adj[start]];
      if (degu < degv || degu == degv && i < adj[start])
        bins[ranges.size()][bin_sizes[ranges.size()].first++] = start; //adj[start];
    }
  }
}
}

template <typename EN>
bool check_bins(vector<EN *> bins, vector<EN> bin_sizes, EN num_edges){
vector<EN> edges(num_edges, 0);
EN counter = 0;
for (int i =0; i<bins.size(); i++)
  for (int j =0; j<bin_sizes[i]; j++)
    if (edges[bins[i][j]] ==0){
      counter++;
      edges[bins[i][j]] =1;
    }
if (counter != num_edges) return false;
else return true;
}

template <bool directed, typename EN, typename VID, typename E>
vector<tuple<string, pair<unsigned long long, unsigned long long>, double>> binning_based_jaccard_async_strat_onestream(
  // GPU variables
  VID * is_d, EN* xadj_d, VID* adj_d, EN * tadj_d, EN * xadj_start_d, E* d_jac,
  // CPU variables
  VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, E* h_jac,
  // ints
  VID n, EN m, 
  // splitter 
  SEP_FUNC<EN, VID> sep_f, vector<EN> ranges, 
  // strategy function - takes the split, the ranges, the bin, max_threads, and w/e else and returns a function, G, and A values
  STRAT_FUNC<directed, EN, VID, E> strat_func, size_t max_shared_memory){
  // jaccard kernel drivers
  int num_bins = ranges.size()+1;
  vector<EN *> bins(num_bins);
  for (int  i =0; i<num_bins; i++) bins[i] = new EN[m];
  vector<EN *> bins_d(num_bins);
  vector<pair<unsigned long long, unsigned long long>> bin_sizes(num_bins, make_pair(0,0));
  vector<tuple<string, pair<unsigned long long, unsigned long long>, double>> timings(4+num_bins);
  // split the vertices into their respective bins
  double start, end;
  start = omp_get_wtime();
  sep_f(is, xadj, adj, tadj, xadj_start, n, bins, bin_sizes, ranges);
  end = omp_get_wtime();
  timings[0] = make_tuple("Binning", make_pair(0,0), end-start);

  start = omp_get_wtime();
  for (int i =0; i<num_bins; i++){
    gpuErrchk( cudaMalloc((void**)&bins_d[i], sizeof(EN) * bin_sizes[i].first ) );
    gpuErrchk( cudaMemcpyAsync(bins_d[i], bins[i], sizeof(EN) * bin_sizes[i].first , cudaMemcpyHostToDevice, 0) );
  }
  vector<cudaEvent_t> events(num_bins+1);
  vector<string> kernel_names;
  for (auto& event : events) gpuErrchk( cudaEventCreate(&event) );
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  timings[1]=make_tuple("Bin alloc/copy", make_pair(0,0), end-start);
  gpuErrchk ( cudaEventRecord(events[0]) );
  start = omp_get_wtime();
#define MAX_THREADS 22118400
  // TODO make this smart
  for (int i =0; i < num_bins; i++){
    EN range = (i<ranges.size()) ? ranges[i] : 99999999;
    std::tuple<JAC_FUNC_GA<directed, EN, VID, E>, size_t, size_t, size_t> strategy = strat_func(i, range, ranges, bin_sizes, MAX_THREADS, max_shared_memory);
    JAC_FUNC_GA<directed, EN, VID, E> jac_kernel_driver = get<0>(strategy); size_t g = get<1>(strategy); size_t a = get<2>(strategy); size_t sm_fac = get<3>(strategy);
    string kernel_name = jac_kernel_driver(is_d, xadj_d, adj_d, tadj_d, xadj_start_d, n, d_jac, bins_d[i], bin_sizes[i].first, range, g, a, nullptr);
    kernel_name="os"+kernel_name;
    kernel_names.push_back(kernel_name);
    gpuErrchk( cudaEventRecord(events[i+1]) );
  }
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  timings[num_bins+3]=make_tuple("GPU Jaccard calculation", make_pair(0,0), end-start);
  for (int i =0; i< num_bins; i++){
    float time = 0;
    gpuErrchk( cudaEventElapsedTime(&time, events[i], events[i+1]));
    timings[i+2] = make_tuple(kernel_names[i], bin_sizes[i] , time/1000);
  }
  start = omp_get_wtime();
  //timings.push_back(make_tuple("Overall", make_pair(0,0), end-start));
  gpuErrchk( cudaMemcpy(h_jac, d_jac, sizeof(E) * m , cudaMemcpyDeviceToHost) );
  end = omp_get_wtime();
  timings[num_bins+2]=make_tuple("Copy back", make_pair(0,0), end-start);

  for (int i =0; i<num_bins; i++) gpuErrchk( cudaFree(bins_d[i]));
  for (int i =0; i<num_bins; i++) 
    if(bin_sizes[i].first>0)
      delete [] bins[i];
  return timings;
}
template <bool directed, typename EN, typename VID, typename E>
vector<tuple<string, pair<unsigned long long, unsigned long long>, double>> binning_based_jaccard_async_strat(
  // GPU variables
  VID * is_d, EN* xadj_d, VID* adj_d, EN * tadj_d, EN * xadj_start_d, E* d_jac,
  // CPU variables
  VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, E* h_jac,
  // ints
  VID n, EN m, 
  // splitter 
  SEP_FUNC<EN, VID> sep_f, vector<EN> ranges, 
  // strategy function - takes the split, the ranges, the bin, max_threads, and w/e else and returns a function, G, and A values
  STRAT_FUNC<directed, EN, VID, E> strat_func, size_t max_shared_memory){
  // jaccard kernel drivers
  int num_bins = ranges.size()+1;
  vector<EN *> bins(num_bins);
  for (int  i =0; i<num_bins; i++) bins[i] = new EN[m];
  vector<EN *> bins_d(num_bins);
  vector<pair<unsigned long long, unsigned long long>> bin_sizes(num_bins, make_pair(0,0));
  vector<tuple<string, pair<unsigned long long, unsigned long long>, double>> timings(4+num_bins);
  // split the vertices into their respective bins
  double start, end;
  start = omp_get_wtime();
  sep_f(is, xadj, adj, tadj, xadj_start, n, bins, bin_sizes, ranges);
  end = omp_get_wtime();
  timings[0] = make_tuple("Binning", make_pair(0,0), end-start);

  start = omp_get_wtime();
  for (int i =0; i<num_bins; i++){
    gpuErrchk( cudaMalloc((void**)&bins_d[i], sizeof(EN) * bin_sizes[i].first ) );
  }
  end = omp_get_wtime();
  timings[1]=make_tuple("Bin alloc", make_pair(0,0), end-start);
  start = omp_get_wtime();
#define MAX_THREADS 22118400
  // TODO make this smart
#pragma omp parallel for num_threads(num_bins)
  for (int i =0; i < num_bins; i++){
    EN range = (i<ranges.size()) ? ranges[i] : 99999999;
    string kernel_name = get_kernel_name("skipped", range, 0, 0, dim3(), dim3(), 0);
    double kstart = omp_get_wtime();
    if (bin_sizes[i].first != 0){
      cudaStream_t stream;
      gpuErrchk( cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      gpuErrchk( cudaMemcpyAsync(bins_d[i], bins[i], sizeof(EN) * bin_sizes[i].first , cudaMemcpyHostToDevice, stream) );
      std::tuple<JAC_FUNC_GA<directed, EN, VID, E>, size_t, size_t, size_t> strategy = strat_func(i, range, ranges, bin_sizes, MAX_THREADS, max_shared_memory);
      JAC_FUNC_GA<directed, EN, VID, E> jac_kernel_driver = get<0>(strategy); size_t g = get<1>(strategy); size_t a = get<2>(strategy); size_t sm_fac = get<3>(strategy);
      kernel_name = jac_kernel_driver(is_d, xadj_d, adj_d, tadj_d, xadj_start_d, n, d_jac, bins_d[i], bin_sizes[i].first, range, g, a, &stream);
      gpuErrchk(cudaStreamSynchronize(stream));
      gpuErrchk(cudaStreamDestroy(stream));
    }
    double kend = omp_get_wtime();
    double time = kend-kstart;
    timings[i+2] = make_tuple(kernel_name, bin_sizes[i] , time);
  }
  end = omp_get_wtime();
  timings[num_bins+3]=make_tuple("GPU Jaccard calculation", make_pair(0,0), end-start);
  start = omp_get_wtime();
  //timings.push_back(make_tuple("Overall", make_pair(0,0), end-start));
  gpuErrchk( cudaMemcpy(h_jac, d_jac, sizeof(E) * m , cudaMemcpyDeviceToHost) );
  end = omp_get_wtime();
  timings[num_bins+2]=make_tuple("Copy back", make_pair(0,0), end-start);

  for (int i =0; i<num_bins; i++) gpuErrchk( cudaFree(bins_d[i]));
  for (int i =0; i<num_bins; i++) 
    if(bin_sizes[i].first>0)
      delete [] bins[i];
  return timings;
}

template <typename EN, typename VID, typename E>
__global__ void jac_binning_atomic_second_kernel_twoarray(const EN* __restrict__ xadj, const VID* __restrict__ adj, const VID* __restrict__ is, const EN n, EN* sum_d, E* emetrics){
  unsigned long long block_local_id =  blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  unsigned long long grid_id = gridDim.x*gridDim.y*blockIdx.z+gridDim.x*blockIdx.y+blockIdx.x;
  unsigned long long tid = (long long)blockDim.z * blockDim.x * blockDim.y * grid_id + (unsigned long long)block_local_id;
  unsigned long long m = xadj[n];
  for (unsigned long long e = tid; e < m; e+=gridDim.x*gridDim.y*gridDim.z*blockDim.x*blockDim.y*blockDim.z){
    emetrics[e] = emetrics[e]/(float)(sum_d[e]-emetrics[e]);
  }
}
// function will take the graph and the jaccard array, as well as a function that will speerate vertices into bins, and a kernel for each bin. It will execute the corresponding kernels on their bins and return the timings of the kernels
// SEP: A seperating function - that takes a vertex ID and returns a class for the vertex
// A list of functions that will be used with each class that SEP creates
template <bool directed, typename EN, typename VID, typename E>
vector<tuple<string, pair<unsigned long long, unsigned long long>, double>> binning_based_jaccard_twostep_twoarray(
  // GPU variables
  VID * is_d, EN* xadj_d, VID* adj_d, EN * tadj_d, EN * xadj_start_d, E* d_jac,
  // CPU variables
  VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, E* h_jac,
  // ints
  VID n, EN m, 
  // splitter 
  SEP_FUNC<EN, VID> sep_f, vector<EN> ranges, 
  // jaccard kernel drivers
  vector<tuple<string, JAC_FUNC<directed, EN, VID, E>, dim3, dim3, VID>> jaccard_kernels){
  if (jaccard_kernels.size() == 0) throw "PASSED 0 KERNELS";
  int num_bins = jaccard_kernels.size();
  vector<EN *> bins(num_bins);
  for (int  i =0; i<num_bins; i++) bins[i] = new EN[m];
  vector<EN *> bins_d(num_bins);
  vector<pair<unsigned long long, unsigned long long>> bin_sizes(num_bins, make_pair(0,0));
  vector<tuple<string, pair<unsigned long long, unsigned long long>, double>> timings;
  // split the vertices into their respective bins

  double start, end;
  EN * d_sum;
  start = omp_get_wtime();
  gpuErrchk( cudaMalloc((void**)&d_sum, sizeof(EN) * m ) );
  end = omp_get_wtime();
  timings.push_back(make_tuple("alloc 2nd arr", make_pair(0,0), end-start));
  start = omp_get_wtime();
  sep_f(is, xadj, adj, tadj, xadj_start, n, bins, bin_sizes, ranges);
  end = omp_get_wtime();
  timings.push_back(make_tuple("Binning", make_pair(0,0), end-start));

  start = omp_get_wtime();
  for (int i =0; i < num_bins; i++){
    gpuErrchk( cudaMalloc((void**)&bins_d[i], sizeof(EN) * bin_sizes[i].first ) );
    gpuErrchk( cudaMemcpy(bins_d[i], bins[i], sizeof(EN) * bin_sizes[i].first , cudaMemcpyHostToDevice) );
  }
  end = omp_get_wtime();
  timings.push_back(make_tuple("Bin alloc/copy", make_pair(0,0), end-start));

  double time;
  start = omp_get_wtime();
  for (int i =0; i < num_bins; i++){
    EN lower_limit = (i == 0) ? 0: ranges[i-1]; // the edge count after which to ignore an edge (don't calculate its jaccard)
    EN upper_limit = (i < num_bins-1) ? ranges[i] : EN(0xffffffffffffffff); // the edge count after which to ignore an edge (don't calculate its jaccard)
    time = get<1>(jaccard_kernels[i])(is_d, xadj_d, adj_d, tadj_d, d_sum, n, d_jac, bins_d[i], bin_sizes[i].first, get<4>(jaccard_kernels[i]), get<2>(jaccard_kernels[i]), get<3>(jaccard_kernels[i]), lower_limit, upper_limit);
    timings.push_back(make_tuple(get<0>(jaccard_kernels[i]), bin_sizes[i], time));
  }
  end = omp_get_wtime();
  timings.push_back(make_tuple("Intersection calculation", make_pair(0,0), end-start));
  start = omp_get_wtime();
  dim3 second_grid(1,1,1);
  unsigned threads_per_block = 512;
  second_grid.x = max(1, min(MAX_GRID_DIM, n/threads_per_block));
  second_grid.y = max(1, min(MAX_GRID_DIM, n/second_grid.x/threads_per_block));
  second_grid.y = max(1, min(MAX_GRID_DIM, n/second_grid.y/threads_per_block));
  jac_binning_atomic_second_kernel_twoarray<<<second_grid, threads_per_block>>>(xadj_d, adj_d, is_d, n, d_sum, d_jac);
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  timings.push_back(make_tuple("Second kernel 2array", make_pair(0,0), end-start));
  start = omp_get_wtime();
  gpuErrchk( cudaMemcpy(h_jac, d_jac, sizeof(E) * m , cudaMemcpyDeviceToHost) );
  end = omp_get_wtime();
  timings.push_back(make_tuple("Copy back", make_pair(0,0), end-start));
  cudaFree(d_sum);
  for (int i =0; i<num_bins; i++) gpuErrchk( cudaFree(bins_d[i]));
  for (int i =0; i<num_bins; i++) 
    if(bin_sizes[i].first>0)
      delete [] bins[i];
  return timings;
}

template <typename EN, typename VID, typename E>
__global__ void jac_binning_atomic_second_kernel(const EN* __restrict__ xadj, const VID* __restrict__ adj, const VID* __restrict__ is, const EN n, E* emetrics){
  unsigned long long block_local_id =  blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  unsigned long long grid_id = gridDim.x*gridDim.y*blockIdx.z+gridDim.x*blockIdx.y+blockIdx.x;
  unsigned long long tid = (long long)blockDim.z * blockDim.x * blockDim.y * grid_id + (unsigned long long)block_local_id;
  unsigned long long m = xadj[n];
  for (unsigned long long e = tid; e < m; e+=gridDim.x*gridDim.y*gridDim.z*blockDim.x*blockDim.y*blockDim.z){
    emetrics[e] = emetrics[e]/(float)(xadj[is[e]+1]-xadj[is[e]]+xadj[adj[e]+1]-xadj[adj[e]]-emetrics[e]);
  }
}
// function will take the graph and the jaccard array, as well as a function that will speerate vertices into bins, and a kernel for each bin. It will execute the corresponding kernels on their bins and return the timings of the kernels
// SEP: A seperating function - that takes a vertex ID and returns a class for the vertex
// A list of functions that will be used with each class that SEP creates
template <bool directed, typename EN, typename VID, typename E>
vector<tuple<string, pair<unsigned long long, unsigned long long>, double>> binning_based_jaccard_twostep(
  // GPU variables
  VID * is_d, EN* xadj_d, VID* adj_d, EN * tadj_d, EN * xadj_start_d, E* d_jac,
  // CPU variables
  VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, E* h_jac,
  // ints
  VID n, EN m, 
  // splitter 
  SEP_FUNC<EN, VID> sep_f, vector<EN> ranges, 
  // jaccard kernel drivers
  vector<tuple<string, JAC_FUNC<directed, EN, VID, E>, dim3, dim3, VID>> jaccard_kernels){
  if (jaccard_kernels.size() == 0) throw "PASSED 0 KERNELS";
  int num_bins = jaccard_kernels.size();
  vector<EN *> bins(num_bins);
  for (int  i =0; i<num_bins; i++) bins[i] = new EN[m];
  vector<EN *> bins_d(num_bins);
  vector<pair<unsigned long long, unsigned long long>> bin_sizes(num_bins, make_pair(0,0));
  vector<tuple<string, pair<unsigned long long, unsigned long long>, double>> timings;
  // split the vertices into their respective bins
  double start, end;

  start = omp_get_wtime();
  sep_f(is, xadj, adj, tadj, xadj_start, n, bins, bin_sizes, ranges);
  end = omp_get_wtime();
  timings.push_back(make_tuple("Binning", make_pair(0,0), end-start));

  start = omp_get_wtime();
  for (int i =0; i < num_bins; i++){
    gpuErrchk( cudaMalloc((void**)&bins_d[i], sizeof(EN) * bin_sizes[i].first ) );
    gpuErrchk( cudaMemcpy(bins_d[i], bins[i], sizeof(EN) * bin_sizes[i].first , cudaMemcpyHostToDevice) );
  }
  end = omp_get_wtime();
  timings.push_back(make_tuple("Bin alloc/copy", make_pair(0,0), end-start));

  double time;
  start = omp_get_wtime();
  for (int i =0; i < num_bins; i++){
    EN lower_limit = (i == 0) ? 0: ranges[i-1]; // the edge count after which to ignore an edge (don't calculate its jaccard)
    EN upper_limit = (i < num_bins-1) ? ranges[i] : EN(0xffffffffffffffff); // the edge count after which to ignore an edge (don't calculate its jaccard)
    time = get<1>(jaccard_kernels[i])(is_d, xadj_d, adj_d, tadj_d, xadj_start_d, n, d_jac, bins_d[i], bin_sizes[i].first, get<4>(jaccard_kernels[i]), get<2>(jaccard_kernels[i]), get<3>(jaccard_kernels[i]), lower_limit, upper_limit);
    timings.push_back(make_tuple(get<0>(jaccard_kernels[i]), bin_sizes[i], time));
  }
  end = omp_get_wtime();
  timings.push_back(make_tuple("Intersection calculation", make_pair(0,0), end-start));
  start = omp_get_wtime();
  dim3 second_grid(1,1,1);
  unsigned threads_per_block = 512;
  second_grid.x = max(1, min(MAX_GRID_DIM, n/threads_per_block));
  second_grid.y = max(1, min(MAX_GRID_DIM, n/second_grid.x/threads_per_block));
  second_grid.y = max(1, min(MAX_GRID_DIM, n/second_grid.y/threads_per_block));
  jac_binning_atomic_second_kernel<<<second_grid, threads_per_block>>>(xadj_d, adj_d, is_d, n, d_jac);
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  timings.push_back(make_tuple("Second kernel", make_pair(0,0), end-start));
  start = omp_get_wtime();
  gpuErrchk( cudaMemcpy(h_jac, d_jac, sizeof(E) * m , cudaMemcpyDeviceToHost) );
  end = omp_get_wtime();
  timings.push_back(make_tuple("Copy back", make_pair(0,0), end-start));
  for (int i =0; i<num_bins; i++) gpuErrchk( cudaFree(bins_d[i]));
  for (int i =0; i<num_bins; i++) 
    if(bin_sizes[i].first>0)
      delete [] bins[i];
  return timings;
}

// function will take the graph and the jaccard array, as well as a function that will speerate vertices into bins, and a kernel for each bin. It will execute the corresponding kernels on their bins and return the timings of the kernels
// SEP: A seperating function - that takes a vertex ID and returns a class for the vertex
// A list of functions that will be used with each class that SEP creates
template <bool directed, typename EN, typename VID, typename E>
vector<tuple<string, pair<unsigned long long, unsigned long long>, double>> binning_based_jaccard_edgefilter(
  // GPU variables
  VID * is_d, EN* xadj_d, VID* adj_d, EN * tadj_d, EN * xadj_start_d, E* d_jac,
  // CPU variables
  VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, E* h_jac,
  // ints
  VID n, EN m, 
  // splitter 
  SEP_FUNC<EN, VID> sep_f, vector<EN> ranges, 
  // jaccard kernel drivers
  vector<tuple<string, JAC_FUNC<directed, EN, VID, E>, dim3, dim3, VID>> jaccard_kernels){
  if (jaccard_kernels.size() == 0) throw "PASSED 0 KERNELS";
  int num_bins = jaccard_kernels.size();
  vector<EN *> bins(num_bins);
  for (int  i =0; i<num_bins; i++) bins[i] = new EN[m];
  vector<EN *> bins_d(num_bins);
  vector<pair<unsigned long long, unsigned long long>> bin_sizes(num_bins, make_pair(0,0));
  vector<tuple<string, pair<unsigned long long, unsigned long long>, double>> timings;
  // split the vertices into their respective bins
  double start, end;

  start = omp_get_wtime();
  sep_f(is, xadj, adj, tadj, xadj_start, n, bins, bin_sizes, ranges);
  end = omp_get_wtime();
  timings.push_back(make_tuple("Binning", make_pair(0,0), end-start));

  start = omp_get_wtime();
  for (int i =0; i < num_bins; i++){
    gpuErrchk( cudaMalloc((void**)&bins_d[i], sizeof(EN) * bin_sizes[i].first ) );
    gpuErrchk( cudaMemcpy(bins_d[i], bins[i], sizeof(EN) * bin_sizes[i].first , cudaMemcpyHostToDevice) );
  }
  end = omp_get_wtime();
  timings.push_back(make_tuple("Bin alloc/copy", make_pair(0,0), end-start));

  double time;
  for (int i =0; i < num_bins; i++){
    EN lower_limit = (i == 0) ? 0: ranges[i-1]; // the edge count after which to ignore an edge (don't calculate its jaccard)
    EN upper_limit = (i < num_bins-1) ? ranges[i] : EN(0xffffffffffffffff); // the edge count after which to ignore an edge (don't calculate its jaccard)
    time = get<1>(jaccard_kernels[i])(is_d, xadj_d, adj_d, tadj_d, xadj_start_d, n, d_jac, bins_d[i], bin_sizes[i].first, get<4>(jaccard_kernels[i]), get<2>(jaccard_kernels[i]), get<3>(jaccard_kernels[i]), bin_sizes[i].second, upper_limit);
    timings.push_back(make_tuple(get<0>(jaccard_kernels[i]), bin_sizes[i], time));
  }
  start = omp_get_wtime();
  gpuErrchk( cudaMemcpy(h_jac, d_jac, sizeof(E) * m , cudaMemcpyDeviceToHost) );
  end = omp_get_wtime();
  timings.push_back(make_tuple("Copy back", make_pair(0,0), end-start));
  for (int i =0; i<num_bins; i++) gpuErrchk( cudaFree(bins_d[i]));
  for (int i =0; i<num_bins; i++) 
    if(bin_sizes[i].first>0)
      delete [] bins[i];
  return timings;
}
// function will take the graph and the jaccard array, as well as a function that will speerate vertices into bins, and a kernel for each bin. It will execute the corresponding kernels on their bins and return the timings of the kernels
// SEP: A seperating function - that takes a vertex ID and returns a class for the vertex
// A list of functions that will be used with each class that SEP creates
template <bool directed, typename EN, typename VID, typename E>
vector<tuple<string, pair<unsigned long long, unsigned long long>, double, nlohmann::json>> binning_based_jaccard(
  // GPU variables
  VID * is_d, EN* xadj_d, VID* adj_d, EN * tadj_d, EN * xadj_start_d, E* d_jac,
  // CPU variables
  VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, E* h_jac,
  // ints
  VID n, EN m, 
  // splitter 
  SEP_FUNC<EN, VID> sep_f, vector<EN> ranges, 
  // jaccard kernel drivers
  vector<tuple<string,JAC_FUNC<directed, EN, VID, E>, dim3, dim3, VID, nlohmann::json>> jaccard_kernels){
  if (jaccard_kernels.size() == 0) throw "PASSED 0 KERNELS";
  int num_bins = jaccard_kernels.size();
  vector<EN *> bins(num_bins);
  for (int  i =0; i<num_bins; i++) bins[i] = new EN[m];
  vector<EN *> bins_d(num_bins);
  vector<pair<unsigned long long, unsigned long long>> bin_sizes(num_bins, make_pair(0,0));
  vector<tuple<string, pair<unsigned long long, unsigned long long>, double, nlohmann::json>> timings;
  // split the vertices into their respective bins
  double start, end;

  start = omp_get_wtime();
  sep_f(is, xadj, adj, tadj, xadj_start, n, bins, bin_sizes, ranges);
  end = omp_get_wtime();
  timings.push_back(make_tuple("Binning", make_pair(0,0), end-start, nlohmann::json()));

  start = omp_get_wtime();
  for (int i =0; i < num_bins; i++){
    gpuErrchk( cudaMalloc((void**)&bins_d[i], sizeof(EN) * bin_sizes[i].first ) );
    gpuErrchk( cudaMemcpy(bins_d[i], bins[i], sizeof(EN) * bin_sizes[i].first , cudaMemcpyHostToDevice) );
  }
  end = omp_get_wtime();
  timings.push_back(make_tuple("Bin alloc/copy", make_pair(0,0), end-start, nlohmann::json()));

  double time;
  for (int i =0; i < num_bins; i++){
    EN lower_limit = (i == 0) ? 0: ranges[i-1]; // the edge count after which to ignore an edge (don't calculate its jaccard)
    EN upper_limit = (i < num_bins-1) ? ranges[i] : EN(0xffffffffffffffff); // the edge count after which to ignore an edge (don't calculate its jaccard)
    time = get<1>(jaccard_kernels[i])(is_d, xadj_d, adj_d, tadj_d, xadj_start_d, n, d_jac, bins_d[i], bin_sizes[i].first, get<4>(jaccard_kernels[i]), get<2>(jaccard_kernels[i]), get<3>(jaccard_kernels[i]), lower_limit, upper_limit);
    timings.push_back(make_tuple(get<0>(jaccard_kernels[i]), bin_sizes[i], time, get<5>(jaccard_kernels[i])));
  }
  start = omp_get_wtime();
  gpuErrchk( cudaMemcpy(h_jac, d_jac, sizeof(E) * m , cudaMemcpyDeviceToHost) );
  end = omp_get_wtime();
  timings.push_back(make_tuple("Copy back", make_pair(0,0), end-start, nlohmann::json()));
  for (int i =0; i<num_bins; i++) gpuErrchk( cudaFree(bins_d[i]));
  for (int i =0; i<num_bins; i++) 
    if(bin_sizes[i].first>0)
      delete [] bins[i];
  return timings;
}

template <bool directed, typename EN, typename VID, typename E>
double fake(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, VID* bin, VID bin_size){
  double start, end;

  start = omp_get_wtime();

  end = omp_get_wtime();
  return end-start;

}


template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_gpu_u_per_warp_bst_sm_kernel(const EN* __restrict__ xadj, const VID* __restrict__ adj, const EN* __restrict__ xadj_start, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {
  int no_threads = blockDim.y * blockDim.x * gridDim.x; //multiple of 32
  unsigned long long tid = blockDim.y*blockDim.x*blockIdx.x + blockDim.x*threadIdx.y + threadIdx.x;
  //if (tid == 0) printf("Running jac_binning_gpu_u_per_warp_bst_sm_kernel\n");
  int block_local_wid = (tid % (blockDim.x * blockDim.y)) / 32;
  unsigned mask = calculateMask(blockDim.x, tid);//threadIdx.y*blockDim.x+threadIdx.x);
  extern __shared__ VID glob_u_adj[]; 
  int grid_local_wid = tid / 32;;
  int grid_no_warps = no_threads / 32;
  int warp_local_tid = tid % 32;
  for (VID t = grid_local_wid; t < bin_size; t+= grid_no_warps){
    VID u = bin[t];
    EN degu = xadj[u+1] - xadj[u];
    __syncwarp();
    VID* u_adj = glob_u_adj + (block_local_wid * SM_FAC);
    for(EN ptr = warp_local_tid; ptr < degu; ptr += 32) {
      u_adj[ptr] = adj[xadj[u] + ptr]; 
    }
    __syncwarp();

    // each group of blockDim.x threads in a warp will handle a single neighbor
    //for (EN ptr = threadIdx.y+xadj_start[u]; ptr < degu; ptr+=blockDim.y){
    for (EN ptr = threadIdx.y; ptr < degu; ptr+=blockDim.y){
      VID v = u_adj[ptr];
      EN v_start = xadj[v];
      if (xadj[v+1]-xadj[v] > degu || (xadj[v+1]-xadj[v] == degu && v < u) ){
        EN degv = xadj[v+1] - xadj[v];
        EN intersection_size = 0;
        EN other_ptr = xadj[v]+bst_spec(&adj[v_start], degv-1, u);
        for (EN t_ptr = threadIdx.x; t_ptr < degv; t_ptr+=blockDim.x){
          EN loc = bst_spec(u_adj, degu-1, adj[v_start+t_ptr]);
          intersection_size+=(loc!=(EN)-1);
        } 
        intersection_size = warpReduce(intersection_size, blockDim.x, mask);
        if (threadIdx.x == 0){
          emetrics[(xadj[u] + ptr)] = emetrics[other_ptr] = float(intersection_size)/float(degu+degv-intersection_size);

        }
      }
    }
  }
  
}

template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_warp_bst_sm_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  start = omp_get_wtime();
/*
  dim3 block; 
  int threads_per_search = 16;

  block.x = threads_per_search;
  block.y = 32/threads_per_search;
  block.z = 1;
  int no_blocks = 1024;
*/
  int no_threads = block.x*block.y;
  unsigned long long shared_memory = SM_FAC * (no_threads / 32) * sizeof(VID);
  jac_binning_gpu_u_per_warp_bst_sm_kernel<directed, EN, VID, E><<<grid, block, shared_memory>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, SM_FAC, lower_limit, upper_limit);
  gpuErrchk( cudaGetLastError());
  cudaDeviceSynchronize();

  end = omp_get_wtime();
  return end-start;

}

template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_gpu_u_per_warp_bst_inv_sm_kernel(const EN* __restrict__ xadj, const VID* __restrict__ adj, const EN* __restrict__ xadj_start, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {
  int no_threads = blockDim.y * blockDim.x * gridDim.x; //multiple of 32
  unsigned long long tid = blockDim.y*blockDim.x*blockIdx.x + blockDim.x*threadIdx.y + threadIdx.x;
  //if (tid == 0) printf("Running jac_binning_gpu_u_per_warp_bst_inv_sm_kernel\n");
  int block_local_wid = (tid % (blockDim.x * blockDim.y)) / 32;
  extern __shared__ VID glob_u_adj[]; 
  int grid_local_wid = tid / 32;;
  int grid_no_warps = no_threads / 32;
  int warp_local_tid = tid % 32;
  unsigned mask = calculateMask(blockDim.x, tid);//threadIdx.y*blockDim.x+threadIdx.x);
  for (VID t = grid_local_wid; t < bin_size; t+= grid_no_warps){
    VID u = bin[t];
    EN degu = xadj[u+1] - xadj[u];

    __syncwarp();
    VID* u_adj = glob_u_adj + (block_local_wid * SM_FAC);
    for(EN ptr = warp_local_tid; ptr < degu; ptr += 32) {
      u_adj[ptr] = adj[xadj[u] + ptr]; 
    }
    __syncwarp();

    // each group of blockDim.x threads in a warp will handle a single neighbor
    for (EN ptr = threadIdx.y; ptr < degu; ptr+=blockDim.y){
    //for (EN ptr = threadIdx.y + xadj_start[u]; ptr < degu; ptr+=blockDim.y){
      VID v = u_adj[ptr];
      if (xadj[v+1]-xadj[v] > degu || (xadj[v+1]-xadj[v] == degu && v < u) ){
        EN degv = xadj[v+1] - xadj[v];
        EN intersection_size = 0;
        EN other_ptr = bst(xadj, adj, v, u);
        for (EN t_ptr = threadIdx.x; t_ptr < degu; t_ptr+=blockDim.x){
          EN loc = bst(xadj, adj, v, u_adj[t_ptr]);
          intersection_size+=(loc!=(EN)-1);
        } 
        intersection_size = warpReduce(intersection_size, blockDim.x, mask);
        //intersection_size = warpReduce(intersection_size, blockDim.x, tid%32, threadIdx.x==0);
        if (threadIdx.x == 0){
          emetrics[(xadj[u] + ptr)] = emetrics[other_ptr] = float(intersection_size)/float(degu+degv-intersection_size);

        }
      }
    }
  }
  
}

template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_warp_bst_inv_sm_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  start = omp_get_wtime();
/*
  dim3 block; 
  int threads_per_search = 16;
  block.x = threads_per_search;
  block.y = 32/threads_per_search;
  block.z = 1;
  int no_blocks = 1024;
*/
  int no_threads = block.x*block.y;
  unsigned long long shared_memory = SM_FAC * (no_threads / 32) * sizeof(VID);
  jac_binning_gpu_u_per_warp_bst_inv_sm_kernel<directed, EN, VID, E><<<grid, block, shared_memory>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, SM_FAC, lower_limit, upper_limit);
  gpuErrchk( cudaGetLastError());
  cudaDeviceSynchronize();

  end = omp_get_wtime();
  return end-start;

}
#define MAX_VAL (0xffffffffffffffff)
template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_gpu_u_per_warp_sm_kernel(const EN* __restrict__ xadj, const VID* __restrict__ adj, const EN * __restrict__ xadj_start, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {
  int no_threads = blockDim.y * blockDim.x * gridDim.x; //multiple of 32
  unsigned long long tid = blockDim.y * blockDim.x * blockDim.z * blockIdx.x + blockDim.x * threadIdx.y + threadIdx.x;
  //if (tid==0)printf("Running jac_binning_gpu_u_per_warp_sm_kernel\n");
  int block_local_wid = (tid % (blockDim.x * blockDim.y)) / 32;
  int warp_local_tid = tid % 32;

  unsigned mask = calculateMask(blockDim.x, tid);//threadIdx.y*blockDim.x+threadIdx.x);
  extern __shared__ VID glob_u_adj[];
  int grid_local_wid = tid / 32;;
  int grid_no_warps = no_threads / 32;
  for(VID t = grid_local_wid; t < bin_size; t += grid_no_warps) {	
    VID u = bin[t];
    EN degu = xadj[u+1] - xadj[u];

    VID* u_adj = glob_u_adj + (block_local_wid * SM_FAC);
    for(EN ptr = warp_local_tid; ptr < degu; ptr += 32) {
      u_adj[ptr] = adj[xadj[u] + ptr]; 
    }
    //if (xadj_start[u]+xadj[u] < xadj[u] || xadj_start[u]+xadj[u] > xadj[u+1]) printf(" u %d xadj (%d, %d) xadj_start %d\n", u, xadj[u], xadj[u+1], xadj_start[u]);
    for(EN ptr = warp_local_tid; ptr < degu; ptr += 32) {
    //for(EN ptr = warp_local_tid+xadj_start[u]; ptr < degu; ptr += 32) {
      VID v = u_adj[ptr];

      if (xadj[v+1]-xadj[v] > degu || (xadj[v+1]-xadj[v] == degu && v < u) ){
        EN other_ptr=EN(MAX_VAL), intersection_size = 0;

        EN ptr_u = 0;
        EN ptr_v = xadj[v];

        while(ptr_u < degu && ptr_v < xadj[v+1]) {
          VID u_ngh = u_adj[ptr_u];

          VID v_ngh = adj[ptr_v];
          if(v_ngh == u) {
            other_ptr = ptr_v;
          }

          if(u_ngh == v_ngh) {
            intersection_size++;
            ptr_u++; ptr_v++;
          } else if(u_ngh < v_ngh) {
            ptr_u++;
          } else {
            ptr_v++;
          }
        }
        if(other_ptr == EN(MAX_VAL)){
          while(adj[ptr_v] != u) ptr_v++;
          other_ptr = ptr_v;
        }
#ifdef AOS
      EN eu =  xadj[u+1] - xadj[u];
      EN ev =  xadj[v+1] - xadj[v];
      emetrics[(xadj[u] + ptr)] = emetrics[other_ptr] = float(intersection_size)/float(eu+ev-intersection_size);
#elif defined SOA
      EN eu =  xadj[u+1] - xadj[u];
      EN ev =  xadj[v+1] - xadj[v];
      emetrics[(xadj[u] + ptr)] = emetrics[other_ptr] = float(intersection_size)/float(eu+ev-intersection_size);
#endif
      }
    }    
  }
}
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_warp_sm_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  start = omp_get_wtime();
/*
  int no_threads = 256;
  int no_blocks = 1024;
*/
  int no_threads =block.x;
  unsigned long long shared_memory = SM_FAC * (no_threads / 32) * sizeof(VID);
  jac_binning_gpu_u_per_warp_sm_kernel<directed, EN, VID, E><<<grid, block, shared_memory>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, SM_FAC, lower_limit, upper_limit);
  gpuErrchk( cudaGetLastError());
  cudaDeviceSynchronize();

  end = omp_get_wtime();
  return end-start;

}

/*
template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_gpu_u_per_block_sm_kernel(const EN* __restrict__ xadj, const VID* __restrict__ adj, const EN* __restrict__ xadj_start, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {
  int no_threads = blockDim.y * blockDim.x * gridDim.x; //multiple of 32
  int tid = blockDim.y * blockDim.x * blockDim.z * blockIdx.x + blockDim.x * threadIdx.y + threadIdx.x;
  //if (tid==0)printf("Running jac_binning_gpu_u_per_block_sm_kernel\n");
  int block_local_wid = (tid % (blockDim.x * blockDim.y)) / 32;
  int warp_local_tid = tid % 32;

  extern __shared__ VID glob_u_adj[];
  int grid_local_wid = tid / 32;;
  int grid_no_warps = no_threads / 32;

  for(VID t = grid_local_wid; t < bin_size; t += grid_no_warps) {	
    VID u = bin[t];
    EN degu = xadj[u+1] - xadj[u];

    VID* u_adj = glob_u_adj + (block_local_wid * SM_FAC);
    for(EN ptr = warp_local_tid; ptr < degu; ptr += 32) {
      u_adj[ptr] = adj[xadj[u] + ptr]; 
    }
    __syncthreads();

#ifdef PSEUDO_SORT
    for(EN ptr = warp_local_tid; ptr < degu; ptr += 32) {
#else
    for(EN ptr = warp_local_tid + xadj_start[u]; ptr < degu; ptr += 32) {
#endif
      VID v = u_adj[ptr];

#ifdef PSEUDO_SORT
      if(xadj[v+1]-xadj[v] >= lower_limit && (xadj[v+1]-xadj[v]>= upper_limit ||(xadj[v+1]-xadj[v]< upper_limit && u<v))) {
     //if(xadj[v+1]-xadj[v] >= lower_limit) {
#endif
        EN other_ptr, intersection_size = 0;

        EN ptr_u = 0;
        EN ptr_v = xadj[v];

        while(ptr_u < degu && ptr_v < xadj[v+1]) {
          VID u_ngh = u_adj[ptr_u];

          VID v_ngh = adj[ptr_v];
          if(v_ngh == u) {
            other_ptr = ptr_v;
          }

          if(u_ngh == v_ngh) {
            intersection_size++;
            ptr_u++; ptr_v++;
          } else if(u_ngh < v_ngh) {
            ptr_u++;
          } else {
            ptr_v++;
          }
        }
#ifdef AOS
      EN eu =  xadj[u+1] -xadj[u];
      EN ev =  xadj[v+1] -xadj[v];
      emetrics[(xadj[u] + ptr)] = emetrics[other_ptr] = float(intersection_size)/float(eu+ev-intersection_size);
#elif defined SOA
      EN eu =  xadj[u+1] -xadj[u];
      EN ev =  xadj[v+1] -xadj[v];
      emetrics[(xadj[u] + ptr)] = emetrics[other_ptr] = float(intersection_size)/float(eu+ev-intersection_size);
#endif
#ifdef PSEUDO_SORT
      }
#endif
    }    
    __syncthreads();
  }
}
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_block_sm_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  start = omp_get_wtime();

  unsigned long long size_of_shared_memory = SM_FAC*sizeof(VID);
  jac_binning_gpu_u_per_block_sm_kernel<directed, EN, VID, E><<<grid, block, size_of_shared_memory>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, SM_FAC, lower_limit, upper_limit);
  gpuErrchk( cudaGetLastError());
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  return end-start;

}
*/
#include <stdio.h>
#include <stdint.h>

static __device__ __inline__ uint32_t __mysmid(){
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;}

static __device__ __inline__ uint32_t __mywarpid(){
  uint32_t warpid;
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
  return warpid;}

static __device__ __inline__ uint32_t __mylaneid(){
  uint32_t laneid;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
  return laneid;}

template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_gpu_u_per_block_bst_kernel(const EN* __restrict__ xadj, const VID* __restrict__ adj, const VID* __restrict xadj_start, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {
  
  //int no_threads = blockDim.z * blockDim.y * blockDim.x * gridDim.x;
  int block_local_id =  blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  unsigned long long tid = blockDim.z * blockDim.x * blockDim.y * blockIdx.x + block_local_id;
  //if (tid==0)printf("Running jac_binning_gpu_u_per_block_bst_kernel\n");
  //int block_local_id = blockDim.x * threadIdx.y + threadIdx.x;
  //int tid = blockDim.x * blockDim.y * blockIdx.x + block_local_id;
  //printf("block_local_id %d tid %d threadidx x %d y %d z %d blockidx x %d y %d z %d\n", block_local_id, tid, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);


  unsigned mask = calculateMask(blockDim.x, tid);//threadIdx.y*blockDim.x+threadIdx.x);
  for(EN ptr = blockIdx.x; ptr < bin_size; ptr += gridDim.x) {
    EN u = bin[ptr];
    EN degu = xadj[u+1] - xadj[u];

    //for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z + xadj_start[u]; neigh_ptr< degu; neigh_ptr+=blockDim.y * blockDim.z){
    for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z; neigh_ptr< degu; neigh_ptr+=blockDim.y * blockDim.z){
      VID v = adj[neigh_ptr+xadj[u]];
      if (xadj[v+1]-xadj[v] > degu || (xadj[v+1]-xadj[v] == degu && v < u) ){
        EN intersection_size = 0;
        EN other_ptr = bst(xadj, adj, v, u);

        for (EN t_ptr = threadIdx.x; t_ptr < degu; t_ptr+=blockDim.x){
          EN loc = bst(xadj, adj, v, adj[xadj[u]+t_ptr]);
          intersection_size+=(loc!=(EN)-1);
        } 
        intersection_size = warpReduce(intersection_size, blockDim.x, mask);

        //intersection_size = warpReduce(intersection_size, blockDim.x, tid%32, threadIdx.x==0);
        if (threadIdx.x == 0){
          emetrics[(xadj[u] + neigh_ptr)] = emetrics[other_ptr] = float(intersection_size)/float(degu+(xadj[v+1]-xadj[v])-intersection_size);

        }
      }
    }
//    __syncthreads();
  }
}

// Driver function that will call the u_per_block_bst kernel
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_block_bst_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  start = omp_get_wtime();
/*
  dim3 block;
  block.x = 8;
  block.y = 4;
  block.z = 4;
  int no_blocks = 2048;
*/
  jac_binning_gpu_u_per_block_bst_kernel<directed, EN, VID, E><<<grid, block>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, SM_FAC, lower_limit, upper_limit);
  gpuErrchk( cudaGetLastError());
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  return end-start;

}

template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_gpu_u_per_subwarp_bst_sm_kernel(const EN* __restrict__ xadj, const VID* __restrict__ adj, const EN* __restrict__ xadj_start, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {
  
  //int no_threads = blockDim.z * blockDim.y * blockDim.x * gridDim.x;
  int block_local_id =  blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  unsigned int grid_id = gridDim.x*gridDim.y*blockIdx.z+gridDim.x*blockIdx.y+blockIdx.x;
  unsigned long long tid = (long long)blockDim.z * blockDim.x * blockDim.y * grid_id + (unsigned long long)block_local_id;
  int block_local_wid = (tid % (blockDim.x * blockDim.y)) / 32;
  int warp_local_tid = tid % 32;
  //unsigned long long tid = blockIdx.x * blockDim.x * blockDim.y * blockDim.z;  
  unsigned mask = calculateMask(blockDim.x, tid);//threadIdx.y*blockDim.x+threadIdx.x);
  extern __shared__ VID glob_u_adj[]; 

  VID* u_adj = glob_u_adj + (block_local_wid * SM_FAC);

  for(EN ptr = blockIdx.y*gridDim.x*blockDim.z+blockIdx.x*blockDim.z+threadIdx.z; ptr < bin_size; ptr += gridDim.y * gridDim.x*blockDim.z) {
    EN u = bin[ptr];
    EN degu = xadj[u+1] - xadj[u];
    __syncwarp(mask);
    for(EN ptr = warp_local_tid; ptr < degu; ptr += 32) {
      u_adj[ptr] = adj[xadj[u] + ptr]; 
    }
    __syncwarp(mask);

    for (EN neigh_ptr = threadIdx.y; neigh_ptr< degu; neigh_ptr+=blockDim.y){
    //for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z + blockIdx.x*blockDim.y*blockDim.z + xadj_start[u]; neigh_ptr< degu; neigh_ptr+=blockDim.y * blockDim.z*gridDim.x){
      VID v = u_adj[neigh_ptr];
      if (xadj[v+1]-xadj[v] > degu || (xadj[v+1]-xadj[v] == degu && v < u) ){
        EN intersection_size = 0;
        EN other_ptr = bst(xadj, adj, v, u);

        for (EN t_ptr = threadIdx.x; t_ptr < degu; t_ptr+=blockDim.x){
          EN loc = bst(xadj, adj, v, u_adj[t_ptr]);
          intersection_size+=(loc!=(EN)-1);
        } 

        intersection_size = warpReduce(intersection_size, blockDim.x, mask);
        //intersection_size = warpReduce(intersection_size, blockDim.x, tid%32, threadIdx.x==0);
        if (threadIdx.x == 0){
          emetrics[(xadj[u] + neigh_ptr)] = emetrics[other_ptr] = float(intersection_size)/float(degu+(xadj[v+1]-xadj[v])-intersection_size);

        }
      }
    }
//    __syncthreads();
  }
}
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_subwarp_bst_sm_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  start = omp_get_wtime();
/*
  dim3 block;
  block.x = 8;
  block.y = 4;
  block.z = 4;
  int no_blocks = 2048;
*/
  grid.x = min(MAX_GRID_DIM,max(1,bin_size/block.z));
  grid.y = min(MAX_GRID_DIM, max(1, bin_size/grid.x/block.z));
  grid.z = 1;
  int no_threads = block.x*block.y*block.z;
  unsigned long long shared_memory = SM_FAC * (no_threads / 32) * sizeof(VID);
  jac_binning_gpu_u_per_subwarp_bst_sm_kernel<directed, EN, VID, E><<<grid, block, shared_memory>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, SM_FAC, lower_limit, upper_limit);
  gpuErrchk( cudaGetLastError());
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  return end-start;

}

template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_gpu_u_per_subwarp_bst_kernel(const EN* __restrict__ xadj, const VID* __restrict__ adj, const EN* __restrict__ xadj_start, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {
  
  //int no_threads = blockDim.z * blockDim.y * blockDim.x * gridDim.x;
  int block_local_id =  blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  unsigned int grid_id = gridDim.x*gridDim.y*blockIdx.z+gridDim.x*blockIdx.y+blockIdx.x;
  unsigned long long tid = (long long)blockDim.z * blockDim.x * blockDim.y * grid_id + (unsigned long long)block_local_id;
  //unsigned long long tid = blockIdx.x * blockDim.x * blockDim.y * blockDim.z;  
  unsigned mask = calculateMask(blockDim.x, tid);//threadIdx.y*blockDim.x+threadIdx.x);


  for(EN ptr = blockIdx.y*gridDim.x*blockDim.z+blockIdx.x*blockDim.z+threadIdx.z; ptr < bin_size; ptr += gridDim.y * gridDim.x*blockDim.z) {
    EN u = bin[ptr];
    EN degu = xadj[u+1] - xadj[u];

    for (EN neigh_ptr = threadIdx.y; neigh_ptr< degu; neigh_ptr+=blockDim.y){
    //for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z + blockIdx.x*blockDim.y*blockDim.z + xadj_start[u]; neigh_ptr< degu; neigh_ptr+=blockDim.y * blockDim.z*gridDim.x){
      VID v = adj[neigh_ptr+xadj[u]];
      if (xadj[v+1]-xadj[v] > degu || (xadj[v+1]-xadj[v] == degu && v < u) ){
        EN intersection_size = 0;
        EN other_ptr = bst(xadj, adj, v, u);

        for (EN t_ptr = threadIdx.x; t_ptr < degu; t_ptr+=blockDim.x){
          EN loc = bst(xadj, adj, v, adj[xadj[u]+t_ptr]);
          intersection_size+=(loc!=(EN)-1);
        } 

        intersection_size = warpReduce(intersection_size, blockDim.x, mask);
        //intersection_size = warpReduce(intersection_size, blockDim.x, tid%32, threadIdx.x==0);
        if (threadIdx.x == 0){
          emetrics[(xadj[u] + neigh_ptr)] = emetrics[other_ptr] = float(intersection_size)/float(degu+(xadj[v+1]-xadj[v])-intersection_size);

        }
      }
    }
//    __syncthreads();
  }
}
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_subwarp_bst_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  start = omp_get_wtime();
/*
  dim3 block;
  block.x = 8;
  block.y = 4;
  block.z = 4;
  int no_blocks = 2048;
*/
  grid.x = min(MAX_GRID_DIM,max(1,bin_size/block.z));
  grid.y = min(MAX_GRID_DIM, max(1, bin_size/grid.x/block.z));
  grid.z = 1;
  jac_binning_gpu_u_per_subwarp_bst_kernel<directed, EN, VID, E><<<grid, block>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, SM_FAC, lower_limit, upper_limit);
  gpuErrchk( cudaGetLastError());
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  return end-start;

}
template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_gpu_edge_based_small_filter(const EN* __restrict__ xadj, const VID* __restrict__ adj, const EN* __restrict__ is, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {
  
  //int no_threads = blockDim.z * blockDim.y * blockDim.x * gridDim.x;
  unsigned int block_local_id =  blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  unsigned int grid_id = gridDim.x*gridDim.y*blockIdx.z+gridDim.x*blockIdx.y+blockIdx.x;
  unsigned long long tid = (long long)blockDim.z * blockDim.x * blockDim.y * grid_id + (unsigned long long)block_local_id;
  unsigned mask = calculateMask(blockDim.x, tid);//threadIdx.y*blockDim.x+threadIdx.x);


  for(EN ptr = threadIdx.y+ blockDim.y*blockIdx.x+ blockDim.y*gridDim.x* blockIdx.y+blockDim.y*blockIdx.z*gridDim.y*gridDim.x ; ptr < bin_size; ptr += gridDim.y*gridDim.z*blockDim.y*gridDim.x) {
    EN v = adj[bin[ptr]];
    EN u = is[bin[ptr]];
    EN degu = xadj[u+1] - xadj[u];
    bool skippable = (xadj[v+1]-xadj[v] < degu ||(xadj[v+1]-xadj[v] == degu && v > u));
    if (!directed &&skippable)
      continue;
    EN other_ptr = bst(xadj, adj, v, u);
    if (directed && other_ptr != (EN)-1 && skippable)
      continue;

    EN intersection_size = 0;

    for (EN t_ptr = threadIdx.x; t_ptr < degu; t_ptr+=blockDim.x){
      EN loc = bst(xadj, adj, v, adj[xadj[u]+t_ptr]);
      intersection_size+=(loc!=(EN)-1);
    } 

    intersection_size = warpReduce(intersection_size, blockDim.x, mask);
    if (threadIdx.x == 0){
      E J =float(intersection_size)/float(degu+(xadj[v+1]-xadj[v])-intersection_size);
      emetrics[bin[ptr]] = J;
      if (other_ptr != (EN) -1)emetrics[other_ptr] = J;
    }
  }
}
template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_gpu_edge_based_small(const EN* __restrict__ xadj, const VID* __restrict__ adj, const EN* __restrict__ is, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {
  
  //int no_threads = blockDim.z * blockDim.y * blockDim.x * gridDim.x;
  unsigned int block_local_id =  blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  unsigned int grid_id = gridDim.x*gridDim.y*blockIdx.z+gridDim.x*blockIdx.y+blockIdx.x;
  unsigned long long tid = (long long)blockDim.z * blockDim.x * blockDim.y * grid_id + (unsigned long long)block_local_id;
  unsigned mask = calculateMask(blockDim.x, tid);//threadIdx.y*blockDim.x+threadIdx.x);


  for(EN ptr = threadIdx.y+ blockDim.y* blockIdx.y+blockDim.y*blockIdx.z*gridDim.y; ptr < bin_size; ptr += gridDim.y*gridDim.z*blockDim.y) {
    EN v = adj[bin[ptr]];
    EN u = is[bin[ptr]];
    EN degu = xadj[u+1] - xadj[u];

    EN intersection_size = 0;
    EN other_ptr = bst(xadj, adj, v, u);

    for (EN t_ptr = threadIdx.x; t_ptr < degu; t_ptr+=blockDim.x){
      EN loc = bst(xadj, adj, v, adj[xadj[u]+t_ptr]);
      intersection_size+=(loc!=(EN)-1);
    } 

    intersection_size = warpReduce(intersection_size, blockDim.x, mask);
    if (threadIdx.x == 0){
      E J =float(intersection_size)/float(degu+(xadj[v+1]-xadj[v])-intersection_size);
      emetrics[bin[ptr]] = J;
      if (other_ptr != (EN) -1)emetrics[other_ptr] = J;
    }
  }
}
template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_atomic_twoarray_kernel(const EN* __restrict__ xadj, const VID* __restrict__ adj, EN* __restrict__ sum_d, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {
  
  //int no_threads = blockDim.z * blockDim.y * blockDim.x * gridDim.x;


  for(EN ptr = blockIdx.y+blockIdx.z*gridDim.y; ptr < bin_size; ptr += gridDim.y*gridDim.z) {
    EN u = bin[ptr];
    EN degu = xadj[u+1] - xadj[u];

    for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z + blockIdx.x*blockDim.y*blockDim.z; neigh_ptr< degu; neigh_ptr+=blockDim.y * blockDim.z*gridDim.x){
      VID v = adj[neigh_ptr+xadj[u]];
      EN intersection_size = 0;
      bool skippable = (xadj[v+1]-xadj[v] < degu ||(xadj[v+1]-xadj[v] == degu && v > u));
      if (!directed &&skippable)
        continue;
      EN other_ptr = bst(xadj, adj, v, u);
      if (directed && other_ptr != (EN)-1 && skippable)
        continue;

      for (EN t_ptr = threadIdx.x; t_ptr < degu; t_ptr+=blockDim.x){
        EN loc = bst(xadj, adj, v, adj[xadj[u]+t_ptr]);
        intersection_size+=(loc!=(EN)-1);
      } 
      sum_d[(xadj[u]+neigh_ptr)] = degu + xadj[v+1]-xadj[v];
      atomicAdd(&emetrics[(xadj[u] + neigh_ptr)], intersection_size);
      if (other_ptr != (EN)-1){
        atomicAdd(&emetrics[other_ptr], intersection_size);
        sum_d[other_ptr] = degu + xadj[v+1]-xadj[v];
      }
    }
  }
}

template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_atomic_kernel(const EN* __restrict__ xadj, const VID* __restrict__ adj, const EN* __restrict__ xadj_start, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {
  
  //int no_threads = blockDim.z * blockDim.y * blockDim.x * gridDim.x;


  for(EN ptr = blockIdx.y+blockIdx.z*gridDim.y; ptr < bin_size; ptr += gridDim.y*gridDim.z) {
    EN u = bin[ptr];
    EN degu = xadj[u+1] - xadj[u];

    for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z + blockIdx.x*blockDim.y*blockDim.z; neigh_ptr< degu; neigh_ptr+=blockDim.y * blockDim.z*gridDim.x){
      VID v = adj[neigh_ptr+xadj[u]];
      EN intersection_size = 0;
      bool skippable = (xadj[v+1]-xadj[v] < degu ||(xadj[v+1]-xadj[v] == degu && v > u));
      if (!directed &&skippable)
        continue;
      EN other_ptr = bst(xadj, adj, v, u);
      if (directed && other_ptr != (EN)-1 && skippable)
        continue;

      for (EN t_ptr = threadIdx.x; t_ptr < degu; t_ptr+=blockDim.x){
        EN loc = bst(xadj, adj, v, adj[xadj[u]+t_ptr]);
        intersection_size+=(loc!=(EN)-1);
      } 
      atomicAdd(&emetrics[(xadj[u] + neigh_ptr)], intersection_size);
      if (other_ptr != (EN)-1)
        atomicAdd(&emetrics[other_ptr], intersection_size);
    }
  }
}
template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_gpu_u_per_grid_bst_kernel(const EN* __restrict__ xadj, const VID* __restrict__ adj, const EN* __restrict__ xadj_start, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {
  
  //int no_threads = blockDim.z * blockDim.y * blockDim.x * gridDim.x;
  unsigned int block_local_id =  blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  unsigned int grid_id = gridDim.x*gridDim.y*blockIdx.z+gridDim.x*blockIdx.y+blockIdx.x;
  unsigned long long tid = (long long)blockDim.z * blockDim.x * blockDim.y * grid_id + (unsigned long long)block_local_id;
  unsigned mask = calculateMask(blockDim.x, tid);//threadIdx.y*blockDim.x+threadIdx.x);


  for(EN ptr = blockIdx.y+blockIdx.z*gridDim.y; ptr < bin_size; ptr += gridDim.y*gridDim.z) {
    EN u = bin[ptr];
    EN degu = xadj[u+1] - xadj[u];

    for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z + blockIdx.x*blockDim.y*blockDim.z; neigh_ptr< degu; neigh_ptr+=blockDim.y * blockDim.z*gridDim.x){
    //for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z + blockIdx.x*blockDim.y*blockDim.z + xadj_start[u]; neigh_ptr< degu; neigh_ptr+=blockDim.y * blockDim.z*gridDim.x){
      VID v = adj[neigh_ptr+xadj[u]];
      bool skippable = (xadj[v+1]-xadj[v] < degu ||(xadj[v+1]-xadj[v] == degu && v > u));
      if (!directed &&skippable)
        continue;
      EN other_ptr = bst(xadj, adj, v, u);
      if (directed && other_ptr != (EN)-1 && skippable)
        continue;
      EN intersection_size = 0;

      for (EN t_ptr = threadIdx.x; t_ptr < degu; t_ptr+=blockDim.x){
        EN loc = bst(xadj, adj, v, adj[xadj[u]+t_ptr]);
        intersection_size+=(loc!=(EN)-1);
      } 

      intersection_size = warpReduce(intersection_size, blockDim.x, mask);
      //intersection_size = warpReduce(intersection_size, blockDim.x, tid%32, threadIdx.x==0);
      if (threadIdx.x == 0){
        E J = float(intersection_size)/float(degu+(xadj[v+1]-xadj[v])-intersection_size);
        emetrics[(xadj[u] + neigh_ptr)] = J;
        if (other_ptr != (EN)-1) emetrics[other_ptr] = J;

      }
    }
    //    __syncthreads();
    }
  }

  template <bool directed, typename EN, typename VID, typename E>
    __global__ void jac_binning_gpu_edge_based_filter(const EN* __restrict__ xadj, const VID* __restrict__ adj, const EN* __restrict__ is, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {

      //int no_threads = blockDim.z * blockDim.y * blockDim.x * gridDim.x;
  int block_local_id =  blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  unsigned int grid_id = gridDim.x*gridDim.y*blockIdx.z+gridDim.x*blockIdx.y+blockIdx.x;
  unsigned long long tid = (long long)blockDim.z * blockDim.x * blockDim.y * grid_id + (unsigned long long)block_local_id;
  unsigned mask = calculateMask(blockDim.x, tid);//threadIdx.y*blockDim.x+threadIdx.x);

  //extern __shared__ VID glob_inter[];
  extern __shared__ VID glob_inter_fornow[];
  VID* glob_inter = glob_inter_fornow + (blockDim.x /WARP_SIZE)*(threadIdx.y+threadIdx.z*blockDim.y);

  for(EN ptr = blockIdx.x + blockIdx.y*gridDim.x+blockIdx.z*gridDim.y*gridDim.x; ptr < bin_size; ptr += gridDim.y*gridDim.z*gridDim.x) {
    EN v = adj[bin[ptr]];
    EN u = is[bin[ptr]];
    EN degu = xadj[u+1] - xadj[u];

    EN intersection_size = 0;
    bool skippable = (xadj[v+1]-xadj[v] < degu ||(xadj[v+1]-xadj[v] == degu && v > u));
    if (!directed &&skippable)
      continue;
    EN other_ptr = bst(xadj, adj, v, u);
    if (directed && other_ptr != (EN)-1 && skippable)
      continue;

    for (EN t_ptr = threadIdx.x; t_ptr < degu; t_ptr+=blockDim.x){
      EN loc = bst(xadj, adj, v, adj[xadj[u]+t_ptr]);
      intersection_size+=(loc!=(EN)-1);
    } 

    intersection_size = warpReduce(intersection_size, min(blockDim.x, WARP_SIZE), mask);
    if (threadIdx.x % WARP_SIZE == 0 && blockDim.x > 32) 
      glob_inter[block_local_id/WARP_SIZE] = intersection_size;
    //intersection_size = warpReduce(intersection_size, blockDim.x, tid%32, threadIdx.x==0);
    __syncthreads();
    if (threadIdx.x == 0){
      for (int i =1; i < blockDim.x/WARP_SIZE; i++) intersection_size+=glob_inter[i];
      E J =float(intersection_size)/float(degu+(xadj[v+1]-xadj[v])-intersection_size);
      emetrics[bin[ptr]] = J;
      if (other_ptr != (EN)-1) emetrics[other_ptr] = J;
    }
    __syncthreads();
  }
}
template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_gpu_edge_based(const EN* __restrict__ xadj, const VID* __restrict__ adj, const EN* __restrict__ is, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {
  
  //int no_threads = blockDim.z * blockDim.y * blockDim.x * gridDim.x;
  int block_local_id =  blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  unsigned int grid_id = gridDim.x*gridDim.y*blockIdx.z+gridDim.x*blockIdx.y+blockIdx.x;
  unsigned long long tid = (long long)blockDim.z * blockDim.x * blockDim.y * grid_id + (unsigned long long)block_local_id;
  unsigned mask = calculateMask(blockDim.x, tid);//threadIdx.y*blockDim.x+threadIdx.x);

  //extern __shared__ VID glob_inter[];
  extern __shared__ VID glob_inter_fornow[];
  VID* glob_inter = glob_inter_fornow + (blockDim.x /WARP_SIZE)*(threadIdx.y+threadIdx.z*blockDim.y);

  for(EN ptr = blockIdx.y+blockIdx.z*gridDim.y; ptr < bin_size; ptr += gridDim.y*gridDim.z) {
    EN v = adj[bin[ptr]];
    EN u = is[bin[ptr]];
    EN degu = xadj[u+1] - xadj[u];

    EN intersection_size = 0;
    EN other_ptr = bst(xadj, adj, v, u);

    for (EN t_ptr = threadIdx.x; t_ptr < degu; t_ptr+=blockDim.x){
      EN loc = bst(xadj, adj, v, adj[xadj[u]+t_ptr]);
      intersection_size+=(loc!=(EN)-1);
    } 

    intersection_size = warpReduce(intersection_size, min(blockDim.x, WARP_SIZE), mask);
    if (threadIdx.x % WARP_SIZE == 0 && blockDim.x > 32) 
      glob_inter[block_local_id/WARP_SIZE] = intersection_size;
    //intersection_size = warpReduce(intersection_size, blockDim.x, tid%32, threadIdx.x==0);
    __syncthreads();
    if (threadIdx.x == 0){
      for (int i =1; i < blockDim.x/WARP_SIZE; i++) intersection_size+=glob_inter[i];
      E J =float(intersection_size)/float(degu+(xadj[v+1]-xadj[v])-intersection_size);
      emetrics[bin[ptr]] = J;
      if (other_ptr != (EN)-1) emetrics[other_ptr] = J;

    }
    __syncthreads();
  }
}

template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_gpu_u_per_grid_bst_bigsgroup_sm_kernel_profiled(const EN* __restrict__ xadj, const VID* __restrict__ adj, const EN* __restrict__ xadj_start, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {
  
  //int no_threads = blockDim.z * blockDim.y * blockDim.x * gridDim.x;
  int block_local_id =  blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  unsigned int grid_id = gridDim.x*gridDim.y*blockIdx.z+gridDim.x*blockIdx.y+blockIdx.x;
  unsigned long long tid = (long long)blockDim.z * blockDim.x * blockDim.y * grid_id + (unsigned long long)block_local_id;
  unsigned mask = calculateMask(blockDim.x, tid);//threadIdx.y*blockDim.x+threadIdx.x);

  extern __shared__ VID glob_inter_fornow[];
  VID* glob_inter = glob_inter_fornow + (blockDim.x /WARP_SIZE)*(threadIdx.y+threadIdx.z*blockDim.y);

  for(EN ptr = blockIdx.y+blockIdx.z*gridDim.y; ptr < bin_size; ptr += gridDim.y*gridDim.z) {
    EN u = bin[ptr];
    EN startidx = xadj[u];
    EN degu = xadj[u+1] - startidx;

    for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z + blockIdx.x*blockDim.y*blockDim.z; neigh_ptr< degu; neigh_ptr+=blockDim.y * blockDim.z*gridDim.x){
    //for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z + blockIdx.x*blockDim.y*blockDim.z + xadj_start[u]; neigh_ptr< degu; neigh_ptr+=blockDim.y * blockDim.z*gridDim.x){
      VID v = adj[neigh_ptr+startidx];
      bool skippable = (xadj[v+1]-xadj[v] < degu ||(xadj[v+1]-xadj[v] == degu && v > u));
      if (!directed &&skippable)
        continue;
      EN other_ptr; 
      if (threadIdx.x == 0) other_ptr = bst(xadj, adj, v, u);
      if (directed && other_ptr != (EN)-1 && skippable)
        continue;
      EN intersection_size = 0;

      for (EN t_ptr = threadIdx.x; t_ptr < degu; t_ptr+=blockDim.x){
        EN loc = bst(xadj, adj, v, adj[startidx+t_ptr]);
        intersection_size+=(loc!=(EN)-1);
      } 

      intersection_size = warpReduce(intersection_size, min(blockDim.x, WARP_SIZE), mask);
      if (threadIdx.x % WARP_SIZE == 0 && blockDim.x > 32) 
        glob_inter[block_local_id/WARP_SIZE] = intersection_size;
      //intersection_size = warpReduce(intersection_size, blockDim.x, tid%32, threadIdx.x==0);
      __syncthreads();
      if (threadIdx.x == 0){
        for (int i =1; i < blockDim.x/WARP_SIZE; i++) intersection_size+=glob_inter[i];
        E J = float(intersection_size)/float(degu+(xadj[v+1]-xadj[v])-intersection_size);
        emetrics[(startidx + neigh_ptr)] = J;
        if (other_ptr != (EN)-1)
          emetrics[other_ptr] = J;

      }
      __syncthreads();
    }
//    __syncthreads();
  }
}


template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_gpu_u_per_grid_bst_bigsgroup_sm_noskipping_kernel(const EN* __restrict__ xadj, const VID* __restrict__ adj, const EN* __restrict__ xadj_start, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {
  
  //int no_threads = blockDim.z * blockDim.y * blockDim.x * gridDim.x;
  int block_local_id =  blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  unsigned int grid_id = gridDim.x*gridDim.y*blockIdx.z+gridDim.x*blockIdx.y+blockIdx.x;
  unsigned long long tid = (long long)blockDim.z * blockDim.x * blockDim.y * grid_id + (unsigned long long)block_local_id;
  unsigned mask = calculateMask(blockDim.x, tid);//threadIdx.y*blockDim.x+threadIdx.x);

  extern __shared__ VID glob_inter_fornow[];
  VID* glob_inter = glob_inter_fornow + (blockDim.x /WARP_SIZE)*(threadIdx.y+threadIdx.z*blockDim.y);

  for(EN ptr = blockIdx.y+blockIdx.z*gridDim.y; ptr < bin_size; ptr += gridDim.y*gridDim.z) {
    EN uu = bin[ptr];
    EN deguu = xadj[uu+1] - xadj[uu];

    for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z + blockIdx.x*blockDim.y*blockDim.z; neigh_ptr< deguu; neigh_ptr+=blockDim.y * blockDim.z*gridDim.x){
      VID vv = adj[neigh_ptr+xadj[uu]];
      bool skippable = (xadj[vv+1]-xadj[vv] < deguu);
      VID u = (skippable) ? vv : uu;
      VID v = (skippable) ? uu : vv;
      VID degu = xadj[u+1] - xadj[u];
      //if (!directed &&skippable)
      //  continue;
      EN other_ptr;
      if (threadIdx.x == 0) other_ptr = bst(xadj, adj, v, u);
      //if (directed && other_ptr != (EN)-1 && skippable)
      //  continue;
      EN intersection_size = 0;

      for (EN t_ptr = threadIdx.x; t_ptr < degu; t_ptr+=blockDim.x){
        EN loc = bst(xadj, adj, v, adj[xadj[u]+t_ptr]);
        intersection_size+=(loc!=(EN)-1);
      } 

      intersection_size = warpReduce(intersection_size, min(blockDim.x, WARP_SIZE), mask);
      if (threadIdx.x % WARP_SIZE == 0 && blockDim.x > 32) 
        glob_inter[block_local_id/WARP_SIZE] = intersection_size;
      //intersection_size = warpReduce(intersection_size, blockDim.x, tid%32, threadIdx.x==0);
      __syncthreads();
      if (threadIdx.x == 0){
        for (int i =1; i < blockDim.x/WARP_SIZE; i++) intersection_size+=glob_inter[i];
        E J = float(intersection_size)/float(degu+(xadj[v+1]-xadj[v])-intersection_size);
        emetrics[(xadj[uu] + neigh_ptr)] = J;
        if (other_ptr != (EN)-1)
          emetrics[other_ptr] = J;

      }
      __syncthreads();
    }
  }
}

template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_gpu_u_per_grid_bst_bigsgroup_sm_kernel(const EN* __restrict__ xadj, const VID* __restrict__ adj, const EN* __restrict__ xadj_start, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {
  
  //int no_threads = blockDim.z * blockDim.y * blockDim.x * gridDim.x;
  int block_local_id =  blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  unsigned int grid_id = gridDim.x*gridDim.y*blockIdx.z+gridDim.x*blockIdx.y+blockIdx.x;
  unsigned long long tid = (long long)blockDim.z * blockDim.x * blockDim.y * grid_id + (unsigned long long)block_local_id;
  unsigned mask = calculateMask(blockDim.x, tid);//threadIdx.y*blockDim.x+threadIdx.x);

  extern __shared__ VID glob_inter_fornow[];
  VID* glob_inter = glob_inter_fornow + (blockDim.x /WARP_SIZE)*(threadIdx.y+threadIdx.z*blockDim.y);

  for(EN ptr = blockIdx.y+blockIdx.z*gridDim.y; ptr < bin_size; ptr += gridDim.y*gridDim.z) {
    EN u = bin[ptr];
    EN degu = xadj[u+1] - xadj[u];

    for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z + blockIdx.x*blockDim.y*blockDim.z; neigh_ptr< degu; neigh_ptr+=blockDim.y * blockDim.z*gridDim.x){
    //for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z + blockIdx.x*blockDim.y*blockDim.z + xadj_start[u]; neigh_ptr< degu; neigh_ptr+=blockDim.y * blockDim.z*gridDim.x){
      VID v = adj[neigh_ptr+xadj[u]];
      bool skippable = (xadj[v+1]-xadj[v] < degu ||(xadj[v+1]-xadj[v] == degu && v > u));
      if (!directed &&skippable)
        continue;
      EN other_ptr = bst(xadj, adj, v, u);
      if (directed && other_ptr != (EN)-1 && skippable)
        continue;
      EN intersection_size = 0;

      for (EN t_ptr = threadIdx.x; t_ptr < degu; t_ptr+=blockDim.x){
        EN loc = bst(xadj, adj, v, adj[xadj[u]+t_ptr]);
        intersection_size+=(loc!=(EN)-1);
      } 

      intersection_size = warpReduce(intersection_size, min(blockDim.x, WARP_SIZE), mask);
      if (threadIdx.x % WARP_SIZE == 0 && blockDim.x > 32) 
        glob_inter[block_local_id/WARP_SIZE] = intersection_size;
      //intersection_size = warpReduce(intersection_size, blockDim.x, tid%32, threadIdx.x==0);
      __syncthreads();
      if (threadIdx.x == 0){
        for (int i =1; i < blockDim.x/WARP_SIZE; i++) intersection_size+=glob_inter[i];
        E J = float(intersection_size)/float(degu+(xadj[v+1]-xadj[v])-intersection_size);
        emetrics[(xadj[u] + neigh_ptr)] = J;
        if (other_ptr != (EN)-1)
          emetrics[other_ptr] = J;

      }
      __syncthreads();
    }
//    __syncthreads();
  }
}

string get_kernel_name(string prefix, int range, int g, int a, dim3 grid, dim3 block, int sm){
  return prefix+"-"+"g"+to_string(g)+"-a"+to_string(a)+"_max-"+to_string(range)+"_grid("+to_string(grid.x)+","+to_string(grid.y)+","+to_string(grid.z)+")"+"_block("+to_string(block.x)+","+to_string(block.y)+","+to_string(block.z)+")"+"_sm"+to_string(sm);
}
template <bool directed, typename EN, typename VID, typename E>
string k9_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, size_t range, size_t g, size_t a, cudaStream_t * stream){

  dim3 block(1,1,1);
  block.x = g;

  dim3 grid(1,1,1);
  grid.x = a;
  // Maximize blocks so that each source assembly handles the minimum number of sources
  grid.y = max(min(bin_size, MAX_GRID_DIM),1);
  grid.z = min(max(bin_size/grid.y,1), MAX_GRID_DIM);
  unsigned int size_of_shared_memory = block.x/WARP_SIZE*sizeof(EN)*block.y*block.z;
  if (stream == nullptr){
    jac_binning_gpu_u_per_grid_bst_bigsgroup_sm_noskipping_kernel<directed, EN, VID, E><<<grid, block, size_of_shared_memory>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  } else {
    jac_binning_gpu_u_per_grid_bst_bigsgroup_sm_noskipping_kernel<directed, EN, VID, E><<<grid, block, size_of_shared_memory, *stream>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  }
  gpuErrchk( cudaGetLastError());
  string name = get_kernel_name("k2", range, g, a, grid, block, size_of_shared_memory);
  return name;
}
template <bool directed, typename EN, typename VID, typename E>
string k2_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, size_t range, size_t g, size_t a, cudaStream_t * stream){

  dim3 block(1,1,1);
  block.x = g;

  dim3 grid(1,1,1);
  grid.x = a;
  // Maximize blocks so that each source assembly handles the minimum number of sources
  grid.y = max(min(bin_size, MAX_GRID_DIM),1);
  grid.z = min(max(bin_size/grid.y,1), MAX_GRID_DIM);
  unsigned int size_of_shared_memory = block.x/WARP_SIZE*sizeof(EN)*block.y*block.z;
  if (stream == nullptr){
    jac_binning_gpu_u_per_grid_bst_bigsgroup_sm_kernel<directed, EN, VID, E><<<grid, block, size_of_shared_memory>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  } else {
    jac_binning_gpu_u_per_grid_bst_bigsgroup_sm_kernel<directed, EN, VID, E><<<grid, block, size_of_shared_memory, *stream>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  }
  gpuErrchk( cudaGetLastError());
  string name = get_kernel_name("k2", range, g, a, grid, block, size_of_shared_memory);
  return name;
}

template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_grid_bst_bigsgroup_sm_driver_profiled(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  start = omp_get_wtime();
/*
  dim3 block;
  block.x = 8;
  block.y = 4;
  block.z = 4;
  int no_blocks = 2048;
*/
  // Maximize blocks so that each source assembly handles the minimum number of sources
  grid.y = max(min(bin_size, MAX_GRID_DIM),1);
  grid.z = min(max(bin_size/grid.y,1), MAX_GRID_DIM);
  unsigned int size_of_shared_memory = block.x/WARP_SIZE*sizeof(EN)*block.y*block.z;
  jac_binning_gpu_u_per_grid_bst_bigsgroup_sm_kernel_profiled<directed, EN, VID, E><<<grid, block, size_of_shared_memory>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, SM_FAC, lower_limit, upper_limit);
  gpuErrchk( cudaGetLastError());
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  return end-start;

}
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_grid_bst_bigsgroup_sm_noskipping_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  start = omp_get_wtime();
/*
  dim3 block;
  block.x = 8;
  block.y = 4;
  block.z = 4;
  int no_blocks = 2048;
*/
  // Maximize blocks so that each source assembly handles the minimum number of sources
  grid.y = max(min(bin_size, MAX_GRID_DIM),1);
  grid.z = min(max(bin_size/grid.y,1), MAX_GRID_DIM);
  unsigned int size_of_shared_memory = block.x/WARP_SIZE*sizeof(EN)*block.y*block.z;
  jac_binning_gpu_u_per_grid_bst_bigsgroup_sm_noskipping_kernel<directed, EN, VID, E><<<grid, block, size_of_shared_memory>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, SM_FAC, lower_limit, upper_limit);
  gpuErrchk( cudaGetLastError());
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  return end-start;

}
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_grid_bst_bigsgroup_sm_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  start = omp_get_wtime();
/*
  dim3 block;
  block.x = 8;
  block.y = 4;
  block.z = 4;
  int no_blocks = 2048;
*/
  // Maximize blocks so that each source assembly handles the minimum number of sources
  grid.y = max(min(bin_size, MAX_GRID_DIM),1);
  grid.z = min(max(bin_size/grid.y,1), MAX_GRID_DIM);
  unsigned int size_of_shared_memory = block.x/WARP_SIZE*sizeof(EN)*block.y*block.z;
  jac_binning_gpu_u_per_grid_bst_bigsgroup_sm_kernel<directed, EN, VID, E><<<grid, block, size_of_shared_memory>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, SM_FAC, lower_limit, upper_limit);
  gpuErrchk( cudaGetLastError());
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  return end-start;

}
// Driver function that will call the u_per_block_bst kernel
template <bool directed, typename EN, typename VID, typename E>
double fake_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  return 9999999;

}

//jac_binning_gpu_u_per_grid_bst_inv_sm_kernel
template <bool directed, typename EN, typename VID, typename E>
string k7_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, size_t range, size_t g, size_t a, cudaStream_t * stream){
   

  dim3 block(1,1,1);
  block.x = g;

  dim3 grid(1,1,1);
  grid.y = max(min((int)((float)bin_size/a), MAX_GRID_DIM),1);
  grid.z = min(max((int)((float)bin_size/a)/grid.y,1), MAX_GRID_DIM);

  unsigned size_of_shared_memory = sizeof(EN)*block.x/WARP_SIZE;

  if (stream == nullptr){
    jac_binning_gpu_edge_based_filter<directed, EN, VID, E><<<grid, block, size_of_shared_memory>>>(xadj, adj, is, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  } else {
    jac_binning_gpu_edge_based_filter<directed, EN, VID, E><<<grid, block, size_of_shared_memory, *stream>>>(xadj, adj, is, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  }
  gpuErrchk( cudaGetLastError());
  string name = get_kernel_name("k7", range, g, a, grid, block, 0);
  return name;
}
//jac_binning_gpu_u_per_grid_bst_inv_sm_kernel
template <bool directed, typename EN, typename VID, typename E>
string k8_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, size_t range, size_t g, size_t a, cudaStream_t * stream){
   

  dim3 block(1,1,1);
  block.x = g;
  if (g < WARP_SIZE){
    block.y = WARP_SIZE/g;
  }

  dim3 grid(1,1,1);
  grid.y = max((ull)min((ull)((ull)bin_size)/a/block.y, (ull)MAX_GRID_DIM),(ull)1);
  grid.z = min((ull)max((ull)((float)bin_size)/a/block.y/grid.y,(ull)1), (ull)MAX_GRID_DIM);

  if (stream == nullptr){
    jac_binning_gpu_edge_based_small_filter<directed, EN, VID, E><<<grid, block, 0>>>(xadj, adj, is, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  } else {
    jac_binning_gpu_edge_based_small_filter<directed, EN, VID, E><<<grid, block, 0, *stream>>>(xadj, adj, is, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  }
  gpuErrchk( cudaGetLastError());
  string name = get_kernel_name("k8", range, g, a, grid, block, 0);
  return name;
}
//jac_binning_gpu_u_per_grid_bst_inv_sm_kernel
template <bool directed, typename EN, typename VID, typename E>
string k5_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, size_t range, size_t g, size_t a, cudaStream_t * stream){
   

  dim3 block(1,1,1);
  block.x = g;

  dim3 grid(1,1,1);
  grid.y = max(min((int)((float)bin_size/a), MAX_GRID_DIM),1);
  grid.z = min(max((int)((float)bin_size/a)/grid.y,1), MAX_GRID_DIM);

  unsigned size_of_shared_memory = sizeof(EN)*block.x/WARP_SIZE;

  if (stream == nullptr){
    jac_binning_gpu_edge_based<directed, EN, VID, E><<<grid, block, size_of_shared_memory>>>(xadj, adj, is, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  } else {
    jac_binning_gpu_edge_based<directed, EN, VID, E><<<grid, block, size_of_shared_memory, *stream>>>(xadj, adj, is, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  }
  gpuErrchk( cudaGetLastError());
  string name = get_kernel_name("k5", range, g, a, grid, block, 0);
  return name;
}
//jac_binning_gpu_u_per_grid_bst_inv_sm_kernel
template <bool directed, typename EN, typename VID, typename E>
string k6_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, size_t range, size_t g, size_t a, cudaStream_t * stream){
   

  dim3 block(1,1,1);
  block.x = g;
  if (g < WARP_SIZE){
    block.y = WARP_SIZE/g;
  }

  dim3 grid(1,1,1);
  grid.y = max((ull)min((ull)((ull)bin_size)/a/block.y, (ull)MAX_GRID_DIM),(ull)1);
  grid.z = min((ull)max((ull)((float)bin_size)/a/block.y/grid.y,(ull)1), (ull)MAX_GRID_DIM);

  if (stream == nullptr){
    jac_binning_gpu_edge_based_small<directed, EN, VID, E><<<grid, block, 0>>>(xadj, adj, is, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  } else {
    jac_binning_gpu_edge_based_small<directed, EN, VID, E><<<grid, block, 0, *stream>>>(xadj, adj, is, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  }
  gpuErrchk( cudaGetLastError());
  string name = get_kernel_name("k6", range, g, a, grid, block, 0);
  return name;
}
//jac_binning_gpu_u_per_grid_bst_inv_sm_kernel
template <bool directed, typename EN, typename VID, typename E>
string k1_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, size_t range, size_t g, size_t a, cudaStream_t * stream){
   

  dim3 block(1,1,1);
  block.x = g;
  //block.y = max((size_t)1,(size_t)THREADS_PER_BLOCK/block.x);
  if (g < WARP_SIZE){
    block.y = WARP_SIZE/g;
  }

  dim3 grid(1,1,1);
  //grid.x = max((size_t)1,(size_t)a/block.y);
  grid.x =a;
  grid.y = max(min(bin_size, MAX_GRID_DIM),1);
  grid.z = min(max(bin_size/grid.y,1), MAX_GRID_DIM);

  if (stream == nullptr){
    jac_binning_gpu_u_per_grid_bst_kernel<directed, EN, VID, E><<<grid, block, 0>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  } else {
    jac_binning_gpu_u_per_grid_bst_kernel<directed, EN, VID, E><<<grid, block, 0, *stream>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  }
  gpuErrchk( cudaGetLastError());
  string name = get_kernel_name("k1", range, g, a, grid, block, 0);
  return name;
}
template <bool directed, typename EN, typename VID, typename E>
std::tuple<JAC_FUNC_GA<directed, EN, VID, E>, size_t, size_t, size_t> all_k2(int bin_id, size_t range, vector<EN> ranges, vector<pair<unsigned long long, unsigned long long>> bin_sizes, size_t max_threads, size_t max_shared_memory){
  return make_tuple(k2_driver<directed, EN, VID, E>, 32, 8, 0);
}
template <typename EN, typename VID>
unsigned long long get_shared_memory_requirement_for_k4(EN range, int g){
  return (range)*sizeof(VID) +(g/WARP_SIZE) * sizeof(EN);
}
template <typename EN, typename VID>
unsigned long long get_shared_memory_requirement_for_k3(EN range){
  return range*sizeof(VID);
}
template <bool directed, typename EN, typename VID, typename E>
std::tuple<JAC_FUNC_GA<directed, EN, VID, E>, size_t, size_t, size_t> all_k1(int bin_id, size_t range, vector<EN> ranges, vector<pair<unsigned long long, unsigned long long>> bin_sizes, size_t max_threads, size_t max_shared_memory){
  return make_tuple(k1_driver<directed, EN, VID, E>, 32, 8, 0);
}
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_edge_based_filter_edgefilter_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID a, dim3 grid, dim3 block, EN num_v, EN upper_limit){
  double start, end;

  grid.x = a;
  start = omp_get_wtime();
  grid.y = max(min((int)((float)num_v), MAX_GRID_DIM),1);
  grid.z = min(max((int)((float)num_v)/grid.y,1), MAX_GRID_DIM);

  unsigned size_of_shared_memory = sizeof(EN)*block.x/WARP_SIZE;
  jac_binning_gpu_edge_based_filter<directed, EN, VID, E><<<grid, block, size_of_shared_memory>>>(xadj, adj, is, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  gpuErrchk( cudaGetLastError());
  //string name = get_kernel_name("k6", range, g, a, grid, block, 0);
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  return end-start;

}
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_edge_based_filter_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID a, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  grid.x = a;
  start = omp_get_wtime();
  grid.y = max(min((int)((float)bin_size), MAX_GRID_DIM),1);
  grid.z = min(max((int)((float)bin_size)/grid.y,1), MAX_GRID_DIM);

  unsigned size_of_shared_memory = sizeof(EN)*block.x/WARP_SIZE;
  jac_binning_gpu_edge_based_filter<directed, EN, VID, E><<<grid, block, size_of_shared_memory>>>(xadj, adj, is, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  gpuErrchk( cudaGetLastError());
  //string name = get_kernel_name("k6", range, g, a, grid, block, 0);
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  return end-start;

}
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_edge_based_small_filter_edgefilter_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID a, dim3 grid, dim3 block, EN num_v, EN upper_limit){
  double start, end;

  grid.x = a;
  start = omp_get_wtime();
  grid.y = max(min((int)((float)num_v), MAX_GRID_DIM),1);
  grid.z = min(max((int)((float)num_v)/grid.y,1), MAX_GRID_DIM);

  jac_binning_gpu_edge_based_small_filter<directed, EN, VID, E><<<grid, block, 0>>>(xadj, adj, is, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  gpuErrchk( cudaGetLastError());
  //string name = get_kernel_name("k6", range, g, a, grid, block, 0)//;
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  return end-start;

}
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_edge_based_small_filter_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID a, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  grid.x = a;
  start = omp_get_wtime();
  grid.y = max(min((int)((float)bin_size), MAX_GRID_DIM),1);
  grid.z = min(max((int)((float)bin_size)/grid.y,1), MAX_GRID_DIM);

  jac_binning_gpu_edge_based_small_filter<directed, EN, VID, E><<<grid, block, 0>>>(xadj, adj, is, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  gpuErrchk( cudaGetLastError());
  //string name = get_kernel_name("k6", range, g, a, grid, block, 0)//;
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  return end-start;

}
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_edge_based_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID a, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  start = omp_get_wtime();
  grid.y = max(min((int)((float)bin_size)/a, MAX_GRID_DIM),1);
  grid.z = min(max((int)((float)bin_size)/a/grid.y,1), MAX_GRID_DIM);

  unsigned size_of_shared_memory = sizeof(EN)*block.x/WARP_SIZE;
  jac_binning_gpu_edge_based<directed, EN, VID, E><<<grid, block, size_of_shared_memory>>>(xadj, adj, is, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  gpuErrchk( cudaGetLastError());
  //string name = get_kernel_name("k6", range, g, a, grid, block, 0);
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  return end-start;

}
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_edge_based_small_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID a, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  start = omp_get_wtime();
  grid.y = max(min((int)((float)bin_size)/a/block.y, MAX_GRID_DIM),1);
  grid.z = min(max((int)((float)bin_size)/a/block.y/grid.y,1), MAX_GRID_DIM);

  jac_binning_gpu_edge_based_small<directed, EN, VID, E><<<grid, block, 0>>>(xadj, adj, is, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  gpuErrchk( cudaGetLastError());
  //string name = get_kernel_name("k6", range, g, a, grid, block, 0)//;
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  return end-start;

}
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_atomic_twoarray_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  start = omp_get_wtime();
/*
  dim3 block;
  block.x = 8;
  block.y = 4;
  block.z = 4;
  int no_blocks = 2048;
*/
  grid.y = max(min(bin_size, MAX_GRID_DIM),1);
  grid.z = min(max(bin_size/grid.y,1), MAX_GRID_DIM);
  jac_binning_atomic_twoarray_kernel<directed, EN, VID, E><<<grid, block>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, SM_FAC, lower_limit, upper_limit);
  gpuErrchk( cudaGetLastError());
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  return end-start;

}
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_atomic_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  start = omp_get_wtime();
/*
  dim3 block;
  block.x = 8;
  block.y = 4;
  block.z = 4;
  int no_blocks = 2048;
*/
  grid.y = max(min(bin_size, MAX_GRID_DIM),1);
  grid.z = min(max(bin_size/grid.y,1), MAX_GRID_DIM);
  jac_binning_atomic_kernel<directed, EN, VID, E><<<grid, block>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, SM_FAC, lower_limit, upper_limit);
  gpuErrchk( cudaGetLastError());
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  return end-start;

}
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_grid_bst_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  start = omp_get_wtime();
/*
  dim3 block;
  block.x = 8;
  block.y = 4;
  block.z = 4;
  int no_blocks = 2048;
*/
  grid.y = max(min(bin_size, MAX_GRID_DIM),1);
  grid.z = min(max(bin_size/grid.y,1), MAX_GRID_DIM);
  jac_binning_gpu_u_per_grid_bst_kernel<directed, EN, VID, E><<<grid, block>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, SM_FAC, lower_limit, upper_limit);
  gpuErrchk( cudaGetLastError());
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  return end-start;

}
// Each block will use a single u
// All threads on the same y blockIdx will handle the same u
// each of them will do its own bst
// the bst is done on the global memory
// Things to try:
//  - different block sizes (number of warps)
//  - different number of threads doing searching
template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_gpu_u_per_block_bst_sm_kernel(const EN* __restrict__ xadj, const VID* __restrict__ adj, const EN* __restrict__ xadj_start, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {
  
  //int no_threads = blockDim.z * blockDim.y * blockDim.x * gridDim.x;
  int block_local_id =  blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  unsigned long long tid = blockDim.z * blockDim.x * blockDim.y * blockIdx.x + block_local_id;
  //if (tid==0)printf("Running jac_binning_gpu_u_per_block_bst_sm_kernel\n");
  //int block_local_id = blockDim.x * threadIdx.y + threadIdx.x;
  //int tid = blockDim.x * blockDim.y * blockIdx.x + block_local_id;
  //printf("block_local_id %d tid %d threadidx x %d y %d z %d blockidx x %d y %d z %d\n", block_local_id, tid, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);


  unsigned mask = calculateMask(blockDim.x, tid);//threadIdx.y*blockDim.x+threadIdx.x);
  extern __shared__ VID glob_u_adj[];

  for(EN ptr = blockIdx.x; ptr < bin_size; ptr += gridDim.x) {
    VID u = bin[ptr];
    EN degu = xadj[u+1] - xadj[u];
    for(EN ptr = block_local_id; ptr < degu; ptr += blockDim.x * blockDim.y * blockDim.z) {
      glob_u_adj[ptr] = adj[xadj[u] + ptr]; 
    }
    __syncthreads();

    for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z; neigh_ptr< degu; neigh_ptr+=blockDim.y * blockDim.z){
    //for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z + xadj_start[u]; neigh_ptr< degu; neigh_ptr+=blockDim.y * blockDim.z){
      VID v = glob_u_adj[neigh_ptr];
      EN v_start = xadj[v];
      if (xadj[v+1]-xadj[v] > degu || (xadj[v+1]-xadj[v] == degu && v < u) ){
        EN degv = xadj[v+1]-xadj[v];
        EN intersection_size = 0;
        EN other_ptr = xadj[v]+bst_spec(&adj[v_start], degv-1, u);

        for (EN t_ptr = threadIdx.x; t_ptr < degv; t_ptr+=blockDim.x){
          EN loc = bst_spec(glob_u_adj, degu-1, adj[v_start+t_ptr]);
          intersection_size+=(loc!=(EN)-1);
        } 

        //intersection_size = warpReduce(intersection_size, blockDim.x, tid%32, threadIdx.x==0);
        intersection_size = warpReduce(intersection_size, blockDim.x, mask);
        if (threadIdx.x == 0){
          emetrics[(xadj[u] + neigh_ptr)] = emetrics[other_ptr] = float(intersection_size)/float(degu+degv-intersection_size);

        }
      }
    }
    __syncthreads();
  }
}

template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_block_bst_sm_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  start = omp_get_wtime();

/*
  dim3 block;
  block.x = 8;
  block.y = 4;
  block.z = 4;
  int no_blocks = 2048;
*/
  unsigned long long size_of_shared_memory = SM_FAC*sizeof(VID);
  jac_binning_gpu_u_per_block_bst_sm_kernel<directed, EN, VID, E><<<grid, block, size_of_shared_memory>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, SM_FAC, lower_limit, upper_limit);
  gpuErrchk( cudaGetLastError());
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  return end-start;

}
template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_gpu_u_per_block_bst_inv_sm_bigsgroup_atomic_calc_kernel(const EN* __restrict__ xadj, const VID* __restrict__ adj, const VID* __restrict__ is, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size) {
  //int no_threads = blockDim.z * blockDim.y * blockDim.x * gridDim.x;
  int block_local_id =  blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  unsigned long long tid = blockDim.z * blockDim.x * blockDim.y * blockIdx.x + block_local_id;
  for (size_t ptr = tid; ptr < bin_size; ptr+=blockDim.x*blockDim.y*blockDim.z*gridDim.x*gridDim.y*gridDim.z){
    size_t uni = xadj[adj[ptr]+1]-xadj[adj[ptr]]+xadj[is[ptr]+1]-xadj[is[ptr]];
    emetrics[ptr] = emetrics[ptr]/(uni-emetrics[ptr]);
  }
}
// Each block will use a single u
// All threads on the same y blockIdx will handle the same u
// each of them will do its own bst
// the bst is done on the global memory
// Things to try:
//  - different block sizes (number of warps)
//  - different number of threads doing searching
template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_gpu_u_per_block_bst_inv_sm_bigsgroup_atomic_kernel(const EN* __restrict__ xadj, const VID* __restrict__ adj, const VID* __restrict__ xadj_start, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {
  
  //int no_threads = blockDim.z * blockDim.y * blockDim.x * gridDim.x;
  int block_local_id =  blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  unsigned long long tid = blockDim.z * blockDim.x * blockDim.y * blockIdx.x + block_local_id;
  //if (tid==0)printf("Running jac_binning_gpu_u_per_block_bst_inv_sm_kernel\n");
  //int block_local_id = blockDim.x * threadIdx.y + threadIdx.x;
  //int tid = blockDim.x * blockDim.y * blockIdx.x + block_local_id;
  //printf("block_local_id %d tid %d threadidx x %d y %d z %d blockidx x %d y %d z %d\n", block_local_id, tid, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

  unsigned mask = calculateMask(blockDim.x, tid);//threadIdx.y*blockDim.x+threadIdx.x);

  extern __shared__ VID glob_u_adj[];

  for(EN ptr = blockIdx.x; ptr < bin_size; ptr += gridDim.x) {
    EN u = bin[ptr];
    EN degu = xadj[u+1] - xadj[u];
    __syncthreads();
    for(EN ptr = block_local_id; ptr < degu; ptr += blockDim.x * blockDim.y * blockDim.z) {
      glob_u_adj[ptr] = adj[xadj[u] + ptr]; 
    }
    __syncthreads();

    for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z; neigh_ptr< degu; neigh_ptr+=blockDim.y * blockDim.z){
    //for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z + xadj_start[u]; neigh_ptr< degu; neigh_ptr+=blockDim.y * blockDim.z){
      VID v = glob_u_adj[neigh_ptr];
      if (xadj[v+1]-xadj[v] > degu || (xadj[v+1]-xadj[v] == degu && v < u) ){
        EN intersection_size = 0;
        EN other_ptr = bst(xadj, adj, v, u);

        for (EN t_ptr = threadIdx.x; t_ptr < degu; t_ptr+=blockDim.x){
          EN loc = bst(xadj, adj, v, glob_u_adj[t_ptr]);
          intersection_size+=(loc!=(EN)-1);
        } 

        intersection_size = warpReduce(intersection_size, min(blockDim.x, WARP_SIZE), mask);
        //intersection_size = warpReduce(intersection_size, blockDim.x, tid%32, threadIdx.x==0);
        if (threadIdx.x % min(blockDim.x, WARP_SIZE) == 0){
          atomicAdd(&emetrics[(xadj[u] + neigh_ptr)], intersection_size); atomicAdd(&emetrics[other_ptr], intersection_size);
        }
      }
    }
    __syncthreads();
  }
}

// Driver function that will call the u_per_block_bst_sm kernel
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_block_bst_inv_sm_bigsgroup_atomic_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  start = omp_get_wtime();

/*
  dim3 block;
  block.x = 8;
  block.y = 4;
  block.z = 4;
  int no_blocks = 2048;
*/
  unsigned long long size_of_shared_memory = SM_FAC*sizeof(VID);
  jac_binning_gpu_u_per_block_bst_inv_sm_bigsgroup_atomic_kernel<directed, EN, VID, E><<<grid, block, size_of_shared_memory>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, SM_FAC, lower_limit, upper_limit);
  jac_binning_gpu_u_per_block_bst_inv_sm_bigsgroup_atomic_calc_kernel<directed, EN, VID, E><<<grid,block>>>(xadj, adj, is, n, jac, bin, bin_size);
  gpuErrchk( cudaGetLastError());
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  return end-start;

}
// Each group of gridDim.x blocks will use a single u
// All threads on the same y,z blockIdx will handle the same u
// Each group of threads with the same y,z thread ID and x blockIdx will handle same u,v
// each of them will do its own bst
// the bst is done on the global memory
// Things to try:
//  - different block sizes (number of warps)
//  - different number of threads doing searching
template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_gpu_u_per_grid_bst_inv_sm_bigsgroup_kernel(const EN* __restrict__ xadj, const VID* __restrict__ adj, const VID* __restrict__ xadj_start, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {
  
  //int no_threads = blockDim.z * blockDim.y * blockDim.x * gridDim.x;
  unsigned long long block_local_id =  blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  unsigned long long grid_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  unsigned long long tid = grid_id * blockDim.x * blockDim.y * blockDim.z + block_local_id;
  unsigned long long block_local_sg_id = threadIdx.y+threadIdx.z*blockDim.y;

  unsigned mask = calculateMask(blockDim.x, tid);//threadIdx.y*blockDim.x+threadIdx.x);

  extern __shared__ VID glob_inter_fnow[];
  VID * glob_u_adj= glob_inter_fnow+(blockDim.x/WARP_SIZE)*blockDim.y*blockDim.z;
  VID* glob_inter = glob_inter_fnow+(blockDim.x/WARP_SIZE)*block_local_sg_id;
  for(EN ptr = blockIdx.y+gridDim.y*blockIdx.z; ptr < bin_size; ptr += gridDim.y*gridDim.z) {
    EN u = bin[ptr];
    EN degu = xadj[u+1] - xadj[u];
    __syncthreads();
    for(EN sm_ptr = block_local_id; sm_ptr < degu; sm_ptr += blockDim.x * blockDim.y * blockDim.z) {
      glob_u_adj[sm_ptr] = adj[xadj[u] + sm_ptr]; 
    }
    __syncthreads();

    for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z + blockDim.y * blockDim.z * blockIdx.x; neigh_ptr< degu; neigh_ptr+=blockDim.y * blockDim.z * gridDim.x){
    //for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z + xadj_start[u]; neigh_ptr< degu; neigh_ptr+=blockDim.y * blockDim.z){
      VID v = glob_u_adj[neigh_ptr];
      bool skippable = (xadj[v+1]-xadj[v] < degu ||(xadj[v+1]-xadj[v] == degu && v > u));
      if (!directed &&skippable)
        continue;
      EN other_ptr = bst(xadj, adj, v, u);
      if (directed && other_ptr != (EN)-1 && skippable)
        continue;
      EN intersection_size = 0;

      for (EN t_ptr = threadIdx.x; t_ptr < degu; t_ptr+=blockDim.x){
        EN loc = bst(xadj, adj, v, glob_u_adj[t_ptr]);
        intersection_size+=(loc!=(EN)-1);
      } 

      intersection_size = warpReduce(intersection_size, min(blockDim.x, WARP_SIZE), mask);
      if (threadIdx.x % WARP_SIZE == 0 && blockDim.x > 32)
        glob_inter[block_local_id/WARP_SIZE] = intersection_size;
      //intersection_size = warpReduce(intersection_size, blockDim.x, tid%32, threadIdx.x==0);
      __syncthreads();
      if (threadIdx.x == 0){
        for (int i =1; i< blockDim.x/WARP_SIZE; i++) intersection_size+= glob_inter[i]; 
        E J = float(intersection_size)/float(degu+(xadj[v+1]-xadj[v])-intersection_size);
        emetrics[(xadj[u] + neigh_ptr)] = J;
        if (other_ptr!=(EN)-1)  emetrics[other_ptr] = J;
      }
      __syncthreads();
    }
  }
}

// Driver function that will call the u_per_block_bst_sm kernel
template <bool directed, typename EN, typename VID, typename E>
string k4_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, size_t range, size_t g, size_t a, cudaStream_t * stream){
  dim3 block(1,1,1);
  block.x = g;

  dim3 grid(a, 1,1);
  grid.y = max(min(bin_size, MAX_GRID_DIM),1);
  grid.z = min(max(bin_size/grid.y,1), MAX_GRID_DIM);
  unsigned long long size_of_shared_memory = get_shared_memory_requirement_for_k4<EN, VID>(range, g);
  string name = get_kernel_name("k4", range, g, a, grid, block, size_of_shared_memory);
  if (stream == nullptr){
    jac_binning_gpu_u_per_grid_bst_inv_sm_bigsgroup_kernel<directed, EN, VID, E><<<grid, block, size_of_shared_memory>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  } else {
    jac_binning_gpu_u_per_grid_bst_inv_sm_bigsgroup_kernel<directed, EN, VID, E><<<grid, block, size_of_shared_memory, *stream>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  }

  gpuErrchk( cudaGetLastError());
  return name;
}

// Driver function that will call the u_per_block_bst_sm kernel
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_grid_bst_inv_sm_bigsgroup_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  start = omp_get_wtime();

/*
  dim3 block;
  block.x = 8;
  block.y = 4;
  block.z = 4;
  int no_blocks = 2048;
*/
  grid.y = max(min(bin_size, MAX_GRID_DIM),1);
  grid.z = min(max(bin_size/grid.y,1), MAX_GRID_DIM);
  unsigned long long size_of_shared_memory = SM_FAC*sizeof(VID) + block.x/WARP_SIZE * sizeof(EN) ;
  jac_binning_gpu_u_per_grid_bst_inv_sm_bigsgroup_kernel<directed, EN, VID, E><<<grid, block, size_of_shared_memory>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, SM_FAC, lower_limit, upper_limit);
  gpuErrchk( cudaGetLastError());
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  return end-start;

}
// Each group of gridDim.x blocks will use a single u
// All threads on the same y,z blockIdx will handle the same u
// Each group of threads with the same y,z thread ID and x blockIdx will handle same u,v
// each of them will do its own bst
// the bst is done on the global memory
// Things to try:
//  - different block sizes (number of warps)
//  - different number of threads doing searching
template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_gpu_u_per_grid_bst_inv_sm_kernel(const EN* __restrict__ xadj, const VID* __restrict__ adj, const VID* __restrict__ xadj_start, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {
  
  //int no_threads = blockDim.z * blockDim.y * blockDim.x * gridDim.x;
  unsigned long long block_local_id =  blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  unsigned long long grid_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  unsigned long long tid = grid_id * blockDim.x * blockDim.y * blockDim.z + block_local_id;
  unsigned mask = calculateMask(blockDim.x, tid);//threadIdx.y*blockDim.x+threadIdx.x);

  extern __shared__ VID glob_u_adj[];

  for(EN ptr = blockIdx.y+gridDim.y*blockIdx.z; ptr < bin_size; ptr += gridDim.y*gridDim.z) {
    EN u = bin[ptr];
    EN degu = xadj[u+1] - xadj[u];
    __syncthreads();
    for(EN sm_ptr = block_local_id; sm_ptr < degu; sm_ptr += blockDim.x * blockDim.y * blockDim.z) {
      glob_u_adj[sm_ptr] = adj[xadj[u] + sm_ptr]; 
    }
    __syncthreads();

    for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z + blockDim.y * blockDim.z * blockIdx.x; neigh_ptr< degu; neigh_ptr+=blockDim.y * blockDim.z * gridDim.x){
    //for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z + xadj_start[u]; neigh_ptr< degu; neigh_ptr+=blockDim.y * blockDim.z){
      VID v = glob_u_adj[neigh_ptr];
      bool skippable = (xadj[v+1]-xadj[v] < degu ||(xadj[v+1]-xadj[v] == degu && v > u));
      if (!directed &&skippable)
        continue;
      EN other_ptr = bst(xadj, adj, v, u);
      if (directed && other_ptr != (EN)-1 && skippable)
        continue;
      EN intersection_size = 0;

      for (EN t_ptr = threadIdx.x; t_ptr < degu; t_ptr+=blockDim.x){
        EN loc = bst(xadj, adj, v, glob_u_adj[t_ptr]);
        intersection_size+=(loc!=(EN)-1);
      } 

      intersection_size = warpReduce(intersection_size, blockDim.x, mask);
      //intersection_size = warpReduce(intersection_size, blockDim.x, tid%32, threadIdx.x==0);
      if (threadIdx.x == 0){
        E J = float(intersection_size)/float(degu+(xadj[v+1]-xadj[v])-intersection_size);
        emetrics[(xadj[u] + neigh_ptr)] = J;
        if (other_ptr != (EN)-1)
          emetrics[other_ptr] =  J;

      }
    }
    __syncthreads();
  }
}

//jac_binning_gpu_u_per_grid_bst_inv_sm_kernel
template <bool directed, typename EN, typename VID, typename E>
string k3_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, size_t range, size_t g, size_t a, cudaStream_t * stream){
   

  dim3 block(1,1,1);
  block.x = g;
  //block.y = max((size_t)1,(size_t)THREADS_PER_BLOCK/block.x);
  if (g < WARP_SIZE){
    block.y = WARP_SIZE/g;
  }

  dim3 grid(1,1,1);
  //grid.x = max((size_t)1,(size_t)a/block.y);
  grid.x = a;
  grid.y = max(min(bin_size, MAX_GRID_DIM),1);
  grid.z = min(max(bin_size/grid.y,1), MAX_GRID_DIM);
  unsigned long long size_of_shared_memory = get_shared_memory_requirement_for_k3<EN, VID>(range);
  //double start = omp_get_wtime();
  if (stream == nullptr){
    jac_binning_gpu_u_per_grid_bst_inv_sm_kernel<directed, EN, VID, E><<<grid, block, size_of_shared_memory>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  } else {
    jac_binning_gpu_u_per_grid_bst_inv_sm_kernel<directed, EN, VID, E><<<grid, block, size_of_shared_memory, *stream>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, (VID)0, (VID)0, (VID)0);
  }
  gpuErrchk( cudaGetLastError());
  //double end = omp_get_wtime();
  string name = get_kernel_name("k3", range, g, a, grid, block, size_of_shared_memory);
  //return make_tuple(end-start, name);
  return name;
}

template <bool directed, typename EN, typename VID, typename E>
std::tuple<JAC_FUNC_GA<directed, EN, VID, E>, size_t, size_t, size_t> all_k3(int bin_id, size_t range, vector<EN> ranges, vector<pair<unsigned long long, unsigned long long>> bin_sizes, size_t max_threads, size_t max_shared_memory){
  return make_tuple(k3_driver<directed, EN, VID, E>, 32, 8, get_shared_memory_requirement_for_k3<EN, VID>(ranges[bin_id]));
}

// Driver function that will call the u_per_block_bst_sm kernel
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_grid_bst_inv_sm_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  start = omp_get_wtime();

/*
  dim3 block;
  block.x = 8;
  block.y = 4;
  block.z = 4;
  int no_blocks = 2048;
*/
  grid.y = max(min(bin_size, MAX_GRID_DIM),1);
  grid.z = min(max(bin_size/grid.y,1), MAX_GRID_DIM);
  unsigned long long size_of_shared_memory = SM_FAC*sizeof(VID);
  jac_binning_gpu_u_per_grid_bst_inv_sm_kernel<directed, EN, VID, E><<<grid, block, size_of_shared_memory>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, SM_FAC, lower_limit, upper_limit);
  gpuErrchk( cudaGetLastError());
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  return end-start;

}
// Each block will use a single u
// All threads on the same y blockIdx will handle the same u
// each of them will do its own bst
// the bst is done on the global memory
// Things to try:
//  - different block sizes (number of warps)
//  - different number of threads doing searching
template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_gpu_u_per_block_bst_inv_sm_kernel(const EN* __restrict__ xadj, const VID* __restrict__ adj, const VID* __restrict__ xadj_start, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {
  
  //int no_threads = blockDim.z * blockDim.y * blockDim.x * gridDim.x;
  int block_local_id =  blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  unsigned long long tid = blockDim.z * blockDim.x * blockDim.y * blockIdx.x + block_local_id;
  //if (tid==0)printf("Running jac_binning_gpu_u_per_block_bst_inv_sm_kernel\n");
  //int block_local_id = blockDim.x * threadIdx.y + threadIdx.x;
  //int tid = blockDim.x * blockDim.y * blockIdx.x + block_local_id;
  //printf("block_local_id %d tid %d threadidx x %d y %d z %d blockidx x %d y %d z %d\n", block_local_id, tid, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

  unsigned mask = calculateMask(blockDim.x, tid);//threadIdx.y*blockDim.x+threadIdx.x);

  extern __shared__ VID glob_u_adj[];

  for(EN ptr = blockIdx.x; ptr < bin_size; ptr += gridDim.x) {
    EN u = bin[ptr];
    EN degu = xadj[u+1] - xadj[u];
    __syncthreads();
    for(EN ptr = block_local_id; ptr < degu; ptr += blockDim.x * blockDim.y * blockDim.z) {
      glob_u_adj[ptr] = adj[xadj[u] + ptr]; 
    }
    __syncthreads();

    for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z; neigh_ptr< degu; neigh_ptr+=blockDim.y * blockDim.z){
    //for (EN neigh_ptr = threadIdx.y + blockDim.y * threadIdx.z + xadj_start[u]; neigh_ptr< degu; neigh_ptr+=blockDim.y * blockDim.z){
      VID v = glob_u_adj[neigh_ptr];
      if (xadj[v+1]-xadj[v] > degu || (xadj[v+1]-xadj[v] == degu && v < u) ){
        EN intersection_size = 0;
        EN other_ptr = bst(xadj, adj, v, u);

        for (EN t_ptr = threadIdx.x; t_ptr < degu; t_ptr+=blockDim.x){
          EN loc = bst(xadj, adj, v, glob_u_adj[t_ptr]);
          intersection_size+=(loc!=(EN)-1);
        } 

        intersection_size = warpReduce(intersection_size, blockDim.x, mask);
        //intersection_size = warpReduce(intersection_size, blockDim.x, tid%32, threadIdx.x==0);
        if (threadIdx.x == 0){
          emetrics[(xadj[u] + neigh_ptr)] = emetrics[other_ptr] = float(intersection_size)/float(degu+(xadj[v+1]-xadj[v])-intersection_size);

        }
      }
    }
    __syncthreads();
  }
}

// Driver function that will call the u_per_block_bst_sm kernel
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_block_bst_inv_sm_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  start = omp_get_wtime();

/*
  dim3 block;
  block.x = 8;
  block.y = 4;
  block.z = 4;
  int no_blocks = 2048;
*/
  unsigned long long size_of_shared_memory = SM_FAC*sizeof(VID);
  jac_binning_gpu_u_per_block_bst_inv_sm_kernel<directed, EN, VID, E><<<grid, block, size_of_shared_memory>>>(xadj, adj, xadj_start, n, jac, bin, bin_size, SM_FAC, lower_limit, upper_limit);
  gpuErrchk( cudaGetLastError());
  cudaDeviceSynchronize();
  end = omp_get_wtime();
  return end-start;

}
template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_binning_gpu_u_per_thread_kernel(const VID* __restrict__ is, const EN* __restrict__ xadj, const VID* __restrict__ adj, VID n, E* __restrict__ emetrics, VID* bin, VID bin_size, VID SM_FAC, EN lower_limit, EN upper_limit) {

  int no_threads = blockDim.x * gridDim.x;
  unsigned long long tid = blockDim.x * blockIdx.x + threadIdx.x;
  //if (tid == 0) printf("Running jac_binning_gpu_u_per_thread_kernel\n"); 

  for(EN ptr = tid; ptr < bin_size; ptr += no_threads) {
    EN edge = bin[ptr];
    VID u = is[edge];
    VID v = adj[edge];

      EN other_ptr, intersection_size = 0;

      EN ptr_u = xadj[u];
      EN ptr_v = xadj[v];

      while(ptr_u < xadj[u+1] && ptr_v < xadj[v+1]) {
        VID u_ngh = adj[ptr_u];

        VID v_ngh = adj[ptr_v];
        if(v_ngh == u) {
          other_ptr = ptr_v;
        }

        if(u_ngh == v_ngh) {
          intersection_size++;
          ptr_u++; ptr_v++;
        } else if(u_ngh < v_ngh) {
          ptr_u++;
        } else {
          ptr_v++;
        }
      }     

#ifdef AOS
      EN eu =  xadj[u+1] -xadj[u];
      EN ev =  xadj[v+1] -xadj[v];
      emetrics[edge] = emetrics[other_ptr ] = float(intersection_size)/float(eu+ev-intersection_size);
#elif defined SOA
      EN eu =  xadj[u+1] -xadj[u];
      EN ev =  xadj[v+1] -xadj[v];
      emetrics[edge] = emetrics[other_ptr] = float(intersection_size)/float(eu+ev-intersection_size);
#endif
  }
}

template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_thread_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit){
  double start, end;

  start = omp_get_wtime();

/*
  int no_threads = 256;
  int no_blocks = 1024;
*/
  int no_threads = block.x;
  jac_binning_gpu_u_per_thread_kernel<directed, EN, VID, E><<<grid, no_threads>>>(is, xadj, adj, n, jac, bin, bin_size, SM_FAC, lower_limit, upper_limit);
  gpuErrchk( cudaGetLastError());
  cudaDeviceSynchronize();

  end = omp_get_wtime();
  return end-start;

}

template <bool directed, typename EN, typename VID, typename E>
std::tuple<JAC_FUNC_GA<directed, EN, VID, E>, size_t, size_t, size_t> get_jac_func(size_t g, size_t a, bool sm){
  if (sm){
    if (g < 32){
      return make_tuple(k3_driver<directed, EN, VID, E>, g, a, 0); 
    } else {
      return make_tuple(k4_driver<directed, EN, VID, E>, g, a, 0); 
    }
  } else {
    if (g < 32){
      return make_tuple(k1_driver<directed, EN, VID, E>, g, a, 0); 
    } else {
      return make_tuple(k2_driver<directed, EN, VID, E>, g, a, 0); 
    }
  }
}

template <bool directed, typename EN, typename VID, typename E>
std::tuple<JAC_FUNC_GA<directed, EN, VID, E>, size_t, size_t, size_t> strategy_7(int bin_id, size_t range, vector<EN> ranges, vector<pair<unsigned long long, unsigned long long>> bin_sizes, size_t max_threads, size_t max_shared_memory){
  size_t g;
  size_t bin_size = bin_sizes[bin_id].first;
  size_t prev_range = (bin_id > 0) ? ranges[bin_id-1] : 0;
  // g
  if (prev_range <= 96) g = 32;
  else if (prev_range < 4096) g = 64;
  else g = 128;
// determine SM
  if (g <= 32){
    return make_tuple(k8_driver<directed, EN, VID, E>, g, 1, 0); 
  } else {
    return make_tuple(k7_driver<directed, EN, VID, E>, g, 1, 0); 
  }
}

template <bool directed, typename EN, typename VID, typename E>
std::tuple<JAC_FUNC_GA<directed, EN, VID, E>, size_t, size_t, size_t> strategy_6(int bin_id, size_t range, vector<EN> ranges, vector<pair<unsigned long long, unsigned long long>> bin_sizes, size_t max_threads, size_t max_shared_memory){
  size_t g;
  size_t bin_size = bin_sizes[bin_id].first;
  size_t prev_range = (bin_id > 0) ? ranges[bin_id-1] : 0;
  // g
  if (prev_range <= 96) g = 32;
  else if (prev_range < 1024) g = 32;
  else if (prev_range < 4096) g = 64;
  else g = 128;
// determine SM
  if (g <= 32){
    return make_tuple(k6_driver<directed, EN, VID, E>, g, 1, 0); 
  } else {
    return make_tuple(k5_driver<directed, EN, VID, E>, g, 1, 0); 
  }
}
template <bool directed, typename EN, typename VID, typename E>
std::tuple<JAC_FUNC_GA<directed, EN, VID, E>, size_t, size_t, size_t> strategy_5(int bin_id, size_t range, vector<EN> ranges, vector<pair<unsigned long long, unsigned long long>> bin_sizes, size_t max_threads, size_t max_shared_memory){
  bool sm = false;
  size_t g, a;
  size_t bin_size = bin_sizes[bin_id].first;
  size_t prev_range = (bin_id > 0) ? ranges[bin_id-1] : 0;
  // g
  if (prev_range <= 96) g = 8;
  else if (prev_range < 1024) g = 32;
  else if (prev_range < 4096) g = 64;
  else g = 128;
// determine SM
  if (range < 1024) {
    if (g < 32){
        if (get_shared_memory_requirement_for_k3<EN, VID>(range) <= max_shared_memory)
          sm = true;
        else
          sm = false;
    } else {
        if (get_shared_memory_requirement_for_k4<EN, VID>(range, g) <= max_shared_memory)
          sm = true;
        else
          sm = false;
    }
  }
  // a
  if (bin_size == 0) a =  8;
  else {
    int prev_range_multiples_of_2 = (prev_range != 0) ? pow(2, floor(log2(prev_range))) : 8;
    a = min(1024, (unsigned int)max((unsigned int)8, (unsigned int)min((unsigned int)prev_range_multiples_of_2, (unsigned int)pow(2, floor(log2(max_threads/(g*bin_size)))))));
  }
  sm = false;
  return get_jac_func<directed, EN, VID, E>(g, a, sm);
}
// strategy 1 but never use SM
template <bool directed, typename EN, typename VID, typename E>
std::tuple<JAC_FUNC_GA<directed, EN, VID, E>, size_t, size_t, size_t> strategy_10(int bin_id, size_t range, vector<EN> ranges, vector<pair<unsigned long long, unsigned long long>> bin_sizes, size_t max_threads, size_t max_shared_memory){
  bool sm = false;
  size_t g, a;
  size_t bin_size = bin_sizes[bin_id].first;
  size_t prev_range = (bin_id > 0) ? ranges[bin_id-1] : 0;
  // g
  if (prev_range < 96) g = 32;
  else if (prev_range < 4096) g = 64;
  else g = 128;
// determine SM
  if (bin_size == 0) a =  8;
  else {
    int prev_range_multiples_of_2 = (prev_range != 0) ? pow(2, floor(log2(prev_range))) : 8;
    a = min((unsigned int)max((unsigned int)8, (unsigned int)min((unsigned int)prev_range_multiples_of_2, (unsigned int)pow(2, floor(log2(max_threads/(g*bin_size)))))), 1024);
  }
  return get_jac_func<directed, EN, VID, E>(g,a,sm);
}

// Use SM as much as possible
template <bool directed, typename EN, typename VID, typename E>
std::tuple<JAC_FUNC_GA<directed, EN, VID, E>, size_t, size_t, size_t> strategy_11(int bin_id, size_t range, vector<EN> ranges, vector<pair<unsigned long long, unsigned long long>> bin_sizes, size_t max_threads, size_t max_shared_memory){
  bool sm = false;
  size_t g, a;
  size_t bin_size = bin_sizes[bin_id].first;
  size_t prev_range = (bin_id > 0) ? ranges[bin_id-1] : 0;
  // g
  if (prev_range < 96) g = 32;
  else if (prev_range < 4096) g = 64;
  else g = 128;
// determine SM
  if (g < 32){
      if (get_shared_memory_requirement_for_k3<EN, VID>(range) <= max_shared_memory)
        sm = true;
      else
        sm = false;
  } else {
      if (get_shared_memory_requirement_for_k4<EN, VID>(range, g) <= max_shared_memory)
        sm = true;
      else
        sm = false;
  }
  // a
  if (bin_size == 0) a =  8;
  else {
    int prev_range_multiples_of_2 = (prev_range != 0) ? pow(2, floor(log2(prev_range))) : 8;
    a = min((unsigned int)max((unsigned int)8, (unsigned int)min((unsigned int)prev_range_multiples_of_2, (unsigned int)pow(2, floor(log2(max_threads/(g*bin_size)))))), 1024);
  }
  return get_jac_func<directed, EN, VID, E>(g,a,sm);
}

template <bool directed, typename EN, typename VID, typename E>
std::tuple<JAC_FUNC_GA<directed, EN, VID, E>, size_t, size_t, size_t> strategy_1(int bin_id, size_t range, vector<EN> ranges, vector<pair<unsigned long long, unsigned long long>> bin_sizes, size_t max_threads, size_t max_shared_memory){
  bool sm = false;
  size_t g, a;
  size_t bin_size = bin_sizes[bin_id].first;
  size_t prev_range = (bin_id > 0) ? ranges[bin_id-1] : 0;
  // g
  if (prev_range < 96) g = 32;
  else if (prev_range < 4096) g = 64;
  else g = 128;
// determine SM
  if (range < 1024) {
    if (g < 32){
        if (get_shared_memory_requirement_for_k3<EN, VID>(range) <= max_shared_memory)
          sm = true;
        else
          sm = false;
    } else {
        if (get_shared_memory_requirement_for_k4<EN, VID>(range, g) <= max_shared_memory)
          sm = true;
        else
          sm = false;
    }
  }
  // a
  if (bin_size == 0) a =  8;
  else {
    int prev_range_multiples_of_2 = (prev_range != 0) ? pow(2, floor(log2(prev_range))) : 8;
    a = min((unsigned int)max((unsigned int)8, (unsigned int)min((unsigned int)prev_range_multiples_of_2, (unsigned int)pow(2, floor(log2(max_threads/(g*bin_size)))))), 1024);
  }
  return get_jac_func<directed, EN, VID, E>(g,a,sm);
}


template <bool directed, typename EN, typename VID, typename E>
std::tuple<JAC_FUNC_GA<directed, EN, VID, E>, size_t, size_t, size_t> strategy_2(int bin_id, size_t range, vector<EN> ranges, vector<pair<unsigned long long, unsigned long long>> bin_sizes, size_t max_threads, size_t max_shared_memory){
  bool sm = false;
  size_t g, a;
  size_t bin_size = bin_sizes[bin_id].first;
  size_t prev_range = (bin_id > 0) ? ranges[bin_id-1] : 0;
  // g
  if (prev_range < 96) g = 32;
  else if (prev_range < 4096) g = 64;
  else g = 128;
// determine SM
  if (range < 1024) {
    if (g < 32){
        if (get_shared_memory_requirement_for_k3<EN, VID>(range) <= max_shared_memory)
          sm = true;
        else
          sm = false;
    } else {
        if (get_shared_memory_requirement_for_k4<EN, VID>(range, g) <= max_shared_memory)
          sm = true;
        else
          sm = false;
    }
  }
  // a
  if (bin_size == 0) a =  8;
  else {
    int prev_range_multiples_of_2 = (prev_range != 0) ? pow(2, floor(log2(prev_range))) : 8;
    a = min((unsigned int)max((unsigned int)8, (unsigned int)min((unsigned int)prev_range_multiples_of_2, (unsigned int)pow(2, floor(log2(max_threads/(g*bin_size)))))), 2048);
  }
  return get_jac_func<directed, EN, VID, E>(g,a,sm);
}
template <bool directed, typename EN, typename VID, typename E>
std::tuple<JAC_FUNC_GA<directed, EN, VID, E>, size_t, size_t, size_t> strategy_3(int bin_id, size_t range, vector<EN> ranges, vector<pair<unsigned long long, unsigned long long>> bin_sizes, size_t max_threads, size_t max_shared_memory){
  bool sm = false;
  size_t g, a;
  size_t bin_size = bin_sizes[bin_id].first;
  size_t prev_range = (bin_id > 0) ? ranges[bin_id-1] : 0;
  // g
  if (prev_range < 96) g = 32;
  else if (prev_range < 4096) g = 64;
  else g = 128;
// determine SM
  if (range < 1024) {
    if (g < 32){
        if (get_shared_memory_requirement_for_k3<EN, VID>(range) <= max_shared_memory)
          sm = true;
        else
          sm = false;
    } else {
        if (get_shared_memory_requirement_for_k4<EN, VID>(range, g) <= max_shared_memory)
          sm = true;
        else
          sm = false;
    }
  }
  // a
  if (bin_size == 0) a =  8;
  else {
    int prev_range_multiples_of_2 = (prev_range != 0) ? pow(2, floor(log2(prev_range))) : 8;
    a = (unsigned int)max((unsigned int)8, (unsigned int)min((unsigned int)prev_range_multiples_of_2, (unsigned int)pow(2, floor(log2(max_threads/(g*bin_size))))));
  }
  return get_jac_func<directed, EN, VID, E>(g,a,sm);
}
template <bool directed, typename EN, typename VID, typename E>
std::tuple<JAC_FUNC_GA<directed, EN, VID, E>, size_t, size_t, size_t> strategy_4(int bin_id, size_t range, vector<EN> ranges, vector<pair<unsigned long long, unsigned long long>> bin_sizes, size_t max_threads, size_t max_shared_memory){
  bool sm = false;
  max_threads=max_threads/100;
  size_t g, a;
  size_t bin_size = bin_sizes[bin_id].first;
  size_t prev_range = (bin_id > 0) ? ranges[bin_id-1] : 0;
  // g
  if (prev_range < 96) g = 32;
  else if (prev_range < 4096) g = 64;
  else g = 128;
// determine SM
  if (range < 1024) {
    if (g < 32){
        if (get_shared_memory_requirement_for_k3<EN, VID>(range) <= max_shared_memory)
          sm = true;
        else
          sm = false;
    } else {
        if (get_shared_memory_requirement_for_k4<EN, VID>(range, g) <= max_shared_memory)
          sm = true;
        else
          sm = false;
    }
  }
  // a
  if (bin_size == 0) a =  8;
  else {
    int prev_range_multiples_of_2 = (prev_range != 0) ? pow(2, floor(log2(prev_range))) : 8;
    a = (unsigned int)max((unsigned int)8, (unsigned int)min((unsigned int)prev_range_multiples_of_2, (unsigned int)pow(2, floor(log2(max_threads/(g*bin_size))))));
  }
  return get_jac_func<directed, EN, VID, E>(g,a,sm);
}
