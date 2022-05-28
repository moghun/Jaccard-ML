#include "json.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <utility>
#include <tuple>
#include <cmath>
#include <chrono>
#include "config.h"
#include "io.h"


#ifdef _CUGRAPH
//#include <algorithms.hpp>
//#include <graph.hpp>
#endif

#include "edge_metrics.cu"
#include "edge_metrics_binning.h"
#include "utils.cuh"

#define NUM_THREADS  256
#define NUM_BLOCKS  1024

using namespace std;

typedef unsigned int READ_TYPE; // the format in which binary files are written
typedef unsigned int vid_t; // used for adj and xadj and SHOULD REPRESENT |E| WITHOUT OVERFLOWING
typedef float jac_t; // used for jaccards (or floats in general)
typedef unsigned long long ull;

int main(int argc, char** argv) {
#ifndef SORT_ASC
    cout << "Not sorting CSR by degrees\n";
#elif SORT_ASC==1
    cout << "Sorting CSR by degrees in ascending order\n";
#elif SORT_ASC==0
  cout << "Sorting CSR by degrees in descending order\n";
#else
  cout << "Passed an illegal value for sorting of CSR - undefining the sort variable\n";
#undef SORT_ASC
#endif
#if (!DIRECTED)
    cout << "Treating graph as an undirected graph\n";
#elif (DIRECTED)
    cout << "Treating graph as a directed graph\n";
#else
  cout << "Bad directed preprocessor value. Exiting\n";
  return 1;
#endif
  string output_json_file_name, input_graph_file_name;
  #ifdef BINNING
  string binning_experiment_json_file_name;
  #endif
  int num_average;
  if (!parse_arguments(argc, const_cast<const char**>(argv), input_graph_file_name,
  #ifdef BINNING
  binning_experiment_json_file_name,
  #endif 
  output_json_file_name, num_average)){
    return 1;
  }
  cout << "Using the average of " << num_average << " runs" << endl;

  // Prepare output json
  nlohmann::json output_json;
  // If it's already been written to, read it back, otherwise initialize it
  if (!check_file_exists(output_json_file_name)){
      output_json = initialize_output_json(input_graph_file_name);
  } else {
      output_json = read_json(output_json_file_name);
  }
  cout << "Printing results to the file " << output_json_file_name;

  // Reading graph
  cout << endl << endl << "Graph: " << input_graph_file_name << endl;
  graph<vid_t, vid_t> g = open_graph<vid_t, READ_TYPE>(input_graph_file_name, DIRECTED);
  print_graph_statistics<vid_t, READ_TYPE>(g, output_json);
  cout << "##############################" << endl << endl;

  pretty_print_results(cout, "Algorithm", "Time", "Errors");


  // Create Jaccard array -- will contain ground truth
  jac_t* emetrics = new jac_t[g.m];

  // Check if the jaccard value has been cached before
  string jaccards_output_path = string(input_graph_file_name)+ ".corr.bin";
  ifstream infile_corr_bin(jaccards_output_path , ios::in | ios::binary);

  bool have_correct = false;
  if(infile_corr_bin.is_open()) {
      cout << "Reading correct jaccard values from disk\n";
      have_correct = true;
      infile_corr_bin.read((char*)(emetrics), sizeof(jac_t)*g.m);
  }
  double total_time = 0;
  double start, end;
#ifdef _CPU
    jac_t* emetrics_vanilla = new jac_t[g.m];
  for (int i = 0; i< num_average; i++){
    start = omp_get_wtime();
    edge_based_metrics<DIRECTED, vid_t, vid_t, jac_t>(g.is, g.xadj, g.adj, g.n, emetrics_vanilla);
    end = omp_get_wtime();
    total_time+=end-start;
  }
  validate_and_write(g, "CPU", emetrics, emetrics_vanilla, total_time, num_average, output_json_file_name, output_json, jaccards_output_path, have_correct);

  //Compute edge-based metrics
  jac_t* emetrics_bitmap = new jac_t[g.m];
  total_time = 0;
  for (int i = 0; i< num_average; i++){
    start = omp_get_wtime();
    edge_based_metrics_bitmap<DIRECTED, vid_t, vid_t, jac_t>(g.is, g.xadj, g.adj, g.n, emetrics_bitmap);
    end = omp_get_wtime();
    total_time+=end-start;
  }
  validate_and_write(g, "CPU - bitmap", emetrics, emetrics_bitmap, total_time, num_average, output_json_file_name, output_json, jaccards_output_path, have_correct);

#endif
#ifdef _GPU
    // prepare GPU
  int device_id = get_device_id(0);
  cudaSetDevice(device_id);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device_id);
  const int max_sm = props.sharedMemPerBlock;
  cout << device_id << ": " << props.name << ": " << props.major << "." << props.minor << " - Shared memory per block (B): " << max_sm << endl;
  // Lazy starting the GPU
  float *dummy;
  gpuErrchk( cudaMalloc((void**)&dummy, sizeof(float) ) );
  //preparation for cuda
  graph<vid_t, vid_t> g_d;
  double alloc_copy_start = omp_get_wtime();
  cout << "GPU: allocating xadj\n";
  gpuErrchk( vcudaMalloc((void**)&g_d.xadj, sizeof(vid_t) * (g.n + 1) ) );
  gpuErrchk( cudaMemcpy(g_d.xadj, g.xadj, sizeof(vid_t) * (g.n + 1), cudaMemcpyHostToDevice) );
  //cout << "GPU: allocating tadj\n";
  //gpuErrchk( vcudaMalloc((void**)&g_d.tadj, sizeof(vid_t) * (g.n + 1) ) );
  //gpuErrchk( cudaMemcpy(g_d.tadj, tadj, sizeof(vid_t) * (g.n + 1), cudaMemcpyHostToDevice) );
  cout << "GPU: allocating adj\n";
  gpuErrchk( vcudaMalloc((void**)&g_d.adj, sizeof(vid_t) * g.m ) );
  gpuErrchk( cudaMemcpy(g_d.adj, g.adj, sizeof(vid_t) * g.m, cudaMemcpyHostToDevice) );
  jac_t* emetrics_cuda = new jac_t[(ull)g.m], *emetrics_cuda_d;
  cout << "GPU: allocating emetrics\n";
  gpuErrchk( vcudaMalloc((void**)&emetrics_cuda_d, sizeof(jac_t) * (ull)g.m) );
  double alloc_copy_end = omp_get_wtime();
  double t = alloc_copy_end -alloc_copy_start; 
  output_json["experiments"]["GPU - alloc/copy"] =  get_result_json(t, 0);
  write_json_to_file(output_json_file_name, output_json);
  pretty_print_results(cout, "GPU - alloc/copy" , to_string(t), to_string(0));

  alloc_copy_start = omp_get_wtime();
  cout << "GPU: allocating is\n";
  gpuErrchk( vcudaMalloc((void**)&g_d.is, sizeof(vid_t) * g.m ) );
  gpuErrchk( cudaMemcpy(g_d.is, g.is, sizeof(vid_t) * g.m, cudaMemcpyHostToDevice) );
  alloc_copy_end = omp_get_wtime();
  t = alloc_copy_end -alloc_copy_start; 
  pretty_print_results(cout, "GPU - alloc/copy is" , to_string(t), to_string(0));
  output_json["experiments"]["GPU - alloc/copy"] =  get_result_json(t, 0);
  write_json_to_file(output_json_file_name, output_json);
  if (DIRECTED == 0){
    alloc_copy_start = omp_get_wtime();
    cout << "GPU: allocating xadj_start\n";
    gpuErrchk( vcudaMalloc((void**)&g_d.xadj_start, sizeof(vid_t) * (g.n) ) );
    gpuErrchk( cudaMemcpy(g_d.xadj_start, g.xadj_start, sizeof(vid_t) * (g.n), cudaMemcpyHostToDevice) );
    alloc_copy_end = omp_get_wtime();
    t = alloc_copy_end -alloc_copy_start; 
    pretty_print_results(cout, "GPU - alloc/copy xadj_start" , to_string(t), to_string(0));
      output_json["experiments"]["GPU - alloc/copy xadj_start"] =  get_result_json(t, 0);
      write_json_to_file(output_json_file_name, output_json);
  }

#ifdef SIMPLE_GPU_EDGE
  gpuErrchk( cudaMemset(emetrics_cuda_d, 0, sizeof(jac_t) * g.m) );
  total_time = 0;
  int gg = 32;
  int a = 1;
  {
    dim3 grid(1,1,1);
    grid.y = min(MAX_GRID_DIM, g.m/a);
    grid.z = min(MAX_GRID_DIM, max(1, g.m/a/grid.y));
    for (int i = 0; i< num_average; i++){
      start = omp_get_wtime();
      jac_edge_based_small<DIRECTED, vid_t, vid_t, jac_t><<<grid, gg>>>(g_d.xadj, g_d.adj, g_d.is, g.n, emetrics_cuda_d);
      gpuErrchk( cudaDeviceSynchronize() );
      gpuErrchk( cudaMemcpy(emetrics_cuda, emetrics_cuda_d, (ull)sizeof(jac_t) * g.m, cudaMemcpyDeviceToHost) );
      end = omp_get_wtime();
      total_time+=end-start;
    }
  }
  validate_and_write(g, "GPU - SG per edge g="+to_string(gg)+" a="+to_string(a), emetrics, emetrics_cuda, total_time, num_average, output_json_file_name, output_json, jaccards_output_path, have_correct);

#endif
 
#ifdef SIMPLE_GPU
  //Compute edge-based metrics cuda
  gpuErrchk( cudaMemset(emetrics_cuda_d, 0, sizeof(jac_t) * g.m) );
  total_time = 0;
  for (int i = 0; i< num_average; i++){
    start = omp_get_wtime();
    edge_based_metrics_cuda<true, vid_t, vid_t, jac_t><<<NUM_BLOCKS, NUM_THREADS>>>(g_d.is, g_d.xadj, g_d.adj, g.n, emetrics_cuda_d, 1);
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemcpy(emetrics_cuda, emetrics_cuda_d, (ull)sizeof(jac_t) * g.m * 1, cudaMemcpyDeviceToHost) );
    end = omp_get_wtime();
    total_time+=end-start;
  }
// if no CPU runs are happening, set the reference jaccard values (for error checking) to be this kernel's
  validate_and_write(g,  "GPU - Thread per u", emetrics, emetrics_cuda, total_time, num_average, output_json_file_name, output_json, jaccards_output_path, have_correct);
#endif

#ifdef _DONGARRA
  // Create jaccard containers to generate into 
  vector<int> dongarra_num_threads = {512};
  gpuErrchk(cudaFree( g_d.is ));
  // Create the edge list in the structure needed by algorithm
  vid_t * rowidxJ_h = new vid_t[g.m], *colidxJ_h = new vid_t[g.m];
  dongarra::generate_nonzero_arrays(rowidxJ_h, colidxJ_h, g.xadj, g.adj,g.n);
  // Create the containers of the edge list on GPU
  vid_t * rowidxJ_d, *colidxJ_d;
  gpuErrchk( vcudaMalloc((void**)&rowidxJ_d, sizeof(vid_t) * g.m ) );
  gpuErrchk( vcudaMalloc((void**)&colidxJ_d, sizeof(vid_t) * g.m ) );
  gpuErrchk( cudaMemcpy(rowidxJ_d, rowidxJ_h, sizeof(vid_t) * g.m, cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(colidxJ_d, colidxJ_h, sizeof(vid_t) * g.m, cudaMemcpyHostToDevice) );
  // calculate jaccards
  for (auto dongarra_threads : dongarra_num_threads){
    total_time = 0;
    for (int i = 0; i< num_average; i++){
      gpuErrchk( cudaMemset(emetrics_cuda_d, 0, sizeof(jac_t) * g.m * 1) );
      start = omp_get_wtime();
      int my_no_blocks = g.m/dongarra_threads+(g.m%NUM_THREADS!=0);
      dim3 grid(my_no_blocks, 1,1);
      dongarra::dongarra_jaccard<<<grid,dongarra_threads>>>(g.n, g.n, g.m, rowidxJ_d, colidxJ_d, emetrics_cuda_d, g_d.xadj, g_d.adj, (jac_t*)NULL);
      gpuErrchk( cudaMemcpy(emetrics_cuda, emetrics_cuda_d, sizeof(jac_t) * g.m, cudaMemcpyDeviceToHost) );
      end = omp_get_wtime();
      total_time+=end-start;
    }
    validate_and_write(g,  "GPU - Dongarra - "+to_string(dongarra_threads)+" threads", emetrics, emetrics_cuda, total_time, num_average, output_json_file_name, output_json, jaccards_output_path, have_correct);
  }

  gpuErrchk( cudaFree(rowidxJ_d) );
  gpuErrchk( cudaFree(colidxJ_d) );
  gpuErrchk( vcudaMalloc((void**)&g_d.is, sizeof(vid_t) * g.m ) );
  gpuErrchk( cudaMemcpy(g_d.is, g.is, sizeof(vid_t) * g.m, cudaMemcpyHostToDevice) );
#endif
#if defined(_CUGRAPH) || defined(_INHOUSE_CUGRAPH)
  //cugraph::GraphCSRView<vid_t, vid_t, jac_t> cuCSR (xadj_d, adj_d, nullptr, g.n, g.m); 
  total_time = 0;
  for (int i = 0; i< num_average; i++){
    gpuErrchk( cudaMemset(emetrics_cuda_d, 0, sizeof(jac_t) * g.m * 1) );
    start = omp_get_wtime();
    #ifdef _INHOUSE_CUGRAPH
    inhouse_cugraph::cugraph_jaccard<false, vid_t, vid_t, float>(g_d.is, g_d.xadj, g_d.adj, g.n, g.m, emetrics_cuda_d);
    inhouse_cugraph::cugraph_jaccard_nosum<false, vid_t, vid_t, float>(g_d.is, g_d.xadj, g_d.adj, g.n, g.m, emetrics_cuda_d);
    #else
    cugraph::jaccard(cuCSR, (jac_t*)NULL, emetrics_cuda_d);
    #endif
    //for (int j =0;j <3; j++) total_times[j]+=times[j];
    gpuErrchk( cudaMemcpy(emetrics_cuda, emetrics_cuda_d, sizeof(jac_t) * g.m, cudaMemcpyDeviceToHost) );
    end = omp_get_wtime();
    total_time+=end-start;
  }
  validate_and_write(g,  "GPU - cuGraph", emetrics, emetrics_cuda, total_time, num_average, output_json_file_name, output_json, jaccards_output_path, have_correct);
#endif
  
#ifdef BINNING
  //Each binning experiment will
  cout << "##############################" << endl << "###### Binning #####" << endl;
  gpuErrchk( cudaMemset(emetrics_cuda_d, 0, sizeof(jac_t) * g.m * 1) );
  nlohmann::json binning_experiment_json = read_json(binning_experiment_json_file_name);
  vector<tuple<string, vector<tuple<JAC_FUNC<DIRECTED, vid_t, vid_t, jac_t>, dim3, dim3, vid_t, nlohmann::json>>, SEP_FUNC<vid_t, vid_t>>> all_kernels;
  vector<tuple<JAC_FUNC<DIRECTED, vid_t, vid_t, jac_t>, dim3, dim3, vid_t, nlohmann::json>> kernels;
  string name;
  vector<vid_t> ranges = binning_experiment_json["ranges"];//{32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144}; 
  dim3 block(1,1,1), grid(1,1,1);
  output_json["experiments"]["binning"] = nlohmann::json();
  output_json["experiments"]["binning"]["ranges"] = ranges;


  // for each range, add either this kernel or a fallback kernel (in case SM doesn't work etc.
//////////////////////////////////////////////////////////////////////////
    // SMALL
  if (binning_experiment_json.contains("small-sm")){
    vector<int> g_values = binning_experiment_json["small-sm"]["g"];
    vector<int> a_values = binning_experiment_json["small-sm"]["a"];
    for (auto k : g_values){
      for (auto j : a_values){
        for (int i =0; i< ranges.size(); i++){
            dim3 block(1,1,1), grid(1,1,1); 
            block.x = k; block.y = max(1, WARP_SIZE/block.x); block.z = 1;
            grid.x = max(1, j/block.y);
            int g = block.x, a = block.y*grid.x;
            int sm_fac = ranges[i]; 
            if (sm_fac*sizeof(vid_t) <= max_sm){
              nlohmann::json information = generate_json("u-per-grid-bst-inv-sm", log2(g), log2(a), ranges[i], grid, block, sm_fac);
              kernels.push_back(make_tuple(jac_binning_gpu_u_per_grid_bst_inv_sm_driver<DIRECTED, vid_t, vid_t, jac_t>, grid, block, sm_fac, information));
            } else {
              dim3 block(1,1,1), grid(1,1,1); 
              block.x = k; block.y = 1; block.z = 1;
              grid.x = j;
              nlohmann::json information = generate_json("u-per-grid-bst-bigsgroup", log2(k), log2(j), ranges[i], grid, block, 1000);
              kernels.push_back(make_tuple(
                    jac_binning_gpu_u_per_grid_bst_bigsgroup_sm_driver<DIRECTED, vid_t, vid_t, jac_t>, grid, block, 1000, information));
            }
        }
        block.x = k; block.y = 1; block.z = 1;
        grid.x = j; 
        nlohmann::json information = generate_json("u-per-grid-bst-bigsgroup", log2(k), log2(j), ranges[ranges.size()-1], grid, block, 1000);
        kernels.push_back(make_tuple(
                                        jac_binning_gpu_u_per_grid_bst_bigsgroup_sm_driver<DIRECTED, vid_t, vid_t, jac_t>, grid, block, 1000, information));
        all_kernels.push_back(make_tuple("small-sm-sg"+string(1,(char)((int)log2(k)+'a'))+to_string(k)+"-sa"+string(1,(char)((int)log2(j)+'a'))+to_string(j),kernels, split_vertices_by_ranges_cugraph_heur<vid_t, vid_t>));
        kernels.clear();
      }
    }
  }
/////////////////////////////////////////////////////////////////////////
  if (binning_experiment_json.contains("small")){
    vector<int> g_values = binning_experiment_json["small"]["g"];
    vector<int> a_values = binning_experiment_json["small"]["a"];
    for (auto k : g_values){
      for (auto j : a_values){
        for (int i =0; i< ranges.size(); i++){
            dim3 block(1,1,1), grid(1,1,1); 
            block.x = k; block.y = max(1, WARP_SIZE/block.x); block.z = 1;
            grid.x = max(1, j/block.y);
            int g = block.x, a = block.y*grid.x;
            nlohmann::json information = generate_json("u-per-grid-bst", log2(g), log2(a), ranges[i], grid, block, 1000);
            kernels.push_back(make_tuple(
                                            jac_binning_gpu_u_per_grid_bst_driver<DIRECTED, vid_t, vid_t, jac_t>, grid, block, 1000, information));
        }
        block.x = k; block.y = 1; block.z = 1;
        grid.x = j; 
        nlohmann::json information = generate_json("u-per-grid-bst", log2(k), log2(j), ranges[ranges.size()-1], grid, block, 1000);
        kernels.push_back(make_tuple(
                                        jac_binning_gpu_u_per_grid_bst_driver<DIRECTED, vid_t, vid_t, jac_t>, grid, block, 1000, information));
        all_kernels.push_back(make_tuple("small-nosm-sg"+string(1, (char)((int)log2(k)+'a'))+to_string(k)+"-sa"+string(1,(char)((int)log2(j)+'a'))+to_string(j),kernels, split_vertices_by_ranges_cugraph_heur<vid_t, vid_t>));
        kernels.clear();
      }
    }
  }
/////////////////////////////////////////////////////////////////////////
// LARGE
  if (binning_experiment_json.contains("large")){
    vector<int> g_values = binning_experiment_json["large"]["g"];
    vector<int> a_values = binning_experiment_json["large"]["a"];
    for (auto k : g_values){
      for (auto j : a_values){
        for (int i =0; i< ranges.size(); i++){
            dim3 block(1,1,1), grid(1,1,1); 
            block.x = k; block.y = 1; block.z = 1;
            grid.x = j;
            nlohmann::json information = generate_json("u-per-grid-bst-bigsgroup", log2(k), log2(j), ranges[i], grid, block, 1000);
            kernels.push_back(make_tuple( 
                                            jac_binning_gpu_u_per_grid_bst_bigsgroup_sm_driver<DIRECTED, vid_t, vid_t, jac_t>, grid, block, 1000, information));
        }
        block.x = k; block.y = 1; block.z = 1;
        grid.x = j; 
        nlohmann::json information = generate_json("u-per-grid-bst-bigsgroup", log2(k), log2(j), ranges[ranges.size()-1], grid, block, 1000);
        kernels.push_back(make_tuple(
                                        jac_binning_gpu_u_per_grid_bst_bigsgroup_sm_driver<DIRECTED, vid_t, vid_t, jac_t>, grid, block, 1000, information));
        all_kernels.push_back(make_tuple("large-nosm-sg"+string(1, (char)((int)log2(k)+'a'))+to_string(k)+"-sa"+string(1,(char)((int)log2(j)+'a'))+to_string(j),kernels, split_vertices_by_ranges_cugraph_heur<vid_t, vid_t>));
        kernels.clear();
      }
    }
  }
/////////////////////////////////////////////////////////////////////////
  if (binning_experiment_json.contains("large-sm")){
    vector<int> g_values = binning_experiment_json["large-sm"]["g"];
    vector<int> a_values = binning_experiment_json["large-sm"]["a"];
    for (auto k : g_values){
      for (auto j : a_values){
      for (int i =0; i< ranges.size(); i++){
          dim3 block(1,1,1), grid(1,1,1); 
          block.x = k; block.y = 1; block.z = 1;
          grid.x = j;
          int sm_fac = ranges[i]; 
          if (sm_fac*sizeof(vid_t)+block.x/WARP_SIZE*sizeof(vid_t) <= max_sm){
            nlohmann::json information = generate_json("u-per-grid-bst-inv-sm-biggroup", log2(k), log2(j), ranges[i], grid, block, sm_fac);
            kernels.push_back(make_tuple(jac_binning_gpu_u_per_grid_bst_inv_sm_bigsgroup_driver<DIRECTED, vid_t, vid_t, jac_t>, grid, block, sm_fac, information));
          } else {
            dim3 block(1,1,1), grid(1,1,1); 
            block.x = k; block.y = 1; block.z = 1;
            grid.x = j;
            nlohmann::json information = generate_json("u-per-grid-bst-bigsgroup", log2(k), log2(j), ranges[i], grid, block, 1000);
            kernels.push_back(make_tuple(
                                             jac_binning_gpu_u_per_grid_bst_bigsgroup_sm_driver<DIRECTED, vid_t, vid_t, jac_t>, grid, block, 1000, information));
          }
      }
      block.x = k; block.y = 1; block.z = 1;
      grid.x = j; 
      nlohmann::json information = generate_json("u-per-grid-bst-bigsgroup", log2(k), log2(j), ranges[ranges.size()-1], grid, block, 1000);
      kernels.push_back(make_tuple( 
                                       jac_binning_gpu_u_per_grid_bst_bigsgroup_sm_driver<DIRECTED, vid_t, vid_t, jac_t>, grid, block, 1000, information));
      all_kernels.push_back(make_tuple("large-sm-sg"+string(1, (char)((int)log2(k)+'a'))+to_string(k)+"-sa"+string(1,(char)((int)log2(j)+'a'))+to_string(j),kernels, split_vertices_by_ranges_cugraph_heur<vid_t, vid_t>));
      kernels.clear();
    }
  }
  }
/////////////////////////////////////////////////////////////////////////
  vector<tuple<pair<unsigned long long, unsigned long long>, double, nlohmann::json>> kernel_time;
  vector<vector<tuple<pair<unsigned long long, unsigned long long>, double, nlohmann::json>>> kernel_times;
  for (auto kernel_splitter : all_kernels){
    total_time = 0;
    auto name = get<0>(kernel_splitter);
    auto one_kernels = get<1>(kernel_splitter);
    auto splitter_function = get<2>(kernel_splitter);
    for (int i =0; i<num_average; i++){
      start = omp_get_wtime();
      kernel_time = binning_based_jaccard<DIRECTED, vid_t, vid_t, jac_t>(g_d.is, g_d.xadj, g_d.adj, g_d.tadj, g_d.xadj_start, emetrics_cuda_d, g.is, g.xadj, g.adj, g.tadj, g.xadj_start, emetrics_cuda, g.n, g.m, splitter_function, ranges, one_kernels);
      end = omp_get_wtime();
      total_time+=end-start;
      kernel_times.push_back(kernel_time);
    }
    kernel_time = average_kernel_times(kernel_times);
    kernel_times.clear();
    end = total_time/num_average;
    validate_and_write_binning(g,  kernel_time, name, emetrics, emetrics_cuda, total_time, num_average, output_json_file_name, output_json, jaccards_output_path, have_correct);
    kernel_time.clear();
    gpuErrchk( cudaMemset(emetrics_cuda_d, 0, sizeof(jac_t) * g.m * (ull)1) );
  }
#endif
#endif
    return 0;
}
