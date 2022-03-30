#ifndef _BINNING
#define _BINNING
using namespace std;
// type definitions

template <typename EN, typename VID>
using SEP_FUNC = void (*)(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, vector<EN*> &bins, vector<pair<unsigned long long, unsigned long long>>& bin_sizes, vector<EN> ranges);

template <bool directed, typename EN, typename VID, typename E>
using JAC_FUNC = double (*)(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit);

template <bool directed, typename EN, typename VID, typename E>
using JAC_FUNC_GA = string (*)(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, size_t range, size_t g, size_t a, cudaStream_t * stream);

template <bool directed, typename EN, typename VID, typename E>
using STRAT_FUNC = std::tuple<JAC_FUNC_GA<directed, EN, VID, E>, size_t, size_t, size_t>(*)(int bin_id, size_t range, vector<EN> ranges, vector<pair<unsigned long long, unsigned long long>> bin_sizes, size_t max_threads, size_t max_shared_memory);

// entire binning driver function
template <bool directed, typename EN, typename VID, typename E>
vector<tuple<string, pair<unsigned long long, unsigned long long>, double>> binning_based_jaccard(
    // GPU variables
    VID * is_d, EN* xadj_d, VID* adj_d, EN * tadj_d, EN * xadj_start_d, E* d_jac,
    // CPU variables
    VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, E* h_jac,
    // ints
    VID n, EN m, 
    // splitter 
    SEP_FUNC<EN, VID> sep_f, vector<EN> ranges, 
    // jaccard kernel drivers
    vector<pair<string, JAC_FUNC<directed, EN, VID, E>>> jaccard_kernels,
    vector<VID> SM_FACTORS);

// Splitter functions
template <typename EN, typename VID>
void small_large(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, vector<EN*> &bins, vector<pair<unsigned long long, unsigned long long>>& bin_sizes, vector<EN> ranges);

template <typename EN, typename VID>
void split_vertices_by_ranges(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, vector<EN*> &bins, vector<pair<unsigned long long, unsigned long long>>& bin_sizes, vector<EN> ranges);

// Jaccnar calculation driver functions

template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_thread_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit);

template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_warp_bst_sm_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit);
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_warp_bst_inv_sm_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit);
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_warp_sm_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit);
/*
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_block_sm_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit);
*/
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_block_bst_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit);
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_block_bst_sm_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit);
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_block_bst_inv_sm_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit);

// Paper kernels
template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_grid_bst_inv_sm_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit);

template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_grid_bst_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit);

template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_grid_bst_inv_sm_bigsgroup_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit);

template <bool directed, typename EN, typename VID, typename E>
double jac_binning_gpu_u_per_grid_bst_bigsgroup_sm_driver(VID * is, EN* xadj, VID* adj, EN * tadj, EN * xadj_start, VID n, E * jac, EN* bin, EN bin_size, VID SM_FAC, dim3 grid, dim3 block, EN lower_limit, EN upper_limit);
#include "edge_metrics_binning.cu"

#endif
