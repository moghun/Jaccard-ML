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

#include "metric_formulas.h"
#define MINDEGG 32
#define MAXDEGG 96
#define CUDA_MAX_BLOCKS 65535
#define CUDA_MAX_KERNEL_THREADS 256


using namespace std;

#ifndef _EDGE_METRICS
#define _EDGE_METRICS

template <bool directed, typename EN, typename VID, typename E>
void edge_based_metrics(VID * is, EN* xadj, VID* adj, VID n, E* emetrics) {
#pragma omp parallel
  {
    VID* markers = new VID[n]; for(VID i = 0; i < n; i++) {markers[i] =(VID)-1;}

#pragma omp for schedule(dynamic, 256)
    for(VID u = 0; u < n; u++) {
      for(EN ptr = xadj[u]; ptr < xadj[u+1]; ptr++) {
        markers[adj[ptr]] = u;
      }

      for(EN ptr = xadj[u]; ptr < xadj[u+1]; ptr++) {
        VID v = adj[ptr];
        
        if (!directed){
          if(xadj[u+1]-xadj[u] < xadj[v+1]-xadj[v] || (xadj[u+1]-xadj[u] == xadj[v+1]-xadj[v] && u > v)) {
            continue;
          }
        }
        EN other_ptr, intersection_size = 0;

        for(EN ptr_v = xadj[v]; ptr_v < xadj[v+1]; ptr_v++) {
          VID w = adj[ptr_v];
          if(w == u) {
            other_ptr = ptr_v;
          } else if(markers[w] == u) {
            intersection_size++;
          }
        }

        EN edg_u = xadj[u+1]-xadj[u];
        EN edg_v = xadj[v+1]-xadj[v];
        if (!directed)
          emetrics[ptr] = emetrics[other_ptr] = (float)intersection_size/(edg_u+edg_v-intersection_size);
        else
          emetrics[ptr] = (float)intersection_size/(edg_u+edg_v-intersection_size);
      }
    }
    delete [] markers;
  }
}

template <bool directed, typename EN, typename VID, typename E>
void edge_based_metrics_bitmap(VID * is, EN* xadj, VID* adj, VID n, E* emetrics) {
#pragma omp parallel
  {
    VID bsize = (n + 63) / 64;
    uint64_t* markers = new uint64_t[bsize];
    for(VID i = 0; i < bsize; i++) {
      markers[i] = 0uLL;
    }

#pragma omp for schedule(dynamic, 256)
    for(VID u = 0; u < n; u++) {

      for(EN ptr = xadj[u]; ptr < xadj[u+1]; ptr++) {
        VID nbr = adj[ptr];
        VID bid = nbr / 64;
        VID bindex = nbr % 64;
        markers[bid] |= (1uLL << bindex);
      }

      for(EN ptr = xadj[u]; ptr < xadj[u+1]; ptr++) {
        VID v = adj[ptr];

        if (!directed){
          if(xadj[u+1]-xadj[u] < xadj[v+1]-xadj[v] || (xadj[u+1]-xadj[u] == xadj[v+1]-xadj[v] && u > v)) {
            continue;
          }
        }
        VID other_ptr, intersection_size = 0;

        for(EN ptr_v = xadj[v]; ptr_v < xadj[v+1]; ptr_v++) {
          VID w = adj[ptr_v];
          if(w == u) {
            other_ptr = ptr_v;
          } else {
            VID bid = w / 64;
            VID bindex = w % 64;	      
            if((markers[bid]) & (1uL << bindex)) {
              intersection_size++;
            }
          }
        } 

        EN edg_u = xadj[u+1]-xadj[u];
        EN edg_v = xadj[v+1]-xadj[v];
        if (!directed)
          emetrics[ptr] = emetrics[other_ptr] = (float)intersection_size/(edg_u+edg_v-intersection_size);
        else
          emetrics[ptr] = (float)intersection_size/(edg_u+edg_v-intersection_size);
      }

      for(EN ptr = xadj[u]; ptr < xadj[u+1]; ptr++) {
        VID nbr = adj[ptr];
        VID bid = nbr / 64;
        markers[bid] = 0uLL;
      }      
    }

    delete [] markers;
  }
}
// some trials which did not work 
template <typename EN, typename VID, typename E>
double edge_based_metrics_bitmap_sort(VID * is, EN* xadj, VID* adj, EN* tadj, VID n, E* emetrics) {

  //sort vertices according to edge number (count sort)
  double start = omp_get_wtime();
  
  EN *edge_count = new EN[n + 1]();
  EN *mrkr = new EN[n]();
  for (VID i = 0; i < n; i++)
  {
    edge_count[n - (xadj[i + 1] - xadj[i]) + 1]++;
  }

  for (VID i = 1; i <= n; i++)
  {
    edge_count[i] += edge_count[i - 1];
  }

  vector<pair<EN,VID> > vertices(n);
  for (VID i = 0; i < n; i++)
  {
    EN edges = xadj[i + 1] - xadj[i];
    EN ec = edge_count[edges]; // number of vertices who have more edges than (V[i+1] - V[i])
    vertices[ec + mrkr[ec]].second = i;
    vertices[ec + mrkr[ec]].first = edges;
    mrkr[ec]++;
  }
  delete[] mrkr;
  delete[] edge_count;
  EN* invperm = new VID[n];
  for(VID i = 0; i < n; i++) {
    invperm[vertices[i].second] = i;
  }
  /*
  vector<pair<EN,VID> > vertices(n);
  for(VID u = 0; u < n; u++) {
    vertices[u].second = u;
    vertices[u].first = xadj[u] - xadj[u + 1];
  }
  sort(vertices.begin(), vertices.end());
  EN* invperm = new VID[n];
  for(VID i = 0; i < n; i++) {
    invperm[vertices[i].second] = i;
  }
  */
  double end = omp_get_wtime();
  //cout << "Sorting takes " << end - start << " time" << endl;

#pragma omp parallel
  {
    VID bsize = (n + 63) / 64;
    uint64_t* markers = new uint64_t[bsize];
    for(VID i = 0; i < bsize; i++) {
      markers[i] = 0uLL;
    }

#pragma omp for schedule(dynamic, 256)
    for(VID i = 0; i < n; i++) {
      VID u = vertices[i].second;

      for(EN ptr = xadj[u]; ptr < xadj[u+1]; ptr++) {
        VID nbr = adj[ptr];
        VID bid = nbr / 64;
        VID bindex = nbr % 64;
        markers[bid] |= (1uLL << bindex);
      }

      for(EN ptr = xadj[u]; ptr < xadj[u+1]; ptr++) {
        VID v = adj[ptr];

        if(invperm[u] < invperm[v]) {
          EN other_ptr = tadj[ptr], intersection_size = 0;

          for(int ptr_v = xadj[v]; ptr_v < xadj[v+1]; ptr_v++) {
            VID w = adj[ptr_v];
            VID bid = w / 64;
            VID bindex = w % 64;	      
            if((markers[bid]) & (1uL << bindex)) {
              intersection_size++;
            }
          }

          emetrics[ptr] = emetrics[other_ptr] = intersection_size;
        }
      }

      for(EN ptr = xadj[u]; ptr < xadj[u+1]; ptr++) {
        VID nbr = adj[ptr];
        VID bid = nbr / 64;
        markers[bid] = 0uLL;
      }      
    }

    delete [] markers;
  }
#pragma omp for schedule(dynamic, 256)
    for (EN i = 0; i < xadj[n]; i++){
      EN edg_u = xadj[adj[i]+1]-xadj[adj[i]];
      EN edg_v = xadj[is[i]+1]-xadj[is[i]];
      emetrics[i] = emetrics[i]/(edg_u+edg_v-emetrics[i]);
    }
    delete [] invperm;
    return end-start;
}

namespace dongarra {


// generate an edge list with the first column in rowidxJ and second column in colidxJ
template <typename EN, typename VID>
void generate_nonzero_arrays(VID *&rowidxJ, VID *&colidxJ, EN * xadj, VID * adj, VID num_vertices){
  EN counter = 0;
  for (VID i =0; i<num_vertices;i++){
    for (EN j =0;j<xadj[i+1]-xadj[i]; j++){
      rowidxJ[counter]  = i;
      colidxJ[counter] = adj[counter];
      counter++;
    }
  }
}
// nnzj = number of non-zero elements in j (num edges)
// rowidxJ, colidxJ = arrays with the source-dest of each edge
// i.e given an edge list, rowidxJ is the first col and colidxJ is the second col
// rowptrA - xadj array
// colidxA - adj array
// ASSUMPTION - CSR is sorted (edges of each vertex are sorted by vID)
template <typename EN, typename VID, typename E>
__global__ void dongarra_jaccard(VID num_rows, VID num_cols, EN nnzj, VID * rowidxJ, VID * colidxJ, E * valJ, EN *rowptrA, VID *colidxA, E*valA){
  EN i, j, il, iu, jl, ju;
  EN k=blockDim.x * gridDim.x * blockIdx.y +  // number
    blockDim.x * blockIdx.x + threadIdx.x;
  E sum_i, sum_j, cap;
  // cap is the intersection

  if (k < nnzj){ // for each nonzero Jaccard value
    i = rowidxJ[k]; j=colidxJ[k];

    if (i != j){ // if not a self-loop
      il = rowptrA[i]; iu = rowptrA[j];
      // il - index of first edge of i
      // ij - index of first edge of j
      sum_i = 0; sum_j = 0; cap = 0;
      sum_i = rowptrA[i+1] - rowptrA[i]; // edges of i
      sum_j = rowptrA[j+1] - rowptrA[j]; // edges of j
      while (il < rowptrA[i+1] && iu < rowptrA[j+1]){ // for all the edges of i and j
        jl = colidxJ[il]; ju = colidxJ[iu];
        // jl -  all the neighbors of l
        // ju -  all the neighbors of u 
        //if (k%10000==0)if (ju != j) printf("you dumb\n");// k %d i %d j %d iu %d ju %d\n");
        cap = (jl == ju) ? cap+1 : cap;
        il = (jl <= ju) ? il+1: il;
        iu = (ju <= jl) ? iu+1: iu;
      }
      valJ[k] = cap / (sum_i + sum_j - cap);
    } else {
      valJ[k] = 1.0;
    }
  }
}
}

namespace inhouse_cugraph {

template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
__global__ void jaccard_is_nosum(vertex_t n,
                           edge_t const *csrPtr,
                           vertex_t const *csrInd,
                           weight_t const *v,
                           weight_t *work,
                           weight_t *weight_i)
{
  edge_t i, j, Ni, Nj;
  vertex_t row, col;
  vertex_t ref, cur, ref_col, cur_col, match;
  weight_t ref_val;

  for (row = threadIdx.z + blockIdx.z * blockDim.z; row < n; row += gridDim.z * blockDim.z) {
    for (j = csrPtr[row] + threadIdx.y + blockIdx.y * blockDim.y; j < csrPtr[row + 1];
         j += gridDim.y * blockDim.y) {
      col = csrInd[j];
      // find which row has least elements (and call it reference row)
      Ni  = csrPtr[row + 1] - csrPtr[row];
      Nj  = csrPtr[col + 1] - csrPtr[col];
      ref = (Ni < Nj) ? row : col;
      cur = (Ni < Nj) ? col : row;

      // compute new sum weights

      // compute new intersection weights
      // search for the element with the same column index in the reference row
      for (i = csrPtr[ref] + threadIdx.x + blockIdx.x * blockDim.x; i < csrPtr[ref + 1];
           i += gridDim.x * blockDim.x) {
        match   = (vertex_t)-1;
        ref_col = csrInd[i];
        if (weighted) {
          ref_val = v[ref_col];
        } else {
          ref_val = 1.0;
        }

        // binary search (column indices are sorted within each row)
        edge_t left  = csrPtr[cur]+1;
        edge_t right = csrPtr[cur + 1];
        while (left <= right) {
          edge_t middle = ((unsigned long long)left + (unsigned long long)right) >> 1;
          cur_col       = csrInd[middle-1];
          if (cur_col > ref_col) {
            right = middle - 1;
          } else if (cur_col < ref_col) {
            left = middle + 1;
          } else {
            match = middle-1;
            break;
          }
        }

        // if the element with the same column index in the reference row has been found
        if (match != (vertex_t)-1) { atomicAdd(&weight_i[j], ref_val); }
      }
    }
  }
}

// Volume of intersections (*weight_i) and cumulated volume of neighboors (*weight_s)
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
__global__ void jaccard_is(vertex_t n,
                           edge_t const *csrPtr,
                           vertex_t const *csrInd,
                           weight_t const *v,
                           weight_t *work,
                           weight_t *weight_i,
                           weight_t *weight_s)
{
  edge_t i, j, Ni, Nj;
  vertex_t row, col;
  vertex_t ref, cur, ref_col, cur_col, match;
  weight_t ref_val;

  for (row = threadIdx.z + blockIdx.z * blockDim.z; row < n; row += gridDim.z * blockDim.z) {
    for (j = csrPtr[row] + threadIdx.y + blockIdx.y * blockDim.y; j < csrPtr[row + 1];
         j += gridDim.y * blockDim.y) {
      col = csrInd[j];
      // find which row has least elements (and call it reference row)
      Ni  = csrPtr[row + 1] - csrPtr[row];
      Nj  = csrPtr[col + 1] - csrPtr[col];
      ref = (Ni < Nj) ? row : col;
      cur = (Ni < Nj) ? col : row;

      // compute new sum weights
      weight_s[j] = work[row] + work[col];

      // compute new intersection weights
      // search for the element with the same column index in the reference row
      for (i = csrPtr[ref] + threadIdx.x + blockIdx.x * blockDim.x; i < csrPtr[ref + 1];
           i += gridDim.x * blockDim.x) {
        match   = (vertex_t)-1;
        ref_col = csrInd[i];
        if (weighted) {
          ref_val = v[ref_col];
        } else {
          ref_val = 1.0;
        }

        // binary search (column indices are sorted within each row)
        edge_t left  = csrPtr[cur]+1;
        edge_t right = csrPtr[cur + 1];
        while (left <= right) {
          edge_t middle = ((unsigned long long)left + (unsigned long long)right) >> 1;
          cur_col       = csrInd[middle-1];
          if (cur_col > ref_col) {
            right = middle - 1;
          } else if (cur_col < ref_col) {
            left = middle + 1;
          } else {
            match = middle-1;
            break;
          }
        }

        // if the element with the same column index in the reference row has been found
        if (match != (vertex_t)-1) { atomicAdd(&weight_i[j], ref_val); }
      }
    }
  }
}

// Volume of neighboors (*weight_s)
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
__global__ void jaccard_row_sum(
  vertex_t n, edge_t const *csrPtr, vertex_t const *csrInd, weight_t const *v, weight_t *work)
{
  vertex_t row;
  edge_t start, end, length;
  weight_t sum;

  for (row = threadIdx.y + blockIdx.y * blockDim.y; row < n; row += gridDim.y * blockDim.y) {
    start  = csrPtr[row];
    end    = csrPtr[row + 1];
    length = end - start;

    // compute row sums
    if (weighted) {
      if (threadIdx.x == 0) work[row] = sum;
    } else {
      work[row] = static_cast<weight_t>(length);
    }
  }
}
// Jaccard  weights (*weight)
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
__global__ void jaccard_jw_nosum(edge_t e,
                           weight_t const *weight_i,
                           edge_t const * xadj,
                           vertex_t const * adj,
                           vertex_t const * is,
                           weight_t * weight_j)
{
  edge_t j;
  weight_t Wi, Ws, Wu;

  for (j = threadIdx.x + blockIdx.x * blockDim.x; j < e; j += gridDim.x * blockDim.x) {
    Wi          = weight_i[j];
    Ws          = xadj[adj[j]+1]-xadj[adj[j]] + xadj[is[j]+1]-xadj[is[j]];
    Wu          = Ws - Wi;
    weight_j[j] = (Wi / Wu);
  }
}
// Jaccard  weights (*weight)
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
__global__ void jaccard_jw(edge_t e,
                           weight_t const *weight_i,
                           weight_t const *weight_s,
                           weight_t *weight_j)
{
  edge_t j;
  weight_t Wi, Ws, Wu;

  for (j = threadIdx.x + blockIdx.x * blockDim.x; j < e; j += gridDim.x * blockDim.x) {
    Wi          = weight_i[j];
    Ws          = weight_s[j];
    Wu          = Ws - Wi;
    weight_j[j] = (Wi / Wu);
  }
}

template <bool weighted, typename edge_t, typename vertex_t, typename weight_t>
void cugraph_jaccard_nosum(const vertex_t* is, const edge_t* xadj, const vertex_t* adj, vertex_t n, edge_t m, weight_t* emetrics){
  weight_t *weight_i, *work;
  edge_t e = m;
  edge_t const *csrPtr = xadj;
  vertex_t const * csrInd = adj;
  weight_t const *weight_in = nullptr;
  weight_t  *weight_j = emetrics; 
  cudaEvent_t e1, e2, e3, e4, e5, e6;
  gpuErrchk( cudaEventCreate(&e1) );
  gpuErrchk( cudaEventCreate(&e2) );
  gpuErrchk( cudaEventCreate(&e3) );
  gpuErrchk( cudaEventCreate(&e4) );
  gpuErrchk( cudaEventCreate(&e5) );
  gpuErrchk( cudaEventCreate(&e6) );
  gpuErrchk( vcudaMalloc((void**)&weight_i, sizeof(weight_t) * m ) );
  gpuErrchk( vcudaMalloc((void**)&work, sizeof(weight_t) * n ) );

  dim3 nthreads, nblocks;
  int y = 4;

  // setup launch configuration
  nthreads.x = 32;
  nthreads.y = y;
  nthreads.z = 1;
  nblocks.x  = 1;
  nblocks.y  = min((n + nthreads.y - 1) / nthreads.y, vertex_t{CUDA_MAX_BLOCKS});
  nblocks.z  = 1;
  // launch kernel
  gpuErrchk( cudaEventRecord(e1) );
  jaccard_row_sum<weighted, vertex_t, edge_t, weight_t>
    <<<nblocks, nthreads>>>(n, csrPtr, csrInd, weight_in, work);
  gpuErrchk( cudaEventRecord(e2) );
  cudaDeviceSynchronize();
  gpuErrchk( cudaMemset(weight_i, 0, sizeof(weight_t) * m ) );

  // setup launch configuration
  nthreads.x = 32 / y;
  nthreads.y = y;
  nthreads.z = 8;
  nblocks.x  = 1;
  nblocks.y  = 1;
  nblocks.z  = min((n + nthreads.z - 1) / nthreads.z, vertex_t{CUDA_MAX_BLOCKS});  // 1;

  // launch kernel
  gpuErrchk( cudaEventRecord(e3) );
  jaccard_is_nosum<weighted, vertex_t, edge_t, weight_t>
    <<<nblocks, nthreads>>>(n, csrPtr, csrInd, weight_in, work, weight_i);
  gpuErrchk( cudaEventRecord(e4) );

  // setup launch configuration
  nthreads.x = min(e, edge_t{CUDA_MAX_KERNEL_THREADS});
  nthreads.y = 1;
  nthreads.z = 1;
  nblocks.x  = min((e + nthreads.x - 1) / nthreads.x, edge_t{CUDA_MAX_BLOCKS});
  nblocks.y  = 1;
  nblocks.z  = 1;

  // launch kernel
  gpuErrchk( cudaEventRecord(e5) );
  jaccard_jw_nosum<weighted, vertex_t, edge_t, weight_t>
    <<<nblocks, nthreads>>>(e, weight_i, xadj, adj, is, weight_j);
  gpuErrchk( cudaEventRecord(e6) );
  cudaDeviceSynchronize();
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, e1, e2);
  cout << "jaccard_row_um " << milliseconds << endl;
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, e3, e4);
  cout << "jaccard_is " << milliseconds << endl;
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, e5, e6);
  cout << "jaccard_jw " << milliseconds << endl;
  gpuErrchk( cudaFree(weight_i));
  gpuErrchk( cudaFree(work));
}
template <bool weighted, typename edge_t, typename vertex_t, typename weight_t>
void cugraph_jaccard(const vertex_t* is, const edge_t* xadj, const vertex_t* adj, vertex_t n, edge_t m, weight_t* emetrics){
  weight_t *weight_i, *weight_s, *work;
  edge_t e = m;
  edge_t const *csrPtr = xadj;
  vertex_t const * csrInd = adj;
  weight_t const *weight_in = nullptr;
  weight_t  *weight_j = emetrics; 
  cudaEvent_t e1, e2, e3, e4, e5, e6;
  gpuErrchk( cudaEventCreate(&e1) );
  gpuErrchk( cudaEventCreate(&e2) );
  gpuErrchk( cudaEventCreate(&e3) );
  gpuErrchk( cudaEventCreate(&e4) );
  gpuErrchk( cudaEventCreate(&e5) );
  gpuErrchk( cudaEventCreate(&e6) );
  gpuErrchk( vcudaMalloc((void**)&weight_i, sizeof(weight_t) * m ) );
  gpuErrchk( vcudaMalloc((void**)&weight_s, sizeof(weight_t) * m ) );
  gpuErrchk( vcudaMalloc((void**)&work, sizeof(weight_t) * n ) );

  dim3 nthreads, nblocks;
  int y = 4;

  // setup launch configuration
  nthreads.x = 32;
  nthreads.y = y;
  nthreads.z = 1;
  nblocks.x  = 1;
  nblocks.y  = min((n + nthreads.y - 1) / nthreads.y, vertex_t{CUDA_MAX_BLOCKS});
  nblocks.z  = 1;
  // launch kernel
  gpuErrchk( cudaEventRecord(e1) );
  jaccard_row_sum<weighted, vertex_t, edge_t, weight_t>
    <<<nblocks, nthreads>>>(n, csrPtr, csrInd, weight_in, work);
  gpuErrchk( cudaEventRecord(e2) );
  cudaDeviceSynchronize();
  gpuErrchk( cudaMemset(weight_i, 0, sizeof(weight_t) * m ) );

  // setup launch configuration
  nthreads.x = 32 / y;
  nthreads.y = y;
  nthreads.z = 8;
  nblocks.x  = 1;
  nblocks.y  = 1;
  nblocks.z  = min((n + nthreads.z - 1) / nthreads.z, vertex_t{CUDA_MAX_BLOCKS});  // 1;

  // launch kernel
  gpuErrchk( cudaEventRecord(e3) );
  jaccard_is<weighted, vertex_t, edge_t, weight_t>
    <<<nblocks, nthreads>>>(n, csrPtr, csrInd, weight_in, work, weight_i, weight_s);
  gpuErrchk( cudaEventRecord(e4) );

  // setup launch configuration
  nthreads.x = min(e, edge_t{CUDA_MAX_KERNEL_THREADS});
  nthreads.y = 1;
  nthreads.z = 1;
  nblocks.x  = min((e + nthreads.x - 1) / nthreads.x, edge_t{CUDA_MAX_BLOCKS});
  nblocks.y  = 1;
  nblocks.z  = 1;

  // launch kernel
  gpuErrchk( cudaEventRecord(e5) );
  jaccard_jw<weighted, vertex_t, edge_t, weight_t>
    <<<nblocks, nthreads>>>(e, weight_i, weight_s, weight_j);
  gpuErrchk( cudaEventRecord(e6) );
  cudaDeviceSynchronize();
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, e1, e2);
  cout << "jaccard_row_um " << milliseconds << endl;
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, e3, e4);
  cout << "jaccard_is " << milliseconds << endl;
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, e5, e6);
  cout << "jaccard_jw " << milliseconds << endl;
  gpuErrchk( cudaFree(weight_i));
  gpuErrchk( cudaFree(weight_s));
  gpuErrchk( cudaFree(work));
}

}
template <bool directed, typename EN, typename VID, typename E>
__global__ void jac_edge_based_small(const EN* __restrict__ xadj, const VID* __restrict__ adj, const EN* __restrict__ is, VID n, E* __restrict__ emetrics) {
  
  //int no_threads = blockDim.z * blockDim.y * blockDim.x * gridDim.x;
  unsigned int block_local_id =  blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  unsigned int grid_id = gridDim.x*gridDim.y*blockIdx.z+gridDim.x*blockIdx.y+blockIdx.x;
  unsigned long long tid = (long long)blockDim.z * blockDim.x * blockDim.y * grid_id + (unsigned long long)block_local_id;
  //if (tid==0)printf("Running jac_binning_gpu_u_per_grid_bst_kernel\n");
  unsigned mask = calculateMask(blockDim.x, tid);//threadIdx.y*blockDim.x+threadIdx.x);

  EN m = xadj[n];
  for(EN ptr = threadIdx.y+ blockDim.y* blockIdx.y+blockDim.y*blockIdx.z*gridDim.y; ptr < m; ptr += gridDim.y*gridDim.z*blockDim.y) {
    EN v = adj[ptr];
    EN u = is[ptr];
    EN degu = xadj[u+1] - xadj[u];

    bool skippable = (xadj[v+1]-xadj[v] < degu ||(xadj[v+1]-xadj[v] == degu && v > u));
    if (!directed &&skippable)
      continue;
    EN other_ptr = bst(xadj, adj, v, u);
    if (directed && other_ptr != (EN)-1 && skippable)
      continue;
    
    VID src = (degu < xadj[v+1] - xadj[v]) ? u : v;
    VID dst = (degu < xadj[v+1] - xadj[v]) ? v : u;
    EN intersection_size = 0;

    for (EN t_ptr = threadIdx.x; t_ptr < xadj[src+1]-xadj[src]; t_ptr+=blockDim.x){
      EN loc = bst(xadj, adj, dst, adj[xadj[src]+t_ptr]);
      intersection_size+=(loc!=(EN)-1);
    } 

    intersection_size = warpReduce(intersection_size, blockDim.x, mask);
    if (threadIdx.x == 0){
      E jaccard =  float(intersection_size)/float(degu+(xadj[v+1]-xadj[v])-intersection_size);
      emetrics[ptr] = jaccard;
      if (other_ptr != (EN)-1)
        emetrics[other_ptr] = jaccard;
    }
  }
}

template <bool directed, typename EN, typename VID, typename E>
__global__ void edge_based_metrics_cuda(const VID* __restrict__ is, const EN* __restrict__ xadj, const VID* __restrict__ adj, VID n,
    E* __restrict__ emetrics, VID no_emetrics) {

  int no_threads = blockDim.x * gridDim.x;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  EN m = xadj[n];

  for(EN ptr = tid; ptr < m; ptr += no_threads) {
    VID u = is[ptr];
    VID v = adj[ptr];
    EN other_ptr;
    if (directed) other_ptr = bst(xadj, adj, v, u);


    if(!directed && u < v || (directed && (other_ptr==(EN)-1 || (other_ptr != (EN)-1 && u < v)))) {
      EN intersection_size = 0;

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

      EN eu =  xadj[u+1] -xadj[u];
      EN ev =  xadj[v+1] -xadj[v];
      E J = float(intersection_size)/float(eu+ev-intersection_size);
      emetrics[ptr] = J;
      if (other_ptr != (EN) -1)
        emetrics[other_ptr] = J;
    }
  }
}

template <typename EN, typename VID, typename E>
__global__ void edge_based_metrics_cuda_small(const VID* __restrict__ is, const EN* __restrict__ xadj, const VID* __restrict__ adj, VID n,
    E* __restrict__ emetrics, VID no_emetrics) {

  int no_threads = blockDim.x * gridDim.x;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  EN m = xadj[n];

  for(EN ptr = tid; ptr < m; ptr += no_threads) {
    VID u = is[ptr];

    EN degu = xadj[u+1] - xadj[u];	
    if(degu < MINDEGG || degu > MAXDEGG) {
      VID v = adj[ptr];

      if(u < v) {
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
        emetrics[ptr] = emetrics[other_ptr] = float(intersection_size)/float(eu+ev-intersection_size);
#elif defined SOA
      EN eu =  xadj[u+1] -xadj[u];
      EN ev =  xadj[v+1] -xadj[v];
      emetrics[ptr] = emetrics[other_ptr] = float(intersection_size)/float(eu+ev-intersection_size);
#endif

      }
    }
  }
}



template <typename EN, typename VID, typename E>
__global__ void edge_based_metrics_cuda_large_bst(const VID* __restrict__ is, const EN* __restrict__ xadj, const VID* __restrict__ adj, VID n,  E* __restrict__ emetrics, VID no_emetrics) {

  int no_threads = blockDim.y * blockDim.x * gridDim.x; //multiple of 32
  int tid = blockDim.y * blockDim.x * blockDim.z * blockIdx.x + blockDim.x * threadIdx.y + threadIdx.x;
  int block_local_wid = (tid % (blockDim.x * blockDim.y)) / 32;
  int warp_local_tid = tid % 32;

  extern __shared__ VID glob_u_adj[];
  int grid_local_wid = tid / 32;;
  int grid_no_warps = no_threads / 32;
  for(VID u = grid_local_wid; u < n; u += grid_no_warps) {	
    EN degu = xadj[u+1] - xadj[u];

    VID* u_adj = glob_u_adj + (block_local_wid * MAXDEGG);
    if(degu >= MINDEGG && degu <= MAXDEGG) {
      for(EN ptr = warp_local_tid; ptr < degu; ptr += 32) {
        u_adj[ptr] = adj[xadj[u] + ptr]; 
      }

      for(EN ptr = threadIdx.y; ptr < degu; ptr += blockDim.y) {
        VID v = u_adj[ptr];

        if(u < v) {
          EN intersection_size = 0;
          EN other_ptr = bst(xadj, adj, v, u);

          VID t_ptr = 0;
          for (t_ptr = threadIdx.x; t_ptr<degu; t_ptr+=blockDim.x){
            EN loc = bst(xadj, adj, v, u_adj[t_ptr]);
            intersection_size+= (loc!=-1);
          }
          intersection_size = warpReduce(intersection_size, blockDim.x, tid%32, threadIdx.x==0);
        if (threadIdx.x == 0){
#ifdef AOS
            EN eu =  xadj[u+1] -xadj[u];
            EN ev =  xadj[v+1] -xadj[v];
            emetrics[(xadj[u] + ptr)] = emetrics[other_ptr] =  float(intersection_size)/float(eu+ev-intersection_size);
#elif defined SOA
            EN eu =  xadj[u+1] -xadj[u];
            EN ev =  xadj[v+1] -xadj[v];
            emetrics[(xadj[u] + ptr)] = emetrics[other_ptr] =float(intersection_size)/float(eu+ev-intersection_size);
#endif

          }
        }
      }
    }    
  }
}

template <typename EN, typename VID, typename E>
__global__ void edge_based_metrics_cuda_large(const VID* __restrict__ is, const EN* __restrict__ xadj, const VID* __restrict__ adj, VID n,
    E* __restrict__ emetrics, VID no_emetrics) {

  int no_threads = blockDim.x * gridDim.x; //multiple of 32
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int block_local_wid = (tid % blockDim.x) / 32;
  int warp_local_tid = tid % 32;

  extern __shared__ VID glob_u_adj[];
  int grid_local_wid = tid / 32;;
  int grid_no_warps = no_threads / 32;
  for(VID u = grid_local_wid; u < n; u += grid_no_warps) {	
    EN degu = xadj[u+1] - xadj[u];

    VID* u_adj = glob_u_adj + (block_local_wid * MAXDEGG);
    if(degu >= MINDEGG && degu <= MAXDEGG) {
      for(EN ptr = warp_local_tid; ptr < degu; ptr += 32) {
        u_adj[ptr] = adj[xadj[u] + ptr]; 
      }

      for(EN ptr = warp_local_tid; ptr < degu; ptr += 32) {
        VID v = u_adj[ptr];

        if(u < v) {
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
        }
      }
    }    
  }
}

template <bool directed, typename EN, typename VID, typename E>
__global__ void edge_based_on_host(const VID* __restrict__ is, const EN* __restrict__ xadj, const VID* __restrict__ adj, VID n,
    E* __restrict__ emetrics, VID no_emetrics) {

  int no_threads = blockDim.x * gridDim.x;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  EN m = xadj[n];

  for (EN ptr = tid; ptr < m; ptr += no_threads) {
    VID u = is[ptr];
    VID v = adj[ptr];
    EN other_ptr;
    if (directed)
      other_ptr = bst(xadj, adj, v, u);

    if (!directed && u < v) || (directed && (other_ptr == (EN)-1 || (other_ptr != (EN)-1 && u < v))) {
      EN intersection_size = 0;
      EN intersection_size_adamic_adar = 0;
      EN intersection_size_resource_allocation = 0;

      EN ptr_u = xadj[u];
      EN ptr_v = xadj[v];

      while (ptr_u < xadj[u + 1] && ptr_v < xadj[v + 1]) {
        VID u_ngh = adj[ptr_u];
        VID v_ngh = adj[ptr_v];

        if (v_ngh == u) {
          other_ptr = ptr_v;
        }

        if (u_ngh == v_ngh) {
          intersection_size++;

          // (float)(xadj[u_ngh + 1] - xadj[u_ngh])) --> calculates the degree of current neighbour node
          aa_ar_val = (float)(xadj[u_ngh + 1] - xadj[u_ngh])
          intersection_size_adamic_adar += 1.0 / log(aa_ar_val); //for every intersection
          intersection_size_resource_allocation += 1.0 / aa_ar_val;

          ptr_u++;
          ptr_v++;
        } else if (u_ngh < v_ngh) {
          ptr_u++;
        } else {
          ptr_v++;
        }
      }
      SET_INTERSECTION;
      SET_AA;
      SET_RA;
    }
  }
}

template <bool directed, typename EN, typename VID, typename E>
__global__ void edge_based_on_device(const VID* __restrict__ is, const EN* __restrict__ xadj, const VID* __restrict__ adj, VID n,
    E* __restrict__ emetrics, VID no_emetrics) {

  int no_threads = blockDim.x * gridDim.x;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  EN m = xadj[n];

  for (EN ptr = tid; ptr < m; ptr += no_threads) {
    VID u = is[ptr];
    VID v = adj[ptr];
    EN other_ptr;
    if (directed)
      other_ptr = bst(xadj, adj, v, u);

    if ((!directed && u < v) || (directed && (other_ptr == (EN)-1 || (other_ptr != (EN)-1 && u < v)))) {
      EN intersection_size = 0;
      EN intersection_size_adamic_adar = 0;
      EN intersection_size_resource_allocation = 0;

      EN ptr_u = xadj[u];
      EN ptr_v = xadj[v];

      while (ptr_u < xadj[u + 1] && ptr_v < xadj[v + 1]) {
        VID u_ngh = adj[ptr_u];
        VID v_ngh = adj[ptr_v];

        if (v_ngh == u) {
          other_ptr = ptr_v;
        }

        if (u_ngh == v_ngh) {
          intersection_size++;

          // (float)(xadj[u_ngh + 1] - xadj[u_ngh])) --> calculates the degree of current neighbour node
          int aa_ar_val = (float)(xadj[u_ngh + 1] - xadj[u_ngh]);
          intersection_size_adamic_adar += 1.0 / _logf(aa_ar_val); //for every intersection
          intersection_size_resource_allocation += 1.0 / aa_ar_val;

          ptr_u++;
          ptr_v++;
        } else if (u_ngh < v_ngh) {
          ptr_u++;
        } else {
          ptr_v++;
        }
      }

      CALC_JAC;
      CALC_AA;
      CALC_RA;
      CALC_CN;
      CALC_PA;
      CALC_SL;
      CALC_SI;
    }
  }
}




#undef MAXDEGG
#undef MINDEGG
#endif
