// CSR build for batched (vertex-disjoint) subgraphs.
//
// Input is upper-triangular COO with self-loops allowed and vertex IDs
// already pre-offset by g * seq_len, so a graph_id can be recovered by
// integer-division by seq_len (cf. partition.py / build_adjacency_batched).
//
// We mirror non-self-loop edges so per-vertex CSR rows sum to 2*sum(weights
// of incident non-self edges) + selfloop_weight (counted once). 2m for the
// modularity formula uses the same sum-with-mirror-mirroring convention.

#include "internal.cuh"

#include <cub/cub.cuh>
#include <stdexcept>

namespace community_kv::native_leiden {

namespace {

__global__ void k_expand_mirror(
    const int32_t* __restrict__ src_in,
    const int32_t* __restrict__ dst_in,
    const float*   __restrict__ w_in,
    int64_t E_in,
    int32_t* __restrict__ src_out,
    int32_t* __restrict__ dst_out,
    float*   __restrict__ w_out,
    int64_t* __restrict__ counter)
{
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= E_in) return;
  int32_t u = src_in[i];
  int32_t v = dst_in[i];
  float   w = w_in[i];
  if (u == v) {
    // Standard undirected modularity convention: A_ii = 2*w for self-loops.
    // We store the self-loop ONCE in CSR with doubled weight so k_v from
    // the row sum equals the matrix-sense degree.
    int64_t pos = atomicAdd(reinterpret_cast<unsigned long long*>(counter), 1ULL);
    src_out[pos] = u;
    dst_out[pos] = v;
    w_out[pos] = 2.0f * w;
  } else {
    int64_t pos = atomicAdd(reinterpret_cast<unsigned long long*>(counter), 2ULL);
    src_out[pos] = u;
    dst_out[pos] = v;
    w_out[pos] = w;
    src_out[pos + 1] = v;
    dst_out[pos + 1] = u;
    w_out[pos + 1] = w;
  }
}

__global__ void k_compute_kv_and_selfloop(
    const int64_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_idx,
    const float*   __restrict__ col_w,
    int32_t V_total,
    double* __restrict__ k_v,
    float*  __restrict__ selfloop_w)
{
  int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= V_total) return;
  int64_t a = row_ptr[v];
  int64_t b = row_ptr[v + 1];
  double sum = 0.0;
  float self_w = 0.0f;
  for (int64_t e = a; e < b; ++e) {
    float w = col_w[e];
    sum += (double)w;
    // Self-loops are stored ONCE with doubled weight (A_ii convention).
    // Track the original (un-doubled) weight here for aggregation use.
    if (col_idx[e] == v) self_w += 0.5f * w;
  }
  k_v[v] = sum;
  selfloop_w[v] = self_w;
}

__global__ void k_per_graph_2m(
    const double* __restrict__ k_v,
    int32_t V_total,
    int32_t seq_len,
    double* __restrict__ two_m_per_graph)
{
  int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= V_total) return;
  int32_t g = v / seq_len;
  atomicAdd(&two_m_per_graph[g], k_v[v]);
}

__global__ void k_build_row_ptr(
    const int32_t* __restrict__ row_index_sorted,
    int64_t E_sym,
    int32_t V_total,
    int64_t* __restrict__ row_ptr)
{
  // Each thread does a lower_bound for its vertex v in row_index_sorted.
  // V_total is small (G * seq_len ~ tens to low hundreds of thousands) so
  // O(V log E) is negligible.
  int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v > V_total) return;
  int64_t lo = 0;
  int64_t hi = E_sym;
  while (lo < hi) {
    int64_t mid = (lo + hi) >> 1;
    if (row_index_sorted[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  row_ptr[v] = lo;
}

__global__ void k_iota_i64(int64_t* p, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) p[i] = i;
}

__global__ void k_gather_by_perm(
    const int64_t* __restrict__ perm,
    const int32_t* __restrict__ dst_in,
    const float*   __restrict__ w_in,
    int32_t* __restrict__ dst_out,
    float*   __restrict__ w_out,
    int64_t n)
{
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  int64_t src_idx = perm[i];
  dst_out[i] = dst_in[src_idx];
  w_out[i]   = w_in[src_idx];
}

}  // namespace

void free_csr(BatchedCSR& csr, cudaStream_t stream) {
  if (csr.row_ptr)         cudaFreeAsync(csr.row_ptr,         stream);
  if (csr.col_idx)         cudaFreeAsync(csr.col_idx,         stream);
  if (csr.col_w)           cudaFreeAsync(csr.col_w,           stream);
  if (csr.k_v)             cudaFreeAsync(csr.k_v,             stream);
  if (csr.selfloop_w)      cudaFreeAsync(csr.selfloop_w,      stream);
  if (csr.two_m_per_graph) cudaFreeAsync(csr.two_m_per_graph, stream);
  csr = BatchedCSR{};
}

BatchedCSR build_csr_from_upper_coo(
    const int32_t* edge_src,
    const int32_t* edge_dst,
    const float*   edge_weight,
    int64_t        E_in,
    int32_t        G,
    int32_t        seq_len,
    cudaStream_t   stream)
{
  BatchedCSR csr;
  csr.G = G;
  csr.seq_len = seq_len;
  csr.V_total = G * seq_len;

  // Worst case all edges are non-self-loops -> 2*E_in entries.
  int64_t cap = 2 * E_in;
  int32_t *src_mirror = nullptr, *dst_mirror = nullptr;
  float* w_mirror = nullptr;
  int64_t* counter = nullptr;
  CUDA_CHECK(cudaMallocAsync(&src_mirror, cap * sizeof(int32_t), stream));
  CUDA_CHECK(cudaMallocAsync(&dst_mirror, cap * sizeof(int32_t), stream));
  CUDA_CHECK(cudaMallocAsync(&w_mirror,   cap * sizeof(float),   stream));
  CUDA_CHECK(cudaMallocAsync(&counter,    sizeof(int64_t),       stream));
  CUDA_CHECK(cudaMemsetAsync(counter, 0, sizeof(int64_t), stream));

  if (E_in > 0) {
    int block = 256;
    int64_t grid = (E_in + block - 1) / block;
    k_expand_mirror<<<grid, block, 0, stream>>>(
        edge_src, edge_dst, edge_weight, E_in,
        src_mirror, dst_mirror, w_mirror, counter);
  }

  int64_t E_sym = 0;
  CUDA_CHECK(cudaMemcpyAsync(&E_sym, counter, sizeof(int64_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  csr.E_sym = E_sym;

  CUDA_CHECK(cudaMallocAsync(&csr.col_idx, E_sym * sizeof(int32_t), stream));
  CUDA_CHECK(cudaMallocAsync(&csr.col_w,   E_sym * sizeof(float),   stream));
  CUDA_CHECK(cudaMallocAsync(&csr.row_ptr, (csr.V_total + 1) * sizeof(int64_t), stream));

  // Sort by src using one cub call to get a permutation, then gather
  // dst/w through that permutation. Avoids depending on cub being
  // deterministic across two independent SortPairs calls.
  int64_t* perm_in = nullptr;
  int64_t* perm_out = nullptr;
  int32_t* src_sorted = nullptr;
  CUDA_CHECK(cudaMallocAsync(&perm_in,    E_sym * sizeof(int64_t), stream));
  CUDA_CHECK(cudaMallocAsync(&perm_out,   E_sym * sizeof(int64_t), stream));
  CUDA_CHECK(cudaMallocAsync(&src_sorted, E_sym * sizeof(int32_t), stream));

  if (E_sym > 0) {
    int block = 256;
    int64_t grid = (E_sym + block - 1) / block;
    k_iota_i64<<<grid, block, 0, stream>>>(perm_in, E_sym);
  }

  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
      nullptr, temp_bytes,
      src_mirror, src_sorted,
      perm_in, perm_out,
      E_sym, 0, 32, stream);
  void* temp_storage = nullptr;
  if (temp_bytes > 0) {
    CUDA_CHECK(cudaMallocAsync(&temp_storage, temp_bytes, stream));
  }
  if (E_sym > 0) {
    cub::DeviceRadixSort::SortPairs(
        temp_storage, temp_bytes,
        src_mirror, src_sorted,
        perm_in, perm_out,
        E_sym, 0, 32, stream);

    int block = 256;
    int64_t grid = (E_sym + block - 1) / block;
    k_gather_by_perm<<<grid, block, 0, stream>>>(
        perm_out, dst_mirror, w_mirror, csr.col_idx, csr.col_w, E_sym);
  }

  // Build row_ptr via per-vertex lower_bound on src_sorted.
  if (E_sym > 0) {
    int block = 128;
    int grid = (csr.V_total + 1 + block - 1) / block;
    k_build_row_ptr<<<grid, block, 0, stream>>>(
        src_sorted, E_sym, csr.V_total, csr.row_ptr);
  } else {
    CUDA_CHECK(cudaMemsetAsync(csr.row_ptr, 0,
                               (csr.V_total + 1) * sizeof(int64_t), stream));
  }

  // k_v and self-loop weights
  CUDA_CHECK(cudaMallocAsync(&csr.k_v, csr.V_total * sizeof(double), stream));
  CUDA_CHECK(cudaMallocAsync(&csr.selfloop_w, csr.V_total * sizeof(float), stream));
  {
    int block = 128;
    int grid = (csr.V_total + block - 1) / block;
    k_compute_kv_and_selfloop<<<grid, block, 0, stream>>>(
        csr.row_ptr, csr.col_idx, csr.col_w, csr.V_total,
        csr.k_v, csr.selfloop_w);
  }

  // Per-graph 2m
  CUDA_CHECK(cudaMallocAsync(&csr.two_m_per_graph, G * sizeof(double), stream));
  CUDA_CHECK(cudaMemsetAsync(csr.two_m_per_graph, 0, G * sizeof(double), stream));
  if (csr.V_total > 0) {
    int block = 128;
    int grid = (csr.V_total + block - 1) / block;
    k_per_graph_2m<<<grid, block, 0, stream>>>(
        csr.k_v, csr.V_total, seq_len, csr.two_m_per_graph);
  }

  // Free scratch
  cudaFreeAsync(src_mirror, stream);
  cudaFreeAsync(dst_mirror, stream);
  cudaFreeAsync(w_mirror, stream);
  cudaFreeAsync(counter, stream);
  cudaFreeAsync(perm_in, stream);
  cudaFreeAsync(perm_out, stream);
  cudaFreeAsync(src_sorted, stream);
  if (temp_storage) cudaFreeAsync(temp_storage, stream);

  return csr;
}

}  // namespace community_kv::native_leiden
