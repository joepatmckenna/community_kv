// COO aggregation: contract communities into super-vertices.
//
// Takes an upper-triangular COO (the same format consumed by
// build_csr_from_upper_coo) plus a per-vertex label array. Outputs a new
// upper-triangular COO over super-vertices, suitable for re-feeding into
// launch_batched_leiden as the next level. Self-loops within a community
// become super-self-loops.
//
// Caller (Python) is responsible for:
//   - densifying labels per graph so each graph's communities form a
//     contiguous block of super-IDs in [g * new_seq_len, ...). This keeps
//     the (G, new_seq_len) batched layout invariant.
//   - composing the chain of densified labels back through levels to get
//     the original-vertex -> final-community mapping.

#include "internal.cuh"

#include <cub/cub.cuh>

namespace community_kv::native_leiden {

namespace {

__global__ void k_remap_edges(
    const int32_t* __restrict__ src_in,
    const int32_t* __restrict__ dst_in,
    const float*   __restrict__ w_in,
    const int32_t* __restrict__ labels,
    int64_t E_in,
    int32_t V_super_total,
    int64_t* __restrict__ keys_out,    // canonical_src * V_super_total + canonical_dst
    float*   __restrict__ w_out)
{
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= E_in) return;
  int32_t v = src_in[i];
  int32_t u = dst_in[i];
  int32_t s = labels[v];
  int32_t t = labels[u];
  int32_t a = s < t ? s : t;
  int32_t b = s < t ? t : s;
  keys_out[i] = (int64_t)a * (int64_t)V_super_total + (int64_t)b;
  w_out[i]    = w_in[i];
}

__global__ void k_unpack_keys(
    const int64_t* __restrict__ keys_in,
    int64_t E_out,
    int32_t V_super_total,
    int32_t* __restrict__ src_out,
    int32_t* __restrict__ dst_out)
{
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= E_out) return;
  int64_t k = keys_in[i];
  src_out[i] = (int32_t)(k / (int64_t)V_super_total);
  dst_out[i] = (int32_t)(k % (int64_t)V_super_total);
}

}  // namespace

// Public host API: aggregate an upper-triangular COO using `labels`. Returns
// the deduped output COO via the caller-provided pointers, and writes the
// output edge count into *E_out_host (host int64). The output buffers must
// be at least E_in elements long; the actual count is *E_out_host.
//
// Memory: scratch is allocated via cudaMallocAsync; output buffers must be
// supplied by the caller (typically tensors allocated by the Python side).
void aggregate_upper_coo(
    const int32_t* edge_src,
    const int32_t* edge_dst,
    const float*   edge_weight,
    int64_t        E_in,
    const int32_t* labels,
    int32_t        V_super_total,
    int32_t*       out_src,        // (cap >= E_in,) device
    int32_t*       out_dst,        // (cap >= E_in,) device
    float*         out_weight,     // (cap >= E_in,) device
    int64_t*       E_out_host,     // output edge count, written via D2H
    cudaStream_t   stream);

void aggregate_upper_coo(
    const int32_t* edge_src,
    const int32_t* edge_dst,
    const float*   edge_weight,
    int64_t        E_in,
    const int32_t* labels,
    int32_t        V_super_total,
    int32_t*       out_src,
    int32_t*       out_dst,
    float*         out_weight,
    int64_t*       E_out_host,
    cudaStream_t   stream)
{
  if (E_in == 0) {
    *E_out_host = 0;
    return;
  }

  int64_t* keys_in = nullptr;
  int64_t* keys_sorted = nullptr;
  int64_t* keys_unique = nullptr;
  float*   w_in_remap = nullptr;
  float*   w_sorted = nullptr;
  int64_t* d_E_out = nullptr;
  CUDA_CHECK(cudaMallocAsync(&keys_in,     E_in * sizeof(int64_t), stream));
  CUDA_CHECK(cudaMallocAsync(&keys_sorted, E_in * sizeof(int64_t), stream));
  CUDA_CHECK(cudaMallocAsync(&keys_unique, E_in * sizeof(int64_t), stream));
  CUDA_CHECK(cudaMallocAsync(&w_in_remap,  E_in * sizeof(float),   stream));
  CUDA_CHECK(cudaMallocAsync(&w_sorted,    E_in * sizeof(float),   stream));
  CUDA_CHECK(cudaMallocAsync(&d_E_out,     sizeof(int64_t),        stream));

  // 1. Remap each edge to a canonical (super_src, super_dst) packed key.
  {
    int block = 256;
    int64_t grid = (E_in + block - 1) / block;
    k_remap_edges<<<grid, block, 0, stream>>>(
        edge_src, edge_dst, edge_weight, labels,
        E_in, V_super_total, keys_in, w_in_remap);
  }

  // 2. Sort (key, weight) pairs by key.
  size_t temp_bytes_sort = 0;
  cub::DeviceRadixSort::SortPairs(
      nullptr, temp_bytes_sort,
      keys_in, keys_sorted,
      w_in_remap, w_sorted,
      E_in, 0, sizeof(int64_t) * 8, stream);
  void* temp_sort = nullptr;
  if (temp_bytes_sort > 0) {
    CUDA_CHECK(cudaMallocAsync(&temp_sort, temp_bytes_sort, stream));
  }
  cub::DeviceRadixSort::SortPairs(
      temp_sort, temp_bytes_sort,
      keys_in, keys_sorted,
      w_in_remap, w_sorted,
      E_in, 0, sizeof(int64_t) * 8, stream);

  // 3. Reduce by key: sum weights for matching keys.
  size_t temp_bytes_red = 0;
  cub::DeviceReduce::ReduceByKey(
      nullptr, temp_bytes_red,
      keys_sorted, keys_unique,
      w_sorted, out_weight,
      d_E_out,
      cub::Sum(),
      E_in, stream);
  void* temp_red = nullptr;
  if (temp_bytes_red > 0) {
    CUDA_CHECK(cudaMallocAsync(&temp_red, temp_bytes_red, stream));
  }
  cub::DeviceReduce::ReduceByKey(
      temp_red, temp_bytes_red,
      keys_sorted, keys_unique,
      w_sorted, out_weight,
      d_E_out,
      cub::Sum(),
      E_in, stream);

  // 4. Read back E_out, then unpack keys into (src, dst).
  int64_t E_out = 0;
  CUDA_CHECK(cudaMemcpyAsync(&E_out, d_E_out, sizeof(int64_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  *E_out_host = E_out;

  if (E_out > 0) {
    int block = 256;
    int64_t grid = (E_out + block - 1) / block;
    k_unpack_keys<<<grid, block, 0, stream>>>(
        keys_unique, E_out, V_super_total, out_src, out_dst);
  }

  cudaFreeAsync(keys_in, stream);
  cudaFreeAsync(keys_sorted, stream);
  cudaFreeAsync(keys_unique, stream);
  cudaFreeAsync(w_in_remap, stream);
  cudaFreeAsync(w_sorted, stream);
  cudaFreeAsync(d_E_out, stream);
  if (temp_sort) cudaFreeAsync(temp_sort, stream);
  if (temp_red)  cudaFreeAsync(temp_red, stream);
}

}  // namespace community_kv::native_leiden
