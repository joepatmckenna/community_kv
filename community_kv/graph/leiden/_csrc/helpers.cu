// Helper kernels: Sigma_tot recompute, modularity computation.

#include "internal.cuh"

namespace community_kv::native_leiden {

namespace {

__global__ void k_zero_d(double* p, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) p[i] = 0.0;
}

__global__ void k_sigma_tot_scatter(
    const int32_t* __restrict__ labels,
    const double*  __restrict__ k_v,
    int32_t V_total,
    double* __restrict__ sigma_tot)
{
  int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= V_total) return;
  int32_t c = labels[v];
  if (c < 0) return;
  atomicAdd(&sigma_tot[c], k_v[v]);
}

// Computes per-vertex contribution to graph-level modularity numerator,
// then atomically adds into modularity_per_graph[g]. The formula:
//   Σ_in[C] -- handled by summing CSR weights where both endpoints share label
//   Σ_tot^2 sum -- handled by summing k_v * Σ_tot[c_v] across vertices
//
// We compute, per vertex v in graph g:
//   in_contrib = sum over CSR row of w if labels[u] == labels[v]
//   tot_contrib = k_v * sigma_tot[labels[v]]
//
// Then Q_g = (sum_v in g in_contrib) / (2 m_g)
//          - γ * (sum_v in g tot_contrib) / (2 m_g)^2
__global__ void k_modularity_per_vertex(
    const int64_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_idx,
    const float*   __restrict__ col_w,
    const double*  __restrict__ k_v,
    const int32_t* __restrict__ labels,
    const double*  __restrict__ sigma_tot,
    const double*  __restrict__ two_m_per_graph,
    int32_t V_total,
    int32_t seq_len,
    float resolution,
    double* __restrict__ modularity_per_graph)
{
  int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= V_total) return;
  int32_t g = v / seq_len;
  double two_m = two_m_per_graph[g];
  if (two_m <= 0.0) return;

  int32_t cv = labels[v];
  int64_t a = row_ptr[v];
  int64_t b = row_ptr[v + 1];

  double in_sum = 0.0;
  for (int64_t e = a; e < b; ++e) {
    if (labels[col_idx[e]] == cv) {
      in_sum += (double)col_w[e];
    }
  }
  double tot_sum = k_v[v] * sigma_tot[cv];

  double m_g = 0.5 * two_m;
  double q_contrib = in_sum / two_m
                   - (double)resolution * tot_sum / (4.0 * m_g * m_g);

  atomicAdd(&modularity_per_graph[g], q_contrib);
}

}  // namespace

void compute_sigma_tot(
    const BatchedCSR& csr,
    const int32_t*    labels,
    double*           sigma_tot,
    cudaStream_t      stream)
{
  if (csr.V_total == 0) return;
  {
    int block = 256;
    int grid = (csr.V_total + block - 1) / block;
    k_zero_d<<<grid, block, 0, stream>>>(sigma_tot, csr.V_total);
  }
  {
    int block = 256;
    int grid = (csr.V_total + block - 1) / block;
    k_sigma_tot_scatter<<<grid, block, 0, stream>>>(
        labels, csr.k_v, csr.V_total, sigma_tot);
  }
}

void compute_modularity(
    const BatchedCSR& csr,
    const int32_t*    labels,
    float             resolution,
    double*           modularity_per_graph,
    cudaStream_t      stream)
{
  if (csr.V_total == 0) return;
  // Need a temporary sigma_tot.
  double* sigma_tot = nullptr;
  CUDA_CHECK(cudaMallocAsync(&sigma_tot, csr.V_total * sizeof(double), stream));
  compute_sigma_tot(csr, labels, sigma_tot, stream);
  // Zero output.
  CUDA_CHECK(cudaMemsetAsync(modularity_per_graph, 0,
                             csr.G * sizeof(double), stream));
  int block = 128;
  int grid = (csr.V_total + block - 1) / block;
  k_modularity_per_vertex<<<grid, block, 0, stream>>>(
      csr.row_ptr, csr.col_idx, csr.col_w, csr.k_v, labels,
      sigma_tot, csr.two_m_per_graph, csr.V_total, csr.seq_len,
      resolution, modularity_per_graph);
  cudaFreeAsync(sigma_tot, stream);
}

}  // namespace community_kv::native_leiden
