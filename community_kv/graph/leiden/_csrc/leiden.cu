// Host orchestration for batched Leiden over G vertex-disjoint subgraphs.
//
// Each call runs, on the device:
//   1. Louvain local-moving with an up/down directional gate, sweeping until
//      neither direction can improve modularity.
//   2. Refinement: re-initialize to singletons within each local-moving
//      community, then independent-set local-moving (propose + commit) inside
//      those parent communities.
//   3. Final per-graph modularity.
//
// Initial labels are identity (each vertex its own community). Returns labels
// (V_total,) where labels[v] is the global community id of v, with -1 for
// vertices that have no edges (isolated). The Python wrapper handles per-graph
// dense renumbering, isolated-singleton fill, and multi-level aggregation.

#include "leiden.cuh"
#include "internal.cuh"

#include <stdexcept>
#include <vector>

namespace community_kv::native_leiden {

namespace {

__global__ void k_init_identity_labels(int32_t V_total, int32_t* labels) {
  int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v < V_total) labels[v] = v;
}

__global__ void k_mark_isolated(
    const int64_t* __restrict__ row_ptr,
    int32_t V_total,
    int32_t* __restrict__ labels)
{
  int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= V_total) return;
  if (row_ptr[v] == row_ptr[v + 1]) labels[v] = -1;
}

}  // namespace

void launch_batched_leiden(
    const int32_t* edge_src,
    const int32_t* edge_dst,
    const float*   edge_weight,
    int64_t        E,
    int32_t        G,
    int32_t        seq_len,
    int32_t*       labels,
    double*        modularity_per_graph,
    const LeidenParams& params,
    void*          stream_v)
{
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_v);
  if (G <= 0 || seq_len <= 0) {
    if (G > 0) {
      CUDA_CHECK(cudaMemsetAsync(modularity_per_graph, 0,
                                 G * sizeof(double), stream));
    }
    return;
  }
  int32_t V_total = G * seq_len;

  // Empty-edge fast path: no Leiden needed; mark every vertex isolated.
  if (E == 0) {
    // Memset to 0xFF -> int32 = -1 across the buffer.
    CUDA_CHECK(cudaMemsetAsync(labels, 0xFF, V_total * sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(modularity_per_graph, 0,
                               G * sizeof(double), stream));
    return;
  }

  BatchedCSR csr = build_csr_from_upper_coo(
      edge_src, edge_dst, edge_weight, E, G, seq_len, stream);

  // Initial labels: identity. Vertices with no edges get -1 at the end.
  {
    int block = 256;
    int grid = (V_total + block - 1) / block;
    k_init_identity_labels<<<grid, block, 0, stream>>>(V_total, labels);
  }

  // Working buffers.
  int32_t* labels_other = nullptr;
  CUDA_CHECK(cudaMallocAsync(&labels_other, V_total * sizeof(int32_t), stream));
  int32_t* parent_labels = nullptr;
  CUDA_CHECK(cudaMallocAsync(&parent_labels, V_total * sizeof(int32_t), stream));
  double*  sigma_tot = nullptr;
  CUDA_CHECK(cudaMallocAsync(&sigma_tot, V_total * sizeof(double), stream));
  int32_t* moved_counter = nullptr;
  CUDA_CHECK(cudaMallocAsync(&moved_counter, sizeof(int32_t), stream));
  // parent_sigma_tot: vol(S) for each PARENT label, computed once after LM
  // and held constant across refinement sweeps (parent_labels never changes
  // during refinement). Used by the well-connectedness check on {v}.
  double*  parent_sigma_tot = nullptr;
  CUDA_CHECK(cudaMallocAsync(&parent_sigma_tot, V_total * sizeof(double), stream));
  // Refinement MIS scratch: per-vertex proposed target (refine_prop) and a
  // per-(vertex, sweep) random priority (refine_rank) used by the commit pass
  // to select an independent set of moves.
  int32_t* refine_prop = nullptr;
  CUDA_CHECK(cudaMallocAsync(&refine_prop, V_total * sizeof(int32_t), stream));
  uint32_t* refine_rank = nullptr;
  CUDA_CHECK(cudaMallocAsync(&refine_rank, V_total * sizeof(uint32_t), stream));

  int32_t* cur = labels;
  int32_t* nxt = labels_other;
  uint32_t level = 0;

  int up_down = 1;
  int stale = 0;
  for (int sweep = 0; sweep < params.max_inner_iter; ++sweep) {
    compute_sigma_tot(csr, cur, sigma_tot, stream);
    CUDA_CHECK(cudaMemsetAsync(moved_counter, 0, sizeof(int32_t), stream));
    launch_local_moving(csr, sigma_tot, cur, nxt, moved_counter,
                        params.resolution, level, (uint32_t)sweep,
                        (uint32_t)params.seed, up_down, stream);

    int32_t moved_host = 0;
    CUDA_CHECK(cudaMemcpyAsync(&moved_host, moved_counter, sizeof(int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::swap(cur, nxt);
    // Flip move direction when a pass stalls; converged once neither
    // direction can move. The single-direction gate stops two adjacent
    // vertices from swapping communities on the same (stale) snapshot.
    if (moved_host == 0) {
      up_down ^= 1;
      if (++stale >= 2) break;
    } else {
      stale = 0;
    }
  }

  // Refinement: local-moving within each local-moving community, with a
  // well-connectedness check per vertex. `cur` currently holds the
  // post-local-moving partition P; we keep it as `parent_labels`, then
  // re-initialize labels_in to identity (singletons) and sweep to convergence.
  CUDA_CHECK(cudaMemcpyAsync(parent_labels, cur, V_total * sizeof(int32_t),
                             cudaMemcpyDeviceToDevice, stream));
  // Compute vol(S) for each parent community ONCE — refinement doesn't
  // touch parent_labels, so this is a constant across sweeps. The kernel
  // reads parent_sigma_tot[parent_v] directly.
  compute_sigma_tot(csr, parent_labels, parent_sigma_tot, stream);
  // Re-init: identity in `labels` (caller buffer), use `labels_other` as nxt.
  cur = labels;
  nxt = labels_other;
  {
    int block = 256;
    int grid = (V_total + block - 1) / block;
    k_init_identity_labels<<<grid, block, 0, stream>>>(V_total, cur);
  }

  for (int sweep = 0; sweep < params.max_inner_iter; ++sweep) {
    compute_sigma_tot(csr, cur, sigma_tot, stream);
    CUDA_CHECK(cudaMemsetAsync(moved_counter, 0, sizeof(int32_t), stream));
    // Propose targets (Boltzmann + well-connectedness), then commit only an
    // independent set of them (MIS) so adjacent vertices never move at once.
    launch_refine_propose(csr, parent_labels, sigma_tot, parent_sigma_tot,
                          cur, refine_prop, refine_rank,
                          params.resolution, params.theta,
                          params.use_boltzmann_refine ? 1 : 0,
                          level + 1, (uint32_t)sweep,
                          (uint32_t)params.seed, stream);
    launch_refine_commit(csr, parent_labels, cur, refine_prop, refine_rank,
                         nxt, moved_counter, stream);

    int32_t moved_host = 0;
    CUDA_CHECK(cudaMemcpyAsync(&moved_host, moved_counter, sizeof(int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::swap(cur, nxt);
    if (moved_host == 0) break;
  }

  // Make sure `labels` (the caller-owned buffer) holds the final result.
  if (cur != labels) {
    CUDA_CHECK(cudaMemcpyAsync(labels, cur, V_total * sizeof(int32_t),
                               cudaMemcpyDeviceToDevice, stream));
  }

  // Compute final modularity per graph (on the still-mirrored CSR).
  compute_modularity(csr, labels, params.resolution,
                     modularity_per_graph, stream);

  // Mark isolated vertices as -1 (Python wrapper assigns singletons).
  {
    int block = 256;
    int grid = (V_total + block - 1) / block;
    k_mark_isolated<<<grid, block, 0, stream>>>(csr.row_ptr, V_total, labels);
  }

  cudaFreeAsync(labels_other, stream);
  cudaFreeAsync(parent_labels, stream);
  cudaFreeAsync(sigma_tot, stream);
  cudaFreeAsync(parent_sigma_tot, stream);
  cudaFreeAsync(refine_prop, stream);
  cudaFreeAsync(refine_rank, stream);
  cudaFreeAsync(moved_counter, stream);
  free_csr(csr, stream);
}

}  // namespace community_kv::native_leiden
