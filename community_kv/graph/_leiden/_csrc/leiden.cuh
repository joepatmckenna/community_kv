// Public host API for batched Leiden over G vertex-disjoint subgraphs.
//
// All inputs live on the same CUDA device; the caller is responsible for
// setting cudaSetDevice() before invoking. No host syncs happen inside
// launch_batched_leiden except a single device->host copy of the
// modularity vector at the end.
#pragma once

#include <cstdint>

namespace community_kv::native_leiden {

struct LeidenParams {
  int   max_level   = 6;
  float resolution  = 1.0f;
  float theta       = 0.01f;     // refinement temperature (Boltzmann only)
  int   max_inner_iter = 8;       // local-moving sweeps per level
  int   seed        = 0;
  // Refinement target selection: false = greedy best-gain (default);
  // true = Boltzmann/Gumbel sampling at temperature `theta`.
  bool  use_boltzmann_refine = false;
};

// Inputs are upper-triangular COO (src <= dst) with vertex IDs
// already pre-offset by g * seq_len (cf. build_adjacency_batched).
// Self-loops are allowed.
//
//   edge_src, edge_dst:    (E,) int32, device
//   edge_weight:           (E,) float32, device
//   E:                     number of edges
//   G:                     number of subgraphs
//   seq_len:               vertices per subgraph (so V_total = G * seq_len)
//
// Outputs (caller-allocated, device):
//   labels:                (V_total,) int32; vertices that never appeared
//                          in any edge get -1 (the Python wrapper fills
//                          isolated singletons).
//   modularity_per_graph:  (G,) float64
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
    void*          stream);

}  // namespace community_kv::native_leiden
