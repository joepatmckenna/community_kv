#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace community_kv::native_leiden {

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(_err));                                       \
      throw std::runtime_error(cudaGetErrorString(_err));                      \
    }                                                                          \
  } while (0)

// Symmetric CSR for a batch of G vertex-disjoint subgraphs.
// Vertices are global ([0, V_total)); subgraph membership is recovered as
// graph_id = v / seq_len when needed. We do NOT renumber to per-graph
// local IDs — keeps kernels branch-free and avoids a remap pass.
struct BatchedCSR {
  int32_t  V_total = 0;       // total vertices across all subgraphs
  int32_t  G       = 0;
  int32_t  seq_len = 0;       // vertices per subgraph
  int64_t  E_sym   = 0;       // number of stored (mirrored) edges

  // CSR
  int64_t* row_ptr  = nullptr;   // (V_total + 1,) int64
  int32_t* col_idx  = nullptr;   // (E_sym,)       int32
  float*   col_w    = nullptr;   // (E_sym,)       float32

  // Per-vertex weighted degree, fp64 to avoid drift.
  // Self-loops contribute their weight ONCE to k_v (consistent with
  // the convention used in modularity: self-loops add w once to the
  // 2m total since we mirror non-self-loop edges).
  double*  k_v     = nullptr;    // (V_total,) float64

  // Per-vertex self-loop weight (0 if none). Tracked separately so
  // aggregation can preserve super-self-loops correctly.
  float*   selfloop_w = nullptr; // (V_total,) float32

  // Per-graph 2m (sum of all edge weights inside that subgraph counted
  // with multiplicity). Computed once from k_v at CSR-build time and
  // updated on aggregation.
  double*  two_m_per_graph = nullptr;  // (G,) float64
};

// Free all device allocations owned by `csr`. Safe on a default-constructed
// BatchedCSR. Frees on the supplied stream via cudaFreeAsync — required
// because the buffers were allocated via cudaMallocAsync, and mixing with
// a synchronous cudaFree corrupts the CUDA mempool across calls.
void free_csr(BatchedCSR& csr, cudaStream_t stream);

// Build a symmetric CSR from an upper-triangular COO. Mirrors non-self-loop
// edges (so each non-self edge contributes both (u, v) and (v, u)), keeps
// self-loops once, sorts by row, and computes per-vertex weighted degree
// + per-graph 2m.
BatchedCSR build_csr_from_upper_coo(
    const int32_t* edge_src,
    const int32_t* edge_dst,
    const float*   edge_weight,
    int64_t        E_in,
    int32_t        G,
    int32_t        seq_len,
    cudaStream_t   stream);

// Compute current modularity per graph from the CSR + a labels array.
// labels are global community IDs (per-graph offsetting handled by the
// caller). Writes into pre-allocated `modularity_per_graph` (G,) float64.
void compute_modularity(
    const BatchedCSR& csr,
    const int32_t*    labels,
    float             resolution,
    double*           modularity_per_graph,
    cudaStream_t      stream);

// Recompute Σ_tot[c] = sum of k_v for v with labels[v] == c.
// `sigma_tot` must be pre-allocated (V_total,) and is zeroed inside.
void compute_sigma_tot(
    const BatchedCSR& csr,
    const int32_t*    labels,
    double*           sigma_tot,
    cudaStream_t      stream);

// Local-moving sweep. Reads `labels_in` + `sigma_tot`, writes proposed
// labels into `labels_out`, and increments `moved_counter` per change.
// `up_down` gates move direction (1 = only to higher community id, 0 = only
// to lower) so two adjacent vertices can't swap communities in one pass —
// the caller alternates it across sweeps.
void launch_local_moving(
    const BatchedCSR& csr,
    const double*    sigma_tot,
    const int32_t*   labels_in,
    int32_t*         labels_out,
    int32_t*         moved_counter,
    float            resolution,
    uint32_t         level,
    uint32_t         sweep,
    uint32_t         seed,
    int              up_down,
    cudaStream_t     stream);

// Refinement sweep split into propose + MIS-commit.
//
// `launch_refine_propose` runs Boltzmann-sampled local-moving within parent
// communities and writes, per vertex, a proposed target sub-community label
// into `prop` (== labels_in[v] if the vertex stays) plus a per-(vertex, sweep)
// random priority into `rank`. No move is applied. Candidate scoring is
// Gumbel-max Boltzmann over ΔQ ≥ 0 within-parent targets (P(c) ∝ exp(ΔQ_c/θ)),
// gated by the well-connectedness inequality on {v}:
//     E(v, S\{v}) ≥ γ · k_v · (vol(S) − k_v) / (2m)
// with vol(S) provided via `parent_sigma_tot[v]`.
//
// `launch_refine_commit` then applies an INDEPENDENT SET of those proposals:
// vertex v moves to prop[v] only if it strictly outranks (by (rank, id)) every
// within-parent neighbour that is also a mover. No two adjacent vertices move
// in the same sweep, so the synchronous-swap oscillation can't occur. The
// caller initializes `labels_in` to identity (singletons within parent
// communities) before the first sweep and loops until `moved_counter` is 0.
//
// `sigma_tot`: degree sum per CURRENT sub-community label (recomputed each
//   sweep against labels_in).
// `parent_sigma_tot`: degree sum per PARENT label (computed once, constant).
void launch_refine_propose(
    const BatchedCSR& csr,
    const int32_t*    parent_labels,
    const double*     sigma_tot,
    const double*     parent_sigma_tot,
    const int32_t*    labels_in,
    int32_t*          prop,
    uint32_t*         rank,
    float             resolution,
    float             theta,
    int               use_boltzmann,
    uint32_t          level,
    uint32_t          sweep,
    uint32_t          seed,
    cudaStream_t      stream);

void launch_refine_commit(
    const BatchedCSR& csr,
    const int32_t*    parent_labels,
    const int32_t*    labels_in,
    const int32_t*    prop,
    const uint32_t*   rank,
    int32_t*          labels_out,
    int32_t*          moved_counter,
    cudaStream_t      stream);

// Aggregate the CSR by contracting communities: each unique label becomes
// a super-vertex. Returns a new BatchedCSR. The `super_label_of_vertex`
// output (V_total,) records which super-vertex each input vertex maps to,
// with global super-IDs offset per-graph (super_id = graph_id * seq_len
// of next level + within-graph index). The next level's `seq_len` is
// returned in `out_seq_len`.
//
// `labels` MUST be pre-cleaned: every distinct value in graph g is assigned
// a unique super-id in [g*new_seq_len, g*new_seq_len + n_g) where n_g is
// the number of communities in graph g. Caller (host orchestrator) does
// this densification after each level so super-IDs stay compact.
BatchedCSR aggregate_communities(
    const BatchedCSR& csr,
    const int32_t*    labels,        // (V_total,) — global super-IDs
    int32_t           new_seq_len,   // == max_n_g over graphs (uniform shape)
    cudaStream_t      stream);

}  // namespace community_kv::native_leiden
