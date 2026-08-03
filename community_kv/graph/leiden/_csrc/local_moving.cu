// Synchronous Louvain local-moving over a batched CSR.
//
// Strategy:
//   - Every vertex proposes its best move in parallel using a snapshot of
//     community totals (Sigma_tot per community).
//   - Proposed labels are written to a separate buffer; after the kernel
//     finishes the host recomputes Sigma_tot from scratch (cheap, O(V)).
//   - A round counts "moved" vertices via an atomic on a single int.
//     Convergence = no moves in a sweep.
//
// ΔQ formula (Newman with resolution γ):
//   ΔQ = (k_v_to_newC_excl - k_v_to_oldC_excl) / m
//      + γ * k_v / (2 m^2) * (Σ_tot_oldC_excl - Σ_tot_newC)
//
// where k_v_to_C_excl is the weight of edges from v to vertices in C,
// EXCLUDING any self-loop on v. The CSR stores mirrored weights and a
// doubled self-loop convention; we explicitly skip the self-loop when
// scanning v's row, so k_v_to_oldC_excl is correct.
//
// Per-graph m: m_g = (Σ_v in graph g of k_v) / 2 = (two_m_per_graph[g]) / 2.
//
// Block-per-vertex, shared-memory open-addressed hash keyed on community
// id. We size the hash to HASH_CAP (compile-time) and require at most
// HASH_CAP/2 distinct neighbor communities per vertex. For the sparse
// top-k graphs this runs on (a handful of neighbor communities per vertex)
// this is comfortably bounded.

#include "internal.cuh"
#include "rng.cuh"

namespace community_kv::native_leiden {

namespace {

constexpr int kHashCap = 1024;  // power of 2; supports degree up to ~512

struct HashEntry {
  int32_t comm;   // -1 = empty
  float   wsum;   // accumulated A_vu weight for u in this community (excl self)
};

__device__ __forceinline__
int32_t hash_probe_or_insert(HashEntry* table, int32_t comm, float w_inc) {
  // FNV-ish mix of comm.
  uint32_t h = pcg_hash((uint32_t)comm) & (kHashCap - 1);
  for (int i = 0; i < kHashCap; ++i) {
    int32_t cur = table[h].comm;
    if (cur == comm) {
      atomicAdd(&table[h].wsum, w_inc);
      return h;
    }
    if (cur == -1) {
      int32_t prev = atomicCAS(&table[h].comm, -1, comm);
      if (prev == -1 || prev == comm) {
        atomicAdd(&table[h].wsum, w_inc);
        return h;
      }
      // lost race to another thread — re-check or probe forward
      if (table[h].comm == comm) {
        atomicAdd(&table[h].wsum, w_inc);
        return h;
      }
    }
    h = (h + 1) & (kHashCap - 1);
  }
  return -1;  // overflow — caller treats as "no candidate" for this entry
}

__global__ void k_local_moving(
    // CSR
    const int64_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_idx,
    const float*   __restrict__ col_w,
    const double*  __restrict__ k_v_arr,
    const float*   __restrict__ selfloop_w,
    // Per-graph m (double m_g)
    const double*  __restrict__ two_m_per_graph,
    int32_t        seq_len,
    int32_t        V_total,
    // Per-community totals (size V_total — we use vertex-id space as
    // community-id space; unused communities have Σ_tot = 0).
    const double*  __restrict__ sigma_tot,
    // Current labels, in/out
    const int32_t* __restrict__ labels_in,
    int32_t*       __restrict__ labels_out,
    // Convergence counter
    int32_t*       __restrict__ moved_counter,
    // Algorithm parameters
    float          resolution,
    uint32_t       level,
    uint32_t       sweep,
    uint32_t       seed,
    // Direction gate: on "up" passes (up_down != 0) a vertex may only move to
    // a HIGHER community id; on "down" passes only to a LOWER one. Within a
    // single pass all moves go the same way, so two adjacent vertices can never
    // move into each other's community at once — this prevents the swap
    // oscillation. (Upward moves form a strict-id DAG, so no cyclic conflict is
    // possible either.)
    int            up_down)
{
  __shared__ HashEntry table[kHashCap];

  int32_t v = blockIdx.x;
  if (v >= V_total) return;

  // Reset table
  for (int i = threadIdx.x; i < kHashCap; i += blockDim.x) {
    table[i].comm = -1;
    table[i].wsum = 0.0f;
  }
  __syncthreads();

  int64_t row_a = row_ptr[v];
  int64_t row_b = row_ptr[v + 1];
  int32_t old_c = labels_in[v];
  double  k_v   = k_v_arr[v];

  if (row_b == row_a) {
    // isolated within this graph
    if (threadIdx.x == 0) labels_out[v] = old_c;
    return;
  }

  int32_t graph_id = v / seq_len;
  double  two_m   = two_m_per_graph[graph_id];
  if (two_m <= 0.0) {
    if (threadIdx.x == 0) labels_out[v] = old_c;
    return;
  }
  double  m_g     = 0.5 * two_m;
  double  inv_m   = 1.0 / m_g;
  double  inv_2m2 = 1.0 / (2.0 * m_g * m_g);

  // Populate hash with k_v_to_C_excl for each neighbor community
  for (int64_t e = row_a + threadIdx.x; e < row_b; e += blockDim.x) {
    int32_t u = col_idx[e];
    if (u == v) continue;     // skip self-loop in k_v_to_C_excl
    float   w = col_w[e];
    int32_t c = labels_in[u];
    hash_probe_or_insert(table, c, w);
  }
  __syncthreads();

  // Cooperative reduction: each thread scans a slice of the hash and
  // finds the best (largest ΔQ) candidate in its slice. Then a tree
  // reduction picks the global best.
  __shared__ float    s_best_dq[32];
  __shared__ int32_t  s_best_c[32];

  float    my_best_dq = 0.0f;        // ΔQ ≥ 0 required (Louvain default)
  int32_t  my_best_c  = old_c;

  // k_v_to_oldC_excl for the OUT term — read from hash if present
  float k_v_to_old = 0.0f;
  {
    uint32_t h = pcg_hash((uint32_t)old_c) & (kHashCap - 1);
    for (int i = 0; i < kHashCap; ++i) {
      int32_t cur = table[h].comm;
      if (cur == old_c) { k_v_to_old = table[h].wsum; break; }
      if (cur == -1) break;
      h = (h + 1) & (kHashCap - 1);
    }
  }
  double sigma_tot_old_excl = sigma_tot[old_c] - k_v;

  for (int i = threadIdx.x; i < kHashCap; i += blockDim.x) {
    int32_t c = table[i].comm;
    if (c == -1 || c == old_c) continue;
    // Direction gate: skip candidates that don't match this pass's
    // up/down direction (see up_down comment on the kernel signature).
    if (((c > old_c) ? 1 : 0) != up_down) continue;
    float k_v_to_new = table[i].wsum;
    double sigma_tot_new = sigma_tot[c];
    double dq = ((double)k_v_to_new - (double)k_v_to_old) * inv_m
              + (double)resolution * k_v
                * (sigma_tot_old_excl - sigma_tot_new) * inv_2m2;
    // Tiny jitter to break ties deterministically per (graph, vertex, sweep).
    float jitter = (rand_uniform(graph_id, (uint32_t)v, level, sweep, seed) - 0.5f) * 1e-9f;
    float dq_f = (float)dq + jitter;
    if (dq_f > my_best_dq) {
      my_best_dq = dq_f;
      my_best_c  = c;
    }
  }

  // warp-level reduction
  unsigned mask = 0xFFFFFFFFu;
  for (int off = 16; off > 0; off >>= 1) {
    float    other_dq = __shfl_down_sync(mask, my_best_dq, off);
    int32_t  other_c  = __shfl_down_sync(mask, my_best_c, off);
    if (other_dq > my_best_dq) {
      my_best_dq = other_dq;
      my_best_c  = other_c;
    }
  }
  if ((threadIdx.x & 31) == 0) {
    s_best_dq[threadIdx.x >> 5] = my_best_dq;
    s_best_c[threadIdx.x >> 5]  = my_best_c;
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    int nwarps = (blockDim.x + 31) / 32;
    float   bd = (threadIdx.x < nwarps) ? s_best_dq[threadIdx.x] : 0.0f;
    int32_t bc = (threadIdx.x < nwarps) ? s_best_c[threadIdx.x]  : old_c;
    for (int off = 16; off > 0; off >>= 1) {
      float    other_dq = __shfl_down_sync(mask, bd, off);
      int32_t  other_c  = __shfl_down_sync(mask, bc, off);
      if (other_dq > bd) { bd = other_dq; bc = other_c; }
    }
    if (threadIdx.x == 0) {
      labels_out[v] = bc;
      if (bc != old_c) atomicAdd(moved_counter, 1);
    }
  }
}

}  // namespace

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
    cudaStream_t     stream)
{
  if (csr.V_total == 0) return;
  int block = 128;
  k_local_moving<<<csr.V_total, block, 0, stream>>>(
      csr.row_ptr, csr.col_idx, csr.col_w, csr.k_v, csr.selfloop_w,
      csr.two_m_per_graph, csr.seq_len, csr.V_total,
      sigma_tot, labels_in, labels_out, moved_counter,
      resolution, level, sweep, seed, up_down);
}

}  // namespace community_kv::native_leiden
