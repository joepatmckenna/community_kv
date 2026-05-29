// Refinement: local-moving within parent communities (Traag, Waltman &
// van Eck 2019), with maximal-independent-set (MIS) conflict resolution.
//
// Split into two kernels per sweep:
//   1. k_refine_propose — each vertex computes a proposed target sub-community
//      and a per-(vertex, sweep) random rank. The target is the greedy
//      maximum-ΔQ candidate, or — when use_boltzmann is set — a Gumbel-max
//      Boltzmann sample over the ΔQ ≥ 0 candidates at temperature theta.
//      Candidates are restricted to the vertex's parent community, and a vertex
//      only moves if it satisfies the well-connectedness condition on {v}.
//      Output: prop[v] (== labels_in[v] if v stays) and rank[v].
//   2. k_refine_commit — a vertex applies its move only if it strictly outranks
//      every within-parent neighbour that also wants to move ((rank, id)
//      lexicographic). This commits an INDEPENDENT SET of moves: no two adjacent
//      vertices move in the same sweep, so simultaneous moves never conflict on
//      a stale snapshot. Maximality isn't required per sweep — the outer loop
//      iterates and the globally top-ranked mover always commits, so progress
//      is guaranteed.
//
// Local moving (the Louvain phase) instead resolves conflicts with an up/down
// directional gate; the independent-set machinery lives here in refinement.

#include "internal.cuh"
#include "rng.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

namespace community_kv::native_leiden {

namespace {

constexpr int kHashCapR = 1024;

struct HashEntryR { int32_t comm; float wsum; };

__device__ __forceinline__
void hash_insert_r(HashEntryR* table, int32_t comm, float w_inc) {
  uint32_t h = pcg_hash((uint32_t)comm) & (kHashCapR - 1);
  for (int i = 0; i < kHashCapR; ++i) {
    int32_t cur = table[h].comm;
    if (cur == comm) {
      atomicAdd(&table[h].wsum, w_inc); return;
    }
    if (cur == -1) {
      int32_t prev = atomicCAS(&table[h].comm, -1, comm);
      if (prev == -1 || prev == comm) {
        atomicAdd(&table[h].wsum, w_inc); return;
      }
      if (table[h].comm == comm) {
        atomicAdd(&table[h].wsum, w_inc); return;
      }
    }
    h = (h + 1) & (kHashCapR - 1);
  }
}

// Gumbel(0, 1) noise from a uniform sample. Clamp away from 0 to avoid
// log(0) = -inf which would produce NaNs after the second log.
__device__ __forceinline__
float gumbel_from_uniform(float u) {
  u = fmaxf(u, 1.0e-7f);
  u = fminf(u, 1.0f - 1.0e-7f);
  return -logf(-logf(u));
}

// First half of a refinement sweep: each vertex PROPOSES a target (no move is
// applied yet). Writes prop[v] (the chosen sub-community label; == labels_in[v]
// if the vertex stays) and rank[v] (a per-(vertex, sweep) random priority used
// by the MIS commit). Conflict resolution happens in k_refine_commit.
__global__ void k_refine_propose(
    const int64_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_idx,
    const float*   __restrict__ col_w,
    const double*  __restrict__ k_v_arr,
    const double*  __restrict__ two_m_per_graph,
    int32_t        seq_len,
    int32_t        V_total,
    const int32_t* __restrict__ parent_labels,
    const double*  __restrict__ sigma_tot,
    const double*  __restrict__ parent_sigma_tot,
    const int32_t* __restrict__ labels_in,
    int32_t*       __restrict__ prop,
    uint32_t*      __restrict__ rank,
    float          resolution,
    float          inv_theta,
    int            use_boltzmann,
    uint32_t       level,
    uint32_t       sweep,
    uint32_t       seed)
{
  __shared__ HashEntryR table[kHashCapR];
  __shared__ float s_e_v_to_S;

  int32_t v = blockIdx.x;
  if (v >= V_total) return;

  for (int i = threadIdx.x; i < kHashCapR; i += blockDim.x) {
    table[i].comm = -1;
    table[i].wsum = 0.0f;
  }
  if (threadIdx.x == 0) s_e_v_to_S = 0.0f;
  __syncthreads();

  int32_t graph_id = v / seq_len;
  // Per-(vertex, sweep) random rank for the MIS commit. Deterministic given
  // seed (reproducibility preserved); varies per sweep so the same vertices
  // don't always win — avoids starvation and yields better independent sets.
  if (threadIdx.x == 0) {
    rank[v] = mix4((uint32_t)graph_id, (uint32_t)v, (level << 16) | sweep, seed);
  }

  int64_t row_a = row_ptr[v];
  int64_t row_b = row_ptr[v + 1];
  int32_t old_c = labels_in[v];
  double  k_v   = k_v_arr[v];
  int32_t parent_v = parent_labels[v];

  if (row_b == row_a) {
    if (threadIdx.x == 0) prop[v] = old_c;
    return;
  }

  double  two_m   = two_m_per_graph[graph_id];
  if (two_m <= 0.0) {
    if (threadIdx.x == 0) prop[v] = old_c;
    return;
  }
  double  m_g     = 0.5 * two_m;
  double  inv_m   = 1.0 / m_g;
  double  inv_2m2 = 1.0 / (2.0 * m_g * m_g);

  // Populate hash with k_v_to_C_excl, restricted to within-parent neighbors.
  float my_e_v_to_S = 0.0f;
  for (int64_t e = row_a + threadIdx.x; e < row_b; e += blockDim.x) {
    int32_t u = col_idx[e];
    if (u == v) continue;
    if (parent_labels[u] != parent_v) continue;  // refinement constraint
    float   w = col_w[e];
    int32_t c = labels_in[u];
    hash_insert_r(table, c, w);
    my_e_v_to_S += w;
  }
  atomicAdd(&s_e_v_to_S, my_e_v_to_S);
  __syncthreads();

  // Well-connectedness on {v}: stay put if v's edges into S\{v} don't meet
  // γ · k_v · (vol(S) − k_v) / (2m).
  float e_v_to_S_total = s_e_v_to_S;
  double vol_S = parent_sigma_tot[parent_v];
  double vol_S_minus_v = vol_S - k_v;
  if (vol_S_minus_v < 0.0) vol_S_minus_v = 0.0;
  double well_conn_bound = (double)resolution * k_v * vol_S_minus_v / two_m;
  if ((double)e_v_to_S_total < well_conn_bound) {
    if (threadIdx.x == 0) prop[v] = old_c;
    return;
  }

  __shared__ float    s_best_score[32];
  __shared__ int32_t  s_best_c[32];

  // "Stay" baseline (ΔQ = 0). With Boltzmann it gets its own Gumbel sample so
  // it competes fairly; with greedy best-gain it's just score 0 (the vertex
  // moves only to a strictly-positive-gain target).
  float my_best_score;
  int32_t my_best_c;
  if (threadIdx.x == 0) {
    if (use_boltzmann) {
      float u_stay = rand_uniform_pair(graph_id, (uint32_t)v,
                                       (uint32_t)old_c, level, sweep, seed);
      my_best_score = gumbel_from_uniform(u_stay);  // ΔQ_stay/θ = 0
    } else {
      my_best_score = 0.0f;
    }
    my_best_c     = old_c;
  } else {
    my_best_score = -CUDART_INF_F;
    my_best_c     = old_c;
  }

  float k_v_to_old = 0.0f;
  {
    uint32_t h = pcg_hash((uint32_t)old_c) & (kHashCapR - 1);
    for (int i = 0; i < kHashCapR; ++i) {
      int32_t cur = table[h].comm;
      if (cur == old_c) { k_v_to_old = table[h].wsum; break; }
      if (cur == -1) break;
      h = (h + 1) & (kHashCapR - 1);
    }
  }
  double sigma_tot_old_excl = sigma_tot[old_c] - k_v;

  for (int i = threadIdx.x; i < kHashCapR; i += blockDim.x) {
    int32_t c = table[i].comm;
    if (c == -1 || c == old_c) continue;
    float k_v_to_new = table[i].wsum;
    double sigma_tot_new = sigma_tot[c];
    double dq = ((double)k_v_to_new - (double)k_v_to_old) * inv_m
              + (double)resolution * k_v
                * (sigma_tot_old_excl - sigma_tot_new) * inv_2m2;
    if (dq < 0.0) continue;  // only ΔQ ≥ 0 candidates.

    float score;
    if (use_boltzmann) {
      float u = rand_uniform_pair(graph_id, (uint32_t)v, (uint32_t)c,
                                  level, sweep, seed);
      float gumbel = gumbel_from_uniform(u);
      score = (float)(dq * (double)inv_theta) + gumbel;  // P(c) ∝ exp(ΔQ_c/θ)
    } else {
      score = (float)dq;  // greedy best-gain
    }

    if (score > my_best_score) {
      my_best_score = score;
      my_best_c     = c;
    }
  }

  unsigned mask = 0xFFFFFFFFu;
  for (int off = 16; off > 0; off >>= 1) {
    float    other_s = __shfl_down_sync(mask, my_best_score, off);
    int32_t  other_c = __shfl_down_sync(mask, my_best_c, off);
    if (other_s > my_best_score) {
      my_best_score = other_s;
      my_best_c     = other_c;
    }
  }
  if ((threadIdx.x & 31) == 0) {
    s_best_score[threadIdx.x >> 5] = my_best_score;
    s_best_c[threadIdx.x >> 5]     = my_best_c;
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    int nwarps = (blockDim.x + 31) / 32;
    float    bs = (threadIdx.x < nwarps) ? s_best_score[threadIdx.x] : -CUDART_INF_F;
    int32_t  bc = (threadIdx.x < nwarps) ? s_best_c[threadIdx.x]     : old_c;
    for (int off = 16; off > 0; off >>= 1) {
      float    other_s = __shfl_down_sync(mask, bs, off);
      int32_t  other_c = __shfl_down_sync(mask, bc, off);
      if (other_s > bs) { bs = other_s; bc = other_c; }
    }
    if (threadIdx.x == 0) {
      prop[v] = bc;  // == old_c if "stay" won
    }
  }
}

// Second half of a refinement sweep: apply an INDEPENDENT SET of the proposed
// moves. Vertex v commits prop[v] iff it is a mover AND it strictly outranks
// every within-parent neighbour that is also a mover ((rank, id) lexicographic
// tie-break). Otherwise it stays. Guarantees no two adjacent vertices move in
// the same sweep.
__global__ void k_refine_commit(
    const int64_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_idx,
    const int32_t* __restrict__ parent_labels,
    const int32_t* __restrict__ labels_in,
    const int32_t* __restrict__ prop,
    const uint32_t* __restrict__ rank,
    int32_t        V_total,
    int32_t*       __restrict__ labels_out,
    int32_t*       __restrict__ moved_counter)
{
  __shared__ int s_lost;

  int32_t v = blockIdx.x;
  if (v >= V_total) return;

  int32_t old_c = labels_in[v];
  int32_t pv    = prop[v];

  // Not a mover -> just carry the current label forward.
  if (pv == old_c) {
    if (threadIdx.x == 0) labels_out[v] = old_c;
    return;
  }

  if (threadIdx.x == 0) s_lost = 0;
  __syncthreads();

  int32_t  parent_v = parent_labels[v];
  uint32_t rank_v   = rank[v];
  int64_t  row_a    = row_ptr[v];
  int64_t  row_b    = row_ptr[v + 1];

  int my_lost = 0;
  for (int64_t e = row_a + threadIdx.x; e < row_b; e += blockDim.x) {
    int32_t u = col_idx[e];
    if (u == v) continue;
    if (parent_labels[u] != parent_v) continue;       // outside S: irrelevant
    if (prop[u] == labels_in[u]) continue;            // u isn't a mover
    uint32_t rank_u = rank[u];
    // u beats v if it outranks v (id breaks ties) -> v must yield this sweep.
    if (rank_u > rank_v || (rank_u == rank_v && u > v)) my_lost = 1;
  }
  if (my_lost) atomicOr(&s_lost, 1);
  __syncthreads();

  if (threadIdx.x == 0) {
    if (s_lost) {
      labels_out[v] = old_c;
    } else {
      labels_out[v] = pv;
      atomicAdd(moved_counter, 1);
    }
  }
}

}  // namespace

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
    cudaStream_t      stream)
{
  if (csr.V_total == 0) return;
  // Guard against degenerate θ (would divide-by-zero into inv_theta).
  float t = theta;
  if (!(t > 0.0f)) t = 1.0e-3f;
  float inv_theta = 1.0f / t;

  int block = 128;
  k_refine_propose<<<csr.V_total, block, 0, stream>>>(
      csr.row_ptr, csr.col_idx, csr.col_w, csr.k_v,
      csr.two_m_per_graph, csr.seq_len, csr.V_total,
      parent_labels, sigma_tot, parent_sigma_tot,
      labels_in, prop, rank,
      resolution, inv_theta, use_boltzmann, level, sweep, seed);
}

void launch_refine_commit(
    const BatchedCSR& csr,
    const int32_t*    parent_labels,
    const int32_t*    labels_in,
    const int32_t*    prop,
    const uint32_t*   rank,
    int32_t*          labels_out,
    int32_t*          moved_counter,
    cudaStream_t      stream)
{
  if (csr.V_total == 0) return;
  int block = 128;
  k_refine_commit<<<csr.V_total, block, 0, stream>>>(
      csr.row_ptr, csr.col_idx, parent_labels, labels_in, prop, rank,
      csr.V_total, labels_out, moved_counter);
}

}  // namespace community_kv::native_leiden
