#pragma once

#include <cstdint>

namespace community_kv::native_leiden {

// PCG-style hash. Cheap, decent quality for sampling/jitter, fully
// reproducible from (graph_id, vertex_id, level, sweep, seed).
__device__ __host__ __forceinline__
uint32_t pcg_hash(uint32_t x) {
  uint32_t state = x * 747796405u + 2891336453u;
  uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

__device__ __host__ __forceinline__
uint32_t mix4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  uint32_t h = pcg_hash(a);
  h = pcg_hash(h ^ b);
  h = pcg_hash(h ^ c);
  h = pcg_hash(h ^ d);
  return h;
}

// Returns a uniform float in [0, 1).
__device__ __host__ __forceinline__
float rand_uniform(uint32_t graph_id, uint32_t vertex_id,
                   uint32_t level, uint32_t sweep, uint32_t seed) {
  uint32_t h = mix4(graph_id, vertex_id, (level << 16) | sweep, seed);
  // 24 bits of mantissa, divided by 2^24.
  return (h >> 8) * (1.0f / 16777216.0f);
}

// Per-(vertex, candidate) uniform — used for Gumbel-max Boltzmann sampling
// in refinement, where each (v, c) pair needs an independent Gumbel draw.
// We fold candidate_id into the sweep slot via a mix step so the random
// stream remains reproducible for fixed (graph, v, c, level, sweep, seed).
__device__ __host__ __forceinline__
float rand_uniform_pair(uint32_t graph_id, uint32_t vertex_id,
                        uint32_t candidate_id, uint32_t level,
                        uint32_t sweep, uint32_t seed) {
  uint32_t s = pcg_hash((sweep << 16) ^ candidate_id);
  uint32_t h = mix4(graph_id, vertex_id, (level << 16) | (s & 0xFFFFu), seed);
  return (h >> 8) * (1.0f / 16777216.0f);
}

}  // namespace community_kv::native_leiden
