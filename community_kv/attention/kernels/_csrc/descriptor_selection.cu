#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

constexpr int kThreads = 1024;
constexpr int kWarpSize = 32;
constexpr int kWarps = kThreads / kWarpSize;
constexpr int kHistogramBins = 256;
constexpr int kTile = 64;
constexpr int kPartFields = 8;
constexpr int kClusterParts = 8;

struct Int4 {
  int high_count;
  int high_mass;
  int equal_count;
  int equal_mass;
};

__device__ __forceinline__ uint16_t ordered_fp16_key(uint16_t bits) {
  const uint16_t flip = (bits & 0x8000u) != 0 ? 0xffffu : 0x8000u;
  return bits ^ flip;
}

__device__ __forceinline__ Int4 block_exclusive_sum(
    Int4 value,
    Int4* warp_prefix,
    Int4* block_total) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  Int4 inclusive = value;
#pragma unroll
  for (int offset = 1; offset < kWarpSize; offset <<= 1) {
    const Int4 other = {
        __shfl_up_sync(0xffffffffu, inclusive.high_count, offset),
        __shfl_up_sync(0xffffffffu, inclusive.high_mass, offset),
        __shfl_up_sync(0xffffffffu, inclusive.equal_count, offset),
        __shfl_up_sync(0xffffffffu, inclusive.equal_mass, offset)};
    if (lane >= offset) {
      inclusive.high_count += other.high_count;
      inclusive.high_mass += other.high_mass;
      inclusive.equal_count += other.equal_count;
      inclusive.equal_mass += other.equal_mass;
    }
  }
  if (lane == kWarpSize - 1) {
    warp_prefix[warp] = inclusive;
  }
  __syncthreads();

  if (warp == 0) {
    const Int4 warp_value =
        lane < kWarps ? warp_prefix[lane] : Int4{0, 0, 0, 0};
    Int4 warp_inclusive = warp_value;
#pragma unroll
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
      const Int4 other = {
          __shfl_up_sync(
              0xffffffffu, warp_inclusive.high_count, offset),
          __shfl_up_sync(
              0xffffffffu, warp_inclusive.high_mass, offset),
          __shfl_up_sync(
              0xffffffffu, warp_inclusive.equal_count, offset),
          __shfl_up_sync(
              0xffffffffu, warp_inclusive.equal_mass, offset)};
      if (lane >= offset) {
        warp_inclusive.high_count += other.high_count;
        warp_inclusive.high_mass += other.high_mass;
        warp_inclusive.equal_count += other.equal_count;
        warp_inclusive.equal_mass += other.equal_mass;
      }
    }
    if (lane < kWarps) {
      warp_prefix[lane] = {
          warp_inclusive.high_count - warp_value.high_count,
          warp_inclusive.high_mass - warp_value.high_mass,
          warp_inclusive.equal_count - warp_value.equal_count,
          warp_inclusive.equal_mass - warp_value.equal_mass};
    }
    if (lane == kWarps - 1) {
      *block_total = warp_inclusive;
    }
  }
  __syncthreads();
  const Int4 base = warp_prefix[warp];
  return {
      base.high_count + inclusive.high_count - value.high_count,
      base.high_mass + inclusive.high_mass - value.high_mass,
      base.equal_count + inclusive.equal_count - value.equal_count,
      base.equal_mass + inclusive.equal_mass - value.equal_mass};
}

__device__ __forceinline__ void block_sum(
    Int4 value,
    Int4* warp_sums,
    Int4* block_total) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value.high_count +=
        __shfl_down_sync(0xffffffffu, value.high_count, offset);
    value.high_mass +=
        __shfl_down_sync(0xffffffffu, value.high_mass, offset);
    value.equal_count +=
        __shfl_down_sync(0xffffffffu, value.equal_count, offset);
    value.equal_mass +=
        __shfl_down_sync(0xffffffffu, value.equal_mass, offset);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  if (warp == 0) {
    Int4 total = lane < kWarps ? warp_sums[lane] : Int4{0, 0, 0, 0};
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      total.high_count +=
          __shfl_down_sync(0xffffffffu, total.high_count, offset);
      total.high_mass +=
          __shfl_down_sync(0xffffffffu, total.high_mass, offset);
      total.equal_count +=
          __shfl_down_sync(0xffffffffu, total.equal_count, offset);
      total.equal_mass +=
          __shfl_down_sync(0xffffffffu, total.equal_mass, offset);
    }
    if (lane == 0) {
      *block_total = total;
    }
  }
}

__device__ __forceinline__ void emit_descriptor_tile_starts(
    int32_t* __restrict__ tile_descriptors,
    int kv_head,
    int tile_count,
    int num_sink,
    int descriptor,
    int selected_start,
    int selected_end) {
  if (selected_start == 0) {
    tile_descriptors[kv_head * tile_count] = descriptor;
  }
  const int first_tile =
      max(1, (selected_start + num_sink + kTile - 1) / kTile);
  const int last_tile =
      min(tile_count - 1, (selected_end + num_sink - 1) / kTile);
  for (int tile = first_tile; tile <= last_tile; ++tile) {
    tile_descriptors[kv_head * tile_count + tile] = descriptor;
  }
}

template <bool kLowByte>
__global__ __launch_bounds__(kThreads) void partial_histogram_kernel(
    const uint16_t* __restrict__ scores,
    const int32_t* __restrict__ community_sizes,
    const int32_t* __restrict__ community_counts,
    int32_t* __restrict__ histogram_workspace,
    const int32_t* __restrict__ threshold_state,
    int community_capacity,
    int parts) {
  const int kv_head = blockIdx.x / parts;
  const int part = blockIdx.x - kv_head * parts;
  const int thread = threadIdx.x;
  const int active_communities = max(
      0,
      min(community_capacity, community_counts[kv_head]));
  __shared__ int histogram[kHistogramBins];
  if (thread < kHistogramBins) {
    histogram[thread] = 0;
  }
  __syncthreads();

  const int high_bin =
      kLowByte ? (threshold_state[kv_head * 4] >> 8) : 0;
  for (int community = part * kThreads + thread;
       community < active_communities;
       community += parts * kThreads) {
    const int size =
        community_sizes[kv_head * community_capacity + community];
    if (size <= 0) {
      continue;
    }
    const uint16_t key = ordered_fp16_key(
        scores[kv_head * community_capacity + community]);
    if (!kLowByte || (key >> 8) == high_bin) {
      const int bin = kLowByte ? (key & 0xff) : (key >> 8);
      atomicAdd(histogram + bin, size);
    }
  }
  __syncthreads();
  if (thread < kHistogramBins) {
    histogram_workspace[
        (kv_head * parts + part) * kHistogramBins + thread] =
        histogram[thread];
  }
}

template <bool kLowByte>
__global__ __launch_bounds__(kHistogramBins) void threshold_kernel(
    const int32_t* __restrict__ histogram_workspace,
    int32_t* __restrict__ threshold_state,
    int selected_budget,
    int parts) {
  const int kv_head = blockIdx.x;
  const int bin = threadIdx.x;
  __shared__ int histogram[kHistogramBins];
  int mass = 0;
  for (int part = 0; part < parts; ++part) {
    mass += histogram_workspace[
        (kv_head * parts + part) * kHistogramBins + bin];
  }
  histogram[bin] = mass;
  __syncthreads();

  if (bin == 0) {
    int cumulative = kLowByte ? threshold_state[kv_head * 4 + 1] : 0;
    int selected_bin = 0;
    int mass_above = cumulative;
    for (int descending = kHistogramBins - 1;
         descending >= 0;
         --descending) {
      if (cumulative + histogram[descending] >= selected_budget) {
        selected_bin = descending;
        mass_above = cumulative;
        break;
      }
      cumulative += histogram[descending];
    }
    if (kLowByte) {
      const int high_bin = threshold_state[kv_head * 4] >> 8;
      threshold_state[kv_head * 4] =
          (high_bin << 8) | selected_bin;
    } else {
      threshold_state[kv_head * 4] = selected_bin << 8;
    }
    threshold_state[kv_head * 4 + 1] = mass_above;
  }
}

__global__ __launch_bounds__(kThreads) void descriptor_part_totals_kernel(
    const uint16_t* __restrict__ scores,
    const int32_t* __restrict__ community_sizes,
    const int32_t* __restrict__ community_counts,
    const int32_t* __restrict__ threshold_state,
    int32_t* __restrict__ part_state,
    int community_capacity,
    int parts) {
  const int kv_head = blockIdx.x / parts;
  const int part = blockIdx.x - kv_head * parts;
  const int thread = threadIdx.x;
  const int active_communities = max(
      0,
      min(community_capacity, community_counts[kv_head]));
  const int communities_per_part =
      (active_communities + parts - 1) / parts;
  const int part_start =
      min(active_communities, part * communities_per_part);
  const int part_end =
      min(active_communities, part_start + communities_per_part);
  const int threshold_key = threshold_state[kv_head * 4];

  int high_count = 0;
  int high_mass = 0;
  int equal_count = 0;
  int equal_mass = 0;
  for (int community = part_start + thread;
       community < part_end;
       community += kThreads) {
    const int size =
        community_sizes[kv_head * community_capacity + community];
    if (size <= 0) {
      continue;
    }
    const uint16_t key = ordered_fp16_key(
        scores[kv_head * community_capacity + community]);
    if (key > threshold_key) {
      ++high_count;
      high_mass += size;
    } else if (key == threshold_key) {
      ++equal_count;
      equal_mass += size;
    }
  }

  __shared__ Int4 warp_prefix[kWarps];
  __shared__ Int4 block_total;
  block_exclusive_sum(
      {high_count, high_mass, equal_count, equal_mass},
      warp_prefix,
      &block_total);
  if (thread == 0) {
    int32_t* state =
        part_state + (kv_head * parts + part) * kPartFields;
    state[0] = block_total.high_count;
    state[1] = block_total.high_mass;
    state[2] = block_total.equal_count;
    state[3] = block_total.equal_mass;
  }
}

__global__ void descriptor_part_prefix_kernel(
    int32_t* __restrict__ threshold_state,
    int32_t* __restrict__ part_state,
    int32_t* __restrict__ descriptor_counts,
    int parts) {
  const int kv_head = blockIdx.x;
  if (threadIdx.x == 0) {
    int high_count = 0;
    int high_mass = 0;
    int equal_count = 0;
    int equal_mass = 0;
    for (int part = 0; part < parts; ++part) {
      int32_t* state =
          part_state + (kv_head * parts + part) * kPartFields;
      state[4] = high_count;
      state[5] = high_mass;
      state[6] = equal_count;
      state[7] = equal_mass;
      high_count += state[0];
      high_mass += state[1];
      equal_count += state[2];
      equal_mass += state[3];
    }
    threshold_state[kv_head * 4 + 2] = high_count;
    threshold_state[kv_head * 4 + 3] = high_mass;
    descriptor_counts[kv_head] = high_count;
  }
}

__global__ __launch_bounds__(kThreads) void compact_descriptors_kernel(
    const uint16_t* __restrict__ scores,
    const int32_t* __restrict__ community_sizes,
    const int32_t* __restrict__ community_counts,
    const int32_t* __restrict__ threshold_state,
    const int32_t* __restrict__ part_state,
    int32_t* __restrict__ descriptor_communities,
    int32_t* __restrict__ descriptor_cumulative_ends,
    int32_t* __restrict__ descriptor_counts,
    int community_capacity,
    int descriptor_capacity,
    int parts,
    int selected_budget) {
  const int kv_head = blockIdx.x / parts;
  const int part = blockIdx.x - kv_head * parts;
  const int thread = threadIdx.x;
  const int active_communities = max(
      0,
      min(community_capacity, community_counts[kv_head]));
  const int communities_per_part =
      (active_communities + parts - 1) / parts;
  const int part_start =
      min(active_communities, part * communities_per_part);
  const int part_end =
      min(active_communities, part_start + communities_per_part);
  const int threshold_key = threshold_state[kv_head * 4];
  const int total_high_count = threshold_state[kv_head * 4 + 2];
  const int total_high_mass = threshold_state[kv_head * 4 + 3];
  const int32_t* prefix =
      part_state + (kv_head * parts + part) * kPartFields;

  __shared__ Int4 warp_prefix[kWarps];
  __shared__ Int4 block_total;
  int local_high_count = 0;
  int local_high_mass = 0;
  int local_equal_count = 0;
  int local_equal_mass = 0;
  int thread_selected_equal = 0;
  for (int block_start = part_start;
       block_start < part_end;
       block_start += kThreads) {
    const int community = block_start + thread;
    const bool valid = community < part_end;
    const int size = valid
        ? community_sizes[kv_head * community_capacity + community]
        : 0;
    const uint16_t key = size > 0
        ? ordered_fp16_key(
              scores[kv_head * community_capacity + community])
        : 0;
    const bool selected_high = size > 0 && key > threshold_key;
    const bool selected_equal = size > 0 && key == threshold_key;

    const Int4 block_prefix = block_exclusive_sum(
        {
            selected_high ? 1 : 0,
            selected_high ? size : 0,
            selected_equal ? 1 : 0,
            selected_equal ? size : 0,
        },
        warp_prefix,
        &block_total);

    if (selected_high) {
      const int descriptor =
          prefix[4] + local_high_count + block_prefix.high_count;
      const int cumulative_end =
          prefix[5] + local_high_mass + block_prefix.high_mass + size;
      if (descriptor < descriptor_capacity) {
        descriptor_communities[
            kv_head * descriptor_capacity + descriptor] = community;
        descriptor_cumulative_ends[
            kv_head * descriptor_capacity + descriptor] = cumulative_end;
      }
    }

    const int equal_start =
        total_high_mass
        + prefix[7]
        + local_equal_mass
        + block_prefix.equal_mass;
    const bool stored_equal =
        selected_equal && equal_start < selected_budget;
    if (stored_equal) {
      const int descriptor =
          total_high_count
          + prefix[6]
          + local_equal_count
          + block_prefix.equal_count;
      const int cumulative_end =
          min(selected_budget, equal_start + size);
      if (descriptor < descriptor_capacity) {
        descriptor_communities[
            kv_head * descriptor_capacity + descriptor] = community;
        descriptor_cumulative_ends[
            kv_head * descriptor_capacity + descriptor] = cumulative_end;
      }
      ++thread_selected_equal;
    }

    local_high_count += block_total.high_count;
    local_high_mass += block_total.high_mass;
    local_equal_count += block_total.equal_count;
    local_equal_mass += block_total.equal_mass;
  }

  block_exclusive_sum(
      {thread_selected_equal, 0, 0, 0},
      warp_prefix,
      &block_total);
  if (thread == 0 && block_total.high_count > 0) {
    atomicAdd(descriptor_counts + kv_head, block_total.high_count);
  }
}

__global__ void tile_descriptor_kernel(
    const int32_t* __restrict__ descriptor_cumulative_ends,
    const int32_t* __restrict__ descriptor_counts,
    int32_t* __restrict__ tile_descriptors,
    int descriptor_capacity,
    int tile_count,
    int num_sink) {
  const int kv_head = blockIdx.x;
  const int tile = threadIdx.x;
  if (tile >= tile_count) {
    return;
  }
  const int descriptor_count = descriptor_counts[kv_head];
  const int selected_offset = max(tile * kTile - num_sink, 0);
  int low = 0;
  int high = descriptor_count;
  while (low < high) {
    const int middle = (low + high) / 2;
    const int cumulative_end =
        descriptor_cumulative_ends[
            kv_head * descriptor_capacity + middle];
    if (cumulative_end <= selected_offset) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  const int start_descriptor =
      max(0, min(low, descriptor_count - 1));
  tile_descriptors[kv_head * tile_count + tile] = start_descriptor;
}

template <int kRetainedItems>
__global__ __cluster_dims__(
    kClusterParts,
    1,
    1) __launch_bounds__(kThreads) void cluster_descriptor_select_kernel(
    const uint16_t* __restrict__ scores,
    const int32_t* __restrict__ community_sizes,
    const int32_t* __restrict__ community_counts,
    int32_t* __restrict__ threshold_state,
    int32_t* __restrict__ descriptor_communities,
    int32_t* __restrict__ descriptor_cumulative_ends,
    int32_t* __restrict__ descriptor_counts,
    int32_t* __restrict__ tile_descriptors,
    int community_capacity,
    int descriptor_capacity,
    int tile_count,
    int selected_budget,
    int num_sink) {
  cooperative_groups::cluster_group cluster =
      cooperative_groups::this_cluster();
  const int cluster_rank = static_cast<int>(cluster.block_rank());
  const int kv_head = blockIdx.x / kClusterParts;
  const int thread = threadIdx.x;
  const int active_communities = max(
      0,
      min(community_capacity, community_counts[kv_head]));
  const int communities_per_part =
      (active_communities + kClusterParts - 1) / kClusterParts;
  const int part_start =
      min(active_communities, cluster_rank * communities_per_part);
  const int part_end =
      min(active_communities, part_start + communities_per_part);
  const int part_iterations = kRetainedItems > 0
      ? kRetainedItems
      : (part_end - part_start + kThreads - 1) / kThreads;
  const int cached_threshold_key = threshold_state[kv_head * 4];
  const int cached_high_bin =
      cached_threshold_key >= 0 ? cached_threshold_key >> 8 : -1;
  uint16_t retained_keys[kRetainedItems > 0 ? kRetainedItems : 1];
  int retained_sizes[kRetainedItems > 0 ? kRetainedItems : 1];

  __shared__ int high_histogram[kHistogramBins];
  __shared__ int low_histogram[kHistogramBins];
  __shared__ int high_bin;
  __shared__ int high_mass_above;
  __shared__ int threshold_key;
  __shared__ int threshold_mass_above;
  __shared__ int total_high_count;
  __shared__ int total_high_mass;
  __shared__ int total_selected_equal;
  __shared__ Int4 scan_warp_prefix[kWarps];
  __shared__ Int4 scan_total;
  __shared__ Int4 part_prefix;

  int* root_high_bin = cluster.map_shared_rank(&high_bin, 0);
  int* root_high_mass_above =
      cluster.map_shared_rank(&high_mass_above, 0);
  int* root_threshold_key =
      cluster.map_shared_rank(&threshold_key, 0);
  int* root_threshold_mass_above =
      cluster.map_shared_rank(&threshold_mass_above, 0);
  int* root_total_high_count =
      cluster.map_shared_rank(&total_high_count, 0);
  int* root_total_high_mass =
      cluster.map_shared_rank(&total_high_mass, 0);
  int* root_total_selected_equal =
      cluster.map_shared_rank(&total_selected_equal, 0);


  if (thread < kHistogramBins) {
    high_histogram[thread] = 0;
    low_histogram[thread] = 0;
  }
  if (cluster_rank == 0 && thread == 0) {
    total_selected_equal = 0;
  }
  __syncthreads();
  #pragma unroll
  for (int item = 0; item < part_iterations; ++item) {
    const int community = part_start + thread * part_iterations + item;
    const bool valid = community < part_end;
    const int size = valid
        ? community_sizes[kv_head * community_capacity + community]
        : 0;
    const uint16_t key = size > 0
        ? ordered_fp16_key(
              scores[kv_head * community_capacity + community])
        : 0;
    if constexpr (kRetainedItems > 0) {
      retained_sizes[item] = size;
      retained_keys[item] = key;
    }
    if (size > 0) {
      atomicAdd(high_histogram + (key >> 8), size);
      if ((key >> 8) == cached_high_bin) {
        atomicAdd(low_histogram + (key & 0xff), size);
      }
    }
  }
  cluster.sync();

  if (cluster_rank == 0) {
    if (thread < kHistogramBins) {
      int total = 0;
#pragma unroll
      for (int rank = 0; rank < kClusterParts; ++rank) {
        total +=
            cluster.map_shared_rank(high_histogram, rank)[thread];
      }
      high_histogram[thread] = total;
    }
    __syncthreads();
    if (thread == 0) {
      int cumulative = 0;
      high_bin = 0;
      high_mass_above = 0;
      for (int bin = kHistogramBins - 1; bin >= 0; --bin) {
        if (cumulative + high_histogram[bin] >= selected_budget) {
          high_bin = bin;
          high_mass_above = cumulative;
          break;
        }
        cumulative += high_histogram[bin];
      }
    }
  }
  cluster.sync();


  const int selected_high_bin = *root_high_bin;
  if (cached_high_bin != selected_high_bin) {
    if (thread < kHistogramBins) {
      low_histogram[thread] = 0;
    }
    __syncthreads();
    #pragma unroll
    for (int item = 0; item < part_iterations; ++item) {
      const int community = part_start + thread * part_iterations + item;
      const bool valid = community < part_end;
      const int size = kRetainedItems > 0
          ? retained_sizes[item]
          : (valid
                 ? community_sizes[
                       kv_head * community_capacity + community]
                 : 0);
      if (size > 0) {
        const uint16_t key = kRetainedItems > 0
            ? retained_keys[item]
            : ordered_fp16_key(
                  scores[kv_head * community_capacity + community]);
        if ((key >> 8) == selected_high_bin) {
          atomicAdd(low_histogram + (key & 0xff), size);
        }
      }
    }
    cluster.sync();
  }

  if (cluster_rank == 0) {
    if (thread < kHistogramBins) {
      int total = 0;
#pragma unroll
      for (int rank = 0; rank < kClusterParts; ++rank) {
        total +=
            cluster.map_shared_rank(low_histogram, rank)[thread];
      }
      low_histogram[thread] = total;
    }
    __syncthreads();
    if (thread == 0) {
      int cumulative = high_mass_above;
      int selected_low_bin = 0;
      threshold_mass_above = high_mass_above;
      for (int bin = kHistogramBins - 1; bin >= 0; --bin) {
        if (cumulative + low_histogram[bin] >= selected_budget) {
          selected_low_bin = bin;
          threshold_mass_above = cumulative;
          break;
        }
        cumulative += low_histogram[bin];
      }
      threshold_key = (high_bin << 8) | selected_low_bin;
      threshold_state[kv_head * 4] = threshold_key;
    }
  }
  cluster.sync();


  const int selected_threshold_key = *root_threshold_key;
  int high_count = 0;
  int high_mass = 0;
  int equal_count = 0;
  int equal_mass = 0;
  #pragma unroll
  for (int item = 0; item < part_iterations; ++item) {
    const int community = part_start + thread * part_iterations + item;
    const bool valid = community < part_end;
    const int size = kRetainedItems > 0
        ? retained_sizes[item]
        : (valid
               ? community_sizes[kv_head * community_capacity + community]
               : 0);
    if (size <= 0) {
      continue;
    }
    const uint16_t key = kRetainedItems > 0
        ? retained_keys[item]
        : ordered_fp16_key(
              scores[kv_head * community_capacity + community]);
    if (key > selected_threshold_key) {
      ++high_count;
      high_mass += size;
    } else if (key == selected_threshold_key) {
      ++equal_count;
      equal_mass += size;
    }
  }
  const Int4 thread_prefix = block_exclusive_sum(
      {high_count, high_mass, equal_count, equal_mass},
      scan_warp_prefix,
      &scan_total);
  cluster.sync();

  if (cluster_rank == 0 && thread == 0) {
    Int4 prefix = {0, 0, 0, 0};
#pragma unroll
    for (int rank = 0; rank < kClusterParts; ++rank) {
      Int4* remote_prefix =
          cluster.map_shared_rank(&part_prefix, rank);
      const Int4 remote_total =
          *cluster.map_shared_rank(&scan_total, rank);
      *remote_prefix = prefix;
      prefix.high_count += remote_total.high_count;
      prefix.high_mass += remote_total.high_mass;
      prefix.equal_count += remote_total.equal_count;
      prefix.equal_mass += remote_total.equal_mass;
    }
    total_high_count = prefix.high_count;
    total_high_mass = prefix.high_mass;
  }
  cluster.sync();

  int thread_high_count = 0;
  int thread_high_mass = 0;
  int thread_equal_count = 0;
  int thread_equal_mass = 0;
  int selected_equal_count = 0;
  #pragma unroll
  for (int item = 0; item < part_iterations; ++item) {
    const int community = part_start + thread * part_iterations + item;
    const bool valid = community < part_end;
    const int size = kRetainedItems > 0
        ? retained_sizes[item]
        : (valid
               ? community_sizes[kv_head * community_capacity + community]
               : 0);
    const uint16_t key = kRetainedItems > 0
        ? retained_keys[item]
        : (size > 0
               ? ordered_fp16_key(
                     scores[kv_head * community_capacity + community])
               : 0);
    const bool selected_high =
        size > 0 && key > selected_threshold_key;
    const bool selected_equal =
        size > 0 && key == selected_threshold_key;

    if (selected_high) {
      const int descriptor =
          part_prefix.high_count
          + thread_prefix.high_count
          + thread_high_count;
      const int selected_start =
          part_prefix.high_mass
          + thread_prefix.high_mass
          + thread_high_mass;
      const int cumulative_end =
          selected_start + size;
      if (descriptor < descriptor_capacity) {
        descriptor_communities[
            kv_head * descriptor_capacity + descriptor] = community;
        descriptor_cumulative_ends[
            kv_head * descriptor_capacity + descriptor] = cumulative_end;
        emit_descriptor_tile_starts(
            tile_descriptors,
            kv_head,
            tile_count,
            num_sink,
            descriptor,
            selected_start,
            cumulative_end);
      }
      ++thread_high_count;
      thread_high_mass += size;
    }

    const int equal_start =
        *root_total_high_mass
        + part_prefix.equal_mass
        + thread_prefix.equal_mass
        + thread_equal_mass;
    const bool stored_equal =
        selected_equal && equal_start < selected_budget;
    if (stored_equal) {
      const int descriptor =
          *root_total_high_count
          + part_prefix.equal_count
          + thread_prefix.equal_count
          + thread_equal_count;
      const int cumulative_end =
          min(selected_budget, equal_start + size);
      if (descriptor < descriptor_capacity) {
        descriptor_communities[
            kv_head * descriptor_capacity + descriptor] = community;
        descriptor_cumulative_ends[
            kv_head * descriptor_capacity + descriptor] = cumulative_end;
        emit_descriptor_tile_starts(
            tile_descriptors,
            kv_head,
            tile_count,
            num_sink,
            descriptor,
            equal_start,
            cumulative_end);
      }
      ++selected_equal_count;
    }
    if (selected_equal) {
      ++thread_equal_count;
      thread_equal_mass += size;
    }
  }
  block_sum(
      {selected_equal_count, 0, 0, 0},
      scan_warp_prefix,
      &scan_total);
  if (thread == 0 && scan_total.high_count > 0) {
    atomicAdd(root_total_selected_equal, scan_total.high_count);
  }
  cluster.sync();


  if (cluster_rank == 0 && thread == 0) {
    descriptor_counts[kv_head] =
        total_high_count + total_selected_equal;
  }

}

void require_cuda_contiguous(
    const torch::Tensor& tensor,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void require_int32(const torch::Tensor& tensor, const char* name) {
  require_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must use int32");
}

}  // namespace

void select_weighted_descriptors_cuda(
    torch::Tensor scores,
    torch::Tensor community_sizes,
    torch::Tensor community_counts,
    torch::Tensor histogram_workspace,
    torch::Tensor threshold_state,
    torch::Tensor part_state,
    torch::Tensor descriptor_communities,
    torch::Tensor descriptor_cumulative_ends,
    torch::Tensor descriptor_counts,
    torch::Tensor tile_descriptors,
    int64_t num_sink,
    bool use_cluster) {
  require_cuda_contiguous(scores, "scores");
  TORCH_CHECK(scores.scalar_type() == at::kHalf, "scores must use FP16");
  require_int32(community_sizes, "community_sizes");
  require_int32(community_counts, "community_counts");
  require_int32(histogram_workspace, "histogram_workspace");
  require_int32(threshold_state, "threshold_state");
  require_int32(part_state, "part_state");
  require_int32(descriptor_communities, "descriptor_communities");
  require_int32(
      descriptor_cumulative_ends,
      "descriptor_cumulative_ends");
  require_int32(descriptor_counts, "descriptor_counts");
  require_int32(tile_descriptors, "tile_descriptors");

  TORCH_CHECK(scores.dim() == 2, "scores must be rank two");
  TORCH_CHECK(
      community_sizes.sizes() == scores.sizes(),
      "community_sizes must match scores");
  const int64_t kv_heads = scores.size(0);
  const int64_t community_capacity = scores.size(1);
  TORCH_CHECK(
      community_counts.sizes() == torch::IntArrayRef({kv_heads}),
      "community_counts must have one entry per KV head");
  TORCH_CHECK(
      histogram_workspace.dim() == 3 &&
          histogram_workspace.size(0) == kv_heads &&
          histogram_workspace.size(2) == kHistogramBins,
      "histogram_workspace must have shape [Hkv, parts, 256]");
  const int64_t parts = histogram_workspace.size(1);
  TORCH_CHECK(
      parts == 1 || parts == 2 || parts == 4 ||
          parts == 8 || parts == 16,
      "selector parts must be a power of two in [1, 16]");
  TORCH_CHECK(
      threshold_state.sizes() == torch::IntArrayRef({kv_heads, 4}),
      "threshold_state must have shape [Hkv, 4]");
  TORCH_CHECK(
      part_state.sizes() ==
          torch::IntArrayRef({kv_heads, parts, kPartFields}),
      "part_state must have shape [Hkv, parts, 8]");
  TORCH_CHECK(
      descriptor_communities.dim() == 2 &&
          descriptor_communities.size(0) == kv_heads,
      "descriptor_communities must have one row per KV head");
  TORCH_CHECK(
      descriptor_cumulative_ends.sizes() ==
          descriptor_communities.sizes(),
      "descriptor cumulative ends must match communities");
  TORCH_CHECK(
      descriptor_counts.sizes() == torch::IntArrayRef({kv_heads}),
      "descriptor_counts must have one entry per KV head");
  TORCH_CHECK(
      tile_descriptors.dim() == 2 &&
          tile_descriptors.size(0) == kv_heads,
      "tile_descriptors must have one row per KV head");
  const int64_t tile_count = tile_descriptors.size(1);
  TORCH_CHECK(tile_count > 0, "at least one attention tile is required");
  const int64_t token_budget = tile_count * kTile;
  TORCH_CHECK(
      num_sink >= 0 && num_sink + 1 < token_budget,
      "num_sink leaves no selected-token budget");
  const int64_t selected_budget = token_budget - num_sink - 1;
  TORCH_CHECK(
      descriptor_communities.size(1) >= selected_budget,
      "descriptor capacity must cover the singleton case");
  TORCH_CHECK(
      kv_heads <= std::numeric_limits<int>::max() &&
          community_capacity <= std::numeric_limits<int>::max() &&
          selected_budget <= std::numeric_limits<int>::max(),
      "descriptor-selection dimensions exceed int32");

  const auto device = scores.device();
  for (const auto& tensor : {
           community_sizes,
           community_counts,
           histogram_workspace,
           threshold_state,
           part_state,
           descriptor_communities,
           descriptor_cumulative_ends,
           descriptor_counts,
           tile_descriptors}) {
    TORCH_CHECK(tensor.device() == device, "all tensors must share a device");
  }

  const c10::cuda::CUDAGuard guard(device);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index());
  const int head_count = static_cast<int>(kv_heads);
  const int part_count = static_cast<int>(parts);
  const int capacity = static_cast<int>(community_capacity);
  const int budget = static_cast<int>(selected_budget);
  const int tiles = static_cast<int>(tile_count);
  const int descriptor_capacity =
      static_cast<int>(descriptor_communities.size(1));
  const int histogram_blocks = head_count * part_count;

  if (use_cluster) {
    if (capacity <= kClusterParts * kThreads) {
      cluster_descriptor_select_kernel<1>
          <<<head_count * kClusterParts, kThreads, 0, stream>>>(
              reinterpret_cast<const uint16_t*>(
                  scores.data_ptr<at::Half>()),
              community_sizes.data_ptr<int32_t>(),
              community_counts.data_ptr<int32_t>(),
              threshold_state.data_ptr<int32_t>(),
              descriptor_communities.data_ptr<int32_t>(),
              descriptor_cumulative_ends.data_ptr<int32_t>(),
              descriptor_counts.data_ptr<int32_t>(),
              tile_descriptors.data_ptr<int32_t>(),
              capacity,
              descriptor_capacity,
              tiles,
              budget,
              static_cast<int>(num_sink));
    } else if (capacity <= 2 * kClusterParts * kThreads) {
      cluster_descriptor_select_kernel<2>
          <<<head_count * kClusterParts, kThreads, 0, stream>>>(
              reinterpret_cast<const uint16_t*>(
                  scores.data_ptr<at::Half>()),
              community_sizes.data_ptr<int32_t>(),
              community_counts.data_ptr<int32_t>(),
              threshold_state.data_ptr<int32_t>(),
              descriptor_communities.data_ptr<int32_t>(),
              descriptor_cumulative_ends.data_ptr<int32_t>(),
              descriptor_counts.data_ptr<int32_t>(),
              tile_descriptors.data_ptr<int32_t>(),
              capacity,
              descriptor_capacity,
              tiles,
              budget,
              static_cast<int>(num_sink));
    } else if (capacity <= 3 * kClusterParts * kThreads) {
      cluster_descriptor_select_kernel<3>
          <<<head_count * kClusterParts, kThreads, 0, stream>>>(
              reinterpret_cast<const uint16_t*>(
                  scores.data_ptr<at::Half>()),
              community_sizes.data_ptr<int32_t>(),
              community_counts.data_ptr<int32_t>(),
              threshold_state.data_ptr<int32_t>(),
              descriptor_communities.data_ptr<int32_t>(),
              descriptor_cumulative_ends.data_ptr<int32_t>(),
              descriptor_counts.data_ptr<int32_t>(),
              tile_descriptors.data_ptr<int32_t>(),
              capacity,
              descriptor_capacity,
              tiles,
              budget,
              static_cast<int>(num_sink));
    } else if (capacity <= 4 * kClusterParts * kThreads) {
      cluster_descriptor_select_kernel<4>
          <<<head_count * kClusterParts, kThreads, 0, stream>>>(
              reinterpret_cast<const uint16_t*>(
                  scores.data_ptr<at::Half>()),
              community_sizes.data_ptr<int32_t>(),
              community_counts.data_ptr<int32_t>(),
              threshold_state.data_ptr<int32_t>(),
              descriptor_communities.data_ptr<int32_t>(),
              descriptor_cumulative_ends.data_ptr<int32_t>(),
              descriptor_counts.data_ptr<int32_t>(),
              tile_descriptors.data_ptr<int32_t>(),
              capacity,
              descriptor_capacity,
              tiles,
              budget,
              static_cast<int>(num_sink));
    } else if (capacity <= 5 * kClusterParts * kThreads) {
      cluster_descriptor_select_kernel<5>
          <<<head_count * kClusterParts, kThreads, 0, stream>>>(
              reinterpret_cast<const uint16_t*>(
                  scores.data_ptr<at::Half>()),
              community_sizes.data_ptr<int32_t>(),
              community_counts.data_ptr<int32_t>(),
              threshold_state.data_ptr<int32_t>(),
              descriptor_communities.data_ptr<int32_t>(),
              descriptor_cumulative_ends.data_ptr<int32_t>(),
              descriptor_counts.data_ptr<int32_t>(),
              tile_descriptors.data_ptr<int32_t>(),
              capacity,
              descriptor_capacity,
              tiles,
              budget,
              static_cast<int>(num_sink));
    } else if (capacity <= 6 * kClusterParts * kThreads) {
      cluster_descriptor_select_kernel<6>
          <<<head_count * kClusterParts, kThreads, 0, stream>>>(
              reinterpret_cast<const uint16_t*>(
                  scores.data_ptr<at::Half>()),
              community_sizes.data_ptr<int32_t>(),
              community_counts.data_ptr<int32_t>(),
              threshold_state.data_ptr<int32_t>(),
              descriptor_communities.data_ptr<int32_t>(),
              descriptor_cumulative_ends.data_ptr<int32_t>(),
              descriptor_counts.data_ptr<int32_t>(),
              tile_descriptors.data_ptr<int32_t>(),
              capacity,
              descriptor_capacity,
              tiles,
              budget,
              static_cast<int>(num_sink));
    } else if (capacity <= 7 * kClusterParts * kThreads) {
      cluster_descriptor_select_kernel<7>
          <<<head_count * kClusterParts, kThreads, 0, stream>>>(
              reinterpret_cast<const uint16_t*>(
                  scores.data_ptr<at::Half>()),
              community_sizes.data_ptr<int32_t>(),
              community_counts.data_ptr<int32_t>(),
              threshold_state.data_ptr<int32_t>(),
              descriptor_communities.data_ptr<int32_t>(),
              descriptor_cumulative_ends.data_ptr<int32_t>(),
              descriptor_counts.data_ptr<int32_t>(),
              tile_descriptors.data_ptr<int32_t>(),
              capacity,
              descriptor_capacity,
              tiles,
              budget,
              static_cast<int>(num_sink));
    } else if (capacity <= 8 * kClusterParts * kThreads) {
      cluster_descriptor_select_kernel<8>
          <<<head_count * kClusterParts, kThreads, 0, stream>>>(
              reinterpret_cast<const uint16_t*>(
                  scores.data_ptr<at::Half>()),
              community_sizes.data_ptr<int32_t>(),
              community_counts.data_ptr<int32_t>(),
              threshold_state.data_ptr<int32_t>(),
              descriptor_communities.data_ptr<int32_t>(),
              descriptor_cumulative_ends.data_ptr<int32_t>(),
              descriptor_counts.data_ptr<int32_t>(),
              tile_descriptors.data_ptr<int32_t>(),
              capacity,
              descriptor_capacity,
              tiles,
              budget,
              static_cast<int>(num_sink));
    } else {
      cluster_descriptor_select_kernel<0>
          <<<head_count * kClusterParts, kThreads, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(
                scores.data_ptr<at::Half>()),
            community_sizes.data_ptr<int32_t>(),
            community_counts.data_ptr<int32_t>(),
            threshold_state.data_ptr<int32_t>(),
            descriptor_communities.data_ptr<int32_t>(),
            descriptor_cumulative_ends.data_ptr<int32_t>(),
            descriptor_counts.data_ptr<int32_t>(),
            tile_descriptors.data_ptr<int32_t>(),
            capacity,
            descriptor_capacity,
            tiles,
            budget,
            static_cast<int>(num_sink));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  partial_histogram_kernel<false>
      <<<histogram_blocks, kThreads, 0, stream>>>(
          reinterpret_cast<const uint16_t*>(
              scores.data_ptr<at::Half>()),
          community_sizes.data_ptr<int32_t>(),
          community_counts.data_ptr<int32_t>(),
          histogram_workspace.data_ptr<int32_t>(),
          threshold_state.data_ptr<int32_t>(),
          capacity,
          part_count);
  threshold_kernel<false><<<head_count, kHistogramBins, 0, stream>>>(
      histogram_workspace.data_ptr<int32_t>(),
      threshold_state.data_ptr<int32_t>(),
      budget,
      part_count);
  partial_histogram_kernel<true>
      <<<histogram_blocks, kThreads, 0, stream>>>(
          reinterpret_cast<const uint16_t*>(
              scores.data_ptr<at::Half>()),
          community_sizes.data_ptr<int32_t>(),
          community_counts.data_ptr<int32_t>(),
          histogram_workspace.data_ptr<int32_t>(),
          threshold_state.data_ptr<int32_t>(),
          capacity,
          part_count);
  threshold_kernel<true><<<head_count, kHistogramBins, 0, stream>>>(
      histogram_workspace.data_ptr<int32_t>(),
      threshold_state.data_ptr<int32_t>(),
      budget,
      part_count);
  descriptor_part_totals_kernel
      <<<histogram_blocks, kThreads, 0, stream>>>(
          reinterpret_cast<const uint16_t*>(
              scores.data_ptr<at::Half>()),
          community_sizes.data_ptr<int32_t>(),
          community_counts.data_ptr<int32_t>(),
          threshold_state.data_ptr<int32_t>(),
          part_state.data_ptr<int32_t>(),
          capacity,
          part_count);
  descriptor_part_prefix_kernel<<<head_count, 256, 0, stream>>>(
      threshold_state.data_ptr<int32_t>(),
      part_state.data_ptr<int32_t>(),
      descriptor_counts.data_ptr<int32_t>(),
      part_count);
  compact_descriptors_kernel
      <<<histogram_blocks, kThreads, 0, stream>>>(
          reinterpret_cast<const uint16_t*>(
              scores.data_ptr<at::Half>()),
          community_sizes.data_ptr<int32_t>(),
          community_counts.data_ptr<int32_t>(),
          threshold_state.data_ptr<int32_t>(),
          part_state.data_ptr<int32_t>(),
          descriptor_communities.data_ptr<int32_t>(),
          descriptor_cumulative_ends.data_ptr<int32_t>(),
          descriptor_counts.data_ptr<int32_t>(),
          capacity,
          descriptor_capacity,
          part_count,
          budget);
  const int tile_threads = std::max(32, std::min(1024, tiles));
  tile_descriptor_kernel<<<head_count, tile_threads, 0, stream>>>(
      descriptor_cumulative_ends.data_ptr<int32_t>(),
      descriptor_counts.data_ptr<int32_t>(),
      tile_descriptors.data_ptr<int32_t>(),
      descriptor_capacity,
      tiles,
      static_cast<int>(num_sink));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
