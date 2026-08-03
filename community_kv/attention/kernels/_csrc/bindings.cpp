#include <torch/extension.h>

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
    bool use_cluster);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "select_weighted_descriptors",
      &select_weighted_descriptors_cuda,
      "CommunityKV exact weighted descriptor selection (CUDA)");
}
