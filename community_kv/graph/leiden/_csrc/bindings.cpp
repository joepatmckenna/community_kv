// pybind11 bindings exposing launch_batched_leiden as a torch-compatible
// callable. Inputs are torch tensors on the same CUDA device; we extract
// raw pointers via .data_ptr() and forward to the CUDA library.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>

#include "leiden.cuh"

namespace py = pybind11;

namespace community_kv::native_leiden {
// Forward declaration from aggregate.cu
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
    cudaStream_t   stream);
}

namespace {

void check_cuda_int32(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(),    name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.dtype() == torch::kInt32, name, " must be int32");
}
void check_cuda_float32(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(),    name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
}

// Returns (labels (V_total,) int32, modularity_per_graph (G,) float64).
// Vertices that don't appear in any edge come back as -1 in labels;
// the Python caller fills singletons.
std::tuple<torch::Tensor, torch::Tensor> batched_leiden(
    torch::Tensor edge_src,
    torch::Tensor edge_dst,
    torch::Tensor edge_weight,
    int64_t       G,
    int64_t       seq_len,
    int           max_level,
    double        resolution,
    double        theta,
    int           max_inner_iter,
    int           seed,
    bool          use_boltzmann)
{
  check_cuda_int32(edge_src, "edge_src");
  check_cuda_int32(edge_dst, "edge_dst");
  check_cuda_float32(edge_weight, "edge_weight");
  TORCH_CHECK(edge_src.numel() == edge_dst.numel(),
              "edge_src and edge_dst must have equal length");
  TORCH_CHECK(edge_src.numel() == edge_weight.numel(),
              "edge_src and edge_weight must have equal length");
  TORCH_CHECK(G > 0,       "G must be positive");
  TORCH_CHECK(seq_len > 0, "seq_len must be positive");

  int64_t V_total = G * seq_len;
  auto opts_i32 = torch::TensorOptions()
      .dtype(torch::kInt32).device(edge_src.device());
  auto opts_f64 = torch::TensorOptions()
      .dtype(torch::kFloat64).device(edge_src.device());
  torch::Tensor labels = torch::empty({V_total}, opts_i32);
  torch::Tensor modularity = torch::empty({G}, opts_f64);

  community_kv::native_leiden::LeidenParams p;
  p.max_level      = max_level;
  p.resolution     = (float)resolution;
  p.theta          = (float)theta;
  p.max_inner_iter = max_inner_iter;
  p.seed           = seed;
  p.use_boltzmann_refine = use_boltzmann;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(edge_src.device().index());

  community_kv::native_leiden::launch_batched_leiden(
      edge_src.data_ptr<int32_t>(),
      edge_dst.data_ptr<int32_t>(),
      edge_weight.data_ptr<float>(),
      edge_src.numel(),
      (int32_t)G,
      (int32_t)seq_len,
      labels.data_ptr<int32_t>(),
      modularity.data_ptr<double>(),
      p,
      (void*)stream);

  return {labels, modularity};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Native CUDA batched Leiden for community_kv";
  // Both entry points release the GIL while the C++ / CUDA work runs so
  // that Python threads issuing calls on different CUDA streams can
  // overlap on the device. Without this, we'd serialize on the GIL
  // regardless of stream affinity. Tensor refcounts are bumped during
  // argument conversion (still under GIL) and decremented after return
  // (re-acquired automatically), so the body is safe to run nogil.
  m.def("batched_leiden", &batched_leiden,
        py::arg("edge_src"),
        py::arg("edge_dst"),
        py::arg("edge_weight"),
        py::arg("G"),
        py::arg("seq_len"),
        py::arg("max_level")      = 6,
        py::arg("resolution")     = 1.0,
        py::arg("theta")          = 0.01,
        py::arg("max_inner_iter") = 8,
        py::arg("seed")           = 0,
        py::arg("use_boltzmann")  = false,
        py::call_guard<py::gil_scoped_release>());

  m.def("aggregate_coo", [](
      torch::Tensor edge_src,
      torch::Tensor edge_dst,
      torch::Tensor edge_weight,
      torch::Tensor labels,
      int64_t       V_super_total) {
    check_cuda_int32(edge_src, "edge_src");
    check_cuda_int32(edge_dst, "edge_dst");
    check_cuda_float32(edge_weight, "edge_weight");
    check_cuda_int32(labels, "labels");
    TORCH_CHECK(edge_src.numel() == edge_dst.numel(),
                "edge_src and edge_dst must have equal length");
    TORCH_CHECK(edge_src.numel() == edge_weight.numel(),
                "edge_src and edge_weight must have equal length");
    TORCH_CHECK(V_super_total > 0, "V_super_total must be positive");

    int64_t E_in = edge_src.numel();
    auto opts_i32 = torch::TensorOptions()
        .dtype(torch::kInt32).device(edge_src.device());
    auto opts_f32 = torch::TensorOptions()
        .dtype(torch::kFloat32).device(edge_src.device());

    // Allocate worst-case output capacity (E_in); shrink after we get the
    // actual count back.
    torch::Tensor out_src = torch::empty({E_in}, opts_i32);
    torch::Tensor out_dst = torch::empty({E_in}, opts_i32);
    torch::Tensor out_w   = torch::empty({E_in}, opts_f32);
    int64_t E_out = 0;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(edge_src.device().index());
    community_kv::native_leiden::aggregate_upper_coo(
        edge_src.data_ptr<int32_t>(),
        edge_dst.data_ptr<int32_t>(),
        edge_weight.data_ptr<float>(),
        E_in,
        labels.data_ptr<int32_t>(),
        (int32_t)V_super_total,
        out_src.data_ptr<int32_t>(),
        out_dst.data_ptr<int32_t>(),
        out_w.data_ptr<float>(),
        &E_out,
        stream);

    return std::make_tuple(
        out_src.narrow(0, 0, E_out).contiguous(),
        out_dst.narrow(0, 0, E_out).contiguous(),
        out_w.narrow(0, 0, E_out).contiguous());
  },
  py::arg("edge_src"),
  py::arg("edge_dst"),
  py::arg("edge_weight"),
  py::arg("labels"),
  py::arg("V_super_total"),
  "Aggregate an upper-triangular COO using `labels` (densified per-graph).\n"
  "Returns (new_src, new_dst, new_weight) for the contracted graph.",
  py::call_guard<py::gil_scoped_release>());
}
