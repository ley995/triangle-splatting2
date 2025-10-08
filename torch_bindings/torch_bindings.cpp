
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "triangulation_bindings.h"
#include "utils/cuda_array.h"

namespace triangulation {

struct TorchBuffer : public OpaqueBuffer {
    torch::Tensor tensor;

    TorchBuffer(size_t bytes) {
        // allocate on CUDA device
        // int64 dtype for alignment
        size_t num_words = (bytes + sizeof(int64_t) - 1) / sizeof(int64_t);
        tensor = torch::empty({(int64_t)num_words},
                              torch::dtype(torch::kInt64).device(torch::kCUDA));
    }

    void *data() override { return tensor.data_ptr(); }
};

std::unique_ptr<OpaqueBuffer> allocate_buffer(size_t bytes) {
    return std::make_unique<TorchBuffer>(bytes);
}

} // namespace triangulation

namespace triangulation_bindings {

PYBIND11_MODULE(torch_bindings, module) {
    using namespace triangulation_bindings;

    module.doc() = "triangulation pytorch bindings module";

    init_triangulation_bindings(module);

}

} // namespace triangulation_bindings
