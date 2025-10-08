#include <memory>

#include "triangulation_bindings.h"

#include "utils/cuda_helpers.h"
#include "delaunay/delaunay.h"
#include "delaunay/triangulation_ops.h"

namespace triangulation_bindings {

std::unique_ptr<Triangulation> create_triangulation(torch::Tensor points) {
    if (points.size(-1) != 3) {
        throw std::runtime_error("points must have 3 as the last dimension");
    }
    if (points.device().type() != at::kCUDA) {
        throw std::runtime_error("points must be on CUDA device");
    }
    if (points.scalar_type() != torch::kFloat32) {
        throw std::runtime_error("points must have float32 dtype");
    }

    uint32_t num_points = points.numel() / 3;

    set_default_stream();

    return Triangulation::create_triangulation(points.data_ptr(), num_points);
}

bool rebuild(Triangulation &triangulation,
             torch::Tensor points,
             bool incremental) {
    if (points.size(-1) != 3) {
        throw std::runtime_error("points must have 3 as the last dimension");
    }
    if (points.device().type() != at::kCUDA) {
        throw std::runtime_error("points must be on CUDA device");
    }
    if (points.scalar_type() != torch::kFloat32) {
        throw std::runtime_error("points must have float32 dtype");
    }

    set_default_stream();

    return triangulation.rebuild(
        points.data_ptr(), points.numel() / 3, incremental);
}

torch::Tensor permutation(const Triangulation &triangulation) {
    const uint32_t *permutation = triangulation.permutation();
    uint32_t num_points = triangulation.num_points();

    at::TensorOptions options =
        at::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA);

    return torch::from_blob(
        const_cast<uint32_t *>(permutation), {num_points}, options);
}

torch::Tensor get_tets(const Triangulation &triangulation) {
    const IndexedTet *tets = triangulation.tets();
    uint32_t num_tets = triangulation.num_tets();

    at::TensorOptions options =
        at::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA);

    return torch::from_blob(
        const_cast<IndexedTet *>(tets), {num_tets, 4}, options);
}

torch::Tensor get_tet_adjacency(const Triangulation &triangulation) {
    const uint32_t *tet_adjacency = triangulation.tet_adjacency();
    uint32_t num_tets = triangulation.num_tets();

    at::TensorOptions options =
        at::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA);

    return torch::from_blob(
        const_cast<uint32_t *>(tet_adjacency), {num_tets, 4}, options);
}

torch::Tensor get_point_adjacency(const Triangulation &triangulation) {
    const uint32_t *point_adjacency = triangulation.point_adjacency();
    uint32_t point_adjacency_size = triangulation.point_adjacency_size();

    at::TensorOptions options =
        at::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA);

    return torch::from_blob(const_cast<uint32_t *>(point_adjacency),
                            {point_adjacency_size},
                            options);
}

torch::Tensor get_point_adjacency_offsets(const Triangulation &triangulation) {
    const uint32_t *point_adjacency_offsets =
        triangulation.point_adjacency_offsets();
    uint32_t num_points = triangulation.num_points();

    at::TensorOptions options =
        at::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA);

    return torch::from_blob(const_cast<uint32_t *>(point_adjacency_offsets),
                            {num_points + 1},
                            options);
}

torch::Tensor get_vert_to_tet(const Triangulation &triangulation) {
    const uint32_t *vert_to_tet = triangulation.vert_to_tet();
    uint32_t num_points = triangulation.num_points();

    at::TensorOptions options =
        at::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA);

    return torch::from_blob(
        const_cast<uint32_t *>(vert_to_tet), {num_points}, options);
}


void init_triangulation_bindings(py::module &module) {
    triangulation::global_cuda_init();

    py::register_exception<triangulation::TriangulationFailedError>(
        module, "TriangulationFailedError");

    py::class_<Triangulation, std::unique_ptr<Triangulation>>(module,
                                                              "Triangulation")
        .def(py::init(&create_triangulation), py::arg("points"))
        .def("tets", &get_tets)
        .def("tet_adjacency", &get_tet_adjacency)
        .def("point_adjacency", &get_point_adjacency)
        .def("point_adjacency_offsets", &get_point_adjacency_offsets)
        .def("vert_to_tet", &get_vert_to_tet)
        .def("rebuild",
             &rebuild,
             py::arg("points"),
             py::arg("incremental") = false)
        .def("permutation", &permutation);
}

} // namespace triangulation_bindings