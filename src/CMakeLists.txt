find_package(CUDAToolkit REQUIRED)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -D_USE_MATH_DEFINES)

list(
  APPEND
  TRIANGULATION_INCLUDES
  ${CMAKE_SOURCE_DIR}/submodules/eigen)

set(TRIANGULATION_CXX_SOURCES)

list(
  APPEND
  TRIANGULATION_CUDA_SOURCES
  aabb_tree/aabb_tree.cu
  delaunay/delaunay.cu
  delaunay/sample_initial_tets.cu
  delaunay/growth_iteration.cu
  delaunay/delete_violations.cu
  delaunay/triangulation_ops.cu)

add_library(triangulation STATIC ${TRIANGULATION_CXX_SOURCES} ${TRIANGULATION_CUDA_SOURCES})
target_include_directories(triangulation PUBLIC ${TRIANGULATION_INCLUDES})

target_link_libraries(triangulation PUBLIC CUDA::cudart CUDA::cuda_driver TBB::tbb)

target_compile_options(triangulation PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -O3
                                       -Wall -Wno-unknown-pragmas> -O3)
