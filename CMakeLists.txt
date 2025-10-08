cmake_minimum_required(VERSION 3.27)
project(RadFoam VERSION 1.0.0)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib")
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/lib")
endif()

if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_CUDA_COMPILER "$ENV{CONDA_PREFIX}/bin/nvcc")
    set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CONDA_PREFIX}/targets/x86_64-linux")
endif()

cmake_policy(SET CMP0060 NEW)

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GPU_DEBUG
    ON
    CACHE BOOL "Enable GPU debug features")
add_definitions(-DGPU_DEBUG=$<BOOL:${GPU_DEBUG}>)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE STRING "Build type")
endif()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/submodules"
     "${CMAKE_SOURCE_DIR}/submodules")

find_package(
  Python3
  COMPONENTS Interpreter Development.Module
  REQUIRED)

find_package(pybind11 REQUIRED)

if(NOT Torch_DIR)
  set(Torch_DIR ${Python3_SITELIB}/torch/share/cmake/Torch)
endif()

find_package(Torch REQUIRED)

if(DEFINED ENV{CONDA_PREFIX})
    include_directories(SYSTEM "$ENV{CONDA_PREFIX}/include")
endif()

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS
             "${TORCH_INSTALL_PREFIX}/lib"
             "${Python3_SITELIB}/torch/lib"
             NO_DEFAULT_PATH)

find_package(TBB GLOBAL)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX
      "${CMAKE_SOURCE_DIR}/triangulation"
      CACHE PATH "..." FORCE)
endif()

set(TRIANGULATION_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

add_subdirectory(src)
add_subdirectory(torch_bindings)
