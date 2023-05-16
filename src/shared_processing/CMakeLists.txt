
# Project name
project(processing)


##### SOURCE FILES

# List source files
file(GLOB_RECURSE src_proc_core_h core/*.h)
file(GLOB_RECURSE src_proc_core_cpp core/*.cpp)
file(GLOB src_proc_thread_h src/shared_base/core/asThread.h)
file(GLOB src_proc_thread_cpp src/shared_base/core/asThread.cpp)
list(APPEND src_shared_processing ${src_proc_core_h} ${src_proc_thread_h})
list(APPEND src_shared_processing ${src_proc_core_cpp} ${src_proc_thread_cpp})

# CUDA library
if (USE_CUDA)
    file(GLOB_RECURSE src_proc_core_cu core/*.cu)
    file(GLOB_RECURSE src_proc_core_cuh core/*.cuh)
    list(APPEND src_cuda ${src_proc_core_cu})
    list(APPEND src_cuda ${src_proc_core_cuh})
    list(APPEND src_app_optimizer ${src_proc_core_cuh})
endif (USE_CUDA)

# Remove eventual duplicates
list(REMOVE_DUPLICATES src_shared_processing)

# Include source directories
list(APPEND inc_dirs
        "${CMAKE_SOURCE_DIR}/src/shared_base/core/"
        "${PROJECT_SOURCE_DIR}/core/"
        )
if (USE_GUI)
    list(APPEND inc_dirs
            "${CMAKE_SOURCE_DIR}/src/shared_base/gui/"
            )
endif (USE_GUI)
include_directories(${inc_dirs})

##### LIBRARIES

# CUDA
if (USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    add_library(ascuda STATIC ${src_cuda})
    target_link_libraries(ascuda CUDA::cudart_static)
    add_definitions(-DUSE_CUDA)
endif (USE_CUDA)

# Own libraries
add_library(asprocessing STATIC ${src_shared_processing})


##### DECLARE EXECUTABLE

# Cppcheck
if (USE_CPPCHECK)
    add_cppcheck(asprocessing STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
endif (USE_CPPCHECK)


##### LINKING

add_dependencies(asprocessing asbase)
target_link_libraries(asprocessing asbase)
