
# Project name
project(processing)


##### DEPENDENCIES

find_package(PROJ CONFIG REQUIRED)


##### SOURCE FILES

# List source files
file(GLOB src_proc_core_h core/*.h)
file(GLOB src_proc_core_cpp core/*.cpp)
file(GLOB src_proc_thread_h src/shared_base/core/asThread.h)
file(GLOB src_proc_thread_cpp src/shared_base/core/asThread.cpp)
list(APPEND src_shared_processing ${src_proc_core_h} ${src_proc_thread_h})
list(APPEND src_shared_processing ${src_proc_core_cpp} ${src_proc_thread_cpp})

# CUDA library
if (USE_CUDA)
    file(GLOB src_proc_core_cu core/*.cu)
    file(GLOB src_proc_core_cuh core/*.cuh)
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

##### LIBRARIES

# Own libraries
add_library(asprocessing STATIC ${src_shared_processing})

# PUBLIC include dirs and link so consumers (apps, tests, benchmark) inherit transitively.
target_include_directories(asprocessing PUBLIC ${inc_dirs})

# CUDA
if (USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(ascuda STATIC ${src_cuda})
    target_include_directories(ascuda PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(ascuda PRIVATE CUDA::cudart_static atmoswing_compile_options)
    # USE_CUDA is needed by both asprocessing sources and downstream apps/tests.
    target_compile_definitions(asprocessing PUBLIC USE_CUDA)
    target_compile_definitions(ascuda PRIVATE USE_CUDA)
endif (USE_CUDA)


##### DECLARE EXECUTABLE

# Cppcheck
if (USE_CPPCHECK)
    add_cppcheck(asprocessing STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
endif (USE_CPPCHECK)


##### LINKING

# asbase PUBLIC-links to atmoswing_compile_options, so consumers of asprocessing inherit
# the shared compile options transitively through this chain.
add_dependencies(asprocessing asbase)
target_link_libraries(asprocessing PUBLIC asbase PROJ::proj)
