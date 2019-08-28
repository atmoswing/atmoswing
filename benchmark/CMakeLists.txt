
# Project name
project(benchmark)

# Output path
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/benchmark)


##### SOURCE FILES

# List source files
file(GLOB_RECURSE src_benchmark_h src/*.h)
file(GLOB_RECURSE src_benchmark_cpp src/*.cpp)
file(GLOB_RECURSE src_optim_core_h ${CMAKE_SOURCE_DIR}/src/app_optimizer/core/*.h)
file(GLOB_RECURSE src_optim_core_cpp ${CMAKE_SOURCE_DIR}/src/app_optimizer/core/*.cpp)
list(APPEND src_benchmark ${src_benchmark_h} ${src_optim_core_h})
list(APPEND src_benchmark ${src_benchmark_cpp} ${src_optim_core_cpp})
list(REMOVE_ITEM src_benchmark "${CMAKE_SOURCE_DIR}/src/app_optimizer/core/AtmoswingAppOptimizer.h")
list(REMOVE_ITEM src_benchmark "${CMAKE_SOURCE_DIR}/src/app_optimizer/core/AtmoswingAppOptimizer.cpp")


# Remove eventual duplicates
list(REMOVE_DUPLICATES src_benchmark)

# Include source directories
list(APPEND inc_dirs
        "${CMAKE_SOURCE_DIR}/src/shared_base/core/"
        "${CMAKE_SOURCE_DIR}/src/shared_processing/core/"
        "${CMAKE_SOURCE_DIR}/src/app_optimizer/core/"
        "${PROJECT_SOURCE_DIR}/benchmark/src/"
        )
include_directories(${inc_dirs})


##### LIBRARIES

include(define_common_libraries)

# Google benchmark
if (MINGW OR MSYS)
    set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)
endif ()
ExternalProject_Add(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark
        UPDATE_COMMAND ""
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_DIR} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBENCHMARK_DOWNLOAD_DEPENDENCIES=1 -Dgtest_force_shared_crt=ON
        BUILD_COMMAND cmake --build . --config ${CMAKE_BUILD_TYPE}
        )

##### DECLARE EXECUTABLE

if (WIN32)
    add_executable(atmoswing-benchmark WIN32 ${src_benchmark})
elseif (UNIX AND NOT APPLE)
    add_executable(atmoswing-benchmark ${src_benchmark})
elseif (APPLE)
    add_executable(atmoswing-benchmark MACOSX_BUNDLE ${src_benchmark})
else ()
    add_executable(atmoswing-benchmark ${src_benchmark})
endif ()

# Cppcheck
if (USE_CPPCHECK)
    add_cppcheck(atmoswing-benchmark STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
endif ()


##### DEFINITIONS

if(WIN32)
    set_target_properties(atmoswing-benchmark PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
    set_target_properties(atmoswing-benchmark PROPERTIES COMPILE_DEFINITIONS "UNIT_TESTING; wxUSE_GUI=0; _CONSOLE")
else()
    set_target_properties(atmoswing-benchmark PROPERTIES COMPILE_DEFINITIONS "UNIT_TESTING; wxUSE_GUI=0")
endif()


##### LINKING

add_dependencies(atmoswing-benchmark googlebenchmark)
add_dependencies(atmoswing-benchmark asbase)
add_dependencies(atmoswing-benchmark asprocessing)
target_link_libraries(atmoswing-benchmark asbase)
target_link_libraries(atmoswing-benchmark asprocessing)
if(WIN32)
    target_link_libraries(atmoswing-benchmark debug benchmarkd debug benchmark_maind debug gtestd Shlwapi)
    target_link_libraries(atmoswing-benchmark optimized benchmark optimized benchmark_main optimized gtest Shlwapi)
else()
    target_link_libraries(atmoswing-benchmark benchmark benchmark_main gtest)
endif()
if (UNIX)
    target_link_libraries(atmoswing-benchmark pthread)
endif ()
if (USE_VLD)
    target_link_libraries(atmoswing-benchmark ${VLD_LIBRARIES})
endif ()
if (USE_CUDA)
    add_dependencies(atmoswing-benchmark ascuda)
    target_link_libraries(ascuda ${CUDA_LIBRARIES})
    target_link_libraries(atmoswing-benchmark ascuda)
    add_definitions(-DUSE_CUDA)
endif ()


##### DLLS

if (WIN32)
    file (TO_NATIVE_PATH "${CMAKE_BINARY_DIR}/*.dll" ALL_DLL_PATH_ORIGIN)
    file (TO_NATIVE_PATH "${CMAKE_BINARY_DIR}/benchmark" ALL_DLL_PATH_DESTINATION)
    add_custom_command(TARGET atmoswing-benchmark POST_BUILD
            COMMAND copy
            ${ALL_DLL_PATH_ORIGIN}
            ${ALL_DLL_PATH_DESTINATION})

    if (BUILD_VIEWER)
        add_custom_command(TARGET atmoswing-benchmark POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${GDAL_DLL_NAME}"
                "${ALL_DLL_PATH_DESTINATION}")
    endif ()
endif ()


##### Get data file

if (NOT EXISTS "${CMAKE_BINARY_DIR}/benchmark/z.nc")
    file(DOWNLOAD https://bitbucket.org/phorton/atmoswing-benchmark-files/downloads/z.nc ${CMAKE_BINARY_DIR}/benchmark/z.nc SHOW_PROGRESS)
endif ()
add_definitions(-DBENCHMARK_DATA_DIR="${CMAKE_BINARY_DIR}/benchmark")