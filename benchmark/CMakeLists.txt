
# Project name
project(benchmark)

# Output path
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/benchmark)


##### SOURCE FILES

# List source files
file(GLOB src_benchmark_h src/*.h)
file(GLOB src_benchmark_cpp src/*.cpp)
file(GLOB src_optim_core_h ${CMAKE_SOURCE_DIR}/src/app_optimizer/core/*.h)
file(GLOB src_optim_core_cpp ${CMAKE_SOURCE_DIR}/src/app_optimizer/core/*.cpp)
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


##### LIBRARIES

# Google benchmark
if (MINGW OR MSYS)
    set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)
endif ()


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

target_include_directories(atmoswing-benchmark PRIVATE ${inc_dirs})

# Cppcheck
if (USE_CPPCHECK)
    add_cppcheck(atmoswing-benchmark STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
endif ()


##### DEFINITIONS

# UNIT_TESTING is target-specific. USE_GUI / version macros come from atmoswing_compile_options.
target_compile_definitions(atmoswing-benchmark PRIVATE UNIT_TESTING)
if (WIN32)
    set_target_properties(atmoswing-benchmark PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
    target_compile_definitions(atmoswing-benchmark PRIVATE _CONSOLE)
endif ()


##### LINKING

add_dependencies(atmoswing-benchmark asbase)
add_dependencies(atmoswing-benchmark asprocessing)
target_link_libraries(atmoswing-benchmark PRIVATE asbase asprocessing)
if(WIN32)
    target_link_libraries(atmoswing-benchmark PRIVATE debug benchmarkd debug gtest Shlwapi)
    target_link_libraries(atmoswing-benchmark PRIVATE optimized benchmark optimized gtest Shlwapi)
else()
    target_link_libraries(atmoswing-benchmark PRIVATE benchmark benchmark_main gtest)
endif()
if (UNIX)
    target_link_libraries(atmoswing-benchmark PRIVATE pthread)
endif ()
if (USE_CUDA)
    add_dependencies(atmoswing-benchmark ascuda)
    target_link_libraries(ascuda PRIVATE ${CUDA_LIBRARIES})
    target_link_libraries(atmoswing-benchmark PRIVATE ascuda)
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
target_compile_definitions(atmoswing-benchmark PRIVATE BENCHMARK_DATA_DIR="${CMAKE_BINARY_DIR}/benchmark")