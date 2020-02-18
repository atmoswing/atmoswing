
# Project name
project(tests)

# Output path
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/tests)


##### SOURCE FILES

# List source files
file(GLOB_RECURSE src_tests_h src/*.h)
file(GLOB_RECURSE src_tests_cpp src/*.cpp)
list(APPEND src_tests ${src_tests_h} ${src_viewer_core_h}) # Include optimization files anyway (to test the analogue method)
list(APPEND src_tests ${src_tests_cpp} ${src_viewer_core_cpp})
if (BUILD_FORECASTER)
    file(GLOB_RECURSE src_forec_core_h ${CMAKE_SOURCE_DIR}/src/app_forecaster/core/*.h)
    file(GLOB_RECURSE src_forec_core_cpp ${CMAKE_SOURCE_DIR}/src/app_forecaster/core/*.cpp)
    list(APPEND src_tests ${src_forec_core_h})
    list(APPEND src_tests ${src_forec_core_cpp})
    list(REMOVE_ITEM src_tests "${CMAKE_SOURCE_DIR}/src/app_forecaster/core/asGlobVarsForecaster.h")
    list(REMOVE_ITEM src_tests "${CMAKE_SOURCE_DIR}/src/app_forecaster/core/asGlobVarsForecaster.cpp")
    list(REMOVE_ITEM src_tests "${CMAKE_SOURCE_DIR}/src/app_forecaster/core/AtmoswingAppForecaster.h")
    list(REMOVE_ITEM src_tests "${CMAKE_SOURCE_DIR}/src/app_forecaster/core/AtmoswingAppForecaster.cpp")
else ()
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asPredictorOperNwsGfsTest.cpp")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asPredictorOperNwsGfsTest.cpp")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asPredictorOperEcmwfIfsTest.cpp")
endif ()
if (BUILD_OPTIMIZER)
    file(GLOB_RECURSE src_optim_core_h ${CMAKE_SOURCE_DIR}/src/app_optimizer/core/*.h)
    file(GLOB_RECURSE src_optim_core_cpp ${CMAKE_SOURCE_DIR}/src/app_optimizer/core/*.cpp)
    list(APPEND src_tests ${src_optim_core_h})
    list(APPEND src_tests ${src_optim_core_cpp})
else ()
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asScoreTest.cpp")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asMethodCalibratorTest.cpp")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asParametersTest.cpp")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asParametersOptimizationTest.cpp")
endif ()
if (BUILD_DOWNSCALER)
    file(GLOB_RECURSE src_downscaler_core_h ${CMAKE_SOURCE_DIR}/src/app_downscaler/core/*.h)
    file(GLOB_RECURSE src_downscaler_core_cpp ${CMAKE_SOURCE_DIR}/src/app_downscaler/core/*.cpp)
    list(APPEND src_tests ${src_downscaler_core_h})
    list(APPEND src_tests ${src_downscaler_core_cpp})
else ()
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asPredictorProjCmip5Test.cpp")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asPredictorProjCordexTest.cpp")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asParametersDownscalingTest.cpp")
endif ()
list(REMOVE_ITEM src_tests "${CMAKE_SOURCE_DIR}/src/app_viewer/core/asGlobVarsViewer.h")
list(REMOVE_ITEM src_tests "${CMAKE_SOURCE_DIR}/src/app_viewer/core/asGlobVarsViewer.cpp")
list(REMOVE_ITEM src_tests "${CMAKE_SOURCE_DIR}/src/app_downscaler/core/asGlobVarsDownscaler.h")
list(REMOVE_ITEM src_tests "${CMAKE_SOURCE_DIR}/src/app_downscaler/core/asGlobVarsDownscaler.cpp")
list(REMOVE_ITEM src_tests "${CMAKE_SOURCE_DIR}/src/app_viewer/core/AtmoswingAppViewer.h")
list(REMOVE_ITEM src_tests "${CMAKE_SOURCE_DIR}/src/app_viewer/core/AtmoswingAppViewer.cpp")
list(REMOVE_ITEM src_tests "${CMAKE_SOURCE_DIR}/src/app_optimizer/core/AtmoswingAppOptimizer.h")
list(REMOVE_ITEM src_tests "${CMAKE_SOURCE_DIR}/src/app_optimizer/core/AtmoswingAppOptimizer.cpp")
list(REMOVE_ITEM src_tests "${CMAKE_SOURCE_DIR}/src/app_downscaler/core/AtmoswingAppDownscaler.h")
list(REMOVE_ITEM src_tests "${CMAKE_SOURCE_DIR}/src/app_downscaler/core/AtmoswingAppDownscaler.cpp")

# CUDA library
if (USE_CUDA)
    file(GLOB_RECURSE src_tests_cu src/*.cu)
    file(GLOB_RECURSE src_tests_cuh src/*.cuh)
    list(APPEND src_tests_cuda ${src_tests_cu})
    list(APPEND src_tests_cuda ${src_tests_cuh})
    list(APPEND tests ${src_tests_cuh})
endif (USE_CUDA)

# Remove eventual duplicates
list(REMOVE_DUPLICATES src_tests)

# Include source directories
list(APPEND inc_dirs
        "${CMAKE_SOURCE_DIR}/src/shared_base/core/"
        "${CMAKE_SOURCE_DIR}/src/shared_processing/core/"
        "${CMAKE_SOURCE_DIR}/src/app_forecaster/core/"
        "${CMAKE_SOURCE_DIR}/src/app_viewer/core/"
        "${CMAKE_SOURCE_DIR}/src/app_optimizer/core/"
        "${CMAKE_SOURCE_DIR}/src/app_downscaler/core/"
        "${PROJECT_SOURCE_DIR}/test/src/"
        )
include_directories(${inc_dirs})


##### LIBRARIES

include(define_common_libraries)

# Google Test
if (MINGW OR MSYS)
    set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)
endif ()
ExternalProject_Add(googletest
        GIT_REPOSITORY https://github.com/google/googletest
        UPDATE_COMMAND ""
        CMAKE_ARGS -DBUILD_GMOCK=OFF -DCMAKE_INSTALL_PREFIX=${EXTERNAL_DIR} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -Dgtest_force_shared_crt=ON -std=c++11
        BUILD_COMMAND cmake --build . --config ${CMAKE_BUILD_TYPE}
        )

# CUDA
if (USE_CUDA)
    set(CUDA_SEPARABLE_COMPILATION ON)
    find_package(CUDA 9.0 REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE};-O3)
    set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};-G)
    set(CUDA_NVCC_FLAGS_RELWITHDEBINFO ${CUDA_NVCC_FLAGS_RELWITHDEBINFO};-lineinfo)
    cuda_add_library(astestscuda STATIC ${src_tests_cuda})
    target_link_libraries(astestscuda ${CUDA_cudadevrt_LIBRARY})
    target_link_libraries(astestscuda ${CUDA_cudart_static_LIBRARY})
    add_definitions(-DUSE_CUDA)
else (USE_CUDA)
    # unset for wxhgversion
    unset(CUDA_INCLUDE_DIRS CACHE)
    unset(CUDA_CUDA_LIBRARY CACHE)
endif (USE_CUDA)

##### DECLARE EXECUTABLE

if (WIN32)
    add_executable(atmoswing-tests WIN32 ${src_tests})
elseif (UNIX AND NOT APPLE)
    add_executable(atmoswing-tests ${src_tests})
elseif (APPLE)
    add_executable(atmoswing-tests MACOSX_BUNDLE ${src_tests})
else ()
    add_executable(atmoswing-tests ${src_tests})
endif ()

# Cppcheck
if (USE_CPPCHECK)
    add_cppcheck(atmoswing-tests STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
endif ()


##### DEFINITIONS

if(WIN32)
    set_target_properties(atmoswing-tests PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
    set_target_properties(atmoswing-tests PROPERTIES COMPILE_DEFINITIONS "UNIT_TESTING; wxUSE_GUI=0; _CONSOLE")
else()
    set_target_properties(atmoswing-tests PROPERTIES COMPILE_DEFINITIONS "UNIT_TESTING; wxUSE_GUI=0")
endif()


##### LINKING

add_dependencies(atmoswing-tests googletest)
add_dependencies(atmoswing-tests asbase)
add_dependencies(atmoswing-tests asprocessing)
target_link_libraries(atmoswing-tests asbase)
target_link_libraries(atmoswing-tests asprocessing)
if(WIN32)
    target_link_libraries(atmoswing-tests debug gtestd debug gtest_maind)
    target_link_libraries(atmoswing-tests optimized gtest optimized gtest_main)
else()
    target_link_libraries(atmoswing-tests gtest gtest_main)
endif()
if (UNIX)
    target_link_libraries(atmoswing-tests pthread)
endif ()
if (USE_VLD)
    target_link_libraries(atmoswing-tests ${VLD_LIBRARIES})
endif ()
if (USE_CUDA)
    add_dependencies(atmoswing-tests ascuda)
    add_dependencies(atmoswing-tests astestscuda)
    target_link_libraries(astestscuda ${CUDA_LIBRARIES})
    target_link_libraries(atmoswing-tests ascuda)
    target_link_libraries(atmoswing-tests astestscuda)
    add_definitions(-DUSE_CUDA)
endif ()


##### DLLS

if (WIN32)
    file (TO_NATIVE_PATH "${CMAKE_BINARY_DIR}/*.dll" ALL_DLL_PATH_ORIGIN)
    file (TO_NATIVE_PATH "${CMAKE_BINARY_DIR}/tests" ALL_DLL_PATH_DESTINATION)
    add_custom_command(TARGET atmoswing-tests POST_BUILD
            COMMAND copy
            ${ALL_DLL_PATH_ORIGIN}
            ${ALL_DLL_PATH_DESTINATION})

    if (BUILD_VIEWER)
        add_custom_command(TARGET atmoswing-tests POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${GDAL_DLL_NAME}"
                "${ALL_DLL_PATH_DESTINATION}")
    endif ()
endif ()


##### CTEST

set(CTEST_TEST_TIMEOUT 3600)
set(DART_TESTING_TIMEOUT 3600)
set(CTEST_OUTPUT_ON_FAILURE TRUE)
add_test(AtmoSwing-tests atmoswing-tests)
include(CTest)
enable_testing()