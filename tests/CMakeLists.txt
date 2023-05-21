
# Project name
project(tests)

# Output path
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/tests)


##### SOURCE FILES

# List source files
file(GLOB_RECURSE src_tests_h src/*.h)
file(GLOB_RECURSE src_tests_cpp src/*.cpp)
file(GLOB_RECURSE src_tests_gui_h src/gui/*.h)
file(GLOB_RECURSE src_tests_gui_cpp src/gui/*.cpp)
list(APPEND src_tests ${src_tests_h} ${src_viewer_core_h}) # Include optimization files anyway (to test the analogue method)
list(APPEND src_tests ${src_tests_cpp} ${src_viewer_core_cpp})
if (USE_GUI)
    list(APPEND src_tests ${src_tests_gui_h})
    list(APPEND src_tests ${src_tests_gui_cpp})
endif (USE_GUI)

if (BUILD_FORECASTER)
    file(GLOB_RECURSE src_forec_core_h ${CMAKE_SOURCE_DIR}/src/app_forecaster/core/*.h)
    file(GLOB_RECURSE src_forec_core_cpp ${CMAKE_SOURCE_DIR}/src/app_forecaster/core/*.cpp)
    file(GLOB_RECURSE src_forec_gui_h ${CMAKE_SOURCE_DIR}/src/app_forecaster/gui/*.h)
    file(GLOB_RECURSE src_forec_gui_cpp ${CMAKE_SOURCE_DIR}/src/app_forecaster/gui/*.cpp)
    list(APPEND src_tests ${src_forec_core_h})
    list(APPEND src_tests ${src_forec_core_cpp})
    if (USE_GUI)
        list(APPEND src_app_forecaster ${src_forec_gui_h})
        list(APPEND src_app_forecaster ${src_forec_gui_cpp})
    endif (USE_GUI)
    list(REMOVE_ITEM src_tests "${CMAKE_SOURCE_DIR}/src/app_forecaster/core/asGlobVarsForecaster.h")
    list(REMOVE_ITEM src_tests "${CMAKE_SOURCE_DIR}/src/app_forecaster/core/asGlobVarsForecaster.cpp")
    list(REMOVE_ITEM src_tests "${CMAKE_SOURCE_DIR}/src/app_forecaster/core/AtmoswingAppForecaster.h")
    list(REMOVE_ITEM src_tests "${CMAKE_SOURCE_DIR}/src/app_forecaster/core/AtmoswingAppForecaster.cpp")
else ()
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asPredictorOperCustomMeteoFvgTest.cpp")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asPredictorOperCustomVigicruesIfsTest.cpp")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asPredictorOperEcmwfIfsTest.cpp")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asPredictorOperMfArpegeTest.cpp")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asPredictorOperNwsGfsTest.cpp")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asParametersForecastingTest.cpp")
endif ()

if (BUILD_OPTIMIZER)
    file(GLOB_RECURSE src_optim_core_h ${CMAKE_SOURCE_DIR}/src/app_optimizer/core/*.h)
    file(GLOB_RECURSE src_optim_core_cpp ${CMAKE_SOURCE_DIR}/src/app_optimizer/core/*.cpp)
    file(GLOB_RECURSE src_optim_gui_h ${CMAKE_SOURCE_DIR}/src/app_optimizer/gui/*.h)
    file(GLOB_RECURSE src_optim_gui_cpp ${CMAKE_SOURCE_DIR}/src/app_optimizer/gui/*.cpp)
    list(APPEND src_tests ${src_optim_core_h})
    list(APPEND src_tests ${src_optim_core_cpp})
    if (USE_GUI)
        list(APPEND src_app_optimizer ${src_optim_gui_h})
        list(APPEND src_app_optimizer ${src_optim_gui_cpp})
    endif (USE_GUI)
else ()
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asScoreTest.cpp")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asMethodCalibratorTest.cpp")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asParametersTest.cpp")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asParametersOptimizationTest.cpp")
endif ()

if (BUILD_DOWNSCALER)
    file(GLOB_RECURSE src_downscaler_core_h ${CMAKE_SOURCE_DIR}/src/app_downscaler/core/*.h)
    file(GLOB_RECURSE src_downscaler_core_cpp ${CMAKE_SOURCE_DIR}/src/app_downscaler/core/*.cpp)
    file(GLOB_RECURSE src_downscaler_gui_h ${CMAKE_SOURCE_DIR}/src/app_downscaler/gui/*.h)
    file(GLOB_RECURSE src_downscaler_gui_cpp ${CMAKE_SOURCE_DIR}/src/app_downscaler/gui/*.cpp)
    list(APPEND src_tests ${src_downscaler_core_h})
    list(APPEND src_tests ${src_downscaler_core_cpp})
    if (USE_GUI)
        list(APPEND src_tests ${src_downsc_gui_h})
        list(APPEND src_tests ${src_downsc_gui_cpp})
    endif (USE_GUI)
else ()
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asPredictorProjCmip5Test.cpp")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asPredictorProjCordexTest.cpp")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asParametersDownscalingTest.cpp")
endif ()

if (BUILD_VIEWER)
    file(GLOB_RECURSE src_viewer_core_h ${CMAKE_SOURCE_DIR}/src/app_viewer/core/*.h)
    file(GLOB_RECURSE src_viewer_core_cpp ${CMAKE_SOURCE_DIR}/src/app_viewer/core/*.cpp)
    file(GLOB_RECURSE src_viewer_gui_h ${CMAKE_SOURCE_DIR}/src/app_viewer/gui/*.h)
    file(GLOB_RECURSE src_viewer_gui_cpp ${CMAKE_SOURCE_DIR}/src/app_viewer/gui/*.cpp)
    list(APPEND src_tests ${src_viewer_core_h})
    list(APPEND src_tests ${src_viewer_core_cpp})
    if (USE_GUI)
        list(APPEND src_tests ${src_viewer_gui_h})
        list(APPEND src_tests ${src_viewer_gui_cpp})
    endif (USE_GUI)
else ()
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/asFrameViewerTest.cpp")
endif ()

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
    option(TEST_CUDA "Should CUDA implementation be tested ?" ON)
    file(GLOB_RECURSE src_tests_cu src/*.cu)
    file(GLOB_RECURSE src_tests_cuh src/*.cuh)
    list(APPEND src_tests_cuda ${src_tests_cu})
    list(APPEND src_tests_cuda ${src_tests_cuh})
    list(APPEND tests ${src_tests_cuh})
    if (TEST_CUDA)
        add_definitions(-DTEST_CUDA)
    endif (TEST_CUDA)
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
if (USE_GUI)
    list(APPEND inc_dirs
            "${CMAKE_SOURCE_DIR}/src/shared_base/gui/"
            "${CMAKE_SOURCE_DIR}/src/shared_base/libs/awxled/"
            "${CMAKE_SOURCE_DIR}/src/shared_processing/gui/"
            "${CMAKE_SOURCE_DIR}/src/app_forecaster/gui/"
            "${CMAKE_SOURCE_DIR}/src/app_optimizer/gui/"
            "${CMAKE_SOURCE_DIR}/src/app_downscaler/gui/"
            "${CMAKE_SOURCE_DIR}/src/app_viewer/gui/"
            "${CMAKE_SOURCE_DIR}/src/app_viewer/libs/wxmathplot/"
            "${CMAKE_SOURCE_DIR}/src/app_viewer/libs/wxplotctrl/"
            "${CMAKE_SOURCE_DIR}/src/app_viewer/libs/wxplotctrl/include/"
            "${CMAKE_SOURCE_DIR}/src/app_viewer/libs/wxthings/include/"
            )
endif (USE_GUI)
include_directories(${inc_dirs})

if (USE_GUI)
    include_directories("${CMAKE_SOURCE_DIR}/src/shared_base/gui/")
endif ()


##### LIBRARIES

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
endif (USE_CUDA)

if (USE_GUI)
    # vroomgis
    FetchContent_MakeAvailable(vroomgis)
    include_directories(${vroomgis_SOURCE_DIR}/vroomgis/src)
    include_directories(${vroomgis_SOURCE_DIR}/vroomgis/art)
endif (USE_GUI)

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

if (WIN32)
    set_target_properties(atmoswing-tests PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
    if (USE_GUI)
        set_target_properties(atmoswing-tests PROPERTIES COMPILE_DEFINITIONS "UNIT_TESTING; USE_GUI=1; _CONSOLE")
    else (USE_GUI)
        set_target_properties(atmoswing-tests PROPERTIES COMPILE_DEFINITIONS "UNIT_TESTING; USE_GUI=0; _CONSOLE")
    endif (USE_GUI)
else ()
    if (USE_GUI)
        set_target_properties(atmoswing-tests PROPERTIES COMPILE_DEFINITIONS "UNIT_TESTING; USE_GUI=1")
    else (USE_GUI)
        set_target_properties(atmoswing-tests PROPERTIES COMPILE_DEFINITIONS "UNIT_TESTING; USE_GUI=0")
    endif (USE_GUI)
endif ()


##### LINKING

add_dependencies(atmoswing-tests asbase)
add_dependencies(atmoswing-tests asprocessing)
target_link_libraries(atmoswing-tests asbase)
target_link_libraries(atmoswing-tests asprocessing)
target_link_libraries(atmoswing-tests CONAN_PKG::gtest)
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
if (USE_GUI)
    add_dependencies(atmoswing-tests vroomgis)
    add_dependencies(atmoswing-tests wxplotctrl)
    target_link_libraries(atmoswing-tests vroomgis)
    target_link_libraries(atmoswing-tests wxplotctrl)
    target_link_libraries(atmoswing-tests CONAN_PKG::gdal)
endif (USE_GUI)

##### CTEST

set(CTEST_TEST_TIMEOUT 3600)
set(DART_TESTING_TIMEOUT 3600)
set(CTEST_OUTPUT_ON_FAILURE TRUE)
add_test(AtmoSwing-tests atmoswing-tests)
include(CTest)
enable_testing()