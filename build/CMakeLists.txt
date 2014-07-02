cmake_minimum_required(VERSION 2.8)


#### PROJECT

if (UNIX)
    project(atmoswing)
else (UNIX)
    project(AtmoSwing)
endif (UNIX)

# Version number
set(VERSION_MAJOR "1")
set(VERSION_MINOR "3")
set(VERSION_PATCH "1")

# Choice of the targets
set(BUILD_FORECASTER OFF CACHE BOOL "Do you want to build AtmoSwing Forecaster ?" )
set(BUILD_VIEWER OFF CACHE BOOL "Do you want to build AtmoSwing Viewer ?" )
set(BUILD_CALIBRATOR OFF CACHE BOOL "Do you want to build AtmoSwing Calibrator ?" )
if (BUILD_FORECASTER OR BUILD_CALIBRATOR)
    set(BUILD_TESTS ON CACHE BOOL "Do you want to build the tests (recommended) ?" )
    mark_as_advanced(CLEAR BUILD_TESTS)
else (BUILD_FORECASTER OR BUILD_CALIBRATOR)
    set(BUILD_TESTS OFF)
    mark_as_advanced(BUILD_TESTS)
endif (BUILD_FORECASTER OR BUILD_CALIBRATOR)

if (NOT BUILD_FORECASTER AND NOT BUILD_VIEWER AND NOT BUILD_CALIBRATOR)
    message(FATAL_ERROR "Please select one or multiple target(s) to build.")
endif (NOT BUILD_FORECASTER AND NOT BUILD_VIEWER AND NOT BUILD_CALIBRATOR)

# Output path
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BUILD_TYPE})

# Enable Visual Leak Detector
if (WIN32)
    set(USE_VLD OFF CACHE BOOL "Sould we use Visual Leak Detector (https://vld.codeplex.com) ?" )
else (WIN32)
    set(USE_VLD OFF)
endif (WIN32)

# Enable Cppcheck
if (WIN32)
    set(USE_CPPCHECK OFF CACHE BOOL "Sould we use Cppcheck (http://cppcheck.sourceforge.net/) ?" )
else (WIN32)
    set(USE_CPPCHECK OFF)
endif (WIN32)

# Enable GUIs
if (BUILD_FORECASTER OR BUILD_CALIBRATOR AND NOT BUILD_VIEWER)
    set(USE_GUI OFF CACHE BOOL "Sould we build the Forecaster / Calibrator with a GUI ?" )
else (BUILD_FORECASTER OR BUILD_CALIBRATOR AND NOT BUILD_VIEWER)
    set(USE_GUI ON)
endif (BUILD_FORECASTER OR BUILD_CALIBRATOR AND NOT BUILD_VIEWER)

# Enable CUDA
if (BUILD_CALIBRATOR)
    set(USE_CUDA OFF CACHE BOOL "Sould we compile with CUDA GPU support (not stable yet) ?" )
    mark_as_advanced(CLEAR USE_CUDA)
else (BUILD_CALIBRATOR)
    set(USE_CUDA OFF)
    mark_as_advanced(USE_CUDA)
endif (BUILD_CALIBRATOR)

# Addition of custom modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


#### SOURCE FILES LIST

# Folder shared_base
file(GLOB_RECURSE src_base_core_h ../src/shared_base/core/*.h)
file(GLOB_RECURSE src_base_core_cpp ../src/shared_base/core/*.cpp)
file(GLOB_RECURSE src_base_gui_h ../src/shared_base/gui/*.h)
file(GLOB_RECURSE src_base_gui_cpp ../src/shared_base/gui/*.cpp)
file(GLOB_RECURSE src_base_img_h ../src/shared_base/gui/img/*.h)
file(GLOB_RECURSE src_base_img_cpp ../src/shared_base/gui/img/*.cpp)
file(GLOB_RECURSE src_lib_awxled_h ../src/shared_base/libs/awxled/*.h)
file(GLOB_RECURSE src_lib_awxled_cpp ../src/shared_base/libs/awxled/*.cpp)
file(GLOB_RECURSE src_lib_ticpp_h ../src/shared_base/libs/ticpp/*.h)
file(GLOB_RECURSE src_lib_ticpp_cpp ../src/shared_base/libs/ticpp/*.cpp)
list(APPEND src_shared_base ${src_base_core_h} ${src_lib_ticpp_h})
list(APPEND src_shared_base ${src_base_core_cpp} ${src_lib_ticpp_cpp})
if (USE_GUI)
    list(APPEND src_shared_base ${src_base_gui_h} ${src_base_img_h} ${src_lib_awxled_h})
    list(APPEND src_shared_base ${src_base_gui_cpp} ${src_base_img_cpp} ${src_lib_awxled_cpp})
endif (USE_GUI)
if (NOT BUILD_FORECASTER AND NOT BUILD_VIEWER)
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/../src/shared_base/core/asInternet.h")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/../src/shared_base/core/asInternet.cpp")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/../src/shared_base/core/asThreadInternetDownload.h")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/../src/shared_base/core/asThreadInternetDownload.cpp")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/../src/shared_base/core/asFileGrib2.h")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/../src/shared_base/core/asFileGrib2.cpp")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/../src/shared_base/core/asDataPredictorRealtime.h")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/../src/shared_base/core/asDataPredictorRealtime.cpp")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/../src/shared_base/core/asDataPredictorRealtimeGfsForecast.h")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/../src/shared_base/core/asDataPredictorRealtimeGfsForecast.cpp")
endif (NOT BUILD_FORECASTER AND NOT BUILD_VIEWER)

# Folder shared_processing
file(GLOB_RECURSE src_proc_core_h ../src/shared_processing/core/*.h)
file(GLOB_RECURSE src_proc_core_cpp ../src/shared_processing/core/*.cpp)
file(GLOB src_proc_thread_h ../src/shared_base/core/asThread.h)
file(GLOB src_proc_thread_cpp ../src/shared_base/core/asThread.cpp)
list(APPEND src_shared_processing ${src_proc_core_h} ${src_proc_thread_h})
list(APPEND src_shared_processing ${src_proc_core_cpp} ${src_proc_thread_cpp})

# Folder app_forecaster
file(GLOB_RECURSE src_forec_app_h ../src/app_forecaster/app/*.h)
file(GLOB_RECURSE src_forec_app_cpp ../src/app_forecaster/app/*.cpp)
file(GLOB_RECURSE src_forec_core_h ../src/app_forecaster/core/*.h)
file(GLOB_RECURSE src_forec_core_cpp ../src/app_forecaster/core/*.cpp)
file(GLOB_RECURSE src_forec_gui_h ../src/app_forecaster/gui/*.h)
file(GLOB_RECURSE src_forec_gui_cpp ../src/app_forecaster/gui/*.cpp)
list(APPEND src_app_forecaster ${src_forec_app_h} ${src_forec_core_h})
list(APPEND src_app_forecaster ${src_forec_app_cpp} ${src_forec_core_cpp})
if (USE_GUI)
    list(APPEND src_app_forecaster ${src_forec_gui_h})
    list(APPEND src_app_forecaster ${src_forec_gui_cpp})
else (USE_GUI)
    list(REMOVE_ITEM src_app_forecaster "${PROJECT_SOURCE_DIR}/../src/app_forecaster/app/AtmoswingMainForecaster.h")
    list(REMOVE_ITEM src_app_forecaster "${PROJECT_SOURCE_DIR}/../src/app_forecaster/app/AtmoswingMainForecaster.cpp")
endif (USE_GUI)

# Folder app_viewer
file(GLOB_RECURSE src_viewer_app_h ../src/app_viewer/app/*.h)
file(GLOB_RECURSE src_viewer_app_cpp ../src/app_viewer/app/*.cpp)
file(GLOB_RECURSE src_viewer_core_h ../src/app_viewer/core/*.h)
file(GLOB_RECURSE src_viewer_core_cpp ../src/app_viewer/core/*.cpp)
file(GLOB_RECURSE src_viewer_gui_h ../src/app_viewer/gui/*.h)
file(GLOB_RECURSE src_viewer_gui_cpp ../src/app_viewer/gui/*.cpp)
file(GLOB_RECURSE src_lib_wxmathplot_h ../src/app_viewer/libs/wxmathplot/*.h)
file(GLOB_RECURSE src_lib_wxmathplot_cpp ../src/app_viewer/libs/wxmathplot/*.cpp)
file(GLOB_RECURSE src_lib_wxplotctrl_h ../src/app_viewer/libs/wxplotctrl/src/*.h)
file(GLOB_RECURSE src_lib_wxplotctrl_hh ../src/app_viewer/libs/wxplotctrl/src/*.hh)
file(GLOB_RECURSE src_lib_wxplotctrl_cpp ../src/app_viewer/libs/wxplotctrl/src/*.cpp)
file(GLOB_RECURSE src_lib_wxplotctrl_c ../src/app_viewer/libs/wxplotctrl/src/*.c)
file(GLOB_RECURSE src_lib_wxthings_cpp ../src/app_viewer/libs/wxthings/src/*.cpp)
list(APPEND src_app_viewer ${src_viewer_app_h} ${src_viewer_core_h} ${src_viewer_gui_h} ${src_lib_wxmathplot_h} ${src_lib_wxplotctrl_h} ${src_lib_wxplotctrl_hh})
list(APPEND src_app_viewer ${src_viewer_app_cpp} ${src_viewer_core_cpp} ${src_viewer_gui_cpp} ${src_lib_wxmathplot_cpp} ${src_lib_wxplotctrl_cpp} ${src_lib_wxplotctrl_c} ${src_lib_wxthings_cpp})

# Folder app_calibrator
file(GLOB_RECURSE src_calib_app_h ../src/app_calibrator/app/*.h)
file(GLOB_RECURSE src_calib_app_cpp ../src/app_calibrator/app/*.cpp)
file(GLOB_RECURSE src_calib_core_h ../src/app_calibrator/core/*.h)
file(GLOB_RECURSE src_calib_core_cpp ../src/app_calibrator/core/*.cpp)
file(GLOB_RECURSE src_calib_gui_h ../src/app_calibrator/gui/*.h)
file(GLOB_RECURSE src_calib_gui_cpp ../src/app_calibrator/gui/*.cpp)
list(APPEND src_app_calibrator ${src_calib_app_h} ${src_calib_core_h})
list(APPEND src_app_calibrator ${src_calib_app_cpp} ${src_calib_core_cpp})
if (USE_GUI)
    list(APPEND src_app_calibrator ${src_calib_gui_h})
    list(APPEND src_app_calibrator ${src_calib_gui_cpp})
else (USE_GUI)
    list(REMOVE_ITEM src_app_calibrator "${PROJECT_SOURCE_DIR}/../src/app_calibrator/app/AtmoswingMainCalibrator.h")
    list(REMOVE_ITEM src_app_calibrator "${PROJECT_SOURCE_DIR}/../src/app_calibrator/app/AtmoswingMainCalibrator.cpp")
endif (USE_GUI)

# Folder test
file(GLOB_RECURSE src_tests_h ../test/src/*.h)
file(GLOB_RECURSE src_tests_cpp ../test/src/*.cpp)
list(APPEND src_tests ${src_tests_h} ${src_viewer_core_h} ${src_calib_core_h}) # Include calibration files anyway (to test the analogue method)
list(APPEND src_tests ${src_tests_cpp} ${src_viewer_core_cpp} ${src_calib_core_cpp})
if (BUILD_FORECASTER)
    list(APPEND src_tests ${src_forec_core_h})
    list(APPEND src_tests ${src_forec_core_cpp})
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/../src/app_forecaster/core/asGlobVarsForecaster.h")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/../src/app_forecaster/core/asGlobVarsForecaster.cpp")
else (BUILD_FORECASTER)
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/../test/src/asDataPredictorRealtimeGeneralGridTest.cpp")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/../test/src/asDataPredictorRealtimeRegularGridTest.cpp")
endif (BUILD_FORECASTER)
list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/../src/app_viewer/core/asGlobVarsViewer.h")
list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/../src/app_viewer/core/asGlobVarsViewer.cpp")

# CUDA library
if (USE_CUDA)
    file(GLOB_RECURSE src_proc_core_cu ../src/shared_processing/core/*.cu)
    file(GLOB_RECURSE src_proc_core_cuh ../src/shared_processing/core/*.cuh)
    list(APPEND src_cuda ${src_proc_core_cu})
    list(APPEND src_cuda ${src_proc_core_cuh})
    list(APPEND src_app_calibrator ${src_proc_core_cuh})
    if (WIN32)
        # Bug with VS 2012 and CUDA in previous versions
        cmake_minimum_required(VERSION 2.8.12.2)
    endif (WIN32)
endif (USE_CUDA)

# Remove eventual duplicates
list(REMOVE_DUPLICATES src_shared_base)
list(REMOVE_DUPLICATES src_shared_processing)
list(REMOVE_DUPLICATES src_app_forecaster)
list(REMOVE_DUPLICATES src_app_viewer)
list(REMOVE_DUPLICATES src_app_calibrator)
list(REMOVE_DUPLICATES src_tests)


#### INCLUDE DIRECTORIES

list(APPEND inc_dirs 
    "${PROJECT_SOURCE_DIR}/../src/shared_base/core/"
    "${PROJECT_SOURCE_DIR}/../src/shared_base/libs/ticpp/"
    )

if (BUILD_FORECASTER)
    list(APPEND inc_dirs 
        "${PROJECT_SOURCE_DIR}/../src/shared_processing/core/"
        "${PROJECT_SOURCE_DIR}/../src/app_forecaster/app/"
        "${PROJECT_SOURCE_DIR}/../src/app_forecaster/core/"
        )
    if (USE_GUI)
        list(APPEND inc_dirs 
            "${PROJECT_SOURCE_DIR}/../src/shared_base/gui/"
            "${PROJECT_SOURCE_DIR}/../src/shared_base/gui/img/"
            "${PROJECT_SOURCE_DIR}/../src/shared_base/libs/awxled/"
            "${PROJECT_SOURCE_DIR}/../src/app_forecaster/gui/"
            )
    endif (USE_GUI)
endif (BUILD_FORECASTER)

if (BUILD_VIEWER)
    list(APPEND inc_dirs 
        "${PROJECT_SOURCE_DIR}/../src/shared_base/gui/"
        "${PROJECT_SOURCE_DIR}/../src/shared_base/gui/img/"
        "${PROJECT_SOURCE_DIR}/../src/shared_base/libs/awxled/"
        "${PROJECT_SOURCE_DIR}/../src/app_viewer/app/"
        "${PROJECT_SOURCE_DIR}/../src/app_viewer/core/"
        "${PROJECT_SOURCE_DIR}/../src/app_viewer/gui/"
        "${PROJECT_SOURCE_DIR}/../src/app_viewer/libs/wxmathplot/"
        "${PROJECT_SOURCE_DIR}/../src/app_viewer/libs/wxplotctrl/"
        "${PROJECT_SOURCE_DIR}/../src/app_viewer/libs/wxplotctrl/include/"
        "${PROJECT_SOURCE_DIR}/../src/app_viewer/libs/wxthings/include/"
        )
endif (BUILD_VIEWER)

if (BUILD_CALIBRATOR)
    list(APPEND inc_dirs 
        "${PROJECT_SOURCE_DIR}/../src/shared_processing/core/"
        "${PROJECT_SOURCE_DIR}/../src/app_calibrator/app/"
        "${PROJECT_SOURCE_DIR}/../src/app_calibrator/core/"
        )
    if (USE_GUI)
        list(APPEND inc_dirs 
            "${PROJECT_SOURCE_DIR}/../src/shared_base/gui/"
            "${PROJECT_SOURCE_DIR}/../src/shared_base/gui/img/"
            "${PROJECT_SOURCE_DIR}/../src/shared_base/libs/awxled/"
            "${PROJECT_SOURCE_DIR}/../src/app_calibrator/gui/"
            )
    endif (USE_GUI)
endif (BUILD_CALIBRATOR)

if (BUILD_TESTS)
    list(APPEND inc_dirs 
        "${PROJECT_SOURCE_DIR}/../test/src/"
        "${PROJECT_SOURCE_DIR}/../src/shared_processing/core/"
        "${PROJECT_SOURCE_DIR}/../src/app_forecaster/core/"
        "${PROJECT_SOURCE_DIR}/../src/app_viewer/core/"
        "${PROJECT_SOURCE_DIR}/../src/app_calibrator/core/"
        )
endif (BUILD_TESTS)

include_directories(${inc_dirs})


#### LIBRARIES

# Own libraries
if (WIN32)
    set(SHARED_BASE_LIBRARY lib_shared_base)
    set(SHARED_PROCESSING_LIBRARY lib_shared_processing)
else (WIN32)
    set(SHARED_BASE_LIBRARY asbase)
    set(SHARED_PROCESSING_LIBRARY asprocessing)
endif (WIN32)
add_library(${SHARED_BASE_LIBRARY} STATIC ${src_shared_base})
if (BUILD_FORECASTER OR BUILD_CALIBRATOR)
    add_library(${SHARED_PROCESSING_LIBRARY} STATIC ${src_shared_processing})
endif (BUILD_FORECASTER OR BUILD_CALIBRATOR)

# Provided libraries
if (WIN32)
    set(USE_PROVIDED_LIBRARIES ON CACHE BOOL "Use the libraries downloaded from https://bitbucket.org/atmoswing/atmoswing")
    if(USE_PROVIDED_LIBRARIES)
        set(USE_PROVIDED_LIBRARIES_PATH CACHE PATH "Path to the libraries downloaded from https://bitbucket.org/atmoswing/atmoswing")
        if ("${USE_PROVIDED_LIBRARIES_PATH}" STREQUAL "")
            message(FATAL_ERROR "Please provide the path to the downloaded libraries, or disable the option USE_PROVIDED_LIBRARIES.")
        else()
            set(CUSTOM_LIBRARY_PATH "${USE_PROVIDED_LIBRARIES_PATH}")
            file(GLOB sub-dir ${CUSTOM_LIBRARY_PATH}/*)
            foreach(dir ${sub-dir})
                if(IS_DIRECTORY ${dir})
                    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};${dir})
                endif()
            endforeach()
        endif()
    endif(USE_PROVIDED_LIBRARIES)
endif (WIN32)

# WxWidgets (adv lib nedded for the caldendar widget)
mark_as_advanced(wxWidgets_wxrc_EXECUTABLE)
if (USE_GUI)
    find_package(wxWidgets COMPONENTS core base adv REQUIRED)
else (USE_GUI)
    find_package(wxWidgets COMPONENTS base REQUIRED)
endif (USE_GUI)
include( "${wxWidgets_USE_FILE}" )
include_directories(${wxWidgets_INCLUDE_DIRS})

# NetCDF (has to be before GDAL)
mark_as_advanced(CLEAR NetCDF_INCLUDE_DIRECTORIES)
mark_as_advanced(CLEAR NetCDF_C_LIBRARY)
find_package(NetCDF REQUIRED)
include_directories(${NetCDF_INCLUDE_DIRECTORIES})

# libcURL
if (BUILD_FORECASTER OR BUILD_VIEWER)
    mark_as_advanced(CLEAR CURL_INCLUDE_DIR)
    mark_as_advanced(CLEAR CURL_LIBRARY)
    find_package(CURL REQUIRED)
    include_directories(${CURL_INCLUDE_DIRS})	
else (BUILD_FORECASTER OR BUILD_VIEWER)
	# unset for wxhgversion
	unset(CURL_INCLUDE_DIR CACHE)
	unset(CURL_LIBRARY CACHE)
endif (BUILD_FORECASTER OR BUILD_VIEWER)

# Proj.4
find_package(Proj4 REQUIRED)
include_directories(${PROJ4_INCLUDE_DIR})

# GDAL
if (BUILD_FORECASTER OR BUILD_VIEWER)
    find_package(GDAL REQUIRED)
    include_directories(${GDAL_INCLUDE_DIRS})
else (BUILD_FORECASTER OR BUILD_VIEWER)
	# unset for wxhgversion
	unset(GDAL_INCLUDE_DIR CACHE)
	unset(GDAL_LIBRARY CACHE)
endif (BUILD_FORECASTER OR BUILD_VIEWER)

# Eigen
mark_as_advanced(CLEAR EIGEN3_INCLUDE_DIR)
find_package(Eigen3 3.0.0 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# vroomgis
if (BUILD_VIEWER)
    mark_as_advanced(SEARCH_GDAL)
    mark_as_advanced(SEARCH_GEOS)
    mark_as_advanced(SEARCH_GIS_LIB_PATH)
    mark_as_advanced(SEARCH_VROOMGIS_LIBS)
    mark_as_advanced(SEARCH_VROOMGIS_WXWIDGETS)
    mark_as_advanced(SQLITE_INCLUDE_DIR)
    mark_as_advanced(SQLITE_PATH)
    mark_as_advanced(wxWIDGETS_USING_SVN)
    include("../src/app_viewer/libs/vroomgis/vroomgis/build/cmake/Use_vroomGISlib.cmake")
    link_libraries(${wxWidgets_LIBRARIES})
endif (BUILD_VIEWER)

# wxhgversion
if (USE_GUI)
    set(USE_WXHGVERSION 1)
    mark_as_advanced(USE_WXHGVERSION)
    include("../src/shared_base/libs/wxhgversion/build/use_wxhgversion.cmake")
else (USE_GUI)
    set(USE_WXHGVERSION 0)
    mark_as_advanced(USE_WXHGVERSION)
endif (USE_GUI)

# CUDA
if (USE_CUDA)
    find_package(CUDA 5.0 REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_20;-use_fast_math)
    set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE};-O3)
    set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};-G)
    set(CUDA_NVCC_FLAGS_RELWITHDEBINFO ${CUDA_NVCC_FLAGS_RELWITHDEBINFO};-lineinfo)
    cuda_add_library(lib_cuda_functions STATIC ${src_cuda})
else (USE_CUDA)
	# unset for wxhgversion
	unset(CUDA_INCLUDE_DIRS CACHE)
	unset(CUDA_CUDA_LIBRARY CACHE)
endif (USE_CUDA)

# UnitTest++
if (BUILD_TESTS)
    include(cmake/FindUnitTest++.cmake)
    include_directories(${UNITTEST++_INCLUDE_DIR})
    mark_as_advanced(UnitTest++_DIR)
else (BUILD_TESTS)
	# unset for wxhgversion
	unset(UNITTEST++_INCLUDE_DIR CACHE)
	unset(UNITTEST++_LIBRARY_DEBUG CACHE)
	unset(UNITTEST++_LIBRARY_RELEASE CACHE)
endif (BUILD_TESTS)

# Visual Leak Detector
if (USE_VLD)
    find_package(VLD)
    include_directories(${VLD_INCLUDE_DIRS})
else (USE_VLD)
	# unset for wxhgversion
	unset(VLD_INCLUDE_DIR CACHE)
	unset(VLD_LIBRARY CACHE)
	unset(VLD_LIBRARY_DEBUG CACHE)
	unset(VLD_LIBRARY_DIR CACHE)
	unset(VLD_ROOT_DIR CACHE)
endif (USE_VLD)

# Cppcheck
if (USE_CPPCHECK)
	include(cmake/Findcppcheck.cmake)
	include(cmake/CppcheckTargets.cmake)
endif (USE_CPPCHECK)


#### DECLARE EXECUTABLE

if (WIN32)

    if (USE_GUI)
        if (BUILD_FORECASTER)
            add_executable (atmoswing-forecaster WIN32 ${src_app_forecaster} ${PROJECT_SOURCE_DIR}/../src/shared_base/gui/resource.rc)
        endif (BUILD_FORECASTER)
        if (BUILD_CALIBRATOR)
            add_executable (atmoswing-calibrator WIN32 ${src_app_calibrator} ${PROJECT_SOURCE_DIR}/../src/shared_base/gui/resource.rc)
        endif (BUILD_CALIBRATOR)
    else (USE_GUI)
        if (BUILD_FORECASTER)
            add_executable (atmoswing-forecaster ${src_app_forecaster})
        endif (BUILD_FORECASTER)
        if (BUILD_CALIBRATOR)
            add_executable (atmoswing-calibrator ${src_app_calibrator})
        endif (BUILD_CALIBRATOR)
    endif (USE_GUI)
    if (BUILD_VIEWER)
        add_executable (atmoswing-viewer WIN32 ${src_app_viewer} ${PROJECT_SOURCE_DIR}/../src/shared_base/gui/resource.rc)
    endif (BUILD_VIEWER)
    if (BUILD_TESTS)
        add_executable (atmoswing-tests WIN32 ${src_tests})
    endif (BUILD_TESTS)
    
elseif (UNIX AND NOT APPLE)

    if (BUILD_FORECASTER)
        add_executable (atmoswing-forecaster ${src_app_forecaster})
    endif (BUILD_FORECASTER)
    if (BUILD_VIEWER)
        add_executable (atmoswing-viewer ${src_app_viewer})
    endif (BUILD_VIEWER)
    if (BUILD_CALIBRATOR)
        add_executable (atmoswing-calibrator ${src_app_calibrator})
    endif (BUILD_CALIBRATOR)
    if (BUILD_TESTS)
        add_executable (atmoswing-tests ${src_tests})
    endif (BUILD_TESTS)
    
elseif (APPLE)

    if (BUILD_FORECASTER)
        add_executable (atmoswing-forecaster MACOSX_BUNDLE ${src_app_forecaster})
    endif (BUILD_FORECASTER)
    if (BUILD_VIEWER)
        add_executable (atmoswing-viewer MACOSX_BUNDLE ${src_app_viewer})
    endif (BUILD_VIEWER)
    if (BUILD_CALIBRATOR)
        add_executable (atmoswing-calibrator MACOSX_BUNDLE ${src_app_calibrator})
    endif (BUILD_CALIBRATOR)
    if (BUILD_TESTS)
        add_executable (atmoswing-tests MACOSX_BUNDLE ${src_tests})
    endif (BUILD_TESTS)
    
else (WIN32)

    if (BUILD_FORECASTER)
        add_executable (atmoswing-forecaster ${src_app_forecaster})
    endif (BUILD_FORECASTER)
    if (BUILD_VIEWER)
        add_executable (atmoswing-viewer ${src_app_viewer})
    endif (BUILD_VIEWER)
    if (BUILD_CALIBRATOR)
        add_executable (atmoswing-calibrator ${src_app_calibrator})
    endif (BUILD_CALIBRATOR)
    if (BUILD_TESTS)
        add_executable (atmoswing-tests ${src_tests})
    endif (BUILD_TESTS)
    
endif (WIN32)


#### Cppcheck

if (USE_CPPCHECK)

    if (BUILD_FORECASTER)
        add_cppcheck (atmoswing-forecaster STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
    endif (BUILD_FORECASTER)
    if (BUILD_VIEWER)
        add_cppcheck (atmoswing-viewer STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
    endif (BUILD_VIEWER)
    if (BUILD_CALIBRATOR)
        add_cppcheck (atmoswing-calibrator STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
    endif (BUILD_CALIBRATOR)
    if (BUILD_TESTS)
        add_cppcheck (atmoswing-tests STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
    endif (BUILD_TESTS)
    
endif (USE_CPPCHECK)


#### FLAGS & DEFINES

# Compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${wxWidgets_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DwxDEBUG_LEVEL=0 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG -D__WXDEBUG__")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DwxDEBUG_LEVEL=0 -DNDEBUG ")

if (UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-strict-aliasing -Wno-sign-compare")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer ")
elseif (WIN32)
    set_target_properties(${SHARED_BASE_LIBRARY} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
    if (BUILD_FORECASTER)
        set_target_properties(atmoswing-forecaster PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
        set_target_properties(${SHARED_PROCESSING_LIBRARY} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
    endif (BUILD_FORECASTER)
    if (BUILD_VIEWER)
        set_target_properties(atmoswing-viewer PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
        set_target_properties(vroomgis PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W2")
    endif (BUILD_VIEWER)
    if (BUILD_CALIBRATOR)
        set_target_properties(atmoswing-calibrator PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
        set_target_properties(${SHARED_PROCESSING_LIBRARY} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
    endif (BUILD_CALIBRATOR)
    if (BUILD_TESTS)
        set_target_properties(atmoswing-tests PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
    endif (BUILD_TESTS)
    if (USE_WXHGVERSION)
        set_target_properties(wxhgversion PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W2")
    endif (USE_WXHGVERSION)
endif (UNIX AND NOT APPLE)

# Global definitions
add_definitions(-DUSE_JPEG2000 -DTIXML_USE_TICPP -DTIXML_USE_STL)

if (WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_CRTDBG_MAP_ALLOC)
endif (WIN32)

if (USE_VLD)
    add_definitions(-DUSE_VLD)
endif (USE_VLD)

if (USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif (USE_CUDA)

# Specific definitions
if (USE_GUI)
    set_target_properties(${SHARED_BASE_LIBRARY} PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=1")
else (USE_GUI)
    set_target_properties(${SHARED_BASE_LIBRARY} PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=0")
endif (USE_GUI)
    
if (BUILD_FORECASTER)
    if (USE_GUI)
        set_target_properties(atmoswing-forecaster PROPERTIES COMPILE_DEFINITIONS "APP_FORECASTER; wxUSE_GUI=1")
        set_target_properties(${SHARED_PROCESSING_LIBRARY} PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=1")
    else (USE_GUI)
        set_target_properties(atmoswing-forecaster PROPERTIES COMPILE_DEFINITIONS "APP_FORECASTER; wxUSE_GUI=0")
        set_target_properties(${SHARED_PROCESSING_LIBRARY} PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=0")
    endif (USE_GUI)
endif (BUILD_FORECASTER)

if (BUILD_VIEWER)
    set_target_properties(atmoswing-viewer PROPERTIES COMPILE_DEFINITIONS "APP_VIEWER")
endif (BUILD_VIEWER)

if (BUILD_CALIBRATOR)
    if (USE_GUI)
        set_target_properties(atmoswing-calibrator PROPERTIES COMPILE_DEFINITIONS "APP_CALIBRATOR; MINIMAL_LINKS; wxUSE_GUI=1")
        set_target_properties(${SHARED_PROCESSING_LIBRARY} PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=1")
    else (USE_GUI)
        set_target_properties(atmoswing-calibrator PROPERTIES COMPILE_DEFINITIONS "APP_CALIBRATOR; MINIMAL_LINKS; wxUSE_GUI=0")
        set_target_properties(${SHARED_PROCESSING_LIBRARY} PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=0")
    endif (USE_GUI)
endif (BUILD_CALIBRATOR)

if (BUILD_TESTS)
    if(WIN32)
        set_target_properties(atmoswing-tests PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
        set_target_properties(atmoswing-tests PROPERTIES COMPILE_DEFINITIONS "UNIT_TESTING; wxUSE_GUI=0; _CONSOLE")
    else(WIN32)
        set_target_properties(atmoswing-tests PROPERTIES COMPILE_DEFINITIONS "UNIT_TESTING; wxUSE_GUI=0")
    endif(WIN32)
endif (BUILD_TESTS)

message("CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message("CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")
message("CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}")
message("CMAKE_CXX_FLAGS_RELWITHDEBINFO = ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")


#### LINKING

if (BUILD_FORECASTER)
	target_link_libraries(${SHARED_BASE_LIBRARY} ${GDAL_LIBRARIES})
    target_link_libraries(atmoswing-forecaster ${SHARED_BASE_LIBRARY})
    target_link_libraries(atmoswing-forecaster ${SHARED_PROCESSING_LIBRARY})
    target_link_libraries(atmoswing-forecaster ${wxWidgets_LIBRARIES})
    target_link_libraries(atmoswing-forecaster ${NetCDF_LIBRARIES})
    target_link_libraries(atmoswing-forecaster ${CURL_LIBRARIES})
    target_link_libraries(atmoswing-forecaster ${PROJ4_LIBRARIES})
    target_link_libraries(atmoswing-forecaster ${GDAL_LIBRARIES})
    if (USE_WXHGVERSION)
        target_link_libraries(atmoswing-forecaster ${WXHGVERSION_NAME})
    endif (USE_WXHGVERSION)
    if (USE_VLD)
        target_link_libraries(atmoswing-forecaster ${VLD_LIBRARIES})
    endif (USE_VLD)
endif (BUILD_FORECASTER)

if (BUILD_VIEWER)
	target_link_libraries(${SHARED_BASE_LIBRARY} ${GDAL_LIBRARIES})
    target_link_libraries(atmoswing-viewer ${SHARED_BASE_LIBRARY})
    target_link_libraries(atmoswing-viewer ${wxWidgets_LIBRARIES})
    target_link_libraries(atmoswing-viewer ${NetCDF_LIBRARIES})
    target_link_libraries(atmoswing-viewer ${CURL_LIBRARIES})
    target_link_libraries(atmoswing-viewer ${PROJ4_LIBRARIES})
    target_link_libraries(atmoswing-viewer ${GDAL_LIBRARIES})
    target_link_libraries(atmoswing-viewer ${VROOMGIS_LIBRARY})
    if (USE_WXHGVERSION)
      target_link_libraries(atmoswing-viewer ${WXHGVERSION_NAME})
    endif (USE_WXHGVERSION)
    if (USE_VLD)
        target_link_libraries(atmoswing-viewer ${VLD_LIBRARIES})
    endif (USE_VLD)
    if (WIN32)
        target_link_libraries(atmoswing-viewer Psapi.lib)
    endif (WIN32)
endif (BUILD_VIEWER)

if (BUILD_CALIBRATOR)
    target_link_libraries(atmoswing-calibrator ${SHARED_BASE_LIBRARY})
    target_link_libraries(atmoswing-calibrator ${SHARED_PROCESSING_LIBRARY})
    target_link_libraries(atmoswing-calibrator ${wxWidgets_LIBRARIES})
    target_link_libraries(atmoswing-calibrator ${NetCDF_LIBRARIES})
    target_link_libraries(atmoswing-calibrator ${PROJ4_LIBRARIES})
    if (USE_WXHGVERSION)
        target_link_libraries(atmoswing-calibrator ${WXHGVERSION_NAME})
    endif (USE_WXHGVERSION)
    if (USE_VLD)
        target_link_libraries(atmoswing-calibrator ${VLD_LIBRARIES})
    endif (USE_VLD)
    if (USE_CUDA)
        target_link_libraries(lib_cuda_functions ${CUDA_CUDA_LIBRARY})
        add_dependencies(atmoswing-calibrator lib_cuda_functions)
        target_link_libraries(atmoswing-calibrator lib_cuda_functions)
    endif (USE_CUDA)
endif (BUILD_CALIBRATOR)

if (BUILD_TESTS)
    target_link_libraries(atmoswing-tests ${SHARED_BASE_LIBRARY})
    target_link_libraries(atmoswing-tests ${SHARED_PROCESSING_LIBRARY})
    target_link_libraries(atmoswing-tests ${wxWidgets_LIBRARIES})
    target_link_libraries(atmoswing-tests ${NetCDF_LIBRARIES})
    target_link_libraries(atmoswing-tests ${CURL_LIBRARIES})
    target_link_libraries(atmoswing-tests ${PROJ4_LIBRARIES})
    target_link_libraries(atmoswing-tests ${UNITTEST++_LIBRARIES})
    if (BUILD_FORECASTER)
        target_link_libraries(atmoswing-tests ${GDAL_LIBRARIES})
    endif (BUILD_FORECASTER)
    if (USE_VLD)
        target_link_libraries(atmoswing-tests ${VLD_LIBRARIES})
    endif (USE_VLD)
    if (USE_CUDA)
        target_link_libraries(lib_cuda_functions ${CUDA_CUDA_LIBRARY})
        add_dependencies(atmoswing-tests lib_cuda_functions)
        target_link_libraries(atmoswing-tests lib_cuda_functions)
    endif (USE_CUDA)
endif (BUILD_TESTS)


#### TESTS

if (BUILD_TESTS)
    set(CTEST_OUTPUT_ON_FAILURE TRUE)
    add_test(AtmoSwing-tests atmoswing-tests)
    include(CTest)
endif (BUILD_TESTS)


#### INSTALLER
 
if (BUILD_FORECASTER)
install ( TARGETS atmoswing-forecaster DESTINATION bin)
endif (BUILD_FORECASTER)

if (BUILD_VIEWER)
install ( TARGETS atmoswing-viewer DESTINATION bin)
endif (BUILD_VIEWER)

if (BUILD_CALIBRATOR)
install ( TARGETS atmoswing-calibrator DESTINATION bin)
endif (BUILD_CALIBRATOR)

if (WIN32)
    if (USE_PROVIDED_LIBRARIES)
        install (
          PROGRAMS ${USE_PROVIDED_LIBRARIES_PATH}/gdal110.dll ${USE_PROVIDED_LIBRARIES_PATH}/libcurl.dll ${USE_PROVIDED_LIBRARIES_PATH}/netcdf.dll ${USE_PROVIDED_LIBRARIES_PATH}/zlib1.dll ${USE_PROVIDED_LIBRARIES_PATH}/msvcr110.dll
          DESTINATION .
          )
    else (USE_PROVIDED_LIBRARIES)
        install (
          PROGRAMS ../bin/gdal110.dll ../bin/libcurl.dll ../bin/netcdf.dll ../bin/zlib1.dll ../bin/msvcr110.dll
          DESTINATION .
          )
    endif (USE_PROVIDED_LIBRARIES)
endif (WIN32)
  
install (
  FILES ../license.txt ../notice.txt
  DESTINATION share/atmoswing
  )
  
install (
  DIRECTORY ../config
  DESTINATION share/atmoswing
  )
 
# COMMON PROPERTIES

set(CPACK_PACKAGE_VERSION_MAJOR "${VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${VERSION_PATCH}")
set(CPACK_PACKAGE_VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "AtmoSwing stands for Analog Technique Model for Statistical weather forecastING. The software allows for real-time precipitation forecasting based on a downscaling method, the analogue technique. It identifies analogue days, in terms of atmospheric circulation and humidity variables, in a long archive of past situations and then uses the corresponding measured precipitation to establish an empirical conditional distribution considered as the probabilistic forecast for the target day.")
set(CPACK_PACKAGE_VENDOR "Terr@num")
set(CPACK_STRIP_FILES ON) # tell cpack to strip all debug symbols from all files

# IDENTIFY ARCHITECTURE

set(CPACK_PACKAGE_ARCH "unkown-architecture")

if(${CMAKE_SYSTEM_NAME} MATCHES Windows)
    if(CMAKE_CL_64)
        set(CPACK_PACKAGE_ARCH "win64")
    elseif(MINGW)
        set(CPACK_PACKAGE_ARCH "mingw32")
    elseif(WIN32)
        set(CPACK_PACKAGE_ARCH "win32")
    endif()
endif(${CMAKE_SYSTEM_NAME} MATCHES Windows)

if(${CMAKE_SYSTEM_NAME} MATCHES Linux)
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES i686)
        set(CPACK_PACKAGE_ARCH "linux32")
    elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86_64)
        if(${CMAKE_CXX_FLAGS} MATCHES " -m32 ")
            set(CPACK_PACKAGE_ARCH "linux32")
        else()
            set(CPACK_PACKAGE_ARCH "linux64")
        endif(${CMAKE_CXX_FLAGS} MATCHES " -m32 ")
    else()
        set(CPACK_PACKAGE_ARCH "linux")
    endif()
endif(${CMAKE_SYSTEM_NAME} MATCHES Linux)

if(${CMAKE_SYSTEM_NAME} MATCHES Darwin)
    set(CPACK_PACKAGE_ARCH "mac64")
endif(${CMAKE_SYSTEM_NAME} MATCHES Darwin)

# OS SPECIFIC PROPERTIES

if (APPLE)
    set(CPACK_GENERATOR "DragNDrop")
    set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}_${CPACK_PACKAGE_VERSION}")
    set(CPACK_DMG_VOLUME_NAME "${CMAKE_PROJECT_NAME}")
    set(CPACK_DMG_FORMAT "UDBZ")
endif (APPLE)

if (WIN32)
    set(CPACK_GENERATOR "NSIS")
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "AtmoSwing")
    set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_ARCH}-Setup")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_LIST_DIR}/cpack/license.txt")
    set(CPACK_NSIS_DISPLAY_NAME "AtmoSwing")
    set(CPACK_NSIS_CONTACT "Pascal Horton pascal.horton@terranum.ch")
    set(CPACK_NSIS_HELP_LINK "www.bitbucket.org/atmoswing/atmoswing")
    set(CPACK_NSIS_EXECUTABLES_DIRECTORY ".")
    set(CPACK_NSIS_URL_INFO_ABOUT "www.bitbucket.org/atmoswing/atmoswing")
    set(CPACK_NSIS_MENU_LINKS 
        "${CMAKE_PROJECT_NAME}" "${CMAKE_PROJECT_NAME}" 
        "https://www.bitbucket.org/atmoswing/atmoswing" "www.bitbucket.org")
    set(CPACK_NSIS_COMPRESSOR "lzma")
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL "OFF")
    set(CPACK_NSIS_MODIFY_PATH "ON")
    if (CMAKE_CL_64) 
        set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64") 
    else()
        set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES") 
    endif()
endif(WIN32)

if (UNIX AND NOT APPLE)
    install(FILES ../art/icon/atmoswing.png DESTINATION share/pixmaps)
    if (BUILD_FORECASTER)
        install(FILES cpack/atmoswing-forecaster.desktop DESTINATION share/applications)
    endif (BUILD_FORECASTER)
    if (BUILD_VIEWER)
        install(FILES cpack/atmoswing-viewer.desktop DESTINATION share/applications)
    endif (BUILD_VIEWER)
    if (BUILD_CALIBRATOR)
        install(FILES cpack/atmoswing-calibrator.desktop DESTINATION share/applications)
    endif (BUILD_CALIBRATOR)
    set(CPACK_GENERATOR "DEB")
    set(CPACK_PACKAGE_NAME "${CMAKE_PROJECT_NAME}")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Pascal Horton <pascal.horton@terranum.ch>")
    set(CPACK_PACKAGE_VENDOR "Terr@num")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS " ...  MRC ... ")
    set(CPACK_PACKAGE_DESCRIPTION "AtmoSwing - Analog Technique Model for Statistical weather forecastING.")
    set(CPACK_DEBIAN_PACKAGE_SECTION "science")
    set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
endif(UNIX AND NOT APPLE)
  
include(CPack)
