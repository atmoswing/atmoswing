cmake_minimum_required(VERSION 2.8)


#### PROJECT

project(AtmoSwingTests)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BUILD_TYPE})
if (WIN32)
    set(USE_VLD 0 CACHE BOOL "Sould we use Visual Leak Detector (https://vld.codeplex.com) ?" )
endif (WIN32)
set(TEST_FORECASTER 0 CACHE BOOL "Sould we test grib2 files (only needed for the forecaster) ?" )


#### ADDITION OF CUSTOM MODULES

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../build/_cmake)


#### SOURCE FILES LIST

file(GLOB_RECURSE src_test ../src/*)
file(GLOB_RECURSE src_base_core ../../src/shared_base/core/*)
file(GLOB_RECURSE src_lib_ticpp ../../src/shared_base/libs/ticpp/*)
file(GLOB_RECURSE src_proc_core ../../src/shared_processing/core/*)
file(GLOB_RECURSE src_viewer_core ../../src/app_viewer/core/*)
file(GLOB_RECURSE src_calib_core ../../src/app_calibrator/core/*)
list(APPEND src_all ${src_test} ${src_base_core} ${src_lib_ticpp} ${src_proc_core} ${src_viewer_core} ${src_calib_core})

if (TEST_FORECASTER)
    file(GLOB_RECURSE src_forec_core ../../src/app_forecaster/core/*)
	list(APPEND src_all ${src_all} ${src_forec_core})
endif (TEST_FORECASTER)


# Remove some files
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/app_viewer/core/asGlobVars.h")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/app_viewer/core/asGlobVars.cpp")
if (TEST_FORECASTER)
    list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/app_forecaster/core/asGlobVars.h")
    list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/app_forecaster/core/asGlobVars.cpp")
else (TEST_FORECASTER)
    list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../src/asDataPredictorRealtimeGeneralGridTest.h")
    list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../src/asDataPredictorRealtimeGeneralGridTest.cpp")
    list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../src/asDataPredictorRealtimeRegularGridTest.h")
    list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../src/asDataPredictorRealtimeRegularGridTest.cpp")
    list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/asFileGrib2.h")
    list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/asFileGrib2.cpp")
    list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/asDataPredictorRealtime.h")
    list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/asDataPredictorRealtime.cpp")
    list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/asDataPredictorRealtimeGfsForecast.h")
    list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/asDataPredictorRealtimeGfsForecast.cpp")
endif (TEST_FORECASTER)
    
# Remove eventual duplicates
list(REMOVE_DUPLICATES src_all)

# Directories to include
list(APPEND inc_dirs 
    "${PROJECT_SOURCE_DIR}/../src/"
    "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/"
    "${PROJECT_SOURCE_DIR}/../../src/shared_base/libs/ticpp/"
    "${PROJECT_SOURCE_DIR}/../../src/shared_processing/core/"
    "${PROJECT_SOURCE_DIR}/../../src/app_forecaster/core/"
    "${PROJECT_SOURCE_DIR}/../../src/app_viewer/core/"
    "${PROJECT_SOURCE_DIR}/../../src/app_calibrator/core/"
    )
include_directories(${inc_dirs})


#### LIBRARIES

if (WIN32)
    set(USE_PROVIDED_LIBRARIES 1 CACHE BOOL "Use the libraries downloaded from https://bitbucket.org/atmoswing/atmoswing")
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
find_package(wxWidgets COMPONENTS base REQUIRED)
include( "${wxWidgets_USE_FILE}" )
include_directories(${wxWidgets_INCLUDE_DIRS})

# NetCDF (has to be before GDAL)
mark_as_advanced(CLEAR NetCDF_INCLUDE_DIRECTORIES)
mark_as_advanced(CLEAR NetCDF_C_LIBRARY)
find_package(NetCDF REQUIRED)
include_directories(${NetCDF_INCLUDE_DIRECTORIES})

# libcURL
mark_as_advanced(CLEAR CURL_INCLUDE_DIR)
mark_as_advanced(CLEAR CURL_LIBRARY)
find_package(CURL REQUIRED)
include_directories(${CURL_INCLUDE_DIRS})

# Proj.4
find_package(Proj4 REQUIRED)
include_directories(${PROJ4_INCLUDE_DIR})

# Eigen
mark_as_advanced(CLEAR EIGEN3_INCLUDE_DIR)
find_package(Eigen3 3.0.0 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# GDAL
if (TEST_FORECASTER)
    find_package(GDAL REQUIRED)
    include_directories(${GDAL_INCLUDE_DIRS})
endif (TEST_FORECASTER)

# UnitTest++
find_package(UnitTest++ REQUIRED)
include_directories(${UNITTEST++_INCLUDE_DIR})

# Visual Leak Detector
if (WIN32)
    if (USE_VLD)
        find_package(VLD REQUIRED)
        include_directories(${VLD_INCLUDE_DIRS})
    endif (USE_VLD)
endif (WIN32)


#### DECLARE EXECUTABLE

# Simple case (no GUI)
add_executable (${CMAKE_PROJECT_NAME} ${src_all})


#### FLAGS & DEFINES

if (UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-strict-aliasing -Wno-sign-compare")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -s -march=core2")
elseif (WIN32)
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
endif (UNIX AND NOT APPLE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${wxWidgets_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DwxDEBUG_LEVEL=0 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG -D__WXDEBUG__")

add_definitions(-DUSE_JPEG2000 -DTIXML_USE_TICPP -DTIXML_USE_STL -DwxUSE_GUI=0 -DUNIT_TESTING)

if (WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_CRTDBG_MAP_ALLOC)
    if (USE_VLD)
        add_definitions(-DUSE_VLD)
    endif (USE_VLD)
endif (WIN32)

message("CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message("CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")
message("CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}")


#### LIBRARIES LINKING

target_link_libraries(${CMAKE_PROJECT_NAME} ${wxWidgets_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${NetCDF_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${CURL_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${PROJ4_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${UNITTEST++_LIBRARIES})
if (TEST_FORECASTER)
    target_link_libraries(${CMAKE_PROJECT_NAME} ${GDAL_LIBRARIES})
endif (TEST_FORECASTER)
if (WIN32)
    if (USE_VLD)
        target_link_libraries(${CMAKE_PROJECT_NAME} ${VLD_LIBRARIES})
    endif (USE_VLD)
endif (WIN32)

