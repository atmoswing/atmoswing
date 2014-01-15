cmake_minimum_required(VERSION 2.8)


#### PROJECT

project(AtmoSwingTests)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BUILD_TYPE})
if (WIN32)
    set(USE_VLD 0 CACHE BOOL "Sould we use Visual Leak Detector (https://vld.codeplex.com) ?" )
endif (WIN32)


#### ADDITION OF CUSTOM MODULES

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../build/_cmake)


#### SOURCE FILES LIST

file(GLOB_RECURSE src_test ../src/*)
file(GLOB_RECURSE src_base_core ../../src/shared_base/core/*)
file(GLOB_RECURSE src_lib_ticpp ../../src/shared_base/libs/ticpp/*)
file(GLOB_RECURSE src_proc_core ../../src/shared_processing/core/*)
file(GLOB_RECURSE src_forec_core ../../src/app_forecaster/core/*)
file(GLOB_RECURSE src_viewer_core ../../src/app_viewer/core/*)
file(GLOB_RECURSE src_calib_core ../../src/app_calibrator/core/*)
list(APPEND src_all ${src_test} ${src_base_core} ${src_lib_ticpp} ${src_proc_core} ${src_forec_core} ${src_viewer_core} ${src_calib_core})

# Remove some files
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/app_forecaster/core/asGlobVars.h")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/app_forecaster/core/asGlobVars.cpp")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/app_viewer/core/asGlobVars.h")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/app_viewer/core/asGlobVars.cpp")
    
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
include("../../build/_cmake/FindNetCDF.cmake")
include_directories(${NetCDF_INCLUDE_DIRECTORIES})

# libcURL
mark_as_advanced(CLEAR CURL_INCLUDE_DIR)
mark_as_advanced(CLEAR CURL_LIBRARY)
find_package(CURL REQUIRED)
include_directories(${CURL_INCLUDE_DIRS})

# Proj.4
include("../../build/_cmake/FindProj4.cmake")
include_directories(${PROJ4_INCLUDE_DIR})

# Eigen
mark_as_advanced(CLEAR EIGEN3_INCLUDE_DIR)
find_package(Eigen3 3.0.0)
include_directories(${EIGEN3_INCLUDE_DIR})

# GDAL
find_package(GDAL REQUIRED)
include_directories(${GDAL_INCLUDE_DIRS})

# UnitTest++
include("../../build/_cmake/FindUnitTest++.cmake")
include_directories(${UNITTEST++_INCLUDE_DIR})

# Visual Leak Detector
if (WIN32)
    if (USE_VLD)
        find_package(VLD)
        include_directories(${VLD_INCLUDE_DIRS})
    endif (USE_VLD)
endif (WIN32)


#### FLAGS & DEFINES

if (UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-sign-compare")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -s -march=core2")
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


#### DECLARE EXECUTABLE

# Simple case (no GUI)
add_executable (${CMAKE_PROJECT_NAME} ${src_all})


#### LIBRARIES LINKING

target_link_libraries(${CMAKE_PROJECT_NAME} ${wxWidgets_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${NetCDF_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${CURL_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${PROJ4_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${GDAL_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${UNITTEST++_LIBRARY})
if (WIN32)
    if (USE_VLD)
        target_link_libraries(${CMAKE_PROJECT_NAME} ${VLD_LIBRARIES})
    endif (USE_VLD)
endif (WIN32)

