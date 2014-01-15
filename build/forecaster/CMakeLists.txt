cmake_minimum_required(VERSION 2.8)


#### PROJECT

project(AtmoSwingForecaster)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BUILD_TYPE})
set(USE_GUI 1 CACHE BOOL "Sould we build AtmoSwingForecaster with a GUI ?" )
if (WIN32)
    set(USE_VLD 0 CACHE BOOL "Sould we use Visual Leak Detector (https://vld.codeplex.com) ?" )
endif (WIN32)


#### ADDITION OF CUSTOM MODULES

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../_cmake)


#### SOURCE FILES LIST

file(GLOB_RECURSE src_base_core ../../src/shared_base/core/*)
file(GLOB_RECURSE src_base_gui ../../src/shared_base/gui/*)
file(GLOB_RECURSE src_base_img ../../src/shared_base/gui/img/*)
file(GLOB_RECURSE src_lib_awxled ../../src/shared_base/libs/awxled/*)
file(GLOB_RECURSE src_lib_ticpp ../../src/shared_base/libs/ticpp/*)
file(GLOB_RECURSE src_proc_core ../../src/shared_processing/core/*)
file(GLOB_RECURSE src_forec_app ../../src/app_forecaster/app/*)
file(GLOB_RECURSE src_forec_core ../../src/app_forecaster/core/*)
file(GLOB_RECURSE src_forec_gui ../../src/app_forecaster/gui/*)
list(APPEND src_all ${src_base_core} ${src_lib_ticpp} ${src_proc_core} ${src_forec_app} ${src_forec_core})

if (USE_GUI)
    list(APPEND src_all ${src_base_gui} ${src_forec_gui} ${src_lib_awxled} ${src_base_img})
else (USE_GUI)
    list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/app_forecaster/app/AtmoswingMainForecaster.h")
    list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/app_forecaster/app/AtmoswingMainForecaster.cpp")
endif (USE_GUI)

# Remove eventual duplicates
list(REMOVE_DUPLICATES src_all)

# Directories to include
list(APPEND inc_dirs 
    "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/"
    "${PROJECT_SOURCE_DIR}/../../src/shared_base/libs/ticpp/"
    "${PROJECT_SOURCE_DIR}/../../src/shared_processing/core/"
    "${PROJECT_SOURCE_DIR}/../../src/app_forecaster/app/"
    "${PROJECT_SOURCE_DIR}/../../src/app_forecaster/core/"
    )
if (USE_GUI)
    list(APPEND inc_dirs 
        "${PROJECT_SOURCE_DIR}/../../src/shared_base/gui/"
        "${PROJECT_SOURCE_DIR}/../../src/shared_base/gui/img/"
        "${PROJECT_SOURCE_DIR}/../../src/shared_base/libs/awxled/"
        "${PROJECT_SOURCE_DIR}/../../src/app_forecaster/gui/"
        )
endif (USE_GUI)
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
include("../_cmake/FindNetCDF.cmake")
include_directories(${NetCDF_INCLUDE_DIRECTORIES})

# libcURL
mark_as_advanced(CLEAR CURL_INCLUDE_DIR)
mark_as_advanced(CLEAR CURL_LIBRARY)
find_package(CURL REQUIRED)
include_directories(${CURL_INCLUDE_DIRS})

# Proj.4
include("../_cmake/FindProj4.cmake")
include_directories(${PROJ4_INCLUDE_DIR})

# GDAL
find_package(GDAL REQUIRED)
include_directories(${GDAL_INCLUDE_DIRS})

# Eigen
mark_as_advanced(CLEAR EIGEN3_INCLUDE_DIR)
find_package(Eigen3 3.0.0)
include_directories(${EIGEN3_INCLUDE_DIR})

# wxhgversion
if (USE_GUI)
    set(USE_WXHGVERSION 1)
    mark_as_advanced(USE_WXHGVERSION)
    include("../../src/shared_base/libs/wxhgversion/build/use_wxhgversion.cmake")
endif (USE_GUI)

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

add_definitions(-DUSE_JPEG2000 -DTIXML_USE_TICPP -DTIXML_USE_STL -DAPP_FORECASTER)

if (WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_CRTDBG_MAP_ALLOC)
    if (USE_VLD)
        add_definitions(-DUSE_VLD)
    endif (USE_VLD)
endif (WIN32)

if (USE_GUI)
    add_definitions(-DwxUSE_GUI=1)
else (USE_GUI)
    add_definitions(-DwxUSE_GUI=0)
endif (USE_GUI)

message("CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message("CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")
message("CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}")


#### DECLARE EXECUTABLE

if (WIN32)
    if (USE_GUI)
        add_executable (${CMAKE_PROJECT_NAME} WIN32 ${src_all} ${PROJECT_SOURCE_DIR}/../../src/shared_base/gui/resource.rc)
    else (USE_GUI)
        add_executable (${CMAKE_PROJECT_NAME} ${src_all})
    endif (USE_GUI)
elseif (UNIX AND NOT APPLE)
    add_executable (${CMAKE_PROJECT_NAME} ${src_all})
elseif (APPLE)
    add_executable (${CMAKE_PROJECT_NAME} MACOSX_BUNDLE ${src_all})
else (WIN32)
    add_executable (${CMAKE_PROJECT_NAME} ${src_all})
endif (WIN32)


#### LIBRARIES LINKING

target_link_libraries(${CMAKE_PROJECT_NAME} ${wxWidgets_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${NetCDF_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${CURL_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${PROJ4_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${GDAL_LIBRARIES})
if (USE_WXHGVERSION)
    target_link_libraries(${CMAKE_PROJECT_NAME} ${WXHGVERSION_NAME})
endif (USE_WXHGVERSION)
if (WIN32)
    if (USE_VLD)
        target_link_libraries(${CMAKE_PROJECT_NAME} ${VLD_LIBRARIES})
    endif (USE_VLD)
endif (WIN32)
