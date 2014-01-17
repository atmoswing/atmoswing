cmake_minimum_required(VERSION 2.8)


#### PROJECT

project(AtmoSwingViewer)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BUILD_TYPE})
if (WIN32)
    set(USE_VLD 0 CACHE BOOL "Sould we use Visual Leak Detector (https://vld.codeplex.com) ?" )
endif (WIN32)


#### ADDITION OF CUSTOM MODULES

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../_cmake)


#### SOURCE FILES LIST

file(GLOB_RECURSE src_base_core ../../src/shared_base/core/*)
file(GLOB_RECURSE src_base_gui ../../src/shared_base/gui/*)
file(GLOB_RECURSE src_base_img ../../src/shared_base/gui/img/*)
file(GLOB_RECURSE src_lib_ticpp ../../src/shared_base/libs/ticpp/*)
file(GLOB_RECURSE src_lib_awxled ../../src/shared_base/libs/awxled/*)
file(GLOB_RECURSE src_viewer_app ../../src/app_viewer/app/*)
file(GLOB_RECURSE src_viewer_core ../../src/app_viewer/core/*)
file(GLOB_RECURSE src_viewer_gui ../../src/app_viewer/gui/*)
file(GLOB_RECURSE src_lib_wxmathplot ../../src/app_viewer/libs/wxmathplot/*)
file(GLOB_RECURSE src_lib_wxplotctrl ../../src/app_viewer/libs/wxplotctrl/src/*)
file(GLOB_RECURSE src_lib_wxthings ../../src/app_viewer/libs/wxthings/src/*)
list(APPEND src_all ${src_base_core} ${src_base_gui} ${src_base_img} ${src_lib_ticpp}  ${src_lib_awxled} ${src_viewer_app} ${src_viewer_core} ${src_viewer_gui} ${src_lib_wxmathplot} ${src_lib_wxplotctrl} ${src_lib_wxthings})

# Remove eventual duplicates
list(REMOVE_DUPLICATES src_all)

# Directories to include
list(APPEND inc_dirs 
    "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/"
    "${PROJECT_SOURCE_DIR}/../../src/shared_base/gui/"
    "${PROJECT_SOURCE_DIR}/../../src/shared_base/gui/img/"
    "${PROJECT_SOURCE_DIR}/../../src/shared_base/libs/ticpp/"
    "${PROJECT_SOURCE_DIR}/../../src/shared_base/libs/awxled/"
    "${PROJECT_SOURCE_DIR}/../../src/app_viewer/app/"
    "${PROJECT_SOURCE_DIR}/../../src/app_viewer/core/"
    "${PROJECT_SOURCE_DIR}/../../src/app_viewer/gui/"
    "${PROJECT_SOURCE_DIR}/../../src/app_viewer/libs/wxmathplot/"
    "${PROJECT_SOURCE_DIR}/../../src/app_viewer/libs/wxplotctrl/"
    "${PROJECT_SOURCE_DIR}/../../src/app_viewer/libs/wxplotctrl/include/"
    "${PROJECT_SOURCE_DIR}/../../src/app_viewer/libs/wxthings/include/"
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
find_package(wxWidgets COMPONENTS core base adv REQUIRED)
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

# GDAL
find_package(GDAL REQUIRED)
include_directories(${GDAL_INCLUDE_DIRS})

# Eigen
mark_as_advanced(CLEAR EIGEN3_INCLUDE_DIR)
find_package(Eigen3 3.0.0 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# vroomgis
mark_as_advanced(SEARCH_GDAL)
mark_as_advanced(SEARCH_GEOS)
mark_as_advanced(SEARCH_GIS_LIB_PATH)
mark_as_advanced(SEARCH_VROOMGIS_LIBS)
mark_as_advanced(SEARCH_VROOMGIS_WXWIDGETS)
mark_as_advanced(SQLITE_INCLUDE_DIR)
mark_as_advanced(SQLITE_PATH)
mark_as_advanced(wxWIDGETS_USING_SVN)
include("../../src/app_viewer/libs/vroomgis/vroomgis/build/cmake/Use_vroomGISlib.cmake")
link_libraries(${wxWidgets_LIBRARIES})

# wxhgversion
set(USE_WXHGVERSION 1)
mark_as_advanced(USE_WXHGVERSION)
include("../../src/shared_base/libs/wxhgversion/build/use_wxhgversion.cmake")

# Visual Leak Detector
if (WIN32)
    if (USE_VLD)
        find_package(VLD)
        include_directories(${VLD_INCLUDE_DIRS})
    endif (USE_VLD)
endif (WIN32)


#### DECLARE EXECUTABLE

if (WIN32)
    add_executable (${CMAKE_PROJECT_NAME} WIN32 ${src_all} ${PROJECT_SOURCE_DIR}/../../src/shared_base/gui/resource.rc)
elseif (UNIX AND NOT APPLE)
    add_executable (${CMAKE_PROJECT_NAME} ${src_all})
elseif (APPLE)
    add_executable (${CMAKE_PROJECT_NAME} MACOSX_BUNDLE ${src_all})
else (WIN32)
    add_executable (${CMAKE_PROJECT_NAME} ${src_all})
endif (WIN32)


#### FLAGS & DEFINES

if (UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-strict-aliasing -Wno-sign-compare")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -s -march=core2")
elseif (WIN32)
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
    set_target_properties(vroomgis PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W2")
    set_target_properties(wxhgversion PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W2")
endif (UNIX AND NOT APPLE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${wxWidgets_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DwxDEBUG_LEVEL=0 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG -D__WXDEBUG__")

add_definitions(-DUSE_JPEG2000 -DTIXML_USE_TICPP -DTIXML_USE_STL -DAPP_VIEWER)

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
target_link_libraries(${CMAKE_PROJECT_NAME} ${GDAL_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${VROOMGIS_LIBRARY})
if (USE_WXHGVERSION)
  target_link_libraries(${CMAKE_PROJECT_NAME} ${WXHGVERSION_NAME})
endif (USE_WXHGVERSION)
if (WIN32)
    target_link_libraries(${CMAKE_PROJECT_NAME} Psapi.lib)
endif (WIN32)
if (WIN32)
    if (USE_VLD)
        target_link_libraries(${CMAKE_PROJECT_NAME} ${VLD_LIBRARIES})
    endif (USE_VLD)
endif (WIN32)
