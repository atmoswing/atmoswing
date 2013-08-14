cmake_minimum_required(VERSION 2.8)


#### PROJECT

project(AtmoswingTests)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BUILD_TYPE})


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
	"${PROJECT_SOURCE_DIR}/../../src/shared_base/libs/eigen/"
	"${PROJECT_SOURCE_DIR}/../../src/shared_base/libs/ticpp/"
	"${PROJECT_SOURCE_DIR}/../../src/shared_processing/core/"
	"${PROJECT_SOURCE_DIR}/../../src/app_forecaster/core/"
	"${PROJECT_SOURCE_DIR}/../../src/app_viewer/core/"
	"${PROJECT_SOURCE_DIR}/../../src/app_calibrator/core/"
	)
include_directories(${inc_dirs})

# Select flags
if (UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "-Wall -fno-strict-aliasing -Wno-sign-compare -Wno-attributes -Wno-unused-local-typedefs -Wno-unused-variable")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s -march=core2")
endif (UNIX AND NOT APPLE)

# Display some variables
message(STATUS "Project source directory : " ${PROJECT_SOURCE_DIR})
message(STATUS "Building in : " ${CMAKE_BINARY_DIR})
message(STATUS "Building in (check) : " ${CMAKE_CURRENT_BINARY_DIR})
message(STATUS "Executable will be : " ${EXECUTABLE_OUTPUT_PATH})
message(STATUS "Build type is : " ${CMAKE_BUILD_TYPE})


#### DEFINES

add_definitions(-DUSE_JPEG2000 -DTIXML_USE_TICPP -DTIXML_USE_STL -D_CRT_SECURE_NO_WARNINGS -DwxUSE_GUI=0 -DUNIT_TESTING)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DwxDEBUG_LEVEL=0 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")


#### LIBRARIES

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

# GDAL
find_package(GDAL REQUIRED)
include_directories(${GDAL_INCLUDE_DIRS})

# UnitTest++
include("../../build/_cmake/FindUnitTest++.cmake")
include_directories(${UNITTEST++_INCLUDE_DIR})


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
