cmake_minimum_required(VERSION 2.8)


#### PROJECT

project(AtmoswingForecaster)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BUILD_TYPE})


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
list(APPEND src_all ${src_base_core} ${src_base_gui} ${src_base_img} ${src_lib_awxled} ${src_lib_ticpp} ${src_proc_core} ${src_forec_app} ${src_forec_core} ${src_forec_gui})

# Remove eventual duplicates
list(REMOVE_DUPLICATES src_all)

# Display the files list
message(STATUS "Files : ")
foreach (currfile ${src_all})
	message(STATUS "  ${currfile}")
endforeach (currfile)

# Directories to include
list(APPEND inc_dirs 
	"${PROJECT_SOURCE_DIR}/../../src/shared_base/core/"
	"${PROJECT_SOURCE_DIR}/../../src/shared_base/gui/"
	"${PROJECT_SOURCE_DIR}/../../src/shared_base/gui/img/"
	"${PROJECT_SOURCE_DIR}/../../src/shared_base/libs/awxled/"
	"${PROJECT_SOURCE_DIR}/../../src/shared_base/libs/eigen/"
	"${PROJECT_SOURCE_DIR}/../../src/shared_base/libs/ticpp/"
	"${PROJECT_SOURCE_DIR}/../../src/shared_processing/core/"
	"${PROJECT_SOURCE_DIR}/../../src/app_forecaster/app/"
	"${PROJECT_SOURCE_DIR}/../../src/app_forecaster/core/"
	"${PROJECT_SOURCE_DIR}/../../src/app_forecaster/gui/"
	)
include_directories(${inc_dirs})

# Display the directories to include
message(STATUS "Directories : ")
foreach (currdir ${inc_dirs})
	message(STATUS "  ${currdir}")
endforeach (currdir)

# Select flags
if (UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "-Wall -fno-strict-aliasing -Wno-sign-compare -Wno-attributes")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s -march=core2")
endif (UNIX AND NOT APPLE)

# Display some variables
message(STATUS "Project source directory : " ${PROJECT_SOURCE_DIR})
message(STATUS "Building in : " ${CMAKE_BINARY_DIR})
message(STATUS "Building in (check) : " ${CMAKE_CURRENT_BINARY_DIR})
message(STATUS "Executable will be : " ${EXECUTABLE_OUTPUT_PATH})
message(STATUS "Build type is : " ${CMAKE_BUILD_TYPE})


#### DEFINES

add_definitions(-DUSE_JPEG2000 -DTIXML_USE_TICPP -DTIXML_USE_STL -DNDEBUG -DAPP_FORECASTER -D_CRT_SECURE_NO_WARNINGS)


#### LIBRARIES

# WxWidgets (adv lib nedded for the caldendar widget)
mark_as_advanced(wxWidgets_wxrc_EXECUTABLE)
find_package(wxWidgets COMPONENTS core base adv REQUIRED)
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


#### LIBRARIES LINKING

target_link_libraries(AtmoswingForecaster ${wxWidgets_LIBRARIES})
target_link_libraries(AtmoswingForecaster ${NetCDF_LIBRARIES})
target_link_libraries(AtmoswingForecaster ${CURL_LIBRARIES})
target_link_libraries(AtmoswingForecaster ${PROJ4_LIBRARIES})
target_link_libraries(AtmoswingForecaster ${GDAL_LIBRARIES})
