cmake_minimum_required(VERSION 2.8)

#### PROJECT

project(AtmoswingForecaster)
set(EXECUTABLE_OUTPUT_PATH forecaster/${CMAKE_BUILD_TYPE})


#### SOURCE FILES LIST

file(GLOB_RECURSE src_main ../../src/AtmoswingMainForecaster.* PATTERN ".svn" EXCLUDE)
file(GLOB_RECURSE src_app ../../src/AtmoswingAppForecaster.* PATTERN ".svn" EXCLUDE)
file(GLOB_RECURSE src_core_fcst ../../src/core/forecaster/* PATTERN ".svn" EXCLUDE)
file(GLOB_RECURSE src_core_shd ../../src/core/shared/* PATTERN ".svn" EXCLUDE)
file(GLOB_RECURSE src_data_fcst ../../src/data/forecaster/* PATTERN ".svn" EXCLUDE)
file(GLOB_RECURSE src_data_shd ../../src/data/shared/* PATTERN ".svn" EXCLUDE)
file(GLOB_RECURSE src_defs_fcst ../../src/defs/forecaster/* PATTERN ".svn" EXCLUDE)
file(GLOB_RECURSE src_defs_shd ../../src/defs/shared/* PATTERN ".svn" EXCLUDE)
file(GLOB_RECURSE src_gui_fcst ../../src/gui/forecaster/* PATTERN ".svn" EXCLUDE)
file(GLOB_RECURSE src_gui_shd ../../src/gui/shared/* PATTERN ".svn" EXCLUDE)
file(GLOB_RECURSE src_gui_shdimg ../../src/gui/shared/img/* PATTERN ".svn" EXCLUDE)
file(GLOB_RECURSE src_lib_awxled ../../src/libs/awxled/* PATTERN ".svn" EXCLUDE)
file(GLOB_RECURSE src_lib_eigen ../../src/libs/eigen/* PATTERN ".svn" EXCLUDE)
file(GLOB_RECURSE src_lib_ticpp ../../src/libs/ticpp/* PATTERN ".svn" EXCLUDE)
file(GLOB_RECURSE src_lib_wxemail ../../src/libs/wxemail/* PATTERN ".svn" EXCLUDE)
list(APPEND src_all ${src_main} ${src_app} ${src_core_fcst} ${src_core_shd} ${src_data_fcst} ${src_data_shd} ${src_defs_fcst} ${src_defs_shd} ${src_gui_fcst} ${src_gui_shd} ${src_gui_shdimg} ${src_lib_awxled} ${src_lib_eigen} ${src_lib_ticpp} ${src_lib_wxemail})

# Remove svn and tmp files
file(GLOB_RECURSE files_to_remove1 ../../src/*.svn*)
list(REMOVE_ITEM src_all ${files_to_remove1})
file(GLOB_RECURSE files_to_remove2 ../../src/*.tmp)
#list(REMOVE_ITEM src_all ${files_to_remove2})
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/core/forecaster/asPostprocessor.cpp")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/core/forecaster/asPostprocessor.h")
list(REMOVE_DUPLICATES src_all)

# Display the files list
message(STATUS "Files : ")
foreach(currfile ${src_all})
	message(STATUS "  ${currfile}")
endforeach(currfile)

# Directories to include
list(APPEND inc_dirs 
    "${PROJECT_SOURCE_DIR}/../../src/"
	"${PROJECT_SOURCE_DIR}/../../src/core/forecaster/"
	"${PROJECT_SOURCE_DIR}/../../src/core/shared/"
	"${PROJECT_SOURCE_DIR}/../../src/data/forecaster/"
	"${PROJECT_SOURCE_DIR}/../../src/data/shared/"
	"${PROJECT_SOURCE_DIR}/../../src/defs/forecaster/"
	"${PROJECT_SOURCE_DIR}/../../src/defs/shared/"
	"${PROJECT_SOURCE_DIR}/../../src/gui/forecaster/"
	"${PROJECT_SOURCE_DIR}/../../src/gui/shared/"
	"${PROJECT_SOURCE_DIR}/../../src/gui/shared/img/"
	"${PROJECT_SOURCE_DIR}/../../src/libs/awxled/"
	"${PROJECT_SOURCE_DIR}/../../src/libs/eigen/"
	"${PROJECT_SOURCE_DIR}/../../src/libs/ticpp/"
    "${PROJECT_SOURCE_DIR}/../../src/libs/wxemail/"
	"${PROJECT_SOURCE_DIR}/../../src/"
	)
include_directories(${inc_dirs})

# Display the directories to include
message(STATUS "Directories : ")
foreach(currdir ${inc_dirs})
	message(STATUS "  ${currdir}")
endforeach(currdir)

# Select flags.
if( UNIX )
    set(CMAKE_CXX_FLAGS "-Wall -fno-strict-aliasing -Wno-sign-compare -Wno-attributes")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s -march=core2")
endif( UNIX )

# Display some variables
message(STATUS "Project source directory : " ${PROJECT_SOURCE_DIR})
message(STATUS "Building in : " ${CMAKE_BINARY_DIR})
message(STATUS "Building in (check) : " ${CMAKE_CURRENT_BINARY_DIR})
message(STATUS "Executable will be : " ${EXECUTABLE_OUTPUT_PATH})
message(STATUS "Build type is : " ${CMAKE_BUILD_TYPE})


#### DEFINES

add_definitions(-DUSE_JPEG2000 -DTIXML_USE_TICPP -DTIXML_USE_STL -DNDEBUG -DAPP_FORECASTER)


#### DECLARE EXECUTABLE

add_executable(AtmoswingForecaster ${src_all})


#### LIBRARIES

# WxWidgets
include(FindwxWidgets)
set( wxWidgets_USE_UNICODE ON ) 
set( wxWidgets_USE_STATIC OFF )
set( wxWidgets_USE_DEBUG ON )
set( wxWidgets_USE_MONOLITHIC ON )
find_package(wxWidgets REQUIRED core base)
include(${wxWidgets_USE_FILE})
include_directories(${wxWidgets_INCLUDE_DIRS})
target_link_libraries(AtmoswingForecaster ${wxWidgets_LIBRARIES})

# GDAL
include(FindGDAL)
find_package(GDAL REQUIRED)
include_directories(${GDAL_INCLUDE_DIRS})
target_link_libraries(AtmoswingForecaster ${GDAL_LIBRARIES})

# Proj
include("../_cmake/FindProj4.cmake")
include_directories(${PROJ4_INCLUDE_DIR})
target_link_libraries(AtmoswingForecaster ${PROJ4_LIBRARIES})

# Netcdf
include("../_cmake/FindNetcdf.cmake")
include_directories(${NetCDF_INCLUDE_DIRECTORIES})
target_link_libraries(AtmoswingForecaster ${NetCDF_LIBRARIES})

# libcurl
include("../_cmake/FindCURL.cmake")
find_package(CURL REQUIRED)
set(CURL_INCLUDE_DIRS ${CURL_INCLUDE_DIRS}/curl)
include_directories(${CURL_INCLUDE_DIRS})
target_link_libraries(AtmoswingForecaster ${CURL_LIBRARIES})
if(CURL_FOUND)
	message(STATUS "Curl found")
	message("    Please check the cURL include directory.")
	message("    include: ${CURL_INCLUDE_DIRS}")
else(CURL_FOUND)
	message("Curl not found. please check the cURL include directory.")
	message("    include: ${CURL_INCLUDE_DIRS}")
endif(CURL_FOUND)

# GTK
if( UNIX )
	find_package( GTK2 2.10 REQUIRED gtk )
	include_directories(${GTK2_INCLUDE_DIRS})
	target_link_libraries(AtmoswingForecaster ${GTK2_LIBRARIES})
endif( UNIX )


#### VARIOUS

# To use the MT version by default under Windows
IF (WIN32)
	include ("../_cmake/UseVisualstudioMT.cmake")
ENDIF (WIN32)

