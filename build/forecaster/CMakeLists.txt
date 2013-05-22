cmake_minimum_required(VERSION 2.8)
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../_cmake/")

#### PROJECT

project(AtmoswingForecaster)
set(EXECUTABLE_OUTPUT_PATH forecaster/${CMAKE_BUILD_TYPE})


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
list(APPEND src_all ${src_base_core} ${src_base_gui} ${src_base_img} ${src_lib_awxled} ${src_lib_eigen} ${src_lib_ticpp} ${src_proc_core} ${src_forec_app} ${src_forec_core} ${src_forec_gui})

# Remove eventual duplicates
list(REMOVE_DUPLICATES src_all)

# Display the files list
message(STATUS "Files : ")
foreach(currfile ${src_all})
	message(STATUS "  ${currfile}")
endforeach(currfile)

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
if( UNIX )
        set( wxWidgets_USE_MONOLITHIC OFF )
else( UNIX )
        set( wxWidgets_USE_MONOLITHIC ON )
endif( UNIX )
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
#find_package(PROJ4 REQUIRED)
include_directories(${PROJ4_INCLUDE_DIR})
target_link_libraries(AtmoswingForecaster ${PROJ4_LIBRARIES})

# Netcdf
include("../_cmake/FindNetCDF.cmake")
#find_package(NetCDF REQUIRED)
include_directories(${NetCDF_INCLUDE_DIRECTORIES})
target_link_libraries(AtmoswingForecaster ${NetCDF_LIBRARIES})

# libcurl
include("../_cmake/FindcURL.cmake")
find_package(CURL REQUIRED)
set(CURL_INCLUDE_DIRS ${CURL_INCLUDE_DIRS}/curl)
include_directories(${CURL_INCLUDE_DIRS})
target_link_libraries(AtmoswingForecaster ${CURL_LIBRARIES})
if( CURL_FOUND )
	message(STATUS "Curl found")
	message("    Please check the cURL include directory.")
	message("    include: ${CURL_INCLUDE_DIRS}")
else( CURL_FOUND )
	message("Curl not found. please check the cURL include directory.")
	message("    include: ${CURL_INCLUDE_DIRS}")
endif( CURL_FOUND )

# GTK
if( UNIX )
	find_package( GTK2 2.10 REQUIRED gtk )
	include_directories(${GTK2_INCLUDE_DIRS})
	target_link_libraries(AtmoswingForecaster ${GTK2_LIBRARIES})
endif( UNIX )


