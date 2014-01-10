cmake_minimum_required(VERSION 2.8)


#### PROJECT

project(AtmoSwingCalibrator)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BUILD_TYPE})
set(USE_GUI 1 CACHE BOOL "Sould we build AtmoswingCalibrator with a GUI ?" )


#### ADDITION OF CUSTOM MODULES

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../_cmake)


#### SOURCE FILES LIST

file(GLOB_RECURSE src_base_core ../../src/shared_base/core/*)
file(GLOB_RECURSE src_base_gui ../../src/shared_base/gui/*)
file(GLOB_RECURSE src_base_img ../../src/shared_base/gui/img/*)
file(GLOB_RECURSE src_lib_awxled ../../src/shared_base/libs/awxled/*)
file(GLOB_RECURSE src_lib_ticpp ../../src/shared_base/libs/ticpp/*)
file(GLOB_RECURSE src_proc_core ../../src/shared_processing/core/*)
file(GLOB_RECURSE src_calib_app ../../src/app_calibrator/app/*)
file(GLOB_RECURSE src_calib_core ../../src/app_calibrator/core/*)
file(GLOB_RECURSE src_calib_gui ../../src/app_calibrator/gui/*)
list(APPEND src_all ${src_base_core} ${src_lib_ticpp} ${src_proc_core} ${src_calib_app} ${src_calib_core})

if (USE_GUI)
	list(APPEND src_all ${src_base_gui} ${src_calib_gui} ${src_lib_awxled} ${src_base_img})
else (USE_GUI)
	list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/app_calibrator/app/AtmoswingMainCalibrator.h")
	list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/app_calibrator/app/AtmoswingMainCalibrator.cpp")
endif (USE_GUI)

# Remove some files
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/app_forecaster/core/asMethodForecasting.h")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/app_forecaster/core/asMethodForecasting.cpp")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/asInternet.h")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/asInternet.cpp")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/asParametersForecast.h")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/asParametersForecast.cpp")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/asThreadInternetDownload.h")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/asThreadInternetDownload.cpp")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/app_forecaster/core/asFileForecastingModels.h")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/app_forecaster/core/asFileForecastingModels.cpp")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/asDataPredictorRealtime.h")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/asDataPredictorRealtime.cpp")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/asDataPredictorRealtimeGfsForecast.h")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/asDataPredictorRealtimeGfsForecast.cpp")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/asFileGrib2.h")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/asFileGrib2.cpp")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/asFileParametersForecast.h")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/asFileParametersForecast.cpp")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/asResultsAnalogsForecast.h")
list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/shared_base/core/asResultsAnalogsForecast.cpp")

# Remove eventual duplicates
list(REMOVE_DUPLICATES src_all)

# Directories to include
list(APPEND inc_dirs 
	"${PROJECT_SOURCE_DIR}/../../src/shared_base/core/"
	"${PROJECT_SOURCE_DIR}/../../src/shared_base/libs/ticpp/"
	"${PROJECT_SOURCE_DIR}/../../src/shared_processing/core/"
	"${PROJECT_SOURCE_DIR}/../../src/app_calibrator/app/"
	"${PROJECT_SOURCE_DIR}/../../src/app_calibrator/core/"
	)
if (USE_GUI)
	list(APPEND inc_dirs 
		"${PROJECT_SOURCE_DIR}/../../src/shared_base/gui/"
		"${PROJECT_SOURCE_DIR}/../../src/shared_base/gui/img/"
		"${PROJECT_SOURCE_DIR}/../../src/shared_base/libs/awxled/"
		"${PROJECT_SOURCE_DIR}/../../src/app_calibrator/gui/"
		)
endif (USE_GUI)
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

add_definitions(-DUSE_JPEG2000 -DTIXML_USE_TICPP -DTIXML_USE_STL -DNDEBUG -DAPP_CALIBRATOR -D_CRT_SECURE_NO_WARNINGS -DMINIMAL_LINKS)
if (USE_GUI)
	add_definitions(-DwxUSE_GUI=1)
else (USE_GUI)
	add_definitions(-DwxUSE_GUI=0)
endif (USE_GUI)

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

# NetCDF
mark_as_advanced(CLEAR NetCDF_INCLUDE_DIRECTORIES)
mark_as_advanced(CLEAR NetCDF_C_LIBRARY)
include("../_cmake/FindNetCDF.cmake")
include_directories(${NetCDF_INCLUDE_DIRECTORIES})

# Proj.4
include("../_cmake/FindProj4.cmake")
include_directories(${PROJ4_INCLUDE_DIR})

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
target_link_libraries(${CMAKE_PROJECT_NAME} ${PROJ4_LIBRARIES})
if (USE_WXHGVERSION)
    target_link_libraries(${CMAKE_PROJECT_NAME} ${WXHGVERSION_NAME})
endif (USE_WXHGVERSION)
