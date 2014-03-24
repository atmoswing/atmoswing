cmake_minimum_required(VERSION 2.8)


#### PROJECT

if (WIN32)
    project(AtmoSwingCalibrator)
elseif (UNIX AND NOT APPLE)
    project(atmoswing-calibrator)
elseif (APPLE)
    project(AtmoSwingCalibrator)
else (WIN32)
    project(AtmoSwingCalibrator)
endif (WIN32)

set(VERSION_MAJOR "1")
set(VERSION_MINOR "3")
set(VERSION_PATCH "0")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BUILD_TYPE})
set(USE_GUI 1 CACHE BOOL "Sould we build AtmoswingCalibrator with a GUI ?" )
if (WIN32)
    set(USE_VLD 0 CACHE BOOL "Sould we use Visual Leak Detector (https://vld.codeplex.com) ?" )
endif (WIN32)


#### ADDITION OF CUSTOM MODULES

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../_cmake)


#### SOURCE FILES LIST

file(GLOB_RECURSE src_base_core_h ../../src/shared_base/core/*.h)
file(GLOB_RECURSE src_base_core_cpp ../../src/shared_base/core/*.cpp)
file(GLOB_RECURSE src_base_gui_h ../../src/shared_base/gui/*.h)
file(GLOB_RECURSE src_base_gui_cpp ../../src/shared_base/gui/*.cpp)
file(GLOB_RECURSE src_base_img_h ../../src/shared_base/gui/img/*.h)
file(GLOB_RECURSE src_base_img_cpp ../../src/shared_base/gui/img/*.cpp)
file(GLOB_RECURSE src_lib_awxled_h ../../src/shared_base/libs/awxled/*.h)
file(GLOB_RECURSE src_lib_awxled_cpp ../../src/shared_base/libs/awxled/*.cpp)
file(GLOB_RECURSE src_lib_ticpp_h ../../src/shared_base/libs/ticpp/*.h)
file(GLOB_RECURSE src_lib_ticpp_cpp ../../src/shared_base/libs/ticpp/*.cpp)
file(GLOB_RECURSE src_proc_core_h ../../src/shared_processing/core/*.h)
file(GLOB_RECURSE src_proc_core_cpp ../../src/shared_processing/core/*.cpp)
file(GLOB_RECURSE src_calib_app_h ../../src/app_calibrator/app/*.h)
file(GLOB_RECURSE src_calib_app_cpp ../../src/app_calibrator/app/*.cpp)
file(GLOB_RECURSE src_calib_core_h ../../src/app_calibrator/core/*.h)
file(GLOB_RECURSE src_calib_core_cpp ../../src/app_calibrator/core/*.cpp)
file(GLOB_RECURSE src_calib_gui_h ../../src/app_calibrator/gui/*.h)
file(GLOB_RECURSE src_calib_gui_cpp ../../src/app_calibrator/gui/*.cpp)
list(APPEND src_all ${src_base_core_h} ${src_lib_ticpp_h} ${src_proc_core_h} ${src_calib_app_h} ${src_calib_core_h})
list(APPEND src_all ${src_base_core_cpp} ${src_lib_ticpp_cpp} ${src_proc_core_cpp} ${src_calib_app_cpp} ${src_calib_core_cpp})

if (USE_GUI)
    list(APPEND src_all ${src_base_gui_h} ${src_calib_gui_h} ${src_lib_awxled_h} ${src_base_img_h})
    list(APPEND src_all ${src_base_gui_cpp} ${src_calib_gui_cpp} ${src_lib_awxled_cpp} ${src_base_img_cpp})
else (USE_GUI)
    list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/app_calibrator/app/AtmoswingMainCalibrator.h")
    list(REMOVE_ITEM src_all "${PROJECT_SOURCE_DIR}/../../src/app_calibrator/app/AtmoswingMainCalibrator.cpp")
endif (USE_GUI)

# Remove some files
list(REMOVE_ITEM src_all "*.orig")
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
find_package(NetCDF REQUIRED)
include_directories(${NetCDF_INCLUDE_DIRECTORIES})

# Proj.4
find_package(Proj4 REQUIRED)
include_directories(${PROJ4_INCLUDE_DIR})

# Eigen
mark_as_advanced(CLEAR EIGEN3_INCLUDE_DIR)
find_package(Eigen3 3.0.0 REQUIRED)
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


#### FLAGS & DEFINES

if (UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-strict-aliasing -Wno-sign-compare")
elseif (WIN32)
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
    set_target_properties(wxhgversion PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W2")
endif (UNIX AND NOT APPLE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${wxWidgets_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DwxDEBUG_LEVEL=0 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG -D__WXDEBUG__")

add_definitions(-DTIXML_USE_TICPP -DTIXML_USE_STL -DAPP_CALIBRATOR -DMINIMAL_LINKS)

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


#### LIBRARIES LINKING

target_link_libraries(${CMAKE_PROJECT_NAME} ${wxWidgets_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${NetCDF_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${PROJ4_LIBRARIES})
if (USE_WXHGVERSION)
    target_link_libraries(${CMAKE_PROJECT_NAME} ${WXHGVERSION_NAME})
endif (USE_WXHGVERSION)
if (WIN32)
    if (USE_VLD)
        target_link_libraries(${CMAKE_PROJECT_NAME} ${VLD_LIBRARIES})
    endif (USE_VLD)
endif (WIN32)


#### INSTALLER
  
install (
  TARGETS ${CMAKE_PROJECT_NAME} 
  DESTINATION .
  )
  
if (WIN32)
	if (USE_PROVIDED_LIBRARIES)
		install (
		  PROGRAMS ${USE_PROVIDED_LIBRARIES_PATH}/gdal110.dll ${USE_PROVIDED_LIBRARIES_PATH}/libcurl.dll ${USE_PROVIDED_LIBRARIES_PATH}/netcdf.dll ${USE_PROVIDED_LIBRARIES_PATH}/zlib1.dll
		  DESTINATION .
		  )
	else (USE_PROVIDED_LIBRARIES)
		install (
		  PROGRAMS ../../bin/gdal110.dll ../../bin/libcurl.dll ../../bin/netcdf.dll ../../bin/zlib1.dll
		  DESTINATION .
		  )
	endif (USE_PROVIDED_LIBRARIES)
endif (WIN32)
  
install (
  FILES ../../license.txt ../../notice.txt
  DESTINATION .
  )
  
install (
  DIRECTORY ../../config
  DESTINATION .
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
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_LIST_DIR}/../_cpack/license.txt")
  set(CPACK_NSIS_DISPLAY_NAME "AtmoSwing Calibrator")
  set(CPACK_NSIS_CONTACT "Pascal Horton pascal.horton@terranum.ch")
  set(CPACK_NSIS_HELP_LINK "www.bitbucket.org/atmoswing/atmoswing")
  set(CPACK_NSIS_EXECUTABLES_DIRECTORY ".")
  set(CPACK_NSIS_MUI_FINISHPAGE_RUN "AtmoSwingCalibrator.exe")
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
  install(FILES "_cpack/atmoswing.png" DESTINATION share/pixmaps)
  install(FILES "_cpack/atmoswing-calibrator.desktop" DESTINATION share/applications)
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
  
 INCLUDE(CPack)