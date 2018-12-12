cmake_minimum_required(VERSION 2.8.12.2) # Bug with VS 2012 and CUDA in previous versions

# Project name
if (UNIX)
    project(atmoswing)
else (UNIX)
    project(AtmoSwing)
endif (UNIX)


##### OPTIONS

# Version number
set(VERSION_MAJOR "2")
set(VERSION_MINOR "0")
set(VERSION_PATCH "1")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/build/cmake ${CMAKE_MODULE_PATH})

# Choice of the targets
option(BUILD_FORECASTER "Do you want to build AtmoSwing Forecaster ?" ON)
option(BUILD_VIEWER "Do you want to build AtmoSwing Viewer ?" ON)
option(BUILD_OPTIMIZER "Do you want to build AtmoSwing Optimizer ?" ON)
option(BUILD_DOWNSCALER "Do you want to build AtmoSwing Downscaler ?" ON)
option(CREATE_INSTALLER "Do you want to generate installer packages ?" ON)
if (BUILD_FORECASTER OR BUILD_OPTIMIZER OR BUILD_DOWNSCALER)
    set(BUILD_TESTS ON CACHE BOOL "Do you want to build the tests (recommended) ?" )
    mark_as_advanced(CLEAR BUILD_TESTS)
else (BUILD_FORECASTER OR BUILD_OPTIMIZER OR BUILD_DOWNSCALER)
    set(BUILD_TESTS OFF)
    mark_as_advanced(BUILD_TESTS)
endif (BUILD_FORECASTER OR BUILD_OPTIMIZER OR BUILD_DOWNSCALER)

if (NOT BUILD_FORECASTER AND NOT BUILD_VIEWER AND NOT BUILD_OPTIMIZER AND NOT BUILD_DOWNSCALER)
    message(FATAL_ERROR "Please select one or multiple target(s) to build.")
endif (NOT BUILD_FORECASTER AND NOT BUILD_VIEWER AND NOT BUILD_OPTIMIZER AND NOT BUILD_DOWNSCALER)

# Output path
if (WIN32)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
else (WIN32)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})
endif (WIN32)

# Libraries version
set(EIGEN_VERSION 3.3.5)

# MSYS condition
if (WIN32)
    option(USE_MSYS2 "Do you want to use MSYS2 ?" OFF)
    if(USE_MSYS2)
        set(MINGW false)
        set(MSYS true)
        set(MINGW_PATH "C:/msys64/mingw64" CACHE PATH "Path to installed libraries in MINGW")
    endif()
endif ()

# Enable Visual Leak Detector
if (WIN32)
    option(USE_VLD "Sould we use Visual Leak Detector (https://vld.codeplex.com) ?" OFF)
else (WIN32)
    set(USE_VLD OFF)
endif (WIN32)

# Enable Cppcheck
option(USE_CPPCHECK "Sould we use Cppcheck (http://cppcheck.sourceforge.net/) ?" OFF)

# Enable code coverage
if (CMAKE_COMPILER_IS_GNUCXX)
    option(USE_CODECOV "Sould we do code coverage with lcov ?" OFF)
else (CMAKE_COMPILER_IS_GNUCXX)
    set(USE_CODECOV OFF)
endif ()

# Enable GUIs
if (BUILD_VIEWER)
    option(USE_GUI "Sould we build AtmoSwing with a GUI ?" ON)
else (BUILD_VIEWER)
    set(USE_GUI OFF)
endif (BUILD_VIEWER)

# External projects
include(ExternalProject)
set(EXTERNAL_DIR ${CMAKE_BINARY_DIR}/external)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${EXTERNAL_DIR}")
include_directories(${EXTERNAL_DIR}/include)
link_directories(${EXTERNAL_DIR}/lib)
link_directories(${EXTERNAL_DIR}/lib64)

# Installer
set(INSTALL_DIR_BIN bin)
set(INSTALL_DIR_SHARE share/atmoswing)


##### FIND ANALYSIS TOOLS

# Visual Leak Detector
if (USE_VLD)
    find_package(VLD)
    include_directories(${VLD_INCLUDE_DIRS})
else (USE_VLD)
    # unset for wxhgversion
    unset(VLD_INCLUDE_DIR CACHE)
    unset(VLD_LIBRARY CACHE)
    unset(VLD_LIBRARY_DEBUG CACHE)
    unset(VLD_LIBRARY_DIR CACHE)
    unset(VLD_ROOT_DIR CACHE)
endif (USE_VLD)

# Cppcheck
if (USE_CPPCHECK)
    include(build/cmake/Findcppcheck.cmake)
    include(build/cmake/CppcheckTargets.cmake)
endif (USE_CPPCHECK)

# Code coverage
if (USE_CODECOV)
    include(CodeCoverage)
    append_coverage_compiler_flags()
    set(COVERAGE_EXCLUDES '/usr/.*' '.*/external/.*' 'tests/.*' '.*/tests/.*' 'tests/src/*' '.*/tests/src/*' '.*/libs/.*')
    setup_target_for_coverage(NAME atmoswing-coverage EXECUTABLE atmoswing-tests DEPENDENCIES atmoswing-tests)
endif (USE_CODECOV)


##### EXTERNAL PROJECTS

# Eigen
ExternalProject_Add(eigen
        URL http://bitbucket.org/eigen/eigen/get/${EIGEN_VERSION}.tar.gz
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND
        ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/eigen-prefix/src/eigen/Eigen
        ${EXTERNAL_DIR}/include/Eigen
        )


##### DISPLAY SOME INFORMATIONS

include(define_shared_compilation_flags)

# Add components
add_subdirectory(src/shared_base)
if (BUILD_FORECASTER OR BUILD_OPTIMIZER OR BUILD_DOWNSCALER)
    add_subdirectory(src/shared_processing)
endif (BUILD_FORECASTER OR BUILD_OPTIMIZER OR BUILD_DOWNSCALER)
if (BUILD_FORECASTER)
    add_subdirectory(src/app_forecaster)
endif(BUILD_FORECASTER)
if (BUILD_OPTIMIZER)
    add_subdirectory(src/app_optimizer)
endif(BUILD_OPTIMIZER)
if (BUILD_DOWNSCALER)
    add_subdirectory(src/app_downscaler)
endif(BUILD_DOWNSCALER)
if (BUILD_VIEWER)
    add_subdirectory(src/app_viewer)
endif(BUILD_VIEWER)
if (BUILD_TESTS)
    add_subdirectory(tests)
endif (BUILD_TESTS)


##### DISPLAY SOME INFORMATIONS

message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELWITHDEBINFO = ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")


#### INSTALLER

if (CREATE_INSTALLER)
    include(define_installer_options)
endif ()