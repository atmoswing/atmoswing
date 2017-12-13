cmake_minimum_required(VERSION 2.8.12.2) # Bug with VS 2012 and CUDA in previous versions

# Project name
if (UNIX)
    project(atmoswing)
else (UNIX)
    project(AtmoSwing)
endif (UNIX)


##### OPTIONS

# Version number
set(VERSION_MAJOR "1")
set(VERSION_MINOR "4")
set(VERSION_PATCH "2")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/build/cmake ${CMAKE_MODULE_PATH})

# Choice of the targets
option(BUILD_FORECASTER "Do you want to build AtmoSwing Forecaster ?" OFF)
option(BUILD_VIEWER "Do you want to build AtmoSwing Viewer ?" OFF)
option(BUILD_OPTIMIZER "Do you want to build AtmoSwing Optimizer ?" OFF)
if (BUILD_FORECASTER OR BUILD_OPTIMIZER)
    set(BUILD_TESTS ON CACHE BOOL "Do you want to build the tests (recommended) ?" )
    mark_as_advanced(CLEAR BUILD_TESTS)
else (BUILD_FORECASTER OR BUILD_OPTIMIZER)
    set(BUILD_TESTS OFF)
    mark_as_advanced(BUILD_TESTS)
endif (BUILD_FORECASTER OR BUILD_OPTIMIZER)

if (NOT BUILD_FORECASTER AND NOT BUILD_VIEWER AND NOT BUILD_OPTIMIZER)
    message(FATAL_ERROR "Please select one or multiple target(s) to build.")
endif (NOT BUILD_FORECASTER AND NOT BUILD_VIEWER AND NOT BUILD_OPTIMIZER)

# Output path
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# Libraries version
set(EIGEN_VERSION 3.3.4)

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
if (BUILD_FORECASTER OR BUILD_OPTIMIZER AND NOT BUILD_VIEWER)
    option(USE_GUI "Sould we build the Forecaster / Optimizer with a GUI ?" OFF)
else (BUILD_FORECASTER OR BUILD_OPTIMIZER AND NOT BUILD_VIEWER)
    set(USE_GUI ON)
endif (BUILD_FORECASTER OR BUILD_OPTIMIZER AND NOT BUILD_VIEWER)

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
if (WIN32)
    set(INSTALL_DIR_BIN .)
    set(INSTALL_DIR_SHARE .)
endif (WIN32)


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
    setup_target_for_coverage(${PROJECT_NAME}-coverage atmoswing-tests coverage)
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


##### COMPONENTS

if (WIN32)
    set(DLL_DIR ${CMAKE_PREFIX_PATH})
    foreach(PATH ${CMAKE_PREFIX_PATH})
        list(APPEND DLL_DIRS ${PATH}/bin)
    endforeach()
endif ()


##### DISPLAY SOME INFORMATIONS

include(define_shared_compilation_flags)

# Add components
add_subdirectory(src/shared_base)
if (BUILD_FORECASTER OR BUILD_OPTIMIZER)
    add_subdirectory(src/shared_processing)
endif (BUILD_FORECASTER OR BUILD_OPTIMIZER)
if (BUILD_FORECASTER)
    add_subdirectory(src/app_forecaster)
endif(BUILD_FORECASTER)
if (BUILD_OPTIMIZER)
    add_subdirectory(src/app_optimizer)
endif(BUILD_OPTIMIZER)
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

include(define_installer_options)
