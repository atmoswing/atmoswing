cmake_minimum_required(VERSION 3.18) # To support FetchContent / FetchContent_MakeAvailable / SOURCE_SUBDIR

if (POLICY CMP0043)
    cmake_policy(SET CMP0043 NEW)
endif ()
if (POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif ()
if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif ()

# Project name
if (UNIX)
    project(atmoswing)
else (UNIX)
    project(AtmoSwing)
endif (UNIX)


##### DEPENDENCIES

include(FetchContent)

FetchContent_Declare(
        vroomgis
        GIT_REPOSITORY https://github.com/terranum-ch/vroomgis.git
        GIT_TAG 780cc0f52979a221ba1f96e42b8ab3cef48eca60
        SOURCE_SUBDIR vroomgis)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

include_directories(${CONAN_INCLUDE_DIRS})


##### OPTIONS

# Version number
set(VERSION_MAJOR "3")
set(VERSION_MINOR "0")
set(VERSION_PATCH "8")
add_definitions(-DATMOSWING_MAJOR_VERSION=${VERSION_MAJOR})
add_definitions(-DATMOSWING_MINOR_VERSION=${VERSION_MINOR})
add_definitions(-DATMOSWING_PATCH_VERSION=${VERSION_PATCH})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/build/cmake ${CMAKE_MODULE_PATH})

# Choice of the targets
option(BUILD_FORECASTER "Do you want to build AtmoSwing Forecaster ?" ON)
option(BUILD_VIEWER "Do you want to build AtmoSwing Viewer ?" ON)
option(BUILD_OPTIMIZER "Do you want to build AtmoSwing Optimizer ?" ON)
option(BUILD_DOWNSCALER "Do you want to build AtmoSwing Downscaler ?" ON)
option(CREATE_INSTALLER "Do you want to generate installer packages ?" OFF)
if (BUILD_FORECASTER OR BUILD_OPTIMIZER OR BUILD_DOWNSCALER)
    option(BUILD_TESTS "Do you want to build the tests (recommended) ?" ON)
    mark_as_advanced(CLEAR BUILD_TESTS)
else ()
    set(BUILD_TESTS OFF)
    mark_as_advanced(BUILD_TESTS)
endif ()
if (BUILD_OPTIMIZER)
    option(BUILD_BENCHMARK "Do you want to build the benchmark suite ?" OFF)
    mark_as_advanced(CLEAR BUILD_BENCHMARK)
else ()
    set(BUILD_BENCHMARK OFF)
    mark_as_advanced(BUILD_BENCHMARK)
endif ()

if (NOT BUILD_FORECASTER AND NOT BUILD_VIEWER AND NOT BUILD_OPTIMIZER AND NOT BUILD_DOWNSCALER)
    message(FATAL_ERROR "Please select one or multiple target(s) to build.")
endif ()

# Output path
if (WIN32)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
else (WIN32)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})
endif (WIN32)

# Docker option
option(ON_DOCKER "Enable only if AtmoSwing is built for Docker." OFF)
if (ON_DOCKER)
    add_definitions(-DON_DOCKER)
endif(ON_DOCKER)

# Using the Intel MKL
option(USE_MKL "Do you want to use Intel MKL ?" OFF)

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
    option(USE_VLD "Should we use Visual Leak Detector (https://vld.codeplex.com) ?" OFF)
else (WIN32)
    set(USE_VLD OFF)
endif (WIN32)

# Enable Cppcheck
option(USE_CPPCHECK "Should we use Cppcheck (http://cppcheck.sourceforge.net/) ?" OFF)

# Enable code coverage
if (CMAKE_COMPILER_IS_GNUCXX)
    option(USE_CODECOV "Should we do code coverage with lcov ?" OFF)
else (CMAKE_COMPILER_IS_GNUCXX)
    set(USE_CODECOV OFF)
endif ()

# Enable GUIs
if (BUILD_VIEWER)
    set(USE_GUI ON)
else (BUILD_VIEWER)
    option(USE_GUI "Should AtmoSwing be built with a GUI ?" OFF)
endif (BUILD_VIEWER)

if (USE_GUI)
    add_definitions(-DUSE_GUI)
endif(USE_GUI)

# Disable testing tree
set(BUILD_TESTING OFF)
mark_as_advanced(BUILD_TESTING)

# Installer
if (WIN32)
    set(INSTALL_DIR_BIN .)
else (WIN32)
    set(INSTALL_DIR_BIN bin)
endif (WIN32)


##### FIND ANALYSIS TOOLS

# Visual Leak Detector
if (USE_VLD)
    find_package(VLD)
    include_directories(${VLD_INCLUDE_DIRS})
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
    set(COVERAGE_EXCLUDES '/usr/*'
        '**/tests/*'
        '**/.conan/*'
        '**/_deps/*')
    setup_target_for_coverage(NAME atmoswing-coverage EXECUTABLE atmoswing-tests DEPENDENCIES atmoswing-tests)
endif (USE_CODECOV)


##### DISPLAY SOME INFORMATION

include(atmoswing_definitions_and_flags)

# Add components
add_subdirectory(src/shared_base)
if (BUILD_FORECASTER OR BUILD_OPTIMIZER OR BUILD_DOWNSCALER)
    add_subdirectory(src/shared_processing)
endif ()
if (BUILD_FORECASTER)
    add_subdirectory(src/app_forecaster)
endif()
if (BUILD_OPTIMIZER)
    add_subdirectory(src/app_optimizer)
endif()
if (BUILD_DOWNSCALER)
    add_subdirectory(src/app_downscaler)
endif()
if (BUILD_VIEWER)
    add_subdirectory(src/app_viewer)
endif()
if (BUILD_TESTS)
    add_subdirectory(tests)
endif ()
if (BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()


##### DISPLAY SOME INFORMATION

message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELWITHDEBINFO = ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")


#### INSTALLER

if (CREATE_INSTALLER)
    include(atmoswing_installer_options)
endif ()