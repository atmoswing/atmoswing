cmake_minimum_required(VERSION 3.0)

#### HUNTER
option(DOWNLOAD_LIBRARIES "Do you want to download the required libraries ?" OFF)
if (DOWNLOAD_LIBRARIES)
    include(build/cmake/HunterGate.cmake)
    set(HUNTER_ROOT ${CMAKE_BINARY_DIR}/hunter)
    HunterGate(
            URL "https://github.com/ruslo/hunter/archive/v0.19.14.tar.gz"
            SHA1 "d4216c82be9c273ce2567b35f75356da045b884f"
            FILEPATH "/home/phorton/Development/atmoswing/build/cmake/HunterConfig.cmake"
    )
endif (DOWNLOAD_LIBRARIES)

#### PROJECT
if (UNIX)
    project(atmoswing)
else (UNIX)
    project(AtmoSwing)
endif (UNIX)

# Version number
set(VERSION_MAJOR "1")
set(VERSION_MINOR "4")
set(VERSION_PATCH "2")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/build/cmake ${CMAKE_MODULE_PATH})

# Options
include(ASoptions)


#### SOURCE FILES LIST
include(ASlistSourceFiles)


#### INCLUDE DIRECTORIES
include(ASsourceIncludes)


#### LIBRARIES
include(ASgetLibraries)


#### DECLARE EXECUTABLE
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/build/cmake) # repeat
include(ASdeclareExecutable)


#### FLAGS & DEFINES
include(ASdefinitions)


#### LINKING
include(ASlinking)


#### TESTS
if (BUILD_TESTS)
    set(CTEST_TEST_TIMEOUT 3600)
    set(DART_TESTING_TIMEOUT 3600)
    set(CTEST_OUTPUT_ON_FAILURE TRUE)
    add_test(AtmoSwing-tests atmoswing-tests)
    include(CTest)
    enable_testing()
endif (BUILD_TESTS)


#### INSTALLER
include(ASinstaller)
