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
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BUILD_TYPE})

# Libraries
set(EIGEN_VERSION 3.3.4)
set(GDAL_VERSION 2.2.1)

# Libraries paths
if (BUILD_VIEWER)
    set(GDAL_ROOT CACHE PATH "Path to installed Gdal libraries")
endif (BUILD_VIEWER)

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

# Enable CUDA
if (BUILD_OPTIMIZER)
    option(USE_CUDA "Sould we compile with CUDA GPU support (not stable yet) ?" OFF)
    mark_as_advanced(CLEAR USE_CUDA)
else (BUILD_OPTIMIZER)
    set(USE_CUDA OFF)
    mark_as_advanced(USE_CUDA)
endif (BUILD_OPTIMIZER)
