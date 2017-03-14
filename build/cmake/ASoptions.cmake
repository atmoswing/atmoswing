# Choice of the targets
set(BUILD_FORECASTER OFF CACHE BOOL "Do you want to build AtmoSwing Forecaster ?" )
set(BUILD_VIEWER OFF CACHE BOOL "Do you want to build AtmoSwing Viewer ?" )
set(BUILD_OPTIMIZER OFF CACHE BOOL "Do you want to build AtmoSwing Optimizer ?" )
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

# MSYS condition
if (WIN32)
    set(USE_MSYS2 ON CACHE BOOL "Do you want to use MSYS2 ?" )
    if(USE_MSYS2)
        set(MINGW false)
        set(MSYS true)
        set(MINGW_PATH "C:/msys64/mingw64" CACHE PATH "Path to installed libraries in MINGW")
    endif()
endif ()

# Enable Visual Leak Detector
if (WIN32)
    set(USE_VLD OFF CACHE BOOL "Sould we use Visual Leak Detector (https://vld.codeplex.com) ?" )
else (WIN32)
    set(USE_VLD OFF)
endif (WIN32)

# Enable Cppcheck
set(USE_CPPCHECK OFF CACHE BOOL "Sould we use Cppcheck (http://cppcheck.sourceforge.net/) ?" )

# Enable GUIs
if (BUILD_FORECASTER OR BUILD_OPTIMIZER AND NOT BUILD_VIEWER)
    set(USE_GUI OFF CACHE BOOL "Sould we build the Forecaster / Optimizer with a GUI ?" )
else (BUILD_FORECASTER OR BUILD_OPTIMIZER AND NOT BUILD_VIEWER)
    set(USE_GUI ON)
endif (BUILD_FORECASTER OR BUILD_OPTIMIZER AND NOT BUILD_VIEWER)

# Enable CUDA
if (BUILD_OPTIMIZER)
    set(USE_CUDA OFF CACHE BOOL "Sould we compile with CUDA GPU support (not stable yet) ?" )
    mark_as_advanced(CLEAR USE_CUDA)
else (BUILD_OPTIMIZER)
    set(USE_CUDA OFF)
    mark_as_advanced(USE_CUDA)
endif (BUILD_OPTIMIZER)