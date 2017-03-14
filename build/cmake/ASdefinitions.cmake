# Compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${wxWidgets_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DwxDEBUG_LEVEL=0 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG -DwxDEBUG_LEVEL=1 -D__WXDEBUG__")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DwxDEBUG_LEVEL=0 -DNDEBUG ")

if (MINGW OR MSYS OR UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-strict-aliasing -Wno-sign-compare -Wno-attributes")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer ")
    if (BUILD_VIEWER)
        set_target_properties(vroomgis PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
        set_target_properties(wxplotctrl PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable -Wno-attributes")
    endif (BUILD_VIEWER)
    if (USE_WXHGVERSION)
        set_target_properties(wxhgversion PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
    endif (USE_WXHGVERSION)
elseif (WIN32)
    set_target_properties(asbase PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
    if (BUILD_FORECASTER)
        set_target_properties(atmoswing-forecaster PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
        set_target_properties(asprocessing PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
    endif (BUILD_FORECASTER)
    if (BUILD_VIEWER)
        set_target_properties(atmoswing-viewer PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
        set_target_properties(vroomgis PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W2")
        set_target_properties(wxplotctrl PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W2")
    endif (BUILD_VIEWER)
    if (BUILD_OPTIMIZER)
        set_target_properties(atmoswing-optimizer PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
        set_target_properties(asprocessing PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
    endif (BUILD_OPTIMIZER)
    if (BUILD_TESTS)
        set_target_properties(atmoswing-tests PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
    endif (BUILD_TESTS)
    if (USE_WXHGVERSION)
        set_target_properties(wxhgversion PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W2")
    endif (USE_WXHGVERSION)
endif ()

# Global definitions
add_definitions(-DUSE_JPEG2000)
add_definitions(-DEIGEN_NO_DEBUG)
#add_definitions(-std=c++11)

if (WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_CRTDBG_MAP_ALLOC)
endif (WIN32)

if (USE_VLD)
    add_definitions(-DUSE_VLD)
endif (USE_VLD)

if (USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif (USE_CUDA)

# Specific definitions
set_target_properties(g2clib PROPERTIES COMPILE_DEFINITIONS "USE_JPEG2000; USE_PNG")

if (USE_GUI)
    set_target_properties(asbase PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=1")
else (USE_GUI)
    set_target_properties(asbase PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=0")
endif (USE_GUI)

if (BUILD_FORECASTER)
    if (USE_GUI)
        set_target_properties(atmoswing-forecaster PROPERTIES COMPILE_DEFINITIONS "APP_FORECASTER; wxUSE_GUI=1")
        set_target_properties(asprocessing PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=1")
    else (USE_GUI)
        set_target_properties(atmoswing-forecaster PROPERTIES COMPILE_DEFINITIONS "APP_FORECASTER; wxUSE_GUI=0")
        set_target_properties(asprocessing PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=0")
    endif (USE_GUI)
endif (BUILD_FORECASTER)

if (BUILD_VIEWER)
    set_target_properties(atmoswing-viewer PROPERTIES COMPILE_DEFINITIONS "APP_VIEWER")
endif (BUILD_VIEWER)

if (BUILD_OPTIMIZER)
    if (USE_GUI)
        set_target_properties(atmoswing-optimizer PROPERTIES COMPILE_DEFINITIONS "APP_OPTIMIZER; MINIMAL_LINKS; wxUSE_GUI=1")
        set_target_properties(asprocessing PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=1")
    else (USE_GUI)
        set_target_properties(atmoswing-optimizer PROPERTIES COMPILE_DEFINITIONS "APP_OPTIMIZER; MINIMAL_LINKS; wxUSE_GUI=0")
        set_target_properties(asprocessing PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=0")
    endif (USE_GUI)
endif (BUILD_OPTIMIZER)

if (BUILD_TESTS)
    if(WIN32)
        set_target_properties(atmoswing-tests PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
        set_target_properties(atmoswing-tests PROPERTIES COMPILE_DEFINITIONS "UNIT_TESTING; wxUSE_GUI=0; _CONSOLE")
    else(WIN32)
        set_target_properties(atmoswing-tests PROPERTIES COMPILE_DEFINITIONS "UNIT_TESTING; wxUSE_GUI=0")
    endif(WIN32)
endif (BUILD_TESTS)

message("CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message("CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")
message("CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}")
message("CMAKE_CXX_FLAGS_RELWITHDEBINFO = ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")