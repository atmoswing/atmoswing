
# Project name
project(viewer)


##### SOURCE FILES

# List source files
file(GLOB_RECURSE src_viewer_core_h core/*.h)
file(GLOB_RECURSE src_viewer_core_cpp core/*.cpp)
file(GLOB_RECURSE src_viewer_gui_h gui/*.h)
file(GLOB_RECURSE src_viewer_gui_cpp gui/*.cpp)
list(APPEND src_app_viewer ${src_viewer_core_h} ${src_viewer_gui_h})
list(APPEND src_app_viewer ${src_viewer_core_cpp} ${src_viewer_gui_cpp})
list(APPEND src_app_viewer
        ${CMAKE_SOURCE_DIR}/src/app_forecaster/core/asResultsForecast.h
        ${CMAKE_SOURCE_DIR}/src/app_forecaster/core/asResultsForecast.cpp
        ${CMAKE_SOURCE_DIR}/src/app_forecaster/core/asResultsForecastAggregator.h
        ${CMAKE_SOURCE_DIR}/src/app_forecaster/core/asResultsForecastAggregator.cpp
        ${CMAKE_SOURCE_DIR}/src/app_forecaster/core/asFileParametersForecast.h
        ${CMAKE_SOURCE_DIR}/src/app_forecaster/core/asFileParametersForecast.cpp
        )

# wxPlotCtrl library (and dependences)
file(GLOB_RECURSE src_lib_wxmathplot_h libs/wxmathplot/*.h)
file(GLOB_RECURSE src_lib_wxmathplot_cpp libs/wxmathplot/*.cpp)
file(GLOB_RECURSE src_lib_wxplotctrl_h libs/wxplotctrl/src/*.h)
file(GLOB_RECURSE src_lib_wxplotctrl_hh libs/wxplotctrl/src/*.hh)
file(GLOB_RECURSE src_lib_wxplotctrl_cpp libs/wxplotctrl/src/*.cpp)
file(GLOB_RECURSE src_lib_wxplotctrl_c libs/wxplotctrl/src/*.c)
file(GLOB_RECURSE src_lib_wxthings_cpp libs/wxthings/src/*.cpp)
list(APPEND src_lib_wxplotctrl ${src_lib_wxmathplot_h} ${src_lib_wxplotctrl_h} ${src_lib_wxplotctrl_hh})
list(APPEND src_lib_wxplotctrl ${src_lib_wxmathplot_cpp} ${src_lib_wxplotctrl_cpp} ${src_lib_wxplotctrl_c} ${src_lib_wxthings_cpp})

# Remove eventual duplicates
list(REMOVE_DUPLICATES src_app_viewer)

# Include source directories
list(APPEND inc_dirs
        "${CMAKE_SOURCE_DIR}/src/shared_base/core/"
        "${CMAKE_SOURCE_DIR}/src/shared_base/gui/"
        "${CMAKE_SOURCE_DIR}/src/shared_base/gui/img/"
        "${CMAKE_SOURCE_DIR}/src/shared_base/libs/awxled/"
        "${CMAKE_SOURCE_DIR}/src/app_forecaster/core/"
        "${PROJECT_SOURCE_DIR}/core/"
        "${PROJECT_SOURCE_DIR}/gui/"
        "${PROJECT_SOURCE_DIR}/libs/wxmathplot/"
        "${PROJECT_SOURCE_DIR}/libs/wxplotctrl/"
        "${PROJECT_SOURCE_DIR}/libs/wxplotctrl/include/"
        "${PROJECT_SOURCE_DIR}/libs/wxthings/include/"
        )
include_directories(${inc_dirs})


##### LIBRARIES

include(define_common_libraries)

# vroomgis
set(CMAKE_INSTALL_PREFIX:STRING ${EXTERNAL_DIR})
set(SEARCH_VROOMGIS_LIBS:BOOL ON)
set(SEARCH_GDAL:BOOL ON)
set(SEARCH_GEOS:BOOL OFF)
set(VROOMGIS_PATH:STRING vroomgis/src)
set(SEARCH_GIS_LIB_PATH:STRING ${GDAL_ROOT})
configure_file(${CMAKE_SOURCE_DIR}/build/cmake/GetVroomgis.txt ${CMAKE_BINARY_DIR}/vroomgis/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" . WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/vroomgis")
execute_process(COMMAND "${CMAKE_COMMAND}" --build . WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/vroomgis")
add_subdirectory(${CMAKE_BINARY_DIR}/vroomgis/vroomgis-prefix/src/vroomgis/vroomgis ${EXTERNAL_DIR})
include_directories(${CMAKE_BINARY_DIR}/vroomgis/vroomgis-prefix/src/vroomgis/vroomgis/src)
include_directories(${CMAKE_BINARY_DIR}/vroomgis/vroomgis-prefix/src/vroomgis/vroomgis/art)

# wxplotctrl
add_library(wxplotctrl STATIC ${src_lib_wxplotctrl})


##### DECLARE EXECUTABLE

if (WIN32)
    add_executable(atmoswing-viewer WIN32 ${src_app_viewer} ${CMAKE_SOURCE_DIR}/src/shared_base/gui/resource.rc)
elseif (UNIX AND NOT APPLE)
    add_executable(atmoswing-viewer ${src_app_viewer})
elseif (APPLE)
    add_executable(atmoswing-viewer MACOSX_BUNDLE ${src_app_viewer})
else (WIN32)
    add_executable(atmoswing-viewer ${src_app_viewer})
endif (WIN32)

# Cppcheck
if (USE_CPPCHECK)
    add_cppcheck(atmoswing-viewer STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
endif (USE_CPPCHECK)


##### DEFINITIONS

# Compilation flags
if (MINGW OR MSYS OR UNIX AND NOT APPLE)
    set_target_properties(vroomgis PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
    set_target_properties(wxplotctrl PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable -Wno-attributes")
elseif (WIN32)
    set_target_properties(vroomgis PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} /W2")
    set_target_properties(wxplotctrl PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} /W2")
endif ()

# Specific definitions
set_target_properties(atmoswing-viewer PROPERTIES COMPILE_DEFINITIONS "APP_VIEWER")


##### LINKING

add_dependencies(atmoswing-viewer vroomgis)
target_link_libraries(atmoswing-viewer ${GDAL_LIBRARIES})
target_link_libraries(atmoswing-viewer asbase)
target_link_libraries(atmoswing-viewer vroomgis)
target_link_libraries(atmoswing-viewer ${VROOMGIS_LIBRARY})
target_link_libraries(atmoswing-viewer wxplotctrl)
if (USE_WXHGVERSION)
    add_dependencies(atmoswing-viewer wxhgversion)
    target_link_libraries(atmoswing-viewer ${WXHGVERSION_NAME})
endif (USE_WXHGVERSION)
if (USE_VLD)
    target_link_libraries(atmoswing-viewer ${VLD_LIBRARIES})
endif (USE_VLD)
if (WIN32)
    target_link_libraries(atmoswing-viewer Psapi.lib)
endif (WIN32)


##### INSTALLER

install(TARGETS atmoswing-viewer DESTINATION ${INSTALL_DIR_BIN})

# Dlls
if (WIN32)

    find_file(GDAL_DLL_NAME
            NAMES gdal200.dll gdal201.dll gdal202.dll gdal203.dll
            HINTS ${CMAKE_PREFIX_PATH}
            PATH_SUFFIXES bin
            NO_DEFAULT_PATH)
    if (NOT GDAL_DLL_NAME)
        message(SEND_ERROR "GDAL DLL not found.")
    endif ()
    message(STATUS "GDAL DLL: ${GDAL_DLL_NAME}")
    add_custom_command(TARGET atmoswing-viewer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GDAL_DLL_NAME}"
            "${CMAKE_BINARY_DIR}")
    install(FILES ${GDAL_DLL_NAME} DESTINATION bin)

endif (WIN32)