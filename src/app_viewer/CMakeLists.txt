
# Project name
project(viewer)


##### DEPENDENCIES

find_package(GDAL REQUIRED)
find_package(PNG REQUIRED)
find_package(PROJ CONFIG REQUIRED)

##### SOURCE FILES

# List source files
file(GLOB src_viewer_core_h core/*.h)
file(GLOB src_viewer_core_cpp core/*.cpp)
file(GLOB src_viewer_gui_h gui/*.h)
file(GLOB src_viewer_gui_cpp gui/*.cpp)
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
file(GLOB src_lib_wxmathplot_h libs/wxmathplot/*.h)
file(GLOB src_lib_wxmathplot_cpp libs/wxmathplot/*.cpp)
file(GLOB src_lib_wxplotctrl_h libs/wxplotctrl/src/*.h)
file(GLOB src_lib_wxplotctrl_hh libs/wxplotctrl/src/*.hh)
file(GLOB src_lib_wxplotctrl_cpp libs/wxplotctrl/src/*.cpp)
file(GLOB src_lib_wxplotctrl_c libs/wxplotctrl/src/*.c)
file(GLOB src_lib_wxthings_cpp libs/wxthings/src/*.cpp)
list(APPEND src_lib_wxplotctrl ${src_lib_wxmathplot_h} ${src_lib_wxplotctrl_h} ${src_lib_wxplotctrl_hh})
list(APPEND src_lib_wxplotctrl ${src_lib_wxmathplot_cpp} ${src_lib_wxplotctrl_cpp} ${src_lib_wxplotctrl_c} ${src_lib_wxthings_cpp})

# Remove eventual duplicates
list(REMOVE_DUPLICATES src_app_viewer)

# Include source directories
list(APPEND inc_dirs
        "${CMAKE_SOURCE_DIR}/src/shared_base/core/"
        "${CMAKE_SOURCE_DIR}/src/shared_base/gui/"
        "${CMAKE_SOURCE_DIR}/src/shared_base/libs/awxled/"
        "${CMAKE_SOURCE_DIR}/src/app_forecaster/core/"
        "${PROJECT_SOURCE_DIR}/core/"
        "${PROJECT_SOURCE_DIR}/gui/"
        "${PROJECT_SOURCE_DIR}/libs/wxmathplot/"
        "${PROJECT_SOURCE_DIR}/libs/wxplotctrl/"
        "${PROJECT_SOURCE_DIR}/libs/wxplotctrl/include/"
        "${PROJECT_SOURCE_DIR}/libs/wxthings/include/"
)


##### LIBRARIES

# wxplotctrl (uses wxWidgets headers from atmoswing_compile_options).
add_library(wxplotctrl STATIC ${src_lib_wxplotctrl})
target_include_directories(wxplotctrl PUBLIC
        "${PROJECT_SOURCE_DIR}/libs/wxmathplot/"
        "${PROJECT_SOURCE_DIR}/libs/wxplotctrl/"
        "${PROJECT_SOURCE_DIR}/libs/wxplotctrl/include/"
        "${PROJECT_SOURCE_DIR}/libs/wxthings/include/")
target_link_libraries(wxplotctrl PRIVATE atmoswing_compile_options)


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

target_include_directories(atmoswing-viewer PRIVATE
        ${inc_dirs}
        ${vroomgis_SOURCE_DIR}/vroomgis/src
        ${vroomgis_SOURCE_DIR}/vroomgis/art)

# Cppcheck
if (USE_CPPCHECK)
    add_cppcheck(atmoswing-viewer STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
endif (USE_CPPCHECK)


##### DEFINITIONS

# Compilation flags
if (MINGW OR MSYS OR UNIX AND NOT APPLE)
    target_compile_options(vroomgis PRIVATE -Wno-unused-but-set-variable)
    target_compile_options(wxplotctrl PRIVATE -Wno-unused-but-set-variable -Wno-attributes)
elseif (WIN32)
    target_compile_options(vroomgis PRIVATE /W2)
    target_compile_options(wxplotctrl PRIVATE /W2)
endif ()

# APP_VIEWER is target-specific. USE_GUI / version macros come from atmoswing_compile_options
# via asbase's PUBLIC link.
target_compile_definitions(atmoswing-viewer PRIVATE APP_VIEWER)


##### LINKING

add_dependencies(atmoswing-viewer vroomgis)
target_link_libraries(atmoswing-viewer PRIVATE PNG::PNG GDAL::GDAL PROJ::proj asbase vroomgis wxplotctrl)
if (WIN32)
    target_link_libraries(atmoswing-viewer PRIVATE Psapi.lib)
endif (WIN32)


##### RUNTIME DLL STAGING (Windows)

atmoswing_stage_runtime_dlls(atmoswing-viewer)


##### INSTALLER

install(TARGETS atmoswing-viewer DESTINATION ${INSTALL_DIR_BIN})

