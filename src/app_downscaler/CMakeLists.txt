
# Project name
project(downscaler)


##### SOURCE FILES

# List source files
file(GLOB src_downsc_core_h core/*.h)
file(GLOB src_downsc_core_cpp core/*.cpp)
file(GLOB src_downsc_gui_h gui/*.h)
file(GLOB src_downsc_gui_cpp gui/*.cpp)
list(APPEND src_app_downscaler ${src_downsc_core_h})
list(APPEND src_app_downscaler ${src_downsc_core_cpp})
if (USE_GUI)
    list(APPEND src_app_downscaler ${src_downsc_gui_h})
    list(APPEND src_app_downscaler ${src_downsc_gui_cpp})
endif (USE_GUI)

# Remove eventual duplicates
list(REMOVE_DUPLICATES src_app_downscaler)

# Include source directories
list(APPEND inc_dirs
        "${CMAKE_SOURCE_DIR}/src/shared_base/core/"
        "${CMAKE_SOURCE_DIR}/src/shared_processing/core/"
        "${PROJECT_SOURCE_DIR}/app/"
        "${PROJECT_SOURCE_DIR}/core/"
)
if (USE_GUI)
    list(APPEND inc_dirs
            "${CMAKE_SOURCE_DIR}/src/shared_base/gui/"
            "${CMAKE_SOURCE_DIR}/src/shared_base/libs/awxled/"
            "${PROJECT_SOURCE_DIR}/gui/"
    )
endif (USE_GUI)


##### DECLARE EXECUTABLE

if (WIN32)
    if (USE_GUI)
        add_executable(atmoswing-downscaler WIN32 ${src_app_downscaler} ${win_resources})
    else (USE_GUI)
        add_executable(atmoswing-downscaler ${src_app_downscaler})
    endif (USE_GUI)
elseif (UNIX AND NOT APPLE)
    add_executable(atmoswing-downscaler ${src_app_downscaler})
elseif (APPLE)
    add_executable(atmoswing-downscaler MACOSX_BUNDLE ${src_app_downscaler})
else (WIN32)
    add_executable(atmoswing-downscaler ${src_app_downscaler})
endif (WIN32)

target_include_directories(atmoswing-downscaler PRIVATE ${inc_dirs})

# Cppcheck
if (USE_CPPCHECK)
    add_cppcheck(atmoswing-downscaler STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
endif (USE_CPPCHECK)


##### DEFINITIONS

# APP_DOWNSCALER and MINIMAL_LINKS are target-specific. USE_GUI / USE_CUDA / version macros
# are inherited from atmoswing_compile_options via the asbase/asprocessing PUBLIC link chain.
target_compile_definitions(atmoswing-downscaler PRIVATE APP_DOWNSCALER MINIMAL_LINKS)


##### LINKING

target_link_libraries(atmoswing-downscaler PRIVATE asbase asprocessing)
if (USE_CUDA)
    add_dependencies(atmoswing-downscaler ascuda)
    target_link_libraries(atmoswing-downscaler PRIVATE ascuda)
endif (USE_CUDA)


##### RUNTIME DLL STAGING (Windows)

atmoswing_stage_runtime_dlls(atmoswing-downscaler)


##### INSTALLER

install(TARGETS atmoswing-downscaler DESTINATION ${INSTALL_DIR_BIN})
