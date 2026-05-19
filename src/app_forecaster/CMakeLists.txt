
# Project name
project(forecaster)


##### DEPENDENCIES

find_package(OpenSSL REQUIRED)


##### SOURCE FILES

# List source files
file(GLOB src_forec_core_h core/*.h)
file(GLOB src_forec_core_cpp core/*.cpp)
file(GLOB src_forec_gui_h gui/*.h)
file(GLOB src_forec_gui_cpp gui/*.cpp)
list(APPEND src_app_forecaster ${src_forec_core_h})
list(APPEND src_app_forecaster ${src_forec_core_cpp})
if (USE_GUI)
    list(APPEND src_app_forecaster ${src_forec_gui_h})
    list(APPEND src_app_forecaster ${src_forec_gui_cpp})
endif (USE_GUI)

# Remove eventual duplicates
list(REMOVE_DUPLICATES src_app_forecaster)

# Include source directories
list(APPEND inc_dirs
        "${CMAKE_SOURCE_DIR}/src/shared_base/core/"
        "${CMAKE_SOURCE_DIR}/src/shared_processing/core/"
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
        add_executable(atmoswing-forecaster WIN32 ${src_app_forecaster} ${CMAKE_SOURCE_DIR}/src/shared_base/gui/resource.rc)
    else (USE_GUI)
        add_executable(atmoswing-forecaster ${src_app_forecaster})
    endif (USE_GUI)
elseif (UNIX AND NOT APPLE)
    add_executable(atmoswing-forecaster ${src_app_forecaster})
elseif (APPLE)
    add_executable(atmoswing-forecaster MACOSX_BUNDLE ${src_app_forecaster})
else (WIN32)
    add_executable(atmoswing-forecaster ${src_app_forecaster})
endif (WIN32)

target_include_directories(atmoswing-forecaster PRIVATE ${inc_dirs})

# Cppcheck
if (USE_CPPCHECK)
    add_cppcheck(atmoswing-forecaster STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
endif (USE_CPPCHECK)


##### DEFINITIONS

# APP_FORECASTER is the only target-specific define here. USE_GUI / USE_CUDA / version macros
# are inherited from atmoswing_compile_options via the asbase/asprocessing PUBLIC link chain.
target_compile_definitions(atmoswing-forecaster PRIVATE APP_FORECASTER)


##### LINKING

target_link_libraries(atmoswing-forecaster PRIVATE asbase asprocessing OpenSSL::SSL)
if (USE_GUI)
    target_link_libraries(atmoswing-forecaster PRIVATE wx::stc)
endif ()
if (USE_VLD)
    target_link_libraries(atmoswing-forecaster PRIVATE ${VLD_LIBRARIES})
endif (USE_VLD)
if (USE_CUDA)
    add_dependencies(atmoswing-forecaster ascuda)
    target_link_libraries(atmoswing-forecaster PRIVATE ascuda)
endif (USE_CUDA)


##### RUNTIME DLL STAGING (Windows)

atmoswing_stage_runtime_dlls(atmoswing-forecaster)


##### INSTALLER

install(TARGETS atmoswing-forecaster DESTINATION ${INSTALL_DIR_BIN})
