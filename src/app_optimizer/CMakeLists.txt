
# Project name
project(optimizer)


##### OPTIONS

# Enable CUDA
if (BUILD_OPTIMIZER)
    option(USE_CUDA "Sould we compile with CUDA GPU support (not stable yet) ?" OFF)
    mark_as_advanced(CLEAR USE_CUDA)
else (BUILD_OPTIMIZER)
    set(USE_CUDA OFF)
    mark_as_advanced(USE_CUDA)
endif (BUILD_OPTIMIZER)


##### SOURCE FILES

# List source files
file(GLOB src_optim_core_h core/*.h)
file(GLOB src_optim_core_cpp core/*.cpp)
file(GLOB src_optim_gui_h gui/*.h)
file(GLOB src_optim_gui_cpp gui/*.cpp)
list(APPEND src_app_optimizer ${src_optim_core_h})
list(APPEND src_app_optimizer ${src_optim_core_cpp})
if (USE_GUI)
    list(APPEND src_app_optimizer ${src_optim_gui_h})
    list(APPEND src_app_optimizer ${src_optim_gui_cpp})
endif (USE_GUI)

# Remove eventual duplicates
list(REMOVE_DUPLICATES src_app_optimizer)

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
        add_executable(atmoswing-optimizer WIN32 ${src_app_optimizer} ${CMAKE_SOURCE_DIR}/src/shared_base/gui/resource.rc)
    else (USE_GUI)
        add_executable(atmoswing-optimizer ${src_app_optimizer})
    endif (USE_GUI)
elseif (UNIX AND NOT APPLE)
    add_executable(atmoswing-optimizer ${src_app_optimizer})
elseif (APPLE)
    add_executable(atmoswing-optimizer MACOSX_BUNDLE ${src_app_optimizer})
else (WIN32)
    add_executable(atmoswing-optimizer ${src_app_optimizer})
endif (WIN32)

target_include_directories(atmoswing-optimizer PRIVATE ${inc_dirs})

# Cppcheck
if (USE_CPPCHECK)
    add_cppcheck(atmoswing-optimizer STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
endif (USE_CPPCHECK)


##### DEFINITIONS

# APP_OPTIMIZER and MINIMAL_LINKS are target-specific. USE_GUI / USE_CUDA / version macros
# are inherited from atmoswing_compile_options via the asbase/asprocessing PUBLIC link chain.
target_compile_definitions(atmoswing-optimizer PRIVATE APP_OPTIMIZER MINIMAL_LINKS)


##### LINKING

target_link_libraries(atmoswing-optimizer PRIVATE asbase asprocessing)
if (USE_VLD)
    target_link_libraries(atmoswing-optimizer PRIVATE ${VLD_LIBRARIES})
endif (USE_VLD)
if (USE_CUDA)
    add_dependencies(atmoswing-optimizer ascuda)
    target_link_libraries(ascuda PRIVATE ${CUDA_LIBRARIES})
    target_link_libraries(atmoswing-optimizer PRIVATE ascuda)
endif (USE_CUDA)


##### RUNTIME DLL STAGING (Windows)

atmoswing_stage_runtime_dlls(atmoswing-optimizer)


##### INSTALLER

install(TARGETS atmoswing-optimizer DESTINATION ${INSTALL_DIR_BIN})
