
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
file(GLOB_RECURSE src_optim_core_h core/*.h)
file(GLOB_RECURSE src_optim_core_cpp core/*.cpp)
file(GLOB_RECURSE src_optim_gui_h gui/*.h)
file(GLOB_RECURSE src_optim_gui_cpp gui/*.cpp)
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
            "${CMAKE_SOURCE_DIR}/src/shared_base/gui/img/"
            "${CMAKE_SOURCE_DIR}/src/shared_base/libs/awxled/"
            "${PROJECT_SOURCE_DIR}/gui/"
            )
endif (USE_GUI)
include_directories(${inc_dirs})


##### LIBRARIES

include(define_common_libraries)


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

# Cppcheck
if (USE_CPPCHECK)
    add_cppcheck(atmoswing-optimizer STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
endif (USE_CPPCHECK)


##### DEFINITIONS

# Global definitions
if (USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif (USE_CUDA)

if (USE_GUI)
    set_target_properties(atmoswing-optimizer PROPERTIES COMPILE_DEFINITIONS "APP_OPTIMIZER; MINIMAL_LINKS; wxUSE_GUI=1")
    set_target_properties(asprocessing PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=1")
else (USE_GUI)
    set_target_properties(atmoswing-optimizer PROPERTIES COMPILE_DEFINITIONS "APP_OPTIMIZER; MINIMAL_LINKS; wxUSE_GUI=0")
    set_target_properties(asprocessing PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=0")
endif (USE_GUI)


##### LINKING

target_link_libraries(atmoswing-optimizer asbase)
target_link_libraries(atmoswing-optimizer asprocessing)
if (USE_WXHGVERSION)
    target_link_libraries(atmoswing-optimizer ${WXHGVERSION_NAME})
endif (USE_WXHGVERSION)
if (USE_VLD)
    target_link_libraries(atmoswing-optimizer ${VLD_LIBRARIES})
endif (USE_VLD)
if (USE_CUDA)
    add_dependencies(atmoswing-optimizer ascuda)
    target_link_libraries(ascuda ${CUDA_LIBRARIES})
    target_link_libraries(atmoswing-optimizer ascuda)
endif (USE_CUDA)


##### INSTALLER

install(TARGETS atmoswing-optimizer DESTINATION ${INSTALL_DIR_BIN})
