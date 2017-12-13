
# Project name
project(forecaster)


##### SOURCE FILES

# List source files
file(GLOB_RECURSE src_forec_core_h core/*.h)
file(GLOB_RECURSE src_forec_core_cpp core/*.cpp)
file(GLOB_RECURSE src_forec_gui_h gui/*.h)
file(GLOB_RECURSE src_forec_gui_cpp gui/*.cpp)
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
            "${CMAKE_SOURCE_DIR}/src/shared_base/gui/img/"
            "${CMAKE_SOURCE_DIR}/src/shared_base/libs/awxled/"
            "${PROJECT_SOURCE_DIR}/gui/"
            )
endif (USE_GUI)
include_directories(${inc_dirs})


##### LIBRARIES

include(define_common_libraries)

# libcURL
mark_as_advanced(CLEAR CURL_INCLUDE_DIR)
mark_as_advanced(CLEAR CURL_LIBRARY)
if (UNIX)
    find_package(OpenSSL REQUIRED)
endif (UNIX)
find_package(CURL REQUIRED)
include_directories(${CURL_INCLUDE_DIRS})


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

# Cppcheck
if (USE_CPPCHECK)
    add_cppcheck(atmoswing-forecaster STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
endif (USE_CPPCHECK)


##### DEFINITIONS

# Specific definitions
if (USE_GUI)
    set_target_properties(atmoswing-forecaster PROPERTIES COMPILE_DEFINITIONS "APP_FORECASTER; wxUSE_GUI=1")
    set_target_properties(asprocessing PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=1")
else (USE_GUI)
    set_target_properties(atmoswing-forecaster PROPERTIES COMPILE_DEFINITIONS "APP_FORECASTER; wxUSE_GUI=0")
    set_target_properties(asprocessing PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=0")
endif (USE_GUI)


##### LINKING

target_link_libraries(atmoswing-forecaster asbase)
target_link_libraries(atmoswing-forecaster asprocessing)
if (USE_WXHGVERSION)
    target_link_libraries(atmoswing-forecaster ${WXHGVERSION_NAME})
endif (USE_WXHGVERSION)
if (USE_VLD)
    target_link_libraries(atmoswing-forecaster ${VLD_LIBRARIES})
endif (USE_VLD)


##### INSTALLER

install ( TARGETS atmoswing-forecaster DESTINATION ${INSTALL_DIR_BIN})
