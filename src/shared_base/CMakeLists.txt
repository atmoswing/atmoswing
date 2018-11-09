
# Project name
project(base)


##### SOURCE FILES

# List source files
file(GLOB_RECURSE src_base_core_h core/*.h)
file(GLOB_RECURSE src_base_core_cpp core/*.cpp)
file(GLOB_RECURSE src_base_gui_h gui/*.h)
file(GLOB_RECURSE src_base_gui_cpp gui/*.cpp)
file(GLOB_RECURSE src_base_img_h gui/img/*.h)
file(GLOB_RECURSE src_base_img_cpp gui/img/*.cpp)
file(GLOB_RECURSE src_lib_awxled_h libs/awxled/*.h)
file(GLOB_RECURSE src_lib_awxled_cpp libs/awxled/*.cpp)
list(APPEND src_shared_base ${src_base_core_h})
list(APPEND src_shared_base ${src_base_core_cpp})
if (USE_GUI)
    list(APPEND src_shared_base ${src_base_gui_h} ${src_base_img_h} ${src_lib_awxled_h})
    list(APPEND src_shared_base ${src_base_gui_cpp} ${src_base_img_cpp} ${src_lib_awxled_cpp})
endif (USE_GUI)
if (NOT BUILD_FORECASTER AND NOT BUILD_VIEWER)
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/core/asInternet.h")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/core/asInternet.cpp")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/core/asThreadInternetDownload.h")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/core/asThreadInternetDownload.cpp")
endif (NOT BUILD_FORECASTER AND NOT BUILD_VIEWER)

# Remove eventual duplicates
list(REMOVE_DUPLICATES src_shared_base)

# Include source directories
list(APPEND inc_dirs "${PROJECT_SOURCE_DIR}/core/")
if (USE_GUI)
    list(APPEND inc_dirs "${PROJECT_SOURCE_DIR}/gui/")
    list(APPEND inc_dirs "${PROJECT_SOURCE_DIR}/gui/img")
endif (USE_GUI)
include_directories(${inc_dirs})


##### LIBRARIES

include(define_common_libraries)

# libcURL
if (BUILD_FORECASTER OR BUILD_VIEWER)
    mark_as_advanced(CLEAR CURL_INCLUDE_DIR)
    mark_as_advanced(CLEAR CURL_LIBRARY)
    if (UNIX)
        find_package(OpenSSL REQUIRED)
    endif (UNIX)
    find_package(CURL REQUIRED)
    include_directories(${CURL_INCLUDE_DIRS})
endif (BUILD_FORECASTER OR BUILD_VIEWER)

# Own libraries
add_library(asbase STATIC ${src_shared_base})

# g2clib
add_subdirectory(libs/g2clib)
add_subdirectory(libs/lsversion)


##### DECLARE EXECUTABLE

# Cppcheck
if (USE_CPPCHECK)
    add_cppcheck(asbase STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
endif (USE_CPPCHECK)


##### DEFINITIONS

# Specific definitions
if (USE_GUI)
    set_target_properties(asbase PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=1")
else (USE_GUI)
    set_target_properties(asbase PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=0")
endif (USE_GUI)


##### LINKING

add_dependencies(asbase eigen)
target_link_libraries(asbase g2clib)

if (BUILD_FORECASTER OR BUILD_VIEWER)
    target_link_libraries(asbase ${CURL_LIBRARIES})
endif (BUILD_FORECASTER OR BUILD_VIEWER)

if (USE_VERSION)
    target_link_libraries(asbase lsVersion)
endif (USE_VERSION)


#### INSTALLER

# Dlls
if (WIN32)

    # Pack the Visual C++ Redistributable for Visual Studio
    include(InstallRequiredSystemLibraries)
    install(FILES ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS} DESTINATION bin)

    # Png
    find_file(PNG_DLL_NAME
            NAMES libpng14.dll libpng15.dll libpng16.dll libpng17.dll libpng18.dll
            HINTS ${CMAKE_PREFIX_PATH}
            PATH_SUFFIXES bin
            NO_DEFAULT_PATH)
    if (NOT PNG_DLL_NAME)
        message(SEND_ERROR "PNG DLL not found.")
    endif ()
    message(STATUS "PNG DLL: ${PNG_DLL_NAME}")
    add_custom_command(TARGET asbase POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PNG_DLL_NAME}"
            "${CMAKE_BINARY_DIR}")
    install(FILES ${PNG_DLL_NAME} DESTINATION bin)

    # NetCDF
    find_file(NETCDF_DLL_NAME
            netcdf.dll
            HINTS ${CMAKE_PREFIX_PATH}
            PATH_SUFFIXES bin
            NO_DEFAULT_PATH)
    if (NOT NETCDF_DLL_NAME)
        message(SEND_ERROR "NETCDF DLL not found.")
    endif ()
    message(STATUS "NETCDF DLL: ${NETCDF_DLL_NAME}")
    add_custom_command(TARGET asbase POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${NETCDF_DLL_NAME}"
            "${CMAKE_BINARY_DIR}")
    install(FILES ${NETCDF_DLL_NAME} DESTINATION bin)

    # Zlib
    find_file(ZLIB_DLL_NAME
            zlib.dll
            HINTS ${CMAKE_PREFIX_PATH}
            PATH_SUFFIXES bin
            NO_DEFAULT_PATH)
    if (NOT ZLIB_DLL_NAME)
        message(SEND_ERROR "ZLIB DLL not found.")
    endif ()
    message(STATUS "ZLIB DLL: ${ZLIB_DLL_NAME}")
    add_custom_command(TARGET asbase POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ZLIB_DLL_NAME}"
            "${CMAKE_BINARY_DIR}")
    install(FILES ${ZLIB_DLL_NAME} DESTINATION bin)

    # cURL
    find_file(CURL_DLL_NAME
            libcurl.dll
            HINTS ${CMAKE_PREFIX_PATH}
            PATH_SUFFIXES bin
            NO_DEFAULT_PATH)
    if (NOT CURL_DLL_NAME)
        message(SEND_ERROR "CURL DLL not found.")
    endif ()
    message(STATUS "CURL DLL: ${CURL_DLL_NAME}")
    add_custom_command(TARGET asbase POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CURL_DLL_NAME}"
            "${CMAKE_BINARY_DIR}")
    install(FILES ${CURL_DLL_NAME} DESTINATION bin)

    # Proj
    find_file(PROJ_DLL_NAME
            proj.dll
            HINTS ${CMAKE_PREFIX_PATH}
            PATH_SUFFIXES bin
            NO_DEFAULT_PATH)
    if (NOT PROJ_DLL_NAME)
        message(SEND_ERROR "PROJ DLL not found.")
    endif ()
    message(STATUS "PROJ DLL: ${PROJ_DLL_NAME}")
    add_custom_command(TARGET asbase POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PROJ_DLL_NAME}"
            "${CMAKE_BINARY_DIR}")
    install(FILES ${PROJ_DLL_NAME} DESTINATION bin)

endif (WIN32)