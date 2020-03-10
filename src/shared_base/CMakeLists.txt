
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

if (BUILD_FORECASTER OR BUILD_VIEWER)
    target_link_libraries(asbase ${CURL_LIBRARIES})
endif (BUILD_FORECASTER OR BUILD_VIEWER)

target_link_libraries(asbase wxVersion)


#### INSTALLER

# Dlls
if (WIN32)

    # Pack the Visual C++ Redistributable for Visual Studio
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION ${INSTALL_DIR_BIN})
    include(InstallRequiredSystemLibraries)
    install(FILES ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS} DESTINATION ${INSTALL_DIR_BIN})

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
    install(FILES ${PNG_DLL_NAME} DESTINATION ${INSTALL_DIR_BIN})

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
    install(FILES ${NETCDF_DLL_NAME} DESTINATION ${INSTALL_DIR_BIN})

    # OpenJPEG
    find_file(OpenJPEG_DLL_NAME
            openjp2.dll
            HINTS ${CMAKE_PREFIX_PATH}
            PATH_SUFFIXES bin
            NO_DEFAULT_PATH)
    if (NOT NETCDF_DLL_NAME)
        message(SEND_ERROR "OpenJPEG DLL not found.")
    endif ()
    message(STATUS "OpenJPEG DLL: ${OpenJPEG_DLL_NAME}")
    add_custom_command(TARGET asbase POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OpenJPEG_DLL_NAME}"
            "${CMAKE_BINARY_DIR}")
    install(FILES ${OpenJPEG_DLL_NAME} DESTINATION ${INSTALL_DIR_BIN})

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
    install(FILES ${ZLIB_DLL_NAME} DESTINATION ${INSTALL_DIR_BIN})

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
    install(FILES ${CURL_DLL_NAME} DESTINATION ${INSTALL_DIR_BIN})

    # Proj
    find_file(PROJ_DLL_NAME
            proj.dll NAMES proj_6_0.dll proj_6_1.dll proj_6_2.dll proj_6_3.dll
            HINTS ${CMAKE_PREFIX_PATH}
            PATH_SUFFIXES bin
            NO_DEFAULT_PATH)
    if (NOT PROJ_DLL_NAME)
        message(SEND_ERROR "PROJ DLL not found.")
    endif ()
    message(STATUS "PROJ DLL: ${PROJ_DLL_NAME}")
    add_custom_command(TARGET asbase POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            "${PROJ_DLL_NAME}"
            "${CMAKE_BINARY_DIR}")
    install(FILES ${PROJ_DLL_NAME} DESTINATION ${INSTALL_DIR_BIN})

    # Proj share
    find_path(PROJ_SHARE_PATH
            NAMES epsg ntv1_can.dat esri.extra proj.db
            HINTS ${CMAKE_PREFIX_PATH}
            PATH_SUFFIXES share share/proj
            NO_DEFAULT_PATH)
    if (NOT PROJ_SHARE_PATH)
        message(SEND_ERROR "PROJ share not found.")
    endif ()
    message(STATUS "PROJ share: ${PROJ_SHARE_PATH}")
    add_custom_command(TARGET asbase POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${PROJ_SHARE_PATH}"
            "${CMAKE_BINARY_DIR}/share/proj")
    install(DIRECTORY ${PROJ_SHARE_PATH} DESTINATION share)

    # SQLite
    find_file(SQLITE_DLL_NAME
            sqlite.dll NAMES sqlite3.dll
            HINTS ${CMAKE_PREFIX_PATH}
            PATH_SUFFIXES bin
            NO_DEFAULT_PATH)
    if (NOT SQLITE_DLL_NAME)
        message(SEND_ERROR "SQLITE DLL not found.")
    endif ()
    message(STATUS "SQLITE DLL: ${SQLITE_DLL_NAME}")
    add_custom_command(TARGET asbase POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            "${SQLITE_DLL_NAME}"
            "${CMAKE_BINARY_DIR}")
    install(FILES ${SQLITE_DLL_NAME} DESTINATION ${INSTALL_DIR_BIN})

    # ecCodes share
    find_path(ECCODES_SHARE_PATH
            NAMES definitions
            HINTS ${CMAKE_PREFIX_PATH}
            PATH_SUFFIXES share share/eccodes
            NO_DEFAULT_PATH)
    if (NOT ECCODES_SHARE_PATH)
        message(SEND_ERROR "ecCodes share not found.")
    endif ()
    message(STATUS "ecCodes share: ${ECCODES_SHARE_PATH}")
    add_custom_command(TARGET asbase POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${ECCODES_SHARE_PATH}"
            "${CMAKE_BINARY_DIR}/share/eccodes")
    install(DIRECTORY ${ECCODES_SHARE_PATH} DESTINATION share)

endif (WIN32)
