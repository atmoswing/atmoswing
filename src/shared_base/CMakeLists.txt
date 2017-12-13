
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
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/core/asPredictorOper.h")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/core/asPredictorOper.cpp")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/core/asPredictorOperGfsForecast.h")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/core/asPredictorOperGfsForecast.cpp")
endif (NOT BUILD_FORECASTER AND NOT BUILD_VIEWER)

# Grib2c library
file(GLOB_RECURSE src_lib_g2clib_h libs/g2clib/*.h)
set(src_lib_g2clib_c
        libs/g2clib/gridtemplates.c
        libs/g2clib/drstemplates.c
        libs/g2clib/pdstemplates.c
        libs/g2clib/gbits.c
        libs/g2clib/g2_unpack1.c
        libs/g2clib/g2_unpack2.c
        libs/g2clib/g2_unpack3.c
        libs/g2clib/g2_unpack4.c
        libs/g2clib/g2_unpack5.c
        libs/g2clib/g2_unpack6.c
        libs/g2clib/g2_unpack7.c
        libs/g2clib/g2_free.c
        libs/g2clib/g2_info.c
        libs/g2clib/g2_getfld.c
        libs/g2clib/simunpack.c
        libs/g2clib/comunpack.c
        libs/g2clib/pack_gp.c
        libs/g2clib/reduce.c
        libs/g2clib/specpack.c
        libs/g2clib/specunpack.c
        libs/g2clib/rdieee.c
        libs/g2clib/mkieee.c
        libs/g2clib/int_power.c
        libs/g2clib/simpack.c
        libs/g2clib/compack.c
        libs/g2clib/cmplxpack.c
        libs/g2clib/misspack.c
        libs/g2clib/jpcpack.c
        libs/g2clib/jpcunpack.c
        libs/g2clib/pngpack.c
        libs/g2clib/pngunpack.c
        libs/g2clib/dec_jpeg2000.c
        libs/g2clib/enc_jpeg2000.c
        libs/g2clib/dec_png.c
        libs/g2clib/enc_png.c
        libs/g2clib/g2_create.c
        libs/g2clib/g2_addlocal.c
        libs/g2clib/g2_addgrid.c
        libs/g2clib/g2_addfield.c
        libs/g2clib/g2_gribend.c
        libs/g2clib/getdim.c
        libs/g2clib/g2_miss.c
        libs/g2clib/getpoly.c
        libs/g2clib/seekgb.c
        )
list(APPEND src_lib_g2clib ${src_lib_g2clib_h} ${src_lib_g2clib_c})

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

# Grib2c
add_library(g2clib STATIC ${src_lib_g2clib})

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

set_target_properties(g2clib PROPERTIES COMPILE_DEFINITIONS "USE_JPEG2000; USE_PNG")


##### LINKING

add_dependencies(asbase eigen)
target_link_libraries(asbase g2clib)

if (BUILD_FORECASTER OR BUILD_VIEWER)
    target_link_libraries(asbase ${CURL_LIBRARIES})
endif (BUILD_FORECASTER OR BUILD_VIEWER)


#### INSTALLER

# Dlls
if (WIN32)
    find_file(PNG_DLL_NAME
            libpng14.dll libpng15.dll libpng16.dll libpng17.dll libpng18.dll
            HINTS ${DLL_DIRS}
            NO_DEFAULT_PATH)
    if (NOT PNG_DLL_NAME)
        message(SEND_ERROR "PNG DLL not found in ${DLL_DIRS}.")
    endif ()
    add_custom_command(TARGET asbase POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PNG_DLL_NAME}"
            "${CMAKE_BINARY_DIR}")
    message (STATUS "PNG DLL: ${PNG_DLL_NAME}")

    find_file(NETCDF_DLL_NAME
            netcdf.dll
            HINTS ${DLL_DIRS}
            NO_DEFAULT_PATH)
    if (NOT NETCDF_DLL_NAME)
        message(SEND_ERROR "NETCDF DLL not found in ${DLL_DIRS}.")
    endif ()
    add_custom_command(TARGET asbase POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${NETCDF_DLL_NAME}"
            "${CMAKE_BINARY_DIR}")
    message (STATUS "NETCDF DLL: ${NETCDF_DLL_NAME}")

    find_file(ZLIB_DLL_NAME
            zlib.dll
            HINTS ${DLL_DIRS}
            NO_DEFAULT_PATH)
    if (NOT ZLIB_DLL_NAME)
        message(SEND_ERROR "ZLIB DLL not found in ${DLL_DIRS}.")
    endif ()
    add_custom_command(TARGET asbase POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ZLIB_DLL_NAME}"
            "${CMAKE_BINARY_DIR}")
    message (STATUS "ZLIB DLL: ${ZLIB_DLL_NAME}")

endif (WIN32)