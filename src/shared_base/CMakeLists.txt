
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
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/src/shared_base/core/asInternet.h")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/src/shared_base/core/asInternet.cpp")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/src/shared_base/core/asThreadInternetDownload.h")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/src/shared_base/core/asThreadInternetDownload.cpp")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/src/shared_base/core/asPredictorOper.h")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/src/shared_base/core/asPredictorOper.cpp")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/src/shared_base/core/asPredictorOperGfsForecast.h")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/src/shared_base/core/asPredictorOperGfsForecast.cpp")
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
list(APPEND inc_dirs
        "${PROJECT_SOURCE_DIR}/src/shared_base/core/"
        )
include_directories(${inc_dirs})


##### EXTERNAL PROJECTS

# Eigen
ExternalProject_Add(eigen
        URL http://bitbucket.org/eigen/eigen/get/${EIGEN_VERSION}.tar.gz
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND
        ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/eigen-prefix/src/eigen/Eigen
        ${EXTERNAL_DIR}/include/Eigen
        )


##### LIBRARIES

# WxWidgets (adv lib nedded for the caldendar widget)
mark_as_advanced(wxWidgets_wxrc_EXECUTABLE)
mark_as_advanced(wxWidgets_with_GUI)
if (USE_MSYS2)
    set(wxWidgets_CONFIG_OPTIONS --prefix=${MINGW_PATH})
endif (USE_MSYS2)
if (USE_GUI)
    set(wxWidgets_with_GUI TRUE)
    find_package(wxWidgets REQUIRED core base adv xml net)
else (USE_GUI)
    set(wxWidgets_with_GUI FALSE)
    find_package(wxWidgets REQUIRED base xml net)
endif (USE_GUI)
include("${wxWidgets_USE_FILE}")
include_directories(${wxWidgets_INCLUDE_DIRS})
link_libraries(${wxWidgets_LIBRARIES})

# PNG
set(PNG_FIND_QUIETLY OFF)
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIRS})
link_libraries(${PNG_LIBRARIES})

# NetCDF (has to be before GDAL)
mark_as_advanced(CLEAR NETCDF_INCLUDE_DIR)
mark_as_advanced(CLEAR NETCDF_LIBRARY)
find_package(NetCDF 4 MODULE REQUIRED)
include_directories(${NETCDF_INCLUDE_DIRS})
link_libraries(${NETCDF_LIBRARIES})

# Jasper
find_package(Jasper REQUIRED)
include_directories(${JASPER_INCLUDE_DIR})
link_libraries(${JASPER_LIBRARIES})

# Jpeg
include_directories(${JPEG_INCLUDE_DIR})
link_libraries(${JPEG_LIBRARY})

# Grib2c
add_library(g2clib STATIC ${src_lib_g2clib})
include_directories("src/shared_base/libs/g2clib")

# wxhgversion
if (USE_GUI)
    set(USE_WXHGVERSION 0)

    #    set(USE_WXHGVERSION 1)
    #    ExternalProject_Add(wxhgversion
    #            URL "https://bitbucket.org/terranum/wxhgversion/get/tip.tar.gz"
    #            PATCH_COMMAND cp build/use_wxhgversion.cmake CMakeLists.txt
    #            CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_DIR}
    #            )
    #    include_directories(${EXTERNAL_DIR}/include)
    #    link_directories(${EXTERNAL_DIR}/lib)
else (USE_GUI)
    set(USE_WXHGVERSION 0)
endif (USE_GUI)

# Own libraries
add_library(asbase STATIC ${src_shared_base})


##### DECLARE EXECUTABLE

# Cppcheck
if (USE_CPPCHECK)
    add_cppcheck(asbase STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
endif (USE_CPPCHECK)


##### DEFINITIONS

include(define_shared_compilation_flags)

# Specific definitions
if (USE_GUI)
    set_target_properties(asbase PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=1")
else (USE_GUI)
    set_target_properties(asbase PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=0")
endif (USE_GUI)


##### LINKING

add_dependencies(asbase eigen)
target_link_libraries(asbase g2clib)