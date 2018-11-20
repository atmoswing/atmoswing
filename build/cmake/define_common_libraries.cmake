
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

# Jasper
find_package(Jasper REQUIRED)
include_directories(${JASPER_INCLUDE_DIR})
link_libraries(${JASPER_LIBRARIES})

# Jpeg
include_directories(${JPEG_INCLUDE_DIR})
link_libraries(${JPEG_LIBRARY})

# NetCDF (has to be before GDAL)
mark_as_advanced(CLEAR NETCDF_INCLUDE_DIR)
mark_as_advanced(CLEAR NETCDF_LIBRARY)
find_package(NetCDF 4 MODULE REQUIRED)
include_directories(${NETCDF_INCLUDE_DIRS})
link_libraries(${NETCDF_LIBRARIES})

if (BUILD_VIEWER)

    # GDAL
    if (GDAL_ROOT)
        set(ENV{GDAL_ROOT} ${GDAL_ROOT})
    endif ()
    find_package(GDAL 2 REQUIRED)
    include_directories(${GDAL_INCLUDE_DIRS})
    link_libraries(${GDAL_LIBRARIES})

endif ()

# g2clib
include_directories("${CMAKE_SOURCE_DIR}/src/shared_base/libs/g2clib/src")

# lsversion
include_directories("${CMAKE_SOURCE_DIR}/src/shared_base/libs/lsversion/src")
include_directories("${CMAKE_BINARY_DIR}")

# lsversion
if (USE_GUI)
    set(USE_VERSION 1)
else (USE_GUI)
    set(USE_VERSION 0)
endif (USE_GUI)