
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

# g2clib
include_directories("${CMAKE_SOURCE_DIR}/src/shared_base/libs/g2clib")

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