
# Eigen
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
endif()
include_directories(${eigen_SOURCE_DIR})
set(USE_EIGEN TRUE)

# Intel MKL
if (USE_MKL)
    find_package(MKL REQUIRED)
    include_directories(${MKL_INCLUDE_DIRS})
    link_libraries(${MKL_LIBRARIES})
    add_definitions(-DEIGEN_USE_MKL_ALL)
endif (USE_MKL)

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
link_libraries(${JASPER_LIBRARY_RELEASE})

# Jpeg
include_directories(${JPEG_INCLUDE_DIR})
link_libraries(${JPEG_LIBRARY})

# OpenJpeg
find_package(OpenJPEG REQUIRED)
include_directories(${OpenJPEG_INCLUDE_DIR})
link_libraries(${OpenJPEG_LIBRARY})

# Proj
find_package(PROJ 4.9 REQUIRED)
include_directories(${PROJ_INCLUDE_DIR})
link_libraries(${PROJ_LIBRARIES})

# NetCDF (has to be before GDAL)
mark_as_advanced(CLEAR NETCDF_INCLUDE_DIR)
mark_as_advanced(CLEAR NETCDF_LIBRARY)
find_package(NetCDF 4 MODULE REQUIRED)
include_directories(${NETCDF_INCLUDE_DIRS})
link_libraries(${NETCDF_LIBRARIES})

if (BUILD_VIEWER)

    # GDAL
    if (GDAL_PATH)
        set(ENV{GDAL_ROOT} ${GDAL_PATH})
    endif ()
    find_package(GDAL 2 REQUIRED)
    include_directories(${GDAL_INCLUDE_DIRS})
    link_libraries(${GDAL_LIBRARIES})

    # SQLite 3
    find_package(SQLite3 REQUIRED)
    include_directories(${SQLITE3_INCLUDE_DIR})
    link_libraries(${SQLITE3_LIBRARY})

else ()

    unset(GDAL_INCLUDE_DIR CACHE)
    unset(GDAL_LIBRARIES CACHE)

endif ()

# ecCodes
find_package(eccodes MODULE REQUIRED)
include_directories(${ECCODES_INCLUDE_DIR})
include_directories(${ECCODES_INCLUDE_DIRS})
link_libraries(${ECCODES_LIBRARIES})

# wxVersion
if (USE_GUI)
    FetchContent_MakeAvailable(wxVersion)
    include_directories(${wxVersion_SOURCE_DIR}/src)
    include_directories(${wxVersion_BINARY_DIR})
else()
    set(USE_WXVERSION 0)
endif()