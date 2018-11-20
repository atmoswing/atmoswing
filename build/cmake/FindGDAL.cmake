# FindGDAL
# --------
#
# Locate gdal
#
# This module accepts the following environment variables:
#
# ::
#
#     GDAL_DIR or GDAL_ROOT - Specify the location of GDAL
#
#
#
# This module defines the following CMake variables:
#
# ::
#
#     GDAL_FOUND - True if libgdal is found
#     GDAL_LIBRARY - A variable pointing to the GDAL library
#     GDAL_INCLUDE_DIR - Where to find the headers

if (NOT GDAL_DIR AND GDAL_ROOT)
    SET(GDAL_DIR ${GDAL_ROOT})
elseif (NOT GDAL_DIR AND ENV{GDAL_ROOT})
    SET(GDAL_DIR ENV{GDAL_ROOT})
endif ()

find_path(GDAL_INCLUDE_DIR gdal.h
        HINTS
        ${GDAL_DIR}
        ENV GDAL_DIR
        ENV GDAL_ROOT
        PATH_SUFFIXES
        include/gdal
        include/GDAL
        include
        NO_DEFAULT_PATH
        )

if (WIN32)

    find_library(GDAL_LIBRARY
            gdal_i
            HINTS ${GDAL_DIR}/lib
            ${GDAL_DIR}
            NO_DEFAULT_PATH)

elseif (APPLE)

    # If mac, use the dynamic library
    if (GDAL_DIR)
        find_library(GDAL_LIBRARY
                gdal NAMES gdal1 gdal1.6.0 gdal1.7.0 gdal1.8.0 gdal1.9.0
                PATHS
                ${GDAL_DIR}/lib
                ${GDAL_DIR}
                NO_DEFAULT_PATH)

    else ()
        message(STATUS "Searching GDAL on standard PATHS")
        find_path(GDAL_INCLUDE_DIR gdal.h
                PATH_SUFFIXES gdal)

        find_library(GDAL_LIBRARY
                gdal NAMES gdal1 gdal1.6.0 gdal1.7.0 gdal1.8.0 gdal1.9.0)
    endif ()

else ()

    # If linux, use the static library
    if (GDAL_DIR)
        find_library(GDAL_LIBRARY
                NAMES libgdal.a gdal1 gdal1.6.0 gdal1.7.0 gdal1.8.0 gdal1.9.0
                PATHS
                ${GDAL_DIR}/lib
                ${GDAL_DIR}
                NO_DEFAULT_PATH)

        find_program(GDAL_CONFIG gdal-config
                ${GDAL_DIR}/bin/
                NO_DEFAULT_PATH)

    else ()
        message(STATUS "Searching GDAL on standard PATHS")
        find_path(GDAL_INCLUDE_DIR gdal.h
                PATH_SUFFIXES gdal)

        find_library(GDAL_LIBRARY
                NAMES libgdal.a gdal1 gdal1.6.0 gdal1.7.0 gdal1.8.0 gdal1.9.0)

        find_program(GDAL_CONFIG gdal-config)

    endif ()

    if (GDAL_CONFIG)
        exec_program(${GDAL_CONFIG} ARGS --dep-libs OUTPUT_VARIABLE GDAL_DEP_LIBS)
        list(APPEND GDAL_LIBRARY ${GDAL_DEP_LIBS})
    endif (GDAL_CONFIG)

endif (WIN32)

mark_as_advanced(
        GDAL_INCLUDE_DIR
        GDAL_LIBRARY
)

set(GDAL_LIBRARIES ${GDAL_LIBRARY})
set(GDAL_INCLUDE_DIRS ${GDAL_INCLUDE_DIR})