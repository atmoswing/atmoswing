#
# copyright : (c) 2016 Pascal Horton. Based on FindGrib.cmake from Maxime Lenoir,
#                      Alain Coulais, Sylwester Arabas and Orion Poplawski (2010)
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#


find_library(GRIB2C_LIBRARIES NAMES grib2c libgrib2c libg2c)
find_path(GRIB2C_INCLUDE_DIR NAMES grib2.h)

include(FindPackageHandleStandardArgs)
# since there's no grib_api.pc let's check if this installation of grib required jasper and jpeg
include(CheckLibraryExists)
set(CMAKE_REQUIRED_LIBRARIES m)
check_library_exists(${GRIB2C_LIBRARIES} grib_index_new_from_file "" GRIB2C_COMPILES)
if (GRIB2C_COMPILES)
    find_package_handle_standard_args(GRIB2C DEFAULT_MSG GRIB2C_LIBRARIES GRIB2C_INCLUDE_DIR)
else (GRIB2C_COMPILES)
    find_package(Jasper)
    if (JASPER_FOUND)
        set(CMAKE_REQUIRED_LIBRARIES ${JASPER_LIBRARIES} m)
        check_library_exists(${GRIB2C_LIBRARIES} grib_index_new_from_file "" GRIB2C_COMPILES_JASPER)
        if (GRIB2C_COMPILES_JASPER)
            set(GRIB2C_LIBRARIES ${GRIB2C_LIBRARIES} ${JASPER_LIBRARIES})
            set(GRIB2C_INCLUDE_DIR ${GRIB2C_INCLUDE_DIR} ${JASPER_INCLUDE_DIR})
            find_package_handle_standard_args(GRIB2C DEFAULT_MSG GRIB2C_LIBRARIES GRIB2C_INCLUDE_DIR)
        endif (GRIB2C_COMPILES_JASPER)
    endif (JASPER_FOUND)
endif (GRIB2C_COMPILES)
set(CMAKE_REQUIRED_LIBRARIES)

mark_as_advanced(
    GRIB2C_COMPILES
    GRIB2C_COMPILES_JASPER
)
