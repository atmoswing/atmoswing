#.rst:
# FindJasper
# ----------
#
# Try to find the Jasper JPEG2000 library
#
# Once done this will define
#
# ::
#
#   JASPER_FOUND - system has Jasper
#   JASPER_INCLUDE_DIR - the Jasper include directory
#   JASPER_LIBRARIES - the libraries needed to use Jasper
#   JASPER_VERSION_STRING - the version of Jasper found (since CMake 2.8.8)

#=============================================================================
# Copyright 2006-2009 Kitware, Inc.
# Copyright 2006 Alexander Neundorf <neundorf@kde.org>
# Copyright 2012 Rolf Eike Beer <eike@sf-mail.de>
# Copyright 2016 Pascal Horton <pascal.horton@giub.unibe.ch>
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)

find_path(JASPER_INCLUDE_DIR jasper/jasper.h)

find_library(JASPER_LIBRARIES NAMES jasper libjasper)

find_package(JPEG)

if (JASPER_INCLUDE_DIR AND EXISTS "${JASPER_INCLUDE_DIR}/jasper/jas_config.h")
    file(STRINGS "${JASPER_INCLUDE_DIR}/jasper/jas_config.h" jasper_version_str REGEX "^#define[\t ]+JAS_VERSION[\t ]+\".*\".*")

    string(REGEX REPLACE "^#define[\t ]+JAS_VERSION[\t ]+\"([^\"]+)\".*" "\\1" JASPER_VERSION_STRING "${jasper_version_str}")
endif ()

# handle the QUIETLY and REQUIRED arguments and set JASPER_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Jasper
                                  REQUIRED_VARS JASPER_LIBRARIES JASPER_INCLUDE_DIR JPEG_LIBRARIES
                                  VERSION_VAR JASPER_VERSION_STRING)

if (JASPER_FOUND)
   set(JASPER_LIBRARIES ${JASPER_LIBRARIES} ${JPEG_LIBRARIES} )
endif ()

