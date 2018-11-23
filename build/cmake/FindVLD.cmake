# Module for locating Visual Leak Detector.
#
# Customizable variables:
#   VLD_ROOT_DIR
#     This variable points to the Visual Leak Detector root directory. By
#     default, the module looks for the installation directory by examining the
#     Program Files/Program Files (x86) folders and the VLDROOT environment
#     variable.
#
# Read-only variables:
#   VLD_FOUND
#     Indicates that the library has been found.
#
#   VLD_INCLUDE_DIRS
#     Points to the Visual Leak Detector include directory.
#
#   VLD_LIBRARY_DIRS
#     Points to the Visual Leak Detector directory that contains the libraries.
#     The content of this variable can be passed to link_directories.
#
#   VLD_LIBRARIES
#     Points to the Visual Leak Detector libraries that can be passed to
#     target_link_libararies.
#
#
# Copyright (c) 2012 Sergiu Dotenco
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

include(FindPackageHandleStandardArgs)

set(_VLD_POSSIBLE_LIB_SUFFIXES lib)

# Version 2.0 uses vld_x86 and vld_x64 instead of simply vld as library names
if (CMAKE_SIZEOF_VOID_P EQUAL 4)
    list(APPEND _VLD_POSSIBLE_LIB_SUFFIXES lib/Win32)
elseif (CMAKE_SIZEOF_VOID_P EQUAL 8)
    list(APPEND _VLD_POSSIBLE_LIB_SUFFIXES lib/Win64)
endif (CMAKE_SIZEOF_VOID_P EQUAL 4)

find_path(VLD_ROOT_DIR
        NAMES include/vld.h
        PATHS ENV VLDROOT
        "$ENV{PROGRAMFILES}/Visual Leak Detector"
        "$ENV{PROGRAMFILES(X86)}/Visual Leak Detector"
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Visual Leak Detector;InstallLocation]"
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Wow6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Visual Leak Detector;InstallLocation]"
        DOC "VLD root directory")

find_path(VLD_INCLUDE_DIR
        NAMES vld.h
        HINTS ${VLD_ROOT_DIR}
        PATH_SUFFIXES include
        DOC "VLD include directory")

find_library(VLD_LIBRARY_DEBUG
        NAMES vld
        HINTS ${VLD_ROOT_DIR}
        PATH_SUFFIXES ${_VLD_POSSIBLE_LIB_SUFFIXES}
        DOC "VLD debug library")

if (VLD_ROOT_DIR)
    set(_VLD_VERSION_FILE ${VLD_ROOT_DIR}/CHANGES.txt)

    if (EXISTS ${_VLD_VERSION_FILE})
        set(_VLD_VERSION_REGEX
                "Visual Leak Detector \\(VLD\\) Version (([0-9]+)\\.([0-9]+)([a-z]|(.([0-9]+)))?)")
        file(STRINGS ${_VLD_VERSION_FILE} _VLD_VERSION_TMP REGEX
                ${_VLD_VERSION_REGEX})

        string(REGEX REPLACE ${_VLD_VERSION_REGEX} "\\1" _VLD_VERSION_TMP
                "${_VLD_VERSION_TMP}")

        string(REGEX REPLACE "([0-9]+).([0-9]+).*" "\\1" VLD_VERSION_MAJOR
                "${_VLD_VERSION_TMP}")
        string(REGEX REPLACE "([0-9]+).([0-9]+).*" "\\2" VLD_VERSION_MINOR
                "${_VLD_VERSION_TMP}")

        set(VLD_VERSION ${VLD_VERSION_MAJOR}.${VLD_VERSION_MINOR})

        if ("${_VLD_VERSION_TMP}" MATCHES "^([0-9]+).([0-9]+).([0-9]+)$")
            # major.minor.patch version numbering scheme
            string(REGEX REPLACE "([0-9]+).([0-9]+).([0-9]+)" "\\3"
                    VLD_VERSION_PATCH "${_VLD_VERSION_TMP}")
            set(VLD_VERSION "${VLD_VERSION}.${VLD_VERSION_PATCH}")
            set(VLD_VERSION_COUNT 3)
        else ("${_VLD_VERSION_TMP}" MATCHES "^([0-9]+).([0-9]+).([0-9]+)$")
            # major.minor version numbering scheme. The trailing letter is ignored.
            set(VLD_VERSION_COUNT 2)
        endif ("${_VLD_VERSION_TMP}" MATCHES "^([0-9]+).([0-9]+).([0-9]+)$")
    endif (EXISTS ${_VLD_VERSION_FILE})
endif (VLD_ROOT_DIR)

if (VLD_LIBRARY_DEBUG)
    set(VLD_LIBRARY debug ${VLD_LIBRARY_DEBUG} CACHE DOC "VLD library")
    get_filename_component(_VLD_LIBRARY_DIR ${VLD_LIBRARY_DEBUG} PATH)
    set(VLD_LIBRARY_DIR ${_VLD_LIBRARY_DIR} CACHE PATH "VLD library directory")
endif (VLD_LIBRARY_DEBUG)

set(VLD_INCLUDE_DIRS ${VLD_INCLUDE_DIR})
set(VLD_LIBRARY_DIRS ${VLD_LIBRARY_DIR})
set(VLD_LIBRARIES ${VLD_LIBRARY})

mark_as_advanced(VLD_INCLUDE_DIR VLD_LIBRARY_DIR VLD_LIBRARY_DEBUG VLD_LIBRARY)

find_package_handle_standard_args(VLD REQUIRED_VARS VLD_ROOT_DIR
        VLD_INCLUDE_DIR VLD_LIBRARY VERSION_VAR VLD_VERSION)
