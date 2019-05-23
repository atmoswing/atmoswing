# - Find PROJ4
# Find the PROJ4 includes and library
#
#  PROJ4_INCLUDE_DIR - Where to find PROJ4 includes
#  PROJ4_LIBRARIES   - List of libraries when using PROJ4
#  PROJ4_FOUND       - True if PROJ4 was found

if (PROJ4_INCLUDE_DIR)
    set(PROJ4_FIND_QUIETLY TRUE)
endif (PROJ4_INCLUDE_DIR)

find_path(PROJ4_INCLUDE_DIR
        NAMES proj.h proj_api.h
        PATHS
          ${CMAKE_PREFIX_PATH}
          $ENV{EXTERNLIBS}
          $ENV{EXTERNLIBS}/proj4
          ~/Library/Frameworks
          /Library/Frameworks
          /usr/local
          /usr
          /opt
        PATH_SUFFIXES
          include
        DOC "PROJ4 - Headers path"
        )

find_file(PROJ4_HEADER
        NAMES proj.h proj_api.h
        PATHS
          ${PROJ4_INCLUDE_DIR}
        DOC "PROJ4 - Headers file"
        )

find_library(PROJ4_LIBRARY
        NAMES Proj4 proj proj_4_9 proj_6_0 proj_6_1
        PATHS
          ${CMAKE_PREFIX_PATH}
          $ENV{EXTERNLIBS}
          $ENV{EXTERNLIBS}/proj4
          ~/Library/Frameworks
          /Library/Frameworks
          /usr/local
          /usr
          /opt
        PATH_SUFFIXES
          lib
          lib64
        DOC "PROJ4 - Library"
        )

include(FindPackageHandleStandardArgs)

set(PROJ4_LIBRARIES ${PROJ4_LIBRARY})

find_package_handle_standard_args(PROJ4 DEFAULT_MSG PROJ4_LIBRARY PROJ4_INCLUDE_DIR)

mark_as_advanced(PROJ4_LIBRARY PROJ4_INCLUDE_DIR)

if (PROJ4_FOUND)
  set(PROJ4_INCLUDE_DIRS ${PROJ4_INCLUDE_DIR})
  set(PROJ4_HEADER_NAME PROJ4_HEADER_NAME-NOTFOUND)
  string(LENGTH ${PROJ4_INCLUDE_DIR} INCLUDE_PATH_LENGTH)
  MATH(EXPR INCLUDE_PATH_LENGTH "${INCLUDE_PATH_LENGTH}+1")
  string(FIND ${PROJ4_HEADER} ${PROJ4_INCLUDE_DIR} PROJ4_HEADER_SAME_PATH)
  if (${PROJ4_HEADER_SAME_PATH} GREATER -1)
    string(SUBSTRING ${PROJ4_HEADER} ${INCLUDE_PATH_LENGTH} -1 PROJ4_HEADER_NAME)
  else ()
    message(FATAL_ERROR "The path to Proj header file and the include directory do not match:\n
            PROJ4_HEADER=${PROJ4_HEADER} and \n
            PROJ4_INCLUDE_DIR=${PROJ4_INCLUDE_DIR}")
  endif ()
endif (PROJ4_FOUND)
