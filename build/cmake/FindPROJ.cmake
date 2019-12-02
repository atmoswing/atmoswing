# - Find PROJ
# Find the PROJ includes and library
#
#  PROJ_INCLUDE_DIR - Where to find PROJ includes
#  PROJ_LIBRARIES   - List of libraries when using PROJ
#  PROJ_FOUND       - True if PROJ was found

if (PROJ_INCLUDE_DIR)
    set(PROJ_FIND_QUIETLY TRUE)
endif (PROJ_INCLUDE_DIR)

find_path(PROJ_INCLUDE_DIR
        NAMES proj.h proj_api.h
        PATHS
          ${CMAKE_PREFIX_PATH}
          $ENV{EXTERNLIBS}
          $ENV{EXTERNLIBS}/PROJ
          ~/Library/Frameworks
          /Library/Frameworks
          /usr/local
          /usr
          /opt
        PATH_SUFFIXES
          include
        DOC "PROJ - Headers path"
        )

find_file(PROJ_HEADER
        NAMES proj.h proj_api.h
        PATHS
          ${PROJ_INCLUDE_DIR}
        DOC "PROJ - Headers file"
        )

find_library(PROJ_LIBRARY
        NAMES Proj4 proj proj_4_9 proj_6_0 proj_6_1 proj_6_2 proj_6_3 proj_6_4
        PATHS
          ${CMAKE_PREFIX_PATH}
          $ENV{EXTERNLIBS}
          ~/Library/Frameworks
          /Library/Frameworks
          /usr/local
          /usr
          /opt
        PATH_SUFFIXES
          lib
          lib64
        DOC "PROJ - Library"
        )

include(FindPackageHandleStandardArgs)

set(PROJ_LIBRARIES ${PROJ_LIBRARY})

find_package_handle_standard_args(PROJ DEFAULT_MSG PROJ_LIBRARY PROJ_INCLUDE_DIR)

mark_as_advanced(PROJ_LIBRARY PROJ_INCLUDE_DIR)

if (PROJ_FOUND)
  set(PROJ_INCLUDE_DIRS ${PROJ_INCLUDE_DIR})
  set(PROJ_HEADER_NAME PROJ_HEADER_NAME-NOTFOUND)
  string(LENGTH ${PROJ_INCLUDE_DIR} INCLUDE_PATH_LENGTH)
  MATH(EXPR INCLUDE_PATH_LENGTH "${INCLUDE_PATH_LENGTH}+1")
  string(FIND ${PROJ_HEADER} ${PROJ_INCLUDE_DIR} PROJ_HEADER_SAME_PATH)
  if (${PROJ_HEADER_SAME_PATH} GREATER -1)
    string(SUBSTRING ${PROJ_HEADER} ${INCLUDE_PATH_LENGTH} -1 PROJ_HEADER_NAME)
  else ()
    message(FATAL_ERROR "The path to Proj header file and the include directory do not match:\n
            PROJ_HEADER=${PROJ_HEADER} and \n
            PROJ_INCLUDE_DIR=${PROJ_INCLUDE_DIR}")
  endif ()
endif (PROJ_FOUND)
