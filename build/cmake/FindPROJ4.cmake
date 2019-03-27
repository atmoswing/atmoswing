# - Find PROJ4
# Find the PROJ4 includes and library
#
#  PROJ4_INCLUDE_DIR - Where to find PROJ4 includes
#  PROJ4_LIBRARIES   - List of libraries when using PROJ4
#  PROJ4_FOUND       - True if PROJ4 was found

if(PROJ4_INCLUDE_DIR)
  set(PROJ4_FIND_QUIETLY TRUE)
endif(PROJ4_INCLUDE_DIR)

find_path(PROJ4_INCLUDE_DIR "proj_api.h"
        PATHS
          ${CMAKE_PREFIX_PATH}
          $ENV{EXTERNLIBS}
          $ENV{EXTERNLIBS}/proj4
          ~/Library/Frameworks
          /Library/Frameworks
          /usr/local
          /usr
          /sw # Fink
          /opt/local # DarwinPorts
          /opt/csw # Blastwave
          /opt
        PATH_SUFFIXES
          include
        DOC "PROJ4 - Headers"
        )

set(PROJ4_NAMES Proj4 proj proj_4_9 proj_6_0)
set(PROJ4_DBG_NAMES Proj4D projD proj_4_9_D proj_6_0_D)

find_library(PROJ4_LIBRARY NAMES ${PROJ4_NAMES}
        PATHS
          ${CMAKE_PREFIX_PATH}
          $ENV{EXTERNLIBS}
          $ENV{EXTERNLIBS}/proj4
          ~/Library/Frameworks
          /Library/Frameworks
          /usr/local
          /usr
          /sw
          /opt/local
          /opt/csw
          /opt
        PATH_SUFFIXES
          lib
          lib64
        DOC "PROJ4 - Library"
        )

include(FindPackageHandleStandardArgs)

#[[if(MSVC)
  # VisualStudio needs a debug version
  find_library(PROJ4_LIBRARY_DEBUG NAMES ${PROJ4_DBG_NAMES}
    PATHS
    $ENV{EXTERNLIBS}/proj4/lib
    DOC "PROJ4 - Library (Debug)"
  )
  
  if(PROJ4_LIBRARY_DEBUG AND PROJ4_LIBRARY)
    set(PROJ4_LIBRARIES optimized ${PROJ4_LIBRARY} debug ${PROJ4_LIBRARY_DEBUG})
  endif(PROJ4_LIBRARY_DEBUG AND PROJ4_LIBRARY)

  find_package_handle_standard_args(PROJ4 DEFAULT_MSG PROJ4_LIBRARY PROJ4_LIBRARY_DEBUG PROJ4_INCLUDE_DIR)

  mark_as_advanced(PROJ4_LIBRARY PROJ4_LIBRARY_DEBUG PROJ4_INCLUDE_DIR)
  
else(MSVC)]]
  # rest of the world
  set(PROJ4_LIBRARIES ${PROJ4_LIBRARY})

  find_package_handle_standard_args(PROJ4 DEFAULT_MSG PROJ4_LIBRARY PROJ4_INCLUDE_DIR)
  
  mark_as_advanced(PROJ4_LIBRARY PROJ4_INCLUDE_DIR)
  
#[[endif(MSVC)]]

if(PROJ4_FOUND)
  set(PROJ4_INCLUDE_DIRS ${PROJ4_INCLUDE_DIR})
endif(PROJ4_FOUND)
