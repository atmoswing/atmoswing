# - Try to find OpenJPEG
# Once done, this will define
#
#  OpenJPEG_FOUND - system has OpenJPEG
#  OpenJPEG_INCLUDE_DIR - the OpenJPEG include directories
#  OpenJPEG_LIBRARY - link these to use OpenJPEG


# Include dir
find_path(OpenJPEG_INCLUDE_DIR
  NAMES openjpeg.h
    PATH_SUFFIXES
      include
      include/openjpeg-2.3
      include/openjpeg-2.4
)

# Finally the library itself
find_library(OpenJPEG_LIBRARY
  NAMES openjp2
)

