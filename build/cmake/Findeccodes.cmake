find_path(ECCODES_INCLUDE_DIR
        NAMES
            eccodes.h
        HINTS
            ${ECCODES_DIR}
            ENV ECCODES_DIR
        PATH_SUFFIXES
            include
        PATHS
            ${COMMON_INSTALL_DIRS}
        )

find_library(ECCODES_LIBRARIES
        NAMES
            eccodes ECCODES
        HINTS
            ${ECCODES_DIR}
            ENV ECCODES_DIR
        PATH_SUFFIXES
            lib64
            lib
        PATHS
            ${COMMON_INSTALL_DIRS}
        )

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(ECCODES REQUIRED_VARS ECCODES_LIBRARIES ECCODES_INCLUDE_DIR)
