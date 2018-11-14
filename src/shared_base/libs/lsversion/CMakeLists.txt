
# Project name
project(lsVersion)

#OPTIONS - PARAMETERS 
set(USE_VERSION 1 CACHE BOOL "Use lsVersion?")
if (USE_VERSION)
    set(lsVERSION_ENABLED 1)
    set(lsVERSION_SOFT_NAME ${CMAKE_PROJECT_NAME})

    # IF GEOS WAS SEARCHED
    if (GEOS_INCLUDE_DIR)
        set(lsVERSION_HAS_GEOS 1)
    endif (GEOS_INCLUDE_DIR)

    # IF GDAL WAS SEARCHED
    if (GDAL_INCLUDE_DIR)
        set(lsVERSION_HAS_GDAL 1)
        include_directories(${GDAL_INCLUDE_DIR})
    endif (GDAL_INCLUDE_DIR)

    # IF CURL WAS SEARCHED
    if (SEARCH_CURL)
        set(lsVERSION_HAS_CURL 1)
    endif (SEARCH_CURL)

    # IF SQLITE WAS SEARCHED
    if (SQLITE_LIBRARIES)
        set(lsVERSION_HAS_SQLITE 1)
    endif (SQLITE_LIBRARIES)

    #IF MYSQL WAS SEARCHED
    if (MYSQL_INCLUDE_DIR)
        set(lsVERSION_HAS_MYSQL 1)
    endif (MYSQL_INCLUDE_DIR)

    # IF VROOMGIS IS SET
    if (VROOMGIS_PATH)
        # GET VROOMGIS VERSION
        execute_process(COMMAND "svnversion" -n
                WORKING_DIRECTORY ${VROOMGIS_PATH}
                TIMEOUT 20
                OUTPUT_VARIABLE lsVERSION_VROOMGIS_SVN
                ERROR_VARIABLE MY_ERROR2)
        if (MY_ERROR2)
            message(FATAL_ERROR "myError2 : ${MY_ERROR2}")
        endif (MY_ERROR2)
        message(STATUS "vroomGis version ${lsVERSION_VROOMGIS_SVN}")
    endif (VROOMGIS_PATH)

    #FINDING OUR PATH
    FIND_PATH(LSVERSION_PATH "lsversion_param.h.in"
            "plugins/lsversion/src/"
            "../plugins/lsversion/src/"
            "../../plugins/lsversion/src/"
            "../../../plugins/lsversion/src/"
            "../../../../plugins/lsversion/src/"
            "lib/lsversion/src/"
            "../lib/lsversion/src/"
            "../../lib/lsversion/src/"
            "../../../lib/lsversion/src/"
            "../../../../lib/lsversion/src/"
            "lsversion/src/"
            "../lsversion/src/"
            "../../lsversion/src/"
            "../../../lsversion/src/"
            "../../../../lsversion/src/"
            NO_DEFAULT_PATH)

    # GET PROGRAM REVISON (NUMBER)
    execute_process(COMMAND "git" rev-list --count HEAD
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            TIMEOUT 20
            OUTPUT_VARIABLE lsVERSION_SOFT_VERSION
            ERROR_VARIABLE MY_ERROR1)
    if (MY_ERROR1)
        message(FATAL_ERROR "myError1 : ${MY_ERROR1}")
    endif (MY_ERROR1)
    string(STRIP ${lsVERSION_SOFT_VERSION} lsVERSION_SOFT_VERSION)

    # GET PROGRAM BRANCH
    execute_process(COMMAND "git" rev-parse --abbrev-ref HEAD
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            TIMEOUT 20
            OUTPUT_VARIABLE lsVERSION_SOFT_VERSION_BRANCH
            ERROR_VARIABLE MY_ERROR1)
    if (MY_ERROR1)
        message(FATAL_ERROR "myError1 : ${MY_ERROR1}")
    endif (MY_ERROR1)
    string(STRIP ${lsVERSION_SOFT_VERSION_BRANCH} lsVERSION_SOFT_VERSION_BRANCH)

    # GET PROGRAM REVISION (SHORT)
    execute_process(COMMAND "git" rev-parse --short HEAD
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            TIMEOUT 20
            OUTPUT_VARIABLE lsVERSION_SOFT_VERSION_REVISION
            ERROR_VARIABLE MY_ERROR1)
    if (MY_ERROR1)
        message(FATAL_ERROR "myError1 : ${MY_ERROR1}")
    endif (MY_ERROR1)
    string(STRIP ${lsVERSION_SOFT_VERSION_REVISION} lsVERSION_SOFT_VERSION_REVISION)

    message(STATUS "lsversion path: ${LSVERSION_PATH}")
    message(STATUS "${lsVERSION_SOFT_NAME} version: ${VERSION_MAJOR}.${VERSION_MINOR}")
    message(STATUS "${lsVERSION_SOFT_NAME} version: ${lsVERSION_SOFT_VERSION}")
    message(STATUS "${lsVERSION_SOFT_NAME} branch: ${lsVERSION_SOFT_VERSION_BRANCH}")
    message(STATUS "${lsVERSION_SOFT_NAME} revision: ${lsVERSION_SOFT_VERSION_REVISION}")

    #
    # CREATE LIBRARY
    #
    #search all source files
    #message(${LSVERSION_PATH})
    file(GLOB LSVERSION_SRC_FILES "${LSVERSION_PATH}/*.cpp")
    source_group(source FILES ${LSVERSION_SRC_FILES})

    #COUNTING NUMBER OF SOURCES FOUND
    list(LENGTH LSVERSION_SRC_FILES MY_VERSION_LENGTH)
    if (NOT MY_VERSION_LENGTH)
        message(FATAL_ERROR "${MY_VERSION_LENGTH} source found for lsVersion Plugin")
    endif (NOT MY_VERSION_LENGTH)

    # search all headers files
    file(GLOB LSVERSION_HEAD_FILES "${LSVERSION_PATH}/*.h")
    source_group(header FILES ${LSVERSION_HEAD_FILES})
    include_directories("${LSVERSION_PATH}")
    add_library("lsVersion" STATIC ${LSVERSION_SRC_FILES} ${LSVERSION_HEAD_FILES})


ELSE (USE_VERSION)
    message(AUTHOR_WARNING "Not using lsVersion Component")
endif (USE_VERSION)

configure_file("${LSVERSION_PATH}/lsversion_param.h.in" "${CMAKE_BINARY_DIR}/lsversion_param.h")
include_directories(${CMAKE_BINARY_DIR})


