# - Find curl
# Find the native CURL headers and libraries.
#
#  CURL_INCLUDE_DIR - where to find curl/curl.h, etc.
#  CURL_LIBRARY    - List of libraries when using curl.
#  CURL_FOUND        - True if curl found.

# Look for the header file.
FIND_PATH(CURL_INCLUDE_DIR 
	NAMES curl/curl.h
	DOC "Path to CURL include directory")

# Look for the library.
FIND_LIBRARY(CURL_LIBRARY 
	NAMES curl
	DOC "Path to CURL library file")

# handle the QUIETLY and REQUIRED arguments and set CURL_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CURL DEFAULT_MSG CURL_LIBRARY CURL_INCLUDE_DIR)

IF ("${CURL_INCLUDE_DIR}" MATCHES "NOTFOUND")
    SET (CURL_LIBRARY)
    SET (CURL_INCLUDE_DIR)
ELSEIF ("${CURL_LIBRARY}" MATCHES "NOTFOUND")
    SET (CURL_LIBRARY)
    SET (CURL_INCLUDE_DIR)
ELSE ("${CURL_INCLUDE_DIR}" MATCHES "NOTFOUND")
    SET (CURL_FOUND 1)
    SET (CURL_LIBRARIES ${CURL_LIBRARY})
    SET (CURL_INCLUDE_DIRS ${CURL_INCLUDE_DIR})
ENDIF ("${CURL_INCLUDE_DIR}" MATCHES "NOTFOUND")