# Find UnitTest++
#
#

SET (UNITTEST++_FOUND FALSE)
SET (UNITTEST++_LIBRARIES    "")

FIND_PATH (UNITTEST++_INCLUDE_DIR UnitTest++.h
	/usr/include/unittest++ 
	/usr/include/UnitTest++ 
	/usr/local/include/unittest++ 
	/usr/local/include/UnitTest++ 
	/opt/local/include/unittest++
	/opt/local/include/UnitTest++
	$ENV{UNITTESTXX_PATH}/src 
	$ENV{UNITTESTXX_INCLUDE_PATH}
	)

FIND_LIBRARY (UNITTEST++_LIBRARY_RELEASE NAMES UnitTest++ PATHS 
	/usr/lib 
	/usr/local/lib 
	/opt/local/lib 
	${UNITTEST++_INCLUDE_DIR}/../lib/Release
	)
mark_as_advanced(UNITTEST++_LIBRARY_RELEASE)

FIND_LIBRARY (UNITTEST++_LIBRARY_DEBUG NAMES UnitTest++ PATHS 
	/usr/lib 
	/usr/local/lib 
	/opt/local/lib 
	${UNITTEST++_INCLUDE_DIR}/../lib/Debug
	)
mark_as_advanced(UNITTEST++_LIBRARY_DEBUG)

IF (UNITTEST++_INCLUDE_DIR AND UNITTEST++_LIBRARY_RELEASE AND UNITTEST++_LIBRARY_DEBUG)
	list(APPEND UNITTEST++_LIBRARIES
		debug ${UNITTEST++_LIBRARY_DEBUG} optimized ${UNITTEST++_LIBRARY_RELEASE}
		)
	SET (UNITTEST++_FOUND TRUE)
ENDIF (UNITTEST++_INCLUDE_DIR AND UNITTEST++_LIBRARY_RELEASE AND UNITTEST++_LIBRARY_DEBUG)

IF (UNITTEST++_FOUND)
   IF (NOT UnitTest++_FIND_QUIETLY)
      MESSAGE(STATUS "Found UnitTest++ Release : ${UNITTEST++_LIBRARY_RELEASE}")
      MESSAGE(STATUS "Found UnitTest++ Debug: ${UNITTEST++_LIBRARY_DEBUG}")
   ENDIF (NOT UnitTest++_FIND_QUIETLY)
ELSE (UNITTEST++_FOUND)
   IF (UnitTest++_FIND_REQUIRED)
      MESSAGE(FATAL_ERROR "Could not find UnitTest++")
   ENDIF (UnitTest++_FIND_REQUIRED)
ENDIF (UNITTEST++_FOUND)

