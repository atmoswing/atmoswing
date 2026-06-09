cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED SOURCE_DIR OR SOURCE_DIR STREQUAL "")
	message(FATAL_ERROR "SOURCE_DIR is required")
endif()

set(targetFile "${SOURCE_DIR}/cmake/ecbuild_configure_file.cmake")
if(NOT EXISTS "${targetFile}")
	message(FATAL_ERROR "Target file does not exist: ${targetFile}")
endif()

file(READ "${targetFile}" fileContent)

set(patchedLine [=[file(GENERATE OUTPUT "${_PAR_FILENAME}" INPUT "${tmp_file}" TARGET eccodes)]=])
string(FIND "${fileContent}" "${patchedLine}" patchedLineIndex)
if(NOT patchedLineIndex EQUAL -1)
	message(STATUS "ecbuild configure_file target patch already applied, skipping: ${targetFile}")
	return()
endif()

set(originalLine [=[file(GENERATE OUTPUT "${_PAR_FILENAME}" INPUT "${tmp_file}" )]=])
string(FIND "${fileContent}" "${originalLine}" originalLineIndex)
if(originalLineIndex EQUAL -1)
	message(FATAL_ERROR "Expected ecbuild configure_file line not found in ${targetFile}")
endif()

string(REPLACE "${originalLine}" "${patchedLine}" fileContent "${fileContent}")
file(WRITE "${targetFile}" "${fileContent}")
message(STATUS "Applied ecbuild configure_file target patch: ${targetFile}")

