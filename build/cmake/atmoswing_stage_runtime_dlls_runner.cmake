# Invoked at build time (POST_BUILD) to glob runtime DLLs and copy them
# next to a target executable. Globbing must happen at build time because the
# DLLs are produced by FetchContent sub-builds and do not exist at configure time.
#
# Expected -D variables:
#   DEST_DIR          — destination directory (typically $<TARGET_FILE_DIR:exe>)
#   WX_BUILD_DIR      — wxWidgets_BINARY_DIR
#   ECCODES_BUILD_DIR — eccodes_BINARY_DIR
#   BUILD_DIR         — CMAKE_BINARY_DIR
#   VCPKG_BIN_DIR     — the vcpkg bin dir for the current config, i.e.
#                       ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}[/debug]/bin
#                       (may be empty)
#   USE_GUI           — ON/OFF

set(_dlls)

if (USE_GUI)
    file(GLOB_RECURSE _wx_dlls
            "${WX_BUILD_DIR}/lib/*.dll"
            "${WX_BUILD_DIR}/libs/webp-build/*.dll")
    list(APPEND _dlls ${_wx_dlls})

    # Pick up vcpkg DLLs that applocal misses because they're reached only
    # through FetchContent libs (e.g. jasper, pulled in by eccodes' JPEG backend).
    if (VCPKG_BIN_DIR AND IS_DIRECTORY "${VCPKG_BIN_DIR}")
        file(GLOB _vcpkg_dlls "${VCPKG_BIN_DIR}/*.dll")
        list(APPEND _dlls ${_vcpkg_dlls})
    endif ()
endif ()

file(GLOB_RECURSE _eccodes_dlls
        "${ECCODES_BUILD_DIR}/bin/*.dll"
        "${BUILD_DIR}/bin/*.dll")
list(APPEND _dlls ${_eccodes_dlls})

if (NOT _dlls)
    return()
endif ()

list(REMOVE_DUPLICATES _dlls)

# copy_if_different compares content, not timestamps. That matters when the
# build type of an existing build dir is switched: the debug and release DLLs
# share their names, and the stale one is not necessarily older than its
# replacement, so a timestamp check would leave the wrong config in place.
execute_process(
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${_dlls} "${DEST_DIR}"
        RESULT_VARIABLE _copy_result)

if (NOT _copy_result EQUAL 0)
    message(FATAL_ERROR "Failed staging runtime DLLs to ${DEST_DIR}")
endif ()
