# Invoked at build time (POST_BUILD) to glob runtime DLLs and copy them
# next to a target executable. Globbing must happen at build time because the
# DLLs are produced by FetchContent sub-builds and do not exist at configure time.
#
# Expected -D variables:
#   DEST_DIR          — destination directory (typically $<TARGET_FILE_DIR:exe>)
#   WX_BUILD_DIR      — wxWidgets_BINARY_DIR
#   ECCODES_BUILD_DIR — eccodes_BINARY_DIR
#   BUILD_DIR         — CMAKE_BINARY_DIR
#   VCPKG_BIN_DIR     — ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin (may be empty)
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

foreach (_dll IN LISTS _dlls)
    get_filename_component(_dll_name "${_dll}" NAME)
    if (NOT EXISTS "${DEST_DIR}/${_dll_name}"
            OR NOT "${DEST_DIR}/${_dll_name}" IS_NEWER_THAN "${_dll}")
        message(STATUS "Staging ${_dll_name} -> ${DEST_DIR}")
        file(COPY "${_dll}" DESTINATION "${DEST_DIR}")
    endif ()
endforeach ()
