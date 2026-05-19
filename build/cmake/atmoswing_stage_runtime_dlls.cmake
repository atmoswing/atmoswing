# On Windows, copy runtime DLLs produced by FetchContent deps (wxWidgets webp
# sub-build, eccodes, and any DLLs that land in ${CMAKE_BINARY_DIR}/bin) next
# to each app executable so local runs work without PATH tweaks.
#
# Defines the function `atmoswing_stage_runtime_dlls(<target>)`. Each app
# subdirectory must call it after its `add_executable` so the staging runs as
# a POST_BUILD step on that executable — that way it fires whenever the exe is
# built, including when the user builds a single target (not just `all`).
#
# Globbing happens at build time (via a -P script) because these DLLs are
# emitted by sub-builds and do not exist at configure time.

if (WIN32)
    set(_atmoswing_stage_script
            "${CMAKE_SOURCE_DIR}/build/cmake/atmoswing_stage_runtime_dlls_runner.cmake")

    function(atmoswing_stage_runtime_dlls targetName)
        if (NOT TARGET ${targetName})
            return()
        endif ()

        add_custom_command(TARGET ${targetName} POST_BUILD
                COMMAND ${CMAKE_COMMAND}
                -DDEST_DIR=$<TARGET_FILE_DIR:${targetName}>
                -DWX_BUILD_DIR=${wxWidgets_BINARY_DIR}
                -DECCODES_BUILD_DIR=${eccodes_BINARY_DIR}
                -DBUILD_DIR=${CMAKE_BINARY_DIR}
                -DUSE_GUI=${USE_GUI}
                -P "${_atmoswing_stage_script}"
                COMMENT "Staging runtime DLLs next to ${targetName}"
                VERBATIM)
    endfunction()
elseif (APPLE)
    # macOS: set rpath so the executable finds dylibs that vcpkg / FetchContent install
    # under the install prefix's lib/ directory. Defaults match a typical "<prefix>/bin/<exe>
    # → <prefix>/lib/<libfoo.dylib>" layout. Untested in CI — macOS isn't in the active matrix
    # but the option is enabled here so any future macOS build picks it up automatically.
    function(atmoswing_stage_runtime_dlls targetName)
        if (NOT TARGET ${targetName})
            return()
        endif ()
        set_target_properties(${targetName} PROPERTIES
                MACOSX_RPATH ON
                BUILD_WITH_INSTALL_RPATH ON
                INSTALL_RPATH "@executable_path/../lib")
    endfunction()
else ()
    function(atmoswing_stage_runtime_dlls targetName)
    endfunction()
endif ()
