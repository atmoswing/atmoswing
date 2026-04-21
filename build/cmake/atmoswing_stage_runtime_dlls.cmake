# On Windows, copy runtime DLLs produced by FetchContent deps next to app executables.
# This keeps local runs working even when wxWidgets/eccodes are built as shared libs.
if (WIN32)
    set(_atmoswing_runtime_dlls)

    if (USE_GUI)
        file(GLOB_RECURSE _wx_runtime_dlls CONFIGURE_DEPENDS
                "${wxWidgets_BINARY_DIR}/lib/*.dll")
        list(APPEND _atmoswing_runtime_dlls ${_wx_runtime_dlls})
    endif ()

    file(GLOB_RECURSE _eccodes_runtime_dlls CONFIGURE_DEPENDS
            "${eccodes_BINARY_DIR}/bin/eccodes*.dll"
            "${CMAKE_BINARY_DIR}/bin/eccodes*.dll")
    list(APPEND _atmoswing_runtime_dlls ${_eccodes_runtime_dlls})

    list(REMOVE_DUPLICATES _atmoswing_runtime_dlls)

    if (_atmoswing_runtime_dlls)
        message(STATUS "Staging wxWidgets/eccodes DLLs: ${_atmoswing_runtime_dlls}")
    endif ()

    add_custom_target(atmoswing-stage-runtime-dlls ALL)

    function(atmoswing_stage_runtime_dlls targetName)
        if (NOT TARGET ${targetName} OR NOT _atmoswing_runtime_dlls)
            return()
        endif ()

        set(_stage_stamp "${CMAKE_BINARY_DIR}/CMakeFiles/${targetName}-runtime-dlls.stamp")
        add_custom_command(
                OUTPUT "${_stage_stamp}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_atmoswing_runtime_dlls}
                "$<TARGET_FILE_DIR:${targetName}>"
                COMMAND ${CMAKE_COMMAND} -E touch "${_stage_stamp}"
                DEPENDS ${targetName} ${_atmoswing_runtime_dlls}
                VERBATIM)

        add_custom_target(${targetName}-stage-runtime-dlls DEPENDS "${_stage_stamp}")
        add_dependencies(atmoswing-stage-runtime-dlls ${targetName}-stage-runtime-dlls)
    endfunction()

    if (BUILD_FORECASTER)
        atmoswing_stage_runtime_dlls(atmoswing-forecaster)
    endif ()
    if (BUILD_OPTIMIZER)
        atmoswing_stage_runtime_dlls(atmoswing-optimizer)
    endif ()
    if (BUILD_DOWNSCALER)
        atmoswing_stage_runtime_dlls(atmoswing-downscaler)
    endif ()
    if (BUILD_VIEWER)
        atmoswing_stage_runtime_dlls(atmoswing-viewer)
    endif ()
endif ()

