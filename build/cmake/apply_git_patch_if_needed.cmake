cmake_minimum_required(VERSION 3.18)

if (NOT DEFINED SOURCE_DIR OR SOURCE_DIR STREQUAL "")
    message(FATAL_ERROR "SOURCE_DIR is required")
endif ()
if (NOT DEFINED PATCH_FILE OR PATCH_FILE STREQUAL "")
    message(FATAL_ERROR "PATCH_FILE is required")
endif ()

if (NOT EXISTS "${PATCH_FILE}")
    message(FATAL_ERROR "Patch file does not exist: ${PATCH_FILE}")
endif ()

# Keep patching idempotent: apply if possible, otherwise treat 'already applied' as success.
execute_process(
        COMMAND git apply --check "${PATCH_FILE}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE _apply_check_result
)

if (_apply_check_result EQUAL 0)
    execute_process(
            COMMAND git apply "${PATCH_FILE}"
            WORKING_DIRECTORY "${SOURCE_DIR}"
            RESULT_VARIABLE _apply_result
    )
    if (NOT _apply_result EQUAL 0)
        message(FATAL_ERROR "Failed to apply patch: ${PATCH_FILE}")
    endif ()
    message(STATUS "Applied patch: ${PATCH_FILE}")
    return()
endif ()

execute_process(
        COMMAND git apply --reverse --check "${PATCH_FILE}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE _reverse_check_result
)

if (_reverse_check_result EQUAL 0)
    message(STATUS "Patch already applied, skipping: ${PATCH_FILE}")
    return()
endif ()

# The tree is in an inconsistent state: the patch neither applies nor reverse-applies.
# This happens when a CI cache restores a source tree patched with a *different* version
# of this patch (the prefix restore-keys fallback can bring back a stale tree). Reset the
# tracked files to the pristine upstream checkout and apply once more before giving up.
message(STATUS "Patch does not apply cleanly; resetting source tree and retrying: ${PATCH_FILE}")
execute_process(
        COMMAND git reset --hard
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE _reset_result
)
if (NOT _reset_result EQUAL 0)
    message(FATAL_ERROR "Failed to reset source tree before re-applying patch: ${PATCH_FILE}")
endif ()

execute_process(
        COMMAND git apply "${PATCH_FILE}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE _retry_result
)
if (NOT _retry_result EQUAL 0)
    message(FATAL_ERROR "Patch cannot be applied cleanly and is not already applied: ${PATCH_FILE}")
endif ()
message(STATUS "Applied patch after resetting source tree: ${PATCH_FILE}")

