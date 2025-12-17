# Apply multiple patches to the fetched ecbuild source
# Usage: cmake -D SRC_DIR=<source_dir> -P apply-ecbuild-patches.cmake

if(NOT DEFINED SRC_DIR)
  message(FATAL_ERROR "SRC_DIR not set (pass -D SRC_DIR=<source_dir>)")
endif()

# Resolve patch paths relative to this script's directory
get_filename_component(_SCRIPT_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
set(_PATCHES
  "${_SCRIPT_DIR}/ecbuild-version.patch"
  "${_SCRIPT_DIR}/ecbuild-windows-bash.patch"
)

# Verify git exists
execute_process(COMMAND git --version RESULT_VARIABLE _git_rv)
if(NOT _git_rv EQUAL 0)
  message(FATAL_ERROR "git is required to apply patches (git not found in PATH)")
endif()

foreach(_patch IN LISTS _PATCHES)
  if(NOT EXISTS "${_patch}")
    message(FATAL_ERROR "Patch file not found: ${_patch}")
  endif()
  message(STATUS "Checking patch: ${_patch}")
  # Check if patch can be applied cleanly
  execute_process(
    COMMAND git apply --check --ignore-whitespace --whitespace=fix "${_patch}"
    WORKING_DIRECTORY "${SRC_DIR}"
    RESULT_VARIABLE _check_rv
    OUTPUT_VARIABLE _check_out
    ERROR_VARIABLE  _check_err
  )
  if(_check_rv EQUAL 0)
    message(STATUS "Applying patch: ${_patch}")
    execute_process(
      COMMAND git apply --ignore-whitespace --whitespace=fix "${_patch}"
      WORKING_DIRECTORY "${SRC_DIR}"
      RESULT_VARIABLE _rv
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE  _err
    )
    if(NOT _rv EQUAL 0)
      message(FATAL_ERROR "Failed to apply patch: ${_patch}\n${_err}\n${_out}")
    endif()
  else()
    # Maybe already applied? Try reverse-check
    execute_process(
      COMMAND git apply -R --check --ignore-whitespace --whitespace=fix "${_patch}"
      WORKING_DIRECTORY "${SRC_DIR}"
      RESULT_VARIABLE _rev_check_rv
      OUTPUT_VARIABLE _rev_out
      ERROR_VARIABLE  _rev_err
    )
    if(_rev_check_rv EQUAL 0)
      message(STATUS "Patch already applied, skipping: ${_patch}")
    else()
      message(FATAL_ERROR "Patch does not apply and is not already applied: ${_patch}\n${_check_err}")
    endif()
  endif()
endforeach()

message(STATUS "All ecbuild patches applied successfully (or already present).")
