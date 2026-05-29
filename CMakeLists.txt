cmake_minimum_required(VERSION 3.20) # cxx_std_23 + FetchContent / FetchContent_MakeAvailable / SOURCE_SUBDIR

include(FetchContent)

if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif ()
if (POLICY CMP0168)
    # Avoid FetchContent sub-builds (Ninja recompact/restat), which can fail on
    # Windows reconfigure when subbuild files are transiently locked.
    cmake_policy(SET CMP0168 NEW)
endif ()

# vcpkg
if (NOT DEFINED ENV{VCPKG_ROOT})
    message(FATAL_ERROR "Please define the environment variable VCPKG_ROOT")
endif ()
set(CMAKE_TOOLCHAIN_FILE
        "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
set(VCPKG_LIBRARY_LINKAGE "static")

# ---- Build target selection (declared before project() so VCPKG_MANIFEST_FEATURES can be derived) ----
set(BUILD_FORECASTER ON  CACHE BOOL "Do you want to build AtmoSwing Forecaster ?")
set(BUILD_VIEWER     ON  CACHE BOOL "Do you want to build AtmoSwing Viewer ?")
set(BUILD_OPTIMIZER  ON  CACHE BOOL "Do you want to build AtmoSwing Optimizer ?")
set(BUILD_DOWNSCALER ON  CACHE BOOL "Do you want to build AtmoSwing Downscaler ?")

# Derive BUILD_TESTS / BUILD_BENCHMARK defaults following the same logic used after project()
if (BUILD_FORECASTER OR BUILD_OPTIMIZER OR BUILD_DOWNSCALER)
    set(BUILD_TESTS ON  CACHE BOOL "Do you want to build the tests (recommended) ?")
else ()
    set(BUILD_TESTS OFF CACHE BOOL "Do you want to build the tests (recommended) ?")
endif ()
if (BUILD_OPTIMIZER)
    set(BUILD_BENCHMARK OFF CACHE BOOL "Do you want to build the benchmark suite ?")
else ()
    set(BUILD_BENCHMARK OFF CACHE BOOL "Do you want to build the benchmark suite ?")
endif ()

# ---- Map build targets → vcpkg manifest features ----
# Note: VCPKG_MANIFEST_FEATURES is FORCE-set so it recomputes on every cmake run.
# If you change a BUILD_* option, run cmake twice: first run updates the cache,
# second run invokes vcpkg with the new feature set.
set(_vcpkg_features)
if (BUILD_VIEWER)
    list(APPEND _vcpkg_features "viewer")
endif ()
if (BUILD_FORECASTER)
    list(APPEND _vcpkg_features "forecaster")
endif ()
if (BUILD_TESTS)
    list(APPEND _vcpkg_features "tests")
endif ()
if (BUILD_BENCHMARK)
    list(APPEND _vcpkg_features "benchmark")
endif ()
set(VCPKG_MANIFEST_FEATURES "${_vcpkg_features}" CACHE STRING
        "vcpkg manifest features (auto-derived from BUILD_* options; do not edit manually)" FORCE)
message(STATUS "VCPKG_MANIFEST_FEATURES = ${VCPKG_MANIFEST_FEATURES}")
message(STATUS "VCPKG_TARGET_TRIPLET    = ${VCPKG_TARGET_TRIPLET}")

# Project name
project(AtmoSwing)


# ---- Shared compile options ----
# All project targets (asbase, asprocessing, executables, tests, benchmark) link to this INTERFACE
# library so they pick up the shared compile definitions, include paths, and C++ standard. Avoids
# global add_definitions / include_directories leaking into FetchContent sub-projects (wxWidgets,
# eccodes, vroomgis).
add_library(atmoswing_compile_options INTERFACE)
set(ATMOSWING_PROJECT_CXX_STANDARD "23" CACHE STRING
        "C++ standard for AtmoSwing project targets (dependencies use CMAKE_CXX_STANDARD)")
set_property(CACHE ATMOSWING_PROJECT_CXX_STANDARD PROPERTY STRINGS 23)
if (NOT ATMOSWING_PROJECT_CXX_STANDARD STREQUAL "23")
    message(FATAL_ERROR "AtmoSwing currently supports project targets only with C++23.")
endif ()
target_compile_features(atmoswing_compile_options INTERFACE cxx_std_23)


##### OPTIONS

# Version number
set(VERSION_MAJOR "3")
set(VERSION_MINOR "0")
set(VERSION_PATCH "11")
target_compile_definitions(atmoswing_compile_options INTERFACE
        ATMOSWING_MAJOR_VERSION=${VERSION_MAJOR}
        ATMOSWING_MINOR_VERSION=${VERSION_MINOR}
        ATMOSWING_PATCH_VERSION=${VERSION_PATCH})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/build/cmake ${CMAKE_MODULE_PATH})

# Choice of the targets
# BUILD_FORECASTER / BUILD_VIEWER / BUILD_OPTIMIZER / BUILD_DOWNSCALER are already in the cache
# (set before project() above). Just make them visible in cmake-gui.
mark_as_advanced(CLEAR BUILD_FORECASTER BUILD_VIEWER BUILD_OPTIMIZER BUILD_DOWNSCALER)
option(CREATE_INSTALLER "Do you want to generate installer packages ?" OFF)
if (BUILD_FORECASTER OR BUILD_OPTIMIZER OR BUILD_DOWNSCALER)
    mark_as_advanced(CLEAR BUILD_TESTS)
else ()
    set(BUILD_TESTS OFF)    # shadow cache so no test target is configured
    mark_as_advanced(BUILD_TESTS)
endif ()
if (BUILD_OPTIMIZER)
    mark_as_advanced(CLEAR BUILD_BENCHMARK)
else ()
    set(BUILD_BENCHMARK OFF)    # shadow cache
    mark_as_advanced(BUILD_BENCHMARK)
endif ()

if (NOT BUILD_FORECASTER AND NOT BUILD_VIEWER AND NOT BUILD_OPTIMIZER AND NOT BUILD_DOWNSCALER)
    message(FATAL_ERROR "Please select one or multiple target(s) to build.")
endif ()

# Output path
if (WIN32)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
else (WIN32)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})
endif (WIN32)

# Docker option
option(ON_DOCKER "Enable only if AtmoSwing is built for Docker." OFF)
if (ON_DOCKER)
    target_compile_definitions(atmoswing_compile_options INTERFACE ON_DOCKER)
endif (ON_DOCKER)

# Using the Intel MKL
option(USE_MKL "Do you want to use Intel MKL ?" OFF)

# MSYS condition
if (WIN32)
    option(USE_MSYS2 "Do you want to use MSYS2 ?" OFF)
    if(USE_MSYS2)
        set(MINGW false)
        set(MSYS true)
        set(MINGW_PATH "C:/msys64/mingw64" CACHE PATH "Path to installed libraries in MINGW")
    endif()
endif ()

# Enable Cppcheck
option(USE_CPPCHECK "Should we use Cppcheck (http://cppcheck.sourceforge.net/) ?" OFF)

# Enable code coverage
if (CMAKE_COMPILER_IS_GNUCXX)
    option(USE_CODECOV "Should we do code coverage with lcov ?" OFF)
else (CMAKE_COMPILER_IS_GNUCXX)
    set(USE_CODECOV OFF)
endif ()

# Enable GUIs
if (BUILD_VIEWER)
    set(USE_GUI ON)
else (BUILD_VIEWER)
    option(USE_GUI "Should AtmoSwing be built with a GUI ?" OFF)
endif (BUILD_VIEWER)

if (USE_GUI)
    target_compile_definitions(atmoswing_compile_options INTERFACE USE_GUI=1)
else (USE_GUI)
    target_compile_definitions(atmoswing_compile_options INTERFACE USE_GUI=0)
endif (USE_GUI)

# Disable testing tree
set(BUILD_TESTING OFF)
mark_as_advanced(BUILD_TESTING)

# Installer
if (WIN32)
    set(INSTALL_DIR_BIN .)
else (WIN32)
    set(INSTALL_DIR_BIN bin)
endif (WIN32)


##### COMPILE FLAGS (must be set before FetchContent so sub-projects inherit the same flags)

# Applying definitions and flags here ensures that FetchContent dependencies (wxWidgets, eccodes,
# vroomgis) are compiled with the same wxDEBUG_LEVEL as AtmoSwing itself.  Without this ordering,
# wxbase would be compiled with wxDEBUG_LEVEL=1 (the wx default) while AtmoSwing release objects
# use wxDEBUG_LEVEL=0, causing an LNK2005 duplicate-symbol error for wxFormatString::Validate.
include(atmoswing_definitions_and_flags)
message(STATUS "AtmoSwing target C++ standard = ${ATMOSWING_PROJECT_CXX_STANDARD}")
message(STATUS "FetchContent default C++ standard = ${ATMOSWING_THIRD_PARTY_CXX_STANDARD}")


##### DEPENDENCIES

# Prevent re-cloning/updating already-populated FetchContent dependencies on reconfigure.
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# wxWidgets
if (USE_GUI)
    set(wxUSE_GUI ON)
    set(WX_LIB_BASENAME "mswu")
else (USE_GUI)
    set(wxUSE_GUI OFF)
    set(WX_LIB_BASENAME "baseu")
endif (USE_GUI)
set(wxBUILD_INSTALL OFF)
set(wxBUILD_SAMPLES OFF)
set(wxBUILD_TESTS OFF)
set(wxBUILD_SHARED OFF)
set(wxUSE_XML ON)

FetchContent_Declare(wxWidgets
        GIT_REPOSITORY https://github.com/wxWidgets/wxWidgets.git
        GIT_TAG        v3.3.2
        UPDATE_DISCONNECTED TRUE)
FetchContent_MakeAvailable(wxWidgets)

target_include_directories(atmoswing_compile_options INTERFACE ${wxWidgets_SOURCE_DIR}/include)
if (WIN32)
    target_include_directories(atmoswing_compile_options INTERFACE
            ${wxWidgets_BINARY_DIR}/lib/vc_x64_lib/$<IF:$<CONFIG:Debug>,${WX_LIB_BASENAME}d,${WX_LIB_BASENAME}>)
    target_include_directories(atmoswing_compile_options INTERFACE
            ${wxWidgets_BINARY_DIR}/lib/vc_x64_dll/$<IF:$<CONFIG:Debug>,${WX_LIB_BASENAME}d,${WX_LIB_BASENAME}>)
else ()
    # On Unix-like systems, wxWidgets generates setup.h in lib/wx/include/<toolkit>-<configuration> Find the actual
    # setup.h directory dynamically (version-agnostic)
    target_include_directories(atmoswing_compile_options INTERFACE ${wxWidgets_BINARY_DIR}/lib)

    # Try to find the setup.h directory dynamically. First try the more common non-debug variant, then fall back to
    # debug variant if needed
    file(GLOB WX_SETUP_INCLUDE_DIR "${wxWidgets_BINARY_DIR}/lib/wx/include/base-unicode-static-*")
    if (NOT WX_SETUP_INCLUDE_DIR)
        file(GLOB WX_SETUP_INCLUDE_DIR "${wxWidgets_BINARY_DIR}/lib/wx/include/base-unicode-debug-static-*")
    endif ()

    if (WX_SETUP_INCLUDE_DIR)
        target_include_directories(atmoswing_compile_options INTERFACE ${WX_SETUP_INCLUDE_DIR})
        message(STATUS "wxWidgets setup dir = ${WX_SETUP_INCLUDE_DIR}")
    else ()
        message(WARNING "Could not find wxWidgets setup.h directory. Build may fail.")
    endif ()
endif ()
target_link_directories(atmoswing_compile_options INTERFACE ${wxWidgets_BINARY_DIR}/lib)
message(STATUS "wxWidgets include dir = ${wxWidgets_SOURCE_DIR}/include")
message(STATUS "wxWidgets lib dir = ${wxWidgets_BINARY_DIR}/lib")

# Make wxWidgets discoverable for find_package calls in dependencies
#set(wxWidgets_DIR "${wxWidgets_BINARY_DIR}" CACHE PATH "Path to wxWidgets config files")
#set(wxWidgets_FOUND TRUE CACHE BOOL "wxWidgets found via FetchContent")

# vroomgis (only needed by the Viewer; GDAL/GEOS are not available in server builds)
if (BUILD_VIEWER)
    set(BUILD_VROOMGIS_WITH_CONAN OFF)
    FetchContent_Declare(vroomgis
            GIT_REPOSITORY https://github.com/terranum-ch/vroomgis.git
            GIT_TAG        c87215c9d9fb086eefe75d6efbd10b98ac8b7290
            SOURCE_SUBDIR  vroomgis
            UPDATE_DISCONNECTED TRUE)
    FetchContent_MakeAvailable(vroomgis)
endif ()

# eccodes
set(ENABLE_FORTRAN OFF)
set(ENABLE_TESTS OFF)
set(ENABLE_EXTRA_TESTS OFF)
set(ENABLE_EXAMPLES OFF)
set(ENABLE_SAMPLES OFF)
set(ENABLE_SYMLINKS OFF)
if (BUILD_VIEWER)
    set(ENABLE_JPG ON CACHE BOOL "ecCodes JPEG support" FORCE)
    set(ENABLE_JPG_LIBJASPER ON CACHE BOOL "ecCodes JPEG backend: JasPer" FORCE)
    set(ENABLE_JPG_LIBOPENJPEG OFF CACHE BOOL "ecCodes JPEG backend: OpenJPEG" FORCE) # Prefer JasPer backend; avoids Linux CI link failures on opj_* symbols
else ()
    # Server/headless builds do not pull the vcpkg "viewer" feature, so JasPer is unavailable.
    set(ENABLE_JPG OFF CACHE BOOL "ecCodes JPEG support" FORCE)
    set(ENABLE_JPG_LIBJASPER OFF CACHE BOOL "ecCodes JPEG backend: JasPer" FORCE)
    set(ENABLE_JPG_LIBOPENJPEG OFF CACHE BOOL "ecCodes JPEG backend: OpenJPEG" FORCE)
endif ()
set(ECBUILD_LOG_LEVEL "OFF")  # Disable all logging
set(ECBUILD_LOG_FILE "${CMAKE_BINARY_DIR}/ecbuild.log")  # Set proper log file location
if (WIN32)
    set(ECCODES_ON_WINDOWS 1)
    set(BASH_EXE "C:/Program Files/Git/bin/bash.exe")
endif ()
FetchContent_Declare(eccodes
        GIT_REPOSITORY https://github.com/ecmwf/eccodes.git
        GIT_TAG 2.46.0
        UPDATE_DISCONNECTED TRUE
        PATCH_COMMAND ${CMAKE_COMMAND}
                      -DSOURCE_DIR=<SOURCE_DIR>
                      -DPATCH_FILE=${CMAKE_SOURCE_DIR}/build/patches/eccodes-conditional-subdirs.patch
                      -P ${CMAKE_SOURCE_DIR}/build/cmake/apply_git_patch_if_needed.cmake
)
FetchContent_MakeAvailable(eccodes)

# Allow find_package(eccodes) to locate ecbuild's generated config in sub-builds
# (needed when CMP0168 is not available, i.e. CMake < 3.30)
list(APPEND CMAKE_PREFIX_PATH "${eccodes_BINARY_DIR}")

# Mirror ecCodes definitions into this build tree using a filtered file set (only if not already done).
#
# Many files in eccodes' definitions tree are git symlinks (mode 120000) — e.g.
# definitions/mars/grib.oper.an.def → base.def. On Windows without Developer
# Mode (or with core.symlinks=false), git materializes them as small text files
# whose contents are the literal target name. Copying those verbatim makes
# eccodes choke with "Parser: syntax error at line 1". Resolve them at mirror
# time by detecting the symlink-shaped content and copying the target instead.
set(ECCODES_SHARE_DIR "${CMAKE_BINARY_DIR}/share/eccodes")
set(_eccodes_def_src "${eccodes_SOURCE_DIR}/definitions")
set(_eccodes_def_dst "${ECCODES_SHARE_DIR}/definitions")
# Bumped stamp suffix when the symlink-resolving logic landed so existing
# mirrors get rebuilt rather than reused with broken fake-symlink copies.
set(_eccodes_stamp_file "${_eccodes_def_dst}/.eccodes_copied_v2")

function(_atmoswing_resolve_eccodes_symlink src_path out_var)
    set(_current "${src_path}")
    set(_seen "${_current}")
    foreach (_iter RANGE 1 16)
        if (NOT EXISTS "${_current}")
            break()
        endif ()
        file(SIZE "${_current}" _size)
        # Git symlink payloads are tiny (a single relative path). Real def files
        # this small either contain syntax (quotes, =, ;) or newlines and won't
        # match the path-shaped regex below.
        if (_size EQUAL 0 OR _size GREATER 256)
            break()
        endif ()
        file(READ "${_current}" _content)
        string(STRIP "${_content}" _content)
        if (NOT _content MATCHES "^[A-Za-z0-9._/-]+$")
            break()
        endif ()
        get_filename_component(_dir "${_current}" DIRECTORY)
        get_filename_component(_target "${_dir}/${_content}" ABSOLUTE)
        if (NOT EXISTS "${_target}")
            break()
        endif ()
        list(FIND _seen "${_target}" _seen_idx)
        if (NOT _seen_idx EQUAL -1)
            break()
        endif ()
        list(APPEND _seen "${_target}")
        set(_current "${_target}")
    endforeach ()
    set(${out_var} "${_current}" PARENT_SCOPE)
endfunction()

if (EXISTS "${_eccodes_def_src}" AND NOT EXISTS "${_eccodes_stamp_file}")
    # Wipe any prior mirror (e.g. left by v1 with unresolved fake symlinks) so
    # we don't rely on configure_file's mtime check to overwrite stale content.
    file(REMOVE_RECURSE "${_eccodes_def_dst}")
    file(MAKE_DIRECTORY "${_eccodes_def_dst}")
    file(GLOB_RECURSE _eccodes_def_files RELATIVE "${_eccodes_def_src}"
            "${_eccodes_def_src}/*.def"
            "${_eccodes_def_src}/*.txt"
            "${_eccodes_def_src}/*.list"
            "${_eccodes_def_src}/*.table")
    foreach(_eccodes_rel_file ${_eccodes_def_files})
        get_filename_component(_eccodes_rel_dir "${_eccodes_rel_file}" DIRECTORY)
        if (NOT _eccodes_rel_dir STREQUAL "")
            file(MAKE_DIRECTORY "${_eccodes_def_dst}/${_eccodes_rel_dir}")
        endif ()
        _atmoswing_resolve_eccodes_symlink(
                "${_eccodes_def_src}/${_eccodes_rel_file}"
                _eccodes_resolved_src)
        configure_file("${_eccodes_resolved_src}" "${_eccodes_def_dst}/${_eccodes_rel_file}" COPYONLY)
    endforeach()
    file(TOUCH "${_eccodes_stamp_file}")
endif ()


##### FIND ANALYSIS TOOLS

# Cppcheck
if (USE_CPPCHECK)
    include(build/cmake/Findcppcheck.cmake)
    include(build/cmake/CppcheckTargets.cmake)
endif (USE_CPPCHECK)

# Code coverage
if (USE_CODECOV)
    include(CodeCoverage)
    append_coverage_compiler_flags()
    set(COVERAGE_EXCLUDES '/usr/*'
        '**/tests/*'
        '**/.conan/*'
        '**/_deps/*')
    setup_target_for_coverage(NAME atmoswing-coverage EXECUTABLE atmoswing-tests DEPENDENCIES atmoswing-tests)
endif (USE_CODECOV)


##### ADD COMPONENTS

# Provides atmoswing_stage_runtime_dlls(<target>); each app subdir calls it on
# its executable so the DLL staging runs as POST_BUILD on the exe.
include(atmoswing_stage_runtime_dlls)

add_subdirectory(src/shared_base)
if (BUILD_FORECASTER OR BUILD_OPTIMIZER OR BUILD_DOWNSCALER)
    add_subdirectory(src/shared_processing)
endif ()
if (BUILD_FORECASTER)
    add_subdirectory(src/app_forecaster)
endif()
if (BUILD_OPTIMIZER)
    add_subdirectory(src/app_optimizer)
endif()
if (BUILD_DOWNSCALER)
    add_subdirectory(src/app_downscaler)
endif()
if (BUILD_VIEWER)
    add_subdirectory(src/app_viewer)
endif()
if (BUILD_TESTS)
    add_subdirectory(tests)
endif ()
if (BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()


##### DISPLAY SOME INFORMATION

message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELWITHDEBINFO = ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")


#### INSTALLER

if (CREATE_INSTALLER)
    include(atmoswing_installer_options)
endif ()