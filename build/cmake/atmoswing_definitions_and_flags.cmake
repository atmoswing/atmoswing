# wxWidgets_CXX_FLAGS must stay in CMAKE_CXX_FLAGS so that FetchContent sub-projects
# (wxWidgets, eccodes, vroomgis) all see the same wxWidgets ABI flags. Per-project
# warning flags are attached to atmoswing_compile_options instead, so they never
# leak into third-party builds.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${wxWidgets_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DwxDEBUG_LEVEL=0 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG -DwxDEBUG_LEVEL=1 -D__WXDEBUG__")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DwxDEBUG_LEVEL=0 -DNDEBUG ")

message(STATUS "CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")
if (MINGW OR MSYS OR UNIX AND NOT APPLE)
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
        target_compile_options(atmoswing_compile_options INTERFACE -Wall -fno-strict-aliasing -Wno-sign-compare -Wno-attributes)
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
        target_compile_options(atmoswing_compile_options INTERFACE -Wall -fno-strict-aliasing -Wno-sign-compare -Wno-attributes)
    else ()
        target_compile_options(atmoswing_compile_options INTERFACE -Wall -fno-strict-aliasing -Wno-sign-compare -Wno-attributes -msse2)
    endif ()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer ")
elseif (WIN32)
    if (MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
        target_compile_options(atmoswing_compile_options INTERFACE /W4)
    endif ()
endif ()

# Keep CMAKE_CXX_STANDARD set globally for FetchContent dependencies (wxWidgets, eccodes,
# vroomgis). AtmoSwing project targets are set separately to C++23 via atmoswing_compile_options.
set(ATMOSWING_THIRD_PARTY_CXX_STANDARD "17" CACHE STRING
        "Default C++ standard for third-party FetchContent dependencies")
set_property(CACHE ATMOSWING_THIRD_PARTY_CXX_STANDARD PROPERTY STRINGS 17 20 23)
set(CMAKE_CXX_STANDARD ${ATMOSWING_THIRD_PARTY_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Shared compile definitions are attached to atmoswing_compile_options (declared at top-level
# CMakeLists.txt). Every project target links to this INTERFACE library, so it picks up these
# definitions without leaking them into FetchContent sub-projects.
target_compile_definitions(atmoswing_compile_options INTERFACE
        USE_JPEG2000
        EIGEN_NO_DEBUG)

if (WIN32)
    target_compile_definitions(atmoswing_compile_options INTERFACE
            _CRT_SECURE_NO_WARNINGS
            _CRTDBG_MAP_ALLOC)
endif ()

