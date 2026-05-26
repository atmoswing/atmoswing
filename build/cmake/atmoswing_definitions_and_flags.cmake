# Compilation flags (still applied globally via CMAKE_CXX_FLAGS so FetchContent sub-projects
# inherit consistent wxDEBUG_LEVEL / NDEBUG settings — see top-level CMakeLists for context).
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${wxWidgets_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DwxDEBUG_LEVEL=0 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG -DwxDEBUG_LEVEL=1 -D__WXDEBUG__")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DwxDEBUG_LEVEL=0 -DNDEBUG ")

message(STATUS "CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")
if (MINGW OR MSYS OR UNIX AND NOT APPLE)
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-strict-aliasing -Wno-sign-compare -Wno-attributes")
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-strict-aliasing -Wno-sign-compare -Wno-attributes")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-strict-aliasing -Wno-sign-compare -Wno-attributes -msse2")
    endif ()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer ")
elseif (WIN32)
    if (MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
        # Force to always compile with W4
        if (CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
            string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        else ()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
        endif ()
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

