# Compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${wxWidgets_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DwxDEBUG_LEVEL=0 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG -DwxDEBUG_LEVEL=1 -D__WXDEBUG__")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DwxDEBUG_LEVEL=0 -DNDEBUG ")

if (MINGW OR MSYS OR UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-strict-aliasing -Wno-sign-compare -Wno-attributes -msse2")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer ")
    if (USE_WXHGVERSION)
        set_target_properties(wxhgversion PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
    endif (USE_WXHGVERSION)
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
    if (USE_WXHGVERSION)
        set_target_properties(wxhgversion PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} /W2")
    endif (USE_WXHGVERSION)
endif ()

# Global definitions
add_definitions(-DUSE_JPEG2000)
add_definitions(-DEIGEN_NO_DEBUG)

if (UNIX)
    add_definitions(-std=c++11)
endif (UNIX)

if (WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_CRTDBG_MAP_ALLOC)
endif (WIN32)

if (USE_VLD)
    add_definitions(-DUSE_VLD)
endif (USE_VLD)

