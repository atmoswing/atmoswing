
# Project name
project(base)


##### SOURCE FILES

# List source files
file(GLOB_RECURSE src_base_core_h core/*.h)
file(GLOB_RECURSE src_base_core_cpp core/*.cpp)
file(GLOB_RECURSE src_base_gui_h gui/*.h)
file(GLOB_RECURSE src_base_gui_cpp gui/*.cpp)
file(GLOB_RECURSE src_base_img_h gui/img/*.h)
file(GLOB_RECURSE src_base_img_cpp gui/img/*.cpp)
file(GLOB_RECURSE src_lib_awxled_h libs/awxled/*.h)
file(GLOB_RECURSE src_lib_awxled_cpp libs/awxled/*.cpp)
list(APPEND src_shared_base ${src_base_core_h})
list(APPEND src_shared_base ${src_base_core_cpp})
if (USE_GUI)
    list(APPEND src_shared_base ${src_base_gui_h} ${src_base_img_h} ${src_lib_awxled_h})
    list(APPEND src_shared_base ${src_base_gui_cpp} ${src_base_img_cpp} ${src_lib_awxled_cpp})
endif (USE_GUI)
if (NOT BUILD_FORECASTER AND NOT BUILD_VIEWER)
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/core/asInternet.h")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/core/asInternet.cpp")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/core/asThreadInternetDownload.h")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/core/asThreadInternetDownload.cpp")
endif (NOT BUILD_FORECASTER AND NOT BUILD_VIEWER)

# Remove eventual duplicates
list(REMOVE_DUPLICATES src_shared_base)

# Include source directories
list(APPEND inc_dirs "${PROJECT_SOURCE_DIR}/core/")
if (USE_GUI)
    list(APPEND inc_dirs "${PROJECT_SOURCE_DIR}/gui/")
    list(APPEND inc_dirs "${PROJECT_SOURCE_DIR}/gui/img")
endif (USE_GUI)
include_directories(${inc_dirs})


##### LIBRARIES

# Intel MKL
if (USE_MKL)
    find_package(MKL REQUIRED)
    include_directories(${MKL_INCLUDE_DIRS})
    link_libraries(${MKL_LIBRARIES})
    add_definitions(-DEIGEN_USE_MKL_ALL)
endif (USE_MKL)

# wxVersion
if (USE_GUI)
    FetchContent_MakeAvailable(wxVersion)
    include_directories(${wxVersion_SOURCE_DIR}/src)
    include_directories(${wxVersion_BINARY_DIR})
else()
    set(USE_WXVERSION 0)
endif()

# Own libraries
add_library(asbase STATIC ${src_shared_base})


##### DECLARE EXECUTABLE

# Cppcheck
if (USE_CPPCHECK)
    add_cppcheck(asbase STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
endif (USE_CPPCHECK)


##### DEFINITIONS

# Specific definitions
if (USE_GUI)
    set_target_properties(asbase PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=1")
else (USE_GUI)
    set_target_properties(asbase PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=0")
endif (USE_GUI)


##### LINKING

target_link_libraries(asbase CONAN_PKG::netcdf)
target_link_libraries(asbase CONAN_PKG::proj)
target_link_libraries(asbase CONAN_PKG::eccodes)
target_link_libraries(asbase CONAN_PKG::pcre2)

if (BUILD_FORECASTER OR BUILD_VIEWER)
    target_link_libraries(asbase CONAN_PKG::libcurl)
endif (BUILD_FORECASTER OR BUILD_VIEWER)

if (USE_GUI)
    target_link_libraries(asbase CONAN_PKG::wxwidgets)
else (USE_GUI)
    target_link_libraries(asbase CONAN_PKG::wxbase)
endif (USE_GUI)

if (USE_WXVERSION)
    target_link_libraries(asbase wxVersion)
endif ()

#### INSTALLER

if (WIN32)

    # Pack the Visual C++ Redistributable for Visual Studio
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION ${INSTALL_DIR_BIN})
    include(InstallRequiredSystemLibraries)
    install(FILES ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS} DESTINATION ${INSTALL_DIR_BIN})

endif (WIN32)
