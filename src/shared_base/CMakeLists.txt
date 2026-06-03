
# Project name
project(base)

##### DEPENDENCIES

find_package(CURL REQUIRED)
find_package(ZLIB REQUIRED)
if (NOT TARGET eccodes)
    find_package(eccodes REQUIRED)
endif ()
find_package(netCDF CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)


##### SOURCE FILES

# List source files
file(GLOB src_base_core_h core/*.h)
file(GLOB src_base_core_cpp core/*.cpp)
file(GLOB src_base_gui_h gui/*.h)
file(GLOB src_base_gui_cpp gui/*.cpp)
file(GLOB_RECURSE src_lib_awxled_h libs/awxled/*.h)
file(GLOB_RECURSE src_lib_awxled_cpp libs/awxled/*.cpp)
list(APPEND src_shared_base ${src_base_core_h})
list(APPEND src_shared_base ${src_base_core_cpp})
if (USE_GUI)
    list(APPEND src_shared_base ${src_base_gui_h} ${src_lib_awxled_h})
    list(APPEND src_shared_base ${src_base_gui_cpp} ${src_lib_awxled_cpp})
endif (USE_GUI)

# Remove eventual duplicates
list(REMOVE_DUPLICATES src_shared_base)

# Include source directories
list(APPEND inc_dirs "${PROJECT_SOURCE_DIR}/core/")
if (USE_GUI)
    list(APPEND inc_dirs "${PROJECT_SOURCE_DIR}/gui/")
endif (USE_GUI)


##### LIBRARIES

# Own libraries
add_library(asbase STATIC ${src_shared_base})

# Include directories: PUBLIC so consumers (apps, tests, benchmark) inherit transitively.
target_include_directories(asbase PUBLIC ${inc_dirs})

# Intel MKL (optional)
if (USE_MKL)
    find_package(MKL REQUIRED)
    target_include_directories(asbase PRIVATE ${MKL_INCLUDE_DIRS})
    target_link_libraries(asbase PUBLIC ${MKL_LIBRARIES})
    target_compile_definitions(asbase PUBLIC EIGEN_USE_MKL_ALL)
endif (USE_MKL)


##### DECLARE EXECUTABLE

# Cppcheck
if (USE_CPPCHECK)
    add_cppcheck(asbase STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
endif (USE_CPPCHECK)


##### LINKING

# PUBLIC link so consumers inherit shared defines/includes (USE_GUI, version macros,
# wxWidgets paths, USE_JPEG2000, EIGEN_NO_DEBUG, ...).
target_link_libraries(asbase PUBLIC atmoswing_compile_options)

target_link_libraries(asbase PUBLIC netCDF::netcdf eccodes Eigen3::Eigen)

if (BUILD_FORECASTER OR BUILD_VIEWER)
    target_link_libraries(asbase PUBLIC CURL::libcurl ZLIB::ZLIB)
endif (BUILD_FORECASTER OR BUILD_VIEWER)

if (USE_GUI)
    target_link_libraries(asbase PUBLIC wx::core wx::xml)
else (USE_GUI)
    target_link_libraries(asbase PUBLIC wx::base wx::xml)
endif (USE_GUI)


#### INSTALLER

if (WIN32)

    # Pack the Visual C++ Redistributable for Visual Studio
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION ${INSTALL_DIR_BIN})
    include(InstallRequiredSystemLibraries)
    install(FILES ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS} DESTINATION ${INSTALL_DIR_BIN})

endif (WIN32)
