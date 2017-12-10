# External projects
include(ExternalProject)
set(EXTERNAL_DIR ${CMAKE_BINARY_DIR}/external)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${EXTERNAL_DIR}")

# WxWidgets (adv lib nedded for the caldendar widget)
mark_as_advanced(wxWidgets_wxrc_EXECUTABLE)
mark_as_advanced(wxWidgets_with_GUI)
if (USE_MSYS2)
    set(wxWidgets_CONFIG_OPTIONS --prefix=${MINGW_PATH})
endif (USE_MSYS2)
if (USE_GUI)
    set(wxWidgets_with_GUI TRUE)
    find_package(wxWidgets REQUIRED core base adv xml net)
else (USE_GUI)
    set(wxWidgets_with_GUI FALSE)
    find_package(wxWidgets REQUIRED base xml net)
endif (USE_GUI)
include("${wxWidgets_USE_FILE}")
include_directories(${wxWidgets_INCLUDE_DIRS})

# libcURL
if (BUILD_FORECASTER OR BUILD_VIEWER)
    mark_as_advanced(CLEAR CURL_INCLUDE_DIR)
    mark_as_advanced(CLEAR CURL_LIBRARY)
    if (UNIX)
        find_package(OpenSSL REQUIRED)
    endif (UNIX)
    find_package(CURL REQUIRED)
    include_directories(${CURL_INCLUDE_DIRS})
else (BUILD_FORECASTER OR BUILD_VIEWER)
    # unset for wxhgversion
    unset(CURL_INCLUDE_DIR CACHE)
    unset(CURL_LIBRARY CACHE)
endif (BUILD_FORECASTER OR BUILD_VIEWER)

# PNG
set(PNG_FIND_QUIETLY OFF)
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIRS})

# NetCDF (has to be before GDAL)
mark_as_advanced(CLEAR NETCDF_INCLUDE_DIR)
mark_as_advanced(CLEAR NETCDF_LIBRARY)
find_package(NetCDF 4 MODULE REQUIRED)
include_directories(${NETCDF_INCLUDE_DIRS})

# Jasper
find_package(Jasper REQUIRED)
include_directories(${JASPER_INCLUDE_DIR})
include_directories(${JPEG_INCLUDE_DIR})

# GDAL
if (BUILD_VIEWER)
    if (GDAL_ROOT)
        message(STATUS "GDAL_ROOT: ${GDAL_ROOT}")
        set(ENV{GDAL_ROOT} ${GDAL_ROOT})
    endif ()
    find_package(GDAL 2 REQUIRED)
    include_directories(${GDAL_INCLUDE_DIRS})
else (BUILD_VIEWER)
    # unset for wxhgversion
    unset(GDAL_INCLUDE_DIR CACHE)
    unset(GDAL_LIBRARY CACHE)
endif (BUILD_VIEWER)

# Eigen
ExternalProject_Add(eigen
        URL http://bitbucket.org/eigen/eigen/get/${EIGEN_VERSION}.tar.gz
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND
            ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_BINARY_DIR}/eigen-prefix/src/eigen/Eigen
            ${EXTERNAL_DIR}/include/Eigen
        )

# CUDA
if (USE_CUDA)
    find_package(CUDA 5.0 REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_20;-use_fast_math)
    set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE};-O3)
    set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};-G)
    set(CUDA_NVCC_FLAGS_RELWITHDEBINFO ${CUDA_NVCC_FLAGS_RELWITHDEBINFO};-lineinfo)
    cuda_add_library(ascuda STATIC ${src_cuda})
else (USE_CUDA)
    # unset for wxhgversion
    unset(CUDA_INCLUDE_DIRS CACHE)
    unset(CUDA_CUDA_LIBRARY CACHE)
endif (USE_CUDA)


# Link to all following targets
link_libraries(${wxWidgets_LIBRARIES})
link_libraries(${CURL_LIBRARIES})
link_libraries(${HDF5_LIBRARIES})
link_libraries(${JASPER_LIBRARIES})
link_libraries(${NETCDF_LIBRARIES})
link_libraries(${PNG_LIBRARIES})
link_libraries(${JPEG_LIBRARY})


# Own libraries
add_library(asbase STATIC ${src_shared_base})
if (BUILD_FORECASTER OR BUILD_OPTIMIZER)
    add_library(asprocessing STATIC ${src_shared_processing})
endif (BUILD_FORECASTER OR BUILD_OPTIMIZER)

# wxplotctrl
if (BUILD_VIEWER)
    add_library(wxplotctrl STATIC ${src_lib_wxplotctrl})
endif (BUILD_VIEWER)

# Grib2c
add_library(g2clib STATIC ${src_lib_g2clib})
include_directories("src/shared_base/libs/g2clib")

# vroomgis
if (BUILD_VIEWER)
    configure_file(build/cmake/GetVroomgis.txt ${CMAKE_BINARY_DIR}/vroomgis/CMakeLists.txt)
    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" . WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/vroomgis" )
    execute_process(COMMAND "${CMAKE_COMMAND}" --build . WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/vroomgis" )
    set(CMAKE_INSTALL_PREFIX:STRING ${EXTERNAL_DIR})
    set(SEARCH_VROOMGIS_LIBS:BOOL ON)
    set(SEARCH_GDAL:BOOL ON)
    set(SEARCH_GEOS:BOOL OFF)
    set(VROOMGIS_PATH:STRING vroomgis/src)
    set(SEARCH_GIS_LIB_PATH:STRING ${GDAL_ROOT})
    add_subdirectory(${CMAKE_BINARY_DIR}/vroomgis/vroomgis-prefix/src/vroomgis/vroomgis)
    include_directories(${CMAKE_BINARY_DIR}/vroomgis/vroomgis-prefix/src/vroomgis/vroomgis/src)
    include_directories(${CMAKE_BINARY_DIR}/vroomgis/vroomgis-prefix/src/vroomgis/vroomgis/art)
endif (BUILD_VIEWER)

# wxhgversion
if (USE_GUI)
    set(USE_WXHGVERSION 0)

#    set(USE_WXHGVERSION 1)
#    ExternalProject_Add(wxhgversion
#            URL "https://bitbucket.org/terranum/wxhgversion/get/tip.tar.gz"
#            PATCH_COMMAND cp build/use_wxhgversion.cmake CMakeLists.txt
#            CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_DIR}
#            )
#    include_directories(${EXTERNAL_DIR}/include)
#    link_directories(${EXTERNAL_DIR}/lib)
else (USE_GUI)
    set(USE_WXHGVERSION 0)
endif (USE_GUI)

# Google Test
if (BUILD_TESTS)
    if (MINGW OR MSYS)
        set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)
    endif ()
    ExternalProject_Add(googletest
            GIT_REPOSITORY https://github.com/google/googletest
            UPDATE_COMMAND ""
            CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_DIR}
            BUILD_COMMAND
                cmake
                --build .
                --config Release
            )
endif (BUILD_TESTS)

# Visual Leak Detector
if (USE_VLD)
    find_package(VLD)
    include_directories(${VLD_INCLUDE_DIRS})
else (USE_VLD)
    # unset for wxhgversion
    unset(VLD_INCLUDE_DIR CACHE)
    unset(VLD_LIBRARY CACHE)
    unset(VLD_LIBRARY_DEBUG CACHE)
    unset(VLD_LIBRARY_DIR CACHE)
    unset(VLD_ROOT_DIR CACHE)
endif (USE_VLD)

# Cppcheck
if (USE_CPPCHECK)
    include(build/cmake/Findcppcheck.cmake)
    include(build/cmake/CppcheckTargets.cmake)
endif (USE_CPPCHECK)

# Code coverage
if (USE_CODECOV)
    include(CodeCoverage)
    setup_target_for_coverage(${PROJECT_NAME}-coverage atmoswing-tests coverage)
endif (USE_CODECOV)

include_directories(${EXTERNAL_DIR}/include)
link_directories(${EXTERNAL_DIR}/lib)
link_directories(${EXTERNAL_DIR}/lib64)