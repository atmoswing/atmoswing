# External projects
include(ExternalProject)
set(EXTERNAL_DIR ${CMAKE_BINARY_DIR}/external)

# Own libraries
add_library(asbase STATIC ${src_shared_base})
if (BUILD_FORECASTER OR BUILD_OPTIMIZER)
    add_library(asprocessing STATIC ${src_shared_processing})
endif (BUILD_FORECASTER OR BUILD_OPTIMIZER)

if (BUILD_VIEWER)
    add_library(wxplotctrl STATIC ${src_lib_wxplotctrl})
endif (BUILD_VIEWER)

# WxWidgets (adv lib nedded for the caldendar widget)
mark_as_advanced(wxWidgets_wxrc_EXECUTABLE)
mark_as_advanced(wxWidgets_with_GUI)
if (USE_MSYS2)
    set(wxWidgets_CONFIG_OPTIONS --prefix=${MINGW_PATH})
endif ()
if (USE_GUI)
    set(wxWidgets_with_GUI TRUE)
    find_package(wxWidgets COMPONENTS core base adv xml REQUIRED)
else (USE_GUI)
    set(wxWidgets_with_GUI FALSE)
    find_package(wxWidgets COMPONENTS base xml REQUIRED)
endif (USE_GUI)
include("${wxWidgets_USE_FILE}")
include_directories(${wxWidgets_INCLUDE_DIRS})

# NetCDF (has to be before GDAL)
mark_as_advanced(CLEAR NetCDF_INCLUDE_DIRECTORIES)
mark_as_advanced(CLEAR NetCDF_C_LIBRARY)
find_package(NetCDF REQUIRED)
include_directories(${NetCDF_INCLUDE_DIRECTORIES})

# Jasper
if (PATH_JASPER)
    find_package(Jasper REQUIRED PATHS PATH_JASPER)
    include_directories(${JASPER_INCLUDE_DIR})
    include_directories(${JPEG_INCLUDE_DIR})
else (PATH_JASPER)
    ExternalProject_Add(jasper
            URL "http://www.ece.uvic.ca/~mdadams/jasper/software/jasper-1.900.1.zip"
            #PATCH_COMMAND patch < http://www.linuxfromscratch.org/patches/blfs/svn/jasper-1.900.1-security_fixes-1.patch
            CONFIGURE_COMMAND ./configure --prefix=${EXTERNAL_DIR}
            BUILD_COMMAND make -j4
            INSTALL_COMMAND make install
            )
endif (PATH_JASPER)


# PNG
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIRS})

# Grib2c
add_library(g2clib STATIC ${src_lib_g2clib})
include_directories("src/shared_base/libs/g2clib")

# libcURL
if (BUILD_FORECASTER OR BUILD_VIEWER)
    if (PATH_CURL)
        mark_as_advanced(CLEAR CURL_INCLUDE_DIR)
        mark_as_advanced(CLEAR CURL_LIBRARY)
        find_package(CURL REQUIRED PATHS PATH_CURL)
        include_directories(${CURL_INCLUDE_DIRS})
    else(PATH_CURL)
        ExternalProject_Add(libcurl
                GIT_REPOSITORY https://github.com/curl/curl
                GIT_TAG ${CURL_GIT_TAG}
                CMAKE_ARGS -DHTTP_ONLY=ON -DCURL_STATICLIB=OFF -DBUILD_CURL_EXE=OFF -DBUILD_TESTING=0 -DCMAKE_INSTALL_PREFIX=${EXTERNAL_DIR}
                )
        set(CURL_INCLUDE_DIR ${EXTERNAL_DIR}/include)
    endif(PATH_CURL)
else (BUILD_FORECASTER OR BUILD_VIEWER)
    # unset for wxhgversion
    unset(CURL_INCLUDE_DIR CACHE)
    unset(CURL_LIBRARY CACHE)
endif (BUILD_FORECASTER OR BUILD_VIEWER)

# GDAL
if (BUILD_VIEWER)
    if (PATH_GDAL)
        find_package(GDAL REQUIRED PATHS PATH_GDAL)
        include_directories(${GDAL_INCLUDE_DIRS})
    else(PATH_GDAL)
        if (PATH_CURL)
            set(WITH_CURL_PATH PATH_CURL)
        else (PATH_CURL)
            set(WITH_CURL_PATH ${EXTERNAL_DIR})
        endif (PATH_CURL)

        if (MINGW OR MSYS OR UNIX)

            ExternalProject_Add(gdal
                    DEPENDS curl jasper
                    URL "http://download.osgeo.org/gdal/CURRENT/gdal-${GDAL_VERSION}.tar.gz"
                    CONFIGURE_COMMAND ./configure --prefix=${EXTERNAL_DIR} --with-jasper=${EXTERNAL_DIR} --with-curl=${WITH_CURL_PATH}
                    BUILD_COMMAND make -j4
                    INSTALL_COMMAND make install
                    )
        elseif(WIN32)


        endif()
    endif(PATH_GDAL)
else (BUILD_VIEWER)
    # unset for wxhgversion
    unset(GDAL_INCLUDE_DIR CACHE)
    unset(GDAL_LIBRARY CACHE)
endif (BUILD_VIEWER)

# Eigen
ExternalProject_Add(eigen
        URL "http://bitbucket.org/eigen/eigen/get/${EIGEN_VERSION}.tar.gz"
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND
        ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/eigen-prefix/src/eigen/Eigen
        ${EXTERNAL_DIR}/include/Eigen
        )

# vroomgis
if (BUILD_VIEWER)
    ExternalProject_Add(vroomgis
            URL "https://bitbucket.org/terranum/vroomgis/get/tip.tar.gz"
            SOURCE_SUBDIR vroomgis
            CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_DIR} -DVROOMGIS_PATH=vroomgis/src
            )
    link_libraries(${wxWidgets_LIBRARIES})
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

# Google Test
if (BUILD_TESTS)
    if (MINGW OR MSYS)
        set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)
    endif ()
    ExternalProject_Add(googletest
            GIT_REPOSITORY https://github.com/google/googletest
            CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_DIR}
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