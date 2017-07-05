# External projects
include(ExternalProject)
set(EXTERNAL_DIR ${CMAKE_BINARY_DIR}/external)


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


# Donwload libraries
if (DOWNLOAD_LIBRARIES)

    # WxWidgets
    hunter_add_package(wxWidgets)
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
    include(${wxWidgets_USE_FILE})
    include_directories(${WXWIDGETS_ROOT}/include)



    # OpenSSL
    #hunter_add_package(OpenSSL)
    #find_package(OpenSSL REQUIRED)
    #include_directories("${OPENSSL_INCLUDE_DIR}")

    # libcURL
    #hunter_add_package(CURL)
    #find_package(CURL REQUIRED)
    #set(CURL_LIBRARIES CURL::libcurl)
    #include_directories(${CURL_INCLUDE_DIRS})

    # libcURL
    configure_file(build/cmake/CMakeListsCurl.txt.in ${CMAKE_BINARY_DIR}/libcurl/CMakeLists.txt)
    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/libcurl" )
    execute_process(COMMAND "${CMAKE_COMMAND}" --build .
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/libcurl" )
    find_package(OpenSSL REQUIRED)
    find_package(CURL REQUIRED PATHS ${EXTERNAL_DIR} NO_SYSTEM_ENVIRONMENT_PATH)

    # PNG
    hunter_add_package(PNG)
    find_package(PNG REQUIRED)

    # Jpg
    hunter_add_package(Jpeg)
    find_package(JPEG REQUIRED)

    # HDF5
    hunter_add_package(hdf5)
    find_package(ZLIB REQUIRED)
    find_package(szip REQUIRED)
    find_package(hdf5 REQUIRED)

    # NetCDF
    configure_file(build/cmake/CMakeListsNetcdf.txt.in ${CMAKE_BINARY_DIR}/libnetcdf/CMakeLists.txt)
    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/libnetcdf" )
    execute_process(COMMAND "${CMAKE_COMMAND}" --build .
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/libnetcdf" )
    find_package(netCDF REQUIRED PATHS ${EXTERNAL_DIR} NO_SYSTEM_ENVIRONMENT_PATH)

    # Jasper
    configure_file(build/cmake/CMakeListsJasper.txt.in ${CMAKE_BINARY_DIR}/libjasper/CMakeLists.txt)
    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/libjasper" )
    execute_process(COMMAND "${CMAKE_COMMAND}" --build .
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/libjasper" )
    find_library(JASPER_LIBRARIES NAMES jasper libjasper PATHS ${EXTERNAL_DIR}/lib NO_DEFAULT_PATH)

    # GDAL
    if (BUILD_VIEWER)
        configure_file(build/cmake/CMakeListsGdal.txt.in ${CMAKE_BINARY_DIR}/libgdal/CMakeLists.txt)
        execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/libgdal" )
        execute_process(COMMAND "${CMAKE_COMMAND}" --build .
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/libgdal" )
        find_package(GDAL REQUIRED PATHS ${EXTERNAL_DIR} NO_DEFAULT_PATH)
    else (BUILD_VIEWER)
        # unset for wxhgversion
        unset(GDAL_INCLUDE_DIR CACHE)
        unset(GDAL_LIBRARY CACHE)
    endif (BUILD_VIEWER)

else(DOWNLOAD_LIBRARIES)

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
        find_package(OpenSSL REQUIRED)
        find_package(CURL REQUIRED)
        include_directories(${CURL_INCLUDE_DIRS})
    else (BUILD_FORECASTER OR BUILD_VIEWER)
        # unset for wxhgversion
        unset(CURL_INCLUDE_DIR CACHE)
        unset(CURL_LIBRARY CACHE)
    endif (BUILD_FORECASTER OR BUILD_VIEWER)

    # PNG
    find_package(PNG REQUIRED)
    include_directories(${PNG_INCLUDE_DIRS})

    # NetCDF (has to be before GDAL)
    mark_as_advanced(CLEAR NetCDF_INCLUDE_DIRECTORIES)
    mark_as_advanced(CLEAR NetCDF_C_LIBRARY)
    find_package(NetCDF REQUIRED)
    include_directories(${NetCDF_INCLUDE_DIRECTORIES})

    # Jasper
    find_package(Jasper REQUIRED)
    include_directories(${JASPER_INCLUDE_DIR})
    include_directories(${JPEG_INCLUDE_DIR})

    # GDAL
    if (BUILD_VIEWER)
        find_package(GDAL REQUIRED)
        include_directories(${GDAL_INCLUDE_DIRS})
    else (BUILD_VIEWER)
        # unset for wxhgversion
        unset(GDAL_INCLUDE_DIR CACHE)
        unset(GDAL_LIBRARY CACHE)
    endif (BUILD_VIEWER)


endif(DOWNLOAD_LIBRARIES)















# vroomgis
if (BUILD_VIEWER)
    ExternalProject_Add(vroomgis
            URL https://bitbucket.org/terranum/vroomgis/get/tip.tar.gz
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