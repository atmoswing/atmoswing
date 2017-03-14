# Own libraries
add_library(asbase STATIC ${src_shared_base})
if (BUILD_FORECASTER OR BUILD_OPTIMIZER)
    add_library(asprocessing STATIC ${src_shared_processing})
endif (BUILD_FORECASTER OR BUILD_OPTIMIZER)

if (BUILD_VIEWER)
    add_library(wxplotctrl STATIC ${src_lib_wxplotctrl})
endif (BUILD_VIEWER)

# Provided libraries
if (WIN32)
    set(USE_PROVIDED_LIBRARIES ON CACHE BOOL "Use the libraries downloaded from https://bitbucket.org/atmoswing/atmoswing")
    if(USE_PROVIDED_LIBRARIES)
        set(USE_PROVIDED_LIBRARIES_PATH CACHE PATH "Path to the libraries downloaded from https://bitbucket.org/atmoswing/atmoswing")
        if ("${USE_PROVIDED_LIBRARIES_PATH}" STREQUAL "")
            message(FATAL_ERROR "Please provide the path to the downloaded libraries, or disable the option USE_PROVIDED_LIBRARIES.")
        else()
            set(CUSTOM_LIBRARY_PATH "${USE_PROVIDED_LIBRARIES_PATH}")
            file(GLOB sub-dir ${CUSTOM_LIBRARY_PATH}/*)
            foreach(dir ${sub-dir})
                if(IS_DIRECTORY ${dir})
                    # Get directory to select the correct wxWidgets configuration
                    get_filename_component(lastdir ${dir} NAME)
                    string(FIND ${lastdir} "wxWidgets" dirwx)
                    if (${dirwx} EQUAL 0)
                        string(FIND ${lastdir} "nogui" dirwxnogui)
                        if (${dirwxnogui} GREATER 0)
                            if (NOT USE_GUI)
                                set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};${dir})
                            endif()
                        else ()
                            if (USE_GUI)
                                set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};${dir})
                            endif()
                        endif()
                    else ()
                        set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};${dir})
                    endif()
                endif()
            endforeach()
        endif()
    endif(USE_PROVIDED_LIBRARIES)
endif (WIN32)

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
include( "${wxWidgets_USE_FILE}" )
include_directories(${wxWidgets_INCLUDE_DIRS})

# NetCDF (has to be before GDAL)
mark_as_advanced(CLEAR NetCDF_INCLUDE_DIRECTORIES)
mark_as_advanced(CLEAR NetCDF_C_LIBRARY)
find_package(NetCDF REQUIRED)
include_directories(${NetCDF_INCLUDE_DIRECTORIES})

# Jasper
find_package(Jasper REQUIRED)
include_directories(${JASPER_INCLUDE_DIR})
include_directories(${JPEG_INCLUDE_DIR})

# PNG
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIRS})

# Grib2c
add_library(g2clib STATIC ${src_lib_g2clib})
include_directories("src/shared_base/libs/g2clib")

# libcURL
if (BUILD_FORECASTER OR BUILD_VIEWER)
    mark_as_advanced(CLEAR CURL_INCLUDE_DIR)
    mark_as_advanced(CLEAR CURL_LIBRARY)
    find_package(CURL REQUIRED)
    include_directories(${CURL_INCLUDE_DIRS})
else (BUILD_FORECASTER OR BUILD_VIEWER)
    # unset for wxhgversion
    unset(CURL_INCLUDE_DIR CACHE)
    unset(CURL_LIBRARY CACHE)
endif (BUILD_FORECASTER OR BUILD_VIEWER)

# GDAL
if (BUILD_FORECASTER OR BUILD_VIEWER)
    find_package(GDAL REQUIRED)
    include_directories(${GDAL_INCLUDE_DIRS})
else (BUILD_FORECASTER OR BUILD_VIEWER)
    # unset for wxhgversion
    unset(GDAL_INCLUDE_DIR CACHE)
    unset(GDAL_LIBRARY CACHE)
endif (BUILD_FORECASTER OR BUILD_VIEWER)

# Eigen
include_directories("src/shared_base/libs/eigen")

# vroomgis
if (BUILD_VIEWER)
    mark_as_advanced(SEARCH_GDAL)
    mark_as_advanced(SEARCH_GEOS)
    mark_as_advanced(SEARCH_GIS_LIB_PATH)
    mark_as_advanced(SEARCH_VROOMGIS_LIBS)
    mark_as_advanced(SEARCH_VROOMGIS_WXWIDGETS)
    mark_as_advanced(SQLITE_INCLUDE_DIR)
    mark_as_advanced(SQLITE_PATH)
    mark_as_advanced(wxWIDGETS_USING_SVN)
    include("src/app_viewer/libs/vroomgis/vroomgis/build/cmake/Use_vroomGISlib.cmake")
    link_libraries(${wxWidgets_LIBRARIES})
endif (BUILD_VIEWER)

# wxhgversion
if (USE_GUI)
    set(USE_WXHGVERSION 1)
    mark_as_advanced(USE_WXHGVERSION)
    include("src/shared_base/libs/wxhgversion/build/use_wxhgversion.cmake")
else (USE_GUI)
    set(USE_WXHGVERSION 0)
    mark_as_advanced(USE_WXHGVERSION)
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
    set(BUILD_GTEST ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    if(MINGW OR MSYS)
        set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)
    endif()
    add_subdirectory(test/libs/googletest)
    include_directories("${gtest_SOURCE_DIR}")
    include_directories("${gtest_SOURCE_DIR}/include")
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