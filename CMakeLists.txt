cmake_minimum_required(VERSION 2.8)


#### PROJECT

if (UNIX)
    project(atmoswing)
else (UNIX)
    project(AtmoSwing)
endif (UNIX)

# Version number
set(VERSION_MAJOR "1")
set(VERSION_MINOR "4")
set(VERSION_PATCH "2")

# Choice of the targets
set(BUILD_FORECASTER OFF CACHE BOOL "Do you want to build AtmoSwing Forecaster ?" )
set(BUILD_VIEWER OFF CACHE BOOL "Do you want to build AtmoSwing Viewer ?" )
set(BUILD_OPTIMIZER OFF CACHE BOOL "Do you want to build AtmoSwing Optimizer ?" )
if (BUILD_FORECASTER OR BUILD_OPTIMIZER)
    set(BUILD_TESTS ON CACHE BOOL "Do you want to build the tests (recommended) ?" )
    mark_as_advanced(CLEAR BUILD_TESTS)
else (BUILD_FORECASTER OR BUILD_OPTIMIZER)
    set(BUILD_TESTS OFF)
    mark_as_advanced(BUILD_TESTS)
endif (BUILD_FORECASTER OR BUILD_OPTIMIZER)

if (NOT BUILD_FORECASTER AND NOT BUILD_VIEWER AND NOT BUILD_OPTIMIZER)
    message(FATAL_ERROR "Please select one or multiple target(s) to build.")
endif (NOT BUILD_FORECASTER AND NOT BUILD_VIEWER AND NOT BUILD_OPTIMIZER)

# Output path
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BUILD_TYPE})

# Enable Visual Leak Detector
if (WIN32)
    set(USE_VLD OFF CACHE BOOL "Sould we use Visual Leak Detector (https://vld.codeplex.com) ?" )
else (WIN32)
    set(USE_VLD OFF)
endif (WIN32)

# Enable Cppcheck
set(USE_CPPCHECK OFF CACHE BOOL "Sould we use Cppcheck (http://cppcheck.sourceforge.net/) ?" )

# Enable GUIs
if (BUILD_FORECASTER OR BUILD_OPTIMIZER AND NOT BUILD_VIEWER)
    set(USE_GUI OFF CACHE BOOL "Sould we build the Forecaster / Optimizer with a GUI ?" )
else (BUILD_FORECASTER OR BUILD_OPTIMIZER AND NOT BUILD_VIEWER)
    set(USE_GUI ON)
endif (BUILD_FORECASTER OR BUILD_OPTIMIZER AND NOT BUILD_VIEWER)

# Enable CUDA
if (BUILD_OPTIMIZER)
    set(USE_CUDA OFF CACHE BOOL "Sould we compile with CUDA GPU support (not stable yet) ?" )
    mark_as_advanced(CLEAR USE_CUDA)
else (BUILD_OPTIMIZER)
    set(USE_CUDA OFF)
    mark_as_advanced(USE_CUDA)
endif (BUILD_OPTIMIZER)

# Addition of custom modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/build/cmake)


#### SOURCE FILES LIST

# Folder shared_base
file(GLOB_RECURSE src_base_core_h src/shared_base/core/*.h)
file(GLOB_RECURSE src_base_core_cpp src/shared_base/core/*.cpp)
file(GLOB_RECURSE src_base_gui_h src/shared_base/gui/*.h)
file(GLOB_RECURSE src_base_gui_cpp src/shared_base/gui/*.cpp)
file(GLOB_RECURSE src_base_img_h src/shared_base/gui/img/*.h)
file(GLOB_RECURSE src_base_img_cpp src/shared_base/gui/img/*.cpp)
file(GLOB_RECURSE src_lib_awxled_h src/shared_base/libs/awxled/*.h)
file(GLOB_RECURSE src_lib_awxled_cpp src/shared_base/libs/awxled/*.cpp)
list(APPEND src_shared_base ${src_base_core_h})
list(APPEND src_shared_base ${src_base_core_cpp})
if (USE_GUI)
    list(APPEND src_shared_base ${src_base_gui_h} ${src_base_img_h} ${src_lib_awxled_h})
    list(APPEND src_shared_base ${src_base_gui_cpp} ${src_base_img_cpp} ${src_lib_awxled_cpp})
endif (USE_GUI)
if (NOT BUILD_FORECASTER AND NOT BUILD_VIEWER)
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/src/shared_base/core/asInternet.h")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/src/shared_base/core/asInternet.cpp")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/src/shared_base/core/asThreadInternetDownload.h")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/src/shared_base/core/asThreadInternetDownload.cpp")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/src/shared_base/core/asFileGrib2.h")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/src/shared_base/core/asFileGrib2.cpp")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/src/shared_base/core/asDataPredictorRealtime.h")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/src/shared_base/core/asDataPredictorRealtime.cpp")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/src/shared_base/core/asDataPredictorRealtimeGfsForecast.h")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/src/shared_base/core/asDataPredictorRealtimeGfsForecast.cpp")
endif (NOT BUILD_FORECASTER AND NOT BUILD_VIEWER)

# Folder shared_processing
file(GLOB_RECURSE src_proc_core_h src/shared_processing/core/*.h)
file(GLOB_RECURSE src_proc_core_cpp src/shared_processing/core/*.cpp)
file(GLOB src_proc_thread_h src/shared_base/core/asThread.h)
file(GLOB src_proc_thread_cpp src/shared_base/core/asThread.cpp)
list(APPEND src_shared_processing ${src_proc_core_h} ${src_proc_thread_h})
list(APPEND src_shared_processing ${src_proc_core_cpp} ${src_proc_thread_cpp})

# Folder app_forecaster
file(GLOB_RECURSE src_forec_core_h src/app_forecaster/core/*.h)
file(GLOB_RECURSE src_forec_core_cpp src/app_forecaster/core/*.cpp)
file(GLOB_RECURSE src_forec_gui_h src/app_forecaster/gui/*.h)
file(GLOB_RECURSE src_forec_gui_cpp src/app_forecaster/gui/*.cpp)
list(APPEND src_app_forecaster ${src_forec_core_h})
list(APPEND src_app_forecaster ${src_forec_core_cpp})
if (USE_GUI)
    list(APPEND src_app_forecaster ${src_forec_gui_h})
    list(APPEND src_app_forecaster ${src_forec_gui_cpp})
endif (USE_GUI)

# Folder app_optimizer
file(GLOB_RECURSE src_optim_core_h src/app_optimizer/core/*.h)
file(GLOB_RECURSE src_optim_core_cpp src/app_optimizer/core/*.cpp)
file(GLOB_RECURSE src_optim_gui_h src/app_optimizer/gui/*.h)
file(GLOB_RECURSE src_optim_gui_cpp src/app_optimizer/gui/*.cpp)
list(APPEND src_app_optimizer ${src_optim_core_h})
list(APPEND src_app_optimizer ${src_optim_core_cpp})
if (USE_GUI)
    list(APPEND src_app_optimizer ${src_optim_gui_h})
    list(APPEND src_app_optimizer ${src_optim_gui_cpp})
endif (USE_GUI)

# Folder app_viewer
file(GLOB_RECURSE src_viewer_core_h src/app_viewer/core/*.h)
file(GLOB_RECURSE src_viewer_core_cpp src/app_viewer/core/*.cpp)
file(GLOB_RECURSE src_viewer_gui_h src/app_viewer/gui/*.h)
file(GLOB_RECURSE src_viewer_gui_cpp src/app_viewer/gui/*.cpp)
list(APPEND src_app_viewer ${src_viewer_core_h} ${src_viewer_gui_h})
list(APPEND src_app_viewer ${src_viewer_core_cpp} ${src_viewer_gui_cpp})

# wxPlotCtrl library (and dependences)
file(GLOB_RECURSE src_lib_wxmathplot_h src/app_viewer/libs/wxmathplot/*.h)
file(GLOB_RECURSE src_lib_wxmathplot_cpp src/app_viewer/libs/wxmathplot/*.cpp)
file(GLOB_RECURSE src_lib_wxplotctrl_h src/app_viewer/libs/wxplotctrl/src/*.h)
file(GLOB_RECURSE src_lib_wxplotctrl_hh src/app_viewer/libs/wxplotctrl/src/*.hh)
file(GLOB_RECURSE src_lib_wxplotctrl_cpp src/app_viewer/libs/wxplotctrl/src/*.cpp)
file(GLOB_RECURSE src_lib_wxplotctrl_c src/app_viewer/libs/wxplotctrl/src/*.c)
file(GLOB_RECURSE src_lib_wxthings_cpp src/app_viewer/libs/wxthings/src/*.cpp)
list(APPEND src_lib_wxplotctrl ${src_lib_wxmathplot_h} ${src_lib_wxplotctrl_h} ${src_lib_wxplotctrl_hh})
list(APPEND src_lib_wxplotctrl ${src_lib_wxmathplot_cpp} ${src_lib_wxplotctrl_cpp} ${src_lib_wxplotctrl_c} ${src_lib_wxthings_cpp})

# Folder test
file(GLOB_RECURSE src_tests_h test/src/*.h)
file(GLOB_RECURSE src_tests_cpp test/src/*.cpp)
list(APPEND src_tests ${src_tests_h} ${src_viewer_core_h} ${src_optim_core_h}) # Include optimization files anyway (to test the analogue method)
list(APPEND src_tests ${src_tests_cpp} ${src_viewer_core_cpp} ${src_optim_core_cpp})
if (BUILD_FORECASTER)
    list(APPEND src_tests ${src_forec_core_h})
    list(APPEND src_tests ${src_forec_core_cpp})
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/app_forecaster/core/asGlobVarsForecaster.h")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/app_forecaster/core/asGlobVarsForecaster.cpp")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/app_forecaster/core/AtmoswingAppForecaster.h")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/app_forecaster/core/AtmoswingAppForecaster.cpp")
else (BUILD_FORECASTER)
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/test/src/asDataPredictorRealtimeGeneralGridTest.cpp")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/test/src/asDataPredictorRealtimeRegularGridTest.cpp")
endif (BUILD_FORECASTER)
list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/app_viewer/core/asGlobVarsViewer.h")
list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/app_viewer/core/asGlobVarsViewer.cpp")
list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/app_viewer/core/AtmoswingAppViewer.h")
list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/app_viewer/core/AtmoswingAppViewer.cpp")
list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/app_optimizer/core/AtmoswingAppOptimizer.h")
list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/src/app_optimizer/core/AtmoswingAppOptimizer.cpp")

# CUDA library
if (USE_CUDA)
    file(GLOB_RECURSE src_proc_core_cu src/shared_processing/core/*.cu)
    file(GLOB_RECURSE src_proc_core_cuh src/shared_processing/core/*.cuh)
    list(APPEND src_cuda ${src_proc_core_cu})
    list(APPEND src_cuda ${src_proc_core_cuh})
    list(APPEND src_app_optimizer ${src_proc_core_cuh})
    if (WIN32)
        # Bug with VS 2012 and CUDA in previous versions
        cmake_minimum_required(VERSION 2.8.12.2)
    endif (WIN32)
endif (USE_CUDA)

# Remove eventual duplicates
list(REMOVE_DUPLICATES src_shared_base)
list(REMOVE_DUPLICATES src_shared_processing)
list(REMOVE_DUPLICATES src_app_forecaster)
list(REMOVE_DUPLICATES src_app_optimizer)
list(REMOVE_DUPLICATES src_app_viewer)
list(REMOVE_DUPLICATES src_lib_wxplotctrl)
list(REMOVE_DUPLICATES src_tests)


#### INCLUDE DIRECTORIES

list(APPEND inc_dirs 
    "${PROJECT_SOURCE_DIR}/src/shared_base/core/"
    )

if (BUILD_FORECASTER)
    list(APPEND inc_dirs 
        "${PROJECT_SOURCE_DIR}/src/shared_processing/core/"
        "${PROJECT_SOURCE_DIR}/src/app_forecaster/core/"
        )
    if (USE_GUI)
        list(APPEND inc_dirs 
            "${PROJECT_SOURCE_DIR}/src/shared_base/gui/"
            "${PROJECT_SOURCE_DIR}/src/shared_base/gui/img/"
            "${PROJECT_SOURCE_DIR}/src/shared_base/libs/awxled/"
            "${PROJECT_SOURCE_DIR}/src/app_forecaster/gui/"
            )
    endif (USE_GUI)
endif (BUILD_FORECASTER)

if (BUILD_VIEWER)
    list(APPEND inc_dirs 
        "${PROJECT_SOURCE_DIR}/src/shared_base/gui/"
        "${PROJECT_SOURCE_DIR}/src/shared_base/gui/img/"
        "${PROJECT_SOURCE_DIR}/src/shared_base/libs/awxled/"
        "${PROJECT_SOURCE_DIR}/src/app_viewer/core/"
        "${PROJECT_SOURCE_DIR}/src/app_viewer/gui/"
        "${PROJECT_SOURCE_DIR}/src/app_viewer/libs/wxmathplot/"
        "${PROJECT_SOURCE_DIR}/src/app_viewer/libs/wxplotctrl/"
        "${PROJECT_SOURCE_DIR}/src/app_viewer/libs/wxplotctrl/include/"
        "${PROJECT_SOURCE_DIR}/src/app_viewer/libs/wxthings/include/"
        )
endif (BUILD_VIEWER)

if (BUILD_OPTIMIZER)
    list(APPEND inc_dirs 
        "${PROJECT_SOURCE_DIR}/src/shared_processing/core/"
        "${PROJECT_SOURCE_DIR}/src/app_optimizer/app/"
        "${PROJECT_SOURCE_DIR}/src/app_optimizer/core/"
        )
    if (USE_GUI)
        list(APPEND inc_dirs 
            "${PROJECT_SOURCE_DIR}/src/shared_base/gui/"
            "${PROJECT_SOURCE_DIR}/src/shared_base/gui/img/"
            "${PROJECT_SOURCE_DIR}/src/shared_base/libs/awxled/"
            "${PROJECT_SOURCE_DIR}/src/app_optimizer/gui/"
            )
    endif (USE_GUI)
endif (BUILD_OPTIMIZER)

if (BUILD_TESTS)
    list(APPEND inc_dirs 
        "${PROJECT_SOURCE_DIR}/test/src/"
        "${PROJECT_SOURCE_DIR}/src/shared_processing/core/"
        "${PROJECT_SOURCE_DIR}/src/app_forecaster/core/"
        "${PROJECT_SOURCE_DIR}/src/app_viewer/core/"
        "${PROJECT_SOURCE_DIR}/src/app_optimizer/core/"
        )
endif (BUILD_TESTS)

include_directories(${inc_dirs})


#### LIBRARIES

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
mark_as_advanced(CLEAR EIGEN3_INCLUDE_DIR)
find_package(Eigen3 3.0.0 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

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

# UnitTest++
if (BUILD_TESTS)
    include(build/cmake/FindUnitTest++.cmake)
    include_directories(${UNITTEST++_INCLUDE_DIR})
    mark_as_advanced(UnitTest++_DIR)
else (BUILD_TESTS)
	# unset for wxhgversion
	unset(UNITTEST++_INCLUDE_DIR CACHE)
	unset(UNITTEST++_LIBRARY_DEBUG CACHE)
	unset(UNITTEST++_LIBRARY_RELEASE CACHE)
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


#### DECLARE EXECUTABLE

if (WIN32)

    if (USE_GUI)
        if (BUILD_FORECASTER)
            add_executable (atmoswing-forecaster WIN32 ${src_app_forecaster} ${PROJECT_SOURCE_DIR}/src/shared_base/gui/resource.rc)
        endif (BUILD_FORECASTER)
        if (BUILD_OPTIMIZER)
            add_executable (atmoswing-optimizer WIN32 ${src_app_optimizer} ${PROJECT_SOURCE_DIR}/src/shared_base/gui/resource.rc)
        endif (BUILD_OPTIMIZER)
    else (USE_GUI)
        if (BUILD_FORECASTER)
            add_executable (atmoswing-forecaster ${src_app_forecaster})
        endif (BUILD_FORECASTER)
        if (BUILD_OPTIMIZER)
            add_executable (atmoswing-optimizer ${src_app_optimizer})
        endif (BUILD_OPTIMIZER)
    endif (USE_GUI)
    if (BUILD_VIEWER)
        add_executable (atmoswing-viewer WIN32 ${src_app_viewer} ${PROJECT_SOURCE_DIR}/src/shared_base/gui/resource.rc)
    endif (BUILD_VIEWER)
    if (BUILD_TESTS)
        add_executable (atmoswing-tests WIN32 ${src_tests})
    endif (BUILD_TESTS)
    
elseif (UNIX AND NOT APPLE)

    if (BUILD_FORECASTER)
        add_executable (atmoswing-forecaster ${src_app_forecaster})
    endif (BUILD_FORECASTER)
    if (BUILD_VIEWER)
        add_executable (atmoswing-viewer ${src_app_viewer})
    endif (BUILD_VIEWER)
    if (BUILD_OPTIMIZER)
        add_executable (atmoswing-optimizer ${src_app_optimizer})
    endif (BUILD_OPTIMIZER)
    if (BUILD_TESTS)
        add_executable (atmoswing-tests ${src_tests})
    endif (BUILD_TESTS)
    
elseif (APPLE)

    if (BUILD_FORECASTER)
        add_executable (atmoswing-forecaster MACOSX_BUNDLE ${src_app_forecaster})
    endif (BUILD_FORECASTER)
    if (BUILD_VIEWER)
        add_executable (atmoswing-viewer MACOSX_BUNDLE ${src_app_viewer})
    endif (BUILD_VIEWER)
    if (BUILD_OPTIMIZER)
        add_executable (atmoswing-optimizer MACOSX_BUNDLE ${src_app_optimizer})
    endif (BUILD_OPTIMIZER)
    if (BUILD_TESTS)
        add_executable (atmoswing-tests MACOSX_BUNDLE ${src_tests})
    endif (BUILD_TESTS)
    
else (WIN32)

    if (BUILD_FORECASTER)
        add_executable (atmoswing-forecaster ${src_app_forecaster})
    endif (BUILD_FORECASTER)
    if (BUILD_VIEWER)
        add_executable (atmoswing-viewer ${src_app_viewer})
    endif (BUILD_VIEWER)
    if (BUILD_OPTIMIZER)
        add_executable (atmoswing-optimizer ${src_app_optimizer})
    endif (BUILD_OPTIMIZER)
    if (BUILD_TESTS)
        add_executable (atmoswing-tests ${src_tests})
    endif (BUILD_TESTS)
    
endif (WIN32)


#### Cppcheck

if (USE_CPPCHECK)

	add_cppcheck (asbase STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
	if (BUILD_FORECASTER OR BUILD_OPTIMIZER)
		add_cppcheck (asprocessing STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
	endif (BUILD_FORECASTER OR BUILD_OPTIMIZER)
    if (BUILD_FORECASTER)
        add_cppcheck (atmoswing-forecaster STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
    endif (BUILD_FORECASTER)
    if (BUILD_VIEWER)
        add_cppcheck (atmoswing-viewer STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
    endif (BUILD_VIEWER)
    if (BUILD_OPTIMIZER)
        add_cppcheck (atmoswing-optimizer STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
    endif (BUILD_OPTIMIZER)
    if (BUILD_TESTS)
        add_cppcheck (atmoswing-tests STYLE POSSIBLE_ERROR FAIL_ON_WARNINGS)
    endif (BUILD_TESTS)
    
endif (USE_CPPCHECK)


#### FLAGS & DEFINES

# Compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${wxWidgets_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DwxDEBUG_LEVEL=0 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG -D__WXDEBUG__")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DwxDEBUG_LEVEL=0 -DNDEBUG ")

if (UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-strict-aliasing -Wno-sign-compare -Wno-attributes")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer ")
    if (BUILD_VIEWER)
        set_target_properties(vroomgis PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
        set_target_properties(wxplotctrl PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable -Wno-attributes")
    endif (BUILD_VIEWER)
    if (USE_WXHGVERSION)
        set_target_properties(wxhgversion PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
    endif (USE_WXHGVERSION)
elseif (WIN32)
    set_target_properties(asbase PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
    if (BUILD_FORECASTER)
        set_target_properties(atmoswing-forecaster PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
        set_target_properties(asprocessing PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
    endif (BUILD_FORECASTER)
    if (BUILD_VIEWER)
        set_target_properties(atmoswing-viewer PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
        set_target_properties(vroomgis PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W2")
        set_target_properties(wxplotctrl PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W2")
    endif (BUILD_VIEWER)
    if (BUILD_OPTIMIZER)
        set_target_properties(atmoswing-optimizer PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
        set_target_properties(asprocessing PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
    endif (BUILD_OPTIMIZER)
    if (BUILD_TESTS)
        set_target_properties(atmoswing-tests PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W4")
    endif (BUILD_TESTS)
    if (USE_WXHGVERSION)
        set_target_properties(wxhgversion PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -W2")
    endif (USE_WXHGVERSION)
endif (UNIX AND NOT APPLE)

# Global definitions
add_definitions(-DUSE_JPEG2000 -DTIXML_USE_STL)

if (WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_CRTDBG_MAP_ALLOC)
endif (WIN32)

if (USE_VLD)
    add_definitions(-DUSE_VLD)
endif (USE_VLD)

if (USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif (USE_CUDA)

# Specific definitions
if (USE_GUI)
    set_target_properties(asbase PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=1")
else (USE_GUI)
    set_target_properties(asbase PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=0")
endif (USE_GUI)
    
if (BUILD_FORECASTER)
    if (USE_GUI)
        set_target_properties(atmoswing-forecaster PROPERTIES COMPILE_DEFINITIONS "APP_FORECASTER; wxUSE_GUI=1")
        set_target_properties(asprocessing PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=1")
    else (USE_GUI)
        set_target_properties(atmoswing-forecaster PROPERTIES COMPILE_DEFINITIONS "APP_FORECASTER; wxUSE_GUI=0")
        set_target_properties(asprocessing PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=0")
    endif (USE_GUI)
endif (BUILD_FORECASTER)

if (BUILD_VIEWER)
    set_target_properties(atmoswing-viewer PROPERTIES COMPILE_DEFINITIONS "APP_VIEWER")
endif (BUILD_VIEWER)

if (BUILD_OPTIMIZER)
    if (USE_GUI)
        set_target_properties(atmoswing-optimizer PROPERTIES COMPILE_DEFINITIONS "APP_OPTIMIZER; MINIMAL_LINKS; wxUSE_GUI=1")
        set_target_properties(asprocessing PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=1")
    else (USE_GUI)
        set_target_properties(atmoswing-optimizer PROPERTIES COMPILE_DEFINITIONS "APP_OPTIMIZER; MINIMAL_LINKS; wxUSE_GUI=0")
        set_target_properties(asprocessing PROPERTIES COMPILE_DEFINITIONS "wxUSE_GUI=0")
    endif (USE_GUI)
endif (BUILD_OPTIMIZER)

if (BUILD_TESTS)
    if(WIN32)
        set_target_properties(atmoswing-tests PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
        set_target_properties(atmoswing-tests PROPERTIES COMPILE_DEFINITIONS "UNIT_TESTING; wxUSE_GUI=0; _CONSOLE")
    else(WIN32)
        set_target_properties(atmoswing-tests PROPERTIES COMPILE_DEFINITIONS "UNIT_TESTING; wxUSE_GUI=0")
    endif(WIN32)
endif (BUILD_TESTS)

message("CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message("CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")
message("CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}")
message("CMAKE_CXX_FLAGS_RELWITHDEBINFO = ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")


#### LINKING

target_link_libraries(asbase ${GDAL_LIBRARIES})
	
if (BUILD_FORECASTER)
	target_link_libraries(asbase ${GDAL_LIBRARIES})
    target_link_libraries(atmoswing-forecaster asbase)
    target_link_libraries(atmoswing-forecaster asprocessing)
    target_link_libraries(atmoswing-forecaster ${wxWidgets_LIBRARIES})
    target_link_libraries(atmoswing-forecaster ${NetCDF_LIBRARIES})
    target_link_libraries(atmoswing-forecaster ${CURL_LIBRARIES})
    target_link_libraries(atmoswing-forecaster ${GDAL_LIBRARIES})
    if (USE_WXHGVERSION)
        target_link_libraries(atmoswing-forecaster ${WXHGVERSION_NAME})
    endif (USE_WXHGVERSION)
    if (USE_VLD)
        target_link_libraries(atmoswing-forecaster ${VLD_LIBRARIES})
    endif (USE_VLD)
endif (BUILD_FORECASTER)

if (BUILD_VIEWER)
	target_link_libraries(asbase ${GDAL_LIBRARIES})
    target_link_libraries(atmoswing-viewer asbase)
    target_link_libraries(atmoswing-viewer ${wxWidgets_LIBRARIES})
    target_link_libraries(atmoswing-viewer ${NetCDF_LIBRARIES})
    target_link_libraries(atmoswing-viewer ${CURL_LIBRARIES})
    target_link_libraries(atmoswing-viewer ${GDAL_LIBRARIES})
    target_link_libraries(atmoswing-viewer ${VROOMGIS_LIBRARY})
	target_link_libraries(atmoswing-viewer wxplotctrl)
    if (USE_WXHGVERSION)
      target_link_libraries(atmoswing-viewer ${WXHGVERSION_NAME})
    endif (USE_WXHGVERSION)
    if (USE_VLD)
        target_link_libraries(atmoswing-viewer ${VLD_LIBRARIES})
    endif (USE_VLD)
    if (WIN32)
        target_link_libraries(atmoswing-viewer Psapi.lib)
    endif (WIN32)
endif (BUILD_VIEWER)

if (BUILD_OPTIMIZER)
    target_link_libraries(atmoswing-optimizer asbase)
    target_link_libraries(atmoswing-optimizer asprocessing)
    target_link_libraries(atmoswing-optimizer ${wxWidgets_LIBRARIES})
    target_link_libraries(atmoswing-optimizer ${NetCDF_LIBRARIES})
    if (USE_WXHGVERSION)
        target_link_libraries(atmoswing-optimizer ${WXHGVERSION_NAME})
    endif (USE_WXHGVERSION)
    if (USE_VLD)
        target_link_libraries(atmoswing-optimizer ${VLD_LIBRARIES})
    endif (USE_VLD)
    if (USE_CUDA)
        target_link_libraries(ascuda ${CUDA_CUDA_LIBRARY})
        add_dependencies(atmoswing-optimizer ascuda)
        target_link_libraries(atmoswing-optimizer ascuda)
    endif (USE_CUDA)
endif (BUILD_OPTIMIZER)

if (BUILD_TESTS)
    target_link_libraries(atmoswing-tests asbase)
    target_link_libraries(atmoswing-tests asprocessing)
    target_link_libraries(atmoswing-tests ${wxWidgets_LIBRARIES})
    target_link_libraries(atmoswing-tests ${NetCDF_LIBRARIES})
    target_link_libraries(atmoswing-tests ${CURL_LIBRARIES})
    target_link_libraries(atmoswing-tests ${UNITTEST++_LIBRARIES})
    if (BUILD_FORECASTER)
        target_link_libraries(atmoswing-tests ${GDAL_LIBRARIES})
    endif (BUILD_FORECASTER)
    if (USE_VLD)
        target_link_libraries(atmoswing-tests ${VLD_LIBRARIES})
    endif (USE_VLD)
    if (USE_CUDA)
        target_link_libraries(ascuda ${CUDA_CUDA_LIBRARY})
        add_dependencies(atmoswing-tests ascuda)
        target_link_libraries(atmoswing-tests ascuda)
    endif (USE_CUDA)
endif (BUILD_TESTS)


#### TESTS

if (BUILD_TESTS)
    set(CTEST_OUTPUT_ON_FAILURE TRUE)
    add_test(AtmoSwing-tests atmoswing-tests)
    include(CTest)
endif (BUILD_TESTS)


#### INSTALLER

set(INSTALL_DIR_BIN bin)
set(INSTALL_DIR_SHARE share/atmoswing)
if (WIN32)
    set(INSTALL_DIR_BIN .)
    set(INSTALL_DIR_SHARE .)
endif (WIN32)
 
if (BUILD_FORECASTER)
install ( TARGETS atmoswing-forecaster DESTINATION ${INSTALL_DIR_BIN})
endif (BUILD_FORECASTER)

if (BUILD_VIEWER)
install ( TARGETS atmoswing-viewer DESTINATION ${INSTALL_DIR_BIN})
endif (BUILD_VIEWER)

if (BUILD_OPTIMIZER)
# Do not include the optimizer yet.
# install ( TARGETS atmoswing-optimizer DESTINATION ${INSTALL_DIR_BIN})
endif (BUILD_OPTIMIZER)

if (WIN32)
    if (USE_PROVIDED_LIBRARIES)
        file (GLOB dlls ${USE_PROVIDED_LIBRARIES_PATH}/*.dll)
        install(
            PROGRAMS ${dlls}
            DESTINATION .
        )
    else (USE_PROVIDED_LIBRARIES)
        file (GLOB dlls bin/*.dll)
        install(
            PROGRAMS ${dlls}
            DESTINATION .
        )
    endif (USE_PROVIDED_LIBRARIES)

    # pack the Visual C++ Redistributable for Visual Studio
    include(InstallRequiredSystemLibraries)
endif (WIN32)
  
install (
  FILES license.txt notice.txt
  DESTINATION ${INSTALL_DIR_SHARE}
  )

install (
  DIRECTORY data
  DESTINATION ${INSTALL_DIR_SHARE}
  )
  
# COMMON PROPERTIES

set(CPACK_PACKAGE_VERSION_MAJOR "${VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${VERSION_PATCH}")
set(CPACK_PACKAGE_VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "AtmoSwing stands for Analog Technique Model for Statistical weather forecastING. The software allows for real-time precipitation forecasting based on a downscaling method, the analogue technique. It identifies analogue days, in terms of atmospheric circulation and humidity variables, in a long archive of past situations and then uses the corresponding measured precipitation to establish an empirical conditional distribution considered as the probabilistic forecast for the target day.")
set(CPACK_PACKAGE_VENDOR "Terranum")
set(CPACK_STRIP_FILES ON) # tell cpack to strip all debug symbols from all files

# IDENTIFY ARCHITECTURE

set(CPACK_PACKAGE_ARCH "unkown-architecture")

if(${CMAKE_SYSTEM_NAME} MATCHES Windows)
    if(CMAKE_CL_64)
        set(CPACK_PACKAGE_ARCH "win64")
    elseif(MINGW)
        set(CPACK_PACKAGE_ARCH "mingw32")
    elseif(WIN32)
        set(CPACK_PACKAGE_ARCH "win32")
    endif()
endif(${CMAKE_SYSTEM_NAME} MATCHES Windows)

if(${CMAKE_SYSTEM_NAME} MATCHES Linux)
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES i686)
        set(CPACK_PACKAGE_ARCH "linux32")
    elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86_64)
        if(${CMAKE_CXX_FLAGS} MATCHES " -m32 ")
            set(CPACK_PACKAGE_ARCH "linux32")
        else()
            set(CPACK_PACKAGE_ARCH "linux64")
        endif(${CMAKE_CXX_FLAGS} MATCHES " -m32 ")
    else()
        set(CPACK_PACKAGE_ARCH "linux")
    endif()
endif(${CMAKE_SYSTEM_NAME} MATCHES Linux)

if(${CMAKE_SYSTEM_NAME} MATCHES Darwin)
    set(CPACK_PACKAGE_ARCH "mac64")
endif(${CMAKE_SYSTEM_NAME} MATCHES Darwin)

# OS SPECIFIC PROPERTIES

if (APPLE)
    set(CPACK_GENERATOR "DragNDrop")
    set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}_${CPACK_PACKAGE_VERSION}")
    set(CPACK_DMG_VOLUME_NAME "${CMAKE_PROJECT_NAME}")
    set(CPACK_DMG_FORMAT "UDBZ")
endif (APPLE)

if (WIN32)
    set(CPACK_GENERATOR "NSIS")
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "AtmoSwing")
    set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_ARCH}-Setup")
    set(CPACK_PACKAGE_EXECUTABLES "atmoswing-viewer;AtmoSwing viewer;atmoswing-forecaster;AtmoSwing forecaster")
    set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "${CMAKE_PROJECT_NAME}-${CPACK_PACKAGE_VERSION}")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_LIST_DIR}/cpack/license.txt")
    set(CPACK_NSIS_DISPLAY_NAME "AtmoSwing")
    set(CPACK_NSIS_CONTACT "Pascal Horton pascal.horton@terranum.ch")
    set(CPACK_NSIS_HELP_LINK "www.atmoswing.org")
    set(CPACK_NSIS_EXECUTABLES_DIRECTORY ".")
    set(CPACK_NSIS_URL_INFO_ABOUT "www.atmoswing.org")
    set(CPACK_NSIS_MENU_LINKS
        "http://www.atmoswing.org" "www.atmoswing.org")
    set(CPACK_CREATE_DESKTOP_LINKS "atmoswing-viewer;atmoswing-forecaster")
    set(CPACK_NSIS_COMPRESSOR "lzma")
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL "ON")
    set(CPACK_NSIS_MODIFY_PATH "ON")
    # Icon in the add/remove control panel. Must be an .exe file 
    set(CPACK_NSIS_INSTALLED_ICON_NAME atmoswing-viewer.exe)
    if (CMAKE_CL_64) 
        set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64") 
    else()
        set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES") 
    endif()
endif(WIN32)

if (UNIX AND NOT APPLE)
    install(FILES art/icon/atmoswing.png DESTINATION share/pixmaps)
    if (BUILD_FORECASTER)
        install(FILES cpack/atmoswing-forecaster.desktop DESTINATION share/applications)
    endif (BUILD_FORECASTER)
    if (BUILD_VIEWER)
        install(FILES cpack/atmoswing-viewer.desktop DESTINATION share/applications)
    endif (BUILD_VIEWER)
    if (BUILD_OPTIMIZER)
        install(FILES cpack/atmoswing-optimizer.desktop DESTINATION share/applications)
    endif (BUILD_OPTIMIZER)
    set(CPACK_GENERATOR "DEB")
    set(CPACK_PACKAGE_NAME "${CMAKE_PROJECT_NAME}")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Pascal Horton <pascal.horton@terranum.ch>")
    set(CPACK_PACKAGE_VENDOR "Terranum")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS " ...  MRC ... ")
    set(CPACK_PACKAGE_DESCRIPTION "AtmoSwing - Analog Technique Model for Statistical weather forecastING.")
    set(CPACK_DEBIAN_PACKAGE_SECTION "science")
    set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
endif(UNIX AND NOT APPLE)
  
include(CPack)
