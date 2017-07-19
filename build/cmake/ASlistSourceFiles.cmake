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
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/src/shared_base/core/asPredictorOper.h")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/src/shared_base/core/asPredictorOper.cpp")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/src/shared_base/core/asPredictorOperGfsForecast.h")
    list(REMOVE_ITEM src_shared_base "${PROJECT_SOURCE_DIR}/src/shared_base/core/asPredictorOperGfsForecast.cpp")
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
file(GLOB_RECURSE src_tests_h tests/src/*.h)
file(GLOB_RECURSE src_tests_cpp tests/src/*.cpp)
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
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/tests/src/asPredictorOperNwsGfsGeneralGridTest.cpp")
    list(REMOVE_ITEM src_tests "${PROJECT_SOURCE_DIR}/tests/src/asPredictorOperNwsGfsRegularGridTest.cpp")
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

# Grib2c library
file(GLOB_RECURSE src_lib_g2clib_h src/shared_base/libs/g2clib/*.h)
set(src_lib_g2clib_c
        src/shared_base/libs/g2clib/gridtemplates.c
        src/shared_base/libs/g2clib/drstemplates.c
        src/shared_base/libs/g2clib/pdstemplates.c
        src/shared_base/libs/g2clib/gbits.c
        src/shared_base/libs/g2clib/g2_unpack1.c
        src/shared_base/libs/g2clib/g2_unpack2.c
        src/shared_base/libs/g2clib/g2_unpack3.c
        src/shared_base/libs/g2clib/g2_unpack4.c
        src/shared_base/libs/g2clib/g2_unpack5.c
        src/shared_base/libs/g2clib/g2_unpack6.c
        src/shared_base/libs/g2clib/g2_unpack7.c
        src/shared_base/libs/g2clib/g2_free.c
        src/shared_base/libs/g2clib/g2_info.c
        src/shared_base/libs/g2clib/g2_getfld.c
        src/shared_base/libs/g2clib/simunpack.c
        src/shared_base/libs/g2clib/comunpack.c
        src/shared_base/libs/g2clib/pack_gp.c
        src/shared_base/libs/g2clib/reduce.c
        src/shared_base/libs/g2clib/specpack.c
        src/shared_base/libs/g2clib/specunpack.c
        src/shared_base/libs/g2clib/rdieee.c
        src/shared_base/libs/g2clib/mkieee.c
        src/shared_base/libs/g2clib/int_power.c
        src/shared_base/libs/g2clib/simpack.c
        src/shared_base/libs/g2clib/compack.c
        src/shared_base/libs/g2clib/cmplxpack.c
        src/shared_base/libs/g2clib/misspack.c
        src/shared_base/libs/g2clib/jpcpack.c
        src/shared_base/libs/g2clib/jpcunpack.c
        src/shared_base/libs/g2clib/pngpack.c
        src/shared_base/libs/g2clib/pngunpack.c
        src/shared_base/libs/g2clib/dec_jpeg2000.c
        src/shared_base/libs/g2clib/enc_jpeg2000.c
        src/shared_base/libs/g2clib/dec_png.c
        src/shared_base/libs/g2clib/enc_png.c
        src/shared_base/libs/g2clib/g2_create.c
        src/shared_base/libs/g2clib/g2_addlocal.c
        src/shared_base/libs/g2clib/g2_addgrid.c
        src/shared_base/libs/g2clib/g2_addfield.c
        src/shared_base/libs/g2clib/g2_gribend.c
        src/shared_base/libs/g2clib/getdim.c
        src/shared_base/libs/g2clib/g2_miss.c
        src/shared_base/libs/g2clib/getpoly.c
        src/shared_base/libs/g2clib/seekgb.c
        )
list(APPEND src_lib_g2clib ${src_lib_g2clib_h} ${src_lib_g2clib_c})

# Remove eventual duplicates
list(REMOVE_DUPLICATES src_shared_base)
list(REMOVE_DUPLICATES src_shared_processing)
list(REMOVE_DUPLICATES src_app_forecaster)
list(REMOVE_DUPLICATES src_app_optimizer)
list(REMOVE_DUPLICATES src_app_viewer)
list(REMOVE_DUPLICATES src_lib_wxplotctrl)
list(REMOVE_DUPLICATES src_tests)