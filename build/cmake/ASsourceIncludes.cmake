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