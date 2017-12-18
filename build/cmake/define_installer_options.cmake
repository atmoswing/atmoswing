
if (WIN32)
    file(GLOB dlls ${CMAKE_BINARY_DIR}/*.dll)
    install(PROGRAMS ${dlls} DESTINATION bin)

    # pack the Visual C++ Redistributable for Visual Studio
    include(InstallRequiredSystemLibraries)
    install(FILES ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS} DESTINATION bin)
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
set(CPACK_PACKAGE_VENDOR "AtmoSwing")
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
    set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_ARCH}")
    set(CPACK_PACKAGE_EXECUTABLES "atmoswing-viewer;AtmoSwing viewer;atmoswing-forecaster;AtmoSwing forecaster;atmoswing-optimizer;AtmoSwing optimizer")
    set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "${CMAKE_PROJECT_NAME}-${CPACK_PACKAGE_VERSION}")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_LIST_DIR}/../cpack/license.txt")
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
    install(FILES art/logo/atmoswing.png DESTINATION share/pixmaps)
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