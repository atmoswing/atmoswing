
install(FILES ${CMAKE_SOURCE_DIR}/license.txt ${CMAKE_SOURCE_DIR}/notice.txt DESTINATION share/atmoswing)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/data/ DESTINATION share/atmoswing)
install(DIRECTORY ${CMAKE_BINARY_DIR}/share DESTINATION .)
if (WIN32)
    install(
            DIRECTORY ${CMAKE_BINARY_DIR}/bin/
            DESTINATION .
            FILES_MATCHING
            PATTERN "*.dll"
    )
    install(
            DIRECTORY ${CMAKE_BINARY_DIR}/bin/fr
            DESTINATION .
    )
endif()

# COMMON PROPERTIES

set(CPACK_PACKAGE_VERSION_MAJOR "${VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${VERSION_PATCH}")
set(CPACK_PACKAGE_VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Analog Technique Model for Statistical weather downscalING and forecastING.")
set(CMAKE_PROJECT_DESCRIPTION "AtmoSwing stands for Analog Technique Model for Statistical weather downscalING and forecastING. AtmoSwing is an open source software that implements different Analog method variants in a very flexible way, so that they can be easily configured by means of XML files. It is written in C++, is object-oriented and multi-platform. AtmoSwing provides four tools: the Optimizer to establish the relationship between the predictand and predictors, the Downscaler to apply the method for climate impact studies, the Forecaster to perform operational forecasts, and the Viewer to display the results.")
set(CPACK_PACKAGE_VENDOR "AtmoSwing")
set(CPACK_STRIP_FILES ON) # tell cpack to strip all debug symbols from all files
if (USE_GUI)
    set(CPACK_PACKAGE_TAG "desktop")
else ()
    set(CPACK_PACKAGE_TAG "server")
endif ()

# IDENTIFY ARCHITECTURE

set(CPACK_PACKAGE_ARCH "unkown-architecture")

if (${CMAKE_SYSTEM_NAME} MATCHES Windows)
    if (CMAKE_CL_64)
        set(CPACK_PACKAGE_ARCH "win64")
    elseif (MINGW)
        set(CPACK_PACKAGE_ARCH "mingw32")
    elseif (WIN32)
        set(CPACK_PACKAGE_ARCH "win32")
    endif ()
endif ()

if (${CMAKE_SYSTEM_NAME} MATCHES Linux)
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES i686)
        set(CPACK_PACKAGE_ARCH "linux32")
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES x86_64)
        if (${CMAKE_CXX_FLAGS} MATCHES " -m32 ")
            set(CPACK_PACKAGE_ARCH "linux32")
        else ()
            set(CPACK_PACKAGE_ARCH "linux64")
        endif ()
    else ()
        set(CPACK_PACKAGE_ARCH "linux")
    endif ()
endif ()

if (${CMAKE_SYSTEM_NAME} MATCHES Darwin)
    set(CPACK_PACKAGE_ARCH "mac64")
endif (${CMAKE_SYSTEM_NAME} MATCHES Darwin)

# OS SPECIFIC PROPERTIES

set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${CPACK_PACKAGE_TAG}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_ARCH}")

if (APPLE)
    set(CPACK_GENERATOR "DragNDrop")
    set(CPACK_DMG_VOLUME_NAME "${CMAKE_PROJECT_NAME}")
    set(CPACK_DMG_FORMAT "UDBZ")
endif (APPLE)

if (WIN32)
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "AtmoSwing")
    set(CPACK_PACKAGE_EXECUTABLES "atmoswing-viewer;AtmoSwing viewer;atmoswing-forecaster;AtmoSwing forecaster;atmoswing-optimizer;AtmoSwing optimizer")
    set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "${CMAKE_PROJECT_NAME}-${CPACK_PACKAGE_VERSION}")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_LIST_DIR}/../cpack/license.txt")
    set(CPACK_CREATE_DESKTOP_LINKS "atmoswing-viewer;atmoswing-forecaster")

    # NSIS related parameters
    set(CPACK_GENERATOR "NSIS")
    set(CPACK_NSIS_DISPLAY_NAME "AtmoSwing")
    set(CPACK_NSIS_CONTACT "Pascal Horton pascal.horton@terranum.ch")
    set(CPACK_NSIS_HELP_LINK "www.atmoswing.org")
    set(CPACK_NSIS_EXECUTABLES_DIRECTORY ".")
    set(CPACK_NSIS_URL_INFO_ABOUT "www.atmoswing.org")
    set(CPACK_NSIS_MENU_LINKS "http://www.atmoswing.org" "www.atmoswing.org")
    set(CPACK_NSIS_COMPRESSOR "lzma")
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL "ON")
    set(CPACK_NSIS_MODIFY_PATH "ON")
    # Icon in the add/remove control panel. Must be an .exe file
    set(CPACK_NSIS_INSTALLED_ICON_NAME atmoswing-viewer.exe)
    if (CMAKE_CL_64)
        set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
    else ()
        set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
    endif ()

    # WIX related parameters
    set(CPACK_GENERATOR "WIX")
    set(CPACK_WIX_UPGRADE_GUID "E65EB70B-065D-4330-B901-83548C718A74")
    set(CPACK_WIX_PRODUCT_ICON "${CMAKE_CURRENT_LIST_DIR}/../../art/logo/atmoswing.png")
    set(CPACK_WIX_UI_DIALOG "${CMAKE_CURRENT_LIST_DIR}/../cpack/windows/installer_bg.jpg")
    set(CPACK_WIX_UI_BANNER "${CMAKE_CURRENT_LIST_DIR}/../cpack/windows/installer_top.jpg")
    set(CPACK_WIX_CMAKE_PACKAGE_REGISTRY "AtmoSwing")
    set(CPACK_WIX_PATCH_FILE "${CMAKE_CURRENT_LIST_DIR}/../cpack/windows/set-env-vars.xml")
endif (WIN32)

if (UNIX AND NOT APPLE)
    install(FILES art/logo/atmoswing.png DESTINATION /usr/share/icons)
    if (BUILD_FORECASTER)
        install(FILES build/cpack/linux/atmoswing-forecaster.desktop DESTINATION /usr/share/applications)
    endif (BUILD_FORECASTER)
    if (BUILD_VIEWER)
        install(FILES build/cpack/linux/atmoswing-viewer.desktop DESTINATION /usr/share/applications)
    endif (BUILD_VIEWER)
    if (BUILD_OPTIMIZER)
        install(FILES build/cpack/linux/atmoswing-optimizer.desktop DESTINATION /usr/share/applications)
    endif (BUILD_OPTIMIZER)
    if (BUILD_DOWNSCALER)
        install(FILES build/cpack/linux/atmoswing-downscaler.desktop DESTINATION /usr/share/applications)
    endif (BUILD_DOWNSCALER)
    set(CPACK_GENERATOR "DEB")
    set(CPACK_DEBIAN_PACKAGE_NAME "AtmoSwing")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Pascal Horton <pascal.horton@alumnil.unil.ch>")
    set(CPACK_PACKAGE_VENDOR "AtmoSwing")
    set(CPACK_PACKAGE_DESCRIPTION "AtmoSwing - Analog Technique Model for Statistical weather forecastING.")
    set(CPACK_DEBIAN_PACKAGE_SECTION "science")
    set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
    set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://atmoswing.org/")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_LIST_DIR}/../../license.txt")

    # Getting debian version (ex: 12.04)
    find_program(LSB_RELEASE_CMD lsb_release)
    mark_as_advanced(LSB_RELEASE_CMD)
    if (NOT LSB_RELEASE_CMD)
        message(FATAL_ERROR "Can not find lsb_release in your path.")
        set(DEBIAN_DISTRO_RELEASE "")
    else ()
        execute_process(COMMAND "${LSB_RELEASE_CMD}" -sr
                OUTPUT_VARIABLE DEBIAN_DISTRO_RELEASE
                OUTPUT_STRIP_TRAILING_WHITESPACE
                )
        if (DEBIAN_DISTRO_RELEASE)
            string(REGEX REPLACE "[.]" "" DEBIAN_DISTRO_RELEASE ${DEBIAN_DISTRO_RELEASE})
            message(STATUS "DEBIAN_DISTRO_RELEASE : ${DEBIAN_DISTRO_RELEASE}")
        endif ()
    endif ()

    # Dependencies (check versions on https://pkgs.org/)
    # Autogenerate dependency information (see https://www.guyrutenberg.com/2012/07/19/auto-detect-dependencies-when-building-debs-using-cmake/)
    set(CMAKE_BUILD_RPATH ${CMAKE_PREFIX_PATH})
    set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)

endif (UNIX AND NOT APPLE)

include(CPack)