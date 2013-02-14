# - Find wxWindows (wxWidgets) installation 
# This module finds if wxWindows/wxWidgets is installed and determines where 
# the include files and libraries are. It also determines what the name of
# the library is. This code sets the following variables:
#  
#  WXWIDGETS_FOUND     = system has WxWindows 
#  WXWIDGETS_LIBRARIES = path to the wxWindows libraries
#                        on Unix/Linux with additional 
#                        linker flags from 
#                        "wx-config --libs"
#  CMAKE_WXWIDGETS_CXX_FLAGS  = Compiler flags for wxWindows, 
#                               essentially "`wx-config --cxxflags`"
#                               on Linux
#  WXWIDGETS_INCLUDE_DIR      = where to find "wx/wx.h" and "wx/setup.h"
#  WXWIDGETS_LINK_DIRECTORIES = link directories, useful for rpath on
#                                Unix
#  WXWIDGETS_DEFINITIONS      = extra defines
# 
# DEPRECATED
#  CMAKE_WX_CAN_COMPILE
#  WXWIDGETS_LIBRARY
#  CMAKE_WX_CXX_FLAGS
#  WXWIDGETS_INCLUDE_PATH
#
# OPTIONS 
# If you need OpenGL support please 
#  SET(WXWIDGETS_USE_GL 1) 
# in your CMakeLists.txt *before* you include this file.
# 
# For convenience include Use_wxWindows.cmake in your project's
# CMakeLists.txt using INCLUDE(Use_wxWindows). 
# 
# USAGE 
#  SET(WXWIDGETS_USE_GL 1) 
#  FIND_PACKAGE(wxWindows)
# 
# NOTES
# wxWidgets 2.6.x is supported for monolithic builds 
# e.g. compiled  in wx/build/msw dir as:  
#  nmake -f makefile.vc BUILD=debug SHARED=0 USE_OPENGL=1 MONOLITHIC=1
#
# AUTHOR
# Jan Woetzel <http://www.mip.informatik.uni-kiel.de/~jw> (07/2003-01/2006)


# ------------------------------------------------------------------
# 
# -removed OPTION for CMAKE_WXWIDGETS_USE_GL. Force the developer to SET it before calling this.
# -major update for wx 2.6.2 and monolithic build option. (10/2005)
#
# STATUS 
# tested with:
#  cmake 1.6.7, Linux (Suse 7.3), wxWindows 2.4.0, gcc 2.95
#  cmake 1.6.7, Linux (Suse 8.2), wxWindows 2.4.0, gcc 3.3
#  cmake 1.6.7, Linux (Suse 8.2), wxWindows 2.4.1-patch1,  gcc 3.3
#  cmake 1.6.7, MS Windows XP home, wxWindows 2.4.1, MS Visual Studio .net 7 2002 (static build)
#  cmake 2.0.5 on Windows XP and Suse Linux 9.2
#  cmake 2.0.6 on Windows XP and Suse Linux 9.2, wxWidgets 2.6.2 MONOLITHIC build
#  cmake 2.2.2 on Windows XP, MS Visual Studio .net 2003 7.1 wxWidgets 2.6.2 MONOLITHIC build
#
# TODO
#  -OPTION for unicode builds
#  -further testing of DLL linking under MS WIN32
#  -better support for non-monolithic builds
#


IF(WIN32)
  SET(WIN32_STYLE_FIND 1)
ENDIF(WIN32)
IF(MINGW)
  SET(WIN32_STYLE_FIND 0)
  SET(UNIX_STYLE_FIND 1)
ENDIF(MINGW)
IF(UNIX)
  SET(UNIX_STYLE_FIND 1)
ENDIF(UNIX)


IF(WIN32_STYLE_FIND)

  ## ######################################################################
  ##
  ## Windows specific:
  ##
  ## candidates for root/base directory of wxwindows
  ## should have subdirs include and lib containing include/wx/wx.h
  ## fix the root dir to avoid mixing of headers/libs from different
  ## versions/builds:
  
  SET (WXWIDGETS_POSSIBLE_ROOT_PATHS
    $ENV{WXWIN}
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\wxWidgets_is1;Inno Setup: App Path]"  ## WX 2.6.x
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\wxWindows_is1;Inno Setup: App Path]"  ## WX 2.4.x
    C:\\wxWidgets-2.6.2 
    D:\\wxWidgets-2.6.2 
    C:\\wxWidgets-2.6.1 
    D:\\wxWidgets-2.6.1 
    C:\\wxWindows-2.4.2 
    D:\\wxWindows-2.4.2 
    )
  
  ## WX supports monolithic and multiple smaller libs (since 2.5.x), we prefer monolithic for now.
  ## monolithic = WX is built as a single big library
  ## e.g. compile on WIN32 as  "nmake -f makefile.vc MONOLITHIC=1 BUILD=debug SHARED=0 USE_OPENGL=1" (JW)
  OPTION(WXWIDGETS_USE_MONOLITHIC "Use monolithic build of WX??" OFF)
  MARK_AS_ADVANCED(WXWIDGETS_USE_MONOLITHIC)

  ## GL libs used?
  OPTION(WXWIDGETS_USE_GL "Use Wx with GL support(glcanvas)?" ON)
  MARK_AS_ADVANCED(WXWIDGETS_USE_GL)


  ## avoid mixing of headers and libs between multiple installed WX versions,
  ## select just one tree here: 
  FIND_PATH(WXWIDGETS_ROOT_DIR  include/wx/wx.h 
    ${WXWIDGETS_POSSIBLE_ROOT_PATHS} )  
  # MESSAGE("DBG found WXWIDGETS_ROOT_DIR: ${WXWIDGETS_ROOT_DIR}")
  
  
  ## find libs for combination of static/shared with release/debug
  ## be careful if you add something here, 
  ## avoid mixing of headers and libs of different wx versions, 
  ## there may be multiple WX version s installed. 
  SET (WXWIDGETS_POSSIBLE_LIB_PATHS
    "${WXWIDGETS_ROOT_DIR}/lib"
    ) 
  
  ## monolithic?
  IF (WXWIDGETS_USE_MONOLITHIC)
    
    FIND_LIBRARY(WXWIDGETS_STATIC_LIBRARY
      NAMES wx wxmsw wxmsw26 wxmsw27 wxmsw28 wxmsw29 wxmsw28u wxmsw29u
      PATHS 
      "${WXWIDGETS_ROOT_DIR}/lib/vc_lib"
      ${WXWIDGETS_POSSIBLE_LIB_PATHS}
      DOC "wxWindows static release build library" ) 
    
    FIND_LIBRARY(WXWIDGETS_STATIC_DEBUG_LIBRARY
      NAMES wxd wxmswd wxmsw26d  wxmsw27d wxmsw28d wxmsw29 wxmsw28ud wxmsw29ud
      PATHS 
      "${WXWIDGETS_ROOT_DIR}/lib/vc_lib"
      ${WXWIDGETS_POSSIBLE_LIB_PATHS}       
      DOC "wxWindows static debug build library" )
    
    FIND_LIBRARY(WXWIDGETS_SHARED_LIBRARY
      NAMES wxmsw26 wxmsw262 wxmsw24 wxmsw242 wxmsw241 wxmsw240 wx23_2 wx22_9 
      PATHS 
      "${WXWIDGETS_ROOT_DIR}/lib/vc_dll"
      ${WXWIDGETS_POSSIBLE_LIB_PATHS} 
      DOC "wxWindows shared release build library" )
    
    FIND_LIBRARY(WXWIDGETS_SHARED_DEBUG_LIBRARY
      NAMES wxmsw26d wxmsw262d wxmsw24d wxmsw241d wxmsw240d wx23_2d wx22_9d 
      PATHS 
      "${WXWIDGETS_ROOT_DIR}/lib/vc_dll"
      ${WXWIDGETS_POSSIBLE_LIB_PATHS} 
      DOC "wxWindows shared debug build library " )
	  
	


    ##
    ## required for WXWIDGETS_USE_GL
    ## gl lib is always build separate:
    ##
    FIND_LIBRARY(WXWIDGETS_STATIC_LIBRARY_GL
      NAMES wx_gl wxmsw_gl wxmsw26_gl wxmsw28u_gl wxmsw28_gl wxmsw29_gl
      PATHS 
      "${WXWIDGETS_ROOT_DIR}/lib/vc_lib"
      ${WXWIDGETS_POSSIBLE_LIB_PATHS} 
      DOC "wxWindows static release build GL library" )

    FIND_LIBRARY(WXWIDGETS_STATIC_DEBUG_LIBRARY_GL
      NAMES wxd_gl wxmswd_gl wxmsw26d_gl wxmsw28ud_gl wxmsw28d_gl wxmsw28d_gl
      PATHS 
      "${WXWIDGETS_ROOT_DIR}/lib/vc_lib"
      ${WXWIDGETS_POSSIBLE_LIB_PATHS} 
      DOC "wxWindows static debug build GL library" )
    

    FIND_LIBRARY(WXWIDGETS_STATIC_DEBUG_LIBRARY_PNG
      NAMES wxpngd 
      PATHS 
      "${WXWIDGETS_ROOT_DIR}/lib/vc_lib"
      ${WXWIDGETS_POSSIBLE_LIB_PATHS} 
      DOC "wxWindows static debug png library" )

    FIND_LIBRARY(WXWIDGETS_STATIC_LIBRARY_PNG
      NAMES wxpng
      PATHS 
      "${WXWIDGETS_ROOT_DIR}/lib/vc_lib"
      ${WXWIDGETS_POSSIBLE_LIB_PATHS} 
      DOC "wxWindows static png library" )
    
    FIND_LIBRARY(WXWIDGETS_STATIC_DEBUG_LIBRARY_TIFF
      NAMES wxtiffd 
      PATHS 
      "${WXWIDGETS_ROOT_DIR}/lib/vc_lib"
      ${WXWIDGETS_POSSIBLE_LIB_PATHS} 
      DOC "wxWindows static debug tiff library" )

    FIND_LIBRARY(WXWIDGETS_STATIC_LIBRARY_TIFF
      NAMES wxtiff
      PATHS 
      "${WXWIDGETS_ROOT_DIR}/lib/vc_lib"
      ${WXWIDGETS_POSSIBLE_LIB_PATHS} 
      DOC "wxWindows static tiff library" )
    
    FIND_LIBRARY(WXWIDGETS_STATIC_DEBUG_LIBRARY_JPEG
      NAMES wxjpegd  wxjpgd
      PATHS 
      "${WXWIDGETS_ROOT_DIR}/lib/vc_lib"
      ${WXWIDGETS_POSSIBLE_LIB_PATHS} 
      DOC "wxWindows static debug jpeg library" )

    FIND_LIBRARY(WXWIDGETS_STATIC_LIBRARY_JPEG
      NAMES wxjpeg wxjpg
      PATHS 
      "${WXWIDGETS_ROOT_DIR}/lib/vc_lib"
      ${WXWIDGETS_POSSIBLE_LIB_PATHS} 
      DOC "wxWindows static jpeg library" )
    
    FIND_LIBRARY(WXWIDGETS_STATIC_DEBUG_LIBRARY_ZLIB
      NAMES wxzlibd
      PATHS 
      "${WXWIDGETS_ROOT_DIR}/lib/vc_lib"
      ${WXWIDGETS_POSSIBLE_LIB_PATHS} 
      DOC "wxWindows static debug zlib library" )

    FIND_LIBRARY(WXWIDGETS_STATIC_LIBRARY_ZLIB
      NAMES wxzlib
      PATHS 
      "${WXWIDGETS_ROOT_DIR}/lib/vc_lib"
      ${WXWIDGETS_POSSIBLE_LIB_PATHS} 
      DOC "wxWindows static zib library" )

    FIND_LIBRARY(WXWIDGETS_STATIC_DEBUG_LIBRARY_REGEX
      NAMES wxregexd wxregexud
      PATHS 
      "${WXWIDGETS_ROOT_DIR}/lib/vc_lib"
      ${WXWIDGETS_POSSIBLE_LIB_PATHS} 
      DOC "wxWindows static debug regex library" )

    FIND_LIBRARY(WXWIDGETS_STATIC_LIBRARY_REGEX
      NAMES wxregex wxregexu
      PATHS 
      "${WXWIDGETS_ROOT_DIR}/lib/vc_lib"
      ${WXWIDGETS_POSSIBLE_LIB_PATHS} 
      DOC "wxWindows static regex library" )
 
	
	FIND_LIBRARY (WXWIDGETS_STATIC_DEBUG_LIBRARY_EXPAT
		NAMES wxexpatd wxexpatud
		PATHS
		"${WXWIDGETS_ROOT_DIR}/lib/vc_lib"
		${WXWIDGETS_POSSIBLE_LIB_PATHS} 
		DOC "wxWindows expat (XML) library" )
	
	FIND_LIBRARY (WXWIDGETS_STATIC_LIBRARY_EXPAT
		NAMES wxexpat wxexpatu
		PATHS
		"${WXWIDGETS_ROOT_DIR}/lib/vc_lib"
		${WXWIDGETS_POSSIBLE_LIB_PATHS} 
		DOC "wxWindows expat XML library" )

    
    ## untested:
    FIND_LIBRARY(WXWIDGETS_SHARED_LIBRARY_GL
      NAMES wx_gl wxmsw_gl wxmsw26_gl 
      PATHS 
      "${WXWIDGETS_ROOT_DIR}/lib/vc_dll"
      ${WXWIDGETS_POSSIBLE_LIB_PATHS} 
      DOC "wxWindows shared release build GL library" )

    FIND_LIBRARY(WXWIDGETS_SHARED_DEBUG_LIBRARY_GL
      NAMES wxd_gl wxmswd_gl wxmsw26d_gl 
      PATHS 
      "${WXWIDGETS_ROOT_DIR}/lib/vc_dll"      
      ${WXWIDGETS_POSSIBLE_LIB_PATHS} 
      DOC "wxWindows shared debug build GL library" )            
    
 
    ELSE (WXWIDGETS_USE_MONOLITHIC)
    ## WX is built as multiple small pieces libraries instead of monolithic
    
    ## DEPECATED (jw) replaced by more general WXWIDGETS_USE_MONOLITHIC ON/OFF
    # OPTION(WXWIDGETS_SEPARATE_LIBS_BUILD "Is wxWindows build with separate libs?" OFF)    
    
    ## HACK: This is very dirty.
    ## because the libs of a particular version are explicitly listed
    ## and NOT searched/verified.
    ## TODO:  Really search for each lib, then decide for 
    ## monolithic x debug x shared x GL (=16 combinations) for at least 18 libs 
    ## -->  about 288 combinations 
    ## thus we need a different approach so solve this correctly ...
    
    MESSAGE(STATUS "Warning: You are trying to use wxWidgets without monolithic build (WXWIDGETS_SEPARATE_LIBS_BUILD). This is a HACK, libraries are not verified! (JW).")
    
    SET(WXWIDGETS_STATIC_LIBS ${WXWIDGETS_STATIC_LIBS}
      wxbase26
      wxbase26_net
      wxbase26_odbc
      wxbase26_xml
      wxmsw26_adv
      wxmsw26_core
      wxmsw26_dbgrid
      wxmsw26_gl
      wxmsw26_html
      wxmsw26_media
      wxmsw26_qa
      wxmsw26_xrc
      wxexpat
      wxjpeg
      wxpng
      wxregex
      wxtiff
      wxzlib
      comctl32
      rpcrt4
      wsock32
      )
    ## HACK: feed in to optimized / debug libaries if both were FOUND. 
    SET(WXWIDGETS_STATIC_DEBUG_LIBS ${WXWIDGETS_STATIC_DEBUG_LIBS}
      wxbase26d
      wxbase26d_net
      wxbase26d_odbc
      wxbase26d_xml
      wxmsw26d_adv
      wxmsw26d_core
      wxmsw26d_dbgrid
      wxmsw26d_gl
      wxmsw26d_html
      wxmsw26d_media
      wxmsw26d_qa
      wxmsw26d_xrc
      wxexpatd
      wxjpegd
      wxpngd
      wxregexd
      wxtiffd
      wxzlibd
      comctl32
      rpcrt4
      wsock32
      )
  ENDIF (WXWIDGETS_USE_MONOLITHIC)
  
  
  ##
  ## now we should have found all WX libs available on the system.
  ## let the user decide which of the available onse to use.
  ## 
  
  ## if there is at least one shared lib available
  ## let user choose wether to use shared or static wxwindows libs 
  IF(WXWIDGETS_SHARED_LIBRARY OR WXWIDGETS_SHARED_DEBUG_LIBRARY)
    ## default value OFF because wxWindows MSVS default build is static
    OPTION(WXWIDGETS_USE_SHARED_LIBS
      "Use shared versions (dll) of wxWindows libraries?" OFF)
    MARK_AS_ADVANCED(WXWIDGETS_USE_SHARED_LIBS)
  ENDIF(WXWIDGETS_SHARED_LIBRARY OR WXWIDGETS_SHARED_DEBUG_LIBRARY)    
  
  ## add system libraries wxwindows always seems to depend on
  SET(WXWIDGETS_LIBRARIES ${WXWIDGETS_LIBRARIES}
    comctl32
    rpcrt4
    wsock32
    )  
  
  IF (NOT WXWIDGETS_USE_SHARED_LIBS)
    SET(WXWIDGETS_LIBRARIES ${WXWIDGETS_LIBRARIES}
      ##  these ones dont seem required, in particular  ctl3d32 is not neccesary (Jan Woetzel 07/2003)
      #   ctl3d32
      debug ${WXWIDGETS_STATIC_DEBUG_LIBRARY_ZLIB}   optimized ${WXWIDGETS_STATIC_LIBRARY_ZLIB}
      debug ${WXWIDGETS_STATIC_DEBUG_LIBRARY_REGEX}  optimized ${WXWIDGETS_STATIC_LIBRARY_REGEX}
	  debug ${WXWIDGETS_STATIC_DEBUG_LIBRARY_EXPAT}  optimized ${WXWIDGETS_STATIC_LIBRARY_EXPAT}
      debug ${WXWIDGETS_STATIC_DEBUG_LIBRARY_PNG}    optimized ${WXWIDGETS_STATIC_LIBRARY_PNG}
      debug ${WXWIDGETS_STATIC_DEBUG_LIBRARY_JPEG}   optimized ${WXWIDGETS_STATIC_LIBRARY_JPEG}
      debug ${WXWIDGETS_STATIC_DEBUG_LIBRARY_TIFF}   optimized ${WXWIDGETS_STATIC_LIBRARY_TIFF}
      )
  ENDIF (NOT WXWIDGETS_USE_SHARED_LIBS)

  ## opengl/glu: TODO/FIXME: better use FindOpenGL.cmake here 
  ## assume release versions of glu an dopengl, here.
  IF (WXWIDGETS_USE_GL)
    SET(WXWIDGETS_LIBRARIES ${WXWIDGETS_LIBRARIES}
      opengl32
      glu32 )
  ENDIF (WXWIDGETS_USE_GL)
  
  ##
  ## select between use of  shared or static wxWindows lib then set libs to use
  ## for debug and optimized build.  so the user can switch between debug and
  ## release build e.g. within MS Visual Studio without running cmake with a
  ## different build directory again.
  ## 
  ## then add the build specific include dir for wx/setup.h
  ## 
 
  IF(WXWIDGETS_USE_SHARED_LIBS)
    ##MESSAGE("DBG wxWindows use shared lib selected.")
    ## assume that both builds use the same setup(.h) for simplicity
    
    ## shared: both wx (debug and release) found?
    ## assume that both builds use the same setup(.h) for simplicity
    IF(WXWIDGETS_SHARED_DEBUG_LIBRARY AND WXWIDGETS_SHARED_LIBRARY)
      ##MESSAGE("DBG wx shared: debug and optimized found.")
      FIND_PATH(WXWIDGETS_INCLUDE_DIR_SETUPH  wx/setup.h
        ${WXWIDGETS_ROOT_DIR}/lib/mswdlld
        ${WXWIDGETS_ROOT_DIR}/lib/mswdll 
        ${WXWIDGETS_ROOT_DIR}/lib/vc_dll/mswd
        ${WXWIDGETS_ROOT_DIR}/lib/vc_dll/msw )
      SET(WXWIDGETS_LIBRARIES ${WXWIDGETS_LIBRARIES}
        debug     ${WXWIDGETS_SHARED_DEBUG_LIBRARY}
        optimized ${WXWIDGETS_SHARED_LIBRARY} )
      IF (WXWIDGETS_USE_GL)
        SET(WXWIDGETS_LIBRARIES ${WXWIDGETS_LIBRARIES}
          debug     ${WXWIDGETS_SHARED_DEBUG_LIBRARY_GL}
          optimized ${WXWIDGETS_SHARED_LIBRARY_GL} )
      ENDIF (WXWIDGETS_USE_GL)
    ENDIF(WXWIDGETS_SHARED_DEBUG_LIBRARY AND WXWIDGETS_SHARED_LIBRARY)
    
    ## shared: only debug wx lib found?
    IF(WXWIDGETS_SHARED_DEBUG_LIBRARY)
      IF(NOT WXWIDGETS_SHARED_LIBRARY)
        ##MESSAGE("DBG wx shared: debug (but no optimized) found.")
        FIND_PATH(WXWIDGETS_INCLUDE_DIR_SETUPH  wx/setup.h
          ${WXWIDGETS_ROOT_DIR}/lib/mswdlld 
          ${WXWIDGETS_ROOT_DIR}/lib/vc_dll/mswd  )        
        SET(WXWIDGETS_LIBRARIES ${WXWIDGETS_LIBRARIES}
          ${WXWIDGETS_SHARED_DEBUG_LIBRARY} )
        IF (WXWIDGETS_USE_GL) 
          SET(WXWIDGETS_LIBRARIES ${WXWIDGETS_LIBRARIES}
            ${WXWIDGETS_SHARED_DEBUG_LIBRARY_GL} )
        ENDIF (WXWIDGETS_USE_GL)
      ENDIF(NOT WXWIDGETS_SHARED_LIBRARY)
    ENDIF(WXWIDGETS_SHARED_DEBUG_LIBRARY)
    
    ## shared: only release wx lib found?
    IF(NOT WXWIDGETS_SHARED_DEBUG_LIBRARY)
      IF(WXWIDGETS_SHARED_LIBRARY)
        ##MESSAGE("DBG wx shared: optimized (but no debug) found.")
        FIND_PATH(WXWIDGETS_INCLUDE_DIR_SETUPH  wx/setup.h
          ${WXWIDGETS_ROOT_DIR}/lib/mswdll 
          ${WXWIDGETS_ROOT_DIR}/lib/vc_dll/msw  )
        SET(WXWIDGETS_LIBRARIES ${WXWIDGETS_LIBRARIES}
          ${WXWIDGETS_SHARED_DEBUG_LIBRARY} )
        IF (WXWIDGETS_USE_GL)
          SET(WXWIDGETS_LIBRARIES ${WXWIDGETS_LIBRARIES}
            ${WXWIDGETS_SHARED_DEBUG_LIBRARY_GL} )
        ENDIF (WXWIDGETS_USE_GL)
      ENDIF(WXWIDGETS_SHARED_LIBRARY)    
    ENDIF(NOT WXWIDGETS_SHARED_DEBUG_LIBRARY)
    
    ## shared: none found?
    IF(NOT WXWIDGETS_SHARED_DEBUG_LIBRARY)
      IF(NOT WXWIDGETS_SHARED_LIBRARY)
        MESSAGE(STATUS  
          "No shared wxWindows lib found, but WXWIDGETS_USE_SHARED_LIBS=${WXWIDGETS_USE_SHARED_LIBS}.")
      ENDIF(NOT WXWIDGETS_SHARED_LIBRARY)
    ENDIF(NOT WXWIDGETS_SHARED_DEBUG_LIBRARY)

    #########################################################################################
  ELSE(WXWIDGETS_USE_SHARED_LIBS)

    ##jw: DEPRECATED IF(NOT WXWIDGETS_SEPARATE_LIBS_BUILD)
    
    ## static: both wx (debug and release) found?
    ## assume that both builds use the same setup(.h) for simplicity
    IF(WXWIDGETS_STATIC_DEBUG_LIBRARY AND WXWIDGETS_STATIC_LIBRARY)
      ##MESSAGE("DBG wx static: debug and optimized found.")
      FIND_PATH(WXWIDGETS_INCLUDE_DIR_SETUPH  wx/setup.h
        ${WXWIDGETS_ROOT_DIR}/lib/mswd
        ${WXWIDGETS_ROOT_DIR}/lib/msw 
        ${WXWIDGETS_ROOT_DIR}/lib/vc_lib/mswd 
        ${WXWIDGETS_ROOT_DIR}/lib/vc_lib/msw )
      SET(WXWIDGETS_LIBRARIES ${WXWIDGETS_LIBRARIES}
        debug     ${WXWIDGETS_STATIC_DEBUG_LIBRARY}
        optimized ${WXWIDGETS_STATIC_LIBRARY} )
      IF (WXWIDGETS_USE_GL)
        SET(WXWIDGETS_LIBRARIES ${WXWIDGETS_LIBRARIES}
          debug     ${WXWIDGETS_STATIC_DEBUG_LIBRARY_GL}
          optimized ${WXWIDGETS_STATIC_LIBRARY_GL} )
      ENDIF (WXWIDGETS_USE_GL)          
    ENDIF(WXWIDGETS_STATIC_DEBUG_LIBRARY AND WXWIDGETS_STATIC_LIBRARY)
    
    ## static: only debug wx lib found?
    IF(WXWIDGETS_STATIC_DEBUG_LIBRARY)
      IF(NOT WXWIDGETS_STATIC_LIBRARY)
        ##MESSAGE("DBG wx static: debug (but no optimized) found.")
        FIND_PATH(WXWIDGETS_INCLUDE_DIR_SETUPH  wx/setup.h
          ${WXWIDGETS_ROOT_DIR}/lib/mswd 
          ${WXWIDGETS_ROOT_DIR}/lib/vc_lib/mswd  )
        SET(WXWIDGETS_LIBRARIES ${WXWIDGETS_LIBRARIES}
          ${WXWIDGETS_STATIC_DEBUG_LIBRARY} )
        IF (WXWIDGETS_USE_GL)           
          SET(WXWIDGETS_LIBRARIES ${WXWIDGETS_LIBRARIES}
            ${WXWIDGETS_STATIC_DEBUG_LIBRARY_GL} )
        ENDIF (WXWIDGETS_USE_GL)
      ENDIF(NOT WXWIDGETS_STATIC_LIBRARY)
    ENDIF(WXWIDGETS_STATIC_DEBUG_LIBRARY)
    
    ## static: only release wx lib found?
    IF(NOT WXWIDGETS_STATIC_DEBUG_LIBRARY)
      IF(WXWIDGETS_STATIC_LIBRARY)
        ##MESSAGE("DBG wx static: optimized (but no debug) found.")
        FIND_PATH(WXWIDGETS_INCLUDE_DIR_SETUPH  wx/setup.h
          ${WXWIDGETS_ROOT_DIR}/lib/msw 
          ${WXWIDGETS_ROOT_DIR}/lib/vc_lib/msw ) 
        SET(WXWIDGETS_LIBRARIES ${WXWIDGETS_LIBRARIES}
          ${WXWIDGETS_STATIC_LIBRARY} )
        IF (WXWIDGETS_USE_GL)                           
          SET(WXWIDGETS_LIBRARIES ${WXWIDGETS_LIBRARIES}
            ${WXWIDGETS_STATIC_LIBRARY_GL} )
        ENDIF (WXWIDGETS_USE_GL)                
      ENDIF(WXWIDGETS_STATIC_LIBRARY)
    ENDIF(NOT WXWIDGETS_STATIC_DEBUG_LIBRARY)
    
    ## static: none found?
    IF(NOT WXWIDGETS_STATIC_DEBUG_LIBRARY AND NOT WXWIDGETS_SEPARATE_LIBS_BUILD)
      IF(NOT WXWIDGETS_STATIC_LIBRARY)
        MESSAGE(STATUS 
          "No static wxWindows lib found, but WXWIDGETS_USE_SHARED_LIBS=${WXWIDGETS_USE_SHARED_LIBS}.")
      ENDIF(NOT WXWIDGETS_STATIC_LIBRARY)
    ENDIF(NOT WXWIDGETS_STATIC_DEBUG_LIBRARY AND NOT WXWIDGETS_SEPARATE_LIBS_BUILD)      
  ENDIF(WXWIDGETS_USE_SHARED_LIBS)  
  
  
  ## not neccessary in wxWindows 2.4.1 and 2.6.2
  ## but it may fix a previous bug, see
  ## http://lists.wxwindows.org/cgi-bin/ezmlm-cgi?8:mss:37574:200305:mpdioeneabobmgjenoap
  OPTION(WXWIDGETS_SET_DEFINITIONS "Set additional defines for wxWindows" OFF)
  MARK_AS_ADVANCED(WXWIDGETS_SET_DEFINITIONS)
  IF (WXWIDGETS_SET_DEFINITIONS) 
    SET(WXWIDGETS_DEFINITIONS "-DWINVER=0x400")
  ELSE (WXWIDGETS_SET_DEFINITIONS) 
    # clear:
    SET(WXWIDGETS_DEFINITIONS "")
  ENDIF (WXWIDGETS_SET_DEFINITIONS) 
  
  
  
  ## Find the include directories for wxwindows
  ## the first, build specific for wx/setup.h was determined before.
  ## add inc dir for general for "wx/wx.h"
  FIND_PATH(WXWIDGETS_INCLUDE_DIR  wx/wx.h 
    "${WXWIDGETS_ROOT_DIR}/include" )  
  ## append the build specific include dir for wx/setup.h:
  IF (WXWIDGETS_INCLUDE_DIR_SETUPH)
    SET(WXWIDGETS_INCLUDE_DIR ${WXWIDGETS_INCLUDE_DIR} ${WXWIDGETS_INCLUDE_DIR_SETUPH} )
  ENDIF (WXWIDGETS_INCLUDE_DIR_SETUPH)
  
  
  MARK_AS_ADVANCED(
    WXWIDGETS_ROOT_DIR
    WXWIDGETS_INCLUDE_DIR
    WXWIDGETS_INCLUDE_DIR_SETUPH
    WXWIDGETS_STATIC_LIBRARY
    WXWIDGETS_STATIC_LIBRARY_GL
    WXWIDGETS_STATIC_DEBUG_LIBRARY
    WXWIDGETS_STATIC_DEBUG_LIBRARY_GL
    WXWIDGETS_STATIC_LIBRARY_ZLIB
    WXWIDGETS_STATIC_DEBUG_LIBRARY_ZLIB
    WXWIDGETS_STATIC_LIBRARY_REGEX
    WXWIDGETS_STATIC_DEBUG_LIBRARY_REGEX
	WXWIDGETS_STATIC_LIBRARY_EXPAT
	WXWIDGETS_STATIC_DEBUG_LIBRARY_EXPAT
    WXWIDGETS_STATIC_LIBRARY_PNG
    WXWIDGETS_STATIC_DEBUG_LIBRARY_PNG
    WXWIDGETS_STATIC_LIBRARY_JPEG
    WXWIDGETS_STATIC_DEBUG_LIBRARY_JPEG
    WXWIDGETS_STATIC_DEBUG_LIBRARY_TIFF
    WXWIDGETS_STATIC_LIBRARY_TIFF
    WXWIDGETS_SHARED_LIBRARY
    WXWIDGETS_SHARED_DEBUG_LIBRARY
    WXWIDGETS_SHARED_LIBRARY_GL
    WXWIDGETS_SHARED_DEBUG_LIBRARY_GL    
    )
  
  
ELSE(WIN32_STYLE_FIND)

  IF (UNIX_STYLE_FIND) 
    ## ######################################################################
    ## 
    ## UNIX/Linux specific:
    ## 
    ## use backquoted wx-config to query and set flags and libs:
    ## 06/2003 Jan Woetzel
    ## 
    
    OPTION(WXWIDGETS_USE_SHARED_LIBS "Use shared versions (.so) of wxWindows libraries" ON)
    MARK_AS_ADVANCED(WXWIDGETS_USE_SHARED_LIBS)

    # JW removed option and force the develper th SET it. 
    # OPTION(WXWIDGETS_USE_GL "use wxWindows with GL support (use additional
      # --gl-libs for wx-config)?" OFF)
    
    # wx-config should be in your path anyhow, usually no need to set WXWIN or
    # search in ../wx or ../../wx
    FIND_PROGRAM(CMAKE_WXWIDGETS_WXCONFIG_EXECUTABLE wx-config
      $ENV{WXWIN}
      $ENV{WXWIN}/bin
      ../wx/bin
      ../../wx/bin
	  /users/phorton/local/lib)
    
    # check wether wx-config was found:
    IF(CMAKE_WXWIDGETS_WXCONFIG_EXECUTABLE)    

      # use shared/static wx lib?
      # remember: always link shared to use systems GL etc. libs (no static
        # linking, just link *against* static .a libs)
      IF(WXWIDGETS_USE_SHARED_LIBS)
        SET(WX_CONFIG_ARGS_LIBS "--libs")
		#SET(WX_CONFIG_ARGS_LIBS "--libs all")
      ELSE(WXWIDGETS_USE_SHARED_LIBS)
        SET(WX_CONFIG_ARGS_LIBS "--static --libs all")
      ENDIF(WXWIDGETS_USE_SHARED_LIBS)
      
      # do we need additionial wx GL stuff like GLCanvas ?
      IF(WXWIDGETS_USE_GL)
        SET(WX_CONFIG_ARGS_LIBS "${WX_CONFIG_ARGS_LIBS} --gl-libs" )
      ENDIF(WXWIDGETS_USE_GL)
      ##MESSAGE("DBG: WX_CONFIG_ARGS_LIBS=${WX_CONFIG_ARGS_LIBS}===")

      
      SET(WX_CONFIG_CXXFLAGS_ARGS "--cxxflags")
      
      IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
        SET(WX_CONFIG_ARGS_LIBS "${WX_CONFIG_ARGS_LIBS} --debug=yes" )
        SET(WX_CONFIG_CXXFLAGS_ARGS "${WX_CONFIG_CXXFLAGS_ARGS} --debug=yes")
      ENDIF(CMAKE_BUILD_TYPE STREQUAL "Debug")

      IF(CMAKE_BUILD_TYPE STREQUAL "Release")
        SET(WX_CONFIG_ARGS_LIBS "${WX_CONFIG_ARGS_LIBS} --debug=no" )
        SET(WX_CONFIG_CXXFLAGS_ARGS "${WX_CONFIG_CXXFLAGS_ARGS} --debug=no")
      ENDIF(CMAKE_BUILD_TYPE STREQUAL "Release")

      	  MESSAGE("DBG: WX_CONFIG_ARGS_LIBS=${WX_CONFIG_ARGS_LIBS}")
		  
		  
		  
      	  
      # set CXXFLAGS to be fed into CMAKE_CXX_FLAGS by the user:
      SET(CMAKE_WXWIDGETS_CXX_FLAGS "`${CMAKE_WXWIDGETS_WXCONFIG_EXECUTABLE} --cxxflags|sed -e s/-I/-isystem/g`")
      ##MESSAGE("DBG: for compilation:
        ##CMAKE_WXWIDGETS_CXX_FLAGS=${CMAKE_WXWIDGETS_CXX_FLAGS}===")

      # keep the back-quoted string for clarity
      SET(WXWIDGETS_LIBRARIES "`${CMAKE_WXWIDGETS_WXCONFIG_EXECUTABLE} ${WX_CONFIG_ARGS_LIBS}`")
      MESSAGE("DBG2: for linking WXWIDGETS_LIBRARIES=${WXWIDGETS_LIBRARIES}===")
      
      # evaluate wx-config output to separate linker flags and linkdirs for
      # rpath:
      EXEC_PROGRAM(${CMAKE_WXWIDGETS_WXCONFIG_EXECUTABLE}
        ARGS ${WX_CONFIG_ARGS_LIBS}
        OUTPUT_VARIABLE WX_CONFIG_LIBS )
	  
	  
      ## extract linkdirs (-L) for rpath
      ## use regular expression to match wildcard equivalent "-L*<endchar>"
      ## with <endchar> is a space or a semicolon
      STRING(REGEX MATCHALL "[-][L]([^ ;])+" WXWIDGETS_LINK_DIRECTORIES_WITH_PREFIX "${WX_CONFIG_LIBS}" )
	  set(WXWIDGETS_LINK_DIRECTORIES_WITH_PREFIX "-L/users/phorton/local/lib ${WXWIDGETS_LINK_DIRECTORIES_WITH_PREFIX}")
       MESSAGE("DBG WXWIDGETS_LINK_DIRECTORIES_WITH_PREFIX= ${WXWIDGETS_LINK_DIRECTORIES_WITH_PREFIX}")
      
      ## remove prefix -L because we need the pure directory for LINK_DIRECTORIES
      ## replace -L by ; because the separator seems to be lost otherwise (bug or
        ## feature?)
      IF(WXWIDGETS_LINK_DIRECTORIES_WITH_PREFIX)
        STRING(REGEX REPLACE "[-][L]" ";" WXWIDGETS_LINK_DIRECTORIES ${WXWIDGETS_LINK_DIRECTORIES_WITH_PREFIX} )
         MESSAGE("DBG WXWIDGETS_LINK_DIRECTORIES= ${WXWIDGETS_LINK_DIRECTORIES}")
      ENDIF(WXWIDGETS_LINK_DIRECTORIES_WITH_PREFIX)
      
      
      ## replace space separated string by semicolon separated vector to make it
      ## work with LINK_DIRECTORIES
      SEPARATE_ARGUMENTS(WXWIDGETS_LINK_DIRECTORIES)
      
      
      	EXEC_PROGRAM(${CMAKE_WXWIDGETS_WXCONFIG_EXECUTABLE}
        ARGS ${WX_CONFIG_CXXFLAGS_ARGS}
        OUTPUT_VARIABLE CMAKE_WXWIDGETS_CXX_FLAGS)
		set(CMAKE_WXWIDGETS_CXX_FLAGS "-I/users/phorton/local/lib/wx/include/gtk2-unicode-2.9 -I/users/phorton/local/include/wx -I/users/phorton/local/include ${CMAKE_WXWIDGETS_CXX_FLAGS}")
        MESSAGE("DBG CMAKE_WXWIDGETS_CXX_FLAGS= ${CMAKE_WXWIDGETS_CXX_FLAGS}")
        
      	EXEC_PROGRAM(${CMAKE_WXWIDGETS_WXCONFIG_EXECUTABLE}
        ARGS ${WX_CONFIG_ARGS_LIBS}
        OUTPUT_VARIABLE WXWIDGETS_LIBRARIES)
		
		set(WXWIDGETS_LIBRARIES "-L/users/phorton/local/lib ${WXWIDGETS_LIBRARIES}")

		SET(WX_USE_XML CACHE BOOL "Use Expat library for XML ?" 1)
		IF(WX_USE_XML)
			SET(WXWIDGETS_LIBRARIES "${WXWIDGETS_LIBRARIES} -lexpat")
		ENDIF(WX_USE_XML)

        MESSAGE("DBG WXWIDGETS_LIBRARIES= ${WXWIDGETS_LIBRARIES}")

      
        IF(WX_CONFIG_LIBS)
    		LINK_LIBRARIES(${WX_CONFIG_LIBS})
  		ENDIF(WX_CONFIG_LIBS)
  
  		IF (WX_CONFIG_CXXFLAGS)
    		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WX_CONFIG_CXXFLAGS}")
  		ENDIF(WX_CONFIG_CXXFLAGS)
      
      
      
      
      MARK_AS_ADVANCED(
        CMAKE_WXWIDGETS_CXX_FLAGS
        WXWIDGETS_INCLUDE_DIR
        WXWIDGETS_LIBRARIES
        CMAKE_WXWIDGETS_WXCONFIG_EXECUTABLE
        )
      
      
      ## we really need wx-config...
    ELSE(CMAKE_WXWIDGETS_WXCONFIG_EXECUTABLE)    
      MESSAGE(STATUS "Cannot find wx-config anywhere on the system. Please put the file into your path or specify it in CMAKE_WXWIDGETS_WXCONFIG_EXECUTABLE.")
      MARK_AS_ADVANCED(CMAKE_WXWIDGETS_WXCONFIG_EXECUTABLE)
    ENDIF(CMAKE_WXWIDGETS_WXCONFIG_EXECUTABLE)

    
    
  ELSE(UNIX_STYLE_FIND)
    MESSAGE(STATUS "FindwxWindows.cmake:  Platform unknown/unsupported by FindwxWindows.cmake. It's neither WIN32 nor UNIX")
  ENDIF(UNIX_STYLE_FIND)
ENDIF(WIN32_STYLE_FIND)


IF(WXWIDGETS_LIBRARIES)
  IF(WXWIDGETS_INCLUDE_DIR OR CMAKE_WXWIDGETS_CXX_FLAGS)
    ## found all we need.
    SET(WXWIDGETS_FOUND 1)
    
    ## set deprecated variables for backward compatibility: 
    SET(CMAKE_WX_CAN_COMPILE   ${WXWIDGETS_FOUND})
    SET(WXWIDGETS_LIBRARY     ${WXWIDGETS_LIBRARIES})
    SET(WXWIDGETS_INCLUDE_PATH ${WXWIDGETS_INCLUDE_DIR})
    SET(WXWIDGETS_LINK_DIRECTORIES ${WXWIDGETS_LINK_DIRECTORIES})
    SET(CMAKE_WX_CXX_FLAGS     ${CMAKE_WXWIDGETS_CXX_FLAGS})
    
  ELSE(WXWIDGETS_INCLUDE_DIR OR CMAKE_WXWIDGETS_CXX_FLAGS)
    MESSAGE("WXWIDGETS_INCLUDE_DIR or CMAKE_WXWIDGETS_CXX_FLAGS missing")
  ENDIF(WXWIDGETS_INCLUDE_DIR OR CMAKE_WXWIDGETS_CXX_FLAGS)
ELSE(WXWIDGETS_LIBRARIES)
  MESSAGE("WXWIDGETS_LIBRARIES missing")
ENDIF(WXWIDGETS_LIBRARIES)




IF(WXWIDGETS_FOUND)
  
  #MESSAGE("DBG Use_wxWindows.cmake:  WXWIDGETS_INCLUDE_DIR=${WXWIDGETS_INCLUDE_DIR} WXWIDGETS_LINK_DIRECTORIES=${WXWIDGETS_LINK_DIRECTORIES}     WXWIDGETS_LIBRARIES=${WXWIDGETS_LIBRARIES}  CMAKE_WXWIDGETS_CXX_FLAGS=${CMAKE_WXWIDGETS_CXX_FLAGS} WXWIDGETS_DEFINITIONS=${WXWIDGETS_DEFINITIONS}")
  
  IF(WXWIDGETS_INCLUDE_DIR)
    INCLUDE_DIRECTORIES(${WXWIDGETS_INCLUDE_DIR})
  ENDIF(WXWIDGETS_INCLUDE_DIR)
 
  IF(WXWIDGETS_LINK_DIRECTORIES)
    LINK_DIRECTORIES(${WXWIDGETS_LINK_DIRECTORIES})
  ENDIF(WXWIDGETS_LINK_DIRECTORIES)
  
  IF(WXWIDGETS_LIBRARIES)
    LINK_LIBRARIES(${WXWIDGETS_LIBRARIES})
  ENDIF(WXWIDGETS_LIBRARIES)
  
  IF (CMAKE_WXWIDGETS_CXX_FLAGS)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_WXWIDGETS_CXX_FLAGS}")
  ENDIF(CMAKE_WXWIDGETS_CXX_FLAGS)
  
  IF(WXWIDGETS_DEFINITIONS)
    ADD_DEFINITIONS(${WXWIDGETS_DEFINITIONS})
  ENDIF(WXWIDGETS_DEFINITIONS)

ELSE(WXWIDGETS_FOUND)
  MESSAGE(SEND_ERROR "wxWindows not found by Use_wxWindows.cmake")
ENDIF(WXWIDGETS_FOUND)

