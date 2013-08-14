/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
/** @file asIncludes.h
 *
 *  @note include file for the Atmoswing project
 *
 *  @author Pascal Horton <pascal.horton@unil.ch>
 *
 *  @note Inspired from panoinc_WX.h (hugin project) from Alexandre Jenny <alexandre.jenny@le-geo.com>
 *
 */


#ifndef ATMOSWINGINC_H_INCLUDED
#define ATMOSWINGINC_H_INCLUDED

//---------------------------------
// Automatic leak detection with Microsoft VisualC++
// http://msdn.microsoft.com/en-us/library/e5ewb1h3(v=VS.90).aspx
// http://wiki.wxwidgets.org/Avoiding_Memory_Leaks
//---------------------------------

#ifdef _DEBUG
   #define _CRTDBG_MAP_ALLOC
   #include <stdlib.h>
   #include <crtdbg.h>
   #include <wx/debug.h>   // wxASSERT

   #if !defined(_INC_CRTDBG) || !defined(_CRTDBG_MAP_ALLOC)
        #error Debug CRT functions have not been included!
    #endif
#endif


//---------------------------------
// Standard wxWidgets headers
//---------------------------------

// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"
#ifdef __BORLANDC__
    #pragma hdrstop
#endif

// For all others, include the necessary headers
#ifndef WX_PRECOMP
    #include "wx/wx.h"
#endif

// Memory checking
//#define wxUSE_DEBUG_NEW_ALWAYS 1


//---------------------------------
// wxWidgets library - frequently used classes
//---------------------------------

#ifndef WX_PRECOMP
    #include "wx/log.h"
    #include "wx/string.h"
    #include "wx/arrstr.h"
    #include "wx/utils.h"
    #include "wx/fileconf.h"
#endif

#if defined (__WIN32__)
  #include "wx/msw/regconf.h"   // wxRegConfig class
#endif


//---------------------------------
// Eigen library
//---------------------------------

// SHOULD MAYBE REMOVE THAT !!!
//#if defined (__WIN32__)
    #define EIGEN_NO_STATIC_ASSERT
//#endif

#define EIGEN_DEFAULT_TO_ROW_MAJOR
#ifndef EIGEN_NO_DEBUG
    #define EIGEN_NO_DEBUG
#endif

// Modules and Header files: http://eigen.tuxfamily.org/dox-3.0/QuickRefPage.html#QuickRef_Headers
#include <Eigen/StdVector>
#include <Eigen/Dense>


//---------------------------------
// Standard library
//---------------------------------

#include <algorithm>
#include <vector>
#include <exception>
//#include <math.h>
//#include <cmath>
using namespace std;


//---------------------------------
// Some Atmoswing stuff - frequently used classes
//---------------------------------

#include "asVersion.h"
#include "asConfig.h"
#include "asLog.h"
#include "asLogGlobalFunctions.h"
#include "asGlobEnums.h"
#include "asGlobVars.h"
#include "asTypeDefs.h"
#include "asException.h"
#include "asTools.h"
#include "asTime.h"
#include "asThreadsManagerGlobalFunctions.h"
#include "asThreadsManager.h"
#if wxUSE_GUI
    #include "asDialogFilePicker.h"
    #include "asDialogFileSaver.h"
    #include "asDialogProgressBar.h"
#endif


//---------------------------------
// Others
//---------------------------------

// Remove stupid #defines from the evil windows.h -> from Hugin project
#ifdef __WXMSW__
//    #undef DIFFERENCE
//    #undef FindWindow
//    #undef min
//    #undef max
#endif


#endif // ATMOSWINGINC_H_INCLUDED


//---------------------------------
// Event definition.
//---------------------------------

wxDECLARE_EVENT(asEVT_STATUS_STARTING, wxCommandEvent);
wxDECLARE_EVENT(asEVT_STATUS_RUNNING, wxCommandEvent);
wxDECLARE_EVENT(asEVT_STATUS_FAILED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_STATUS_SUCCESS, wxCommandEvent);
wxDECLARE_EVENT(asEVT_STATUS_DOWNLOADING, wxCommandEvent);
wxDECLARE_EVENT(asEVT_STATUS_DOWNLOADED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_STATUS_LOADING, wxCommandEvent);
wxDECLARE_EVENT(asEVT_STATUS_LOADED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_STATUS_SAVING, wxCommandEvent);
wxDECLARE_EVENT(asEVT_STATUS_SAVED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_STATUS_PROCESSING, wxCommandEvent);
wxDECLARE_EVENT(asEVT_STATUS_PROCESSED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_FORECAST_CLEAR, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_FORECAST_NEW_ADDED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_FORECAST_RATIO_SELECTION_CHANGED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_FORECAST_MODEL_SELECTION_CHANGED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_FORECAST_PERCENTILE_SELECTION_CHANGED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_STATION_SELECTION_CHANGED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_ANALOG_DATE_SELECTION_CHANGED, wxCommandEvent);

