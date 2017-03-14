/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * When distributing Covered Code, include this CDDL Header Notice in
 * each file and include the License file (licence.txt). If applicable,
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 *
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#ifndef ATMOSWINGINC_H_INCLUDED
#define ATMOSWINGINC_H_INCLUDED


//---------------------------------
// Disable some MSVC warnings
//---------------------------------

#ifdef _MSC_VER
#   pragma warning( disable : 4125 ) // C4125: decimal digit terminates octal escape sequence
#   pragma warning( disable : 4100 ) // C4100: unreferenced formal parameter
#   pragma warning( disable : 4515 ) // C4515: namespace uses itself
#endif


//---------------------------------
// Standard wxWidgets headers
//---------------------------------

// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
#   pragma hdrstop
#endif

// For all others, include the necessary headers
#ifndef WX_PRECOMP
#   include "wx/wx.h"
#endif


//---------------------------------
// wxWidgets library - frequently used classes
//---------------------------------

#ifndef WX_PRECOMP
#   include "wx/log.h"
#include "wx/string.h"
#include "wx/arrstr.h"
#include "wx/utils.h"
#include "wx/fileconf.h"
#endif

#if defined (__WIN32__)
#   include "wx/msw/regconf.h"   // wxRegConfig class
#endif


//---------------------------------
// Eigen library
//---------------------------------

#define EIGEN_DEFAULT_TO_ROW_MAJOR
#ifndef EIGEN_NO_DEBUG
#   define EIGEN_NO_DEBUG
#endif

// Modules and Header files: http://eigen.tuxfamily.org/dox-3.0/QuickRefPage.html#QuickRef_Headers
#include <Eigen/StdVector>
#include <Eigen/Core>


//---------------------------------
// Standard library
//---------------------------------

#include <algorithm>
#include <vector>
#include <exception>


//---------------------------------
// Automatic leak detection with Microsoft VisualC++
// http://msdn.microsoft.com/en-us/library/e5ewb1h3(v=VS.90).aspx
// http://wiki.wxwidgets.org/Avoiding_Memory_Leaks
//---------------------------------

#ifdef _DEBUG

#   include <stdlib.h>
#   include <wx/debug.h> // wxASSERT

#   ifdef __WXMSW__
#       include <crtdbg.h>
#       include <wx/msw/msvcrt.h> // redefines the new() operator

#       if !defined(_INC_CRTDBG) || !defined(_CRTDBG_MAP_ALLOC)
#           error Debug CRT functions have not been included!
#       endif
#   endif

#   ifdef USE_VLD
#       include <vld.h> // Visual Leak Detector (https://vld.codeplex.com/)
#   endif

#endif


//---------------------------------
// Some AtmoSwing stuff - frequently used classes
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
#   include "asDialogFilePicker.h"
#   include "asDialogFileSaver.h"
#   include "asDialogProgressBar.h"
#endif

#ifdef APP_FORECASTER
#   include "asGlobVarsForecaster.h"
#endif
#ifdef APP_VIEWER
#   include "asGlobVarsViewer.h"
#endif
#ifdef APP_OPTIMIZER
#   include "asGlobVarsOptimizer.h"
#endif
#ifdef UNIT_TESTING
#   include "asGlobVarsOptimizer.h"
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
wxDECLARE_EVENT(asEVT_ACTION_FORECAST_SELECTION_CHANGED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_FORECAST_SELECT_FIRST, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_FORECAST_QUANTILE_SELECTION_CHANGED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_STATION_SELECTION_CHANGED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_ANALOG_DATE_SELECTION_CHANGED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_LEAD_TIME_SELECTION_CHANGED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_OPEN_WORKSPACE, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_OPEN_BATCHFORECASTS, wxCommandEvent);
