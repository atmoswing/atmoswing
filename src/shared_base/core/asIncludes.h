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

#ifndef AS_INC_H
#define AS_INC_H

// =====================================================================================
// asIncludes.h — implementation-side omnibus header (use from .cpp files only).
//
// SCOPE
//   Pulls in the project's full surface (wx/wx.h, Eigen, asUtils, asTime, asLog, asConfig,
//   asThreadsManager, asEvents, GUI dialogs, ...) for .cpp translation units. The cost is
//   paid once per .cpp; it does NOT cascade through other headers because no project .h
//   should include this file.
//
//   HEADERS must NOT include this file. Use "asHeadersBase.h" instead (small, ~6 wx
//   headers + project type aliases + enums) and add any extra direct includes (e.g.
//   <wx/log.h>, <wx/fileconf.h>, "asTime.h") only when needed. asHeadersBase.h is
//   pulled in below so the shared baseline lives in a single place.
// =====================================================================================

//---------------------------------
// Disable some MSVC warnings
//---------------------------------

#ifdef _MSC_VER
#pragma warning(disable : 4125)  // C4125: decimal digit terminates octal escape sequence
#pragma warning(disable : 4100)  // C4100: unreferenced formal parameter
#pragma warning(disable : 4515)  // C4515: namespace uses itself
#endif

//---------------------------------
// Shared baseline (wxObject/wxString, project type aliases, project enums, ...)
// Pulled in from asHeadersBase.h so the .h and .cpp sides share one source of truth.
// asTypeDefs.h must come before the other project headers, and asHeadersBase.h
// already ensures that ordering.
//---------------------------------

#include "asHeadersBase.h"

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

//---------------------------------
// wxWidgets library - frequently used classes
//---------------------------------

#ifndef WX_PRECOMP

#include "wx/arrstr.h"
#include "wx/log.h"
#include "wx/string.h"
#include "wx/utils.h"

#endif
#include "wx/fileconf.h"

#if defined(__WIN32__)
#include "wx/msw/regconf.h"  // wxRegConfig class
#endif

//---------------------------------
// Eigen library
//---------------------------------

#define EIGEN_DEFAULT_TO_ROW_MAJOR
#ifndef EIGEN_NO_DEBUG
#define EIGEN_NO_DEBUG
#endif

// Modules and Header files: http://eigen.tuxfamily.org/dox-3.0/QuickRefPage.html#QuickRef_Headers
#include <Eigen/Core>
#include <Eigen/StdVector>

//---------------------------------
// Standard library
// (std::vector and `using std::runtime_error` come from asHeadersBase.h)
//---------------------------------

#include <algorithm>
#include <cmath>
#include <exception>

//---------------------------------
// Automatic leak detection with Microsoft VisualC++
// http://msdn.microsoft.com/en-us/library/e5ewb1h3(v=VS.90).aspx
// http://wiki.wxwidgets.org/Avoiding_Memory_Leaks
//---------------------------------

#ifdef _DEBUG

#include <stdlib.h>
#include <wx/debug.h>  // wxASSERT

#ifdef __WXMSW__
#include <crtdbg.h>
#include <wx/msw/msvcrt.h>  // redefines the new() operator

#if !defined(_INC_CRTDBG) || !defined(_CRTDBG_MAP_ALLOC)
#error Debug CRT functions have not been included!
#endif
#endif

#endif

//---------------------------------
// Some AtmoSwing stuff - frequently used classes
// (asTypeDefs.h and asGlobEnums.h are pulled in by asHeadersBase.h above.)
//---------------------------------

#include "asConfig.h"
#include "asGlobVars.h"
#include "asLog.h"
#include "asThreadsManager.h"
#include "asThreadsManagerGlobalFunctions.h"
#include "asTime.h"
#include "asUtils.h"
#include "asVersion.h"

#if USE_GUI
#include "asDialogFilePicker.h"
#include "asDialogFileSaver.h"
#include "asDialogProgressBar.h"
#endif

// g_cmdFileName, g_local, g_runNb now live in asGlobVars.h (already pulled in above).
// Only the Optimizer still has app-specific globals (g_resumePreviousRun).
#if defined(APP_OPTIMIZER) || defined(UNIT_TESTING)
#include "asGlobVarsOptimizer.h"
#endif

// Custom wxEvent declarations live in their own header.
#include "asEvents.h"

#endif  // AS_INC_H
