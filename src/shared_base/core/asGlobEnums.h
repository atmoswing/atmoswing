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

#ifndef ASGLOBENUMS_H_INCLUDED
#define ASGLOBENUMS_H_INCLUDED

#include "wx/wxprec.h"

#ifndef WX_PRECOMP

#include "wx/wx.h"

#endif

//---------------------------------
// Symbolic constants
//---------------------------------

enum
{
    asSHOW_WARNINGS, asHIDE_WARNINGS
};

enum
{
    asOUT_OF_RANGE = -2,
    asNOT_FOUND = -1,
    asFAILED = 0,
    asEMPTY = -1,
    asNONE = -1,
    asCANCELLED = -1,
    asNOT_VALID = -9999,
    asSUCCESS = 1
};

enum
{
    asEDIT_FORBIDDEN, asEDIT_ALLOWED
};

enum
{
    asFLAT_FORBIDDEN, asFLAT_ALLOWED
};

enum
{
    asUSE_NORMAL_METHOD, asUSE_ALTERNATE_METHOD
};

enum
{
    asUTM, asLOCAL
};

enum
{
    asSAMPLE, asENTIRE_POPULATION
};

// Processor methods
enum
{
    asMULTITHREADS = 0, asSTANDARD = 1, asCUDA = 2,
};

// Optimization stages
enum
{
    asINITIALIZATION,
    asREASSESSMENT,
    asCHECK_CONVERGENCE
};

// Windows ID
#if wxUSE_GUI
enum
{
    asWINDOW_MAIN = 101,
    asWINDOW_PREFERENCES,
    asWINDOW_PREDICTANDDB,
    asWINDOW_PLOTS_TIMESERIES,
    asWINDOW_PLOTS_DISTRIBUTIONS,
    asWINDOW_GRID_ANALOGS,
    asWINDOW_PREDICTORS
};

// Menus & Controls ID
enum
{
    asID_PREFERENCES = wxID_HIGHEST + 1,
    asID_OPEN,
    asID_RUN,
    asID_RUN_PREVIOUS,
    asID_CANCEL,
    asID_DB_OPTIONS,
    asID_DB_CREATE,
    asID_PRINT,
    asID_SELECT,
    asID_ZOOM_IN,
    asID_ZOOM_OUT,
    asID_ZOOM_FIT,
    asID_PAN,
    asID_CROSS_MARKER,
    asID_FRAME_VIEWER,
    asID_FRAME_FORECASTER,
    asID_FRAME_DOTS,
    asID_FRAME_PLOTS,
    asID_FRAME_GRID,
};
#endif


//---------------------------------
// Enumerations
//---------------------------------

enum Order
{
    Asc,    // Ascendant
    Desc,   // Descendant
};

enum TimeFormat
{
    ISOdate,
    ISOdateTime,
    YYYYMMDD,
    YYYY_MM_DD,
    YYYY_MM_DD_hh,
    YYYYMMDD_hhmm,
    YYYY_MM_DD_hh_mm,
    YYYY_MM_DD_hh_mm_ss,
    DD_MM_YYYY,
    DD_MM_YYYY_hh_mm,
    DD_MM_YYYY_hh_mm_ss,
    hh_mm,
    guess
};

#endif // ASGLOBENUMS_H_INCLUDED