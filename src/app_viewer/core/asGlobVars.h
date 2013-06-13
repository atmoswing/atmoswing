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
 
#ifndef ASGLOBVARS_H_INCLUDED
#define ASGLOBVARS_H_INCLUDED

#include <wx/colour.h>
#include "asIncludes.h"

extern bool g_SilentMode;
extern bool g_VerboseMode;
extern bool g_Responsive;
extern bool g_UnitTesting;
extern bool g_GuiMode;
extern bool g_AppViewer;
extern bool g_AppForecaster;
extern wxColour g_LinuxBgColour;
extern wxString g_CmdFilename;

// Constants
const extern double g_Cst_Euler;
const extern double g_Cst_Pi;

// Useful variables
const extern wxString DS;

#endif // ASGLOBVARS_H_INCLUDED
