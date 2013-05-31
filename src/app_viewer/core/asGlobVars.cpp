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
 
#include "asGlobVars.h"

bool g_SilentMode = false;
bool g_VerboseMode = true;
bool g_Responsive = true;
bool g_UnitTesting = false;
bool g_GuiMode = true;
bool g_AppViewer = false;
bool g_AppForecaster = true;
wxColour g_LinuxBgColour = wxColour(242,241,240);
wxString g_CmdFilename = wxEmptyString;

// Constants
const double g_Cst_Euler = 0.57721566490153286060651209008240243104215933593992; // http://fr.wikipedia.org/wiki/Constante_d%27Euler-Mascheroni
const double g_Cst_Pi = 3.14159265358979323846264338327950288419716939937510; // http://fr.wikipedia.org/wiki/Pi

// Useful variables
const wxString DS = wxFileName::GetPathSeparator();
