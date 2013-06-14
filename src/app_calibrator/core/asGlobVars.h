#ifndef ASGLOBVARS_H_INCLUDED
#define ASGLOBVARS_H_INCLUDED

#include "asIncludes.h"

using namespace std;

extern bool g_SilentMode;
extern bool g_VerboseMode;
extern bool g_Responsive;
extern bool g_UnitTesting;
extern bool g_GuiMode;
extern bool g_AppViewer;
extern bool g_AppForecaster;
extern bool g_DistributionVersion;
extern bool g_Local;
extern int g_RunNb;
#if wxUSE_GUI
extern wxColour g_LinuxBgColour;
#endif

// Constants
const extern double g_Cst_Euler;
const extern double g_Cst_Pi;

// Useful variables
const extern wxString DS;

#endif // ASGLOBVARS_H_INCLUDED
