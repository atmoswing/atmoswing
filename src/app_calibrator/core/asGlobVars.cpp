#include "asGlobVars.h"

bool g_SilentMode = false;
bool g_VerboseMode = false;
bool g_Responsive = false;
bool g_UnitTesting = false;
bool g_GuiMode = true;
bool g_AppViewer = false;
bool g_AppForecaster = false;
bool g_Local = false;
int g_RunNb = 0;
#ifdef DISTRIBUTION_VERSION
    bool g_DistributionVersion = true;
#else
    bool g_DistributionVersion = false;
#endif
#if wxUSE_GUI
wxColour g_LinuxBgColour = wxColour(242,241,240);
#endif

// Constants
const double g_Cst_Euler = 0.57721566490153286060651209008240243104215933593992; // http://fr.wikipedia.org/wiki/Constante_d%27Euler-Mascheroni
const double g_Cst_Pi = 3.14159265358979323846264338327950288419716939937510; // http://fr.wikipedia.org/wiki/Pi

// Useful variables
const wxString DS = wxFileName::GetPathSeparator();
