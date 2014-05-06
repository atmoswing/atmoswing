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
 * The Original Software is AtmoSwing. The Initial Developer of the
 * Original Software is Pascal Horton of the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 * Portions Copyright 2013 Pascal Horton, Terr@num.
 */

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
extern bool g_ResumePreviousRun;
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
