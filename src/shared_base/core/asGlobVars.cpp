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

#include "asGlobVars.h"

bool g_silentMode = false;
bool g_verboseMode = true;
bool g_responsive = true;
bool g_unitTesting = false;
bool g_guiMode = true;

// Constants
const double g_cst_Euler = 0.5772156649;  // Euler-Mascheroni
const double g_cst_Pi = 3.1415926535;

// Useful variables
#if defined(__WIN32__)
const wxString DS = "\\";
#else
const wxString DS = "/";
#endif
double g_ppiScaleDc = 1.0;