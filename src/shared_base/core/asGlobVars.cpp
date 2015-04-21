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
 */
 
#include "asGlobVars.h"

bool g_silentMode = false;
bool g_verboseMode = true;
bool g_responsive = true;
bool g_unitTesting = false;
bool g_guiMode = true;

// Constants
const double g_cst_Euler = 0.57721566490153286060651209008240243104215933593992; // http://fr.wikipedia.org/wiki/Constante_d%27Euler-Mascheroni
const double g_cst_Pi = 3.14159265358979323846264338327950288419716939937510; // http://fr.wikipedia.org/wiki/Pi

// Useful variables
const wxString DS = wxFileName::GetPathSeparator();
double g_ppiScaleDc = 1.0;