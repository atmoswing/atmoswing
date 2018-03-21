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
 */

#include "asVersion.h"

const wxString g_version = wxString::Format("%d.%d.%d", ATMOSWING_MAJOR_VERSION, ATMOSWING_MINOR_VERSION,
                                            ATMOSWING_PATCH_VERSION);

wxString asVersion::GetFullString()
{
    //ctor
    wxString versionNb = "Version " + g_version;
#if defined(__WXMSW__)
    versionNb << " -Windows";
#elif defined(__WXMAC__)
    versionNb << " -Mac";
#elif defined(__UNIX__)
    versionNb << " -Linux";
#endif

    return versionNb;
}
