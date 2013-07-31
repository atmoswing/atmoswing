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
 
#include "asVersion.h"

const wxString g_Version = wxString::Format("%d.%d.%d", ATMOSWING_MAJOR_VERSION, ATMOSWING_MINOR_VERSION, ATMOSWING_PATCH_VERSION);

asVersion::asVersion()
{
    //ctor
}

asVersion::~asVersion()
{
    //dtor
}

wxString asVersion::GetFullString()
{
    //ctor
    wxString versionNb = "Version " + g_Version;
    #if defined(__WXMSW__)
        versionNb << " -Windows";
    #elif defined(__WXMAC__)
        versionNb << " -Mac";
    #elif defined(__UNIX__)
        versionNb << " -Linux";
    #endif

    #if wxUSE_UNICODE
        versionNb << " -Unicode";
    #else
        versionNb << " -ANSI";
    #endif

    return versionNb;
}
