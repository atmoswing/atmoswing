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

#include "asConfig.h"

#include "wx/stdpaths.h"        // wxStandardPaths returns the standard locations in the file system


asConfig::asConfig()
{
    //ctor
}

asConfig::~asConfig()
{
    //dtor
}

wxString asConfig::GetLogDir()
{
    ThreadsManager().CritSectionConfig().Enter();
    wxString TempDir = wxStandardPaths::Get().GetTempDir();
    ThreadsManager().CritSectionConfig().Leave();
    TempDir.Append(DS);
    return TempDir;
}

wxString asConfig::GetTempDir()
{
    ThreadsManager().CritSectionConfig().Enter();
    wxString TempDir = wxStandardPaths::Get().GetTempDir();
    ThreadsManager().CritSectionConfig().Leave();
    TempDir.Append(DS);
    return TempDir;
}

wxString asConfig::CreateTempFileName(const wxString &prefix)
{
    wxString path = asConfig::GetTempDir() + prefix;
    wxString pathTry;

    static const size_t numTries = 1000;
    for (size_t n = 0; n < numTries; n++) {
        // 3 hex digits is enough for numTries == 1000 < 4096
        pathTry = path + wxString::Format(wxT("%.03x"), (unsigned int) n);
        if (!wxFileName::FileExists(pathTry) && !wxFileName::DirExists(pathTry)) {
            break;
        }

        pathTry.clear();
    }

    return pathTry;
}

wxString asConfig::GetDataDir()
{
    ThreadsManager().CritSectionConfig().Enter();
    wxString DirData = wxStandardPaths::Get().GetDataDir();
    ThreadsManager().CritSectionConfig().Leave();
    DirData.Append(DS);
    return DirData;
}

wxString asConfig::GetSoftDir()
{
    ThreadsManager().CritSectionConfig().Enter();
    wxString appPath = wxStandardPaths::Get().GetExecutablePath();
    ThreadsManager().CritSectionConfig().Leave();
    wxFileName fileName(appPath);
    wxString appDir = fileName.GetPath();
    appDir.Append(DS);
    return appDir;
}

wxString asConfig::GetUserDataDir()
{
    ThreadsManager().CritSectionConfig().Enter();
    wxStandardPathsBase &stdPth = wxStandardPaths::Get();
    stdPth.UseAppInfo(0);
    wxString userDataDir = stdPth.GetUserDataDir();
    ThreadsManager().CritSectionConfig().Leave();

#if defined(__WXMSW__)
    userDataDir.Append(DS+"AtmoSwing");
#elif defined(__WXMAC__)
    userDataDir.Append(DS+"atmoswing");
#elif defined(__UNIX__)
    userDataDir.Append("atmoswing");
#endif

    stdPth.UseAppInfo(1);
    userDataDir.Append(DS);
    return userDataDir;
}

wxString asConfig::GetDocumentsDir()
{
    ThreadsManager().CritSectionConfig().Enter();
    wxString DirDocs = wxStandardPaths::Get().GetDocumentsDir();
    ThreadsManager().CritSectionConfig().Leave();
    DirDocs.Append(DS);
    return DirDocs;
}

wxString asConfig::GetDefaultUserWorkingDir()
{
    wxString DirData = GetUserDataDir() + DS + "Data" + DS;
    return DirData;
}

wxString asConfig::GetDefaultUserConfigDir()
{
    wxString DirConfig = GetUserDataDir() + DS + "Config" + DS;
    return DirConfig;
}
