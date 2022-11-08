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

#include <wx/dir.h>
#include <wx/stdpaths.h>  // wxStandardPaths returns the standard locations in the file system

wxString asConfig::GetLogDir() {
#ifdef ON_DOCKER
    wxString tempDir = "/app/config/";
#else
    ThreadsManager().CritSectionConfig().Enter();
    wxString tempDir = wxStandardPaths::Get().GetTempDir();
    ThreadsManager().CritSectionConfig().Leave();
    tempDir.Append(DS);
#endif
    return tempDir;
}

wxString asConfig::GetTempDir() {
    ThreadsManager().CritSectionConfig().Enter();
    wxString tempDir = wxStandardPaths::Get().GetTempDir();
    ThreadsManager().CritSectionConfig().Leave();
    tempDir.Append(DS);
    return tempDir;
}

wxString asConfig::CreateTempFileName(const wxString& prefix) {
    wxString path = asConfig::GetTempDir() + prefix;

    static const size_t numTries = 1000;
    for (size_t n = 0; n < numTries; n++) {
        wxString pathFile = path + asStrF(wxT("%.03x"), (unsigned int)n);
        if (!wxFileName::FileExists(pathFile) && !wxFileName::DirExists(pathFile)) {
            return pathFile;
        }
    }

    return wxEmptyString;
}

wxString asConfig::CreateTempDir(const wxString& prefix) {
    wxString path = asConfig::GetTempDir() + prefix;

    static const size_t numTries = 1000;
    for (size_t n = 0; n < numTries; n++) {
        wxString pathDir = path + asStrF(wxT("%.03x"), (unsigned int)n);
        if (!wxFileName::FileExists(pathDir) && !wxFileName::DirExists(pathDir)) {
            wxDir::Make(pathDir);
            return pathDir;
        }
    }

    return wxEmptyString;
}

wxString asConfig::GetDataDir() {
    ThreadsManager().CritSectionConfig().Enter();
    wxString dirData = wxStandardPaths::Get().GetDataDir();
    ThreadsManager().CritSectionConfig().Leave();
    dirData.Append(DS);
    return dirData;
}

wxString asConfig::GetSoftDir() {
    ThreadsManager().CritSectionConfig().Enter();
    wxString appPath = wxStandardPaths::Get().GetExecutablePath();
    ThreadsManager().CritSectionConfig().Leave();
    wxFileName fileName(appPath);
    wxString appDir = fileName.GetPath();
    appDir.Append(DS);
    return appDir;
}

wxString asConfig::GetUserDataDir() {
#ifdef ON_DOCKER
    wxString userDataDir = "/app/config/";
#else
    ThreadsManager().CritSectionConfig().Enter();
    wxStandardPathsBase& stdPth = wxStandardPaths::Get();
    stdPth.UseAppInfo(0);
    wxString userDataDir = stdPth.GetUserDataDir();

#if defined(__WXMSW__)
    userDataDir.Append(DS + "AtmoSwing");
#elif defined(__WXMAC__)
    userDataDir.Append(DS + "atmoswing");
#elif defined(__UNIX__)
    userDataDir.Append("atmoswing");
#endif

    stdPth.UseAppInfo(1);
    ThreadsManager().CritSectionConfig().Leave();
    userDataDir.Append(DS);
#endif
    return userDataDir;
}

wxString asConfig::GetDocumentsDir() {
#ifdef ON_DOCKER
    wxString dirDocs = "/app/config/";
#else
    ThreadsManager().CritSectionConfig().Enter();
    wxString dirDocs = wxStandardPaths::Get().GetDocumentsDir();
    ThreadsManager().CritSectionConfig().Leave();
    dirDocs.Append(DS);
#endif
    return dirDocs;
}

wxString asConfig::GetDefaultUserWorkingDir() {
#ifdef ON_DOCKER
    wxString dirData = "/app/config/";
#else
    wxString dirData = GetUserDataDir() + DS + "Data" + DS;
#endif
    return dirData;
}

#if USE_GUI
wxColour asConfig::GetFrameBgColour() {
#if defined(__WIN32__)
    return wxSystemSettings::GetColour(wxSYS_COLOUR_BTNFACE);
#elif defined(__UNIX__)
    return wxSystemSettings::GetColour(wxSYS_COLOUR_BACKGROUND);
#elif defined(__APPLE__)
    return wxSystemSettings::GetColour(wxSYS_COLOUR_BTNFACE);
#else
    return wxSystemSettings::GetColour(wxSYS_COLOUR_BTNFACE);
#endif
}
#endif