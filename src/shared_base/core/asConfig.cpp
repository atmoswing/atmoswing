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
    wxStandardPathsBase &stdPth = wxStandardPaths::Get();
    wxString TempDir = stdPth.GetTempDir();
    ThreadsManager().CritSectionConfig().Leave();
    TempDir.Append(DS);
    return TempDir;
}

wxString asConfig::GetTempDir()
{
    ThreadsManager().CritSectionConfig().Enter();
    wxStandardPathsBase &stdPth = wxStandardPaths::Get();
    wxString TempDir = stdPth.GetTempDir();
    ThreadsManager().CritSectionConfig().Leave();
    TempDir.Append(DS);
    return TempDir;
}

wxString asConfig::CreateTempFileName(const wxString& prefix)
{
    wxString path = asConfig::GetTempDir() + DS + prefix;
    wxString pathTry;

    static const size_t numTries = 1000;
    for ( size_t n = 0; n < numTries; n++ )
    {
        // 3 hex digits is enough for numTries == 1000 < 4096
        pathTry = path + wxString::Format(wxT("%.03x"), (unsigned int) n);
        if ( !wxFileName::FileExists(pathTry) )
        {
            break;
        }

        pathTry.clear();
    }

    return pathTry;
}

wxString asConfig::GetDataDir()
{
    ThreadsManager().CritSectionConfig().Enter();
    wxStandardPathsBase &stdPth = wxStandardPaths::Get();
    wxString DirData = stdPth.GetDataDir();
    ThreadsManager().CritSectionConfig().Leave();
    DirData.Append(DS);
    return DirData;
}

wxString asConfig::GetUserDataDir()
{
    ThreadsManager().CritSectionConfig().Enter();
    wxStandardPathsBase &stdPth = wxStandardPaths::Get();
    wxString DirUserData = stdPth.GetUserDataDir();
    ThreadsManager().CritSectionConfig().Leave();
    DirUserData.Append(DS);
    return DirUserData;
}

wxString asConfig::GetUserDataDir(const wxString &appName)
{
    ThreadsManager().CritSectionConfig().Enter();
    wxStandardPathsBase &stdPth = wxStandardPaths::Get();
    stdPth.UseAppInfo(0);
    wxString DirUserData = stdPth.GetUserDataDir();
    ThreadsManager().CritSectionConfig().Leave();

#if defined(__WXMSW__)
    DirUserData.Append(DS+appName);
#elif defined(__WXMAC__)
    DirUserData.Append(DS+appName);
#elif defined(__UNIX__)
    DirUserData.Append(DS+"."+appName);
#endif

    stdPth.UseAppInfo(1);
    DirUserData.Append(DS);
    return DirUserData;
}

wxString asConfig::GetDocumentsDir()
{
    ThreadsManager().CritSectionConfig().Enter();
    wxStandardPathsBase &stdPth = wxStandardPaths::Get();
    wxString DirDocs = stdPth.GetDocumentsDir ();
    ThreadsManager().CritSectionConfig().Leave();
    DirDocs.Append(DS);
    return DirDocs;
}

wxString asConfig::GetDefaultUserWorkingDir()
{
    wxString DirData = GetUserDataDir("Atmoswing") + DS + "Data" + DS;
    return DirData;
}

wxString asConfig::GetDefaultUserConfigDir()
{
    wxString DirConfig = GetUserDataDir("Atmoswing") + DS + "Config" + DS;
    return DirConfig;
}
