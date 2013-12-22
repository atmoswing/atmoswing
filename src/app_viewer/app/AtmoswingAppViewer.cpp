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

#ifdef WX_PRECOMP
#include "wx_pch.h"
#endif

#ifdef __BORLANDC__
#pragma hdrstop
#endif //__BORLANDC__

#include "AtmoswingAppViewer.h"
#include "AtmoswingMainViewer.h"

IMPLEMENT_APP(AtmoswingAppViewer);

#include <wx/debug.h>
#include "wx/fileconf.h"S
#include <asIncludes.h>
#include <asThreadsManager.h>
#include <asInternet.h>
#include "vroomgis_bmp.h"
#include "img_bullets.h"
#include "img_toolbar.h"
#include "img_logo.h"

static const wxCmdLineEntryDesc g_cmdLineDesc[] =
{
    { wxCMD_LINE_SWITCH, "h", "help", "displays help on the command line parameters" },
    { wxCMD_LINE_SWITCH, "v", "version", "print version" },
    { wxCMD_LINE_OPTION, "l", "loglevel", "set a log level"
                                "\n \t\t\t\t 0: minimum"
                                "\n \t\t\t\t 1: errors"
                                "\n \t\t\t\t 2: warnings"
                                "\n \t\t\t\t 3: verbose" },
    { wxCMD_LINE_PARAM, NULL, NULL, "input file", wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL },
    { wxCMD_LINE_NONE }
};

bool AtmoswingAppViewer::OnInit()
{
    #if _DEBUG
        _CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
    #endif

    // Set application name and create user directory
    wxString appName = "AtmoSwing viewer";
    wxApp::SetAppName(appName);
    wxFileName userDir = wxFileName::DirName(asConfig::GetUserDataDir());
    userDir.Mkdir(wxS_DIR_DEFAULT,wxPATH_MKDIR_FULL);

    g_AppViewer = true;
    g_AppForecaster = false;

    // Set the local config object
    wxFileConfig *pConfig = new wxFileConfig("AtmoSwing",wxEmptyString,asConfig::GetUserDataDir()+"AtmoSwing.ini",asConfig::GetUserDataDir()+"AtmoSwing.ini",wxCONFIG_USE_LOCAL_FILE);
    wxFileConfig::Set(pConfig);

    // Check that it is the unique instance
    bool multipleInstances;
    pConfig->Read("/Standard/MultiInstances", &multipleInstances, false);

    if (!multipleInstances)
    {
        const wxString instanceName = wxString::Format(wxT("AtmoSwingViewer-%s"),wxGetUserId().c_str());
        m_SingleInstanceChecker = new wxSingleInstanceChecker(instanceName);
        if ( m_SingleInstanceChecker->IsAnotherRunning() )
        {
            //asLogError(_("Program already running, aborting."));
            wxMessageBox(_("Program already running, aborting."));
            return false;
        }
    }

    wxInitAllImageHandlers();

    // Initialize images
    initialize_images_bullets();
    initialize_images_toolbar();
	initialize_images_logo();
    vroomgis_initialize_images();

    // Init cURL
    asInternet::Init();

    // Call default behaviour (mandatory for command-line mode)
    if (!wxApp::OnInit()) // When false, we are in CL mode
        return false;

    // Create frame
    AtmoswingFrameViewer* frame = new AtmoswingFrameViewer(0L);
    frame->OnInit();

#ifdef __WXMSW__
    frame->SetIcon(wxICON(myicon)); // To Set App Icon
#endif
    frame->Show();
    SetTopWindow(frame);

    return true;
}

bool AtmoswingAppViewer::InitForCmdLineOnly(long logLevel)
{
    g_UnitTesting = false;
    g_SilentMode = true;

    // Set log level
    if (logLevel<0)
    {
        logLevel = wxFileConfig::Get()->Read("/Standard/LogLevel", 2l);
    }
    Log().CreateFile("AtmoSwingViewer.log");
    Log().SetLevel((int)logLevel);

    return true;
}

void AtmoswingAppViewer::OnInitCmdLine(wxCmdLineParser& parser)
{
    // From http://wiki.wxwidgets.org/Command-Line_Arguments
    parser.SetDesc (g_cmdLineDesc);
    // Must refuse '/' as parameter starter or cannot use "/path" style paths
    parser.SetSwitchChars (wxT("-"));
}

bool AtmoswingAppViewer::OnCmdLineParsed(wxCmdLineParser& parser)
{
    // From http://wiki.wxwidgets.org/Command-Line_Arguments

    // Check if the user asked for command-line help
    if (parser.Found("h"))
    {
        parser.Usage();
        return false;
    }

    // Check if the user asked for the version
    if (parser.Found("v"))
    {
        wxMessageOutput* msgOut = wxMessageOutput::Get();
        if ( msgOut )
        {
            wxString msg;
            wxString date(wxString::FromAscii(__DATE__));
            msg.Printf("AtmoSwing, (c) University of Lausanne, 2011. Version %s, %s", g_Version.c_str(), (const wxChar*) date);

            msgOut->Printf( wxT("%s"), msg.c_str() );
        }
        else
        {
            wxFAIL_MSG( _("No wxMessageOutput object?") );
        }

        return false;
    }

    // Check for a log level option
    wxString logLevelStr = wxEmptyString;
    long logLevel = -1;
    if (parser.Found("l", & logLevelStr))
    {
        logLevelStr.ToLong(&logLevel);

        // Check and apply
        if (logLevel==0)
        {
            Log().SetLevel(0);
        }
        else if (logLevel==1)
        {
            Log().SetLevel(1);
        }
        else if (logLevel==2)
        {
            Log().SetLevel(2);
        }
        else
        {
            Log().SetLevel(3);
        }
    }

    // Check for input files
    if (parser.GetParamCount()>0)
    {
        InitForCmdLineOnly(logLevel);

        g_CmdFilename = parser.GetParam(0);

        // Under Windows when invoking via a document in Explorer, we are passed th short form.
        // So normalize and make the long form.
        wxFileName fName(g_CmdFilename);
        fName.Normalize(wxPATH_NORM_LONG|wxPATH_NORM_DOTS|wxPATH_NORM_TILDE|wxPATH_NORM_ABSOLUTE);
        g_CmdFilename = fName.GetFullPath();

        return true;
    }

    return true;
}

int AtmoswingAppViewer::OnExit()
{
    // Instance checker
    delete m_SingleInstanceChecker;

    // Config file (from wxWidgets samples)
    delete wxFileConfig::Set((wxFileConfig *) NULL);

    // Delete threads manager and log
    DeleteThreadsManager();
    DeleteLog();

    // Cleanup cURL
    asInternet::Cleanup();

    vroomgis_clear_images();

// TODO (phorton#5#): Do the cleanup here
// Override this member function for any processing which needs to be done as the application is about to exit.
// OnExit is called after destroying all application windows and controls, but before wxWidgets cleanup.

    #ifdef _CRTDBG_MAP_ALLOC
        _CrtDumpMemoryLeaks();
    #endif

    return 1;
}

