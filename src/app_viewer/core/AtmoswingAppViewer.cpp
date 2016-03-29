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
 * Portions Copyright 2014-2015 Pascal Horton, Terranum.
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
#include "wx/fileconf.h"
#include <asIncludes.h>
#include <asThreadsManager.h>
#include <asInternet.h>
#include "vroomgis_bmp.h"
#include "images.h"

static const wxCmdLineEntryDesc g_cmdLineDesc[] =
        {{wxCMD_LINE_SWITCH, "h", "help",      "This help text"},
        {wxCMD_LINE_SWITCH, "v", "version",   "Show version number and quit"},
        {wxCMD_LINE_OPTION, "l", "log-level", "set a log level"
                                 "\n \t\t\t\t 0: minimum"
                                 "\n \t\t\t\t 1: errors"
                                 "\n \t\t\t\t 2: warnings"
                                 "\n \t\t\t\t 3: verbose"},
        {wxCMD_LINE_PARAM, NULL, NULL,        "input file", wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL},
        {wxCMD_LINE_NONE}};

bool AtmoswingAppViewer::OnInit()
{
#if _DEBUG
#ifdef __WXMSW__
    _CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif
#endif

    // Set PPI
    wxMemoryDC dcTestPpi;
    wxSize ppiDC = dcTestPpi.GetPPI();
    g_ppiScaleDc = double(ppiDC.x) / 96.0;

    // Set application name and create user directory
    wxString appName = "AtmoSwing viewer";
    wxApp::SetAppName(appName);
    wxFileName userDir = wxFileName::DirName(asConfig::GetUserDataDir());
    userDir.Mkdir(wxS_DIR_DEFAULT, wxPATH_MKDIR_FULL);

    // Set the local config object
    wxFileConfig *pConfig = new wxFileConfig("AtmoSwing", wxEmptyString,
                                             asConfig::GetUserDataDir() + "AtmoSwingViewer.ini",
                                             asConfig::GetUserDataDir() + "AtmoSwingViewer.ini",
                                             wxCONFIG_USE_LOCAL_FILE);
    wxFileConfig::Set(pConfig);

    // Check that it is the unique instance
    m_singleInstanceChecker = NULL;
    bool multipleInstances;
    pConfig->Read("/General/MultiInstances", &multipleInstances, false);

    if (!multipleInstances) {
        const wxString instanceName = wxString::Format(wxT("atmoswing-viewer-%s"), wxGetUserId());
        m_singleInstanceChecker = new wxSingleInstanceChecker(instanceName);
        if (m_singleInstanceChecker->IsAnotherRunning()) {
            //asLogError(_("Program already running, aborting."));
            wxMessageBox(_("Program already running, aborting."));
            return false;
        }
    }

    wxInitAllImageHandlers();

    // Initialize images
    initialize_images(g_ppiScaleDc);
    vroomgis_initialize_images();

    // Init cURL
    asInternet::Init();

    // Call default behaviour (mandatory for command-line mode)
    if (!wxApp::OnInit()) // When false, we are in CL mode
        return false;

    // Create frame
    AtmoswingFrameViewer *frame = new AtmoswingFrameViewer(0L);
    frame->Init();

#ifdef __WXMSW__
    frame->SetIcon(wxICON(myicon)); // To Set App Icon
#endif
    frame->Show();
    SetTopWindow(frame);

    return true;
}

bool AtmoswingAppViewer::InitForCmdLineOnly(long logLevel)
{
    g_unitTesting = false;
    g_silentMode = true;

    // Set log level
    if (logLevel < 0) {
        logLevel = wxFileConfig::Get()->Read("/General/LogLevel", 2l);
    }
    Log().CreateFile("AtmoSwingViewer.log");
    Log().SetLevel((int) logLevel);

    return true;
}

void AtmoswingAppViewer::OnInitCmdLine(wxCmdLineParser &parser)
{
    // From http://wiki.wxwidgets.org/Command-Line_Arguments
    parser.SetDesc(g_cmdLineDesc);
    // Must refuse '/' as parameter starter or cannot use "/path" style paths
    parser.SetSwitchChars(wxT("-"));
}

bool AtmoswingAppViewer::OnCmdLineParsed(wxCmdLineParser &parser)
{
    // From http://wiki.wxwidgets.org/Command-Line_Arguments

    // Check if the user asked for command-line help
    if (parser.Found("h")) {
        parser.Usage();
        return false;
    }

    // Check if the user asked for the version
    if (parser.Found("v")) {
        wxMessageOutput *msgOut = wxMessageOutput::Get();
        if (msgOut) {
            wxString msg;
            wxString date(wxString::FromAscii(__DATE__));
            msg.Printf("AtmoSwing, (c) University of Lausanne, 2011. Version %s, %s", g_version, (const wxChar *) date);

            msgOut->Printf(wxT("%s"), msg);
        } else {
            wxFAIL_MSG(_("No wxMessageOutput object?"));
        }

        return false;
    }

    // Check for a log level option
    wxString logLevelStr = wxEmptyString;
    long logLevel = -1;
    if (parser.Found("l", &logLevelStr)) {
        logLevelStr.ToLong(&logLevel);

        // Check and apply
        if (logLevel == 0) {
            Log().SetLevel(0);
        } else if (logLevel == 1) {
            Log().SetLevel(1);
        } else if (logLevel == 2) {
            Log().SetLevel(2);
        } else {
            Log().SetLevel(3);
        }
    }

    // Check for input files
    if (parser.GetParamCount() > 0) {
        InitForCmdLineOnly(logLevel);

        g_cmdFilename = parser.GetParam(0);

        // Under Windows when invoking via a document in Explorer, we are passed th short form.
        // So normalize and make the long form.
        wxFileName fName(g_cmdFilename);
        fName.Normalize(wxPATH_NORM_LONG | wxPATH_NORM_DOTS | wxPATH_NORM_TILDE | wxPATH_NORM_ABSOLUTE);
        g_cmdFilename = fName.GetFullPath();

        return true;
    }

    return true;
}

int AtmoswingAppViewer::OnExit()
{
    // Instance checker
    wxDELETE(m_singleInstanceChecker);

    // Config file (from wxWidgets samples)
    delete wxFileConfig::Set((wxFileConfig *) NULL);

    // Delete threads manager and log
    DeleteThreadsManager();
    DeleteLog();

    // Cleanup cURL
    asInternet::Cleanup();

    // Cleanup images
    vroomgis_clear_images();
    cleanup_images();

    return 1;
}

