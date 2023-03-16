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
 * Portions Copyright 2017 Pascal Horton, University of Bern.
 */

#ifdef WX_PRECOMP
#include "wx_pch.h"
#endif

#ifdef __BORLANDC__
#pragma hdrstop
#endif  //__BORLANDC__

#include "AtmoswingAppDownscaler.h"

#if USE_GUI

#include "AtmoswingMainDownscaler.h"

#endif

#include "asMethodDownscalerClassic.h"
#include "asParameters.h"

IMPLEMENT_APP(AtmoswingAppDownscaler)

#include "asFileText.h"
#include "asMethodDownscalerClassic.h"

#if USE_GUI

#include "asParameters.h"
#include "images.h"

#endif

static const wxCmdLineEntryDesc g_cmdLineDesc[] = {
    {wxCMD_LINE_SWITCH, "v", "version", "Show version number and quit"},
    {wxCMD_LINE_SWITCH, "s", "silent", "Silent mode"},
    {wxCMD_LINE_SWITCH, "l", "local", "Work in local directory"},
    {wxCMD_LINE_OPTION, "n", "threads-nb", "Number of threads to use"},
    {wxCMD_LINE_OPTION, "r", "run-number", "Choice of number associated with the run"},
    {wxCMD_LINE_OPTION, "f", "file-parameters", "File containing the downscaling parameters"},
    {wxCMD_LINE_OPTION, NULL, "predictand-db", "The predictand DB"},
    {wxCMD_LINE_OPTION, NULL, "station-id", "The predictand station ID"},
    {wxCMD_LINE_OPTION, NULL, "dir-archive-predictors", "The archive predictors directory"},
    {wxCMD_LINE_OPTION, NULL, "dir-scenario-predictors", "The scenario predictors directory"},
    {wxCMD_LINE_OPTION, NULL, "downscaling-method",
     "Choice of the downscaling method"
     "\n \t\t\t\t\t - classic: classic downscaling"},
    {wxCMD_LINE_OPTION, NULL, "log-level",
     "Set a log level"
     "\n \t\t\t\t\t - 1: errors"
     "\n \t\t\t\t\t - 2: warnings"
     "\n \t\t\t\t\t - 3: verbose"},

    {wxCMD_LINE_NONE}};

static const wxString cmdLineLogo = wxT(
    "\n"
    "_________________________________________\n"
    "____ ___ _  _ ____ ____ _ _ _ _ _  _ ____ \n"
    "|__|  |  |\\/| |  | [__  | | | | |\\ | | __ \n"
    "|  |  |  |  | |__| ___] |_|_| | | \\| |__] \n"
    "_________________________________________\n"
    "\n");

bool AtmoswingAppDownscaler::OnInit() {
#if _DEBUG
#ifdef __WXMSW__
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
#endif

    // Set application name
    wxString appName = "AtmoSwing Downscaler";
    wxApp::SetAppName(appName);

    g_local = false;
    m_downscalingParamsFile = wxEmptyString;
    m_predictandDB = wxEmptyString;
    m_predictandStationIds = vi(0);
    m_predictorsArchiveDir = wxEmptyString;
    m_predictorsScenarioDir = wxEmptyString;
    m_downscalingMethod = wxEmptyString;
    m_doProcessing = false;
#if USE_GUI
    g_guiMode = true;
    m_singleInstanceChecker = nullptr;
#else
    g_guiMode = false;
#endif

    // Call default behaviour
    if (!wxApp::OnInit()) {
        return false;
    }

    // Skip frame initialization if needed.
    if (!g_guiMode) {
        return true;
    }

#if USE_GUI
    // Set PPI
    wxMemoryDC dcTestPpi;
    wxSize ppiDC = dcTestPpi.GetPPI();
    g_ppiScaleDc = wxMax(double(ppiDC.x) / 96.0, 1.0);

    m_singleInstanceChecker = nullptr;

    // Check that it is the unique instance
    if (!wxFileConfig::Get()->ReadBool("/General/MultiInstances", false)) {
        const wxString instanceName = asStrF(wxT("atmoswing-downscaler-%s"), wxGetUserId());
        m_singleInstanceChecker = new wxSingleInstanceChecker(instanceName);
        if (m_singleInstanceChecker->IsAnotherRunning()) {
            wxMessageBox(_("Program already running, aborting."));
            return false;
        }
    }

    // Following for GUI only
    wxInitAllImageHandlers();

    // Initialize images
    initialize_images(g_ppiScaleDc);

    // Create frame
    AtmoswingFrameDownscaler* frame = new AtmoswingFrameDownscaler(0L);
    frame->OnInit();

#ifdef __WXMSW__
    frame->SetIcon(wxICON(myicon));  // To Set App Icon
#endif
    frame->Show();
    SetTopWindow(frame);
#endif

    return true;
}

wxString AtmoswingAppDownscaler::GetLocalPath() {
    // Prepare local path
    wxString localPath = wxFileName::GetCwd() + DS;
    if (g_runNb > 0) {
        localPath.Append("runs");
        localPath.Append(DS);
        localPath.Append(asStrF("%d", g_runNb));
        localPath.Append(DS);
    }

    return localPath;
}

bool AtmoswingAppDownscaler::InitLog() {
    if (g_local) {
        wxString fullPath = GetLocalPath();
        fullPath.Append("AtmoSwingDownscaler.log");

#if USE_GUI
        if (!g_guiMode) {
            Log()->CreateFileOnly("AtmoSwingDownscaler.log");
        } else {
            delete wxLog::SetActiveTarget(new asLogGui());
            Log()->CreateFileAtPath(fullPath);
        }
#else
        Log()->CreateFileOnlyAtPath(fullPath);
#endif
    } else {
#if USE_GUI
        if (!g_guiMode) {
            Log()->CreateFileOnly("AtmoSwingDownscaler.log");
        }
        // GUI mode: will be set later
#else
        Log()->CreateFileOnly("AtmoSwingDownscaler.log");
#endif
    }

    return true;
}

bool AtmoswingAppDownscaler::SetUseAsCmdLine() {
    g_guiMode = false;
    g_unitTesting = false;
    g_silentMode = true;
    g_verboseMode = false;
    g_responsive = false;

    return true;
}

bool AtmoswingAppDownscaler::InitForCmdLineOnly() {
    if (g_local) {
        wxString dirData = wxFileName::GetCwd() + DS + "data" + DS;

        wxConfigBase* pConfig = wxFileConfig::Get();

        // Define the default preferences
        pConfig->Write("/General/MultiInstances", true);
        pConfig->Write("/General/GuiOptions", 0l);
        pConfig->Write("/General/Responsive", false);
        pConfig->Write("/General/DisplayLogWindow", false);
        pConfig->Write("/Paths/DataPredictandDBDir", dirData);
        pConfig->Write("/Paths/DownscalerResultsDir", GetLocalPath() + "results");
        pConfig->Write("/Paths/ArchivePredictorsDir", dirData);
        pConfig->Write("/Processing/Method", (long)asMULTITHREADS);
        pConfig->Write("/Processing/ThreadsPriority", 100);
        pConfig->Write("/Processing/AllowMultithreading", true);
        if (pConfig->ReadLong("/Processing/ThreadsNb", 1) > 1) {
            pConfig->Write("/ParallelEvaluations", true);
        }

        pConfig->Flush();
    }

    return true;
}

void AtmoswingAppDownscaler::OnInitCmdLine(wxCmdLineParser& parser) {
    wxAppConsole::OnInitCmdLine(parser);

    parser.SetDesc(g_cmdLineDesc);
    parser.SetLogo(cmdLineLogo);

    // Must refuse '/' as parameter starter or cannot use "/path" style paths
    parser.SetSwitchChars(wxT("-"));
}

bool AtmoswingAppDownscaler::OnCmdLineParsed(wxCmdLineParser& parser) {
    // Check if runs with GUI or CL
    if (parser.Found("downscaling-method")) {
        SetUseAsCmdLine();
    }

    /*
     * General options
     */

    // Check for a run number
    wxString runNbStr = wxEmptyString;
    long runNb = 0;
    if (parser.Found("run-number", &runNbStr)) {
        if (runNbStr.ToLong(&runNb)) {
            g_runNb = (int)runNb;
        } else {
            g_runNb = rand();
        }
    }

    // Local mode
    if (parser.Found("local")) {
        g_local = true;
        wxString localPath = wxFileName::GetCwd() + DS;
        if (g_runNb > 0) {
            localPath.Append("runs");
            localPath.Append(DS);
            localPath.Append(asStrF("%d", g_runNb));
            localPath.Append(DS);

            // Check if path already exists
            if (wxFileName::Exists(localPath)) {
                asLog::PrintToConsole(_("A directory with the same name already exists.\n"));
                wxLogError(_("A directory with the same name already exists."));
                return false;
            } else {
                // Create directory
                wxFileName userDir = wxFileName::DirName(localPath);
                userDir.Mkdir(wxS_DIR_DEFAULT, wxPATH_MKDIR_FULL);
            }
        }

        // Create local ini file
        wxString iniPath = localPath;
        iniPath.Append("AtmoSwing.ini");

        // Set the local config object
        wxFileConfig* pConfig = new wxFileConfig("AtmoSwing", wxEmptyString, iniPath, iniPath, wxCONFIG_USE_LOCAL_FILE);
        wxFileConfig::Set(pConfig);
    } else {
        // Create user directory
        wxFileName userDir = wxFileName::DirName(asConfig::GetUserDataDir());
        userDir.Mkdir(wxS_DIR_DEFAULT, wxPATH_MKDIR_FULL);

        // Set the local config object
        wxFileConfig* pConfig = new wxFileConfig(
            "AtmoSwing", wxEmptyString, asConfig::GetUserDataDir() + "AtmoSwingDownscaler.ini",
            asConfig::GetUserDataDir() + "AtmoSwingDownscaler.ini", wxCONFIG_USE_LOCAL_FILE);
        wxFileConfig::Set(pConfig);
    }

    // Initialize log
    InitLog();

    // Check if the user asked for command-line help
    if (parser.Found("help")) {
        parser.Usage();

        return true;
    }

    // Check if the user asked for the version
    if (parser.Found("version")) {
        wxString date(wxString::FromAscii(__DATE__));
        asLog::PrintToConsole(asStrF("AtmoSwing version %s, %s\n", g_version, date));

        return true;
    }

    // Check for a log level option
    wxString logLevelStr = wxEmptyString;
    if (parser.Found("log-level", &logLevelStr)) {
        long logLevel = -1;
        if (!logLevelStr.ToLong(&logLevel)) {
            asLog::PrintToConsole(_("The value provided for 'log-level' could not be interpreted.\n"));
            return false;
        }

        // Check and apply
        if (logLevel >= 1 && logLevel <= 3) {
            Log()->SetLevel(int(logLevel));
        } else {
            Log()->SetLevel(2);
        }
    } else {
        Log()->SetLevel(wxFileConfig::Get()->ReadLong("/General/LogLevel", 2l));
    }

    // Check for a downscaling params file
    wxString threadsNb = wxEmptyString;
    if (parser.Found("threads-nb", &threadsNb)) {
        wxFileConfig::Get()->Write("/Processing/ThreadsNb", threadsNb);
    }

    // Check for a downscaling params file
    if (parser.Found("file-parameters", &m_downscalingParamsFile)) {
        if (g_local) {
            m_downscalingParamsFile = wxFileName::GetCwd() + DS + m_downscalingParamsFile;
        }

        if (!wxFileName::FileExists(m_downscalingParamsFile)) {
            wxLogError(_("The given downscaling file (%s) couldn't be found."), m_downscalingParamsFile);
            return false;
        }
    }

    // Check for a downscaling predictand DB
    if (parser.Found("predictand-db", &m_predictandDB)) {
        if (g_local) {
            m_predictandDB = wxFileName::GetCwd() + DS + m_predictandDB;
        }

        if (!wxFileName::FileExists(m_predictandDB)) {
            wxLogError(_("The given predictand DB (%s) couldn't be found."), m_predictandDB);
            return false;
        }
    }

    // Check for archive predictors directory
    if (parser.Found("dir-archive-predictors", &m_predictorsArchiveDir)) {
        if (g_local && wxFileName::Exists(wxFileName::GetCwd() + DS + m_predictorsArchiveDir)) {
            m_predictorsArchiveDir = wxFileName::GetCwd() + DS + m_predictorsArchiveDir;
        }

        if (!wxFileName::DirExists(m_predictorsArchiveDir)) {
            wxLogError(_("The given archive predictors directory (%s) couldn't be found."), m_predictorsArchiveDir);
            return false;
        }
    }

    // Check for scenario predictors directory
    if (parser.Found("dir-scenario-predictors", &m_predictorsScenarioDir)) {
        if (g_local && wxFileName::Exists(wxFileName::GetCwd() + DS + m_predictorsScenarioDir)) {
            m_predictorsScenarioDir = wxFileName::GetCwd() + DS + m_predictorsScenarioDir;
        }

        if (!wxFileName::DirExists(m_predictorsScenarioDir)) {
            wxLogError(_("The given scenario predictors directory (%s) couldn't be found."), m_predictorsScenarioDir);
            return false;
        }
    }

    // Station ID
    wxString stationIdStr = wxEmptyString;
    if (parser.Found("station-id", &stationIdStr)) {
        m_predictandStationIds = asParameters::GetFileStationIds(stationIdStr);
    }

    /*
     * Method choice
     */

    // Check for a downscaling method option
    if (parser.Found("downscaling-method", &m_downscalingMethod)) {
        if (!InitForCmdLineOnly()) {
            wxLogError(_("Initialization for command-line interface failed."));
            return false;
        }
        m_doProcessing = true;
        wxLogVerbose(_("Given downscaling method: %s"), m_downscalingMethod);
        return true;
    }

    // Finally, if no option is given in CL mode, display help.
    if (!g_guiMode) {
        parser.Usage();

        return false;
    }

    return true;
}

int AtmoswingAppDownscaler::OnRun() {
    if (g_guiMode) {
        return wxApp::OnRun();
    }

    if (!m_doProcessing) {
        return 0;
    }

    if (m_downscalingParamsFile.IsEmpty()) {
        wxLogError(_("The parameters file is not given."));
        return 1;
    }

    if (m_predictandDB.IsEmpty()) {
        wxLogError(_("The predictand DB is not given."));
        return 1;
    }

    if (m_predictorsArchiveDir.IsEmpty()) {
        wxLogError(_("The predictors directory is not given."));
        return 1;
    }

    try {
        if (m_downscalingMethod.IsSameAs("classic", false)) {
            asMethodDownscalerClassic downscaler;
            downscaler.SetParamsFilePath(m_downscalingParamsFile);
            downscaler.SetPredictandDBFilePath(m_predictandDB);
            downscaler.SetPredictandStationIds(m_predictandStationIds);
            downscaler.SetPredictorDataDir(m_predictorsArchiveDir);
            downscaler.Manager();
        } else {
            asLog::PrintToConsole(asStrF("Wrong downscaling method selection (%s).\n", m_downscalingMethod));
        }
    } catch (std::bad_alloc& ba) {
        wxString msg(ba.what(), wxConvUTF8);
        wxLogError(_("Bad allocation caught: %s"), msg);
        return 1;
    } catch (std::exception& e) {
        wxString msg(e.what(), wxConvUTF8);
        wxLogError(_("Exception caught: %s"), msg);
        return 1;
    }

    wxLogMessage(_("Downscaling over."));
    asLog::PrintToConsole(_("Downscaling over.\n"));

    return 0;
}

int AtmoswingAppDownscaler::OnExit() {
    CleanUp();

    return 0;
}

void AtmoswingAppDownscaler::CleanUp() {
#if USE_GUI
    // Instance checker
    wxDELETE(m_singleInstanceChecker);
#endif

    // Config file (from wxWidgets samples)
    delete wxFileConfig::Set((wxFileConfig*)nullptr);

    // Delete threads manager and log
    DeleteThreadsManager();
    DeleteLog();

#if USE_GUI
    // Delete images
    cleanup_images();
#endif

    // CleanUp
    wxApp::CleanUp();
}

bool AtmoswingAppDownscaler::OnExceptionInMainLoop() {
    wxLogError(_("An exception occured in the main loop"));
    return false;
}

void AtmoswingAppDownscaler::OnFatalException() {
    wxLogError(_("An fatal exception occured"));
}

void AtmoswingAppDownscaler::OnUnhandledException() {
    wxLogError(_("An unhandled exception occured"));
}
