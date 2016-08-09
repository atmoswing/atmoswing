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

#ifdef WX_PRECOMP
#include "wx_pch.h"
#endif

#ifdef __BORLANDC__
#pragma hdrstop
#endif //__BORLANDC__

#include "AtmoswingAppOptimizer.h"

#if wxUSE_GUI

#include "AtmoswingMainOptimizer.h"

#endif

#include "asMethodCalibratorClassicPlus.h"
#include "asMethodCalibratorClassicPlusVarExplo.h"
#include "asMethodCalibratorEvaluateAllScores.h"


IMPLEMENT_APP(AtmoswingAppOptimizer);

#include <wx/debug.h>
#include "wx/fileconf.h"
#include "wx/cmdline.h"
#include <asIncludes.h>
#include <asFileAscii.h>
#include <asMethodCalibratorSingle.h>
#include <asMethodCalibratorClassic.h>

#if wxUSE_GUI

#include "images.h"

#endif

static const wxCmdLineEntryDesc g_cmdLineDesc[] = {{wxCMD_LINE_SWITCH, "v",  "version",                 "Show version number and quit"},
                                                   {wxCMD_LINE_SWITCH, "s",  "silent",                  "Silent mode"},
                                                   {wxCMD_LINE_SWITCH, "l",  "local",                   "Work in local directory"},
                                                   {wxCMD_LINE_OPTION, "n",  "threads-nb",              "Number of threads to use"},
                                                   {wxCMD_LINE_OPTION, "r",  "run-number",              "Choice of number associated with the run"},
                                                   {wxCMD_LINE_OPTION, "f",  "file-parameters",         "File containing the calibration parameters"},
                                                   {wxCMD_LINE_OPTION, NULL, "predictand-db",           "The predictand DB"},
                                                   {wxCMD_LINE_OPTION, NULL, "dir-predictors",          "The predictors directory"},
                                                   {wxCMD_LINE_OPTION, NULL, "skip-valid",              "Skip the validation calculation"},
                                                   {wxCMD_LINE_OPTION, NULL, "calibration-method",      "Choice of the calibration method"
                                                                                                                "\n \t\t\t\t single: single assessment"
                                                                                                                "\n \t\t\t\t classic: classic calibration"
                                                                                                                "\n \t\t\t\t classicp: classic+ calibration"
                                                                                                                "\n \t\t\t\t varexplocp: variables exploration classic+"
                                                                                                                "\n \t\t\t\t evalscores: evaluate all scores"},
                                                   {wxCMD_LINE_OPTION, NULL, "cp-resizing-iteration",   "Classic plus: resizing iteration"},
                                                   {wxCMD_LINE_OPTION, NULL, "cp-lat-step",             "Classic plus: steps in latitudes for the relevance map"},
                                                   {wxCMD_LINE_OPTION, NULL, "cp-lon-step",             "Classic plus: steps in longitudes for the relevance map"},
                                                   {wxCMD_LINE_OPTION, NULL, "cp-proceed-sequentially", "Classic plus: proceed sequentially"},
                                                   {wxCMD_LINE_OPTION, NULL, "ve-step",                 "Variables exploration: step"},
                                                   {wxCMD_LINE_OPTION, NULL, "log-level",               "Set a log level"
                                                                                                                "\n \t\t\t\t 0: minimum"
                                                                                                                "\n \t\t\t\t 1: errors"
                                                                                                                "\n \t\t\t\t 2: warnings"
                                                                                                                "\n \t\t\t\t 3: verbose"},
                                                   {wxCMD_LINE_OPTION, NULL, "log-target",              "Set log target"
                                                                                                                "\n \t\t\t\t file: file only"
                                                                                                                "\n \t\t\t\t prompt: command prompt"
                                                                                                                "\n \t\t\t\t both: command prompt and file (default)"},

                                                   {wxCMD_LINE_NONE}};

bool AtmoswingAppOptimizer::OnInit()
{
#if _DEBUG
#ifdef __WXMSW__
    _CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif
#endif

    // Set application name
    wxString appName = "AtmoSwing Optimizer";
    wxApp::SetAppName(appName);

    g_guiMode = true;
    g_local = false;
    m_calibParamsFile = wxEmptyString;
    m_predictandDB = wxEmptyString;
    m_predictorsDir = wxEmptyString;
    m_calibMethod = wxEmptyString;

    // Call default behaviour (mandatory for command-line mode)
    if (!wxApp::OnInit()) // When false, we are in CL mode
        return false;

#if wxUSE_GUI

    // Set PPI
    wxMemoryDC dcTestPpi;
    wxSize ppiDC = dcTestPpi.GetPPI();
    g_ppiScaleDc = double(ppiDC.x) / 96.0;

    m_singleInstanceChecker = NULL;
    if (g_guiMode) {
        // Check that it is the unique instance
        bool multipleInstances = false;

        wxFileConfig::Get()->Read("/General/MultiInstances", &multipleInstances, false);

        if (!multipleInstances) {
            const wxString instanceName = wxString::Format(wxT("atmoswing-optimizer-%s"), wxGetUserId());
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
        AtmoswingFrameOptimizer *frame = new AtmoswingFrameOptimizer(0L);
        frame->OnInit();

#ifdef __WXMSW__
        frame->SetIcon(wxICON(myicon)); // To Set App Icon
#endif
        frame->Show();
        SetTopWindow(frame);
    }
#endif

    return true;
}

bool AtmoswingAppOptimizer::InitForCmdLineOnly()
{
    g_guiMode = false;
    g_unitTesting = false;
    g_silentMode = true;
    g_verboseMode = false;
    g_responsive = false;

    // Prepare local path
    wxString localPath = wxFileName::GetCwd() + DS;
    if (g_runNb > 0) {
        localPath.Append("runs");
        localPath.Append(DS);
        localPath.Append(wxString::Format("%d", g_runNb));
        localPath.Append(DS);
    }

    if (g_local) {
        wxString fullPath = localPath;
        fullPath.Append("AtmoSwingOptimizer.log");
        Log().CreateFileOnlyAtPath(fullPath);
    } else {
        Log().CreateFileOnly("AtmoSwingOptimizer.log");
    }

    Log().DisableMessageBoxOnError();

    if (g_local) {
        wxString dirData = wxFileName::GetCwd() + DS + "data" + DS;

        wxConfigBase *pConfig = wxFileConfig::Get();

        // Define the default preferences
        pConfig->Write("/General/MultiInstances", true);
        pConfig->Write("/General/GuiOptions", 0l);
        pConfig->Write("/General/Responsive", false);
        pConfig->Write("/General/DisplayLogWindow", false);
        pConfig->Write("/Paths/DataPredictandDBDir", dirData);
        pConfig->Write("/Paths/IntermediateResultsDir", localPath + "temp");
        pConfig->Write("/Paths/OptimizerResultsDir", localPath + "results");
        pConfig->Write("/Paths/ArchivePredictorsDir", dirData);
        pConfig->Write("/Processing/AllowMultithreading", true);
        pConfig->Write("/Processing/Method", (long) asMULTITHREADS);
        pConfig->Write("/Processing/LinAlgebra", (long) asLIN_ALGEBRA_NOVAR);
        pConfig->Write("/Processing/ThreadsPriority", 100);
        pConfig->Write("/Optimizer/ParallelEvaluations", true);
        pConfig->Write("/Optimizer/GeneticAlgorithms/AllowElitismForTheBest", true);

        pConfig->Flush();

    }

    return true;
}

void AtmoswingAppOptimizer::OnInitCmdLine(wxCmdLineParser &parser)
{
    wxAppConsole::OnInitCmdLine(parser);

    // From http://wiki.wxwidgets.org/Command-Line_Arguments
    parser.SetDesc(g_cmdLineDesc);
    // Must refuse '/' as parameter starter or cannot use "/path" style paths
    parser.SetSwitchChars(wxT("-"));
}

bool AtmoswingAppOptimizer::OnCmdLineParsed(wxCmdLineParser &parser)
{
    // From http://wiki.wxwidgets.org/Command-Line_Arguments

    /*
     * General options
     */

    // Check if the user asked for command-line help
    if (parser.Found("help")) {
        parser.Usage();
        return false;
    }

    // Check if the user asked for the version
    if (parser.Found("version")) {
        wxMessageOutput *msgOut = wxMessageOutput::Get();
        if (msgOut) {
            wxString msg;
            wxString date(wxString::FromAscii(__DATE__));
            msg.Printf("AtmoSwing version %s, %s", g_version, (const wxChar *) date);

            msgOut->Printf(msg);
        } else {
            wxFAIL_MSG(_("No wxMessageOutput object?"));
        }

        return false; // We don't want to continue
    }

    // Check for a run number
    wxString runNbStr = wxEmptyString;
    long runNb = 0;
    if (parser.Found("run-number", &runNbStr)) {
        runNbStr.ToLong(&runNb);
        g_runNb = (int) runNb;
    }

    // Local mode
    if (parser.Found("local")) {
        g_local = true;
        wxString localPath = wxFileName::GetCwd() + DS;
        if (g_runNb > 0) {
            localPath.Append("runs");
            localPath.Append(DS);
            localPath.Append(wxString::Format("%d", g_runNb));
            localPath.Append(DS);

            // Create directory
            wxFileName userDir = wxFileName::DirName(localPath);
            userDir.Mkdir(wxS_DIR_DEFAULT, wxPATH_MKDIR_FULL);
        }

        // Create local ini file
        wxString iniPath = localPath;
        iniPath.Append("AtmoSwing.ini");

        // Set the local config object
        wxFileConfig *pConfig = new wxFileConfig("AtmoSwing", wxEmptyString, iniPath, iniPath, wxCONFIG_USE_LOCAL_FILE);
        wxFileConfig::Set(pConfig);
    } else {
        // Create user directory
        wxFileName userDir = wxFileName::DirName(asConfig::GetUserDataDir());
        userDir.Mkdir(wxS_DIR_DEFAULT, wxPATH_MKDIR_FULL);

        // Set the local config object
        wxFileConfig *pConfig = new wxFileConfig("AtmoSwing", wxEmptyString,
                                                 asConfig::GetUserDataDir() + "AtmoSwingOptimizer.ini",
                                                 asConfig::GetUserDataDir() + "AtmoSwingOptimizer.ini",
                                                 wxCONFIG_USE_LOCAL_FILE);
        wxFileConfig::Set(pConfig);
    }

    // Check for a log level option
    wxString logLevelStr = wxEmptyString;
    if (parser.Found("log-level", &logLevelStr)) {
        long logLevel = -1;
        logLevelStr.ToLong(&logLevel);

        // Check and apply
        if (logLevel == 0) {
            Log().SetLevel(0);
        } else if (logLevel == 1) {
            Log().SetLevel(1);
        } else if (logLevel == 2) {
            Log().SetLevel(2);
        } else if (logLevel == 3) {
            Log().SetLevel(3);
        } else {
            Log().SetLevel(2);
        }
    } else {
        long logLevel = wxFileConfig::Get()->Read("/General/LogLevel", 2l);
        Log().SetLevel((int) logLevel);
    }

    // Check for the log target option
    wxString logTargetStr = wxEmptyString;
    if (parser.Found("log-target", &logTargetStr)) {
        // Check and apply
        if (logTargetStr.IsSameAs("file", false)) {
            Log().SetTarget(asLog::File);
        } else if (logTargetStr.IsSameAs("screen", false)) {
            Log().SetTarget(asLog::Screen);
        } else if (logTargetStr.IsSameAs("both", false)) {
            Log().SetTarget(asLog::Both);
        } else {
            Log().SetTarget(asLog::Both);

            wxMessageOutput *msgOut = wxMessageOutput::Get();
            if (msgOut) {
                msgOut->Printf(_("The given log target (%s) does not correspond to any possible option."),
                               logTargetStr);
            }
        }
    }

    // Check if the user asked for the silent mode
    if (parser.Found("silent")) {
        Log().SetTarget(asLog::File);
    }

    // Check for a calibration params file
    wxString threadsNb = wxEmptyString;
    if (parser.Found("threads-nb", &threadsNb)) {
        wxFileConfig::Get()->Write("/Processing/MaxThreadNb", threadsNb);
    }

    // Check for a calibration params file
    if (parser.Found("file-parameters", &m_calibParamsFile)) {
        if (g_local) {
            m_calibParamsFile = wxFileName::GetCwd() + DS + m_calibParamsFile;
        }

        if (!wxFileName::FileExists(m_calibParamsFile)) {
            asLogError(wxString::Format(_("The given calibration file (%s) couldn't be found."), m_calibParamsFile));
            return false;
        }
    }

    // Check for a calibration predictand DB
    if (parser.Found("predictand-db", &m_predictandDB)) {
        if (g_local) {
            m_predictandDB = wxFileName::GetCwd() + DS + m_predictandDB;
        }

        if (!wxFileName::FileExists(m_predictandDB)) {
            asLogError(wxString::Format(_("The given predictand DB (%s) couldn't be found."), m_predictandDB));
            return false;
        }
    }

    // Check for a predictors directory
    if (parser.Found("dir-predictors", &m_predictorsDir)) {
        if (g_local) {
            m_predictorsDir = wxFileName::GetCwd() + DS + m_predictorsDir;
        }

        if (!wxFileName::DirExists(m_predictorsDir)) {
            asLogError(wxString::Format(_("The given predictors directory (%s) couldn't be found."), m_predictorsDir));
            return false;
        }
    }

    /*
     * Methods options
     */

    wxString option = wxEmptyString;

    // Classic+ calibration
    if (parser.Found("cp-resizing-iteration", &option)) {
        wxFileConfig::Get()->Write("/Optimizer/ClassicPlus/ResizingIterations", option);
    }

    if (parser.Found("cp-lat-step", &option)) {
        wxFileConfig::Get()->Write("/Optimizer/ClassicPlus/StepsLatPertinenceMap", option);
    }

    if (parser.Found("cp-lon-step", &option)) {
        wxFileConfig::Get()->Write("/Optimizer/ClassicPlus/StepsLonPertinenceMap", option);
    }

    if (parser.Found("cp-proceed-sequentially", &option)) {
        wxFileConfig::Get()->Write("/Optimizer/ClassicPlus/ProceedSequentially", option);
    }

    // Variables exploration
    if (parser.Found("ve-step", &option)) {
        wxFileConfig::Get()->Write("/Optimizer/VariablesExplo/Step", option);
    }

    // Skip validation option
    if (parser.Found("skip-valid", &option)) {
        wxFileConfig::Get()->Write("/Optimizer/SkipValidation", option);
    }

    /*
     * Method choice
     */

    // Check for a calibration method option
    if (parser.Found("calibration-method", &m_calibMethod)) {
        if (!InitForCmdLineOnly())
            return false;
        asLogMessage(wxString::Format(_("Given calibration method: %s"), m_calibMethod));
        return true;
    }

    return wxAppConsole::OnCmdLineParsed(parser);
}

int AtmoswingAppOptimizer::OnRun()
{
    if (!g_guiMode) {
        if (m_calibParamsFile.IsEmpty()) {
            asLogError(_("The parameters file is not given."));
            return 1001;
        }

        if (m_predictandDB.IsEmpty()) {
            asLogError(_("The predictand DB is not given."));
            return 1002;
        }

        if (m_predictorsDir.IsEmpty()) {
            asLogError(_("The predictors directory is not given."));
            return 1003;
        }

        wxMessageOutput *msgOut = wxMessageOutput::Get();

        try {
            if (m_calibMethod.IsSameAs("single", false)) {
                asMethodCalibratorSingle calibrator;
                calibrator.SetParamsFilePath(m_calibParamsFile);
                calibrator.SetPredictandDBFilePath(m_predictandDB);
                calibrator.SetPredictorDataDir(m_predictorsDir);
                calibrator.Manager();
            } else if (m_calibMethod.IsSameAs("classic", false)) {
                asMethodCalibratorClassic calibrator;
                calibrator.SetParamsFilePath(m_calibParamsFile);
                calibrator.SetPredictandDBFilePath(m_predictandDB);
                calibrator.SetPredictorDataDir(m_predictorsDir);
                calibrator.Manager();
            } else if (m_calibMethod.IsSameAs("classicp", false)) {
                asMethodCalibratorClassicPlus calibrator;
                calibrator.SetParamsFilePath(m_calibParamsFile);
                calibrator.SetPredictandDBFilePath(m_predictandDB);
                calibrator.SetPredictorDataDir(m_predictorsDir);
                calibrator.Manager();
            } else if (m_calibMethod.IsSameAs("varexplocp", false)) {
                asMethodCalibratorClassicPlusVarExplo calibrator;
                calibrator.SetParamsFilePath(m_calibParamsFile);
                calibrator.SetPredictandDBFilePath(m_predictandDB);
                calibrator.SetPredictorDataDir(m_predictorsDir);
                calibrator.Manager();
            } else if (m_calibMethod.IsSameAs("evalscores", false)) {
                asMethodCalibratorEvaluateAllScores calibrator;
                calibrator.SetParamsFilePath(m_calibParamsFile);
                calibrator.SetPredictandDBFilePath(m_predictandDB);
                calibrator.SetPredictorDataDir(m_predictorsDir);
                calibrator.Manager();
            } else {
                if (msgOut) {
                    msgOut->Printf("Wrong calibration method selection (%s).", m_calibMethod);
                }
            }
        } catch (std::bad_alloc &ba) {
            wxString msg(ba.what(), wxConvUTF8);
            asLogError(wxString::Format(_("Bad allocation caught: %s"), msg));
            return 1011;
        } catch (asException &e) {
            wxString fullMessage = e.GetFullMessage();
            if (!fullMessage.IsEmpty()) {
                asLogError(fullMessage);
            }
            asLogError(_("Failed to process the calibration."));
            return 1010;
        }

        asLogMessageImportant(_("Calibration over."));

        return 0;
    }

    return wxApp::OnRun();
}

int AtmoswingAppOptimizer::OnExit()
{
#if wxUSE_GUI
    // Instance checker
    wxDELETE(m_singleInstanceChecker);
#endif

    // Config file (from wxWidgets samples)
    delete wxFileConfig::Set((wxFileConfig *) NULL);

    // Delete threads manager and log
    DeleteThreadsManager();
    DeleteLog();

#if wxUSE_GUI
    // Delete images
    cleanup_images();
#endif

    // CleanUp
    wxApp::CleanUp();

    return 0;
}

bool AtmoswingAppOptimizer::OnExceptionInMainLoop()
{
    asLogError(_("An exception occured in the main loop"));
    return false;
}

void AtmoswingAppOptimizer::OnFatalException()
{
    asLogError(_("An fatal exception occured"));
}

void AtmoswingAppOptimizer::OnUnhandledException()
{
    asLogError(_("An unhandled exception occured"));
}

