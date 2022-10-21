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
#endif  //__BORLANDC__

#include "AtmoswingAppOptimizer.h"

#if USE_GUI

#include "AtmoswingMainOptimizer.h"

#endif

#include "asMethodCalibratorClassic.h"
#include "asMethodCalibratorClassicVarExplo.h"
#include "asMethodCalibratorEvaluateAllScores.h"
#include "asMethodCalibratorSingleOnlyDates.h"
#include "asMethodCalibratorSingleOnlyValues.h"
#include "asMethodOptimizerGeneticAlgorithms.h"
#include "asMethodOptimizerRandomSet.h"

IMPLEMENT_APP(AtmoswingAppOptimizer)

#include "asFileText.h"
#include "asMethodCalibratorSingle.h"

#if USE_GUI

#include "images.h"

#endif

static const wxCmdLineEntryDesc g_cmdLineDesc[] = {
    {wxCMD_LINE_SWITCH, "v", "version", "Show version number and quit"},
    {wxCMD_LINE_SWITCH, "s", "silent", "Silent mode"},
    {wxCMD_LINE_SWITCH, "l", "local", "Work in local directory"},
    {wxCMD_LINE_OPTION, "n", "threads-nb", "Number of threads to use"},
    {wxCMD_LINE_OPTION, "g", "gpus-nb", "Number of gpus to use"},
    {wxCMD_LINE_OPTION, "r", "run-number", "Choice of number associated with the run"},
    {wxCMD_LINE_OPTION, "f", "file-parameters", "File containing the calibration parameters"},
    {wxCMD_LINE_OPTION, NULL, "predictand-db", "The predictand DB"},
    {wxCMD_LINE_OPTION, NULL, "station-id", "The predictand station ID"},
    {wxCMD_LINE_OPTION, NULL, "dir-predictors", "The predictors directory"},
    {wxCMD_LINE_SWITCH, NULL, "skip-valid", "Skip the validation calculation"},
    {wxCMD_LINE_SWITCH, NULL, "no-duplicate-dates",
     "Do not allow to keep several times the same analog dates (e.g. for ensembles)"},
    {wxCMD_LINE_SWITCH, NULL, "dump-predictor-data", "Dump predictor data to binary files to reduce RAM usage"},
    {wxCMD_LINE_SWITCH, NULL, "load-from-dumped-data", "Load dumped predictor data into RAM (faster load)"},
    {wxCMD_LINE_SWITCH, NULL, "replace-nans", "Option to replace NaNs with -9999 (faster processing)"},
    {wxCMD_LINE_SWITCH, NULL, "skip-nans-check", "Do not check for NaNs (faster processing)"},
    {wxCMD_LINE_OPTION, NULL, "calibration-method",
     "Choice of the calibration method"
     "\n \t\t\t\t\t - single: single assessment"
     "\n \t\t\t\t\t - classic: classic calibration"
     "\n \t\t\t\t\t - classicp: classic+ calibration"
     "\n \t\t\t\t\t - varexplocp: variables exploration classic+"
     "\n \t\t\t\t\t - montecarlo: Monte Carlo"
     "\n \t\t\t\t\t - ga: genetic algorithms"
     "\n \t\t\t\t\t - evalscores: evaluate all scores"
     "\n \t\t\t\t\t - onlyvalues: evaluate all scores"
     "\n \t\t\t\t\t - onlydates: evaluate all scores"},
    {wxCMD_LINE_OPTION, NULL, "cp-resizing-iteration", "Classic plus: resizing iteration"},
    {wxCMD_LINE_OPTION, NULL, "cp-lat-step", "Classic plus: steps in latitudes for the relevance map"},
    {wxCMD_LINE_OPTION, NULL, "cp-lon-step", "Classic plus: steps in longitudes for the relevance map"},
    {wxCMD_LINE_SWITCH, NULL, "cp-proceed-sequentially", "Classic plus: proceed sequentially"},
    {wxCMD_LINE_OPTION, NULL, "ve-step", "Variables exploration: step to process"},
    {wxCMD_LINE_OPTION, NULL, "mc-runs-nb", "Monte Carlo: number of runs"},
    {wxCMD_LINE_OPTION, NULL, "ga-config", "GAs: predefined configuration of options (1-5)"},
    {wxCMD_LINE_OPTION, NULL, "ga-use-mini-batches", "GAs: use mini-batches (1/0)"},
    {wxCMD_LINE_OPTION, NULL, "ga-mini-batch-size", "GAs: size of the mini-batches"},
    {wxCMD_LINE_OPTION, NULL, "ga-number-epochs", "GAs: number of epochs if using mini-batches"},
    {wxCMD_LINE_OPTION, NULL, "ga-ope-nat-sel", "GAs: operator choice for natural selection"},
    {wxCMD_LINE_OPTION, NULL, "ga-ope-coup-sel", "GAs: operator choice for couples selection"},
    {wxCMD_LINE_OPTION, NULL, "ga-ope-cross", "GAs: operator choice for chromosome crossover"},
    {wxCMD_LINE_OPTION, NULL, "ga-ope-mut", "GAs: operator choice for mutation"},
    {wxCMD_LINE_OPTION, NULL, "ga-pop-size", "GAs: size of the population"},
    {wxCMD_LINE_OPTION, NULL, "ga-conv-steps", "GAs: number of generations for convergence"},
    {wxCMD_LINE_OPTION, NULL, "ga-interm-gen", "GAs: ratio of the intermediate generation"},
    {wxCMD_LINE_OPTION, NULL, "ga-nat-sel-tour-p", "GAs: natural selection - tournament probability"},
    {wxCMD_LINE_OPTION, NULL, "ga-coup-sel-tour-nb", "GAs: couples selection - tournament candidates (2-3)"},
    {wxCMD_LINE_OPTION, NULL, "ga-cross-mult-pt-nb", "GAs: standard crossover - number of points"},
    {wxCMD_LINE_OPTION, NULL, "ga-cross-blen-pt-nb", "GAs: blending crossover - number of points"},
    {wxCMD_LINE_OPTION, NULL, "ga-cross-blen-share-b", "GAs: blending crossover - beta shared (1/0)"},
    {wxCMD_LINE_OPTION, NULL, "ga-cross-lin-pt-nb", "GAs: linear crossover - number of points"},
    {wxCMD_LINE_OPTION, NULL, "ga-cross-heur-pt-nb", "GAs: heuristic crossover - number of points"},
    {wxCMD_LINE_OPTION, NULL, "ga-cross-heur-share-b", "GAs: heuristic crossover - beta shared (1/0)"},
    {wxCMD_LINE_OPTION, NULL, "ga-cross-bin-pt-nb", "GAs: binary-like crossover - number of points"},
    {wxCMD_LINE_OPTION, NULL, "ga-cross-bin-share-b", "GAs: binary-like crossover - beta shared (1/0)"},
    {wxCMD_LINE_OPTION, NULL, "ga-mut-unif-cst-p", "GAs: uniform mutation - probability"},
    {wxCMD_LINE_OPTION, NULL, "ga-mut-norm-cst-p", "GAs: normal mutation - probability"},
    {wxCMD_LINE_OPTION, NULL, "ga-mut-norm-cst-dev", "GAs: normal mutation - standard deviation"},
    {wxCMD_LINE_OPTION, NULL, "ga-mut-unif-var-gens", "GAs: variable uniform mutation - generations nb"},
    {wxCMD_LINE_OPTION, NULL, "ga-mut-unif-var-p-strt", "GAs: variable uniform mutation - starting probability"},
    {wxCMD_LINE_OPTION, NULL, "ga-mut-unif-var-p-end", "GAs: variable uniform mutation - end probability"},
    {wxCMD_LINE_OPTION, NULL, "ga-mut-norm-var-gens-p",
     "GAs: variable normal mutation - generations nb for probability"},
    {wxCMD_LINE_OPTION, NULL, "ga-mut-norm-var-gens-d",
     "GAs: variable normal mutation - generations nb for std deviation"},
    {wxCMD_LINE_OPTION, NULL, "ga-mut-norm-var-p-strt", "GAs: variable normal mutation - starting probability"},
    {wxCMD_LINE_OPTION, NULL, "ga-mut-norm-var-p-end", "GAs: variable normal mutation - end probability"},
    {wxCMD_LINE_OPTION, NULL, "ga-mut-norm-var-d-strt", "GAs: variable normal mutation - starting std deviation"},
    {wxCMD_LINE_OPTION, NULL, "ga-mut-norm-var-d-end", "GAs: variable normal mutation - end std deviation"},
    {wxCMD_LINE_OPTION, NULL, "ga-mut-non-uni-p", "GAs: non uniform mutation - probability"},
    {wxCMD_LINE_OPTION, NULL, "ga-mut-non-uni-gens", "GAs: non uniform mutation - generations nb"},
    {wxCMD_LINE_OPTION, NULL, "ga-mut-non-uni-min-r", "GAs: non uniform mutation - minimum rate"},
    {wxCMD_LINE_OPTION, NULL, "ga-mut-multi-scale-p", "GAs: multi-scale mutation - probability"},

    {wxCMD_LINE_OPTION, NULL, "log-level",
     "Set a log level"
     "\n \t\t\t\t\t - 1: errors"
     "\n \t\t\t\t\t - 2: warnings"
     "\n \t\t\t\t\t - 3: verbose"},

    {wxCMD_LINE_NONE}};

static const wxString cmdLineLogo =
    wxT("\n"
        "_________________________________________\n"
        "____ ___ _  _ ____ ____ _ _ _ _ _  _ ____ \n"
        "|__|  |  |\\/| |  | [__  | | | | |\\ | | __ \n"
        "|  |  |  |  | |__| ___] |_|_| | | \\| |__] \n"
        "_________________________________________\n"
        "\n");

bool AtmoswingAppOptimizer::OnInit() {
#if _DEBUG
#ifdef __WXMSW__
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
#endif

    // Set application name
    wxString appName = "AtmoSwing Optimizer";
    wxApp::SetAppName(appName);

    g_local = false;
    m_calibParamsFile = wxEmptyString;
    m_predictandDB = wxEmptyString;
    m_predictandStationIds = vi(0);
    m_predictorsDir = wxEmptyString;
    m_calibMethod = wxEmptyString;
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
    AtmoswingFrameOptimizer* frame = new AtmoswingFrameOptimizer(0L);
    frame->OnInit();

#ifdef __WXMSW__
    frame->SetIcon(wxICON(myicon));  // To Set App Icon
#endif
    frame->Show();
    SetTopWindow(frame);
#endif

    return true;
}

wxString AtmoswingAppOptimizer::GetLocalPath() {
    // Prepare local path
    wxString localPath = wxFileName::GetCwd() + DS;
    if (g_runNb > 0) {
        localPath.Append("runs");
        localPath.Append(DS);
        localPath.Append(wxString::Format("%d", g_runNb));
        localPath.Append(DS);
    }

    return localPath;
}

bool AtmoswingAppOptimizer::InitLog() {
    if (g_local) {
        wxString fullPath = GetLocalPath();
        fullPath.Append("AtmoSwingOptimizer.log");
        if (g_resumePreviousRun) {
            int increment = 1;
            while (wxFileName::Exists(fullPath)) {
                increment++;
                fullPath = GetLocalPath();
                fullPath.Append(wxString::Format("AtmoSwingOptimizer-%d.log", increment));
            }
        }

#if USE_GUI
        if (!g_guiMode) {
            Log()->CreateFileOnlyAtPath(fullPath);
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
            Log()->CreateFileOnly("AtmoSwingOptimizer.log");
        }
        // GUI mode: will be set later
#else
        Log()->CreateFileOnly("AtmoSwingOptimizer.log");
#endif
    }

    return true;
}

bool AtmoswingAppOptimizer::SetUseAsCmdLine() {
    g_guiMode = false;
    g_unitTesting = false;
    g_silentMode = true;
    g_verboseMode = false;
    g_responsive = false;

    return true;
}

bool AtmoswingAppOptimizer::InitForCmdLineOnly() {
    // Warn the user if reloading previous results
    if (g_resumePreviousRun) {
        wxLogWarning(_("An existing directory was found for the run number %d"), g_runNb);
        printf("Warning: An existing directory was found for the run number %d\n", g_runNb);
    }

    if (g_local) {
        wxString dirData = wxFileName::GetCwd() + DS + "data" + DS;

        wxConfigBase* pConfig = wxFileConfig::Get();

        // Define the default preferences
        pConfig->Write("/General/MultiInstances", true);
        pConfig->Write("/General/GuiOptions", 0l);
        pConfig->Write("/General/Responsive", false);
        pConfig->Write("/General/DisplayLogWindow", false);
        pConfig->Write("/Paths/DataPredictandDBDir", dirData);
        pConfig->Write("/Paths/ResultsDir", GetLocalPath() + "results");
        pConfig->Write("/Paths/ArchivePredictorsDir", dirData);
        pConfig->Write("/Processing/ThreadsPriority", 100);
        pConfig->Write("/Processing/AllowMultithreading", true);
        pConfig->Write("/Processing/Method", (long)asMULTITHREADS);
        if (pConfig->ReadLong("/Processing/GpusNb", 0) > 0) {
            pConfig->Write("/Processing/Method", (long)asCUDA);
        }
        if (m_calibMethod.IsSameAs("ga", false)) {
            pConfig->Write("/Processing/AllowMultithreading", false);  // Because we are using parallel evaluations
            pConfig->Write("/GAs/AllowElitismForTheBest", true);
        }

        pConfig->Flush();
    }

    // Check that the config files correspond if reloading data
    if (g_resumePreviousRun) {
        wxConfigBase* pConfigNow = wxFileConfig::Get();
        wxString refIniPath = GetLocalPath();
        refIniPath.Append("AtmoSwing.ini");
        wxFileConfig* pConfigRef = new wxFileConfig("AtmoSwing", wxEmptyString, refIniPath, refIniPath,
                                                    wxCONFIG_USE_LOCAL_FILE);

        // Check that the number of groups are identical.
        int groupsNb = pConfigNow->GetNumberOfGroups(true);
        if (groupsNb != pConfigRef->GetNumberOfGroups(true)) {
            wxLogError(_("The number of groups (%d) differ from the previous config file (%d)."), groupsNb,
                       int(pConfigRef->GetNumberOfGroups()));
            return false;
        }

        // We only compare the content of the Calibration group.
        pConfigNow->SetPath("Optimizer");
        pConfigRef->SetPath("Optimizer");

        wxString subGroupName;
        long subGroupIndex;

        if (pConfigNow->GetFirstGroup(subGroupName, subGroupIndex)) {
            do {
                pConfigNow->SetPath(subGroupName);
                pConfigRef->SetPath(subGroupName);

                wxString entryName;
                long entryIndex;

                if (pConfigNow->GetFirstEntry(entryName, entryIndex)) {
                    do {
                        wxString valRef, valNow;
                        pConfigNow->Read(entryName, &valNow);
                        pConfigRef->Read(entryName, &valRef);

                        if (!valNow.IsEmpty() && !valNow.IsSameAs(valRef)) {
                            wxLogError(_("The option %s (under Optimizer/%s) differ from the previous config file (%s "
                                         "!= %s)."),
                                       entryName, subGroupName, valNow, valRef);
                            return false;
                        }
                    } while (pConfigNow->GetNextEntry(entryName, entryIndex));
                }

                pConfigNow->SetPath("..");
                pConfigRef->SetPath("..");
            } while (pConfigNow->GetNextGroup(subGroupName, subGroupIndex));
        }

        wxDELETE(pConfigRef);
    }

    return true;
}

void AtmoswingAppOptimizer::OnInitCmdLine(wxCmdLineParser& parser) {
    wxAppConsole::OnInitCmdLine(parser);

    parser.SetDesc(g_cmdLineDesc);
    parser.SetLogo(cmdLineLogo);

    // Must refuse '/' as parameter starter or cannot use "/path" style paths
    parser.SetSwitchChars(wxT("-"));
}

bool AtmoswingAppOptimizer::OnCmdLineParsed(wxCmdLineParser& parser) {
    // Check if runs with GUI or CL
    if (parser.Found("calibration-method")) {
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
            localPath.Append(wxString::Format("%d", g_runNb));
            localPath.Append(DS);

            // Check if path already exists
            if (wxFileName::Exists(localPath)) {
                g_resumePreviousRun = true;
            } else {
                // Create directory
                wxFileName userDir = wxFileName::DirName(localPath);
                userDir.Mkdir(wxS_DIR_DEFAULT, wxPATH_MKDIR_FULL);
            }
        }

        // Create local ini file
        wxString iniPath = localPath;
        iniPath.Append("AtmoSwing.ini");
        if (g_resumePreviousRun) {
            int increment = 1;
            while (wxFileName::Exists(iniPath)) {
                increment++;
                iniPath = localPath;
                iniPath.Append(wxString::Format("AtmoSwing-%d.ini", increment));
            }
        }

        // Set the local config object
        wxFileConfig* pConfig = new wxFileConfig("AtmoSwing", wxEmptyString, iniPath, iniPath, wxCONFIG_USE_LOCAL_FILE);
        wxFileConfig::Set(pConfig);
    } else {
        // Create user directory
        wxFileName userDir = wxFileName::DirName(asConfig::GetUserDataDir());
        userDir.Mkdir(wxS_DIR_DEFAULT, wxPATH_MKDIR_FULL);

        // Set the local config object
        wxFileConfig* pConfig = new wxFileConfig(
            "AtmoSwing", wxEmptyString, asConfig::GetUserDataDir() + "AtmoSwingOptimizer.ini",
            asConfig::GetUserDataDir() + "AtmoSwingOptimizer.ini", wxCONFIG_USE_LOCAL_FILE);
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
        asLog::PrintToConsole(wxString::Format("AtmoSwing version %s, %s\n", g_version, date));

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
        if (logLevel >= 1 && logLevel <= 4) {
            Log()->SetLevel((int)logLevel);
        } else {
            Log()->SetLevel(2);
        }
    } else {
        Log()->SetLevel(wxFileConfig::Get()->ReadLong("/General/LogLevel", 2l));
    }

    // Number of threads
    wxString threadsNb = wxEmptyString;
    if (parser.Found("threads-nb", &threadsNb)) {
        wxFileConfig::Get()->Write("/Processing/ThreadsNb", threadsNb);
    }

    // Number of gpus
    wxString gpusNb = wxEmptyString;
    if (parser.Found("gpus-nb", &gpusNb)) {
        wxFileConfig::Get()->Write("/Processing/GpusNb", gpusNb);
    }

    // Check for a calibration params file
    if (parser.Found("file-parameters", &m_calibParamsFile)) {
        if (g_local) {
            m_calibParamsFile = wxFileName::GetCwd() + DS + m_calibParamsFile;
        }

        if (!wxFileName::FileExists(m_calibParamsFile)) {
            wxLogError(_("The given calibration file (%s) couldn't be found."), m_calibParamsFile);
            return false;
        }
    }

    // Check for a calibration predictand DB
    if (parser.Found("predictand-db", &m_predictandDB)) {
        if (g_local) {
            m_predictandDB = wxFileName::GetCwd() + DS + m_predictandDB;
        }

        if (!wxFileName::FileExists(m_predictandDB)) {
            wxLogError(_("The given predictand DB (%s) couldn't be found."), m_predictandDB);
            return false;
        }
    }

    // Check for a predictors directory
    if (parser.Found("dir-predictors", &m_predictorsDir)) {
        if (g_local && wxFileName::Exists(wxFileName::GetCwd() + DS + m_predictorsDir)) {
            m_predictorsDir = wxFileName::GetCwd() + DS + m_predictorsDir;
        }

        if (!wxFileName::DirExists(m_predictorsDir)) {
            wxLogError(_("The given predictors directory (%s) couldn't be found."), m_predictorsDir);
            return false;
        }
    }

    // Skip validation option
    wxString optionValid = wxEmptyString;
    if (parser.Found("skip-valid", &optionValid)) {
        wxFileConfig::Get()->Write("/SkipValidation", optionValid);
    }

    // Station ID
    wxString stationIdStr = wxEmptyString;
    if (parser.Found("station-id", &stationIdStr)) {
        m_predictandStationIds = asParameters::GetFileStationIds(stationIdStr);
    }

    // Flag to disable the duplicate dates
    if (parser.Found("no-duplicate-dates")) {
        wxFileConfig::Get()->Write("/Processing/AllowDuplicateDates", false);
    }

    // Memory option
    if (parser.Found("dump-predictor-data")) {
        wxFileConfig::Get()->Write("/General/DumpPredictorData", true);
    }

    // Load from dumped memory files
    if (parser.Found("load-from-dumped-data")) {
        wxFileConfig::Get()->Write("/General/LoadDumpedData", true);
    }

    // Replace NaNs with another value
    if (parser.Found("replace-nans")) {
        wxFileConfig::Get()->Write("/General/ReplaceNans", true);
        wxFileConfig::Get()->Write("/General/SkipNansCheck", true);
    }

    if (parser.Found("skip-nans-check")) {
        wxFileConfig::Get()->Write("/General/SkipNansCheck", true);
    }

    /*
     * Methods options
     */

    wxString option = wxEmptyString;

    // Classic+ calibration
    if (parser.Found("cp-resizing-iteration", &option)) {
        wxFileConfig::Get()->Write("/ClassicPlus/ResizingIterations", option);
    }

    if (parser.Found("cp-lat-step", &option)) {
        wxFileConfig::Get()->Write("/ClassicPlus/StepsLatPertinenceMap", option);
    }

    if (parser.Found("cp-lon-step", &option)) {
        wxFileConfig::Get()->Write("/ClassicPlus/StepsLonPertinenceMap", option);
    }

    if (parser.Found("cp-proceed-sequentially", &option)) {
        wxFileConfig::Get()->Write("/ClassicPlus/ProceedSequentially", option);
    }

    // Variables exploration
    if (parser.Found("ve-step", &option)) {
        wxFileConfig::Get()->Write("/VariablesExplo/Step", option);
    }

    // Monte Carlo
    if (parser.Found("mc-runs-nb", &option)) {
        wxFileConfig::Get()->Write("/MonteCarlo/RandomNb", option);
    }

    // Genetic algorithms
    if (parser.Found("ga-config", &option)) {
        long gaConfig = -1;
        if (!option.ToLong(&gaConfig)) {
            asLog::PrintToConsole(_("The value provided for 'ga-config' could not be interpreted.\n"));
            return false;
        }

        wxFileConfig::Get()->Write("/GAs/ConvergenceStepsNb", 30);
        wxFileConfig::Get()->Write("/GAs/PopulationSize", 500);
        wxFileConfig::Get()->Write("/GAs/RatioIntermediateGeneration", 0.5);
        wxFileConfig::Get()->Write("/GAs/NaturalSelectionOperator", 0);
        wxFileConfig::Get()->Write("/GAs/CouplesSelectionOperator", 2);
        wxFileConfig::Get()->Write("/GAs/CrossoverOperator", 7);
        wxFileConfig::Get()->Write("/GAs/CrossoverBinaryLikePointsNb", 2);
        wxFileConfig::Get()->Write("/GAs/CrossoverBinaryLikeShareBeta", 1);

        switch (gaConfig) {
            case 1:
                wxFileConfig::Get()->Write("/GAs/MutationOperator", 8);
                break;
            case 2:
                wxFileConfig::Get()->Write("/GAs/MutationOperator", 9);
                wxFileConfig::Get()->Write("/GAs/MutationsMultiScaleProbability", 0.1);
                break;
            case 3:
                wxFileConfig::Get()->Write("/GAs/MutationOperator", 4);
                wxFileConfig::Get()->Write("/GAs/MutationsNonUniformProbability", 0.1);
                wxFileConfig::Get()->Write("/GAs/MutationsNonUniformMaxGensNbVar", 50);
                wxFileConfig::Get()->Write("/GAs/MutationsNonUniformMinRate", 0.1);
                break;
            case 4:
                wxFileConfig::Get()->Write("/GAs/MutationOperator", 4);
                wxFileConfig::Get()->Write("/GAs/MutationsNonUniformProbability", 0.1);
                wxFileConfig::Get()->Write("/GAs/MutationsNonUniformMaxGensNbVar", 100);
                wxFileConfig::Get()->Write("/GAs/MutationsNonUniformMinRate", 0.1);
                break;
            case 5:
                wxFileConfig::Get()->Write("/GAs/MutationOperator", 4);
                wxFileConfig::Get()->Write("/GAs/MutationsNonUniformProbability", 0.2);
                wxFileConfig::Get()->Write("/GAs/MutationsNonUniformMaxGensNbVar", 100);
                wxFileConfig::Get()->Write("/GAs/MutationsNonUniformMinRate", 0.1);
                break;
            default:
                asLog::PrintToConsole(_("The value provided for 'ga-config' does not match any option.\n"));
                return false;
        }
    }

    if (parser.Found("ga-use-mini-batches", &option)) {
        wxFileConfig::Get()->Write("/GAs/UseMiniBatches", option);
    }

    if (parser.Found("ga-mini-batch-size", &option)) {
        wxFileConfig::Get()->Write("/GAs/MiniBatchSize", option);
    }

    if (parser.Found("ga-number-epochs", &option)) {
        wxFileConfig::Get()->Write("/GAs/NumberOfEpochs", option);
    }

    if (parser.Found("ga-ope-nat-sel", &option)) {
        wxFileConfig::Get()->Write("/GAs/NaturalSelectionOperator", option);
    }

    if (parser.Found("ga-ope-coup-sel", &option)) {
        wxFileConfig::Get()->Write("/GAs/CouplesSelectionOperator", option);
    }

    if (parser.Found("ga-ope-cross", &option)) {
        wxFileConfig::Get()->Write("/GAs/CrossoverOperator", option);
    }

    if (parser.Found("ga-ope-mut", &option)) {
        wxFileConfig::Get()->Write("/GAs/MutationOperator", option);
    }

    if (parser.Found("ga-pop-size", &option)) {
        wxFileConfig::Get()->Write("/GAs/PopulationSize", option);
    }

    if (parser.Found("ga-conv-steps", &option)) {
        wxFileConfig::Get()->Write("/GAs/ConvergenceStepsNb", option);
    }

    if (parser.Found("ga-interm-gen", &option)) {
        wxFileConfig::Get()->Write("/GAs/RatioIntermediateGeneration", option);
    }

    if (parser.Found("ga-nat-sel-tour-p", &option)) {
        wxFileConfig::Get()->Write("/GAs/NaturalSelectionTournamentProbability", option);
    }

    if (parser.Found("ga-coup-sel-tour-nb", &option)) {
        wxFileConfig::Get()->Write("/GAs/CouplesSelectionTournamentNb", option);
    }

    if (parser.Found("ga-cross-mult-pt-nb", &option)) {
        wxFileConfig::Get()->Write("/GAs/CrossoverMultiplePointsNb", option);
    }

    if (parser.Found("ga-cross-blen-pt-nb", &option)) {
        wxFileConfig::Get()->Write("/GAs/CrossoverBlendingPointsNb", option);
    }

    if (parser.Found("ga-cross-blen-share-b", &option)) {
        wxFileConfig::Get()->Write("/GAs/CrossoverBlendingShareBeta", option);
    }

    if (parser.Found("ga-cross-lin-pt-nb", &option)) {
        wxFileConfig::Get()->Write("/GAs/CrossoverLinearPointsNb", option);
    }

    if (parser.Found("ga-cross-heur-pt-nb", &option)) {
        wxFileConfig::Get()->Write("/GAs/CrossoverHeuristicPointsNb", option);
    }

    if (parser.Found("ga-cross-heur-share-b", &option)) {
        wxFileConfig::Get()->Write("/GAs/CrossoverHeuristicShareBeta", option);
    }

    if (parser.Found("ga-cross-bin-pt-nb", &option)) {
        wxFileConfig::Get()->Write("/GAs/CrossoverBinaryLikePointsNb", option);
    }

    if (parser.Found("ga-cross-bin-share-b", &option)) {
        wxFileConfig::Get()->Write("/GAs/CrossoverBinaryLikeShareBeta", option);
    }

    if (parser.Found("ga-mut-unif-cst-p", &option)) {
        wxFileConfig::Get()->Write("/GAs/MutationsUniformConstantProbability", option);
    }

    if (parser.Found("ga-mut-norm-cst-p", &option)) {
        wxFileConfig::Get()->Write("/GAs/MutationsNormalConstantProbability", option);
    }

    if (parser.Found("ga-mut-norm-cst-dev", &option)) {
        wxFileConfig::Get()->Write("/GAs/MutationsNormalConstantStdDevRatioRange", option);
    }

    if (parser.Found("ga-mut-unif-var-gens", &option)) {
        wxFileConfig::Get()->Write("/GAs/MutationsUniformVariableMaxGensNbVar", option);
    }

    if (parser.Found("ga-mut-unif-var-p-strt", &option)) {
        wxFileConfig::Get()->Write("/GAs/MutationsUniformVariableProbabilityStart", option);
    }

    if (parser.Found("ga-mut-unif-var-p-end", &option)) {
        wxFileConfig::Get()->Write("/GAs/MutationsUniformVariableProbabilityEnd", option);
    }

    if (parser.Found("ga-mut-norm-var-gens-p", &option)) {
        wxFileConfig::Get()->Write("/GAs/MutationsNormalVariableMaxGensNbVarProb", option);
    }

    if (parser.Found("ga-mut-norm-var-gens-d", &option)) {
        wxFileConfig::Get()->Write("/GAs/MutationsNormalVariableMaxGensNbVarStdDev", option);
    }

    if (parser.Found("ga-mut-norm-var-p-strt", &option)) {
        wxFileConfig::Get()->Write("/GAs/MutationsNormalVariableProbabilityStart", option);
    }

    if (parser.Found("ga-mut-norm-var-p-end", &option)) {
        wxFileConfig::Get()->Write("/GAs/MutationsNormalVariableProbabilityEnd", option);
    }

    if (parser.Found("ga-mut-norm-var-d-strt", &option)) {
        wxFileConfig::Get()->Write("/GAs/MutationsNormalVariableStdDevStart", option);
    }

    if (parser.Found("ga-mut-norm-var-d-end", &option)) {
        wxFileConfig::Get()->Write("/GAs/MutationsNormalVariableStdDevEnd", option);
    }

    if (parser.Found("ga-mut-non-uni-p", &option)) {
        wxFileConfig::Get()->Write("/GAs/MutationsNonUniformProbability", option);
    }

    if (parser.Found("ga-mut-non-uni-gens", &option)) {
        wxFileConfig::Get()->Write("/GAs/MutationsNonUniformMaxGensNbVar", option);
    }

    if (parser.Found("ga-mut-non-uni-min-r", &option)) {
        wxFileConfig::Get()->Write("/GAs/MutationsNonUniformMinRate", option);
    }

    if (parser.Found("ga-mut-multi-scale-p", &option)) {
        wxFileConfig::Get()->Write("/GAs/MutationsMultiScaleProbability", option);
    }

    /*
     * Method choice
     */

    // Check for a calibration method option
    if (parser.Found("calibration-method", &m_calibMethod)) {
        if (!InitForCmdLineOnly()) {
            wxLogError(_("Initialization for command-line interface failed."));
            return false;
        }
        m_doProcessing = true;
        wxLogVerbose(_("Given calibration method: %s"), m_calibMethod);

        return true;
    }

    // Finally, if no option is given in CL mode, display help.
    if (!g_guiMode) {
        parser.Usage();

        return false;
    }

    return true;
}

int AtmoswingAppOptimizer::OnRun() {
    if (g_guiMode) {
        return wxApp::OnRun();
    }

    if (!m_doProcessing) {
        return 0;
    }

    if (m_calibParamsFile.IsEmpty()) {
        wxLogError(_("The parameters file is not given."));
        return 1;
    }

    if (m_predictandDB.IsEmpty()) {
        wxLogError(_("The predictand DB is not given."));
        return 1;
    }

    if (m_predictorsDir.IsEmpty()) {
        wxLogError(_("The predictors directory is not given."));
        return 1;
    }

    try {
        if (m_calibMethod.IsSameAs("single", false)) {
            asMethodCalibratorSingle calibrator;
            calibrator.SetParamsFilePath(m_calibParamsFile);
            calibrator.SetPredictandDBFilePath(m_predictandDB);
            calibrator.SetPredictandStationIds(m_predictandStationIds);
            calibrator.SetPredictorDataDir(m_predictorsDir);
            calibrator.Manager();
        } else if (m_calibMethod.IsSameAs("classic", false)) {
            asMethodCalibratorClassic calibrator;
            calibrator.SetParamsFilePath(m_calibParamsFile);
            calibrator.SetPredictandDBFilePath(m_predictandDB);
            calibrator.SetPredictandStationIds(m_predictandStationIds);
            calibrator.SetPredictorDataDir(m_predictorsDir);
            calibrator.Manager();
        } else if (m_calibMethod.IsSameAs("classicp", false)) {
            asMethodCalibratorClassic calibrator;
            calibrator.SetAsCalibrationPlus();
            calibrator.SetParamsFilePath(m_calibParamsFile);
            calibrator.SetPredictandDBFilePath(m_predictandDB);
            calibrator.SetPredictandStationIds(m_predictandStationIds);
            calibrator.SetPredictorDataDir(m_predictorsDir);
            calibrator.Manager();
        } else if (m_calibMethod.IsSameAs("varexplocp", false)) {
            asMethodCalibratorClassicVarExplo calibrator;
            calibrator.SetAsCalibrationPlus();
            calibrator.SetParamsFilePath(m_calibParamsFile);
            calibrator.SetPredictandDBFilePath(m_predictandDB);
            calibrator.SetPredictandStationIds(m_predictandStationIds);
            calibrator.SetPredictorDataDir(m_predictorsDir);
            calibrator.Manager();
        } else if (m_calibMethod.IsSameAs("montecarlo", false)) {
            asMethodOptimizerRandomSet optimizer;
            optimizer.SetParamsFilePath(m_calibParamsFile);
            optimizer.SetPredictandDBFilePath(m_predictandDB);
            optimizer.SetPredictandStationIds(m_predictandStationIds);
            optimizer.SetPredictorDataDir(m_predictorsDir);
            optimizer.Manager();
        } else if (m_calibMethod.IsSameAs("ga", false)) {
            asMethodOptimizerGeneticAlgorithms optimizer;
            optimizer.SetParamsFilePath(m_calibParamsFile);
            optimizer.SetPredictandDBFilePath(m_predictandDB);
            optimizer.SetPredictandStationIds(m_predictandStationIds);
            optimizer.SetPredictorDataDir(m_predictorsDir);
            optimizer.Manager();
        } else if (m_calibMethod.IsSameAs("evalscores", false)) {
            asMethodCalibratorEvaluateAllScores calibrator;
            calibrator.SetParamsFilePath(m_calibParamsFile);
            calibrator.SetPredictandDBFilePath(m_predictandDB);
            calibrator.SetPredictandStationIds(m_predictandStationIds);
            calibrator.SetPredictorDataDir(m_predictorsDir);
            calibrator.Manager();
        } else if (m_calibMethod.IsSameAs("onlyvalues", false)) {
            asMethodCalibratorSingleOnlyValues calibrator;
            calibrator.SetParamsFilePath(m_calibParamsFile);
            calibrator.SetPredictandDBFilePath(m_predictandDB);
            calibrator.SetPredictandStationIds(m_predictandStationIds);
            calibrator.SetPredictorDataDir(m_predictorsDir);
            calibrator.Manager();
        } else if (m_calibMethod.IsSameAs("onlydates", false)) {
            asMethodCalibratorSingleOnlyDates calibrator;
            calibrator.SetParamsFilePath(m_calibParamsFile);
            calibrator.SetPredictorDataDir(m_predictorsDir);
            calibrator.Manager();
        } else {
            asLog::PrintToConsole(wxString::Format("Wrong calibration method selection (%s).\n", m_calibMethod));
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

    wxLogMessage(_("Calibration over."));
    asLog::PrintToConsole(_("Calibration over.\n"));

    return 0;
}

int AtmoswingAppOptimizer::OnExit() {
    CleanUp();

    return 0;
}

void AtmoswingAppOptimizer::CleanUp() {
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

bool AtmoswingAppOptimizer::OnExceptionInMainLoop() {
    wxLogError(_("An exception occured in the main loop"));
    return false;
}

void AtmoswingAppOptimizer::OnFatalException() {
    wxLogError(_("An fatal exception occured"));
}

void AtmoswingAppOptimizer::OnUnhandledException() {
    wxLogError(_("An unhandled exception occured"));
}
