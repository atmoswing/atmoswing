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
 * Portions Copyright 2013 Pascal Horton, Terr@num.
 */

#ifdef WX_PRECOMP
#include "wx_pch.h"
#endif

#ifdef __BORLANDC__
#pragma hdrstop
#endif //__BORLANDC__

#include "AtmoswingAppCalibrator.h"
#if wxUSE_GUI
    #include "AtmoswingMainCalibrator.h"
#endif
#include "asMethodCalibratorClassicPlus.h"
#include "asMethodCalibratorClassicPlusVarExplo.h"
#include "asMethodOptimizerRandomSet.h"
#include "asMethodOptimizerGeneticAlgorithms.h"
#include "asMethodCalibratorEvaluateAllScores.h"


IMPLEMENT_APP(AtmoswingAppCalibrator);

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

static const wxCmdLineEntryDesc g_cmdLineDesc[] =
{
    { wxCMD_LINE_SWITCH, "v", "version", "Show version number and quit" },
    { wxCMD_LINE_SWITCH, "s", "silent", "Silent mode" },
    { wxCMD_LINE_SWITCH, "l", "local", "Work in local directory" },
    { wxCMD_LINE_OPTION, "n", "threads-nb", "Number of threads to use" },
    { wxCMD_LINE_OPTION, "r", "run-number", "Choice of number associated with the run" },
	{ wxCMD_LINE_OPTION, "f", "file-parameters", "File containing the calibration parameters" },
	{ wxCMD_LINE_OPTION, NULL, "predictand-db", "The predictand DB" },
    { wxCMD_LINE_OPTION, NULL, "station-id", "The predictand station ID" },
	{ wxCMD_LINE_OPTION, NULL, "dir-predictors", "The predictors directory" },
	{ wxCMD_LINE_OPTION, NULL, "skip-valid", "Skip the validation calculation" },
	{ wxCMD_LINE_OPTION, NULL, "calibration-method", "Choice of the calibration method"
                                "\n \t\t\t\t single: single assessment"
                                "\n \t\t\t\t classic: classic calibration"
                                "\n \t\t\t\t classicp: classic+ calibration"
                                "\n \t\t\t\t varexplocp: variables exploration classic+"
                                "\n \t\t\t\t evalscores: evaluate all scores" },
    { wxCMD_LINE_OPTION, NULL, "cp-resizing-iteration", "Classic plus: resizing iteration" },
	{ wxCMD_LINE_OPTION, NULL, "cp-lat-step", "Classic plus: steps in latitudes for the relevance map" },
	{ wxCMD_LINE_OPTION, NULL, "cp-lon-step", "Classic plus: steps in longitudes for the relevance map" },
	{ wxCMD_LINE_OPTION, NULL, "cp-proceed-sequentially", "Classic plus: proceed sequentially" },
	{ wxCMD_LINE_OPTION, NULL, "ve-step", "Variables exploration: step" },
    { wxCMD_LINE_OPTION, NULL, "mc-runs-nb", "Monte Carlo: options MonteCarloRandomNb" },
    { wxCMD_LINE_OPTION, NULL, "ga-ope-nat-sel", "GAs: options NaturalSelectionOperator" },
    { wxCMD_LINE_OPTION, NULL, "ga-ope-coup-sel", "GAs: options CouplesSelectionOperator" },
    { wxCMD_LINE_OPTION, NULL, "ga-ope-cross", "GAs: options CrossoverOperator" },
    { wxCMD_LINE_OPTION, NULL, "ga-ope-mut", "GAs: options MutationOperator" },
    { wxCMD_LINE_OPTION, NULL, "ga-pop-size", "GAs: options GAsPopulationSize" },
    { wxCMD_LINE_OPTION, NULL, "ga-conv-steps", "GAs: options GAsConvergenceStepsNb" },
    { wxCMD_LINE_OPTION, NULL, "ga-interm-gen", "GAs: options GAsRatioIntermediateGeneration" },
    { wxCMD_LINE_OPTION, NULL, "ga-nat-sel-tour-p", "GAs: options GAsNaturalSelectionTournamentProbability" },
    { wxCMD_LINE_OPTION, NULL, "ga-coup-sel-tour-nb", "GAs: options GAsCouplesSelectionTournamentNb" },
    { wxCMD_LINE_OPTION, NULL, "ga-cross-mult-pt-nb", "GAs: options GAsCrossoverMultiplePointsNb" },
    { wxCMD_LINE_OPTION, NULL, "ga-cross-blen-pt-nb", "GAs: options GAsCrossoverBlendingPointsNb" },
    { wxCMD_LINE_OPTION, NULL, "ga-cross-blen-share-b", "GAs: options GAsCrossoverBlendingShareBeta" },
    { wxCMD_LINE_OPTION, NULL, "ga-cross-lin-pt-nb", "GAs: options GAsCrossoverLinearPointsNb" },
    { wxCMD_LINE_OPTION, NULL, "ga-cross-heur-pt-nb", "GAs: options GAsCrossoverHeuristicPointsNb" },
    { wxCMD_LINE_OPTION, NULL, "ga-cross-heur-share-b", "GAs: options GAsCrossoverHeuristicShareBeta" },
    { wxCMD_LINE_OPTION, NULL, "ga-cross-bin-pt-nb", "GAs: options GAsCrossoverBinaryLikePointsNb" },
    { wxCMD_LINE_OPTION, NULL, "ga-cross-bin-share-b", "GAs: options GAsCrossoverBinaryLikeShareBeta" },
    { wxCMD_LINE_OPTION, NULL, "ga-mut-unif-cst-p", "GAs: options GAsMutationsUniformConstantProbability" },
    { wxCMD_LINE_OPTION, NULL, "ga-mut-norm-cst-p", "GAs: options GAsMutationsNormalConstantProbability" },
    { wxCMD_LINE_OPTION, NULL, "ga-mut-norm-cst-dev", "GAs: options GAsMutationsNormalConstantStdDevRatioRange" },
    { wxCMD_LINE_OPTION, NULL, "ga-mut-unif-var-gens", "GAs: options GAsMutationsUniformVariableMaxGensNbVar" },
    { wxCMD_LINE_OPTION, NULL, "ga-mut-unif-var-p-strt", "GAs: options GAsMutationsUniformVariableProbabilityStart" },
    { wxCMD_LINE_OPTION, NULL, "ga-mut-unif-var-p-end", "GAs: options GAsMutationsUniformVariableProbabilityEnd" },
    { wxCMD_LINE_OPTION, NULL, "ga-mut-norm-var-gens-p", "GAs: options GAsMutationsNormalVariableMaxGensNbVarProb" },
    { wxCMD_LINE_OPTION, NULL, "ga-mut-norm-var-gens-d", "GAs: options GAsMutationsNormalVariableMaxGensNbVarStdDev" },
    { wxCMD_LINE_OPTION, NULL, "ga-mut-norm-var-p-strt", "GAs: options GAsMutationsNormalVariableProbabilityStart" },
    { wxCMD_LINE_OPTION, NULL, "ga-mut-norm-var-p-end", "GAs: options GAsMutationsNormalVariableProbabilityEnd" },
    { wxCMD_LINE_OPTION, NULL, "ga-mut-norm-var-d-strt", "GAs: options GAsMutationsNormalVariableStdDevStart" },
    { wxCMD_LINE_OPTION, NULL, "ga-mut-norm-var-d-end", "GAs: options GAsMutationsNormalVariableStdDevEnd" },
    { wxCMD_LINE_OPTION, NULL, "ga-mut-non-uni-p", "GAs: options GAsMutationsNonUniformProb" },
    { wxCMD_LINE_OPTION, NULL, "ga-mut-non-uni-gens", "GAs: options GAsMutationsNonUniformMaxGensNbVar" },
    { wxCMD_LINE_OPTION, NULL, "ga-mut-non-uni-min-r", "GAs: options GAsMutationsNonUniformMinRate" },
    { wxCMD_LINE_OPTION, NULL, "ga-mut-multi-scale-p", "GAs: options GAsMutationsMultiScaleProb" },

	{ wxCMD_LINE_OPTION, NULL, "log-level", "Set a log level"
		"\n \t\t\t\t 0: minimum"
		"\n \t\t\t\t 1: errors"
		"\n \t\t\t\t 2: warnings"
		"\n \t\t\t\t 3: verbose" },
	{ wxCMD_LINE_OPTION, NULL, "log-target", "Set log target"
		"\n \t\t\t\t file: file only"
		"\n \t\t\t\t prompt: command prompt"
		"\n \t\t\t\t both: command prompt and file (default)" },

    { wxCMD_LINE_NONE }
};

bool AtmoswingAppCalibrator::OnInit()
{
    #if _DEBUG
        #ifdef __WXMSW__
            _CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
        #endif
    #endif

    // Set application name
    wxString appName = "AtmoSwing Calibrator";
    wxApp::SetAppName(appName);

    g_guiMode = true;
    g_local = false;
    m_calibParamsFile = wxEmptyString;
    m_predictandDB = wxEmptyString;
    m_predictandStationIds = VectorInt(0);
    m_predictorsDir = wxEmptyString;
    m_calibMethod = wxEmptyString;
    m_forceQuit = false;

    // Call default behaviour (mandatory for command-line mode)
    if (!wxApp::OnInit()) // When false, we are in CL mode
        return false;

    #if wxUSE_GUI

		// Set PPI
		wxMemoryDC dcTestPpi;
		wxSize ppiDC = dcTestPpi.GetPPI();
		g_ppiScaleDc = double(ppiDC.x) / 96.0;

	    m_singleInstanceChecker = NULL;
        if (g_guiMode)
        {
            // Check that it is the unique instance
            bool multipleInstances = false;

            wxFileConfig::Get()->Read("/General/MultiInstances", &multipleInstances, false);

            if (!multipleInstances)
            {
                const wxString instanceName = wxString::Format(wxT("atmoswing-calibrator-%s"),wxGetUserId());
                m_singleInstanceChecker = new wxSingleInstanceChecker(instanceName);
                if ( m_singleInstanceChecker->IsAnotherRunning() )
                {
                    wxMessageBox(_("Program already running, aborting."));
                    return false;
                }
            }

            // Following for GUI only
            wxInitAllImageHandlers();

            // Initialize images
			initialize_images(g_ppiScaleDc);

            // Create frame
            AtmoswingFrameCalibrator* frame = new AtmoswingFrameCalibrator(0L);
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

bool AtmoswingAppCalibrator::InitForCmdLineOnly()
{
    g_guiMode = false;
    g_unitTesting = false;
    g_silentMode = true;
    g_verboseMode = false;
    g_responsive = false;

    // Prepare local path
    wxString localPath = wxFileName::GetCwd() + DS;
    if (g_runNb>0)
    {
        localPath.Append("runs");
        localPath.Append(DS);
        localPath.Append(wxString::Format("%d", g_runNb));
        localPath.Append(DS);
    }

    if (g_local)
    {
        wxString fullPath = localPath;
        fullPath.Append("AtmoSwingCalibrator.log");
        if (g_resumePreviousRun)
        {
            int increment = 1;
            while (wxFileName::Exists(fullPath))
            {
                increment++;
                fullPath = localPath;
                fullPath.Append(wxString::Format("AtmoSwingCalibrator-%d.log", increment));
            }
        }

        Log().CreateFileOnlyAtPath(fullPath);
    }
    else
    {
        Log().CreateFileOnly("AtmoSwingCalibrator.log");
    }

    Log().DisableMessageBoxOnError();

    // Warn the user if reloading previous results
    if (g_resumePreviousRun)
    {
        asLogWarning(wxString::Format(_("An existing directory was found for the run number %d"), g_runNb));
        printf("Warning: An existing directory was found for the run number %d\n", g_runNb);
    }

    if (g_local)
    {
        asLogWarning(wxString::Format(_("An existing directory was found for the run number %d"), g_runNb));
        printf("Warning: An existing directory was found for the run number %d\n", g_runNb);
    }

    {
        wxString dirData = wxFileName::GetCwd()+DS+"data"+DS;

        wxConfigBase *pConfig = wxFileConfig::Get();

        // Define the default preferences
        pConfig->Write("/General/MultiInstances", true);
        pConfig->Write("/General/GuiOptions", 0l);
        pConfig->Write("/General/Responsive", false);
        pConfig->Write("/General/DisplayLogWindow", false);
        pConfig->Write("/Paths/DataPredictandDBDir", dirData);
        pConfig->Write("/Paths/IntermediateResultsDir", localPath+"temp");
        pConfig->Write("/Paths/CalibrationResultsDir", localPath+"results");
        pConfig->Write("/Paths/ArchivePredictorsDir", dirData);
        pConfig->Write("/Processing/AllowMultithreading", false); // Because we are using parallel evaluations
        pConfig->Write("/Processing/Method", (long)asMULTITHREADS);
        pConfig->Write("/Processing/LinAlgebra", (long)asLIN_ALGEBRA_NOVAR);
        pConfig->Write("/Processing/ThreadsPriority", 100);
        pConfig->Write("/Calibration/ParallelEvaluations", true);
        pConfig->Write("/Calibration/GeneticAlgorithms/AllowElitismForTheBest", true);

        pConfig->Flush();

    }

    // Check that the config files correspond if reloading data
    if (g_resumePreviousRun)
    {
        wxConfigBase *pConfigNow = wxFileConfig::Get();
        wxString refIniPath = localPath;
        refIniPath.Append("AtmoSwing.ini");
        wxFileConfig *pConfigRef = new wxFileConfig("AtmoSwing",wxEmptyString,refIniPath,refIniPath,wxCONFIG_USE_LOCAL_FILE);

        // Check that the number of groups are identical.
        int groupsNb = pConfigNow->GetNumberOfGroups(true);
        if (groupsNb != pConfigRef->GetNumberOfGroups(true))
        {
            asLogError(wxString::Format(_("The number of groups (%d) differ from the previous config file (%d)."), groupsNb, int(pConfigRef->GetNumberOfGroups())));
            m_forceQuit = true;
        }

        // We only compare the content of the Calibration group.
        pConfigNow->SetPath("Calibration");
        pConfigRef->SetPath("Calibration");

        wxString subGroupName;
        long subGroupIndex;

        if (pConfigNow->GetFirstGroup(subGroupName, subGroupIndex))
        {
            do
            {
                pConfigNow->SetPath(subGroupName);
                pConfigRef->SetPath(subGroupName);

                wxString entryName;
                long entryIndex;

                if (pConfigNow->GetFirstEntry(entryName, entryIndex))
                {
                    do
                    {
                        wxString valRef, valNow;
                        pConfigNow->Read(entryName, &valNow);
                        pConfigRef->Read(entryName, &valRef);

                        if (!valNow.IsEmpty() && !valNow.IsSameAs(valRef))
                        {
                            asLogError(wxString::Format(_("The option %s (under Calibration/%s) differ from the previous config file (%s != %s)."),
                                                        entryName.c_str(), subGroupName.c_str(), valNow.c_str(), valRef.c_str()));
                            m_forceQuit = true;
                        }
                    }
                    while (pConfigNow->GetNextEntry(entryName, entryIndex));
                }

                pConfigNow->SetPath("..");
                pConfigRef->SetPath("..");
            }
            while (pConfigNow->GetNextGroup(subGroupName, subGroupIndex));
        }

        wxDELETE(pConfigRef);
    }

    return true;
}

void AtmoswingAppCalibrator::OnInitCmdLine(wxCmdLineParser& parser)
{
    wxAppConsole::OnInitCmdLine(parser);

    // From http://wiki.wxwidgets.org/Command-Line_Arguments
    parser.SetDesc (g_cmdLineDesc);
    // Must refuse '/' as parameter starter or cannot use "/path" style paths
    parser.SetSwitchChars (wxT("-"));
}

bool AtmoswingAppCalibrator::OnCmdLineParsed(wxCmdLineParser& parser)
{
    // From http://wiki.wxwidgets.org/Command-Line_Arguments

    /*
     * General options
     */

    // Check if the user asked for command-line help
    if (parser.Found("help"))
    {
        parser.Usage();
        return false;
    }

    // Check if the user asked for the version
    if (parser.Found("version"))
    {
        wxMessageOutput* msgOut = wxMessageOutput::Get();
        if ( msgOut )
        {
            wxString msg;
            wxString date(wxString::FromAscii(__DATE__));
            msg.Printf("AtmoSwing version %s, %s", g_version, (const wxChar*) date);

            msgOut->Printf( msg );
        }
        else
        {
            wxFAIL_MSG( _("No wxMessageOutput object?") );
        }

        return false; // We don't want to continue
    }

    // Check for a run number
    wxString runNbStr = wxEmptyString;
    long runNb = 0;
    if (parser.Found("run-number", & runNbStr))
    {
        runNbStr.ToLong(&runNb);
        g_runNb = (int)runNb;
    }

    // Local mode
    if (parser.Found("local"))
    {
        g_local = true;
        wxString localPath = wxFileName::GetCwd() + DS;
        if (g_runNb>0)
        {
            localPath.Append("runs");
            localPath.Append(DS);
            localPath.Append(wxString::Format("%d", g_runNb));
            localPath.Append(DS);

            // Check if path already exists
            if (wxFileName::Exists(localPath))
            {
                g_resumePreviousRun = true;
            }
            else
            {
                // Create directory
                wxFileName userDir = wxFileName::DirName(localPath);
                userDir.Mkdir(wxS_DIR_DEFAULT,wxPATH_MKDIR_FULL);
            }
        }

        // Create local ini file
        wxString iniPath = localPath;
        iniPath.Append("AtmoSwing.ini");
        if (g_resumePreviousRun)
        {
            int increment = 1;
            while (wxFileName::Exists(iniPath))
            {
                increment++;
                iniPath = localPath;
                iniPath.Append(wxString::Format("AtmoSwing-%d.ini", increment));
            }
        }

        // Set the local config object
        wxFileConfig *pConfig = new wxFileConfig("AtmoSwing",wxEmptyString,iniPath,iniPath,wxCONFIG_USE_LOCAL_FILE);
        wxFileConfig::Set(pConfig);
    }
    else
    {
        // Create user directory
        wxFileName userDir = wxFileName::DirName(asConfig::GetUserDataDir());
        userDir.Mkdir(wxS_DIR_DEFAULT,wxPATH_MKDIR_FULL);

        // Set the local config object
        wxFileConfig *pConfig = new wxFileConfig("AtmoSwing",wxEmptyString,asConfig::GetUserDataDir()+"AtmoSwingCalibrator.ini",asConfig::GetUserDataDir()+"AtmoSwingCalibrator.ini",wxCONFIG_USE_LOCAL_FILE);
        wxFileConfig::Set(pConfig);
    }

    // Check for a log level option
    wxString logLevelStr = wxEmptyString;
    if (parser.Found("log-level", & logLevelStr))
    {
        long logLevel = -1;
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
        else if (logLevel==3)
        {
            Log().SetLevel(3);
        }
        else
        {
            Log().SetLevel(2);
        }
    }
    else
    {
        long logLevel = wxFileConfig::Get()->Read("/General/LogLevel", 2l);
        Log().SetLevel((int)logLevel);
    }

    // Check for the log target option
    wxString logTargetStr = wxEmptyString;
    if (parser.Found("log-target", & logTargetStr))
    {
        // Check and apply
        if (logTargetStr.IsSameAs("file", false))
        {
            Log().SetTarget(asLog::File);
        }
        else if (logTargetStr.IsSameAs("screen", false))
        {
            Log().SetTarget(asLog::Screen);
        }
        else if (logTargetStr.IsSameAs("both", false))
        {
            Log().SetTarget(asLog::Both);
        }
        else
        {
            Log().SetTarget(asLog::Both);

            wxMessageOutput* msgOut = wxMessageOutput::Get();
            if ( msgOut )
            {
                msgOut->Printf( _("The given log target (%s) does not correspond to any possible option."), logTargetStr );
            }
        }
    }

    // Check if the user asked for the silent mode
    if (parser.Found("silent"))
    {
        Log().SetTarget(asLog::File);
    }

    // Check for a calibration params file
    wxString threadsNb = wxEmptyString;
    if (parser.Found("threads-nb", & threadsNb))
    {
        wxFileConfig::Get()->Write("/Processing/MaxThreadNb", threadsNb);
    }

    // Check for a calibration params file
    if (parser.Found("file-parameters", & m_calibParamsFile))
    {
        if (g_local)
        {
            m_calibParamsFile = wxFileName::GetCwd() + DS + m_calibParamsFile;
        }

        if (!wxFileName::FileExists(m_calibParamsFile))
        {
            asLogError(wxString::Format(_("The given calibration file (%s) couldn't be found."), m_calibParamsFile));
            return false;
        }
    }

    // Check for a calibration predictand DB
    if (parser.Found("predictand-db", & m_predictandDB))
    {
        if (g_local)
        {
            m_predictandDB = wxFileName::GetCwd() + DS + m_predictandDB;
        }

        if (!wxFileName::FileExists(m_predictandDB))
        {
            asLogError(wxString::Format(_("The given predictand DB (%s) couldn't be found."), m_predictandDB));
            return false;
        }
    }

    // Check for a predictors directory
    if (parser.Found("dir-predictors", & m_predictorsDir))
    {
        if (g_local)
        {
            m_predictorsDir = wxFileName::GetCwd() + DS + m_predictorsDir;
        }

        if (!wxFileName::DirExists(m_predictorsDir))
        {
            asLogError(wxString::Format(_("The given predictors directory (%s) couldn't be found."), m_predictorsDir));
            return false;
        }
    }

    /*
     * Methods options
     */

    wxString option = wxEmptyString;

    // Classic+ calibration
    if (parser.Found("cp-resizing-iteration", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/ClassicPlus/ResizingIterations", option);
    }

    if (parser.Found("cp-lat-step", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/ClassicPlus/StepsLatPertinenceMap", option);
    }

    if (parser.Found("cp-lon-step", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/ClassicPlus/StepsLonPertinenceMap", option);
    }

    if (parser.Found("cp-proceed-sequentially", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/ClassicPlus/ProceedSequentially", option);
    }

    // Variables exploration
    if (parser.Found("ve-step", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/VariablesExplo/Step", option);
    }

    // Monte Carlo
    if (parser.Found("mc-runs-nb", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/MonteCarlo/RandomNb", option);
    }

    // Genetic algorithms
    if (parser.Found("ga-ope-nat-sel", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/GeneticAlgorithms/NaturalSelectionOperator", option);
    }

    if (parser.Found("ga-ope-coup-sel", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/GeneticAlgorithms/CouplesSelectionOperator", option);
    }

    if (parser.Found("ga-ope-cross", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/GeneticAlgorithms/CrossoverOperator", option);
    }

    if (parser.Found("ga-ope-mut", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/GeneticAlgorithms/MutationOperator", option);
    }

    if (parser.Found("ga-pop-size", & option))
    {
        wxFileConfig::Get()->Write("Calibration/GeneticAlgorithms/PopulationSize", option);
    }

    if (parser.Found("ga-conv-steps", & option))
    {
        wxFileConfig::Get()->Write("Calibration/GeneticAlgorithms/ConvergenceStepsNb", option);
    }

    if (parser.Found("ga-interm-gen", & option))
    {
        wxFileConfig::Get()->Write("Calibration/GeneticAlgorithms/RatioIntermediateGeneration", option);
    }

    if (parser.Found("ga-nat-sel-tour-p", & option))
    {
        wxFileConfig::Get()->Write("Calibration/GeneticAlgorithms/NaturalSelectionTournamentProbability", option);
    }

    if (parser.Found("ga-coup-sel-tourn-b", & option))
    {
        wxFileConfig::Get()->Write("Calibration/GeneticAlgorithms/CouplesSelectionTournamentNb", option);
    }

    if (parser.Found("ga-cross-mult-pt-nb", & option))
    {
        wxFileConfig::Get()->Write("Calibration/GeneticAlgorithms/CrossoverMultiplePointsNb", option);
    }

    if (parser.Found("ga-cross-blen-pt-nb", & option))
    {
        wxFileConfig::Get()->Write("Calibration/GeneticAlgorithms/CrossoverBlendingPointsNb", option);
    }

    if (parser.Found("ga-cross-blen-share-b", & option))
    {
        wxFileConfig::Get()->Write("Calibration/GeneticAlgorithms/CrossoverBlendingShareBeta", option);
    }

    if (parser.Found("ga-cross-lin-pt-nb", & option))
    {
        wxFileConfig::Get()->Write("Calibration/GeneticAlgorithms/CrossoverLinearPointsNb", option);
    }

    if (parser.Found("ga-cross-heur-pt-nb", & option))
    {
        wxFileConfig::Get()->Write("Calibration/GeneticAlgorithms/CrossoverHeuristicPointsNb", option);
    }

    if (parser.Found("ga-cross-heur-share-b", & option))
    {
        wxFileConfig::Get()->Write("Calibration/GeneticAlgorithms/CrossoverHeuristicShareBeta", option);
    }

    if (parser.Found("ga-cross-bin-pt-nb", & option))
    {
        wxFileConfig::Get()->Write("Calibration/GeneticAlgorithms/CrossoverBinaryLikePointsNb", option);
    }

    if (parser.Found("ga-cross-bin-share-b", & option))
    {
        wxFileConfig::Get()->Write("Calibration/GeneticAlgorithms/CrossoverBinaryLikeShareBeta", option);
    }

    if (parser.Found("ga-mut-unif-cst-p", & option))
    {
        wxFileConfig::Get()->Write("Calibration/GeneticAlgorithms/MutationsUniformConstantProbability", option);
    }

    if (parser.Found("ga-mut-norm-cst-p", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/GeneticAlgorithms/MutationsNormalConstantProbability", option);
    }

    if (parser.Found("ga-mut-norm-cst-dev", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/GeneticAlgorithms/MutationsNormalConstantStdDevRatioRange", option);
    }

    if (parser.Found("ga-mut-unif-var-gens", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/GeneticAlgorithms/MutationsUniformVariableMaxGensNbVar", option);
    }

    if (parser.Found("ga-mut-unif-var-p-strt", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/GeneticAlgorithms/MutationsUniformVariableProbabilityStart", option);
    }

    if (parser.Found("ga-mut-unif-var-p-end", & option))
    {
       wxFileConfig::Get()->Write("/Calibration/GeneticAlgorithms/MutationsUniformVariableProbabilityEnd", option);
    }

    if (parser.Found("ga-mut-norm-var-gens-p", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/GeneticAlgorithms/MutationsNormalVariableMaxGensNbVarProb", option);
    }

    if (parser.Found("ga-mut-norm-var-gens-d", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/GeneticAlgorithms/MutationsNormalVariableMaxGensNbVarStdDev", option);
    }

    if (parser.Found("ga-mut-norm-var-p-strt", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/GeneticAlgorithms/MutationsNormalVariableProbabilityStart", option);
    }

    if (parser.Found("ga-mut-norm-var-p-end", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/GeneticAlgorithms/MutationsNormalVariableProbabilityEnd", option);
    }

    if (parser.Found("ga-mut-norm-var-d-strt", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/GeneticAlgorithms/MutationsNormalVariableStdDevStart", option);
    }

    if (parser.Found("ga-mut-norm-var-d-end", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/GeneticAlgorithms/MutationsNormalVariableStdDevEnd", option);
    }

    if (parser.Found("ga-mut-non-uni-p", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/GeneticAlgorithms/MutationsNonUniformProbability", option);
    }

    if (parser.Found("ga-mut-non-uni-gens", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/GeneticAlgorithms/MutationsNonUniformMaxGensNbVar", option);
    }

    if (parser.Found("ga-mut-non-uni-min-r", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/GeneticAlgorithms/MutationsNonUniformMinRate", option);
    }

    if (parser.Found("ga-mut-multi-scale-p", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/GeneticAlgorithms/MutationsMultiScaleProbability", option);
    }

    // Skip validation option
    if (parser.Found("skip-valid", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/SkipValidation", option);
    }

	// Station ID
	wxString stationIdStr = wxEmptyString;
	if (parser.Found("stationid", &stationIdStr))
	{
		m_predictandStationIds = asParameters::GetFileStationIds(stationIdStr);
	}

    /*
     * Method choice
     */

    // Check for a calibration method option
    if (parser.Found("calibration-method", & m_calibMethod))
    {
        if(!InitForCmdLineOnly()) return false;
        asLogMessage(wxString::Format(_("Given calibration method: %s"), m_calibMethod));
		return true;
    }

    return wxAppConsole::OnCmdLineParsed(parser);
}

int AtmoswingAppCalibrator::OnRun()
{
    if (m_forceQuit)
    {
        asLogError(_("The calibration will not be processed."));
        return 0;
    }

    if (!g_guiMode)
    {
        if (m_calibParamsFile.IsEmpty())
        {
            asLogError(_("The parameters file is not given."));
            return 1001;
        }

        if (m_predictandDB.IsEmpty())
        {
            asLogError(_("The predictand DB is not given."));
            return 1002;
        }

        if (m_predictorsDir.IsEmpty())
        {
            asLogError(_("The predictors directory is not given."));
            return 1003;
        }

        wxMessageOutput* msgOut = wxMessageOutput::Get();

        try
        {
            if (m_calibMethod.IsSameAs("single", false))
            {
                asMethodCalibratorSingle calibrator;
                calibrator.SetParamsFilePath(m_calibParamsFile);
                calibrator.SetPredictandDBFilePath(m_predictandDB);
                calibrator.SetPredictorDataDir(m_predictorsDir);
                calibrator.Manager();
            }
            else if (m_calibMethod.IsSameAs("classic", false))
            {
                asMethodCalibratorClassic calibrator;
                calibrator.SetParamsFilePath(m_calibParamsFile);
                calibrator.SetPredictandDBFilePath(m_predictandDB);
                calibrator.SetPredictorDataDir(m_predictorsDir);
                calibrator.Manager();
            }
            else if (m_calibMethod.IsSameAs("classicp", false))
            {
                asMethodCalibratorClassicPlus calibrator;
                calibrator.SetParamsFilePath(m_calibParamsFile);
                calibrator.SetPredictandDBFilePath(m_predictandDB);
                calibrator.SetPredictorDataDir(m_predictorsDir);
                calibrator.Manager();
            }
            else if (m_calibMethod.IsSameAs("varexplocp", false))
            {
                asMethodCalibratorClassicPlusVarExplo calibrator;
                calibrator.SetParamsFilePath(m_calibParamsFile);
                calibrator.SetPredictandDBFilePath(m_predictandDB);
                calibrator.SetPredictorDataDir(m_predictorsDir);
                calibrator.Manager();
            }
            else if (m_calibMethod.IsSameAs("montecarlo", false))
            {
                asMethodOptimizerRandomSet optimizer;
                optimizer.SetParamsFilePath(m_calibParamsFile);
                optimizer.SetPredictandDBFilePath(m_predictandDB);
                optimizer.SetPredictandStationIds(m_predictandStationIds);
                optimizer.SetPredictorDataDir(m_predictorsDir);
                optimizer.Manager();
            }
            else if (m_calibMethod.IsSameAs("ga", false))
            {
                asMethodOptimizerGeneticAlgorithms optimizer;
                optimizer.SetParamsFilePath(m_calibParamsFile);
                optimizer.SetPredictandDBFilePath(m_predictandDB);
                optimizer.SetPredictandStationIds(m_predictandStationIds);
                optimizer.SetPredictorDataDir(m_predictorsDir);
                optimizer.Manager();
            }
            else if (m_calibMethod.IsSameAs("evalscores", false))
            {
                asMethodCalibratorEvaluateAllScores calibrator;
                calibrator.SetParamsFilePath(m_calibParamsFile);
                calibrator.SetPredictandDBFilePath(m_predictandDB);
                calibrator.SetPredictorDataDir(m_predictorsDir);
                calibrator.Manager();
            }
            else
            {
                if ( msgOut )
                {
                    msgOut->Printf( "Wrong calibration method selection (%s).", m_calibMethod );
                }
            }
        }
		catch (std::bad_alloc& ba)
        {
            wxString msg(ba.what(), wxConvUTF8);
            asLogError(wxString::Format(_("Bad allocation caught: %s"), msg));
            return 1011;
        }
        catch(asException& e)
        {
            wxString fullMessage = e.GetFullMessage();
            if (!fullMessage.IsEmpty())
            {
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

int AtmoswingAppCalibrator::OnExit()
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

bool AtmoswingAppCalibrator::OnExceptionInMainLoop()
{
    asLogError(_("An exception occured in the main loop"));
    return false;
}

void AtmoswingAppCalibrator::OnFatalException()
{
    asLogError(_("An fatal exception occured"));
}

void AtmoswingAppCalibrator::OnUnhandledException()
{
    asLogError(_("An unhandled exception occured"));
}

