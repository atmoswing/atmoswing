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
    #include "img_toolbar.h"
#endif

static const wxCmdLineEntryDesc g_cmdLineDesc[] =
{
    { wxCMD_LINE_SWITCH, "v", "version", "print version" },
    { wxCMD_LINE_SWITCH, "s", "silent", "silent mode" },
    { wxCMD_LINE_SWITCH, "l", "local", "work in local directory" },
    { wxCMD_LINE_OPTION, "tn", "threadsnb", "number of threads to use" },
    { wxCMD_LINE_OPTION, "ll", "loglevel", "set a log level"
                                "\n \t\t\t\t 0: minimum"
                                "\n \t\t\t\t 1: errors"
                                "\n \t\t\t\t 2: warnings"
                                "\n \t\t\t\t 3: verbose" },
    { wxCMD_LINE_OPTION, "lt", "logtarget", "set log target"
                                "\n \t\t\t\t file: file only"
                                "\n \t\t\t\t prompt: command prompt"
                                "\n \t\t\t\t both: command prompt and file (default)" },
    { wxCMD_LINE_OPTION, "r", "runnumber", "choice of number associated with the run" },
    { wxCMD_LINE_OPTION, "fp", "fileparams", "choice of the calibration parameters file" },
    { wxCMD_LINE_OPTION, "fd", "filepredicand", "choice of the predictand DB" },
    { wxCMD_LINE_OPTION, "di", "dirpredictors", "choice of the predictors directory" },
    { wxCMD_LINE_OPTION, "cm", "calibmethod", "choice of the calibration method"
                                "\n \t\t\t\t single: single assessment"
                                "\n \t\t\t\t classic: classic calibration"
                                "\n \t\t\t\t classicp: classic+ calibration"
                                "\n \t\t\t\t varexplocp: variables exploration classic+"
                                "\n \t\t\t\t evalscores: Evaluate all scores" },
    { wxCMD_LINE_OPTION, "cpresizeite", "cpresizeite", "options ClassicPlusResizingIterations" },
    { wxCMD_LINE_OPTION, "cplatstepmap", "cplatstepmap", "options ClassicPlusStepsLatPertinenceMap" },
    { wxCMD_LINE_OPTION, "cplonstepmap", "cplonstepmap", "options ClassicPlusStepsLonPertinenceMap" },
    { wxCMD_LINE_OPTION, "cpprosseq", "cpprosseq", "options ClassicPlusProceedSequentially" },
    { wxCMD_LINE_OPTION, "varexpstep", "varexpstep", "options VariablesExploStep" },
    { wxCMD_LINE_OPTION, "mcrunsnb", "mcrunsnb", "options MonteCarloRandomNb" },
    { wxCMD_LINE_OPTION, "nmrunsnb", "nmrunsnb", "options NelderMeadNbRuns" },
    { wxCMD_LINE_OPTION, "nmrho", "nmrho", "options NelderMeadRho" },
    { wxCMD_LINE_OPTION, "nmchi", "nmchi", "options NelderMeadChi" },
    { wxCMD_LINE_OPTION, "nmgamma", "nmgamma", "options NelderMeadGamma" },
    { wxCMD_LINE_OPTION, "nmsigma", "nmsigma", "options NelderMeadSigma" },
    { wxCMD_LINE_OPTION, "savedatesstep", "savedatesstep", "options SaveAnalogDatesStep (with given number)" },
    { wxCMD_LINE_OPTION, "loaddatesstep", "loaddatesstep", "options LoadAnalogDatesStep (with given number)" },
    { wxCMD_LINE_OPTION, "savedatesallsteps", "savedatesallsteps", "options SaveAnalogDatesStep (all the steps)" },
    { wxCMD_LINE_OPTION, "savevalues", "savevalues", "options SaveAnalogValues" },
    { wxCMD_LINE_OPTION, "savescores", "savescores", "options SaveForecastScores" },
    { wxCMD_LINE_OPTION, "skipvalid", "skipvalid", "Skip the validation calculation" },

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
    wxString appName = "AtmoSwing calibrator";
    wxApp::SetAppName(appName);

    g_GuiMode = true;
    g_Local = false;
    m_CalibParamsFile = wxEmptyString;
    m_PredictandDB = wxEmptyString;
    m_PredictorsDir = wxEmptyString;
    m_CalibMethod = wxEmptyString;

    // Call default behaviour (mandatory for command-line mode)
    if (!wxApp::OnInit()) // When false, we are in CL mode
        return false;

    #if wxUSE_GUI
	m_SingleInstanceChecker = NULL;
    if (g_GuiMode)
    {
        // Check that it is the unique instance
        bool multipleInstances = false;

        wxFileConfig::Get()->Read("/General/MultiInstances", &multipleInstances, false);

        if (!multipleInstances)
        {
            const wxString instanceName = wxString::Format(wxT("AtmoSwingCalibrator-%s"),wxGetUserId().c_str());
            m_SingleInstanceChecker = new wxSingleInstanceChecker(instanceName);
            if ( m_SingleInstanceChecker->IsAnotherRunning() )
            {
                wxMessageBox(_("Program already running, aborting."));
                return false;
            }
        }

        // Following for GUI only
        wxInitAllImageHandlers();

        // Initialize images
        initialize_images_toolbar();

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
    g_GuiMode = false;
    g_UnitTesting = false;
    g_SilentMode = true;
    g_VerboseMode = false;
    g_Responsive = false;

    // Prepare local path
    wxString localPath = wxFileName::GetCwd() + DS;
    if (g_RunNb>0)
    {
        localPath.Append("runs");
        localPath.Append(DS);
        localPath.Append(wxString::Format("%d", g_RunNb));
        localPath.Append(DS);
    }

    if (g_Local)
    {
        wxString fullPath = localPath;
        fullPath.Append("AtmoSwingCalibrator.log");
        Log().CreateFileOnlyAtPath(fullPath);
    }
    else
    {
        Log().CreateFileOnly("AtmoSwingCalibrator.log");
    }

    Log().DisableMessageBoxOnError();

    if (g_Local)
    {
        wxString dirData = wxFileName::GetCwd()+DS+"data"+DS;

        wxConfigBase *pConfig = wxFileConfig::Get();

        // Define the default preferences
        pConfig->Write("/General/MultiInstances", true);
        pConfig->Write("/General/GuiOptions", 0l);
        pConfig->Write("/General/Responsive", false);
        pConfig->Write("/General/DisplayLogWindow", false);
        pConfig->Write("/General/ProcessingThreadsPriority", 100);
        pConfig->Write("/Paths/DataPredictandDBDir", dirData);
        pConfig->Write("/Paths/IntermediateResultsDir", localPath+"temp");
        pConfig->Write("/Paths/CalibrationResultsDir", localPath+"results");
        pConfig->Write("/Paths/ArchivePredictorsDir", dirData);
        pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", (long)asLIN_ALGEBRA_NOVAR);
        pConfig->Write("/Calibration/ParallelEvaluations", true);
        pConfig->Write("/Calibration/GeneticAlgorithms/AllowElitismForTheBest", true);
        pConfig->Write("/General/AllowMultithreading", true);
        pConfig->Write("/ProcessingOptions/ProcessingMethod", (long)asMULTITHREADS);

        pConfig->Flush();

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
            msg.Printf("AtmoSwing version %s, %s", g_Version.c_str(), (const wxChar*) date);

            msgOut->Printf( wxT("%s"), msg.c_str() );
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
    if (parser.Found("r", & runNbStr))
    {
        runNbStr.ToLong(&runNb);
        g_RunNb = (int)runNb;
    }

    // Local mode
    if (parser.Found("l"))
    {
        g_Local = true;
        wxString localPath = wxFileName::GetCwd() + DS;
        if (g_RunNb>0)
        {
            localPath.Append("runs");
            localPath.Append(DS);
            localPath.Append(wxString::Format("%d", g_RunNb));
            localPath.Append(DS);

            // Create directory
            wxFileName userDir = wxFileName::DirName(localPath);
            userDir.Mkdir(wxS_DIR_DEFAULT,wxPATH_MKDIR_FULL);
        }

        // Create local ini file
        wxString iniPath = localPath;
        iniPath.Append("AtmoSwing.ini");

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
        wxFileConfig *pConfig = new wxFileConfig("AtmoSwing",wxEmptyString,asConfig::GetUserDataDir()+"AtmoSwing.ini",asConfig::GetUserDataDir()+"AtmoSwing.ini",wxCONFIG_USE_LOCAL_FILE);
        wxFileConfig::Set(pConfig);
    }

    // Check for a log level option
    wxString logLevelStr = wxEmptyString;
    if (parser.Found("ll", & logLevelStr))
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
    if (parser.Found("lt", & logTargetStr))
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
                msgOut->Printf( _("The given log target (%s) does not correspond to any possible option."), logTargetStr.c_str() );
            }
        }
    }

    // Check if the user asked for the silent mode
    if (parser.Found("s"))
    {
        Log().SetTarget(asLog::File);
    }

    // Check for a calibration params file
    wxString threadsNb = wxEmptyString;
    if (parser.Found("tn", & threadsNb))
    {
        wxFileConfig::Get()->Write("/General/ProcessingMaxThreadNb", threadsNb);
    }

    // Check for a calibration params file
    if (parser.Found("fp", & m_CalibParamsFile))
    {
        if (g_Local)
        {
            m_CalibParamsFile = wxFileName::GetCwd() + DS + m_CalibParamsFile;
        }

        if (!wxFileName::FileExists(m_CalibParamsFile))
        {
            asLogError(wxString::Format(_("The given calibration file (%s) couldn't be found."), m_CalibParamsFile.c_str()));
            return false;
        }
    }

    // Check for a calibration predictand DB
    if (parser.Found("fd", & m_PredictandDB))
    {
        if (g_Local)
        {
            m_PredictandDB = wxFileName::GetCwd() + DS + m_PredictandDB;
        }

        if (!wxFileName::FileExists(m_PredictandDB))
        {
            asLogError(wxString::Format(_("The given predictand DB (%s) couldn't be found."), m_PredictandDB.c_str()));
            return false;
        }
    }

    // Check for a predictors directory
    if (parser.Found("di", & m_PredictorsDir))
    {
        if (g_Local)
        {
            m_PredictorsDir = wxFileName::GetCwd() + DS + m_PredictorsDir;
        }

        if (!wxFileName::DirExists(m_PredictorsDir))
        {
            asLogError(wxString::Format(_("The given predictors directory (%s) couldn't be found."), m_PredictorsDir.c_str()));
            return false;
        }
    }

    /*
     * Methods options
     */

    wxString option = wxEmptyString;

    // Classic+ calibration
    if (parser.Found("cpresizeite", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/ClassicPlus/ResizingIterations", option);
    }

    if (parser.Found("cplatstepmap", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/ClassicPlus/StepsLatPertinenceMap", option);
    }

    if (parser.Found("cplonstepmap", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/ClassicPlus/StepsLonPertinenceMap", option);
    }

    if (parser.Found("cpprosseq", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/ClassicPlus/ProceedSequentially", option);
    }

    // Variables exploration
    if (parser.Found("varexpstep", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/VariablesExplo/Step", option);
    }

    // Skip validation option
    if (parser.Found("skipvalid", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/SkipValidation", option);
    }

    // Saving and loading of intermediate results files: reinitialized as it may be catastrophic to forget that it is enabled...
    wxFileConfig::Get()->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep1", false);
    wxFileConfig::Get()->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep2", false);
    wxFileConfig::Get()->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep3", false);
    wxFileConfig::Get()->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep4", false);
    wxFileConfig::Get()->Write("/Calibration/IntermediateResults/SaveAnalogDatesAllSteps", false);
    wxFileConfig::Get()->Write("/Calibration/IntermediateResults/SaveAnalogValues", false);
    wxFileConfig::Get()->Write("/Calibration/IntermediateResults/SaveForecastScores", false);
    wxFileConfig::Get()->Write("/Calibration/IntermediateResults/SaveFinalForecastScore", false);
    wxFileConfig::Get()->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep1", false);
    wxFileConfig::Get()->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep2", false);
    wxFileConfig::Get()->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep3", false);
    wxFileConfig::Get()->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep4", false);
    wxFileConfig::Get()->Write("/Calibration/IntermediateResults/LoadAnalogDatesAllSteps", false);
    wxFileConfig::Get()->Write("/Calibration/IntermediateResults/LoadAnalogValues", false);
    wxFileConfig::Get()->Write("/Calibration/IntermediateResults/LoadForecastScores", false);
    wxFileConfig::Get()->Write("/Calibration/IntermediateResults/LoadFinalForecastScore", false);

    wxString saveDatesStep;
    if (parser.Found("savedatesstep", & saveDatesStep))
    {
        long step = -1;
        saveDatesStep.ToLong(&step);
        int intStep = (int)step;

        switch (intStep)
        {
            case 1:
            {
                wxFileConfig::Get()->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep1", true);
                break;
            }
            case 2:
            {
                wxFileConfig::Get()->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep2", true);
                break;
            }
            case 3:
            {
                wxFileConfig::Get()->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep3", true);
                break;
            }
            case 4:
            {
                wxFileConfig::Get()->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep4", true);
                break;
            }
            default:
            {
                asLogError(wxString::Format(_("Wrong given step for the intermediate saving option (%d)."), intStep));
            }
        }
    }

    wxString loadDatesStep;
    if (parser.Found("loaddatesstep", & loadDatesStep))
    {
        long step = -1;
        loadDatesStep.ToLong(&step);
        int intStep = (int)step;

        switch (intStep)
        {
            case 1:
            {
                wxFileConfig::Get()->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep1", true);
                break;
            }
            case 2:
            {
                wxFileConfig::Get()->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep2", true);
                break;
            }
            case 3:
            {
                wxFileConfig::Get()->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep3", true);
                break;
            }
            case 4:
            {
                wxFileConfig::Get()->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep4", true);
                break;
            }
            default:
            {
                asLogError(wxString::Format(_("Wrong given step for the intermediate loading option (%d)."), intStep));
            }
        }
    }

    // Save analogs values
    if (parser.Found("savedatesallsteps", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep1", true);
        wxFileConfig::Get()->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep2", true);
        wxFileConfig::Get()->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep3", true);
        wxFileConfig::Get()->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep4", true);
    }

    // Save analogs values
    if (parser.Found("savevalues", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/IntermediateResults/SaveAnalogValues", option);
    }

    // Save forecast scores
    if (parser.Found("savescores", & option))
    {
        wxFileConfig::Get()->Write("/Calibration/IntermediateResults/SaveForecastScores", option);
    }

    /*
     * Method choice
     */

    // Check for a calibration method option
    if (parser.Found("cm", & m_CalibMethod))
    {
        if(!InitForCmdLineOnly()) return false;
        asLogMessage(wxString::Format(_("Given calibration method: %s"), m_CalibMethod.c_str()));
    }

    return wxAppConsole::OnCmdLineParsed(parser);
}

int AtmoswingAppCalibrator::OnExit()
{
	#if wxUSE_GUI
		// Instance checker
		wxDELETE(m_SingleInstanceChecker);
	#endif

    // Config file (from wxWidgets samples)
    delete wxFileConfig::Set((wxFileConfig *) NULL);

    // Delete threads manager and log
    DeleteThreadsManager();
    DeleteLog();

    // CleanUp
    wxApp::CleanUp();

    return 0;
}

int AtmoswingAppCalibrator::OnRun()
{
    if (!g_GuiMode)
    {
        if (m_CalibParamsFile.IsEmpty())
        {
            asLogError(_("The parameters file is not given."));
            return 1001;
        }

        if (m_PredictandDB.IsEmpty())
        {
            asLogError(_("The predictand DB is not given."));
            return 1002;
        }

        if (m_PredictorsDir.IsEmpty())
        {
            asLogError(_("The predictors directory is not given."));
            return 1003;
        }

        wxMessageOutput* msgOut = wxMessageOutput::Get();

        try
        {
            if (m_CalibMethod.IsSameAs("single", false))
            {
                asMethodCalibratorSingle calibrator;
                calibrator.SetParamsFilePath(m_CalibParamsFile);
                calibrator.SetPredictandDBFilePath(m_PredictandDB);
                calibrator.SetPredictorDataDir(m_PredictorsDir);
                calibrator.Manager();
            }
            else if (m_CalibMethod.IsSameAs("classic", false))
            {
                asMethodCalibratorClassic calibrator;
                calibrator.SetParamsFilePath(m_CalibParamsFile);
                calibrator.SetPredictandDBFilePath(m_PredictandDB);
                calibrator.SetPredictorDataDir(m_PredictorsDir);
                calibrator.Manager();
            }
            else if (m_CalibMethod.IsSameAs("classicp", false))
            {
                asMethodCalibratorClassicPlus calibrator;
                calibrator.SetParamsFilePath(m_CalibParamsFile);
                calibrator.SetPredictandDBFilePath(m_PredictandDB);
                calibrator.SetPredictorDataDir(m_PredictorsDir);
                calibrator.Manager();
            }
            else if (m_CalibMethod.IsSameAs("varexplocp", false))
            {
                asMethodCalibratorClassicPlusVarExplo calibrator;
                calibrator.SetParamsFilePath(m_CalibParamsFile);
                calibrator.SetPredictandDBFilePath(m_PredictandDB);
                calibrator.SetPredictorDataDir(m_PredictorsDir);
                calibrator.Manager();
            }
            else if (m_CalibMethod.IsSameAs("evalscores", false))
            {
                asMethodCalibratorEvaluateAllScores calibrator;
                calibrator.SetParamsFilePath(m_CalibParamsFile);
                calibrator.SetPredictandDBFilePath(m_PredictandDB);
                calibrator.SetPredictorDataDir(m_PredictorsDir);
                calibrator.Manager();
            }
            else
            {
                if ( msgOut )
                {
                    msgOut->Printf( "Wrong calibration method selection (%s).", m_CalibMethod.c_str() );
                }
            }
        }
        catch(bad_alloc& ba)
        {
            wxString msg(ba.what(), wxConvUTF8);
            asLogError(wxString::Format(_("Bad allocation caught: %s"), msg.c_str()));
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

