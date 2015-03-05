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

#include "asFrameCalibration.h"

#include "wx/fileconf.h"

#include "asMethodCalibratorClassic.h"
#include "asMethodCalibratorClassicPlus.h"
#include "asMethodCalibratorClassicPlusVarExplo.h"
#include "asMethodCalibratorSingle.h"
#include "asMethodCalibratorEvaluateAllScores.h"
#include "asMethodCalibratorSingleOnlyValues.h"
#include "img_toolbar.h"
#include "asFramePreferencesCalibrator.h"
#include "asFrameAbout.h"


asFrameCalibration::asFrameCalibration( wxWindow* parent )
:
asFrameCalibrationVirtual( parent )
{
    m_logWindow = NULL;
    m_methodCalibrator = NULL;

    // Toolbar
    m_toolBar->AddTool( asID_RUN, wxT("Run"), img_run, img_run, wxITEM_NORMAL, _("Run calibration"), _("Run calibration now"), NULL );
    m_toolBar->AddTool( asID_CANCEL, wxT("Cancel"), img_run_cancel, img_run_cancel, wxITEM_NORMAL, _("Cancel calibration"), _("Cancel current calibration"), NULL );
	m_toolBar->AddTool( asID_PREFERENCES, wxT("Preferences"), img_preferences, img_preferences, wxITEM_NORMAL, _("Preferences"), _("Preferences"), NULL );
    m_toolBar->Realize();

    // Connect events
    this->Connect( asID_RUN, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameCalibration::Launch ) );
    this->Connect( asID_CANCEL, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameCalibration::Cancel ) );
	this->Connect( asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameCalibration::OpenFramePreferences ) );

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

asFrameCalibration::~asFrameCalibration()
{
    // Disconnect events
    this->Disconnect( asID_RUN, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameCalibration::Launch ) );
    this->Disconnect( asID_CANCEL, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameCalibration::Cancel ) );
	this->Disconnect( asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameCalibration::OpenFramePreferences ) );
}

void asFrameCalibration::OnInit()
{
    // Set the defaults
    LoadOptions();
    DisplayLogLevelMenu();
}

void asFrameCalibration::Update()
{
    DisplayLogLevelMenu();
}

void asFrameCalibration::OpenFramePreferences( wxCommandEvent& event )
{
    asFramePreferencesCalibrator* frame = new asFramePreferencesCalibrator(this);
    frame->Fit();
    frame->Show();
}

void asFrameCalibration::OpenFrameAbout( wxCommandEvent& event )
{
    asFrameAbout* frame = new asFrameAbout(this);
    frame->Fit();
    frame->Show();
}

void asFrameCalibration::OnShowLog( wxCommandEvent& event )
{
    wxASSERT(m_logWindow);
    m_logWindow->DoShow();
}

void asFrameCalibration::OnLogLevel1( wxCommandEvent& event )
{
    Log().SetLevel(1);
    m_menuLogLevel->FindItemByPosition(0)->Check(true);
    m_menuLogLevel->FindItemByPosition(1)->Check(false);
    m_menuLogLevel->FindItemByPosition(2)->Check(false);
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Write("/General/LogLevel", 1l);
    ThreadsManager().CritSectionConfig().Leave();
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameCalibration::OnLogLevel2( wxCommandEvent& event )
{
    Log().SetLevel(2);
    m_menuLogLevel->FindItemByPosition(0)->Check(false);
    m_menuLogLevel->FindItemByPosition(1)->Check(true);
    m_menuLogLevel->FindItemByPosition(2)->Check(false);
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Write("/General/LogLevel", 2l);
    ThreadsManager().CritSectionConfig().Leave();
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameCalibration::OnLogLevel3( wxCommandEvent& event )
{
    Log().SetLevel(3);
    m_menuLogLevel->FindItemByPosition(0)->Check(false);
    m_menuLogLevel->FindItemByPosition(1)->Check(false);
    m_menuLogLevel->FindItemByPosition(2)->Check(true);
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Write("/General/LogLevel", 3l);
    ThreadsManager().CritSectionConfig().Leave();
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameCalibration::DisplayLogLevelMenu()
{
    // Set log level in the menu
    ThreadsManager().CritSectionConfig().Enter();
    int logLevel = (int)wxFileConfig::Get()->Read("/General/LogLevel", 2l);
    ThreadsManager().CritSectionConfig().Leave();
    m_menuLogLevel->FindItemByPosition(0)->Check(false);
    m_menuLogLevel->FindItemByPosition(1)->Check(false);
    m_menuLogLevel->FindItemByPosition(2)->Check(false);
    switch (logLevel)
    {
    case 1:
        m_menuLogLevel->FindItemByPosition(0)->Check(true);
        Log().SetLevel(1);
        break;
    case 2:
        m_menuLogLevel->FindItemByPosition(1)->Check(true);
        Log().SetLevel(2);
        break;
    case 3:
        m_menuLogLevel->FindItemByPosition(2)->Check(true);
        Log().SetLevel(3);
        break;
    default:
        m_menuLogLevel->FindItemByPosition(1)->Check(true);
        Log().SetLevel(2);
    }
}

void asFrameCalibration::Cancel( wxCommandEvent& event )
{
    if (m_methodCalibrator)
    {
        m_methodCalibrator->Cancel();
    }
}

void asFrameCalibration::LoadOptions()
{
    // General stuff
    wxConfigBase *pConfig = wxFileConfig::Get();
    long MethodSelection = pConfig->Read("/Calibration/MethodSelection", 0l);
    m_choiceMethod->SetSelection((int)MethodSelection);
    wxString ParametersFilePath = pConfig->Read("/Calibration/ParametersFilePath", wxEmptyString);
    m_filePickerParameters->SetPath(ParametersFilePath);
    wxString PredictandDBFilePath = pConfig->Read("/Paths/PredictandDBFilePath", wxEmptyString);
    m_filePickerPredictand->SetPath(PredictandDBFilePath);
    wxString PredictorDir = pConfig->Read("/Paths/PredictorDir", wxEmptyString);
    m_dirPickerPredictor->SetPath(PredictorDir);
    wxString CalibrationResultsDir = pConfig->Read("/Paths/CalibrationResultsDir", asConfig::GetDocumentsDir() + "AtmoSwing" + DS + "Calibration");
    m_dirPickerCalibrationResults->SetPath(CalibrationResultsDir);
    bool parallelEvaluations;
    pConfig->Read("/Calibration/ParallelEvaluations", &parallelEvaluations, false);
    m_checkBoxParallelEvaluations->SetValue(parallelEvaluations);

    // Saving and loading of intermediate results files
    bool saveAnalogDatesStep1;
    pConfig->Read("/Calibration/IntermediateResults/SaveAnalogDatesStep1", &saveAnalogDatesStep1, false);
    m_checkBoxSaveAnalogDatesStep1->SetValue(saveAnalogDatesStep1);
    bool saveAnalogDatesStep2;
    pConfig->Read("/Calibration/IntermediateResults/SaveAnalogDatesStep2", &saveAnalogDatesStep2, false);
    m_checkBoxSaveAnalogDatesStep2->SetValue(saveAnalogDatesStep2);
    bool saveAnalogDatesStep3;
    pConfig->Read("/Calibration/IntermediateResults/SaveAnalogDatesStep3", &saveAnalogDatesStep3, false);
    m_checkBoxSaveAnalogDatesStep3->SetValue(saveAnalogDatesStep3);
    bool saveAnalogDatesStep4;
    pConfig->Read("/Calibration/IntermediateResults/SaveAnalogDatesStep4", &saveAnalogDatesStep4, false);
    m_checkBoxSaveAnalogDatesStep4->SetValue(saveAnalogDatesStep4);
    bool saveAnalogDatesAllSteps;
    pConfig->Read("/Calibration/IntermediateResults/SaveAnalogDatesAllSteps", &saveAnalogDatesAllSteps, false);
    m_checkBoxSaveAnalogDatesAllSteps->SetValue(saveAnalogDatesAllSteps);
    bool saveAnalogValues;
    pConfig->Read("/Calibration/IntermediateResults/SaveAnalogValues", &saveAnalogValues, false);
    m_checkBoxSaveAnalogValues->SetValue(saveAnalogValues);
    bool saveForecastScores;
    pConfig->Read("/Calibration/IntermediateResults/SaveForecastScores", &saveForecastScores, false);
    m_checkBoxSaveForecastScores->SetValue(saveForecastScores);
    bool saveFinalForecastScore;
    pConfig->Read("/Calibration/IntermediateResults/SaveFinalForecastScore", &saveFinalForecastScore, false);
    m_checkBoxSaveFinalForecastScore->SetValue(saveFinalForecastScore);
    bool loadAnalogDatesStep1;
    pConfig->Read("/Calibration/IntermediateResults/LoadAnalogDatesStep1", &loadAnalogDatesStep1, false);
    m_checkBoxLoadAnalogDatesStep1->SetValue(loadAnalogDatesStep1);
    bool loadAnalogDatesStep2;
    pConfig->Read("/Calibration/IntermediateResults/LoadAnalogDatesStep2", &loadAnalogDatesStep2, false);
    m_checkBoxLoadAnalogDatesStep2->SetValue(loadAnalogDatesStep2);
    bool loadAnalogDatesStep3;
    pConfig->Read("/Calibration/IntermediateResults/LoadAnalogDatesStep3", &loadAnalogDatesStep3, false);
    m_checkBoxLoadAnalogDatesStep3->SetValue(loadAnalogDatesStep3);
    bool loadAnalogDatesStep4;
    pConfig->Read("/Calibration/IntermediateResults/LoadAnalogDatesStep4", &loadAnalogDatesStep4, false);
    m_checkBoxLoadAnalogDatesStep4->SetValue(loadAnalogDatesStep4);
    bool loadAnalogDatesAllSteps;
    pConfig->Read("/Calibration/IntermediateResults/LoadAnalogDatesAllSteps", &loadAnalogDatesAllSteps, false);
    m_checkBoxLoadAnalogDatesAllSteps->SetValue(loadAnalogDatesAllSteps);
    bool loadAnalogValues;
    pConfig->Read("/Calibration/IntermediateResults/LoadAnalogValues", &loadAnalogValues, false);
    m_checkBoxLoadAnalogValues->SetValue(loadAnalogValues);
    bool loadForecastScores;
    pConfig->Read("/Calibration/IntermediateResults/LoadForecastScores", &loadForecastScores, false);
    m_checkBoxLoadForecastScores->SetValue(loadForecastScores);

    // Classic+ calibration
    wxString ClassicPlusResizingIterations = pConfig->Read("/Calibration/ClassicPlus/ResizingIterations", "1");
    m_textCtrlClassicPlusResizingIterations->SetValue(ClassicPlusResizingIterations);
    wxString ClassicPlusStepsLatPertinenceMap = pConfig->Read("/Calibration/ClassicPlus/StepsLatPertinenceMap", "2");
    m_textCtrlClassicPlusStepsLatPertinenceMap->SetValue(ClassicPlusStepsLatPertinenceMap);
    wxString ClassicPlusStepsLonPertinenceMap = pConfig->Read("/Calibration/ClassicPlus/StepsLonPertinenceMap", "2");
    m_textCtrlClassicPlusStepsLonPertinenceMap->SetValue(ClassicPlusStepsLonPertinenceMap);
    bool proceedSequentially;
    pConfig->Read("/Calibration/ClassicPlus/ProceedSequentially", &proceedSequentially, true);
    m_checkBoxProceedSequentially->SetValue(proceedSequentially);

    // Variables exploration
    wxString VarExploStep = pConfig->Read("/Calibration/VariablesExplo/Step");
    m_textCtrlVarExploStepToExplore->SetValue(VarExploStep);
}

void asFrameCalibration::OnSaveDefault( wxCommandEvent& event )
{
    SaveOptions();
}

void asFrameCalibration::SaveOptions( )
{
    // General stuff
    wxConfigBase *pConfig = wxFileConfig::Get();
    long MethodSelection = (long) m_choiceMethod->GetSelection();
    pConfig->Write("/Calibration/MethodSelection", MethodSelection);
    wxString ParametersFilePath = m_filePickerParameters->GetPath();
    pConfig->Write("/Calibration/ParametersFilePath", ParametersFilePath);
    wxString PredictandDBFilePath = m_filePickerPredictand->GetPath();
    pConfig->Write("/Paths/PredictandDBFilePath", PredictandDBFilePath);
    wxString PredictorDir = m_dirPickerPredictor->GetPath();
    pConfig->Write("/Paths/PredictorDir", PredictorDir);
	wxString CalibrationResultsDir = m_dirPickerCalibrationResults->GetPath();
    pConfig->Write("/Paths/CalibrationResultsDir", CalibrationResultsDir);
    bool parallelEvaluations = m_checkBoxParallelEvaluations->GetValue();
    pConfig->Write("/Calibration/ParallelEvaluations", parallelEvaluations);

    // Saving and loading of intermediate results files
    bool saveAnalogDatesStep1 = m_checkBoxSaveAnalogDatesStep1->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep1", saveAnalogDatesStep1);
    bool saveAnalogDatesStep2 = m_checkBoxSaveAnalogDatesStep2->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep2", saveAnalogDatesStep2);
    bool saveAnalogDatesStep3 = m_checkBoxSaveAnalogDatesStep3->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep3", saveAnalogDatesStep3);
    bool saveAnalogDatesStep4 = m_checkBoxSaveAnalogDatesStep4->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep4", saveAnalogDatesStep4);
    bool saveAnalogDatesAllSteps = m_checkBoxSaveAnalogDatesAllSteps->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/SaveAnalogDatesAllSteps", saveAnalogDatesAllSteps);
    bool saveAnalogValues = m_checkBoxSaveAnalogValues->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/SaveAnalogValues", saveAnalogValues);
    bool saveForecastScores = m_checkBoxSaveForecastScores->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/SaveForecastScores", saveForecastScores);
    bool saveFinalForecastScore = m_checkBoxSaveFinalForecastScore->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/SaveFinalForecastScore", saveFinalForecastScore);
    bool loadAnalogDatesStep1 = m_checkBoxLoadAnalogDatesStep1->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep1", loadAnalogDatesStep1);
    bool loadAnalogDatesStep2 = m_checkBoxLoadAnalogDatesStep2->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep2", loadAnalogDatesStep2);
    bool loadAnalogDatesStep3 = m_checkBoxLoadAnalogDatesStep3->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep3", loadAnalogDatesStep3);
    bool loadAnalogDatesStep4 = m_checkBoxLoadAnalogDatesStep4->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep4", loadAnalogDatesStep4);
    bool loadAnalogDatesAllSteps = m_checkBoxLoadAnalogDatesAllSteps->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/LoadAnalogDatesAllSteps", loadAnalogDatesAllSteps);
    bool loadAnalogValues = m_checkBoxLoadAnalogValues->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/LoadAnalogValues", loadAnalogValues);
    bool loadForecastScores = m_checkBoxLoadForecastScores->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/LoadForecastScores", loadForecastScores);

    // Classic+ calibration
    wxString ClassicPlusResizingIterations = m_textCtrlClassicPlusResizingIterations->GetValue();
    pConfig->Write("/Calibration/ClassicPlus/ResizingIterations", ClassicPlusResizingIterations);
    wxString ClassicPlusStepsLatPertinenceMap = m_textCtrlClassicPlusStepsLatPertinenceMap->GetValue();
    pConfig->Write("/Calibration/ClassicPlus/StepsLatPertinenceMap", ClassicPlusStepsLatPertinenceMap);
    wxString ClassicPlusStepsLonPertinenceMap = m_textCtrlClassicPlusStepsLonPertinenceMap->GetValue();
    pConfig->Write("/Calibration/ClassicPlus/StepsLonPertinenceMap", ClassicPlusStepsLonPertinenceMap);
    bool proceedSequentially = m_checkBoxProceedSequentially->GetValue();
    pConfig->Write("/Calibration/ClassicPlus/ProceedSequentially", proceedSequentially);

    // Variables exploration
    wxString VarExploStep = m_textCtrlVarExploStepToExplore->GetValue();
    pConfig->Write("/Calibration/VariablesExplo/Step", VarExploStep);

    pConfig->Flush();
}
/*
void asFrameCalibration::OnIdle( wxCommandEvent& event )
{
    wxString state = asGetState();
    m_staticTextState->SetLabel(state);
}
*/
void asFrameCalibration::Launch( wxCommandEvent& event )
{
    SaveOptions();

    try
    {
        switch (m_choiceMethod->GetSelection())
        {
            case wxNOT_FOUND:
            {
                asLogError(_("Wrong method selection."));
                break;
            }
            case 0: // Single
            {
                asLogMessage(_("Proceeding to single assessment."));
                m_methodCalibrator = new asMethodCalibratorSingle();
                break;
            }
            case 1: // Classic
            {
                asLogMessage(_("Proceeding to classic calibration."));
                m_methodCalibrator = new asMethodCalibratorClassic();
                break;
            }
            case 2: // Classic+
            {
                asLogMessage(_("Proceeding to classic+ calibration."));
                m_methodCalibrator = new asMethodCalibratorClassicPlus();
                break;
            }
            case 3: // Variables exploration with classic+
            {
                asLogMessage(_("Proceeding to variables exploration."));
                m_methodCalibrator = new asMethodCalibratorClassicPlusVarExplo();
                break;
            }
            case 4: // Scores evaluation
            {
                asLogMessage(_("Proceeding to all scores evaluation."));
                m_methodCalibrator = new asMethodCalibratorEvaluateAllScores();
                break;
            }
            case 5: // Only predictand values
            {
                asLogMessage(_("Proceeding to predictand values saving."));
                m_methodCalibrator = new asMethodCalibratorSingleOnlyValues();
                break;
            }
            default:
                asLogError(_("Chosen method not defined yet."));
        }

        if (m_methodCalibrator)
        {
            m_methodCalibrator->SetParamsFilePath(m_filePickerParameters->GetPath());
            m_methodCalibrator->SetPredictandDBFilePath(m_filePickerPredictand->GetPath());
            m_methodCalibrator->SetPredictorDataDir(m_dirPickerPredictor->GetPath());
            m_methodCalibrator->Manager();
        }
    }
    catch(bad_alloc& ba)
    {
        wxString msg(ba.what(), wxConvUTF8);
        asLogError(wxString::Format(_("Bad allocation caught: %s"), msg.c_str()));
        asLogError(_("Failed to process the calibration."));
    }
    catch(asException& e)
    {
		wxString fullMessage = e.GetFullMessage();
		if (!fullMessage.IsEmpty())
		{
			asLogError(fullMessage);
		}
		asLogError(_("Failed to process the calibration."));
    }

    wxDELETE(m_methodCalibrator);

    wxMessageBox(_("Calibration over."));
}
