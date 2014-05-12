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
#include "asFramePreferences.h"
#include "asFrameAbout.h"


asFrameCalibration::asFrameCalibration( wxWindow* parent )
:
asFrameCalibrationVirtual( parent )
{
    m_MethodCalibrator = NULL;

    // Toolbar
    m_ToolBar->AddTool( asID_RUN, wxT("Run"), img_run, img_run, wxITEM_NORMAL, _("Run calibration"), _("Run calibration now"), NULL );
    m_ToolBar->AddTool( asID_CANCEL, wxT("Cancel"), img_run_cancel, img_run_cancel, wxITEM_NORMAL, _("Cancel calibration"), _("Cancel current calibration"), NULL );
	m_ToolBar->AddTool( asID_PREFERENCES, wxT("Preferences"), img_preferences, img_preferences, wxITEM_NORMAL, _("Preferences"), _("Preferences"), NULL );

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
    asFramePreferences* frame = new asFramePreferences(this);
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
    wxASSERT(m_LogWindow);
    m_LogWindow->DoShow();
}

void asFrameCalibration::OnLogLevel1( wxCommandEvent& event )
{
    Log().SetLevel(1);
    m_MenuLogLevel->FindItemByPosition(0)->Check(true);
    m_MenuLogLevel->FindItemByPosition(1)->Check(false);
    m_MenuLogLevel->FindItemByPosition(2)->Check(false);
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Write("/Standard/LogLevel", 1l);
    ThreadsManager().CritSectionConfig().Leave();
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameCalibration::OnLogLevel2( wxCommandEvent& event )
{
    Log().SetLevel(2);
    m_MenuLogLevel->FindItemByPosition(0)->Check(false);
    m_MenuLogLevel->FindItemByPosition(1)->Check(true);
    m_MenuLogLevel->FindItemByPosition(2)->Check(false);
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Write("/Standard/LogLevel", 2l);
    ThreadsManager().CritSectionConfig().Leave();
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameCalibration::OnLogLevel3( wxCommandEvent& event )
{
    Log().SetLevel(3);
    m_MenuLogLevel->FindItemByPosition(0)->Check(false);
    m_MenuLogLevel->FindItemByPosition(1)->Check(false);
    m_MenuLogLevel->FindItemByPosition(2)->Check(true);
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Write("/Standard/LogLevel", 3l);
    ThreadsManager().CritSectionConfig().Leave();
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameCalibration::DisplayLogLevelMenu()
{
    // Set log level in the menu
    ThreadsManager().CritSectionConfig().Enter();
    int logLevel = (int)wxFileConfig::Get()->Read("/Standard/LogLevel", 2l);
    ThreadsManager().CritSectionConfig().Leave();
    m_MenuLogLevel->FindItemByPosition(0)->Check(false);
    m_MenuLogLevel->FindItemByPosition(1)->Check(false);
    m_MenuLogLevel->FindItemByPosition(2)->Check(false);
    switch (logLevel)
    {
    case 1:
        m_MenuLogLevel->FindItemByPosition(0)->Check(true);
        Log().SetLevel(1);
        break;
    case 2:
        m_MenuLogLevel->FindItemByPosition(1)->Check(true);
        Log().SetLevel(2);
        break;
    case 3:
        m_MenuLogLevel->FindItemByPosition(2)->Check(true);
        Log().SetLevel(3);
        break;
    default:
        m_MenuLogLevel->FindItemByPosition(1)->Check(true);
        Log().SetLevel(2);
    }
}

void asFrameCalibration::Cancel( wxCommandEvent& event )
{
    if (m_MethodCalibrator)
    {
        m_MethodCalibrator->Cancel();
    }
}

void asFrameCalibration::LoadOptions()
{
    // General stuff
    wxConfigBase *pConfig = wxFileConfig::Get();
    long MethodSelection = pConfig->Read("/Calibration/MethodSelection", 0l);
    m_ChoiceMethod->SetSelection((int)MethodSelection);
    wxString ParametersFilePath = pConfig->Read("/Calibration/ParametersFilePath", wxEmptyString);
    m_FilePickerParameters->SetPath(ParametersFilePath);
    wxString PredictandDBFilePath = pConfig->Read("/StandardPaths/PredictandDBFilePath", wxEmptyString);
    m_FilePickerPredictand->SetPath(PredictandDBFilePath);
    wxString PredictorDir = pConfig->Read("/StandardPaths/PredictorDir", wxEmptyString);
    m_DirPickerPredictor->SetPath(PredictorDir);
    wxString CalibrationResultsDir = pConfig->Read("/StandardPaths/CalibrationResultsDir", asConfig::GetDocumentsDir() + "Atmoswing" + DS + "Calibration");
    m_DirPickerCalibrationResults->SetPath(CalibrationResultsDir);
    bool parallelEvaluations;
    pConfig->Read("/Calibration/ParallelEvaluations", &parallelEvaluations, false);
    m_CheckBoxParallelEvaluations->SetValue(parallelEvaluations);

    // Saving and loading of intermediate results files
    bool saveAnalogDatesStep1;
    pConfig->Read("/Calibration/IntermediateResults/SaveAnalogDatesStep1", &saveAnalogDatesStep1, false);
    m_CheckBoxSaveAnalogDatesStep1->SetValue(saveAnalogDatesStep1);
    bool saveAnalogDatesStep2;
    pConfig->Read("/Calibration/IntermediateResults/SaveAnalogDatesStep2", &saveAnalogDatesStep2, false);
    m_CheckBoxSaveAnalogDatesStep2->SetValue(saveAnalogDatesStep2);
    bool saveAnalogDatesStep3;
    pConfig->Read("/Calibration/IntermediateResults/SaveAnalogDatesStep3", &saveAnalogDatesStep3, false);
    m_CheckBoxSaveAnalogDatesStep3->SetValue(saveAnalogDatesStep3);
    bool saveAnalogDatesStep4;
    pConfig->Read("/Calibration/IntermediateResults/SaveAnalogDatesStep4", &saveAnalogDatesStep4, false);
    m_CheckBoxSaveAnalogDatesStep4->SetValue(saveAnalogDatesStep4);
    bool saveAnalogDatesAllSteps;
    pConfig->Read("/Calibration/IntermediateResults/SaveAnalogDatesAllSteps", &saveAnalogDatesAllSteps, false);
    m_CheckBoxSaveAnalogDatesAllSteps->SetValue(saveAnalogDatesAllSteps);
    bool saveAnalogValues;
    pConfig->Read("/Calibration/IntermediateResults/SaveAnalogValues", &saveAnalogValues, false);
    m_CheckBoxSaveAnalogValues->SetValue(saveAnalogValues);
    bool saveForecastScores;
    pConfig->Read("/Calibration/IntermediateResults/SaveForecastScores", &saveForecastScores, false);
    m_CheckBoxSaveForecastScores->SetValue(saveForecastScores);
    bool saveFinalForecastScore;
    pConfig->Read("/Calibration/IntermediateResults/SaveFinalForecastScore", &saveFinalForecastScore, false);
    m_CheckBoxSaveFinalForecastScore->SetValue(saveFinalForecastScore);
    bool loadAnalogDatesStep1;
    pConfig->Read("/Calibration/IntermediateResults/LoadAnalogDatesStep1", &loadAnalogDatesStep1, false);
    m_CheckBoxLoadAnalogDatesStep1->SetValue(loadAnalogDatesStep1);
    bool loadAnalogDatesStep2;
    pConfig->Read("/Calibration/IntermediateResults/LoadAnalogDatesStep2", &loadAnalogDatesStep2, false);
    m_CheckBoxLoadAnalogDatesStep2->SetValue(loadAnalogDatesStep2);
    bool loadAnalogDatesStep3;
    pConfig->Read("/Calibration/IntermediateResults/LoadAnalogDatesStep3", &loadAnalogDatesStep3, false);
    m_CheckBoxLoadAnalogDatesStep3->SetValue(loadAnalogDatesStep3);
    bool loadAnalogDatesStep4;
    pConfig->Read("/Calibration/IntermediateResults/LoadAnalogDatesStep4", &loadAnalogDatesStep4, false);
    m_CheckBoxLoadAnalogDatesStep4->SetValue(loadAnalogDatesStep4);
    bool loadAnalogDatesAllSteps;
    pConfig->Read("/Calibration/IntermediateResults/LoadAnalogDatesAllSteps", &loadAnalogDatesAllSteps, false);
    m_CheckBoxLoadAnalogDatesAllSteps->SetValue(loadAnalogDatesAllSteps);
    bool loadAnalogValues;
    pConfig->Read("/Calibration/IntermediateResults/LoadAnalogValues", &loadAnalogValues, false);
    m_CheckBoxLoadAnalogValues->SetValue(loadAnalogValues);
    bool loadForecastScores;
    pConfig->Read("/Calibration/IntermediateResults/LoadForecastScores", &loadForecastScores, false);
    m_CheckBoxLoadForecastScores->SetValue(loadForecastScores);

    // Classic+ calibration
    wxString ClassicPlusResizingIterations = pConfig->Read("/Calibration/ClassicPlus/ResizingIterations", "1");
    m_TextCtrlClassicPlusResizingIterations->SetValue(ClassicPlusResizingIterations);
    wxString ClassicPlusStepsLatPertinenceMap = pConfig->Read("/Calibration/ClassicPlus/StepsLatPertinenceMap", "2");
    m_TextCtrlClassicPlusStepsLatPertinenceMap->SetValue(ClassicPlusStepsLatPertinenceMap);
    wxString ClassicPlusStepsLonPertinenceMap = pConfig->Read("/Calibration/ClassicPlus/StepsLonPertinenceMap", "2");
    m_TextCtrlClassicPlusStepsLonPertinenceMap->SetValue(ClassicPlusStepsLonPertinenceMap);
    bool proceedSequentially;
    pConfig->Read("/Calibration/ClassicPlus/ProceedSequentially", &proceedSequentially, true);
    m_CheckBoxProceedSequentially->SetValue(proceedSequentially);

    // Variables exploration
    wxString VarExploStep = pConfig->Read("/Calibration/VariablesExplo/Step");
    m_TextCtrlVarExploStepToExplore->SetValue(VarExploStep);
}

void asFrameCalibration::OnSaveDefault( wxCommandEvent& event )
{
    SaveOptions();
}

void asFrameCalibration::SaveOptions( )
{
    // General stuff
    wxConfigBase *pConfig = wxFileConfig::Get();
    long MethodSelection = (long) m_ChoiceMethod->GetSelection();
    pConfig->Write("/Calibration/MethodSelection", MethodSelection);
    wxString ParametersFilePath = m_FilePickerParameters->GetPath();
    pConfig->Write("/Calibration/ParametersFilePath", ParametersFilePath);
    wxString PredictandDBFilePath = m_FilePickerPredictand->GetPath();
    pConfig->Write("/StandardPaths/PredictandDBFilePath", PredictandDBFilePath);
    wxString PredictorDir = m_DirPickerPredictor->GetPath();
    pConfig->Write("/StandardPaths/PredictorDir", PredictorDir);
	wxString CalibrationResultsDir = m_DirPickerCalibrationResults->GetPath();
    pConfig->Write("/StandardPaths/CalibrationResultsDir", CalibrationResultsDir);
    bool parallelEvaluations = m_CheckBoxParallelEvaluations->GetValue();
    pConfig->Write("/Calibration/ParallelEvaluations", parallelEvaluations);

    // Saving and loading of intermediate results files
    bool saveAnalogDatesStep1 = m_CheckBoxSaveAnalogDatesStep1->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep1", saveAnalogDatesStep1);
    bool saveAnalogDatesStep2 = m_CheckBoxSaveAnalogDatesStep2->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep2", saveAnalogDatesStep2);
    bool saveAnalogDatesStep3 = m_CheckBoxSaveAnalogDatesStep3->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep3", saveAnalogDatesStep3);
    bool saveAnalogDatesStep4 = m_CheckBoxSaveAnalogDatesStep4->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/SaveAnalogDatesStep4", saveAnalogDatesStep4);
    bool saveAnalogDatesAllSteps = m_CheckBoxSaveAnalogDatesAllSteps->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/SaveAnalogDatesAllSteps", saveAnalogDatesAllSteps);
    bool saveAnalogValues = m_CheckBoxSaveAnalogValues->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/SaveAnalogValues", saveAnalogValues);
    bool saveForecastScores = m_CheckBoxSaveForecastScores->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/SaveForecastScores", saveForecastScores);
    bool saveFinalForecastScore = m_CheckBoxSaveFinalForecastScore->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/SaveFinalForecastScore", saveFinalForecastScore);
    bool loadAnalogDatesStep1 = m_CheckBoxLoadAnalogDatesStep1->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep1", loadAnalogDatesStep1);
    bool loadAnalogDatesStep2 = m_CheckBoxLoadAnalogDatesStep2->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep2", loadAnalogDatesStep2);
    bool loadAnalogDatesStep3 = m_CheckBoxLoadAnalogDatesStep3->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep3", loadAnalogDatesStep3);
    bool loadAnalogDatesStep4 = m_CheckBoxLoadAnalogDatesStep4->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/LoadAnalogDatesStep4", loadAnalogDatesStep4);
    bool loadAnalogDatesAllSteps = m_CheckBoxLoadAnalogDatesAllSteps->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/LoadAnalogDatesAllSteps", loadAnalogDatesAllSteps);
    bool loadAnalogValues = m_CheckBoxLoadAnalogValues->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/LoadAnalogValues", loadAnalogValues);
    bool loadForecastScores = m_CheckBoxLoadForecastScores->GetValue();
    pConfig->Write("/Calibration/IntermediateResults/LoadForecastScores", loadForecastScores);

    // Classic+ calibration
    wxString ClassicPlusResizingIterations = m_TextCtrlClassicPlusResizingIterations->GetValue();
    pConfig->Write("/Calibration/ClassicPlus/ResizingIterations", ClassicPlusResizingIterations);
    wxString ClassicPlusStepsLatPertinenceMap = m_TextCtrlClassicPlusStepsLatPertinenceMap->GetValue();
    pConfig->Write("/Calibration/ClassicPlus/StepsLatPertinenceMap", ClassicPlusStepsLatPertinenceMap);
    wxString ClassicPlusStepsLonPertinenceMap = m_TextCtrlClassicPlusStepsLonPertinenceMap->GetValue();
    pConfig->Write("/Calibration/ClassicPlus/StepsLonPertinenceMap", ClassicPlusStepsLonPertinenceMap);
    bool proceedSequentially = m_CheckBoxProceedSequentially->GetValue();
    pConfig->Write("/Calibration/ClassicPlus/ProceedSequentially", proceedSequentially);

    // Variables exploration
    wxString VarExploStep = m_TextCtrlVarExploStepToExplore->GetValue();
    pConfig->Write("/Calibration/VariablesExplo/Step", VarExploStep);

    pConfig->Flush();
}
/*
void asFrameCalibration::OnIdle( wxCommandEvent& event )
{
    wxString state = asGetState();
    m_StaticTextState->SetLabel(state);
}
*/
void asFrameCalibration::Launch( wxCommandEvent& event )
{
    SaveOptions();

    try
    {
        switch (m_ChoiceMethod->GetSelection())
        {
            case wxNOT_FOUND:
            {
                asLogError(_("Wrong method selection."));
                break;
            }
            case 0: // Single
            {
                asLogMessage(_("Proceeding to single assessment."));
                m_MethodCalibrator = new asMethodCalibratorSingle();
                break;
            }
            case 1: // Classic
            {
                asLogMessage(_("Proceeding to classic calibration."));
                m_MethodCalibrator = new asMethodCalibratorClassic();
                break;
            }
            case 2: // Classic+
            {
                asLogMessage(_("Proceeding to classic+ calibration."));
                m_MethodCalibrator = new asMethodCalibratorClassicPlus();
                break;
            }
            case 3: // Variables exploration with classic+
            {
                asLogMessage(_("Proceeding to variables exploration."));
                m_MethodCalibrator = new asMethodCalibratorClassicPlusVarExplo();
                break;
            }
            case 4: // Scores evaluation
            {
                asLogMessage(_("Proceeding to all scores evaluation."));
                m_MethodCalibrator = new asMethodCalibratorEvaluateAllScores();
                break;
            }
            case 5: // Only predictand values
            {
                asLogMessage(_("Proceeding to predictand values saving."));
                m_MethodCalibrator = new asMethodCalibratorSingleOnlyValues();
                break;
            }
            default:
                asLogError(_("Chosen method not defined yet."));
        }

        if (m_MethodCalibrator)
        {
            m_MethodCalibrator->SetParamsFilePath(m_FilePickerParameters->GetPath());
            m_MethodCalibrator->SetPredictandDBFilePath(m_FilePickerPredictand->GetPath());
            m_MethodCalibrator->SetPredictorDataDir(m_DirPickerPredictor->GetPath());
            m_MethodCalibrator->Manager();
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

    wxDELETE(m_MethodCalibrator);

    wxMessageBox(_("Calibration over."));
}
