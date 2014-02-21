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
#include "asMethodOptimizerNelderMead.h"
#include "asMethodOptimizerRandomSet.h"
#include "asMethodOptimizerGeneticAlgorithms.h"
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

    // Monte Carlo
    wxString MonteCarloRandomNb = pConfig->Read("/Calibration/MonteCarlo/RandomNb", "1000");
    m_TextCtrlMonteCarloRandomNb->SetValue(MonteCarloRandomNb);

    // Nelder Mead optimization
    wxString NelderMeadNbRuns = pConfig->Read("/Calibration/NelderMead/NbRuns", "20");
    m_TextCtrlNelderMeadNbRuns->SetValue(NelderMeadNbRuns);
    wxString NelderMeadRho = pConfig->Read("/Calibration/NelderMead/Rho", "1.0"); // reflection
    m_TextCtrlNelderMeadRho->SetValue(NelderMeadRho);
    wxString NelderMeadChi = pConfig->Read("/Calibration/NelderMead/Chi", "2.0"); // expansion
    m_TextCtrlNelderMeadChi->SetValue(NelderMeadChi);
    wxString NelderMeadGamma = pConfig->Read("/Calibration/NelderMead/Gamma", "0.5"); // contraction
    m_TextCtrlNelderMeadGamma->SetValue(NelderMeadGamma);
    wxString NelderMeadSigma = pConfig->Read("/Calibration/NelderMead/Sigma", "0.5"); // reduction
    m_TextCtrlNelderMeadSigma->SetValue(NelderMeadSigma);

    // Genetic algorithms
    long NaturalSelectionOperator = pConfig->Read("/Calibration/GeneticAlgorithms/NaturalSelectionOperator", 1l);
    m_ChoiceGAsNaturalSelectionOperator->SetSelection((int)NaturalSelectionOperator);
    long CouplesSelectionOperator = pConfig->Read("/Calibration/GeneticAlgorithms/CouplesSelectionOperator", 3l);
    m_ChoiceGAsCouplesSelectionOperator->SetSelection((int)CouplesSelectionOperator);
    long CrossoverOperator = pConfig->Read("/Calibration/GeneticAlgorithms/CrossoverOperator", 1l);
    m_ChoiceGAsCrossoverOperator->SetSelection((int)CrossoverOperator);
    long MutationOperator = pConfig->Read("/Calibration/GeneticAlgorithms/MutationOperator", 0l);
    m_ChoiceGAsMutationOperator->SetSelection((int)MutationOperator);
    wxString GAsRunNumbers = pConfig->Read("/Calibration/GeneticAlgorithms/NbRuns", "20");
    m_TextCtrlGAsRunNumbers->SetValue(GAsRunNumbers);
    wxString GAsPopulationSize = pConfig->Read("/Calibration/GeneticAlgorithms/PopulationSize", "50");
    m_TextCtrlGAsPopulationSize->SetValue(GAsPopulationSize);
    wxString GAsConvergenceStepsNb = pConfig->Read("/Calibration/GeneticAlgorithms/ConvergenceStepsNb", "20");
    m_TextCtrlGAsConvergenceNb->SetValue(GAsConvergenceStepsNb);
    wxString GAsRatioIntermediateGeneration = pConfig->Read("/Calibration/GeneticAlgorithms/RatioIntermediateGeneration", "0.5");
    m_TextCtrlGAsRatioIntermGen->SetValue(GAsRatioIntermediateGeneration);
    bool GAsAllowElitismForTheBest;
    pConfig->Read("/Calibration/GeneticAlgorithms/AllowElitismForTheBest", &GAsAllowElitismForTheBest, true);
    m_CheckBoxGAsAllowElitism->SetValue(GAsAllowElitismForTheBest);
    wxString GAsNaturalSelectionTournamentProbability = pConfig->Read("/Calibration/GeneticAlgorithms/NaturalSelectionTournamentProbability", "0.9");
    m_TextCtrlGAsNaturalSlctTournamentProb->SetValue(GAsNaturalSelectionTournamentProbability);
    wxString GAsCouplesSelectionTournamentNb = pConfig->Read("/Calibration/GeneticAlgorithms/CouplesSelectionTournamentNb", "3");
    m_TextCtrlGAsCouplesSlctTournamentNb->SetValue(GAsCouplesSelectionTournamentNb);
    wxString GAsCrossoverMultiplePointsNb = pConfig->Read("/Calibration/GeneticAlgorithms/CrossoverMultiplePointsNb", "3");
    m_TextCtrlGAsCrossoverMultipleNbPts->SetValue(GAsCrossoverMultiplePointsNb);
    wxString GAsCrossoverBlendingPointsNb = pConfig->Read("/Calibration/GeneticAlgorithms/CrossoverBlendingPointsNb", "2");
    m_TextCtrlGAsCrossoverBlendingNbPts->SetValue(GAsCrossoverBlendingPointsNb);
    bool GAsCrossoverBlendingShareBeta;
    pConfig->Read("/Calibration/GeneticAlgorithms/CrossoverBlendingShareBeta", &GAsCrossoverBlendingShareBeta, true);
    m_CheckBoxGAsCrossoverBlendingShareBeta->SetValue(GAsCrossoverBlendingShareBeta);
    wxString GAsCrossoverLinearPointsNb = pConfig->Read("/Calibration/GeneticAlgorithms/CrossoverLinearPointsNb", "2");
    m_TextCtrlGAsCrossoverLinearNbPts->SetValue(GAsCrossoverLinearPointsNb);
    wxString GAsCrossoverHeuristicPointsNb = pConfig->Read("/Calibration/GeneticAlgorithms/CrossoverHeuristicPointsNb", "2");
    m_TextCtrlGAsCrossoverHeuristicNbPts->SetValue(GAsCrossoverHeuristicPointsNb);
    bool GAsCrossoverHeuristicShareBeta;
    pConfig->Read("/Calibration/GeneticAlgorithms/CrossoverHeuristicShareBeta", &GAsCrossoverHeuristicShareBeta, true);
    m_CheckBoxGAsCrossoverHeuristicShareBeta->SetValue(GAsCrossoverHeuristicShareBeta);
    wxString GAsCrossoverBinaryLikePointsNb = pConfig->Read("/Calibration/GeneticAlgorithms/CrossoverBinaryLikePointsNb", "2");
    m_TextCtrlGAsCrossoverBinLikeNbPts->SetValue(GAsCrossoverBinaryLikePointsNb);
    bool GAsCrossoverBinaryLikeShareBeta;
    pConfig->Read("/Calibration/GeneticAlgorithms/CrossoverBinaryLikeShareBeta", &GAsCrossoverBinaryLikeShareBeta, true);
    m_CheckBoxGAsCrossoverBinLikeShareBeta->SetValue(GAsCrossoverBinaryLikeShareBeta);
    wxString GAsMutationsUniformConstantProbability = pConfig->Read("/Calibration/GeneticAlgorithms/MutationsUniformConstantProbability", "0.2");
    m_TextCtrlGAsMutationsUniformCstProb->SetValue(GAsMutationsUniformConstantProbability);
    wxString GAsMutationsNormalConstantProbability = pConfig->Read("/Calibration/GeneticAlgorithms/MutationsNormalConstantProbability", "0.2");
    m_TextCtrlGAsMutationsNormalCstProb->SetValue(GAsMutationsNormalConstantProbability);
    wxString GAsMutationsNormalConstantStdDevRatioRange = pConfig->Read("/Calibration/GeneticAlgorithms/MutationsNormalConstantStdDevRatioRange", "0.10");
    m_TextCtrlGAsMutationsNormalCstStdDev->SetValue(GAsMutationsNormalConstantStdDevRatioRange);
    wxString GAsMutationsUniformVariableMaxGensNbVar = pConfig->Read("/Calibration/GeneticAlgorithms/MutationsUniformVariableMaxGensNbVar", "50");
    m_TextCtrlGAsMutationsUniformVarMaxGensNb->SetValue(GAsMutationsUniformVariableMaxGensNbVar);
    wxString GAsMutationsUniformVariableProbabilityStart = pConfig->Read("/Calibration/GeneticAlgorithms/MutationsUniformVariableProbabilityStart", "0.5");
    m_TextCtrlGAsMutationsUniformVarProbStart->SetValue(GAsMutationsUniformVariableProbabilityStart);
    wxString GAsMutationsUniformVariableProbabilityEnd = pConfig->Read("/Calibration/GeneticAlgorithms/MutationsUniformVariableProbabilityEnd", "0.01");
    m_TextCtrlGAsMutationsUniformVarProbEnd->SetValue(GAsMutationsUniformVariableProbabilityEnd);
    wxString GAsMutationsNormalVariableMaxGensNbVarProb = pConfig->Read("/Calibration/GeneticAlgorithms/MutationsNormalVariableMaxGensNbVarProb", "50");
    m_TextCtrlGAsMutationsNormalVarMaxGensNbProb->SetValue(GAsMutationsNormalVariableMaxGensNbVarProb);
    wxString GAsMutationsNormalVariableMaxGensNbVarStdDev = pConfig->Read("/Calibration/GeneticAlgorithms/MutationsNormalVariableMaxGensNbVarStdDev", "50");
    m_TextCtrlGAsMutationsNormalVarMaxGensNbStdDev->SetValue(GAsMutationsNormalVariableMaxGensNbVarStdDev);
    wxString GAsMutationsNormalVariableProbabilityStart = pConfig->Read("/Calibration/GeneticAlgorithms/MutationsNormalVariableProbabilityStart", "0.5");
    m_TextCtrlGAsMutationsNormalVarProbStart->SetValue(GAsMutationsNormalVariableProbabilityStart);
    wxString GAsMutationsNormalVariableProbabilityEnd = pConfig->Read("/Calibration/GeneticAlgorithms/MutationsNormalVariableProbabilityEnd", "0.05");
    m_TextCtrlGAsMutationsNormalVarProbEnd->SetValue(GAsMutationsNormalVariableProbabilityEnd);
    wxString GAsMutationsNormalVariableStdDevStart = pConfig->Read("/Calibration/GeneticAlgorithms/MutationsNormalVariableStdDevStart", "0.5");
    m_TextCtrlGAsMutationsNormalVarStdDevStart->SetValue(GAsMutationsNormalVariableStdDevStart);
    wxString GAsMutationsNormalVariableStdDevEnd = pConfig->Read("/Calibration/GeneticAlgorithms/MutationsNormalVariableStdDevEnd", "0.01");
    m_TextCtrlGAsMutationsNormalVarStdDevEnd->SetValue(GAsMutationsNormalVariableStdDevEnd);
    wxString GAsMutationsNonUniformProb = pConfig->Read("/Calibration/GeneticAlgorithms/MutationsNonUniformProbability", "0.2");
    m_TextCtrlGAsMutationsNonUniformProb->SetValue(GAsMutationsNonUniformProb);
    wxString GAsMutationsNonUniformMaxGensNbVar = pConfig->Read("/Calibration/GeneticAlgorithms/MutationsNonUniformMaxGensNbVar", "50");
    m_TextCtrlGAsMutationsNonUniformGensNb->SetValue(GAsMutationsNonUniformMaxGensNbVar);
    wxString GAsMutationsNonUniformMinRate = pConfig->Read("/Calibration/GeneticAlgorithms/MutationsNonUniformMinRate", "0.20");
    m_TextCtrlGAsMutationsNonUniformMinRate->SetValue(GAsMutationsNonUniformMinRate);
    wxString GAsMutationsMultiScaleProb = pConfig->Read("/Calibration/GeneticAlgorithms/MutationsMultiScaleProbability", "0.20");
    m_TextCtrlGAsMutationsMultiScaleProb->SetValue(GAsMutationsMultiScaleProb);
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

    // Monte Carlo
    wxString MonteCarloRandomNb = m_TextCtrlMonteCarloRandomNb->GetValue();
    pConfig->Write("/Calibration/MonteCarlo/RandomNb", MonteCarloRandomNb);

    // Nelder Mead optimization
    wxString NelderMeadNbRuns = m_TextCtrlNelderMeadNbRuns->GetValue();
    pConfig->Write("/Calibration/NelderMead/NbRuns", NelderMeadNbRuns);
    wxString NelderMeadRho = m_TextCtrlNelderMeadRho->GetValue();
    pConfig->Write("/Calibration/NelderMead/Rho", NelderMeadRho); // reflection
    wxString NelderMeadChi = m_TextCtrlNelderMeadChi->GetValue();
    pConfig->Write("/Calibration/NelderMead/Chi", NelderMeadChi); // expansion
    wxString NelderMeadGamma = m_TextCtrlNelderMeadGamma->GetValue();
    pConfig->Write("/Calibration/NelderMead/Gamma", NelderMeadGamma); // contraction
    wxString NelderMeadSigma = m_TextCtrlNelderMeadSigma->GetValue();
    pConfig->Write("/Calibration/NelderMead/Sigma", NelderMeadSigma); // reduction

    // Genetic algorithms
    long NaturalSelectionOperator = m_ChoiceGAsNaturalSelectionOperator->GetSelection();
    pConfig->Write("/Calibration/GeneticAlgorithms/NaturalSelectionOperator", NaturalSelectionOperator);
    long CouplesSelectionOperator = m_ChoiceGAsCouplesSelectionOperator->GetSelection();
    pConfig->Write("/Calibration/GeneticAlgorithms/CouplesSelectionOperator", CouplesSelectionOperator);
    long CrossoverOperator = m_ChoiceGAsCrossoverOperator->GetSelection();
    pConfig->Write("/Calibration/GeneticAlgorithms/CrossoverOperator", CrossoverOperator);
    long MutationOperator = m_ChoiceGAsMutationOperator->GetSelection();
    pConfig->Write("/Calibration/GeneticAlgorithms/MutationOperator", MutationOperator);
    wxString GAsRunNumbers = m_TextCtrlGAsRunNumbers->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/NbRuns", GAsRunNumbers);
    wxString GAsPopulationSize = m_TextCtrlGAsPopulationSize->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/PopulationSize", GAsPopulationSize);
    wxString GAsConvergenceStepsNb = m_TextCtrlGAsConvergenceNb->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/ConvergenceStepsNb", GAsConvergenceStepsNb);
    wxString GAsRatioIntermediateGeneration = m_TextCtrlGAsRatioIntermGen->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/RatioIntermediateGeneration", GAsRatioIntermediateGeneration);
    bool GAsAllowElitismForTheBest = m_CheckBoxGAsAllowElitism->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/AllowElitismForTheBest", GAsAllowElitismForTheBest);
    wxString GAsNaturalSelectionTournamentProbability = m_TextCtrlGAsNaturalSlctTournamentProb->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/NaturalSelectionTournamentProbability", GAsNaturalSelectionTournamentProbability);
    wxString GAsCouplesSelectionTournamentNb = m_TextCtrlGAsCouplesSlctTournamentNb->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/CouplesSelectionTournamentNb", GAsCouplesSelectionTournamentNb);
    wxString GAsCrossoverMultiplePointsNb = m_TextCtrlGAsCrossoverMultipleNbPts->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/CrossoverMultiplePointsNb", GAsCrossoverMultiplePointsNb);
    wxString GAsCrossoverBlendingPointsNb = m_TextCtrlGAsCrossoverBlendingNbPts->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/CrossoverBlendingPointsNb", GAsCrossoverBlendingPointsNb);
    bool GAsCrossoverBlendingShareBeta = m_CheckBoxGAsCrossoverBlendingShareBeta->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/CrossoverBlendingShareBeta", GAsCrossoverBlendingShareBeta);
    wxString GAsCrossoverLinearPointsNb = m_TextCtrlGAsCrossoverLinearNbPts->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/CrossoverLinearPointsNb", GAsCrossoverLinearPointsNb);
    wxString GAsCrossoverHeuristicPointsNb = m_TextCtrlGAsCrossoverHeuristicNbPts->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/CrossoverHeuristicPointsNb", GAsCrossoverHeuristicPointsNb);
    bool GAsCrossoverHeuristicShareBeta = m_CheckBoxGAsCrossoverHeuristicShareBeta->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/CrossoverHeuristicShareBeta", GAsCrossoverHeuristicShareBeta);
    wxString GAsCrossoverBinaryLikePointsNb = m_TextCtrlGAsCrossoverBinLikeNbPts->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/CrossoverBinaryLikePointsNb", GAsCrossoverBinaryLikePointsNb);
    bool GAsCrossoverBinaryLikeShareBeta = m_CheckBoxGAsCrossoverBinLikeShareBeta->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/CrossoverBinaryLikeShareBeta", GAsCrossoverBinaryLikeShareBeta);
    wxString GAsMutationsUniformConstantProbability = m_TextCtrlGAsMutationsUniformCstProb->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/MutationsUniformConstantProbability", GAsMutationsUniformConstantProbability);
    wxString GAsMutationsNormalConstantProbability = m_TextCtrlGAsMutationsNormalCstProb->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/MutationsNormalConstantProbability", GAsMutationsNormalConstantProbability);
    wxString GAsMutationsNormalConstantStdDevRatioRange = m_TextCtrlGAsMutationsNormalCstStdDev->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/MutationsNormalConstantStdDevRatioRange", GAsMutationsNormalConstantStdDevRatioRange);
    wxString GAsMutationsUniformVariableMaxGensNbVar = m_TextCtrlGAsMutationsUniformVarMaxGensNb->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/MutationsUniformVariableMaxGensNbVar", GAsMutationsUniformVariableMaxGensNbVar);
    wxString GAsMutationsUniformVariableProbabilityStart = m_TextCtrlGAsMutationsUniformVarProbStart->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/MutationsUniformVariableProbabilityStart", GAsMutationsUniformVariableProbabilityStart);
    wxString GAsMutationsUniformVariableProbabilityEnd = m_TextCtrlGAsMutationsUniformVarProbEnd->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/MutationsUniformVariableProbabilityEnd", GAsMutationsUniformVariableProbabilityEnd);
    wxString GAsMutationsNormalVariableMaxGensNbVarProb = m_TextCtrlGAsMutationsNormalVarMaxGensNbProb->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/MutationsNormalVariableMaxGensNbVarProb", GAsMutationsNormalVariableMaxGensNbVarProb);
    wxString GAsMutationsNormalVariableMaxGensNbVarStdDev = m_TextCtrlGAsMutationsNormalVarMaxGensNbStdDev->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/MutationsNormalVariableMaxGensNbVarStdDev", GAsMutationsNormalVariableMaxGensNbVarStdDev);
    wxString GAsMutationsNormalVariableProbabilityStart = m_TextCtrlGAsMutationsNormalVarProbStart->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/MutationsNormalVariableProbabilityStart", GAsMutationsNormalVariableProbabilityStart);
    wxString GAsMutationsNormalVariableProbabilityEnd = m_TextCtrlGAsMutationsNormalVarProbEnd->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/MutationsNormalVariableProbabilityEnd", GAsMutationsNormalVariableProbabilityEnd);
    wxString GAsMutationsNormalVariableStdDevStart = m_TextCtrlGAsMutationsNormalVarStdDevStart->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/MutationsNormalVariableStdDevStart", GAsMutationsNormalVariableStdDevStart);
    wxString GAsMutationsNormalVariableStdDevEnd = m_TextCtrlGAsMutationsNormalVarStdDevEnd->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/MutationsNormalVariableStdDevEnd", GAsMutationsNormalVariableStdDevEnd);
    wxString GAsMutationsNonUniformProb = m_TextCtrlGAsMutationsNonUniformProb->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/MutationsNonUniformProbability", GAsMutationsNonUniformProb);
    wxString GAsMutationsNonUniformMaxGensNbVar = m_TextCtrlGAsMutationsNonUniformGensNb->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/MutationsNonUniformMaxGensNbVar", GAsMutationsNonUniformMaxGensNbVar);
    wxString GAsMutationsNonUniformMinRate = m_TextCtrlGAsMutationsNonUniformMinRate->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/MutationsNonUniformMinRate", GAsMutationsNonUniformMinRate);
    wxString GAsMutationsMultiScaleProb = m_TextCtrlGAsMutationsMultiScaleProb->GetValue();
    pConfig->Write("/Calibration/GeneticAlgorithms/MutationsMultiScaleProbability", GAsMutationsMultiScaleProb);

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
                m_MethodCalibrator = new asMethodCalibratorSingle();
                break;
            }
            case 1: // Classic
            {
                m_MethodCalibrator = new asMethodCalibratorClassic();
                break;
            }
            case 2: // Classic+
            {
                m_MethodCalibrator = new asMethodCalibratorClassicPlus();
                break;
            }
            case 3: // Variables exploration with classic+
            {
                m_MethodCalibrator = new asMethodCalibratorClassicPlusVarExplo();
                break;
            }
            case 4: // Optimization Nelder-Mead
            {
                m_MethodCalibrator = new asMethodOptimizerNelderMead();
                break;
            }
            case 5: // Random sets
            {
                m_MethodCalibrator = new asMethodOptimizerRandomSet();
                break;
            }
            case 6: // Genetic algorithms
            {
                m_MethodCalibrator = new asMethodOptimizerGeneticAlgorithms();
                break;
            }
            case 7: // Scores evaluation
            {
                m_MethodCalibrator = new asMethodCalibratorEvaluateAllScores();
                break;
            }
            case 8: // Only predictand values
            {
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
            m_MethodCalibrator->Cleanup();
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
