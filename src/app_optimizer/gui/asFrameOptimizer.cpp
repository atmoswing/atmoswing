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

#include "asFrameOptimizer.h"

#include "wx/fileconf.h"

#include "asMethodCalibratorClassic.h"
#include "asMethodCalibratorClassicVarExplo.h"
#include "asMethodCalibratorSingle.h"
#include "asMethodOptimizerRandomSet.h"
#include "asMethodOptimizerGeneticAlgorithms.h"
#include "asMethodCalibratorEvaluateAllScores.h"
#include "asMethodCalibratorSingleOnlyValues.h"
#include "images.h"
#include "asFramePreferencesOptimizer.h"
#include "asFrameAbout.h"


asFrameOptimizer::asFrameOptimizer(wxWindow *parent)
        : asFrameOptimizerVirtual(parent)
{
    m_logWindow = NULL;
    m_methodCalibrator = NULL;

    // Toolbar
    m_toolBar->AddTool(asID_RUN, wxT("Run"), *_img_run, *_img_run, wxITEM_NORMAL, _("Run optimizer"),
                       _("Run optimizer now"), NULL);
    m_toolBar->AddTool(asID_CANCEL, wxT("Cancel"), *_img_stop, *_img_stop, wxITEM_NORMAL, _("Cancel optimization"),
                       _("Cancel current optimization"), NULL);
    m_toolBar->AddTool(asID_PREFERENCES, wxT("Preferences"), *_img_preferences, *_img_preferences, wxITEM_NORMAL,
                       _("Preferences"), _("Preferences"), NULL);
    m_toolBar->Realize();

    // Connect events
    this->Connect(asID_RUN, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler(asFrameOptimizer::Launch));
    this->Connect(asID_CANCEL, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler(asFrameOptimizer::Cancel));
    this->Connect(asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED,
                  wxCommandEventHandler(asFrameOptimizer::OpenFramePreferences));

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

asFrameOptimizer::~asFrameOptimizer()
{
    // Disconnect events
    this->Disconnect(asID_RUN, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler(asFrameOptimizer::Launch));
    this->Disconnect(asID_CANCEL, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler(asFrameOptimizer::Cancel));
    this->Disconnect(asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED,
                     wxCommandEventHandler(asFrameOptimizer::OpenFramePreferences));
}

void asFrameOptimizer::OnInit()
{
    // Set the defaults
    LoadOptions();
    DisplayLogLevelMenu();
}

void asFrameOptimizer::Update()
{
    DisplayLogLevelMenu();
}

void asFrameOptimizer::OpenFramePreferences(wxCommandEvent &event)
{
    asFramePreferencesOptimizer *frame = new asFramePreferencesOptimizer(this);
    frame->Fit();
    frame->Show();
}

void asFrameOptimizer::OpenFrameAbout(wxCommandEvent &event)
{
    asFrameAbout *frame = new asFrameAbout(this);
    frame->Fit();
    frame->Show();
}

void asFrameOptimizer::OnShowLog(wxCommandEvent &event)
{
    wxASSERT(m_logWindow);
    m_logWindow->DoShow();
}

void asFrameOptimizer::OnLogLevel1(wxCommandEvent &event)
{
    Log().SetLevel(1);
    m_menuLogLevel->FindItemByPosition(0)->Check(true);
    m_menuLogLevel->FindItemByPosition(1)->Check(false);
    m_menuLogLevel->FindItemByPosition(2)->Check(false);
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Write("/General/LogLevel", 1l);
    ThreadsManager().CritSectionConfig().Leave();
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame)
        prefFrame->Update();
}

void asFrameOptimizer::OnLogLevel2(wxCommandEvent &event)
{
    Log().SetLevel(2);
    m_menuLogLevel->FindItemByPosition(0)->Check(false);
    m_menuLogLevel->FindItemByPosition(1)->Check(true);
    m_menuLogLevel->FindItemByPosition(2)->Check(false);
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Write("/General/LogLevel", 2l);
    ThreadsManager().CritSectionConfig().Leave();
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame)
        prefFrame->Update();
}

void asFrameOptimizer::OnLogLevel3(wxCommandEvent &event)
{
    Log().SetLevel(3);
    m_menuLogLevel->FindItemByPosition(0)->Check(false);
    m_menuLogLevel->FindItemByPosition(1)->Check(false);
    m_menuLogLevel->FindItemByPosition(2)->Check(true);
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Write("/General/LogLevel", 3l);
    ThreadsManager().CritSectionConfig().Leave();
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame)
        prefFrame->Update();
}

void asFrameOptimizer::DisplayLogLevelMenu()
{
    // Set log level in the menu
    ThreadsManager().CritSectionConfig().Enter();
    int logLevel = (int) wxFileConfig::Get()->Read("/General/LogLevel", 2l);
    ThreadsManager().CritSectionConfig().Leave();
    m_menuLogLevel->FindItemByPosition(0)->Check(false);
    m_menuLogLevel->FindItemByPosition(1)->Check(false);
    m_menuLogLevel->FindItemByPosition(2)->Check(false);
    switch (logLevel) {
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

void asFrameOptimizer::Cancel(wxCommandEvent &event)
{
    if (m_methodCalibrator) {
        m_methodCalibrator->Cancel();
    }
}

void asFrameOptimizer::LoadOptions()
{
    // General stuff
    wxConfigBase *pConfig = wxFileConfig::Get();
    long MethodSelection = pConfig->Read("/Optimizer/MethodSelection", 0l);
    m_choiceMethod->SetSelection((int) MethodSelection);
    wxString ParametersFilePath = pConfig->Read("/Optimizer/ParametersFilePath", wxEmptyString);
    m_filePickerParameters->SetPath(ParametersFilePath);
    wxString PredictandDBFilePath = pConfig->Read("/Paths/PredictandDBFilePath", wxEmptyString);
    m_filePickerPredictand->SetPath(PredictandDBFilePath);
    wxString PredictorDir = pConfig->Read("/Paths/PredictorDir", wxEmptyString);
    m_dirPickerPredictor->SetPath(PredictorDir);
    wxString OptimizerResultsDir = pConfig->Read("/Paths/OptimizerResultsDir",
                                                 asConfig::GetDocumentsDir() + "AtmoSwing" + DS + "Optimizer");
    m_dirPickerCalibrationResults->SetPath(OptimizerResultsDir);
    bool parallelEvaluations;
    pConfig->Read("/Optimizer/ParallelEvaluations", &parallelEvaluations, false);
    m_checkBoxParallelEvaluations->SetValue(parallelEvaluations);

    // Classic+ calibration
    wxString ClassicPlusResizingIterations = pConfig->Read("/Optimizer/ClassicPlus/ResizingIterations", "1");
    m_textCtrlClassicPlusResizingIterations->SetValue(ClassicPlusResizingIterations);
    wxString ClassicPlusStepsLatPertinenceMap = pConfig->Read("/Optimizer/ClassicPlus/StepsLatPertinenceMap", "2");
    m_textCtrlClassicPlusStepsLatPertinenceMap->SetValue(ClassicPlusStepsLatPertinenceMap);
    wxString ClassicPlusStepsLonPertinenceMap = pConfig->Read("/Optimizer/ClassicPlus/StepsLonPertinenceMap", "2");
    m_textCtrlClassicPlusStepsLonPertinenceMap->SetValue(ClassicPlusStepsLonPertinenceMap);
    bool proceedSequentially;
    pConfig->Read("/Optimizer/ClassicPlus/ProceedSequentially", &proceedSequentially, true);
    m_checkBoxProceedSequentially->SetValue(proceedSequentially);

    // Variables exploration
    wxString VarExploStep = pConfig->Read("/Optimizer/VariablesExplo/Step");
    m_textCtrlVarExploStepToExplore->SetValue(VarExploStep);

    // Monte Carlo
    wxString MonteCarloRandomNb = pConfig->Read("/Optimizer/MonteCarlo/RandomNb", "1000");
    m_textCtrlMonteCarloRandomNb->SetValue(MonteCarloRandomNb);

    // Genetic algorithms
    long NaturalSelectionOperator = pConfig->Read("/Optimizer/GeneticAlgorithms/NaturalSelectionOperator", 1l);
    m_choiceGAsNaturalSelectionOperator->SetSelection((int) NaturalSelectionOperator);
    long CouplesSelectionOperator = pConfig->Read("/Optimizer/GeneticAlgorithms/CouplesSelectionOperator", 3l);
    m_choiceGAsCouplesSelectionOperator->SetSelection((int) CouplesSelectionOperator);
    long CrossoverOperator = pConfig->Read("/Optimizer/GeneticAlgorithms/CrossoverOperator", 1l);
    m_choiceGAsCrossoverOperator->SetSelection((int) CrossoverOperator);
    long MutationOperator = pConfig->Read("/Optimizer/GeneticAlgorithms/MutationOperator", 0l);
    m_choiceGAsMutationOperator->SetSelection((int) MutationOperator);
    wxString GAsRunNumbers = pConfig->Read("/Optimizer/GeneticAlgorithms/NbRuns", "20");
    m_textCtrlGAsRunNumbers->SetValue(GAsRunNumbers);
    wxString GAsPopulationSize = pConfig->Read("/Optimizer/GeneticAlgorithms/PopulationSize", "50");
    m_textCtrlGAsPopulationSize->SetValue(GAsPopulationSize);
    wxString GAsConvergenceStepsNb = pConfig->Read("/Optimizer/GeneticAlgorithms/ConvergenceStepsNb", "20");
    m_textCtrlGAsConvergenceNb->SetValue(GAsConvergenceStepsNb);
    wxString GAsRatioIntermediateGeneration = pConfig->Read("/Optimizer/GeneticAlgorithms/RatioIntermediateGeneration",
                                                            "0.5");
    m_textCtrlGAsRatioIntermGen->SetValue(GAsRatioIntermediateGeneration);
    bool GAsAllowElitismForTheBest;
    pConfig->Read("/Optimizer/GeneticAlgorithms/AllowElitismForTheBest", &GAsAllowElitismForTheBest, true);
    m_checkBoxGAsAllowElitism->SetValue(GAsAllowElitismForTheBest);
    wxString GAsNaturalSelectionTournamentProbability = pConfig->Read(
            "/Optimizer/GeneticAlgorithms/NaturalSelectionTournamentProbability", "0.9");
    m_textCtrlGAsNaturalSlctTournamentProb->SetValue(GAsNaturalSelectionTournamentProbability);
    wxString GAsCouplesSelectionTournamentNb = pConfig->Read(
            "/Optimizer/GeneticAlgorithms/CouplesSelectionTournamentNb", "3");
    m_textCtrlGAsCouplesSlctTournamentNb->SetValue(GAsCouplesSelectionTournamentNb);
    wxString GAsCrossoverMultiplePointsNb = pConfig->Read("/Optimizer/GeneticAlgorithms/CrossoverMultiplePointsNb",
                                                          "3");
    m_textCtrlGAsCrossoverMultipleNbPts->SetValue(GAsCrossoverMultiplePointsNb);
    wxString GAsCrossoverBlendingPointsNb = pConfig->Read("/Optimizer/GeneticAlgorithms/CrossoverBlendingPointsNb",
                                                          "2");
    m_textCtrlGAsCrossoverBlendingNbPts->SetValue(GAsCrossoverBlendingPointsNb);
    bool GAsCrossoverBlendingShareBeta;
    pConfig->Read("/Optimizer/GeneticAlgorithms/CrossoverBlendingShareBeta", &GAsCrossoverBlendingShareBeta, true);
    m_checkBoxGAsCrossoverBlendingShareBeta->SetValue(GAsCrossoverBlendingShareBeta);
    wxString GAsCrossoverLinearPointsNb = pConfig->Read("/Optimizer/GeneticAlgorithms/CrossoverLinearPointsNb", "2");
    m_textCtrlGAsCrossoverLinearNbPts->SetValue(GAsCrossoverLinearPointsNb);
    wxString GAsCrossoverHeuristicPointsNb = pConfig->Read("/Optimizer/GeneticAlgorithms/CrossoverHeuristicPointsNb",
                                                           "2");
    m_textCtrlGAsCrossoverHeuristicNbPts->SetValue(GAsCrossoverHeuristicPointsNb);
    bool GAsCrossoverHeuristicShareBeta;
    pConfig->Read("/Optimizer/GeneticAlgorithms/CrossoverHeuristicShareBeta", &GAsCrossoverHeuristicShareBeta, true);
    m_checkBoxGAsCrossoverHeuristicShareBeta->SetValue(GAsCrossoverHeuristicShareBeta);
    wxString GAsCrossoverBinaryLikePointsNb = pConfig->Read("/Optimizer/GeneticAlgorithms/CrossoverBinaryLikePointsNb",
                                                            "2");
    m_textCtrlGAsCrossoverBinLikeNbPts->SetValue(GAsCrossoverBinaryLikePointsNb);
    bool GAsCrossoverBinaryLikeShareBeta;
    pConfig->Read("/Optimizer/GeneticAlgorithms/CrossoverBinaryLikeShareBeta", &GAsCrossoverBinaryLikeShareBeta, true);
    m_checkBoxGAsCrossoverBinLikeShareBeta->SetValue(GAsCrossoverBinaryLikeShareBeta);
    wxString GAsMutationsUniformConstantProbability = pConfig->Read(
            "/Optimizer/GeneticAlgorithms/MutationsUniformConstantProbability", "0.2");
    m_textCtrlGAsMutationsUniformCstProb->SetValue(GAsMutationsUniformConstantProbability);
    wxString GAsMutationsNormalConstantProbability = pConfig->Read(
            "/Optimizer/GeneticAlgorithms/MutationsNormalConstantProbability", "0.2");
    m_textCtrlGAsMutationsNormalCstProb->SetValue(GAsMutationsNormalConstantProbability);
    wxString GAsMutationsNormalConstantStdDevRatioRange = pConfig->Read(
            "/Optimizer/GeneticAlgorithms/MutationsNormalConstantStdDevRatioRange", "0.10");
    m_textCtrlGAsMutationsNormalCstStdDev->SetValue(GAsMutationsNormalConstantStdDevRatioRange);
    wxString GAsMutationsUniformVariableMaxGensNbVar = pConfig->Read(
            "/Optimizer/GeneticAlgorithms/MutationsUniformVariableMaxGensNbVar", "50");
    m_textCtrlGAsMutationsUniformVarMaxGensNb->SetValue(GAsMutationsUniformVariableMaxGensNbVar);
    wxString GAsMutationsUniformVariableProbabilityStart = pConfig->Read(
            "/Optimizer/GeneticAlgorithms/MutationsUniformVariableProbabilityStart", "0.5");
    m_textCtrlGAsMutationsUniformVarProbStart->SetValue(GAsMutationsUniformVariableProbabilityStart);
    wxString GAsMutationsUniformVariableProbabilityEnd = pConfig->Read(
            "/Optimizer/GeneticAlgorithms/MutationsUniformVariableProbabilityEnd", "0.01");
    m_textCtrlGAsMutationsUniformVarProbEnd->SetValue(GAsMutationsUniformVariableProbabilityEnd);
    wxString GAsMutationsNormalVariableMaxGensNbVarProb = pConfig->Read(
            "/Optimizer/GeneticAlgorithms/MutationsNormalVariableMaxGensNbVarProb", "50");
    m_textCtrlGAsMutationsNormalVarMaxGensNbProb->SetValue(GAsMutationsNormalVariableMaxGensNbVarProb);
    wxString GAsMutationsNormalVariableMaxGensNbVarStdDev = pConfig->Read(
            "/Optimizer/GeneticAlgorithms/MutationsNormalVariableMaxGensNbVarStdDev", "50");
    m_textCtrlGAsMutationsNormalVarMaxGensNbStdDev->SetValue(GAsMutationsNormalVariableMaxGensNbVarStdDev);
    wxString GAsMutationsNormalVariableProbabilityStart = pConfig->Read(
            "/Optimizer/GeneticAlgorithms/MutationsNormalVariableProbabilityStart", "0.5");
    m_textCtrlGAsMutationsNormalVarProbStart->SetValue(GAsMutationsNormalVariableProbabilityStart);
    wxString GAsMutationsNormalVariableProbabilityEnd = pConfig->Read(
            "/Optimizer/GeneticAlgorithms/MutationsNormalVariableProbabilityEnd", "0.05");
    m_textCtrlGAsMutationsNormalVarProbEnd->SetValue(GAsMutationsNormalVariableProbabilityEnd);
    wxString GAsMutationsNormalVariableStdDevStart = pConfig->Read(
            "/Optimizer/GeneticAlgorithms/MutationsNormalVariableStdDevStart", "0.5");
    m_textCtrlGAsMutationsNormalVarStdDevStart->SetValue(GAsMutationsNormalVariableStdDevStart);
    wxString GAsMutationsNormalVariableStdDevEnd = pConfig->Read(
            "/Optimizer/GeneticAlgorithms/MutationsNormalVariableStdDevEnd", "0.01");
    m_textCtrlGAsMutationsNormalVarStdDevEnd->SetValue(GAsMutationsNormalVariableStdDevEnd);
    wxString GAsMutationsNonUniformProb = pConfig->Read("/Optimizer/GeneticAlgorithms/MutationsNonUniformProbability",
                                                        "0.2");
    m_textCtrlGAsMutationsNonUniformProb->SetValue(GAsMutationsNonUniformProb);
    wxString GAsMutationsNonUniformMaxGensNbVar = pConfig->Read(
            "/Optimizer/GeneticAlgorithms/MutationsNonUniformMaxGensNbVar", "50");
    m_textCtrlGAsMutationsNonUniformGensNb->SetValue(GAsMutationsNonUniformMaxGensNbVar);
    wxString GAsMutationsNonUniformMinRate = pConfig->Read("/Optimizer/GeneticAlgorithms/MutationsNonUniformMinRate",
                                                           "0.20");
    m_textCtrlGAsMutationsNonUniformMinRate->SetValue(GAsMutationsNonUniformMinRate);
    wxString GAsMutationsMultiScaleProb = pConfig->Read("/Optimizer/GeneticAlgorithms/MutationsMultiScaleProbability",
                                                        "0.20");
    m_textCtrlGAsMutationsMultiScaleProb->SetValue(GAsMutationsMultiScaleProb);
}

void asFrameOptimizer::OnSaveDefault(wxCommandEvent &event) const
{
    SaveOptions();
}

void asFrameOptimizer::SaveOptions() const
{
    // General stuff
    wxConfigBase *pConfig = wxFileConfig::Get();
    long MethodSelection = (long) m_choiceMethod->GetSelection();
    pConfig->Write("/Optimizer/MethodSelection", MethodSelection);
    wxString ParametersFilePath = m_filePickerParameters->GetPath();
    pConfig->Write("/Optimizer/ParametersFilePath", ParametersFilePath);
    wxString PredictandDBFilePath = m_filePickerPredictand->GetPath();
    pConfig->Write("/Paths/PredictandDBFilePath", PredictandDBFilePath);
    wxString PredictorDir = m_dirPickerPredictor->GetPath();
    pConfig->Write("/Paths/PredictorDir", PredictorDir);
    wxString OptimizerResultsDir = m_dirPickerCalibrationResults->GetPath();
    pConfig->Write("/Paths/OptimizerResultsDir", OptimizerResultsDir);
    bool parallelEvaluations = m_checkBoxParallelEvaluations->GetValue();
    pConfig->Write("/Optimizer/ParallelEvaluations", parallelEvaluations);

    // Classic+ calibration
    wxString ClassicPlusResizingIterations = m_textCtrlClassicPlusResizingIterations->GetValue();
    pConfig->Write("/Optimizer/ClassicPlus/ResizingIterations", ClassicPlusResizingIterations);
    wxString ClassicPlusStepsLatPertinenceMap = m_textCtrlClassicPlusStepsLatPertinenceMap->GetValue();
    pConfig->Write("/Optimizer/ClassicPlus/StepsLatPertinenceMap", ClassicPlusStepsLatPertinenceMap);
    wxString ClassicPlusStepsLonPertinenceMap = m_textCtrlClassicPlusStepsLonPertinenceMap->GetValue();
    pConfig->Write("/Optimizer/ClassicPlus/StepsLonPertinenceMap", ClassicPlusStepsLonPertinenceMap);
    bool proceedSequentially = m_checkBoxProceedSequentially->GetValue();
    pConfig->Write("/Optimizer/ClassicPlus/ProceedSequentially", proceedSequentially);

    // Variables exploration
    wxString VarExploStep = m_textCtrlVarExploStepToExplore->GetValue();
    pConfig->Write("/Optimizer/VariablesExplo/Step", VarExploStep);

    // Monte Carlo
    wxString MonteCarloRandomNb = m_textCtrlMonteCarloRandomNb->GetValue();
    pConfig->Write("/Optimizer/MonteCarlo/RandomNb", MonteCarloRandomNb);

    // Genetic algorithms
    long NaturalSelectionOperator = m_choiceGAsNaturalSelectionOperator->GetSelection();
    pConfig->Write("/Optimizer/GeneticAlgorithms/NaturalSelectionOperator", NaturalSelectionOperator);
    long CouplesSelectionOperator = m_choiceGAsCouplesSelectionOperator->GetSelection();
    pConfig->Write("/Optimizer/GeneticAlgorithms/CouplesSelectionOperator", CouplesSelectionOperator);
    long CrossoverOperator = m_choiceGAsCrossoverOperator->GetSelection();
    pConfig->Write("/Optimizer/GeneticAlgorithms/CrossoverOperator", CrossoverOperator);
    long MutationOperator = m_choiceGAsMutationOperator->GetSelection();
    pConfig->Write("/Optimizer/GeneticAlgorithms/MutationOperator", MutationOperator);
    wxString GAsRunNumbers = m_textCtrlGAsRunNumbers->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/NbRuns", GAsRunNumbers);
    wxString GAsPopulationSize = m_textCtrlGAsPopulationSize->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/PopulationSize", GAsPopulationSize);
    wxString GAsConvergenceStepsNb = m_textCtrlGAsConvergenceNb->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/ConvergenceStepsNb", GAsConvergenceStepsNb);
    wxString GAsRatioIntermediateGeneration = m_textCtrlGAsRatioIntermGen->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/RatioIntermediateGeneration", GAsRatioIntermediateGeneration);
    bool GAsAllowElitismForTheBest = m_checkBoxGAsAllowElitism->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/AllowElitismForTheBest", GAsAllowElitismForTheBest);
    wxString GAsNaturalSelectionTournamentProbability = m_textCtrlGAsNaturalSlctTournamentProb->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/NaturalSelectionTournamentProbability",
                   GAsNaturalSelectionTournamentProbability);
    wxString GAsCouplesSelectionTournamentNb = m_textCtrlGAsCouplesSlctTournamentNb->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/CouplesSelectionTournamentNb", GAsCouplesSelectionTournamentNb);
    wxString GAsCrossoverMultiplePointsNb = m_textCtrlGAsCrossoverMultipleNbPts->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/CrossoverMultiplePointsNb", GAsCrossoverMultiplePointsNb);
    wxString GAsCrossoverBlendingPointsNb = m_textCtrlGAsCrossoverBlendingNbPts->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/CrossoverBlendingPointsNb", GAsCrossoverBlendingPointsNb);
    bool GAsCrossoverBlendingShareBeta = m_checkBoxGAsCrossoverBlendingShareBeta->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/CrossoverBlendingShareBeta", GAsCrossoverBlendingShareBeta);
    wxString GAsCrossoverLinearPointsNb = m_textCtrlGAsCrossoverLinearNbPts->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/CrossoverLinearPointsNb", GAsCrossoverLinearPointsNb);
    wxString GAsCrossoverHeuristicPointsNb = m_textCtrlGAsCrossoverHeuristicNbPts->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/CrossoverHeuristicPointsNb", GAsCrossoverHeuristicPointsNb);
    bool GAsCrossoverHeuristicShareBeta = m_checkBoxGAsCrossoverHeuristicShareBeta->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/CrossoverHeuristicShareBeta", GAsCrossoverHeuristicShareBeta);
    wxString GAsCrossoverBinaryLikePointsNb = m_textCtrlGAsCrossoverBinLikeNbPts->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/CrossoverBinaryLikePointsNb", GAsCrossoverBinaryLikePointsNb);
    bool GAsCrossoverBinaryLikeShareBeta = m_checkBoxGAsCrossoverBinLikeShareBeta->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/CrossoverBinaryLikeShareBeta", GAsCrossoverBinaryLikeShareBeta);
    wxString GAsMutationsUniformConstantProbability = m_textCtrlGAsMutationsUniformCstProb->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/MutationsUniformConstantProbability",
                   GAsMutationsUniformConstantProbability);
    wxString GAsMutationsNormalConstantProbability = m_textCtrlGAsMutationsNormalCstProb->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/MutationsNormalConstantProbability",
                   GAsMutationsNormalConstantProbability);
    wxString GAsMutationsNormalConstantStdDevRatioRange = m_textCtrlGAsMutationsNormalCstStdDev->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/MutationsNormalConstantStdDevRatioRange",
                   GAsMutationsNormalConstantStdDevRatioRange);
    wxString GAsMutationsUniformVariableMaxGensNbVar = m_textCtrlGAsMutationsUniformVarMaxGensNb->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/MutationsUniformVariableMaxGensNbVar",
                   GAsMutationsUniformVariableMaxGensNbVar);
    wxString GAsMutationsUniformVariableProbabilityStart = m_textCtrlGAsMutationsUniformVarProbStart->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/MutationsUniformVariableProbabilityStart",
                   GAsMutationsUniformVariableProbabilityStart);
    wxString GAsMutationsUniformVariableProbabilityEnd = m_textCtrlGAsMutationsUniformVarProbEnd->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/MutationsUniformVariableProbabilityEnd",
                   GAsMutationsUniformVariableProbabilityEnd);
    wxString GAsMutationsNormalVariableMaxGensNbVarProb = m_textCtrlGAsMutationsNormalVarMaxGensNbProb->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/MutationsNormalVariableMaxGensNbVarProb",
                   GAsMutationsNormalVariableMaxGensNbVarProb);
    wxString GAsMutationsNormalVariableMaxGensNbVarStdDev = m_textCtrlGAsMutationsNormalVarMaxGensNbStdDev->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/MutationsNormalVariableMaxGensNbVarStdDev",
                   GAsMutationsNormalVariableMaxGensNbVarStdDev);
    wxString GAsMutationsNormalVariableProbabilityStart = m_textCtrlGAsMutationsNormalVarProbStart->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/MutationsNormalVariableProbabilityStart",
                   GAsMutationsNormalVariableProbabilityStart);
    wxString GAsMutationsNormalVariableProbabilityEnd = m_textCtrlGAsMutationsNormalVarProbEnd->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/MutationsNormalVariableProbabilityEnd",
                   GAsMutationsNormalVariableProbabilityEnd);
    wxString GAsMutationsNormalVariableStdDevStart = m_textCtrlGAsMutationsNormalVarStdDevStart->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/MutationsNormalVariableStdDevStart",
                   GAsMutationsNormalVariableStdDevStart);
    wxString GAsMutationsNormalVariableStdDevEnd = m_textCtrlGAsMutationsNormalVarStdDevEnd->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/MutationsNormalVariableStdDevEnd",
                   GAsMutationsNormalVariableStdDevEnd);
    wxString GAsMutationsNonUniformProb = m_textCtrlGAsMutationsNonUniformProb->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/MutationsNonUniformProbability", GAsMutationsNonUniformProb);
    wxString GAsMutationsNonUniformMaxGensNbVar = m_textCtrlGAsMutationsNonUniformGensNb->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/MutationsNonUniformMaxGensNbVar", GAsMutationsNonUniformMaxGensNbVar);
    wxString GAsMutationsNonUniformMinRate = m_textCtrlGAsMutationsNonUniformMinRate->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/MutationsNonUniformMinRate", GAsMutationsNonUniformMinRate);
    wxString GAsMutationsMultiScaleProb = m_textCtrlGAsMutationsMultiScaleProb->GetValue();
    pConfig->Write("/Optimizer/GeneticAlgorithms/MutationsMultiScaleProbability", GAsMutationsMultiScaleProb);

    pConfig->Flush();
}

/*
void asFrameOptimizer::OnIdle( wxCommandEvent& event )
{
    wxString state = asGetState();
    m_staticTextState->SetLabel(state);
}
*/
void asFrameOptimizer::Launch(wxCommandEvent &event)
{
    SaveOptions();

    try {
        switch (m_choiceMethod->GetSelection()) {
            case wxNOT_FOUND: {
                wxLogError(_("Wrong method selection."));
                break;
            }
            case 0: // Single
            {
                wxLogVerbose(_("Proceeding to single assessment."));
                m_methodCalibrator = new asMethodCalibratorSingle();
                break;
            }
            case 1: // Classic
            {
                wxLogVerbose(_("Proceeding to classic calibration."));
                m_methodCalibrator = new asMethodCalibratorClassic();
                break;
            }
            case 2: // Classic+
            {
                wxLogVerbose(_("Proceeding to classic+ calibration."));
                m_methodCalibrator = new asMethodCalibratorClassic();
                break;
            }
            case 3: // Variables exploration with classic+
            {
                wxLogVerbose(_("Proceeding to variables exploration."));
                m_methodCalibrator = new asMethodCalibratorClassicVarExplo();
                break;
            }
            case 4: // Random sets
            {
                m_methodCalibrator = new asMethodOptimizerRandomSet();
                break;
            }
            case 5: // Genetic algorithms
            {
                m_methodCalibrator = new asMethodOptimizerGeneticAlgorithms();
                break;
            }
            case 6: // Scores evaluation
            {
                wxLogVerbose(_("Proceeding to all scores evaluation."));
                m_methodCalibrator = new asMethodCalibratorEvaluateAllScores();
                break;
            }
            case 7: // Only predictand values
            {
                wxLogVerbose(_("Proceeding to predictand values saving."));
                m_methodCalibrator = new asMethodCalibratorSingleOnlyValues();
                break;
            }
            default:
                wxLogError(_("Chosen method not defined yet."));
        }

        if (m_methodCalibrator) {
            m_methodCalibrator->SetParamsFilePath(m_filePickerParameters->GetPath());
            m_methodCalibrator->SetPredictandDBFilePath(m_filePickerPredictand->GetPath());
            m_methodCalibrator->SetPredictorDataDir(m_dirPickerPredictor->GetPath());
            m_methodCalibrator->Manager();
        }
    } catch (std::bad_alloc &ba) {
        wxString msg(ba.what(), wxConvUTF8);
        wxLogError(_("Bad allocation caught: %s"), msg);
        wxLogError(_("Failed to process the calibration."));
    } catch (asException &e) {
        wxString fullMessage = e.GetFullMessage();
        if (!fullMessage.IsEmpty()) {
            wxLogError(fullMessage);
        }
        wxLogError(_("Failed to process the optimization."));
    }

    wxDELETE(m_methodCalibrator);

    wxMessageBox(_("Optimizer over."));
}
