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

#include "asFrameDownscaler.h"

#include "wx/fileconf.h"

#include "asMethodDownscalerClassic.h"
#include "asMethodDownscalerClassicVarExplo.h"
#include "asMethodDownscalerSingle.h"
#include "asMethodDownscalerRandomSet.h"
#include "asMethodDownscalerGeneticAlgorithms.h"
#include "asMethodDownscalerEvaluateAllScores.h"
#include "asMethodDownscalerSingleOnlyValues.h"
#include "images.h"
#include "asFramePreferencesDownscaler.h"
#include "asFrameAbout.h"


asFrameDownscaler::asFrameDownscaler(wxWindow *parent)
        : asFrameDownscalerVirtual(parent),
          m_logWindow(nullptr),
          m_methodDownscaler(nullptr)
{
    // Toolbar
    m_toolBar->AddTool(asID_RUN, wxT("Run"), *_img_run, *_img_run, wxITEM_NORMAL, _("Run downscaler"),
                       _("Run downscaler now"), NULL);
    m_toolBar->AddTool(asID_CANCEL, wxT("Cancel"), *_img_stop, *_img_stop, wxITEM_NORMAL, _("Cancel downscaling"),
                       _("Cancel current downscaling"), NULL);
    m_toolBar->AddTool(asID_PREFERENCES, wxT("Preferences"), *_img_preferences, *_img_preferences, wxITEM_NORMAL,
                       _("Preferences"), _("Preferences"), NULL);
    m_toolBar->Realize();

    // Connect events
    this->Connect(asID_RUN, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler(asFrameDownscaler::Launch));
    this->Connect(asID_CANCEL, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler(asFrameDownscaler::Cancel));
    this->Connect(asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED,
                  wxCommandEventHandler(asFrameDownscaler::OpenFramePreferences));

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

asFrameDownscaler::~asFrameDownscaler()
{
    // Disconnect events
    this->Disconnect(asID_RUN, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler(asFrameDownscaler::Launch));
    this->Disconnect(asID_CANCEL, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler(asFrameDownscaler::Cancel));
    this->Disconnect(asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED,
                     wxCommandEventHandler(asFrameDownscaler::OpenFramePreferences));
}

void asFrameDownscaler::OnInit()
{
    // Set the defaults
    LoadOptions();
    DisplayLogLevelMenu();
}

void asFrameDownscaler::Update()
{
    DisplayLogLevelMenu();
}

void asFrameDownscaler::OpenFramePreferences(wxCommandEvent &event)
{
    asFramePreferencesDownscaler *frame = new asFramePreferencesDownscaler(this);
    frame->Fit();
    frame->Show();
}

void asFrameDownscaler::OpenFrameAbout(wxCommandEvent &event)
{
    asFrameAbout *frame = new asFrameAbout(this);
    frame->Fit();
    frame->Show();
}

void asFrameDownscaler::OnShowLog(wxCommandEvent &event)
{
    wxASSERT(m_logWindow);
    m_logWindow->DoShow();
}

void asFrameDownscaler::OnLogLevel1(wxCommandEvent &event)
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

void asFrameDownscaler::OnLogLevel2(wxCommandEvent &event)
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

void asFrameDownscaler::OnLogLevel3(wxCommandEvent &event)
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

void asFrameDownscaler::DisplayLogLevelMenu()
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

void asFrameDownscaler::Cancel(wxCommandEvent &event)
{
    if (m_methodDownscaler) {
        m_methodDownscaler->Cancel();
    }
}

void asFrameDownscaler::LoadOptions()
{
    // General stuff
    wxConfigBase *pConfig = wxFileConfig::Get();
    long methodSelection = pConfig->Read("/Downscaler/MethodSelection", 0l);
    m_choiceMethod->SetSelection((int) methodSelection);
    wxString parametersFilePath = pConfig->Read("/Downscaler/ParametersFilePath", wxEmptyString);
    m_filePickerParameters->SetPath(parametersFilePath);
    wxString predictandDBFilePath = pConfig->Read("/Paths/PredictandDBFilePath", wxEmptyString);
    m_filePickerPredictand->SetPath(predictandDBFilePath);
    wxString predictorDir = pConfig->Read("/Paths/PredictorDir", wxEmptyString);
    m_dirPickerPredictor->SetPath(predictorDir);
    wxString downscalerResultsDir = pConfig->Read("/Paths/DownscalerResultsDir",
                                                 asConfig::GetDocumentsDir() + "AtmoSwing" + DS + "Downscaler");
    m_dirPickerDownscalingResults->SetPath(downscalerResultsDir);
    bool parallelEvaluations;
    pConfig->Read("/Downscaler/ParallelEvaluations", &parallelEvaluations, false);
    m_checkBoxParallelEvaluations->SetValue(parallelEvaluations);

    // Classic+ downscaling
    wxString classicPlusResizingIterations = pConfig->Read("/Downscaler/ClassicPlus/ResizingIterations", "1");
    m_textCtrlClassicPlusResizingIterations->SetValue(classicPlusResizingIterations);
    wxString classicPlusStepsLatPertinenceMap = pConfig->Read("/Downscaler/ClassicPlus/StepsLatPertinenceMap", "2");
    m_textCtrlClassicPlusStepsLatPertinenceMap->SetValue(classicPlusStepsLatPertinenceMap);
    wxString classicPlusStepsLonPertinenceMap = pConfig->Read("/Downscaler/ClassicPlus/StepsLonPertinenceMap", "2");
    m_textCtrlClassicPlusStepsLonPertinenceMap->SetValue(classicPlusStepsLonPertinenceMap);
    bool proceedSequentially;
    pConfig->Read("/Downscaler/ClassicPlus/ProceedSequentially", &proceedSequentially, true);
    m_checkBoxProceedSequentially->SetValue(proceedSequentially);

    // Variables exploration
    wxString varExploStep = pConfig->Read("/Downscaler/VariablesExplo/Step");
    m_textCtrlVarExploStepToExplore->SetValue(varExploStep);

    // Monte Carlo
    wxString monteCarloRandomNb = pConfig->Read("/Downscaler/MonteCarlo/RandomNb", "1000");
    m_textCtrlMonteCarloRandomNb->SetValue(monteCarloRandomNb);

    // Genetic algorithms
    long naturalSelectionOperator = pConfig->Read("/Downscaler/GeneticAlgorithms/NaturalSelectionOperator", 1l);
    m_choiceGAsNaturalSelectionOperator->SetSelection((int) naturalSelectionOperator);
    long couplesSelectionOperator = pConfig->Read("/Downscaler/GeneticAlgorithms/CouplesSelectionOperator", 3l);
    m_choiceGAsCouplesSelectionOperator->SetSelection((int) couplesSelectionOperator);
    long crossoverOperator = pConfig->Read("/Downscaler/GeneticAlgorithms/CrossoverOperator", 1l);
    m_choiceGAsCrossoverOperator->SetSelection((int) crossoverOperator);
    long mutationOperator = pConfig->Read("/Downscaler/GeneticAlgorithms/MutationOperator", 0l);
    m_choiceGAsMutationOperator->SetSelection((int) mutationOperator);
    wxString GAsRunNumbers = pConfig->Read("/Downscaler/GeneticAlgorithms/NbRuns", "20");
    m_textCtrlGAsRunNumbers->SetValue(GAsRunNumbers);
    wxString GAsPopulationSize = pConfig->Read("/Downscaler/GeneticAlgorithms/PopulationSize", "50");
    m_textCtrlGAsPopulationSize->SetValue(GAsPopulationSize);
    wxString GAsConvergenceStepsNb = pConfig->Read("/Downscaler/GeneticAlgorithms/ConvergenceStepsNb", "20");
    m_textCtrlGAsConvergenceNb->SetValue(GAsConvergenceStepsNb);
    wxString GAsRatioIntermediateGeneration = pConfig->Read("/Downscaler/GeneticAlgorithms/RatioIntermediateGeneration",
                                                            "0.5");
    m_textCtrlGAsRatioIntermGen->SetValue(GAsRatioIntermediateGeneration);
    bool GAsAllowElitismForTheBest;
    pConfig->Read("/Downscaler/GeneticAlgorithms/AllowElitismForTheBest", &GAsAllowElitismForTheBest, true);
    m_checkBoxGAsAllowElitism->SetValue(GAsAllowElitismForTheBest);
    wxString GAsNaturalSelectionTournamentProbability = pConfig->Read(
            "/Downscaler/GeneticAlgorithms/NaturalSelectionTournamentProbability", "0.9");
    m_textCtrlGAsNaturalSlctTournamentProb->SetValue(GAsNaturalSelectionTournamentProbability);
    wxString GAsCouplesSelectionTournamentNb = pConfig->Read(
            "/Downscaler/GeneticAlgorithms/CouplesSelectionTournamentNb", "3");
    m_textCtrlGAsCouplesSlctTournamentNb->SetValue(GAsCouplesSelectionTournamentNb);
    wxString GAsCrossoverMultiplePointsNb = pConfig->Read("/Downscaler/GeneticAlgorithms/CrossoverMultiplePointsNb",
                                                          "3");
    m_textCtrlGAsCrossoverMultipleNbPts->SetValue(GAsCrossoverMultiplePointsNb);
    wxString GAsCrossoverBlendingPointsNb = pConfig->Read("/Downscaler/GeneticAlgorithms/CrossoverBlendingPointsNb",
                                                          "2");
    m_textCtrlGAsCrossoverBlendingNbPts->SetValue(GAsCrossoverBlendingPointsNb);
    bool GAsCrossoverBlendingShareBeta;
    pConfig->Read("/Downscaler/GeneticAlgorithms/CrossoverBlendingShareBeta", &GAsCrossoverBlendingShareBeta, true);
    m_checkBoxGAsCrossoverBlendingShareBeta->SetValue(GAsCrossoverBlendingShareBeta);
    wxString GAsCrossoverLinearPointsNb = pConfig->Read("/Downscaler/GeneticAlgorithms/CrossoverLinearPointsNb", "2");
    m_textCtrlGAsCrossoverLinearNbPts->SetValue(GAsCrossoverLinearPointsNb);
    wxString GAsCrossoverHeuristicPointsNb = pConfig->Read("/Downscaler/GeneticAlgorithms/CrossoverHeuristicPointsNb",
                                                           "2");
    m_textCtrlGAsCrossoverHeuristicNbPts->SetValue(GAsCrossoverHeuristicPointsNb);
    bool GAsCrossoverHeuristicShareBeta;
    pConfig->Read("/Downscaler/GeneticAlgorithms/CrossoverHeuristicShareBeta", &GAsCrossoverHeuristicShareBeta, true);
    m_checkBoxGAsCrossoverHeuristicShareBeta->SetValue(GAsCrossoverHeuristicShareBeta);
    wxString GAsCrossoverBinaryLikePointsNb = pConfig->Read("/Downscaler/GeneticAlgorithms/CrossoverBinaryLikePointsNb",
                                                            "2");
    m_textCtrlGAsCrossoverBinLikeNbPts->SetValue(GAsCrossoverBinaryLikePointsNb);
    bool GAsCrossoverBinaryLikeShareBeta;
    pConfig->Read("/Downscaler/GeneticAlgorithms/CrossoverBinaryLikeShareBeta", &GAsCrossoverBinaryLikeShareBeta, true);
    m_checkBoxGAsCrossoverBinLikeShareBeta->SetValue(GAsCrossoverBinaryLikeShareBeta);
    wxString GAsMutationsUniformConstantProbability = pConfig->Read(
            "/Downscaler/GeneticAlgorithms/MutationsUniformConstantProbability", "0.2");
    m_textCtrlGAsMutationsUniformCstProb->SetValue(GAsMutationsUniformConstantProbability);
    wxString GAsMutationsNormalConstantProbability = pConfig->Read(
            "/Downscaler/GeneticAlgorithms/MutationsNormalConstantProbability", "0.2");
    m_textCtrlGAsMutationsNormalCstProb->SetValue(GAsMutationsNormalConstantProbability);
    wxString GAsMutationsNormalConstantStdDevRatioRange = pConfig->Read(
            "/Downscaler/GeneticAlgorithms/MutationsNormalConstantStdDevRatioRange", "0.10");
    m_textCtrlGAsMutationsNormalCstStdDev->SetValue(GAsMutationsNormalConstantStdDevRatioRange);
    wxString GAsMutationsUniformVariableMaxGensNbVar = pConfig->Read(
            "/Downscaler/GeneticAlgorithms/MutationsUniformVariableMaxGensNbVar", "50");
    m_textCtrlGAsMutationsUniformVarMaxGensNb->SetValue(GAsMutationsUniformVariableMaxGensNbVar);
    wxString GAsMutationsUniformVariableProbabilityStart = pConfig->Read(
            "/Downscaler/GeneticAlgorithms/MutationsUniformVariableProbabilityStart", "0.5");
    m_textCtrlGAsMutationsUniformVarProbStart->SetValue(GAsMutationsUniformVariableProbabilityStart);
    wxString GAsMutationsUniformVariableProbabilityEnd = pConfig->Read(
            "/Downscaler/GeneticAlgorithms/MutationsUniformVariableProbabilityEnd", "0.01");
    m_textCtrlGAsMutationsUniformVarProbEnd->SetValue(GAsMutationsUniformVariableProbabilityEnd);
    wxString GAsMutationsNormalVariableMaxGensNbVarProb = pConfig->Read(
            "/Downscaler/GeneticAlgorithms/MutationsNormalVariableMaxGensNbVarProb", "50");
    m_textCtrlGAsMutationsNormalVarMaxGensNbProb->SetValue(GAsMutationsNormalVariableMaxGensNbVarProb);
    wxString GAsMutationsNormalVariableMaxGensNbVarStdDev = pConfig->Read(
            "/Downscaler/GeneticAlgorithms/MutationsNormalVariableMaxGensNbVarStdDev", "50");
    m_textCtrlGAsMutationsNormalVarMaxGensNbStdDev->SetValue(GAsMutationsNormalVariableMaxGensNbVarStdDev);
    wxString GAsMutationsNormalVariableProbabilityStart = pConfig->Read(
            "/Downscaler/GeneticAlgorithms/MutationsNormalVariableProbabilityStart", "0.5");
    m_textCtrlGAsMutationsNormalVarProbStart->SetValue(GAsMutationsNormalVariableProbabilityStart);
    wxString GAsMutationsNormalVariableProbabilityEnd = pConfig->Read(
            "/Downscaler/GeneticAlgorithms/MutationsNormalVariableProbabilityEnd", "0.05");
    m_textCtrlGAsMutationsNormalVarProbEnd->SetValue(GAsMutationsNormalVariableProbabilityEnd);
    wxString GAsMutationsNormalVariableStdDevStart = pConfig->Read(
            "/Downscaler/GeneticAlgorithms/MutationsNormalVariableStdDevStart", "0.5");
    m_textCtrlGAsMutationsNormalVarStdDevStart->SetValue(GAsMutationsNormalVariableStdDevStart);
    wxString GAsMutationsNormalVariableStdDevEnd = pConfig->Read(
            "/Downscaler/GeneticAlgorithms/MutationsNormalVariableStdDevEnd", "0.01");
    m_textCtrlGAsMutationsNormalVarStdDevEnd->SetValue(GAsMutationsNormalVariableStdDevEnd);
    wxString GAsMutationsNonUniformProb = pConfig->Read("/Downscaler/GeneticAlgorithms/MutationsNonUniformProbability",
                                                        "0.2");
    m_textCtrlGAsMutationsNonUniformProb->SetValue(GAsMutationsNonUniformProb);
    wxString GAsMutationsNonUniformMaxGensNbVar = pConfig->Read(
            "/Downscaler/GeneticAlgorithms/MutationsNonUniformMaxGensNbVar", "50");
    m_textCtrlGAsMutationsNonUniformGensNb->SetValue(GAsMutationsNonUniformMaxGensNbVar);
    wxString GAsMutationsNonUniformMinRate = pConfig->Read("/Downscaler/GeneticAlgorithms/MutationsNonUniformMinRate",
                                                           "0.20");
    m_textCtrlGAsMutationsNonUniformMinRate->SetValue(GAsMutationsNonUniformMinRate);
    wxString GAsMutationsMultiScaleProb = pConfig->Read("/Downscaler/GeneticAlgorithms/MutationsMultiScaleProbability",
                                                        "0.20");
    m_textCtrlGAsMutationsMultiScaleProb->SetValue(GAsMutationsMultiScaleProb);
}

void asFrameDownscaler::OnSaveDefault(wxCommandEvent &event)
{
    SaveOptions();
}

void asFrameDownscaler::SaveOptions() const
{
    // General stuff
    wxConfigBase *pConfig = wxFileConfig::Get();
    long methodSelection = (long) m_choiceMethod->GetSelection();
    pConfig->Write("/Downscaler/MethodSelection", methodSelection);
    wxString parametersFilePath = m_filePickerParameters->GetPath();
    pConfig->Write("/Downscaler/ParametersFilePath", parametersFilePath);
    wxString predictandDBFilePath = m_filePickerPredictand->GetPath();
    pConfig->Write("/Paths/PredictandDBFilePath", predictandDBFilePath);
    wxString predictorDir = m_dirPickerPredictor->GetPath();
    pConfig->Write("/Paths/PredictorDir", predictorDir);
    wxString downscalerResultsDir = m_dirPickerDownscalingResults->GetPath();
    pConfig->Write("/Paths/DownscalerResultsDir", downscalerResultsDir);
    bool parallelEvaluations = m_checkBoxParallelEvaluations->GetValue();
    pConfig->Write("/Downscaler/ParallelEvaluations", parallelEvaluations);

    // Classic+ downscaling
    wxString classicPlusResizingIterations = m_textCtrlClassicPlusResizingIterations->GetValue();
    pConfig->Write("/Downscaler/ClassicPlus/ResizingIterations", classicPlusResizingIterations);
    wxString classicPlusStepsLatPertinenceMap = m_textCtrlClassicPlusStepsLatPertinenceMap->GetValue();
    pConfig->Write("/Downscaler/ClassicPlus/StepsLatPertinenceMap", classicPlusStepsLatPertinenceMap);
    wxString classicPlusStepsLonPertinenceMap = m_textCtrlClassicPlusStepsLonPertinenceMap->GetValue();
    pConfig->Write("/Downscaler/ClassicPlus/StepsLonPertinenceMap", classicPlusStepsLonPertinenceMap);
    bool proceedSequentially = m_checkBoxProceedSequentially->GetValue();
    pConfig->Write("/Downscaler/ClassicPlus/ProceedSequentially", proceedSequentially);

    // Variables exploration
    wxString varExploStep = m_textCtrlVarExploStepToExplore->GetValue();
    pConfig->Write("/Downscaler/VariablesExplo/Step", varExploStep);

    // Monte Carlo
    wxString monteCarloRandomNb = m_textCtrlMonteCarloRandomNb->GetValue();
    pConfig->Write("/Downscaler/MonteCarlo/RandomNb", monteCarloRandomNb);

    // Genetic algorithms
    long naturalSelectionOperator = m_choiceGAsNaturalSelectionOperator->GetSelection();
    pConfig->Write("/Downscaler/GeneticAlgorithms/NaturalSelectionOperator", naturalSelectionOperator);
    long couplesSelectionOperator = m_choiceGAsCouplesSelectionOperator->GetSelection();
    pConfig->Write("/Downscaler/GeneticAlgorithms/CouplesSelectionOperator", couplesSelectionOperator);
    long crossoverOperator = m_choiceGAsCrossoverOperator->GetSelection();
    pConfig->Write("/Downscaler/GeneticAlgorithms/CrossoverOperator", crossoverOperator);
    long mutationOperator = m_choiceGAsMutationOperator->GetSelection();
    pConfig->Write("/Downscaler/GeneticAlgorithms/MutationOperator", mutationOperator);
    wxString GAsRunNumbers = m_textCtrlGAsRunNumbers->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/NbRuns", GAsRunNumbers);
    wxString GAsPopulationSize = m_textCtrlGAsPopulationSize->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/PopulationSize", GAsPopulationSize);
    wxString GAsConvergenceStepsNb = m_textCtrlGAsConvergenceNb->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/ConvergenceStepsNb", GAsConvergenceStepsNb);
    wxString GAsRatioIntermediateGeneration = m_textCtrlGAsRatioIntermGen->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/RatioIntermediateGeneration", GAsRatioIntermediateGeneration);
    bool GAsAllowElitismForTheBest = m_checkBoxGAsAllowElitism->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/AllowElitismForTheBest", GAsAllowElitismForTheBest);
    wxString GAsNaturalSelectionTournamentProbability = m_textCtrlGAsNaturalSlctTournamentProb->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/NaturalSelectionTournamentProbability",
                   GAsNaturalSelectionTournamentProbability);
    wxString GAsCouplesSelectionTournamentNb = m_textCtrlGAsCouplesSlctTournamentNb->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/CouplesSelectionTournamentNb", GAsCouplesSelectionTournamentNb);
    wxString GAsCrossoverMultiplePointsNb = m_textCtrlGAsCrossoverMultipleNbPts->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/CrossoverMultiplePointsNb", GAsCrossoverMultiplePointsNb);
    wxString GAsCrossoverBlendingPointsNb = m_textCtrlGAsCrossoverBlendingNbPts->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/CrossoverBlendingPointsNb", GAsCrossoverBlendingPointsNb);
    bool GAsCrossoverBlendingShareBeta = m_checkBoxGAsCrossoverBlendingShareBeta->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/CrossoverBlendingShareBeta", GAsCrossoverBlendingShareBeta);
    wxString GAsCrossoverLinearPointsNb = m_textCtrlGAsCrossoverLinearNbPts->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/CrossoverLinearPointsNb", GAsCrossoverLinearPointsNb);
    wxString GAsCrossoverHeuristicPointsNb = m_textCtrlGAsCrossoverHeuristicNbPts->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/CrossoverHeuristicPointsNb", GAsCrossoverHeuristicPointsNb);
    bool GAsCrossoverHeuristicShareBeta = m_checkBoxGAsCrossoverHeuristicShareBeta->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/CrossoverHeuristicShareBeta", GAsCrossoverHeuristicShareBeta);
    wxString GAsCrossoverBinaryLikePointsNb = m_textCtrlGAsCrossoverBinLikeNbPts->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/CrossoverBinaryLikePointsNb", GAsCrossoverBinaryLikePointsNb);
    bool GAsCrossoverBinaryLikeShareBeta = m_checkBoxGAsCrossoverBinLikeShareBeta->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/CrossoverBinaryLikeShareBeta", GAsCrossoverBinaryLikeShareBeta);
    wxString GAsMutationsUniformConstantProbability = m_textCtrlGAsMutationsUniformCstProb->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/MutationsUniformConstantProbability",
                   GAsMutationsUniformConstantProbability);
    wxString GAsMutationsNormalConstantProbability = m_textCtrlGAsMutationsNormalCstProb->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/MutationsNormalConstantProbability",
                   GAsMutationsNormalConstantProbability);
    wxString GAsMutationsNormalConstantStdDevRatioRange = m_textCtrlGAsMutationsNormalCstStdDev->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/MutationsNormalConstantStdDevRatioRange",
                   GAsMutationsNormalConstantStdDevRatioRange);
    wxString GAsMutationsUniformVariableMaxGensNbVar = m_textCtrlGAsMutationsUniformVarMaxGensNb->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/MutationsUniformVariableMaxGensNbVar",
                   GAsMutationsUniformVariableMaxGensNbVar);
    wxString GAsMutationsUniformVariableProbabilityStart = m_textCtrlGAsMutationsUniformVarProbStart->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/MutationsUniformVariableProbabilityStart",
                   GAsMutationsUniformVariableProbabilityStart);
    wxString GAsMutationsUniformVariableProbabilityEnd = m_textCtrlGAsMutationsUniformVarProbEnd->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/MutationsUniformVariableProbabilityEnd",
                   GAsMutationsUniformVariableProbabilityEnd);
    wxString GAsMutationsNormalVariableMaxGensNbVarProb = m_textCtrlGAsMutationsNormalVarMaxGensNbProb->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/MutationsNormalVariableMaxGensNbVarProb",
                   GAsMutationsNormalVariableMaxGensNbVarProb);
    wxString GAsMutationsNormalVariableMaxGensNbVarStdDev = m_textCtrlGAsMutationsNormalVarMaxGensNbStdDev->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/MutationsNormalVariableMaxGensNbVarStdDev",
                   GAsMutationsNormalVariableMaxGensNbVarStdDev);
    wxString GAsMutationsNormalVariableProbabilityStart = m_textCtrlGAsMutationsNormalVarProbStart->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/MutationsNormalVariableProbabilityStart",
                   GAsMutationsNormalVariableProbabilityStart);
    wxString GAsMutationsNormalVariableProbabilityEnd = m_textCtrlGAsMutationsNormalVarProbEnd->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/MutationsNormalVariableProbabilityEnd",
                   GAsMutationsNormalVariableProbabilityEnd);
    wxString GAsMutationsNormalVariableStdDevStart = m_textCtrlGAsMutationsNormalVarStdDevStart->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/MutationsNormalVariableStdDevStart",
                   GAsMutationsNormalVariableStdDevStart);
    wxString GAsMutationsNormalVariableStdDevEnd = m_textCtrlGAsMutationsNormalVarStdDevEnd->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/MutationsNormalVariableStdDevEnd",
                   GAsMutationsNormalVariableStdDevEnd);
    wxString GAsMutationsNonUniformProb = m_textCtrlGAsMutationsNonUniformProb->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/MutationsNonUniformProbability", GAsMutationsNonUniformProb);
    wxString GAsMutationsNonUniformMaxGensNbVar = m_textCtrlGAsMutationsNonUniformGensNb->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/MutationsNonUniformMaxGensNbVar", GAsMutationsNonUniformMaxGensNbVar);
    wxString GAsMutationsNonUniformMinRate = m_textCtrlGAsMutationsNonUniformMinRate->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/MutationsNonUniformMinRate", GAsMutationsNonUniformMinRate);
    wxString GAsMutationsMultiScaleProb = m_textCtrlGAsMutationsMultiScaleProb->GetValue();
    pConfig->Write("/Downscaler/GeneticAlgorithms/MutationsMultiScaleProbability", GAsMutationsMultiScaleProb);

    pConfig->Flush();
}

/*
void asFrameDownscaler::OnIdle( wxCommandEvent& event )
{
    wxString state = asGetState();
    m_staticTextState->SetLabel(state);
}
*/
void asFrameDownscaler::Launch(wxCommandEvent &event)
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
                m_methodDownscaler = new asMethodDownscalerSingle();
                break;
            }
            case 1: // Classic
            {
                wxLogVerbose(_("Proceeding to classic downscaling."));
                m_methodDownscaler = new asMethodDownscalerClassic();
                break;
            }
            case 2: // Classic+
            {
                wxLogVerbose(_("Proceeding to classic+ downscaling."));
                m_methodDownscaler = new asMethodDownscalerClassic();
                break;
            }
            case 3: // Variables exploration with classic+
            {
                wxLogVerbose(_("Proceeding to variables exploration."));
                m_methodDownscaler = new asMethodDownscalerClassicVarExplo();
                break;
            }
            case 4: // Random sets
            {
                m_methodDownscaler = new asMethodDownscalerRandomSet();
                break;
            }
            case 5: // Genetic algorithms
            {
                m_methodDownscaler = new asMethodDownscalerGeneticAlgorithms();
                break;
            }
            case 6: // Scores evaluation
            {
                wxLogVerbose(_("Proceeding to all scores evaluation."));
                m_methodDownscaler = new asMethodDownscalerEvaluateAllScores();
                break;
            }
            case 7: // Only predictand values
            {
                wxLogVerbose(_("Proceeding to predictand values saving."));
                m_methodDownscaler = new asMethodDownscalerSingleOnlyValues();
                break;
            }
            default:
                wxLogError(_("Chosen method not defined yet."));
        }

        if (m_methodDownscaler) {
            m_methodDownscaler->SetParamsFilePath(m_filePickerParameters->GetPath());
            m_methodDownscaler->SetPredictandDBFilePath(m_filePickerPredictand->GetPath());
            m_methodDownscaler->SetPredictorDataDir(m_dirPickerPredictor->GetPath());
            m_methodDownscaler->Manager();
        }
    } catch (std::bad_alloc &ba) {
        wxString msg(ba.what(), wxConvUTF8);
        wxLogError(_("Bad allocation caught: %s"), msg);
        wxLogError(_("Failed to process the downscaling."));
    } catch (asException &e) {
        wxString fullMessage = e.GetFullMessage();
        if (!fullMessage.IsEmpty()) {
            wxLogError(fullMessage);
        }
        wxLogError(_("Failed to process the downscaling."));
    }

    wxDELETE(m_methodDownscaler);

    wxMessageBox(_("Downscaler over."));
}
