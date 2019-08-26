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
#include "asMethodCalibratorSingleOnlyDates.h"
#include "images.h"
#include "asFramePreferencesOptimizer.h"
#include "asFrameAbout.h"
#include "asFramePredictandDB.h"


asFrameOptimizer::asFrameOptimizer(wxWindow *parent)
        : asFrameOptimizerVirtual(parent),
          m_logWindow(nullptr),
          m_methodCalibrator(nullptr)
{
    // Toolbar
    m_toolBar->AddTool(asID_RUN, wxT("Run"), *_img_run, *_img_run, wxITEM_NORMAL, _("Run optimizer"),
                       _("Run optimizer now"), nullptr);
    m_toolBar->AddTool(asID_CANCEL, wxT("Cancel"), *_img_stop, *_img_stop, wxITEM_NORMAL, _("Cancel optimization"),
                       _("Cancel current optimization"), nullptr);
    m_toolBar->AddTool(asID_PREFERENCES, wxT("Preferences"), *_img_preferences, *_img_preferences, wxITEM_NORMAL,
                       _("Preferences"), _("Preferences"), nullptr);
    m_toolBar->Realize();

    // Connect events
    this->Connect(asID_RUN, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler(asFrameOptimizer::Launch));
    this->Connect(asID_CANCEL, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler(asFrameOptimizer::Cancel));
    this->Connect(asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED,
                  wxCommandEventHandler(asFrameOptimizer::OpenFramePreferences));
    this->Connect(asID_DB_CREATE, wxEVT_COMMAND_TOOL_CLICKED,
                  wxCommandEventHandler(asFrameOptimizer::OpenFramePredictandDB));

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
    this->Disconnect(asID_DB_CREATE, wxEVT_COMMAND_TOOL_CLICKED,
                     wxCommandEventHandler(asFrameOptimizer::OpenFramePredictandDB));
}

void asFrameOptimizer::OnInit()
{
    wxBusyCursor wait;

    // Set the defaults
    LoadOptions();
    DisplayLogLevelMenu();
}

void asFrameOptimizer::Update()
{
    DisplayLogLevelMenu();
}

void asFrameOptimizer::OpenFramePredictandDB(wxCommandEvent &event)
{
    wxBusyCursor wait;

    auto *frame = new asFramePredictandDB(this);
    frame->Fit();
    frame->Show();
}

void asFrameOptimizer::OpenFramePreferences(wxCommandEvent &event)
{
    wxBusyCursor wait;

    auto *frame = new asFramePreferencesOptimizer(this);
    frame->Fit();
    frame->Show();
}

void asFrameOptimizer::OpenFrameAbout(wxCommandEvent &event)
{
    wxBusyCursor wait;

    auto *frame = new asFrameAbout(this);
    frame->Fit();
    frame->Show();
}

void asFrameOptimizer::OnShowLog(wxCommandEvent &event)
{
    wxBusyCursor wait;

    wxASSERT(m_logWindow);
    m_logWindow->DoShow(true);
}

void asFrameOptimizer::OnLogLevel1(wxCommandEvent &event)
{
    Log()->SetLevel(1);
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
    Log()->SetLevel(2);
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
    Log()->SetLevel(3);
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
    int logLevel = (int) wxFileConfig::Get()->ReadLong("/General/LogLevel", 2l);
    ThreadsManager().CritSectionConfig().Leave();
    m_menuLogLevel->FindItemByPosition(0)->Check(false);
    m_menuLogLevel->FindItemByPosition(1)->Check(false);
    m_menuLogLevel->FindItemByPosition(2)->Check(false);
    switch (logLevel) {
        case 1:
            m_menuLogLevel->FindItemByPosition(0)->Check(true);
            Log()->SetLevel(1);
            break;
        case 2:
            m_menuLogLevel->FindItemByPosition(1)->Check(true);
            Log()->SetLevel(2);
            break;
        case 3:
            m_menuLogLevel->FindItemByPosition(2)->Check(true);
            Log()->SetLevel(3);
            break;
        default:
            m_menuLogLevel->FindItemByPosition(1)->Check(true);
            Log()->SetLevel(2);
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
    wxBusyCursor wait;

    // General stuff
    wxConfigBase *pConfig = wxFileConfig::Get();
    m_choiceMethod->SetSelection(pConfig->ReadLong("/MethodSelection", 0l));
    m_filePickerParameters->SetPath(pConfig->Read("/ParametersFilePath", wxEmptyString));
    m_filePickerPredictand->SetPath(pConfig->Read("/Paths/PredictandDBFilePath", wxEmptyString));
    m_dirPickerPredictor->SetPath(pConfig->Read("/Paths/PredictorDir", wxEmptyString));
    m_dirPickerCalibrationResults->SetPath(pConfig->Read("/Paths/ResultsDir",
                                                         asConfig::GetDocumentsDir() + "AtmoSwing" + DS + "Optimizer"));

    // Classic+ calibration
    m_textCtrlClassicPlusResizingIterations->SetValue(pConfig->Read("/ClassicPlus/ResizingIterations", "1"));
    m_textCtrlClassicPlusStepsLatPertinenceMap->SetValue(pConfig->Read("/ClassicPlus/StepsLatPertinenceMap", "2"));
    m_textCtrlClassicPlusStepsLonPertinenceMap->SetValue(pConfig->Read("/ClassicPlus/StepsLonPertinenceMap", "2"));
    m_checkBoxProceedSequentially->SetValue(pConfig->ReadBool("/ClassicPlus/ProceedSequentially", true));

    // Variables exploration
    m_textCtrlVarExploStepToExplore->SetValue(pConfig->Read("/VariablesExplo/Step"));

    // Monte Carlo
    m_textCtrlMonteCarloRandomNb->SetValue(pConfig->Read("/MonteCarlo/RandomNb", "1000"));

    // Genetic algorithms
    m_choiceGAsNaturalSelectionOperator->SetSelection(pConfig->ReadLong("/GAs/NaturalSelectionOperator", 1l));
    m_choiceGAsCouplesSelectionOperator->SetSelection(pConfig->ReadLong("/GAs/CouplesSelectionOperator", 3l));
    m_choiceGAsCrossoverOperator->SetSelection(pConfig->ReadLong("/GAs/CrossoverOperator", 1l));
    m_choiceGAsMutationOperator->SetSelection(pConfig->ReadLong("/GAs/MutationOperator", 0l));
    m_textCtrlGAsRunNumbers->SetValue(pConfig->Read("/GAs/NbRuns", "20"));
    m_textCtrlGAsPopulationSize->SetValue(pConfig->Read("/GAs/PopulationSize", "50"));
    m_textCtrlGAsConvergenceNb->SetValue(pConfig->Read("/GAs/ConvergenceStepsNb", "20"));
    m_textCtrlGAsRatioIntermGen->SetValue(pConfig->Read("/GAs/RatioIntermediateGeneration", "0.5"));
    m_checkBoxGAsAllowElitism->SetValue(pConfig->ReadBool("/GAs/AllowElitismForTheBest", true));
    m_textCtrlGAsNaturalSlctTournamentProb->SetValue(pConfig->Read("/GAs/NaturalSelectionTournamentProbability", "0.9"));
    m_textCtrlGAsCouplesSlctTournamentNb->SetValue(pConfig->Read("/GAs/CouplesSelectionTournamentNb", "3"));
    m_textCtrlGAsCrossoverMultipleNbPts->SetValue(pConfig->Read("/GAs/CrossoverMultiplePointsNb", "3"));
    m_textCtrlGAsCrossoverBlendingNbPts->SetValue(pConfig->Read("/GAs/CrossoverBlendingPointsNb", "2"));
    m_checkBoxGAsCrossoverBlendingShareBeta->SetValue(pConfig->ReadBool("/GAs/CrossoverBlendingShareBeta", true));
    m_textCtrlGAsCrossoverLinearNbPts->SetValue(pConfig->Read("/GAs/CrossoverLinearPointsNb", "2"));
    m_textCtrlGAsCrossoverHeuristicNbPts->SetValue(pConfig->Read("/GAs/CrossoverHeuristicPointsNb", "2"));
    m_checkBoxGAsCrossoverHeuristicShareBeta->SetValue(pConfig->ReadBool("/GAs/CrossoverHeuristicShareBeta", true));
    m_textCtrlGAsCrossoverBinLikeNbPts->SetValue(pConfig->Read("/GAs/CrossoverBinaryLikePointsNb", "2"));
    m_checkBoxGAsCrossoverBinLikeShareBeta->SetValue(pConfig->ReadBool("/GAs/CrossoverBinaryLikeShareBeta", true));
    m_textCtrlGAsMutationsUniformCstProb->SetValue(pConfig->Read("/GAs/MutationsUniformConstantProbability", "0.2"));
    m_textCtrlGAsMutationsNormalCstProb->SetValue(pConfig->Read("/GAs/MutationsNormalConstantProbability", "0.2"));
    m_textCtrlGAsMutationsNormalCstStdDev->SetValue(pConfig->Read("/GAs/MutationsNormalConstantStdDevRatioRange", "0.10"));
    m_textCtrlGAsMutationsUniformVarMaxGensNb->SetValue(pConfig->Read("/GAs/MutationsUniformVariableMaxGensNbVar", "50"));
    m_textCtrlGAsMutationsUniformVarProbStart->SetValue(pConfig->Read("/GAs/MutationsUniformVariableProbabilityStart", "0.5"));
    m_textCtrlGAsMutationsUniformVarProbEnd->SetValue(pConfig->Read("/GAs/MutationsUniformVariableProbabilityEnd", "0.01"));
    m_textCtrlGAsMutationsNormalVarMaxGensNbProb->SetValue(pConfig->Read("/GAs/MutationsNormalVariableMaxGensNbVarProb", "50"));
    m_textCtrlGAsMutationsNormalVarMaxGensNbStdDev->SetValue(pConfig->Read("/GAs/MutationsNormalVariableMaxGensNbVarStdDev", "50"));
    m_textCtrlGAsMutationsNormalVarProbStart->SetValue(pConfig->Read("/GAs/MutationsNormalVariableProbabilityStart", "0.5"));
    m_textCtrlGAsMutationsNormalVarProbEnd->SetValue(pConfig->Read("/GAs/MutationsNormalVariableProbabilityEnd", "0.05"));
    m_textCtrlGAsMutationsNormalVarStdDevStart->SetValue(pConfig->Read("/GAs/MutationsNormalVariableStdDevStart", "0.5"));
    m_textCtrlGAsMutationsNormalVarStdDevEnd->SetValue(pConfig->Read("/GAs/MutationsNormalVariableStdDevEnd", "0.01"));
    m_textCtrlGAsMutationsNonUniformProb->SetValue(pConfig->Read("/GAs/MutationsNonUniformProbability","0.2"));
    m_textCtrlGAsMutationsNonUniformGensNb->SetValue(pConfig->Read("/GAs/MutationsNonUniformMaxGensNbVar", "50"));
    m_textCtrlGAsMutationsNonUniformMinRate->SetValue(pConfig->Read("/GAs/MutationsNonUniformMinRate", "0.20"));
    m_textCtrlGAsMutationsMultiScaleProb->SetValue(pConfig->Read("/GAs/MutationsMultiScaleProbability", "0.20"));
}

void asFrameOptimizer::OnSaveDefault(wxCommandEvent &event)
{
    SaveOptions();
}

void asFrameOptimizer::SaveOptions() const
{
    wxBusyCursor wait;

    // General stuff
    wxConfigBase *pConfig = wxFileConfig::Get();
    auto methodSelection = (long) m_choiceMethod->GetSelection();
    pConfig->Write("/MethodSelection", methodSelection);
    wxString parametersFilePath = m_filePickerParameters->GetPath();
    pConfig->Write("/ParametersFilePath", parametersFilePath);
    wxString predictandDBFilePath = m_filePickerPredictand->GetPath();
    pConfig->Write("/Paths/PredictandDBFilePath", predictandDBFilePath);
    wxString predictorDir = m_dirPickerPredictor->GetPath();
    pConfig->Write("/Paths/PredictorDir", predictorDir);
    wxString optimizerResultsDir = m_dirPickerCalibrationResults->GetPath();
    pConfig->Write("/Paths/ResultsDir", optimizerResultsDir);

    // Classic+ calibration
    wxString classicPlusResizingIterations = m_textCtrlClassicPlusResizingIterations->GetValue();
    pConfig->Write("/ClassicPlus/ResizingIterations", classicPlusResizingIterations);
    wxString classicPlusStepsLatPertinenceMap = m_textCtrlClassicPlusStepsLatPertinenceMap->GetValue();
    pConfig->Write("/ClassicPlus/StepsLatPertinenceMap", classicPlusStepsLatPertinenceMap);
    wxString classicPlusStepsLonPertinenceMap = m_textCtrlClassicPlusStepsLonPertinenceMap->GetValue();
    pConfig->Write("/ClassicPlus/StepsLonPertinenceMap", classicPlusStepsLonPertinenceMap);
    bool proceedSequentially = m_checkBoxProceedSequentially->GetValue();
    pConfig->Write("/ClassicPlus/ProceedSequentially", proceedSequentially);

    // Variables exploration
    wxString varExploStep = m_textCtrlVarExploStepToExplore->GetValue();
    pConfig->Write("/VariablesExplo/Step", varExploStep);

    // Monte Carlo
    wxString monteCarloRandomNb = m_textCtrlMonteCarloRandomNb->GetValue();
    pConfig->Write("/MonteCarlo/RandomNb", monteCarloRandomNb);

    // Genetic algorithms
    long naturalSelectionOperator = m_choiceGAsNaturalSelectionOperator->GetSelection();
    pConfig->Write("/GAs/NaturalSelectionOperator", naturalSelectionOperator);
    long couplesSelectionOperator = m_choiceGAsCouplesSelectionOperator->GetSelection();
    pConfig->Write("/GAs/CouplesSelectionOperator", couplesSelectionOperator);
    long crossoverOperator = m_choiceGAsCrossoverOperator->GetSelection();
    pConfig->Write("/GAs/CrossoverOperator", crossoverOperator);
    long mutationOperator = m_choiceGAsMutationOperator->GetSelection();
    pConfig->Write("/GAs/MutationOperator", mutationOperator);
    wxString GAsRunNumbers = m_textCtrlGAsRunNumbers->GetValue();
    pConfig->Write("/GAs/NbRuns", GAsRunNumbers);
    wxString GAsPopulationSize = m_textCtrlGAsPopulationSize->GetValue();
    pConfig->Write("/GAs/PopulationSize", GAsPopulationSize);
    wxString GAsConvergenceStepsNb = m_textCtrlGAsConvergenceNb->GetValue();
    pConfig->Write("/GAs/ConvergenceStepsNb", GAsConvergenceStepsNb);
    wxString GAsRatioIntermediateGeneration = m_textCtrlGAsRatioIntermGen->GetValue();
    pConfig->Write("/GAs/RatioIntermediateGeneration", GAsRatioIntermediateGeneration);
    bool GAsAllowElitismForTheBest = m_checkBoxGAsAllowElitism->GetValue();
    pConfig->Write("/GAs/AllowElitismForTheBest", GAsAllowElitismForTheBest);
    wxString GAsNaturalSelectionTournamentProbability = m_textCtrlGAsNaturalSlctTournamentProb->GetValue();
    pConfig->Write("/GAs/NaturalSelectionTournamentProbability", GAsNaturalSelectionTournamentProbability);
    wxString GAsCouplesSelectionTournamentNb = m_textCtrlGAsCouplesSlctTournamentNb->GetValue();
    pConfig->Write("/GAs/CouplesSelectionTournamentNb", GAsCouplesSelectionTournamentNb);
    wxString GAsCrossoverMultiplePointsNb = m_textCtrlGAsCrossoverMultipleNbPts->GetValue();
    pConfig->Write("/GAs/CrossoverMultiplePointsNb", GAsCrossoverMultiplePointsNb);
    wxString GAsCrossoverBlendingPointsNb = m_textCtrlGAsCrossoverBlendingNbPts->GetValue();
    pConfig->Write("/GAs/CrossoverBlendingPointsNb", GAsCrossoverBlendingPointsNb);
    bool GAsCrossoverBlendingShareBeta = m_checkBoxGAsCrossoverBlendingShareBeta->GetValue();
    pConfig->Write("/GAs/CrossoverBlendingShareBeta", GAsCrossoverBlendingShareBeta);
    wxString GAsCrossoverLinearPointsNb = m_textCtrlGAsCrossoverLinearNbPts->GetValue();
    pConfig->Write("/GAs/CrossoverLinearPointsNb", GAsCrossoverLinearPointsNb);
    wxString GAsCrossoverHeuristicPointsNb = m_textCtrlGAsCrossoverHeuristicNbPts->GetValue();
    pConfig->Write("/GAs/CrossoverHeuristicPointsNb", GAsCrossoverHeuristicPointsNb);
    bool GAsCrossoverHeuristicShareBeta = m_checkBoxGAsCrossoverHeuristicShareBeta->GetValue();
    pConfig->Write("/GAs/CrossoverHeuristicShareBeta", GAsCrossoverHeuristicShareBeta);
    wxString GAsCrossoverBinaryLikePointsNb = m_textCtrlGAsCrossoverBinLikeNbPts->GetValue();
    pConfig->Write("/GAs/CrossoverBinaryLikePointsNb", GAsCrossoverBinaryLikePointsNb);
    bool GAsCrossoverBinaryLikeShareBeta = m_checkBoxGAsCrossoverBinLikeShareBeta->GetValue();
    pConfig->Write("/GAs/CrossoverBinaryLikeShareBeta", GAsCrossoverBinaryLikeShareBeta);
    wxString GAsMutationsUniformConstantProbability = m_textCtrlGAsMutationsUniformCstProb->GetValue();
    pConfig->Write("/GAs/MutationsUniformConstantProbability", GAsMutationsUniformConstantProbability);
    wxString GAsMutationsNormalConstantProbability = m_textCtrlGAsMutationsNormalCstProb->GetValue();
    pConfig->Write("/GAs/MutationsNormalConstantProbability", GAsMutationsNormalConstantProbability);
    wxString GAsMutationsNormalConstantStdDevRatioRange = m_textCtrlGAsMutationsNormalCstStdDev->GetValue();
    pConfig->Write("/GAs/MutationsNormalConstantStdDevRatioRange", GAsMutationsNormalConstantStdDevRatioRange);
    wxString GAsMutationsUniformVariableMaxGensNbVar = m_textCtrlGAsMutationsUniformVarMaxGensNb->GetValue();
    pConfig->Write("/GAs/MutationsUniformVariableMaxGensNbVar", GAsMutationsUniformVariableMaxGensNbVar);
    wxString GAsMutationsUniformVariableProbabilityStart = m_textCtrlGAsMutationsUniformVarProbStart->GetValue();
    pConfig->Write("/GAs/MutationsUniformVariableProbabilityStart", GAsMutationsUniformVariableProbabilityStart);
    wxString GAsMutationsUniformVariableProbabilityEnd = m_textCtrlGAsMutationsUniformVarProbEnd->GetValue();
    pConfig->Write("/GAs/MutationsUniformVariableProbabilityEnd", GAsMutationsUniformVariableProbabilityEnd);
    wxString GAsMutationsNormalVariableMaxGensNbVarProb = m_textCtrlGAsMutationsNormalVarMaxGensNbProb->GetValue();
    pConfig->Write("/GAs/MutationsNormalVariableMaxGensNbVarProb", GAsMutationsNormalVariableMaxGensNbVarProb);
    wxString GAsMutationsNormalVariableMaxGensNbVarStdDev = m_textCtrlGAsMutationsNormalVarMaxGensNbStdDev->GetValue();
    pConfig->Write("/GAs/MutationsNormalVariableMaxGensNbVarStdDev", GAsMutationsNormalVariableMaxGensNbVarStdDev);
    wxString GAsMutationsNormalVariableProbabilityStart = m_textCtrlGAsMutationsNormalVarProbStart->GetValue();
    pConfig->Write("/GAs/MutationsNormalVariableProbabilityStart", GAsMutationsNormalVariableProbabilityStart);
    wxString GAsMutationsNormalVariableProbabilityEnd = m_textCtrlGAsMutationsNormalVarProbEnd->GetValue();
    pConfig->Write("/GAs/MutationsNormalVariableProbabilityEnd", GAsMutationsNormalVariableProbabilityEnd);
    wxString GAsMutationsNormalVariableStdDevStart = m_textCtrlGAsMutationsNormalVarStdDevStart->GetValue();
    pConfig->Write("/GAs/MutationsNormalVariableStdDevStart", GAsMutationsNormalVariableStdDevStart);
    wxString GAsMutationsNormalVariableStdDevEnd = m_textCtrlGAsMutationsNormalVarStdDevEnd->GetValue();
    pConfig->Write("/GAs/MutationsNormalVariableStdDevEnd", GAsMutationsNormalVariableStdDevEnd);
    wxString GAsMutationsNonUniformProb = m_textCtrlGAsMutationsNonUniformProb->GetValue();
    pConfig->Write("/GAs/MutationsNonUniformProbability", GAsMutationsNonUniformProb);
    wxString GAsMutationsNonUniformMaxGensNbVar = m_textCtrlGAsMutationsNonUniformGensNb->GetValue();
    pConfig->Write("/GAs/MutationsNonUniformMaxGensNbVar", GAsMutationsNonUniformMaxGensNbVar);
    wxString GAsMutationsNonUniformMinRate = m_textCtrlGAsMutationsNonUniformMinRate->GetValue();
    pConfig->Write("/GAs/MutationsNonUniformMinRate", GAsMutationsNonUniformMinRate);
    wxString GAsMutationsMultiScaleProb = m_textCtrlGAsMutationsMultiScaleProb->GetValue();
    pConfig->Write("/GAs/MutationsMultiScaleProbability", GAsMutationsMultiScaleProb);

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
    wxBusyCursor wait;

    SaveOptions();

    try {
        switch (m_choiceMethod->GetSelection()) {
            case wxNOT_FOUND: {
                wxLogError(_("Wrong method selection."));
                break;
            }
            case 0: // Single
            {
                m_methodCalibrator = new asMethodCalibratorSingle();
                break;
            }
            case 1: // Classic
            {
                m_methodCalibrator = new asMethodCalibratorClassic();
                break;
            }
            case 2: // Classic+
            {
                m_methodCalibrator = new asMethodCalibratorClassic();
                break;
            }
            case 3: // Variables exploration with classic+
            {
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
                m_methodCalibrator = new asMethodCalibratorEvaluateAllScores();
                break;
            }
            case 7: // Only predictand values
            {
                m_methodCalibrator = new asMethodCalibratorSingleOnlyValues();
                break;
            }
            case 8: // Only analog dates
            {
                m_methodCalibrator = new asMethodCalibratorSingleOnlyDates();
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
    } catch (std::exception &e) {
        wxString msg(e.what(), wxConvUTF8);
        wxLogError(_("Exception caught: %s"), msg);
        wxLogError(_("Failed to process the optimization."));
    }

    wxDELETE(m_methodCalibrator);

    wxMessageBox(_("Optimizer over."));
}
