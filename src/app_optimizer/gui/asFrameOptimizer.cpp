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

#include "asBitmaps.h"
#include "asFrameAbout.h"
#include "asFramePredictandDB.h"
#include "asFramePreferencesOptimizer.h"
#include "asMethodCalibratorClassic.h"
#include "asMethodCalibratorClassicVarExplo.h"
#include "asMethodCalibratorEvaluateAllScores.h"
#include "asMethodCalibratorSingle.h"
#include "asMethodCalibratorSingleOnlyDates.h"
#include "asMethodCalibratorSingleOnlyValues.h"
#include "asMethodOptimizerGAs.h"
#include "asMethodOptimizerMC.h"
#include "wx/fileconf.h"

asFrameOptimizer::asFrameOptimizer(wxWindow* parent)
    : asFrameOptimizerVirtual(parent),
      _logWindow(nullptr),
      _methodCalibrator(nullptr) {
    // Toolbar
    _toolBar->AddTool(asID_RUN, wxT("Run"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::RUN), wxNullBitmap, wxITEM_NORMAL,
                      _("Run optimizer"), _("Run optimizer now"), nullptr);
    _toolBar->AddTool(asID_CANCEL, wxT("Cancel"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::STOP), wxNullBitmap,
                      wxITEM_NORMAL, _("Cancel optimization"), _("Cancel current optimization"), nullptr);
    _toolBar->AddTool(asID_PREFERENCES, wxT("Preferences"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::PREFERENCES),
                      wxNullBitmap, wxITEM_NORMAL, _("Preferences"), _("Preferences"), nullptr);
    _toolBar->Realize();

    // Connect events
    Bind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameOptimizer::Launch, this, asID_RUN);
    Bind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameOptimizer::Cancel, this, asID_CANCEL);
    Bind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameOptimizer::OpenFramePreferences, this, asID_PREFERENCES);
    Bind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameOptimizer::OpenFramePredictandDB, this, asID_DB_CREATE);

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

asFrameOptimizer::~asFrameOptimizer() {
    // Disconnect events
    Unbind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameOptimizer::Launch, this, asID_RUN);
    Unbind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameOptimizer::Cancel, this, asID_CANCEL);
    Unbind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameOptimizer::OpenFramePreferences, this, asID_PREFERENCES);
    Unbind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameOptimizer::OpenFramePredictandDB, this, asID_DB_CREATE);
}

void asFrameOptimizer::OnInit() {
    wxBusyCursor wait;

    // Set the defaults
    LoadOptions();
    DisplayLogLevelMenu();
}

void asFrameOptimizer::Update() {
    DisplayLogLevelMenu();
}

void asFrameOptimizer::OpenFramePredictandDB(wxCommandEvent& event) {
    wxBusyCursor wait;

    auto frame = new asFramePredictandDB(this);
    frame->Fit();
    frame->Show();
}

void asFrameOptimizer::OpenFramePreferences(wxCommandEvent& event) {
    wxBusyCursor wait;

    auto frame = new asFramePreferencesOptimizer(this);
    frame->Fit();
    frame->Show();
}

void asFrameOptimizer::OpenFrameAbout(wxCommandEvent& event) {
    wxBusyCursor wait;

    auto frame = new asFrameAbout(this);
    frame->Fit();
    frame->Show();
}

void asFrameOptimizer::OnShowLog(wxCommandEvent& event) {
    wxBusyCursor wait;

    wxASSERT(_logWindow);
    _logWindow->DoShow(true);
}

void asFrameOptimizer::OnLogLevel1(wxCommandEvent& event) {
    Log()->SetLevel(1);
    _menuLogLevel->FindItemByPosition(0)->Check(true);
    _menuLogLevel->FindItemByPosition(1)->Check(false);
    _menuLogLevel->FindItemByPosition(2)->Check(false);
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Write("/General/LogLevel", 1l);
    ThreadsManager().CritSectionConfig().Leave();
    wxWindow* prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameOptimizer::OnLogLevel2(wxCommandEvent& event) {
    Log()->SetLevel(2);
    _menuLogLevel->FindItemByPosition(0)->Check(false);
    _menuLogLevel->FindItemByPosition(1)->Check(true);
    _menuLogLevel->FindItemByPosition(2)->Check(false);
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Write("/General/LogLevel", 2l);
    ThreadsManager().CritSectionConfig().Leave();
    wxWindow* prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameOptimizer::OnLogLevel3(wxCommandEvent& event) {
    Log()->SetLevel(3);
    _menuLogLevel->FindItemByPosition(0)->Check(false);
    _menuLogLevel->FindItemByPosition(1)->Check(false);
    _menuLogLevel->FindItemByPosition(2)->Check(true);
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Write("/General/LogLevel", 3l);
    ThreadsManager().CritSectionConfig().Leave();
    wxWindow* prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameOptimizer::DisplayLogLevelMenu() {
    // Set log level in the menu
    ThreadsManager().CritSectionConfig().Enter();
    int logLevel = (int)wxFileConfig::Get()->ReadLong("/General/LogLevel", 2l);
    ThreadsManager().CritSectionConfig().Leave();
    _menuLogLevel->FindItemByPosition(0)->Check(false);
    _menuLogLevel->FindItemByPosition(1)->Check(false);
    _menuLogLevel->FindItemByPosition(2)->Check(false);
    switch (logLevel) {
        case 1:
            _menuLogLevel->FindItemByPosition(0)->Check(true);
            Log()->SetLevel(1);
            break;
        case 2:
            _menuLogLevel->FindItemByPosition(1)->Check(true);
            Log()->SetLevel(2);
            break;
        case 3:
            _menuLogLevel->FindItemByPosition(2)->Check(true);
            Log()->SetLevel(3);
            break;
        default:
            _menuLogLevel->FindItemByPosition(1)->Check(true);
            Log()->SetLevel(2);
    }
}

void asFrameOptimizer::Cancel(wxCommandEvent& event) {
    if (_methodCalibrator) {
        _methodCalibrator->Cancel();
    }
}

void asFrameOptimizer::LoadOptions() {
    wxBusyCursor wait;

    // General stuff
    wxConfigBase* pConfig = wxFileConfig::Get();
    _choiceMethod->SetSelection(pConfig->ReadLong("/MethodSelection", 0l));
    _filePickerParameters->SetPath(pConfig->Read("/ParametersFilePath", wxEmptyString));
    _filePickerPredictand->SetPath(pConfig->Read("/Paths/PredictandDBFilePath", wxEmptyString));
    _dirPickerPredictor->SetPath(pConfig->Read("/Paths/PredictorDir", wxEmptyString));
    _dirPickerCalibrationResults->SetPath(
        pConfig->Read("/Paths/ResultsDir", asConfig::GetDocumentsDir() + "AtmoSwing" + DS + "Optimizer"));

    // Classic+ calibration
    _textCtrlClassicPlusResizingIterations->SetValue(pConfig->Read("/ClassicPlus/ResizingIterations", "1"));
    _textCtrlClassicPlusStepsLatPertinenceMap->SetValue(pConfig->Read("/ClassicPlus/StepsLatPertinenceMap", "2"));
    _textCtrlClassicPlusStepsLonPertinenceMap->SetValue(pConfig->Read("/ClassicPlus/StepsLonPertinenceMap", "2"));
    _checkBoxProceedSequentially->SetValue(pConfig->ReadBool("/ClassicPlus/ProceedSequentially", true));

    // Variables exploration
    _textCtrlVarExploStepToExplore->SetValue(pConfig->Read("/VariablesExplo/Step"));

    // Monte Carlo
    _textCtrlMonteCarloRandomNb->SetValue(pConfig->Read("/MonteCarlo/RandomNb", "1000"));

    // Genetic algorithms
    _choiceGAsNaturalSelectionOperator->SetSelection(pConfig->ReadLong("/GAs/NaturalSelectionOperator", 1l));
    _choiceGAsCouplesSelectionOperator->SetSelection(pConfig->ReadLong("/GAs/CouplesSelectionOperator", 3l));
    _choiceGAsCrossoverOperator->SetSelection(pConfig->ReadLong("/GAs/CrossoverOperator", 1l));
    _choiceGAsMutationOperator->SetSelection(pConfig->ReadLong("/GAs/MutationOperator", 0l));
    _textCtrlGAsRunNumbers->SetValue(pConfig->Read("/GAs/NbRuns", "20"));
    _textCtrlGAsPopulationSize->SetValue(pConfig->Read("/GAs/PopulationSize", "500"));
    _textCtrlGAsConvergenceNb->SetValue(pConfig->Read("/GAs/ConvergenceStepsNb", "30"));
    _textCtrlGAsRatioIntermGen->SetValue(pConfig->Read("/GAs/RatioIntermediateGeneration", "0.5"));
    _checkBoxGAsAllowElitism->SetValue(pConfig->ReadBool("/GAs/AllowElitismForTheBest", true));
    _textCtrlGAsNaturalSlctTournamentProb->SetValue(pConfig->Read("/GAs/NaturalSelectionTournamentProbability", "0.9"));
    _textCtrlGAsCouplesSlctTournamentNb->SetValue(pConfig->Read("/GAs/CouplesSelectionTournamentNb", "3"));
    _textCtrlGAsCrossoverMultipleNbPts->SetValue(pConfig->Read("/GAs/CrossoverMultiplePointsNb", "3"));
    _textCtrlGAsCrossoverBlendingNbPts->SetValue(pConfig->Read("/GAs/CrossoverBlendingPointsNb", "2"));
    _checkBoxGAsCrossoverBlendingShareBeta->SetValue(pConfig->ReadBool("/GAs/CrossoverBlendingShareBeta", true));
    _textCtrlGAsCrossoverLinearNbPts->SetValue(pConfig->Read("/GAs/CrossoverLinearPointsNb", "2"));
    _textCtrlGAsCrossoverHeuristicNbPts->SetValue(pConfig->Read("/GAs/CrossoverHeuristicPointsNb", "2"));
    _checkBoxGAsCrossoverHeuristicShareBeta->SetValue(pConfig->ReadBool("/GAs/CrossoverHeuristicShareBeta", true));
    _textCtrlGAsCrossoverBinLikeNbPts->SetValue(pConfig->Read("/GAs/CrossoverBinaryLikePointsNb", "2"));
    _checkBoxGAsCrossoverBinLikeShareBeta->SetValue(pConfig->ReadBool("/GAs/CrossoverBinaryLikeShareBeta", true));
    _textCtrlGAsMutationsUniformCstProb->SetValue(pConfig->Read("/GAs/MutationsUniformConstantProbability", "0.2"));
    _textCtrlGAsMutationsNormalCstProb->SetValue(pConfig->Read("/GAs/MutationsNormalConstantProbability", "0.2"));
    _textCtrlGAsMutationsNormalCstStdDev->SetValue(
        pConfig->Read("/GAs/MutationsNormalConstantStdDevRatioRange", "0.10"));
    _textCtrlGAsMutationsUniformVarMaxGensNb->SetValue(
        pConfig->Read("/GAs/MutationsUniformVariableMaxGensNbVar", "50"));
    _textCtrlGAsMutationsUniformVarProbStart->SetValue(
        pConfig->Read("/GAs/MutationsUniformVariableProbabilityStart", "0.5"));
    _textCtrlGAsMutationsUniformVarProbEnd->SetValue(
        pConfig->Read("/GAs/MutationsUniformVariableProbabilityEnd", "0.01"));
    _textCtrlGAsMutationsNormalVarMaxGensNbProb->SetValue(
        pConfig->Read("/GAs/MutationsNormalVariableMaxGensNbVarProb", "50"));
    _textCtrlGAsMutationsNormalVarMaxGensNbStdDev->SetValue(
        pConfig->Read("/GAs/MutationsNormalVariableMaxGensNbVarStdDev", "50"));
    _textCtrlGAsMutationsNormalVarProbStart->SetValue(
        pConfig->Read("/GAs/MutationsNormalVariableProbabilityStart", "0.5"));
    _textCtrlGAsMutationsNormalVarProbEnd->SetValue(
        pConfig->Read("/GAs/MutationsNormalVariableProbabilityEnd", "0.05"));
    _textCtrlGAsMutationsNormalVarStdDevStart->SetValue(
        pConfig->Read("/GAs/MutationsNormalVariableStdDevStart", "0.5"));
    _textCtrlGAsMutationsNormalVarStdDevEnd->SetValue(pConfig->Read("/GAs/MutationsNormalVariableStdDevEnd", "0.01"));
    _textCtrlGAsMutationsNonUniformProb->SetValue(pConfig->Read("/GAs/MutationsNonUniformProbability", "0.2"));
    _textCtrlGAsMutationsNonUniformGensNb->SetValue(pConfig->Read("/GAs/MutationsNonUniformMaxGensNbVar", "50"));
    _textCtrlGAsMutationsNonUniformMinRate->SetValue(pConfig->Read("/GAs/MutationsNonUniformMinRate", "0.10"));
    _textCtrlGAsMutationsMultiScaleProb->SetValue(pConfig->Read("/GAs/MutationsMultiScaleProbability", "0.10"));
}

void asFrameOptimizer::OnSaveDefault(wxCommandEvent& event) {
    SaveOptions();
}

void asFrameOptimizer::SaveOptions() const {
    wxBusyCursor wait;

    // General stuff
    wxConfigBase* pConfig = wxFileConfig::Get();
    auto methodSelection = (long)_choiceMethod->GetSelection();
    pConfig->Write("/MethodSelection", methodSelection);
    wxString parametersFilePath = _filePickerParameters->GetPath();
    pConfig->Write("/ParametersFilePath", parametersFilePath);
    wxString predictandDBFilePath = _filePickerPredictand->GetPath();
    pConfig->Write("/Paths/PredictandDBFilePath", predictandDBFilePath);
    wxString predictorDir = _dirPickerPredictor->GetPath();
    pConfig->Write("/Paths/PredictorDir", predictorDir);
    wxString optimizerResultsDir = _dirPickerCalibrationResults->GetPath();
    pConfig->Write("/Paths/ResultsDir", optimizerResultsDir);

    // Classic+ calibration
    wxString classicPlusResizingIterations = _textCtrlClassicPlusResizingIterations->GetValue();
    pConfig->Write("/ClassicPlus/ResizingIterations", classicPlusResizingIterations);
    wxString classicPlusStepsLatPertinenceMap = _textCtrlClassicPlusStepsLatPertinenceMap->GetValue();
    pConfig->Write("/ClassicPlus/StepsLatPertinenceMap", classicPlusStepsLatPertinenceMap);
    wxString classicPlusStepsLonPertinenceMap = _textCtrlClassicPlusStepsLonPertinenceMap->GetValue();
    pConfig->Write("/ClassicPlus/StepsLonPertinenceMap", classicPlusStepsLonPertinenceMap);
    bool proceedSequentially = _checkBoxProceedSequentially->GetValue();
    pConfig->Write("/ClassicPlus/ProceedSequentially", proceedSequentially);

    // Variables exploration
    wxString varExploStep = _textCtrlVarExploStepToExplore->GetValue();
    pConfig->Write("/VariablesExplo/Step", varExploStep);

    // Monte Carlo
    wxString monteCarloRandomNb = _textCtrlMonteCarloRandomNb->GetValue();
    pConfig->Write("/MonteCarlo/RandomNb", monteCarloRandomNb);

    // Genetic algorithms
    long naturalSelectionOperator = _choiceGAsNaturalSelectionOperator->GetSelection();
    pConfig->Write("/GAs/NaturalSelectionOperator", naturalSelectionOperator);
    long couplesSelectionOperator = _choiceGAsCouplesSelectionOperator->GetSelection();
    pConfig->Write("/GAs/CouplesSelectionOperator", couplesSelectionOperator);
    long crossoverOperator = _choiceGAsCrossoverOperator->GetSelection();
    pConfig->Write("/GAs/CrossoverOperator", crossoverOperator);
    long mutationOperator = _choiceGAsMutationOperator->GetSelection();
    pConfig->Write("/GAs/MutationOperator", mutationOperator);
    wxString GAsRunNumbers = _textCtrlGAsRunNumbers->GetValue();
    pConfig->Write("/GAs/NbRuns", GAsRunNumbers);
    wxString GAsPopulationSize = _textCtrlGAsPopulationSize->GetValue();
    pConfig->Write("/GAs/PopulationSize", GAsPopulationSize);
    wxString GAsConvergenceStepsNb = _textCtrlGAsConvergenceNb->GetValue();
    pConfig->Write("/GAs/ConvergenceStepsNb", GAsConvergenceStepsNb);
    wxString GAsRatioIntermediateGeneration = _textCtrlGAsRatioIntermGen->GetValue();
    pConfig->Write("/GAs/RatioIntermediateGeneration", GAsRatioIntermediateGeneration);
    bool GAsAllowElitismForTheBest = _checkBoxGAsAllowElitism->GetValue();
    pConfig->Write("/GAs/AllowElitismForTheBest", GAsAllowElitismForTheBest);
    wxString GAsNaturalSelectionTournamentProbability = _textCtrlGAsNaturalSlctTournamentProb->GetValue();
    pConfig->Write("/GAs/NaturalSelectionTournamentProbability", GAsNaturalSelectionTournamentProbability);
    wxString GAsCouplesSelectionTournamentNb = _textCtrlGAsCouplesSlctTournamentNb->GetValue();
    pConfig->Write("/GAs/CouplesSelectionTournamentNb", GAsCouplesSelectionTournamentNb);
    wxString GAsCrossoverMultiplePointsNb = _textCtrlGAsCrossoverMultipleNbPts->GetValue();
    pConfig->Write("/GAs/CrossoverMultiplePointsNb", GAsCrossoverMultiplePointsNb);
    wxString GAsCrossoverBlendingPointsNb = _textCtrlGAsCrossoverBlendingNbPts->GetValue();
    pConfig->Write("/GAs/CrossoverBlendingPointsNb", GAsCrossoverBlendingPointsNb);
    bool GAsCrossoverBlendingShareBeta = _checkBoxGAsCrossoverBlendingShareBeta->GetValue();
    pConfig->Write("/GAs/CrossoverBlendingShareBeta", GAsCrossoverBlendingShareBeta);
    wxString GAsCrossoverLinearPointsNb = _textCtrlGAsCrossoverLinearNbPts->GetValue();
    pConfig->Write("/GAs/CrossoverLinearPointsNb", GAsCrossoverLinearPointsNb);
    wxString GAsCrossoverHeuristicPointsNb = _textCtrlGAsCrossoverHeuristicNbPts->GetValue();
    pConfig->Write("/GAs/CrossoverHeuristicPointsNb", GAsCrossoverHeuristicPointsNb);
    bool GAsCrossoverHeuristicShareBeta = _checkBoxGAsCrossoverHeuristicShareBeta->GetValue();
    pConfig->Write("/GAs/CrossoverHeuristicShareBeta", GAsCrossoverHeuristicShareBeta);
    wxString GAsCrossoverBinaryLikePointsNb = _textCtrlGAsCrossoverBinLikeNbPts->GetValue();
    pConfig->Write("/GAs/CrossoverBinaryLikePointsNb", GAsCrossoverBinaryLikePointsNb);
    bool GAsCrossoverBinaryLikeShareBeta = _checkBoxGAsCrossoverBinLikeShareBeta->GetValue();
    pConfig->Write("/GAs/CrossoverBinaryLikeShareBeta", GAsCrossoverBinaryLikeShareBeta);
    wxString GAsMutationsUniformConstantProbability = _textCtrlGAsMutationsUniformCstProb->GetValue();
    pConfig->Write("/GAs/MutationsUniformConstantProbability", GAsMutationsUniformConstantProbability);
    wxString GAsMutationsNormalConstantProbability = _textCtrlGAsMutationsNormalCstProb->GetValue();
    pConfig->Write("/GAs/MutationsNormalConstantProbability", GAsMutationsNormalConstantProbability);
    wxString GAsMutationsNormalConstantStdDevRatioRange = _textCtrlGAsMutationsNormalCstStdDev->GetValue();
    pConfig->Write("/GAs/MutationsNormalConstantStdDevRatioRange", GAsMutationsNormalConstantStdDevRatioRange);
    wxString GAsMutationsUniformVariableMaxGensNbVar = _textCtrlGAsMutationsUniformVarMaxGensNb->GetValue();
    pConfig->Write("/GAs/MutationsUniformVariableMaxGensNbVar", GAsMutationsUniformVariableMaxGensNbVar);
    wxString GAsMutationsUniformVariableProbabilityStart = _textCtrlGAsMutationsUniformVarProbStart->GetValue();
    pConfig->Write("/GAs/MutationsUniformVariableProbabilityStart", GAsMutationsUniformVariableProbabilityStart);
    wxString GAsMutationsUniformVariableProbabilityEnd = _textCtrlGAsMutationsUniformVarProbEnd->GetValue();
    pConfig->Write("/GAs/MutationsUniformVariableProbabilityEnd", GAsMutationsUniformVariableProbabilityEnd);
    wxString GAsMutationsNormalVariableMaxGensNbVarProb = _textCtrlGAsMutationsNormalVarMaxGensNbProb->GetValue();
    pConfig->Write("/GAs/MutationsNormalVariableMaxGensNbVarProb", GAsMutationsNormalVariableMaxGensNbVarProb);
    wxString GAsMutationsNormalVariableMaxGensNbVarStdDev = _textCtrlGAsMutationsNormalVarMaxGensNbStdDev->GetValue();
    pConfig->Write("/GAs/MutationsNormalVariableMaxGensNbVarStdDev", GAsMutationsNormalVariableMaxGensNbVarStdDev);
    wxString GAsMutationsNormalVariableProbabilityStart = _textCtrlGAsMutationsNormalVarProbStart->GetValue();
    pConfig->Write("/GAs/MutationsNormalVariableProbabilityStart", GAsMutationsNormalVariableProbabilityStart);
    wxString GAsMutationsNormalVariableProbabilityEnd = _textCtrlGAsMutationsNormalVarProbEnd->GetValue();
    pConfig->Write("/GAs/MutationsNormalVariableProbabilityEnd", GAsMutationsNormalVariableProbabilityEnd);
    wxString GAsMutationsNormalVariableStdDevStart = _textCtrlGAsMutationsNormalVarStdDevStart->GetValue();
    pConfig->Write("/GAs/MutationsNormalVariableStdDevStart", GAsMutationsNormalVariableStdDevStart);
    wxString GAsMutationsNormalVariableStdDevEnd = _textCtrlGAsMutationsNormalVarStdDevEnd->GetValue();
    pConfig->Write("/GAs/MutationsNormalVariableStdDevEnd", GAsMutationsNormalVariableStdDevEnd);
    wxString GAsMutationsNonUniformProb = _textCtrlGAsMutationsNonUniformProb->GetValue();
    pConfig->Write("/GAs/MutationsNonUniformProbability", GAsMutationsNonUniformProb);
    wxString GAsMutationsNonUniformMaxGensNbVar = _textCtrlGAsMutationsNonUniformGensNb->GetValue();
    pConfig->Write("/GAs/MutationsNonUniformMaxGensNbVar", GAsMutationsNonUniformMaxGensNbVar);
    wxString GAsMutationsNonUniformMinRate = _textCtrlGAsMutationsNonUniformMinRate->GetValue();
    pConfig->Write("/GAs/MutationsNonUniformMinRate", GAsMutationsNonUniformMinRate);
    wxString GAsMutationsMultiScaleProb = _textCtrlGAsMutationsMultiScaleProb->GetValue();
    pConfig->Write("/GAs/MutationsMultiScaleProbability", GAsMutationsMultiScaleProb);

    pConfig->Flush();
}

/*
void asFrameOptimizer::OnIdle( wxCommandEvent& event )
{
    wxString state = asGetState();
    _staticTextState->SetLabel(state);
}
*/
void asFrameOptimizer::Launch(wxCommandEvent& event) {
    wxBusyCursor wait;

    SaveOptions();

    try {
        switch (_choiceMethod->GetSelection()) {
            case wxNOT_FOUND: {
                wxLogError(_("Wrong method selection."));
                break;
            }
            case 0:  // Single
            {
                _methodCalibrator = new asMethodCalibratorSingle();
                break;
            }
            case 1:  // Classic
            {
                _methodCalibrator = new asMethodCalibratorClassic();
                break;
            }
            case 2:  // Classic+
            {
                _methodCalibrator = new asMethodCalibratorClassic();
                break;
            }
            case 3:  // Variables exploration with classic+
            {
                _methodCalibrator = new asMethodCalibratorClassicVarExplo();
                break;
            }
            case 4:  // Random sets
            {
                _methodCalibrator = new asMethodOptimizerMC();
                break;
            }
            case 5:  // Genetic algorithms
            {
                _methodCalibrator = new asMethodOptimizerGAs();
                break;
            }
            case 6:  // Scores evaluation
            {
                _methodCalibrator = new asMethodCalibratorEvaluateAllScores();
                break;
            }
            case 7:  // Only predictand values
            {
                _methodCalibrator = new asMethodCalibratorSingleOnlyValues();
                break;
            }
            case 8:  // Only analog dates
            {
                _methodCalibrator = new asMethodCalibratorSingleOnlyDates();
                break;
            }
            default:
                wxLogError(_("Chosen method not defined yet."));
        }

        if (_methodCalibrator) {
            _methodCalibrator->SetParamsFilePath(_filePickerParameters->GetPath());
            _methodCalibrator->SetPredictandDBFilePath(_filePickerPredictand->GetPath());
            _methodCalibrator->SetPredictorDataDir(_dirPickerPredictor->GetPath());
            _methodCalibrator->Manager();
        }
    } catch (std::bad_alloc& ba) {
        wxString msg(ba.what(), wxConvUTF8);
        wxLogError(_("Bad allocation caught: %s"), msg);
        wxLogError(_("Failed to process the calibration."));
    } catch (runtime_error& e) {
        wxString msg(e.what(), wxConvUTF8);
        wxLogError(_("Exception caught: %s"), msg);
        wxLogError(_("Failed to process the optimization."));
    }

    wxDELETE(_methodCalibrator);

    wxMessageBox(_("Optimizer over."));
}
