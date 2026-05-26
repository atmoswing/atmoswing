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

#include "asMethodCalibrator.h"

#include "asIncludes.h"
#include "asThreadPreloadArchiveData.h"

#ifndef UNIT_TESTING

#endif

asMethodCalibrator::asMethodCalibrator()
    : asMethodStandard(),
      _scoreOrder(Asc),
      _scoreValid(NAN),
      _validationMode(false),
      _useBatches(false),
      _batchStart(0),
      _batchEnd(0) {
    // Seeds the random generator
    asInitRandom();
}

asMethodCalibrator::~asMethodCalibrator() {
    DeletePreloadedArchiveData();
}

bool asMethodCalibrator::Manager() {
    // Set unresponsive to speedup
    g_responsive = false;

    // Reset the score of the climatology
    _scoreClimatology.clear();

    // Seeds the random generator
    asInitRandom();

    // Load parameters
    asParametersCalibration params;
    if (!params.LoadFromFile(_paramsFilePath)) {
        return false;
    }
    if (!_predictandStationIds.empty()) {
        vvi idsVect;
        idsVect.push_back(_predictandStationIds);
        params.SetPredictandStationIdsVector(idsVect);
    }
    params.InitValues();
    _originalParams = params;

    // Load the Predictand DB
    if (!LoadPredictandDB(_predictandDBFilePath)) {
        return false;
    }

    // Get the score order
    LoadScoreOrder(params);

    // Watch
    wxStopWatch sw;

    // Calibrate
    if (Calibrate(params)) {
        // Display processing time
        wxLogMessage(_("The whole processing took %.3f min to execute"), float(sw.Time()) / 60000.0f);
#if USE_GUI
        wxLogStatus(_("Calibration over."));
#endif
    } else {
        wxLogError(_("The parameters could not be calibrated"));
    }

    // Delete preloaded data and cleanup
    DeletePreloadedArchiveData();

    return true;
}

void asMethodCalibrator::LoadScoreOrder(asParametersCalibration& params) {
    asScore* score = asScore::GetInstance(params.GetScoreName());
    Order scoreOrder = score->GetOrder();
    SetScoreOrder(scoreOrder);
    wxDELETE(score);
}

void asMethodCalibrator::ClearAll() {
    _parametersTemp.clear();
    _scoresCalibTemp.clear();
    _parameters.clear();
    _scoresCalib.clear();
    _scoreValid = NAN;
}

void asMethodCalibrator::ClearTemp() {
    _parametersTemp.clear();
    _scoresCalibTemp.clear();
}

bool asMethodCalibrator::PushBackBestTemp() {
    if (!SortScoresAndParametersTemp()) {
        return false;
    }

    PushBackFirstTemp();

    return true;
}

void asMethodCalibrator::RemoveNaNsInTemp() {
    wxASSERT(_parametersTemp.size() == _scoresCalibTemp.size());

    vector<asParametersCalibration> copyParametersTemp;
    vf copyScoresCalibTemp;

    for (int i = 0; i < _scoresCalibTemp.size(); i++) {
        if (!isnan(_scoresCalibTemp[i])) {
            copyScoresCalibTemp.push_back(_scoresCalibTemp[i]);
            copyParametersTemp.push_back(_parametersTemp[i]);
        }
    }

    _scoresCalibTemp = copyScoresCalibTemp;
    _parametersTemp = copyParametersTemp;

    wxASSERT(_parametersTemp.size() == _scoresCalibTemp.size());
    wxASSERT(!_parametersTemp.empty());
}

void asMethodCalibrator::KeepBestTemp() {
    SortScoresAndParametersTemp();
    KeepFirstTemp();
}

void asMethodCalibrator::PushBackFirstTemp() {
    wxASSERT(!_parametersTemp.empty());
    wxASSERT(!_scoresCalibTemp.empty());
    _parameters.push_back(_parametersTemp[0]);
    _scoresCalib.push_back(_scoresCalibTemp[0]);
}

void asMethodCalibrator::KeepFirstTemp() {
    wxASSERT(!_parameters.empty());
    wxASSERT(!_parametersTemp.empty());
    wxASSERT(!_scoresCalibTemp.empty());
    _parameters[0] = _parametersTemp[0];
    if (_scoresCalib.empty()) {
        _scoresCalib.push_back(_scoresCalibTemp[0]);
    } else {
        _scoresCalib[0] = _scoresCalibTemp[0];
    }
}

bool asMethodCalibrator::SortScoresAndParametersTemp() {
    wxASSERT(_scoresCalibTemp.size() == _parametersTemp.size());
    wxASSERT(!_scoresCalibTemp.empty());
    wxASSERT(!_parametersTemp.empty());

    if (_parametersTemp.size() == 1) return true;

    // Sort according to the score
    a1f vIndices = a1f::LinSpaced(_scoresCalibTemp.size(), 0, _scoresCalibTemp.size() - 1);
    if (!asSortArrays(&_scoresCalibTemp[0], &_scoresCalibTemp[_scoresCalibTemp.size() - 1], &vIndices[0],
                      &vIndices[_scoresCalibTemp.size() - 1], _scoreOrder)) {
        return false;
    }

    // Sort the parameters sets as the scores
    vector<asParametersCalibration> copyParameters;
    for (int i = 0; i < _scoresCalibTemp.size(); i++) {
        copyParameters.push_back(_parametersTemp[i]);
    }
    for (int i = 0; i < _scoresCalibTemp.size(); i++) {
        int index = (int)vIndices(i);
        _parametersTemp[i] = copyParameters[index];
    }

    return true;
}

bool asMethodCalibrator::PushBackInTempIfBetter(asParametersCalibration& params, asResultsTotalScore& scoreFinal) {
    float thisScore = scoreFinal.GetScore();

    switch (_scoreOrder) {
        case Asc:
            if (thisScore < _scoresCalib[0]) {
                _parametersTemp.push_back(params);
                _scoresCalibTemp.push_back(thisScore);
                return true;
            }
            break;

        case Desc:
            if (thisScore > _scoresCalib[0]) {
                _parametersTemp.push_back(params);
                _scoresCalibTemp.push_back(thisScore);
                return true;
            }
            break;

        default:
            throw runtime_error(_("The score order is not correctly defined."));
    }

    return false;
}

bool asMethodCalibrator::KeepIfBetter(asParametersCalibration& params, asResultsTotalScore& scoreFinal) {
    float thisScore = scoreFinal.GetScore();

    switch (_scoreOrder) {
        case Asc:
            if (thisScore < _scoresCalib[0]) {
                wxASSERT(!_parameters.empty());
                wxASSERT(!_scoresCalib.empty());
                _parameters[0] = params;
                _scoresCalib[0] = thisScore;
                return true;
            }
            break;

        case Desc:
            if (thisScore > _scoresCalib[0]) {
                wxASSERT(!_parameters.empty());
                wxASSERT(!_scoresCalib.empty());
                _parameters[0] = params;
                _scoresCalib[0] = thisScore;
                return true;
            }
            break;

        default:
            throw runtime_error(_("The score order is not correctly defined."));
    }

    return false;
}

bool asMethodCalibrator::SetBestParameters(asResultsParametersArray& results) {
    wxASSERT(!_parameters.empty());
    wxASSERT(!_scoresCalib.empty());

    // Extract selected parameters & best parameters
    float bestScore = _scoresCalib[0];
    int bestScoreRow = 0;

    for (int i = 0; i < _parameters.size(); i++) {
        if (_scoreOrder == Asc) {
            if (_scoresCalib[i] < bestScore) {
                bestScore = _scoresCalib[i];
                bestScoreRow = i;
            }
        } else {
            if (_scoresCalib[i] > bestScore) {
                bestScore = _scoresCalib[i];
                bestScoreRow = i;
            }
        }
    }

    if (bestScoreRow != 0) {
        // Re-validate
        SaveDetails(&_parameters[bestScoreRow]);
        Validate(&_parameters[bestScoreRow]);
    }

    results.Add(_parameters[bestScoreRow], _scoresCalib[bestScoreRow], _scoreValid);

    return true;
}

wxString asMethodCalibrator::GetStationIdsList(vi& stationIds) const {
    wxString id;

    if (stationIds.size() == 1) {
        id << stationIds[0];
    } else if (stationIds.size() > 10) {
        id << stationIds[0];
        id << '-';
        id << stationIds[stationIds.size() - 1];
    } else {
        for (int i = 0; i < (int)stationIds.size(); i++) {
            id << stationIds[i];
            if (i < (int)stationIds.size() - 1) {
                id << ",";
            }
        }
    }

    return id;
}

double asMethodCalibrator::GetTimeStartCalibration(asParametersScoring* params) const {
    return params->GetCalibrationStart() + params->GetTimeShiftDays();
}

double asMethodCalibrator::GetTimeEndCalibration(asParametersScoring* params) const {
    return params->GetCalibrationEnd() - params->GetTimeSpanDays();
}

double asMethodCalibrator::GetEffectiveArchiveDataStart(asParameters* params) const {
    auto paramsScoring = dynamic_cast<asParametersScoring*>(params);
    wxASSERT(paramsScoring);

    return std::min(GetTimeStartCalibration(paramsScoring), GetTimeStartArchive(paramsScoring));
}

double asMethodCalibrator::GetEffectiveArchiveDataEnd(asParameters* params) const {
    auto paramsScoring = dynamic_cast<asParametersScoring*>(params);
    wxASSERT(paramsScoring);

    return std::max(GetTimeEndCalibration(paramsScoring), GetTimeEndArchive(paramsScoring));
}

va1f asMethodCalibrator::GetClimatologyData(asParametersScoring* params) {
    vi stationIds = params->GetPredictandStationIds();

    // Get start and end dates
    a1d predictandTime = _predictandDB->GetTime();
    auto predictandTimeDays = float(params->GetPredictandTimeHours() / 24.0);
    double timeStart, timeEnd;
    timeStart = std::max(predictandTime[0], params->GetCalibrationStart());
    timeStart = floor(timeStart) + predictandTimeDays;
    timeEnd = std::min(predictandTime[predictandTime.size() - 1], params->GetCalibrationEnd());
    timeEnd = floor(timeEnd) + predictandTimeDays;

    if (predictandTime.size() < 1) {
        wxLogError(_("An unexpected error occurred."));
        return {stationIds.size(), a1f(1)};
    }

    // Check if data are effectively available for this period
    int indexPredictandTimeStart = asFindCeil(&predictandTime[0], &predictandTime[predictandTime.size() - 1],
                                              timeStart);
    int indexPredictandTimeEnd = asFindFloor(&predictandTime[0], &predictandTime[predictandTime.size() - 1], timeEnd);

    if (indexPredictandTimeStart < 0 || indexPredictandTimeEnd < 0) {
        wxLogError(_("An unexpected error occurred."));
        return {stationIds.size(), a1f(1)};
    }

    for (int iStat = 0; iStat < (int)stationIds.size(); iStat++) {
        a1f predictandDataNorm = _predictandDB->GetDataNormalizedStation(stationIds[iStat]);

        while (isnan(predictandDataNorm(indexPredictandTimeStart))) {
            indexPredictandTimeStart++;
        }
        while (isnan(predictandDataNorm(indexPredictandTimeEnd))) {
            indexPredictandTimeEnd--;
            if (indexPredictandTimeEnd < 0) {
                wxLogError(_("An unexpected error occurred."));
                return {stationIds.size(), a1f(1)};
            }
        }
    }

    if (indexPredictandTimeStart < 0 || indexPredictandTimeEnd < 0) {
        wxLogError(_("An unexpected error occurred."));
        return {stationIds.size(), a1f(1)};
    }

    timeStart = predictandTime[indexPredictandTimeStart];
    timeStart = floor(timeStart) + predictandTimeDays;
    timeEnd = predictandTime[indexPredictandTimeEnd];
    timeEnd = floor(timeEnd) + predictandTimeDays;
    indexPredictandTimeStart = asFindCeil(&predictandTime[0], &predictandTime[predictandTime.size() - 1], timeStart);
    indexPredictandTimeEnd = asFindFloor(&predictandTime[0], &predictandTime[predictandTime.size() - 1], timeEnd);

    if (indexPredictandTimeStart < 0 || indexPredictandTimeEnd < 0) {
        wxLogError(_("An unexpected error occurred."));
        return {stationIds.size(), a1f(1)};
    }

    // Get index step
    double predictandTimeStep = predictandTime[1] - predictandTime[0];
    double targetTimeStep = params->GetTargetTimeStepHours() / 24.0;
    int indexStep = int(targetTimeStep / predictandTimeStep);

    // Get vector length
    int dataLength = (indexPredictandTimeEnd - indexPredictandTimeStart) / indexStep + 1;

    // Process the climatology score
    va1f climatologyData(stationIds.size(), a1f(dataLength));
    for (int iStat = 0; iStat < stationIds.size(); iStat++) {
        a1f predictandDataNorm = _predictandDB->GetDataNormalizedStation(stationIds[iStat]);

        // Set data
        int counter = 0;
        for (int i = indexPredictandTimeStart; i <= indexPredictandTimeEnd; i += indexStep) {
            climatologyData[iStat][counter] = predictandDataNorm[i];
            counter++;
        }
        wxASSERT(dataLength == counter);
    }

    return climatologyData;
}

bool asMethodCalibrator::PreloadDataOnly(asParametersScoring* params) {
    // Archive date array
    asTimeArray timeArrayArchive(GetTimeStartArchive(params), GetTimeEndArchive(params),
                                 params->GetAnalogsTimeStepHours(), params->GetTimeArrayAnalogsMode());
    if (params->HasValidationPeriod())  // remove validation years
    {
        timeArrayArchive.SetForbiddenYears(params->GetValidationYearsVector());
    }
    timeArrayArchive.Init();

    // Target date array
    asTimeArray timeArrayTarget(GetTimeStartCalibration(params), GetTimeEndCalibration(params),
                                params->GetTargetTimeStepHours(), params->GetTimeArrayTargetMode());

    // Remove validation years
    if (!_validationMode && params->HasValidationPeriod()) {
        timeArrayTarget.SetForbiddenYears(params->GetValidationYearsVector());
    }

    if (params->GetTimeArrayTargetMode().CmpNoCase("predictand_thresholds") == 0 ||
        params->GetTimeArrayTargetMode().CmpNoCase("PredictandThresholds") == 0) {
        vi stations = params->GetPredictandStationIds();
        if (stations.size() > 1) {
            wxLogError(_("You cannot use predictand thresholds with the multivariate approach."));
            return false;
        }

        if (!timeArrayTarget.Init(*_predictandDB, params->GetTimeArrayTargetPredictandSerieName(), stations[0],
                                  params->GetTimeArrayTargetPredictandMinThreshold(),
                                  params->GetTimeArrayTargetPredictandMaxThreshold())) {
            wxLogError(_("The time array mode for the target dates is not correctly defined."));
            return false;
        }
    } else {
        if (!timeArrayTarget.Init()) {
            wxLogError(_("The time array mode for the target dates is not correctly defined."));
            return false;
        }
    }

    // If in validation mode, only keep validation years
    if (_validationMode) {
        timeArrayTarget.KeepOnlyYears(params->GetValidationYearsVector());
    }

    // Data date array
    double timeStartData = std::min(GetTimeStartCalibration(params), GetTimeStartArchive(params));
    double timeEndData = std::max(GetTimeEndCalibration(params), GetTimeEndArchive(params));
    wxString timeArrayMode = params->GetTimeArrayAnalogsMode();
    if (timeArrayMode.IsSameAs("days_interval")) {
        timeArrayMode = "simple";
    }
    asTimeArray timeArrayData(timeStartData, timeEndData, params->GetAnalogsTimeStepHours(), timeArrayMode);
    timeArrayData.Init();

    // Load the predictor data
    vector<asPredictor*> predictors;
    if (!LoadArchiveData(predictors, params, 0, timeStartData, timeEndData)) {
        wxLogError(_("Failed loading predictor data."));
        Cleanup(predictors);
        return false;
    }

    Cleanup(predictors);

    return true;
}

bool asMethodCalibrator::GetAnalogsDates(asResultsDates& results, asParametersScoring* params, int iStep,
                                         bool& containsNaNs) {
    // Initialize the result object
    results.SetCurrentStep(iStep);
    results.Init(params);

    // Archive date array
    asTimeArray timeArrayArchive(GetTimeStartArchive(params), GetTimeEndArchive(params),
                                 params->GetAnalogsTimeStepHours(), params->GetTimeArrayAnalogsMode());
    if (params->HasValidationPeriod())  // remove validation years
    {
        timeArrayArchive.SetForbiddenYears(params->GetValidationYearsVector());
    }
    timeArrayArchive.Init();

    // Target date array
    asTimeArray timeArrayTarget(GetTimeStartCalibration(params), GetTimeEndCalibration(params),
                                params->GetTargetTimeStepHours(), params->GetTimeArrayTargetMode());

    // Remove validation years
    if (!_validationMode && params->HasValidationPeriod()) {
        timeArrayTarget.SetForbiddenYears(params->GetValidationYearsVector());
    }

    if (!_validationMode && (params->GetTimeArrayTargetMode().CmpNoCase("predictand_thresholds") == 0 ||
                             params->GetTimeArrayTargetMode().CmpNoCase("PredictandThresholds") == 0)) {
        vi stations = params->GetPredictandStationIds();
        if (stations.size() > 1) {
            wxLogError(_("You cannot use predictand thresholds with the multivariate approach."));
            return false;
        }

        if (!timeArrayTarget.Init(*_predictandDB, params->GetTimeArrayTargetPredictandSerieName(), stations[0],
                                  params->GetTimeArrayTargetPredictandMinThreshold(),
                                  params->GetTimeArrayTargetPredictandMaxThreshold())) {
            wxLogError(_("The time array mode for the target dates is not correctly defined."));
            return false;
        }
    } else {
        if (!timeArrayTarget.Init()) {
            wxLogError(_("The time array mode for the target dates is not correctly defined."));
            return false;
        }
    }

    // If in validation mode, only keep validation years
    if (_validationMode) {
        timeArrayTarget.KeepOnlyYears(params->GetValidationYearsVector());
    }

    if (!_validationMode && _useBatches) {
        timeArrayTarget.KeepOnlyRange(_batchStart, _batchEnd);
    }

    // Data date array
    double timeStartData = std::min(GetTimeStartCalibration(params), GetTimeStartArchive(params));
    double timeEndData = std::max(GetTimeEndCalibration(params), GetTimeEndArchive(params));
    wxString timeArrayMode = params->GetTimeArrayAnalogsMode();
    if (timeArrayMode.IsSameAs("days_interval")) {
        timeArrayMode = "simple";
    }
    asTimeArray timeArrayData(timeStartData, timeEndData, params->GetAnalogsTimeStepHours(), timeArrayMode);
    timeArrayData.Init();

    // Check on the archive length
    if (timeArrayArchive.GetSize() < 100) {
        wxLogError(_("The time array is not consistent in asMethodCalibrator::GetAnalogsDates: size=%d."),
                   timeArrayArchive.GetSize());
        return false;
    }

    // Load the predictor data
    vector<asPredictor*> predictors;
    if (!LoadArchiveData(predictors, params, iStep, timeStartData, timeEndData)) {
        wxLogError(_("Failed loading predictor data."));
        Cleanup(predictors);
        return false;
    }

    // Create the criterion
    vector<asCriteria*> criteria;
    for (int iPtor = 0; iPtor < params->GetPredictorsNb(iStep); iPtor++) {
        // Instantiate a score object. The vector takes ownership; Cleanup() deletes the
        // raw pointers below, so release the unique_ptr at the boundary.
        criteria.push_back(asCriteria::GetInstance(params->GetPredictorCriteria(iStep, iPtor)).release());
    }

    // Check time sizes
#ifdef _DEBUG
    int prevTimeSize = 0;

    for (int i = 0; i < predictors.size(); i++) {
        if (i > 0) {
            wxASSERT(predictors[i]->GetTimeSize() == prevTimeSize);
        }
        prevTimeSize = predictors[i]->GetTimeSize();
    }
#endif  // _DEBUG

    // Inline the data when possible
    for (int iPtor = 0; iPtor < (int)predictors.size(); iPtor++) {
        if (criteria[iPtor]->CanUseInline()) {
            predictors[iPtor]->Inline();
        }
    }

    if (!asProcessor::GetAnalogsDates(predictors, predictors, timeArrayData, timeArrayArchive, timeArrayData,
                                      timeArrayTarget, criteria, params, iStep, results, containsNaNs)) {
        wxLogError(_("Failed processing the analogs dates."));
        Cleanup(predictors);
        Cleanup(criteria);
        return false;
    }

    Cleanup(predictors);
    Cleanup(criteria);

    return true;
}

bool asMethodCalibrator::GetAnalogsSubDates(asResultsDates& results, asParametersScoring* params,
                                            asResultsDates& anaDates, int iStep, bool& containsNaNs) {
    // Initialize the result object
    results.SetCurrentStep(iStep);
    results.Init(params);

    // Date array object instantiation for the processor
    double timeStart = params->GetArchiveStart();
    double timeEnd = params->GetArchiveEnd() - params->GetTimeSpanDays();
    asTimeArray timeArrayArchive(timeStart, timeEnd, params->GetAnalogsTimeStepHours(),
                                 params->GetTimeArrayTargetMode());
    timeArrayArchive.Init();

    // Load the predictor data
    vector<asPredictor*> predictors;
    if (!LoadArchiveData(predictors, params, iStep, timeStart, timeEnd)) {
        wxLogError(_("Failed loading predictor data."));
        Cleanup(predictors);
        return false;
    }

    // Create the score objects. The vector takes ownership; Cleanup() deletes the
    // raw pointers below, so release the unique_ptr at the boundary.
    vector<asCriteria*> criteria;
    for (int iPtor = 0; iPtor < params->GetPredictorsNb(iStep); iPtor++) {
        criteria.push_back(asCriteria::GetInstance(params->GetPredictorCriteria(iStep, iPtor)).release());
    }

    // Inline the data when possible
    for (int iPtor = 0; iPtor < (int)predictors.size(); iPtor++) {
        if (criteria[iPtor]->CanUseInline()) {
            predictors[iPtor]->Inline();
        }
    }

    // Send data and criteria to processor
    if (!asProcessor::GetAnalogsSubDates(predictors, predictors, timeArrayArchive, timeArrayArchive, anaDates, criteria,
                                         params, iStep, results, containsNaNs)) {
        wxLogError(_("Failed processing the analogs dates."));
        Cleanup(predictors);
        Cleanup(criteria);
        return false;
    }

    Cleanup(predictors);
    Cleanup(criteria);

    return true;
}

bool asMethodCalibrator::GetAnalogsValues(asResultsValues& results, asParametersScoring* params,
                                          asResultsDates& anaDates, int iStep) {
    // Initialize the result object
    results.SetCurrentStep(iStep);
    results.Init(params);

    // Set the predictand values to the corresponding analog dates
    wxASSERT(_predictandDB);
    if (!asProcessor::GetAnalogsValues(*_predictandDB, anaDates, params, results)) {
        wxLogError(_("Failed setting the predictand values to the corresponding analog dates."));
        return false;
    }

    return true;
}

bool asMethodCalibrator::GetAnalogsScores(asResultsScores& results, asParametersScoring* params,
                                          asResultsValues& anaValues, int iStep) {
    // Initialize the result object
    results.SetCurrentStep(iStep);
    results.Init(params);

    // Instantiate a score object
    asScore* score = asScore::GetInstance(params->GetScoreName());
    score->SetQuantile(params->GetScoreQuantile());
    score->SetThreshold(params->GetScoreThreshold());
    score->SetOnMean(params->GetOnMean());

    if (score->UsesClimatology() && _scoreClimatology.empty()) {
        wxLogVerbose(_("Processing the score of the climatology."));

        va1f climatologyData = GetClimatologyData(params);
        vi stationIds = params->GetPredictandStationIds();
        _scoreClimatology.resize(stationIds.size());

        for (int iStat = 0; iStat < (int)stationIds.size(); iStat++) {
            score->ProcessScoreClimatology(anaValues.GetTargetValues()[iStat], climatologyData[iStat]);
            _scoreClimatology[iStat] = score->GetScoreClimatology();
        }
    }

    if (!asProcessorScore::GetAnalogsScores(anaValues, score, params, results, _scoreClimatology)) {
        wxLogError(_("Failed processing the score."));
        wxDELETE(score);
        return false;
    }

    wxDELETE(score);

    return true;
}

bool asMethodCalibrator::GetAnalogsTotalScore(asResultsTotalScore& results, asParametersScoring* params,
                                              asResultsScores& anaScores, int iStep) {
    // Initialize the result object
    results.SetCurrentStep(iStep);
    results.Init();

    // Date array object instantiation for the final score
    double timeStart = params->GetCalibrationStart();
    double timeEnd = params->GetCalibrationEnd() + 1;
    while (timeEnd > params->GetCalibrationEnd() + 0.999) {
        timeEnd -= params->GetTargetTimeStepHours() / 24.0;
    }
    asTimeArray timeArray(timeStart, timeEnd, params->GetTargetTimeStepHours(), params->GetScoreTimeArrayMode());
    timeArray.Init();

    // Pass data and score to processor
    if (!asProcessorScore::GetAnalogsTotalScore(anaScores, timeArray, params, results)) {
        wxLogError(_("Failed to process the final score."));
        return false;
    }

    return true;
}

bool asMethodCalibrator::SubProcessAnalogsNumber(asParametersCalibration& params, asResultsDates& anaDatesPrevious,
                                                 int iStep) {
    vi analogsNbVect = params.GetAnalogsNumberVector(iStep);

    // Cannot be superior to previous analogs nb
    int rowEnd = int(analogsNbVect.size() - 1);
    if (iStep > 0) {
        int prevAnalogsNb = params.GetAnalogsNumber(iStep - 1);
        if (prevAnalogsNb < analogsNbVect[analogsNbVect.size() - 1]) {
            rowEnd = asFindFloor(&analogsNbVect[0], &analogsNbVect[analogsNbVect.size() - 1], prevAnalogsNb);
        }
    }

    asResultsDates anaDates;
    asResultsValues anaValues;

    if (rowEnd < 0) {
        wxLogError(_("Error assessing the number of analogues."));
        return false;
    }

    // Set the maximum and let play with the analogs nb on the score (faster)
    params.SetAnalogsNumber(iStep, analogsNbVect[rowEnd]);

    // Process first the dates and the values
    bool containsNaNs = false;
    if (iStep == 0) {
        if (!GetAnalogsDates(anaDates, &params, iStep, containsNaNs)) return false;
    } else {
        if (!GetAnalogsSubDates(anaDates, &params, anaDatesPrevious, iStep, containsNaNs)) return false;
    }
    if (containsNaNs) {
        wxLogError(_("The dates selection contains NaNs"));
        return false;
    }

    asResultsDates anaDatesTmp(anaDates);
    a2f dates = anaDates.GetAnalogsDates();

    // If at the end of the chain
    if (iStep == params.GetStepsNb() - 1) {
        if (!GetAnalogsValues(anaValues, &params, anaDates, iStep)) return false;

        asResultsScores anaScores;
        asResultsTotalScore anaScoreFinal;

        for (int i = 0; i <= rowEnd; i++) {
            params.SetAnalogsNumber(iStep, analogsNbVect[i]);

            // Fixes and checks
            params.FixAnalogsNb();

            // Extract analogs dates from former results
            a2f subDates = dates.leftCols(params.GetAnalogsNumber(iStep));
            anaDatesTmp.SetAnalogsDates(subDates);

            if (!GetAnalogsScores(anaScores, &params, anaValues, iStep)) return false;
            if (!GetAnalogsTotalScore(anaScoreFinal, &params, anaScores, iStep)) return false;

            _parametersTemp.push_back(params);
            _scoresCalibTemp.push_back(anaScoreFinal.GetScore());
        }

    } else {
        for (int i = 0; i <= rowEnd; i++) {
            params.SetAnalogsNumber(iStep, analogsNbVect[i]);

            // Fixes and checks
            params.FixAnalogsNb();

            // Extract analogs dates from former results
            a2f subDates = dates.leftCols(params.GetAnalogsNumber(iStep));
            anaDatesTmp.SetAnalogsDates(subDates);

            // Continue
            if (!SubProcessAnalogsNumber(params, anaDatesTmp, iStep + 1)) return false;
        }
    }

    return true;
}

bool asMethodCalibrator::SaveDetails(asParametersCalibration* params) {
    asResultsDates anaDatesPrevious;
    asResultsDates anaDates;
    asResultsValues anaValues;
    asResultsScores anaScores;
    asResultsTotalScore anaScoreFinal;

    // Process every step one after the other
    int stepsNb = params->GetStepsNb();
    for (int iStep = 0; iStep < stepsNb; iStep++) {
        bool containsNaNs = false;
        if (iStep == 0) {
            if (!GetAnalogsDates(anaDates, params, iStep, containsNaNs)) return false;
        } else {
            anaDatesPrevious = anaDates;
            if (!GetAnalogsSubDates(anaDates, params, anaDatesPrevious, iStep, containsNaNs)) return false;
        }
        if (containsNaNs) {
            wxLogError(_("The dates selection contains NaNs"));
            return false;
        }
    }
    if (!GetAnalogsValues(anaValues, params, anaDates, stepsNb - 1)) return false;
    if (!GetAnalogsScores(anaScores, params, anaValues, stepsNb - 1)) return false;
    if (!GetAnalogsTotalScore(anaScoreFinal, params, anaScores, stepsNb - 1)) return false;

    anaDates.SetSubFolder("calibration");
    anaDates.Save();
    anaValues.SetSubFolder("calibration");
    anaValues.Save();
    anaScores.SetSubFolder("calibration");
    anaScores.Save();

    return true;
}

bool asMethodCalibrator::Validate(asParametersCalibration* params) {
    if (wxFileConfig::Get()->ReadBool("/SkipValidation", false)) {
        return true;
    }

    if (!params->HasValidationPeriod()) {
        wxLogWarning(_("The parameters have no validation period !"));
        return true;
    }

    _validationMode = true;

    asResultsDates anaDatesPrevious;
    asResultsDates anaDates;
    asResultsValues anaValues;
    asResultsScores anaScores;
    asResultsTotalScore anaScoreFinal;

    // Process every step one after the other
    int stepsNb = params->GetStepsNb();
    for (int iStep = 0; iStep < stepsNb; iStep++) {
        bool containsNaNs = false;
        if (iStep == 0) {
            if (!GetAnalogsDates(anaDates, params, iStep, containsNaNs)) return false;
        } else {
            anaDatesPrevious = anaDates;
            if (!GetAnalogsSubDates(anaDates, params, anaDatesPrevious, iStep, containsNaNs)) return false;
        }
        if (containsNaNs) {
            wxLogError(_("The dates selection contains NaNs"));
            return false;
        }
    }
    if (!GetAnalogsValues(anaValues, params, anaDates, stepsNb - 1)) return false;
    if (!GetAnalogsScores(anaScores, params, anaValues, stepsNb - 1)) return false;
    if (!GetAnalogsTotalScore(anaScoreFinal, params, anaScores, stepsNb - 1)) return false;

    anaDates.SetSubFolder("validation");
    anaDates.Save();
    anaValues.SetSubFolder("validation");
    anaValues.Save();
    anaScores.SetSubFolder("validation");
    anaScores.Save();

    _scoreValid = anaScoreFinal.GetScore();

    _validationMode = false;

    return true;
}
