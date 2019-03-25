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
#include "asThreadPreloadArchiveData.h"

#ifndef UNIT_TESTING

#endif

asMethodCalibrator::asMethodCalibrator()
        : asMethodStandard(),
          m_scoreOrder(Asc),
          m_scoreValid(NaNf),
          m_validationMode(false)
{
    // Seeds the random generator
    asInitRandom();
}

asMethodCalibrator::~asMethodCalibrator()
{
    DeletePreloadedArchiveData();
}

bool asMethodCalibrator::Manager()
{
    // Set unresponsive to speedup
    g_responsive = false;

    // Reset the score of the climatology
    m_scoreClimatology.clear();

    // Seeds the random generator
    asInitRandom();

    // Load parameters
    asParametersCalibration params;
    if (!params.LoadFromFile(m_paramsFilePath)) {
        return false;
    }
    if (!m_predictandStationIds.empty()) {
        vvi idsVect;
        idsVect.push_back(m_predictandStationIds);
        params.SetPredictandStationIdsVector(idsVect);
    }
    params.InitValues();
    m_originalParams = params;

    // Load the Predictand DB
    if (!LoadPredictandDB(m_predictandDBFilePath)) {
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
#if wxUSE_GUI
        wxLogStatus(_("Calibration over."));
#endif
    } else {
        wxLogError(_("The parameters could not be calibrated"));
    }

    // Delete preloaded data and cleanup
    DeletePreloadedArchiveData();

    return true;
}

void asMethodCalibrator::LoadScoreOrder(asParametersCalibration &params)
{
    asScore *score = asScore::GetInstance(params.GetScoreName());
    Order scoreOrder = score->GetOrder();
    SetScoreOrder(scoreOrder);
    wxDELETE(score);
}

void asMethodCalibrator::ClearAll()
{
    m_parametersTemp.clear();
    m_scoresCalibTemp.clear();
    m_parameters.clear();
    m_scoresCalib.clear();
    m_scoreValid = NaNf;
}

void asMethodCalibrator::ClearTemp()
{
    m_parametersTemp.clear();
    m_scoresCalibTemp.clear();
}

bool asMethodCalibrator::PushBackBestTemp()
{
    if (!SortScoresAndParametersTemp()) {
        return false;
    }

    PushBackFirstTemp();

    return true;
}

void asMethodCalibrator::RemoveNaNsInTemp()
{
    wxASSERT(m_parametersTemp.size() == m_scoresCalibTemp.size());

    std::vector<asParametersCalibration> copyParametersTemp;
    vf copyScoresCalibTemp;

    for (unsigned int i = 0; i < m_scoresCalibTemp.size(); i++) {
        if (!asIsNaN(m_scoresCalibTemp[i])) {
            copyScoresCalibTemp.push_back(m_scoresCalibTemp[i]);
            copyParametersTemp.push_back(m_parametersTemp[i]);
        }
    }

    m_scoresCalibTemp = copyScoresCalibTemp;
    m_parametersTemp = copyParametersTemp;

    wxASSERT(m_parametersTemp.size() == m_scoresCalibTemp.size());
    wxASSERT(!m_parametersTemp.empty());
}

void asMethodCalibrator::KeepBestTemp()
{
    SortScoresAndParametersTemp();
    KeepFirstTemp();
}

void asMethodCalibrator::PushBackFirstTemp()
{
    wxASSERT(!m_parametersTemp.empty());
    wxASSERT(!m_scoresCalibTemp.empty());
    m_parameters.push_back(m_parametersTemp[0]);
    m_scoresCalib.push_back(m_scoresCalibTemp[0]);
}

void asMethodCalibrator::KeepFirstTemp()
{
    wxASSERT(!m_parameters.empty());
    wxASSERT(!m_parametersTemp.empty());
    wxASSERT(!m_scoresCalibTemp.empty());
    m_parameters[0] = m_parametersTemp[0];
    if (m_scoresCalib.empty()) {
        m_scoresCalib.push_back(m_scoresCalibTemp[0]);
    } else {
        m_scoresCalib[0] = m_scoresCalibTemp[0];
    }
}

bool asMethodCalibrator::SortScoresAndParametersTemp()
{
    wxASSERT(m_scoresCalibTemp.size() == m_parametersTemp.size());
    wxASSERT(!m_scoresCalibTemp.empty());
    wxASSERT(!m_parametersTemp.empty());

    if (m_parametersTemp.size() == 1)
        return true;

    // Sort according to the score
    a1f vIndices = a1f::LinSpaced(Eigen::Sequential, m_scoresCalibTemp.size(), 0, m_scoresCalibTemp.size() - 1);
    if (!asSortArrays(&m_scoresCalibTemp[0], &m_scoresCalibTemp[m_scoresCalibTemp.size() - 1], &vIndices[0],
                      &vIndices[m_scoresCalibTemp.size() - 1], m_scoreOrder)) {
        return false;
    }

    // Sort the parameters sets as the scores
    std::vector<asParametersCalibration> copyParameters;
    for (unsigned int i = 0; i < m_scoresCalibTemp.size(); i++) {
        copyParameters.push_back(m_parametersTemp[i]);
    }
    for (unsigned int i = 0; i < m_scoresCalibTemp.size(); i++) {
        int index = (int) vIndices(i);
        m_parametersTemp[i] = copyParameters[index];
    }

    return true;
}

bool asMethodCalibrator::PushBackInTempIfBetter(asParametersCalibration &params, asResultsTotalScore &scoreFinal)
{
    float thisScore = scoreFinal.GetScore();

    switch (m_scoreOrder) {
        case Asc:
            if (thisScore < m_scoresCalib[0]) {
                m_parametersTemp.push_back(params);
                m_scoresCalibTemp.push_back(thisScore);
                return true;
            }
            break;

        case Desc:
            if (thisScore > m_scoresCalib[0]) {
                m_parametersTemp.push_back(params);
                m_scoresCalibTemp.push_back(thisScore);
                return true;
            }
            break;

        default:
            asThrowException(_("The score order is not correctly defined."));
    }

    return false;
}

bool asMethodCalibrator::KeepIfBetter(asParametersCalibration &params, asResultsTotalScore &scoreFinal)
{
    float thisScore = scoreFinal.GetScore();

    switch (m_scoreOrder) {
        case Asc:
            if (thisScore < m_scoresCalib[0]) {
                wxASSERT(!m_parameters.empty());
                wxASSERT(!m_scoresCalib.empty());
                m_parameters[0] = params;
                m_scoresCalib[0] = thisScore;
                return true;
            }
            break;

        case Desc:
            if (thisScore > m_scoresCalib[0]) {
                wxASSERT(!m_parameters.empty());
                wxASSERT(!m_scoresCalib.empty());
                m_parameters[0] = params;
                m_scoresCalib[0] = thisScore;
                return true;
            }
            break;

        default:
            asThrowException(_("The score order is not correctly defined."));
    }

    return false;
}

bool asMethodCalibrator::SetSelectedParameters(asResultsParametersArray &results)
{
    // Extract selected parameters & best parameters
    for (unsigned int i = 0; i < m_parameters.size(); i++) {
        results.Add(m_parameters[i], m_scoresCalib[i], m_scoreValid);
    }

    return true;
}

bool asMethodCalibrator::SetBestParameters(asResultsParametersArray &results)
{
    wxASSERT(!m_parameters.empty());
    wxASSERT(!m_scoresCalib.empty());

    // Extract selected parameters & best parameters
    float bestScore = m_scoresCalib[0];
    int bestScoreRow = 0;

    for (unsigned int i = 0; i < m_parameters.size(); i++) {
        if (m_scoreOrder == Asc) {
            if (m_scoresCalib[i] < bestScore) {
                bestScore = m_scoresCalib[i];
                bestScoreRow = i;
            }
        } else {
            if (m_scoresCalib[i] > bestScore) {
                bestScore = m_scoresCalib[i];
                bestScoreRow = i;
            }
        }
    }

    if (bestScoreRow != 0) {
        // Re-validate
        SaveDetails(&m_parameters[bestScoreRow]);
        Validate(&m_parameters[bestScoreRow]);
    }

    results.Add(m_parameters[bestScoreRow], m_scoresCalib[bestScoreRow], m_scoreValid);

    return true;
}

wxString asMethodCalibrator::GetPredictandStationIdsList(vi &stationIds) const
{
    wxString id;

    if (stationIds.size() == 1) {
        id << stationIds[0];
    } else if (stationIds.size() > 10) {
        id << stationIds[0];
        id << '-';
        id << stationIds[stationIds.size() - 1];
    } else {
        for (int i = 0; i < (int) stationIds.size(); i++) {
            id << stationIds[i];
            if (i < (int) stationIds.size() - 1) {
                id << ",";
            }
        }
    }

    return id;
}

double asMethodCalibrator::GetTimeStartCalibration(asParametersScoring *params) const
{
    return params->GetCalibrationStart() + params->GetPredictorsStartDiff();
}

double asMethodCalibrator::GetTimeEndCalibration(asParametersScoring *params) const
{
    double timeEndCalibration = params->GetCalibrationEnd();
    timeEndCalibration = wxMin(timeEndCalibration, timeEndCalibration - params->GetTimeSpanDays());

    return timeEndCalibration;
}

double asMethodCalibrator::GetEffectiveArchiveDataStart(asParameters *params) const
{
    auto *paramsScoring = (asParametersScoring *) params;

    return wxMin(GetTimeStartCalibration(paramsScoring), GetTimeStartArchive(paramsScoring));
}

double asMethodCalibrator::GetEffectiveArchiveDataEnd(asParameters *params) const
{
    auto *paramsScoring = (asParametersScoring *) params;

    return wxMax(GetTimeEndCalibration(paramsScoring), GetTimeEndArchive(paramsScoring));
}

va1f asMethodCalibrator::GetClimatologyData(asParametersScoring *params)
{
    vi stationIds = params->GetPredictandStationIds();

    // Get start and end dates
    a1d predictandTime = m_predictandDB->GetTime();
    auto predictandTimeDays = float(params->GetPredictandTimeHours() / 24.0);
    double timeStart, timeEnd;
    timeStart = wxMax(predictandTime[0], params->GetCalibrationStart());
    timeStart = floor(timeStart) + predictandTimeDays;
    timeEnd = wxMin(predictandTime[predictandTime.size() - 1], params->GetCalibrationEnd());
    timeEnd = floor(timeEnd) + predictandTimeDays;

    if (predictandTime.size() < 1) {
        wxLogError(_("An unexpected error occurred."));
        return va1f(stationIds.size(), a1f(1));
    }

    // Check if data are effectively available for this period
    int indexPredictandTimeStart = asFindCeil(&predictandTime[0], &predictandTime[predictandTime.size() - 1], timeStart);
    int indexPredictandTimeEnd = asFindFloor(&predictandTime[0], &predictandTime[predictandTime.size() - 1], timeEnd);

    if (indexPredictandTimeStart < 0 || indexPredictandTimeEnd < 0) {
        wxLogError(_("An unexpected error occurred."));
        return va1f(stationIds.size(), a1f(1));
    }

    for (int iStat = 0; iStat < (int) stationIds.size(); iStat++) {
        a1f predictandDataNorm = m_predictandDB->GetDataNormalizedStation(stationIds[iStat]);

        while (asIsNaN(predictandDataNorm(indexPredictandTimeStart))) {
            indexPredictandTimeStart++;
        }
        while (asIsNaN(predictandDataNorm(indexPredictandTimeEnd))) {
            indexPredictandTimeEnd--;
            if (indexPredictandTimeEnd < 0) {
                wxLogError(_("An unexpected error occurred."));
                return va1f(stationIds.size(), a1f(1));
            }
        }
    }

    if (indexPredictandTimeStart < 0 || indexPredictandTimeEnd < 0) {
        wxLogError(_("An unexpected error occurred."));
        return va1f(stationIds.size(), a1f(1));
    }

    timeStart = predictandTime[indexPredictandTimeStart];
    timeStart = floor(timeStart) + predictandTimeDays;
    timeEnd = predictandTime[indexPredictandTimeEnd];
    timeEnd = floor(timeEnd) + predictandTimeDays;
    indexPredictandTimeStart = asFindCeil(&predictandTime[0], &predictandTime[predictandTime.size() - 1], timeStart);
    indexPredictandTimeEnd = asFindFloor(&predictandTime[0], &predictandTime[predictandTime.size() - 1], timeEnd);

    if (indexPredictandTimeStart < 0 || indexPredictandTimeEnd < 0) {
        wxLogError(_("An unexpected error occurred."));
        return va1f(stationIds.size(), a1f(1));
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
        a1f predictandDataNorm = m_predictandDB->GetDataNormalizedStation(stationIds[iStat]);

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

bool asMethodCalibrator::GetAnalogsDates(asResultsDates &results, asParametersScoring *params, int iStep,
                                         bool &containsNaNs)
{
    // Initialize the result object
    results.SetCurrentStep(iStep);
    results.Init(params);

    // Archive date array
    asTimeArray timeArrayArchive(GetTimeStartArchive(params), GetTimeEndArchive(params),
                                 params->GetAnalogsTimeStepHours(), params->GetTimeArrayAnalogsMode());
    if (params->HasValidationPeriod()) // remove validation years
    {
        timeArrayArchive.SetForbiddenYears(params->GetValidationYearsVector());
    }
    timeArrayArchive.Init();

    // Target date array
    asTimeArray timeArrayTarget(GetTimeStartCalibration(params), GetTimeEndCalibration(params),
                                params->GetTargetTimeStepHours(), params->GetTimeArrayTargetMode());

    // Remove validation years
    if (!m_validationMode && params->HasValidationPeriod())
    {
        timeArrayTarget.SetForbiddenYears(params->GetValidationYearsVector());
    }

    if (params->GetTimeArrayTargetMode().CmpNoCase("predictand_thresholds") == 0 ||
        params->GetTimeArrayTargetMode().CmpNoCase("PredictandThresholds") == 0) {
        vi stations = params->GetPredictandStationIds();
        if (stations.size() > 1) {
            wxLogError(_("You cannot use predictand thresholds with the multivariate approach."));
            return false;
        }

        if (!timeArrayTarget.Init(*m_predictandDB, params->GetTimeArrayTargetPredictandSerieName(), stations[0],
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
    if (m_validationMode) {
        timeArrayTarget.KeepOnlyYears(params->GetValidationYearsVector());
    }

    // Data date array
    double timeStartData = wxMin(GetTimeStartCalibration(params), GetTimeStartArchive(params));
    double timeEndData = wxMax(GetTimeEndCalibration(params), GetTimeEndArchive(params));
    asTimeArray timeArrayData(timeStartData, timeEndData, params->GetAnalogsTimeStepHours(),
                              params->GetTimeArrayTargetMode());
    timeArrayData.Init();

    // Check on the archive length
    if (timeArrayArchive.GetSize() < 100) {
        wxLogError(_("The time array is not consistent in asMethodCalibrator::GetAnalogsDates: size=%d."),
                   timeArrayArchive.GetSize());
        return false;
    }
    wxLogVerbose(_("Date arrays created."));

    // Load the predictor data
    std::vector<asPredictor *> predictors;
    if (!LoadArchiveData(predictors, params, iStep, timeStartData, timeEndData)) {
        wxLogError(_("Failed loading predictor data."));
        Cleanup(predictors);
        return false;
    }

    // Create the criterion
    std::vector<asCriteria *> criteria;
    for (int iPtor = 0; iPtor < params->GetPredictorsNb(iStep); iPtor++) {
        // Instantiate a score object
        asCriteria *criterion = asCriteria::GetInstance(params->GetPredictorCriteria(iStep, iPtor));
        if (criterion->NeedsDataRange()) {
            wxASSERT(predictors.size() > iPtor);
            wxASSERT(predictors[iPtor]);
            criterion->SetDataRange(predictors[iPtor]);
        }
        criteria.push_back(criterion);
    }

    // Check time sizes
#ifdef _DEBUG
    int prevTimeSize = 0;

    for (unsigned int i = 0; i < predictors.size(); i++) {
        if (i > 0) {
            wxASSERT(predictors[i]->GetTimeSize() == prevTimeSize);
        }
        prevTimeSize = predictors[i]->GetTimeSize();
    }
#endif // _DEBUG

    // Inline the data when possible
    for (int iPtor = 0; iPtor < (int) predictors.size(); iPtor++) {
        if (criteria[iPtor]->CanUseInline()) {
            predictors[iPtor]->Inline();
        }
    }

    // Send data and criteria to processor
    wxLogVerbose(_("Start processing the comparison."));

    if (!asProcessor::GetAnalogsDates(predictors, predictors, timeArrayData, timeArrayArchive, timeArrayData,
                                      timeArrayTarget, criteria, params, iStep, results, containsNaNs)) {
        wxLogError(_("Failed processing the analogs dates."));
        Cleanup(predictors);
        Cleanup(criteria);
        return false;
    }
    wxLogVerbose(_("The processing is over."));

    Cleanup(predictors);
    Cleanup(criteria);

    return true;
}

bool asMethodCalibrator::GetAnalogsSubDates(asResultsDates &results, asParametersScoring *params,
                                            asResultsDates &anaDates, int iStep, bool &containsNaNs)
{
    // Initialize the result object
    results.SetCurrentStep(iStep);
    results.Init(params);

    // Date array object instantiation for the processor
    wxLogVerbose(_("Creating a date arrays for the processor."));
    double timeStart = params->GetArchiveStart();
    double timeEnd = params->GetArchiveEnd();
    timeEnd = wxMin(timeEnd, timeEnd - params->GetTimeSpanDays()); // Adjust so the predictors search won't overtake the array
    asTimeArray timeArrayArchive(timeStart, timeEnd, params->GetAnalogsTimeStepHours(),
                                 params->GetTimeArrayTargetMode());
    timeArrayArchive.Init();
    wxLogVerbose(_("Date arrays created."));

    // Load the predictor data
    std::vector<asPredictor *> predictors;
    if (!LoadArchiveData(predictors, params, iStep, timeStart, timeEnd)) {
        wxLogError(_("Failed loading predictor data."));
        Cleanup(predictors);
        return false;
    }

    // Create the score objects
    std::vector<asCriteria *> criteria;
    for (int iPtor = 0; iPtor < params->GetPredictorsNb(iStep); iPtor++) {
        wxLogVerbose(_("Creating a criterion object."));
        asCriteria *criterion = asCriteria::GetInstance(params->GetPredictorCriteria(iStep, iPtor));
        if (criterion->NeedsDataRange()) {
            wxASSERT(predictors.size() > iPtor);
            wxASSERT(predictors[iPtor]);
            criterion->SetDataRange(predictors[iPtor]);
        }
        criteria.push_back(criterion);
        wxLogVerbose(_("Criterion object created."));
    }

    // Inline the data when possible
    for (int iPtor = 0; iPtor < (int) predictors.size(); iPtor++) {
        if (criteria[iPtor]->CanUseInline()) {
            predictors[iPtor]->Inline();
        }
    }

    // Send data and criteria to processor
    wxLogVerbose(_("Start processing the comparison."));
    if (!asProcessor::GetAnalogsSubDates(predictors, predictors, timeArrayArchive, timeArrayArchive, anaDates, criteria,
                                         params, iStep, results, containsNaNs)) {
        wxLogError(_("Failed processing the analogs dates."));
        Cleanup(predictors);
        Cleanup(criteria);
        return false;
    }
    wxLogVerbose(_("The processing is over."));

    Cleanup(predictors);
    Cleanup(criteria);

    return true;
}

bool asMethodCalibrator::GetAnalogsValues(asResultsValues &results, asParametersScoring *params,
                                          asResultsDates &anaDates, int iStep)
{
    // Initialize the result object
    results.SetCurrentStep(iStep);
    results.Init(params);

    // Set the predictand values to the corresponding analog dates
    wxASSERT(m_predictandDB);
    wxLogVerbose(_("Start setting the predictand values to the corresponding analog dates."));
    if (!asProcessor::GetAnalogsValues(*m_predictandDB, anaDates, params, results)) {
        wxLogError(_("Failed setting the predictand values to the corresponding analog dates."));
        return false;
    }
    wxLogVerbose(_("Predictand association over."));

    return true;
}

bool asMethodCalibrator::GetAnalogsScores(asResultsScores &results, asParametersScoring *params,
                                          asResultsValues &anaValues, int iStep)
{
    // Initialize the result object
    results.SetCurrentStep(iStep);
    results.Init(params);

    // Instantiate a score object
    wxLogVerbose(_("Instantiating a score object"));
    asScore *score = asScore::GetInstance(params->GetScoreName());
    score->SetQuantile(params->GetScoreQuantile());
    score->SetThreshold(params->GetScoreThreshold());

    if (score->UsesClimatology() && m_scoreClimatology.empty()) {
        wxLogVerbose(_("Processing the score of the climatology."));

        va1f climatologyData = GetClimatologyData(params);
        vi stationIds = params->GetPredictandStationIds();
        m_scoreClimatology.resize(stationIds.size());

        for (int iStat = 0; iStat < (int) stationIds.size(); iStat++) {
            score->ProcessScoreClimatology(anaValues.GetTargetValues()[iStat], climatologyData[iStat]);
            m_scoreClimatology[iStat] = score->GetScoreClimatology();
        }
    }

    // Pass data and score to processor
    wxLogVerbose(_("Start processing the score."));

    if (!asProcessorScore::GetAnalogsScores(anaValues, score, params, results, m_scoreClimatology)) {
        wxLogError(_("Failed processing the score."));
        wxDELETE(score);
        return false;
    }

    wxDELETE(score);

    return true;
}

bool asMethodCalibrator::GetAnalogsTotalScore(asResultsTotalScore &results, asParametersScoring *params,
                                              asResultsScores &anaScores, int iStep)
{
    // Initialize the result object
    results.SetCurrentStep(iStep);
    results.Init();

    // Date array object instantiation for the final score
    wxLogVerbose(_("Creating a date array for the final score."));
    double timeStart = params->GetCalibrationStart();
    double timeEnd = params->GetCalibrationEnd() + 1;
    while (timeEnd > params->GetCalibrationEnd() + 0.999) {
        timeEnd -= params->GetTargetTimeStepHours() / 24.0;
    }
    asTimeArray timeArray(timeStart, timeEnd, params->GetTargetTimeStepHours(), params->GetScoreTimeArrayMode());

    // TODO: Add every options for the Init function (generic version)
    //    timeArray.Init(params->GetScoreTimeArrayDate(), params->GetForecastScoreTimeArrayIntervalDays());
    timeArray.Init();
    wxLogVerbose(_("Date array created."));

    // Pass data and score to processor
    wxLogVerbose(_("Start processing the final score."));
    if (!asProcessorScore::GetAnalogsTotalScore(anaScores, timeArray, params, results)) {
        wxLogError(_("Failed to process the final score."));
        return false;
    }
    wxLogVerbose(_("Processing over."));

    return true;
}

bool asMethodCalibrator::SubProcessAnalogsNumber(asParametersCalibration &params, asResultsDates &anaDatesPrevious,
                                                 int iStep)
{
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
        if (!GetAnalogsDates(anaDates, &params, iStep, containsNaNs))
            return false;
    } else {
        if (!GetAnalogsSubDates(anaDates, &params, anaDatesPrevious, iStep, containsNaNs))
            return false;
    }
    if (containsNaNs) {
        wxLogError(_("The dates selection contains NaNs"));
        return false;
    }

    asResultsDates anaDatesTmp(anaDates);
    a2f dates = anaDates.GetAnalogsDates();

    // If at the end of the chain
    if (iStep == params.GetStepsNb() - 1) {

        if (!GetAnalogsValues(anaValues, &params, anaDates, iStep))
            return false;

        asResultsScores anaScores;
        asResultsTotalScore anaScoreFinal;

        for (int i = 0; i <= rowEnd; i++) {
            params.SetAnalogsNumber(iStep, analogsNbVect[i]);

            // Fixes and checks
            params.FixAnalogsNb();

            // Extract analogs dates from former results
            a2f subDates = dates.leftCols(params.GetAnalogsNumber(iStep));
            anaDatesTmp.SetAnalogsDates(subDates);

            if (!GetAnalogsScores(anaScores, &params, anaValues, iStep))
                return false;
            if (!GetAnalogsTotalScore(anaScoreFinal, &params, anaScores, iStep))
                return false;

            m_parametersTemp.push_back(params);
            m_scoresCalibTemp.push_back(anaScoreFinal.GetScore());
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
            if (!SubProcessAnalogsNumber(params, anaDatesTmp, iStep + 1))
                return false;
        }
    }

    return true;
}

bool asMethodCalibrator::SaveDetails(asParametersCalibration *params)
{
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
            if (!GetAnalogsDates(anaDates, params, iStep, containsNaNs))
                return false;
        } else {
            anaDatesPrevious = anaDates;
            if (!GetAnalogsSubDates(anaDates, params, anaDatesPrevious, iStep, containsNaNs))
                return false;
        }
        if (containsNaNs) {
            wxLogError(_("The dates selection contains NaNs"));
            return false;
        }
    }
    if (!GetAnalogsValues(anaValues, params, anaDates, stepsNb - 1))
        return false;
    if (!GetAnalogsScores(anaScores, params, anaValues, stepsNb - 1))
        return false;
    if (!GetAnalogsTotalScore(anaScoreFinal, params, anaScores, stepsNb - 1))
        return false;

    anaDates.SetSubFolder("calibration");
    anaDates.Save();
    anaValues.SetSubFolder("calibration");
    anaValues.Save();
    anaScores.SetSubFolder("calibration");
    anaScores.Save();

    return true;
}

bool asMethodCalibrator::Validate(asParametersCalibration *params)
{
    bool skipValidation = false;
    wxFileConfig::Get()->Read("/Optimizer/SkipValidation", &skipValidation, false);

    if (skipValidation) {
        return true;
    }

    if (!params->HasValidationPeriod()) {
        wxLogWarning("The parameters have no validation period !");
        return true;
    }

    m_validationMode = true;

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
            if (!GetAnalogsDates(anaDates, params, iStep, containsNaNs))
                return false;
        } else {
            anaDatesPrevious = anaDates;
            if (!GetAnalogsSubDates(anaDates, params, anaDatesPrevious, iStep, containsNaNs))
                return false;
        }
        if (containsNaNs) {
            wxLogError(_("The dates selection contains NaNs"));
            return false;
        }
    }
    if (!GetAnalogsValues(anaValues, params, anaDates, stepsNb - 1))
        return false;
    if (!GetAnalogsScores(anaScores, params, anaValues, stepsNb - 1))
        return false;
    if (!GetAnalogsTotalScore(anaScoreFinal, params, anaScores, stepsNb - 1))
        return false;

    anaDates.SetSubFolder("validation");
    anaDates.Save();
    anaValues.SetSubFolder("validation");
    anaValues.Save();
    anaScores.SetSubFolder("validation");
    anaScores.Save();

    m_scoreValid = anaScoreFinal.GetScore();

    m_validationMode = false;

    return true;
}
