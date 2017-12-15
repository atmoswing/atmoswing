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
#include "asThreadPreloadData.h"

#ifndef UNIT_TESTING

#endif

asMethodCalibrator::asMethodCalibrator()
        : asMethodStandard(),
          m_scoreOrder(Asc),
          m_scoreValid(NaNf),
          m_preloaded(false),
          m_validationMode(false)
{
    // Seeds the random generator
    asTools::InitRandom();
}

asMethodCalibrator::~asMethodCalibrator()
{
    DeletePreloadedData();
}

bool asMethodCalibrator::Manager()
{
    // Set unresponsive to speedup
    g_responsive = false;

    // Reset the score of the climatology
    m_scoreClimatology.clear();

    // Seeds the random generator
    asTools::InitRandom();

    // Load parameters
    asParametersCalibration params;
    if (!params.LoadFromFile(m_paramsFilePath)) {
        return false;
    }
    if (m_predictandStationIds.size() > 0) {
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
    DeletePreloadedData();

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

void asMethodCalibrator::PushBackBestTemp()
{
    SortScoresAndParametersTemp();
    PushBackFirstTemp();
}

void asMethodCalibrator::RemoveNaNsInTemp()
{
    wxASSERT(m_parametersTemp.size() == m_scoresCalibTemp.size());

    std::vector<asParametersCalibration> copyParametersTemp;
    vf copyScoresCalibTemp;

    for (unsigned int i = 0; i < m_scoresCalibTemp.size(); i++) {
        if (!asTools::IsNaN(m_scoresCalibTemp[i])) {
            copyScoresCalibTemp.push_back(m_scoresCalibTemp[i]);
            copyParametersTemp.push_back(m_parametersTemp[i]);
        }
    }

    m_scoresCalibTemp = copyScoresCalibTemp;
    m_parametersTemp = copyParametersTemp;

    wxASSERT(m_parametersTemp.size() == m_scoresCalibTemp.size());
    wxASSERT(m_parametersTemp.size() > 0);
}

void asMethodCalibrator::KeepBestTemp()
{
    SortScoresAndParametersTemp();
    KeepFirstTemp();
}

void asMethodCalibrator::PushBackFirstTemp()
{
    m_parameters.push_back(m_parametersTemp[0]);
    m_scoresCalib.push_back(m_scoresCalibTemp[0]);
}

void asMethodCalibrator::KeepFirstTemp()
{
    wxASSERT(m_parameters.size() > 0);
    wxASSERT(m_parametersTemp.size() > 0);
    wxASSERT(m_scoresCalibTemp.size() > 0);
    m_parameters[0] = m_parametersTemp[0];
    if (m_scoresCalib.size() == 0) {
        m_scoresCalib.push_back(m_scoresCalibTemp[0]);
    } else {
        m_scoresCalib[0] = m_scoresCalibTemp[0];
    }
}

void asMethodCalibrator::SortScoresAndParametersTemp()
{
    wxASSERT(m_scoresCalibTemp.size() == m_parametersTemp.size());
    wxASSERT(m_scoresCalibTemp.size() > 0);
    wxASSERT(m_parametersTemp.size() > 0);

    if (m_parametersTemp.size() == 1)
        return;

    // Sort according to the score
    a1f vIndices = a1f::LinSpaced(Eigen::Sequential, m_scoresCalibTemp.size(), 0, m_scoresCalibTemp.size() - 1);
    asTools::SortArrays(&m_scoresCalibTemp[0], &m_scoresCalibTemp[m_scoresCalibTemp.size() - 1], &vIndices[0],
                        &vIndices[m_scoresCalibTemp.size() - 1], m_scoreOrder);

    // Sort the parameters sets as the scores
    std::vector<asParametersCalibration> copyParameters;
    for (unsigned int i = 0; i < m_scoresCalibTemp.size(); i++) {
        copyParameters.push_back(m_parametersTemp[i]);
    }
    for (unsigned int i = 0; i < m_scoresCalibTemp.size(); i++) {
        int index = (int) vIndices(i);
        m_parametersTemp[i] = copyParameters[index];
    }
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
                wxASSERT(m_parameters.size() > 0);
                wxASSERT(m_scoresCalib.size() > 0);
                m_parameters[0] = params;
                m_scoresCalib[0] = thisScore;
                return true;
            }
            break;

        case Desc:
            if (thisScore > m_scoresCalib[0]) {
                wxASSERT(m_parameters.size() > 0);
                wxASSERT(m_scoresCalib.size() > 0);
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
    wxASSERT(m_parameters.size() > 0);
    wxASSERT(m_scoresCalib.size() > 0);

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
        SaveDetails(m_parameters[bestScoreRow]);
        Validate(m_parameters[bestScoreRow]);
    }

    results.Add(m_parameters[bestScoreRow], m_scoresCalib[bestScoreRow], m_scoreValid);

    return true;
}

wxString asMethodCalibrator::GetPredictandStationIdsList(vi &stationIds) const
{
    wxString id;

    if (stationIds.size() == 1) {
        id << stationIds[0];
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

bool asMethodCalibrator::PreloadData(asParametersScoring &params)
{
    if (!m_preloaded) {
        // Set preload to true here, so cleanup is made in case of exceptions.
        m_preloaded = true;

        InitializePreloadedDataContainer(params);

        if (!ProceedToDataPreloading(params))
            return false;

        if (!CheckDataIsPreloaded(params))
            return false;
    }

    return true;
}

bool asMethodCalibrator::ProceedToDataPreloading(asParametersScoring &params)
{
    bool parallelDataLoad = false;
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Read("/General/ParallelDataLoad", &parallelDataLoad, true);
    ThreadsManager().CritSectionConfig().Leave();

    if (parallelDataLoad) {
        wxLogVerbose(_("Preloading data with threads."));
    }

    for (int iStep = 0; iStep < params.GetStepsNb(); iStep++) {
        for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
            if (params.NeedsPreloading(iStep, iPtor)) {

                if (params.NeedsPreprocessing(iStep, iPtor)) {
                    if (PointersShared(params, iStep, iPtor, 0)) {
                        continue;
                    }
                    if (!PreloadDataWithPreprocessing(params, iStep, iPtor)) {
                        return false;
                    }
                } else {
                    for (int iPre = 0; iPre < params.GetPredictorDataIdVector(iStep, iPtor).size(); iPre++) {
                        if (PointersShared(params, iStep, iPtor, iPre)) {
                            continue;
                        }
                        if (parallelDataLoad) {
                            asThreadPreloadData *thread = new asThreadPreloadData(this, params, iStep, iPtor, iPre);
                            if (!ThreadsManager().HasFreeThread(thread->GetType())) {
                                ThreadsManager().WaitForFreeThread(thread->GetType());
                            }
                            ThreadsManager().AddThread(thread);
                        } else {
                            if (!PreloadDataWithoutPreprocessing(params, iStep, iPtor, iPre)) {
                                return false;
                            }
                        }
                    }
                }

                if (parallelDataLoad) {
                    // Wait until all done in order to have non null pointers to copy.
                    ThreadsManager().Wait(asThread::PreloadData);
                }
            }

            if (parallelDataLoad) {
                // Wait until all done in order to have non null pointers to copy.
                ThreadsManager().Wait(asThread::PreloadData);
            }
        }
    }

    if (parallelDataLoad) {
        // Wait until all done
        ThreadsManager().Wait(asThread::PreloadData);
        wxLogVerbose(_("Data preloaded with threads."));
    }

    return true;
}

bool asMethodCalibrator::CheckDataIsPreloaded(const asParametersScoring &params) const
{
    for (int iStep = 0; iStep < params.GetStepsNb(); iStep++) {
        for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
            if (params.NeedsPreloading(iStep, iPtor)) {
                if (!params.NeedsPreprocessing(iStep, iPtor)) {
                    for (int iPre = 0; iPre < params.GetPredictorDataIdVector(iStep, iPtor).size(); iPre++) {
                        if (!HasPreloadedData(iStep, iPtor, iPre)) {
                            wxLogError(_("No data was preloaded for step %d, predictor %d and variable '%s' (num %d)."),
                                       iStep, iPtor, params.GetPredictorDataIdVector(iStep, iPtor)[iPre], iPre);
                            return false;
                        }
                    }
                }
                if (!HasPreloadedData(iStep, iPtor)) {
                    wxLogError(_("No data was preloaded for step %d and predictor %d."), iStep, iPtor);
                    return false;
                }
            }
        }
    }

    return true;
}

bool asMethodCalibrator::HasPreloadedData(int iStep, int iPtor) const
{
    for (int iPre = 0; iPre < m_preloadedArchive[iStep][iPtor].size(); iPre++) {
        for (int iLevel = 0; iLevel < m_preloadedArchive[iStep][iPtor][iPre].size(); iLevel++) {
            for (int iHour = 0; iHour < m_preloadedArchive[iStep][iPtor][iPre][iLevel].size(); iHour++) {
                if (m_preloadedArchive[iStep][iPtor][iPre][iLevel][iHour] != NULL) {
                    return true;
                }
            }
        }
    }

    return false;
}

bool asMethodCalibrator::HasPreloadedData(int iStep, int iPtor, int iPre) const
{
    for (int iLevel = 0; iLevel < m_preloadedArchive[iStep][iPtor][iPre].size(); iLevel++) {
        for (int iHour = 0; iHour < m_preloadedArchive[iStep][iPtor][iPre][iLevel].size(); iHour++) {
            if (m_preloadedArchive[iStep][iPtor][iPre][iLevel][iHour] != NULL) {
                return true;
            }
        }
    }

    return false;
}

void asMethodCalibrator::InitializePreloadedDataContainer(asParametersScoring &params)
{
    if (m_preloadedArchive.size() == 0) {
        m_preloadedArchive.resize((unsigned long) params.GetStepsNb());
        m_preloadedArchivePointerCopy.resize((unsigned long) params.GetStepsNb());
        for (int iStep = 0; iStep < params.GetStepsNb(); iStep++) {
            m_preloadedArchive[iStep].resize((unsigned long) params.GetPredictorsNb(iStep));
            m_preloadedArchivePointerCopy[iStep].resize((unsigned long) params.GetPredictorsNb(iStep));

            for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {

                vwxs preloadDataIds = params.GetPreloadDataIds(iStep, iPtor);
                vf preloadLevels = params.GetPreloadLevels(iStep, iPtor);
                vd preloadTimeHours = params.GetPreloadTimeHours(iStep, iPtor);

                unsigned long preloadDataIdsSize = wxMax(preloadDataIds.size(), 1);
                unsigned long preloadLevelsSize = wxMax(preloadLevels.size(), 1);
                unsigned long preloadTimeHoursSize = wxMax(preloadTimeHours.size(), 1);

                m_preloadedArchivePointerCopy[iStep][iPtor].resize(preloadDataIdsSize);
                m_preloadedArchive[iStep][iPtor].resize(preloadDataIdsSize);

                for (int iPre = 0; iPre < preloadDataIdsSize; iPre++) {
                    m_preloadedArchivePointerCopy[iStep][iPtor][iPre] = false;
                    m_preloadedArchive[iStep][iPtor][iPre].resize(preloadLevelsSize);

                    // Load data for every level and every hour
                    for (int iLevel = 0; iLevel < preloadLevelsSize; iLevel++) {
                        m_preloadedArchive[iStep][iPtor][iPre][iLevel].resize(preloadTimeHoursSize);
                        for (int iHour = 0; iHour < preloadTimeHoursSize; iHour++) {
                            m_preloadedArchive[iStep][iPtor][iPre][iLevel][iHour] = NULL;
                        }
                    }
                }
            }
        }
    }
}

double asMethodCalibrator::GetTimeStartCalibration(asParametersScoring &params) const
{
    double timeStartCalibration = params.GetCalibrationStart();
    timeStartCalibration += std::abs(params.GetTimeShiftDays());

    return timeStartCalibration;
}

double asMethodCalibrator::GetTimeStartArchive(asParametersScoring &params) const
{
    double timeStartArchive = params.GetArchiveStart();
    timeStartArchive += std::abs(params.GetTimeShiftDays());

    return timeStartArchive;
}

double asMethodCalibrator::GetTimeEndCalibration(asParametersScoring &params) const
{
    double timeEndCalibration = params.GetCalibrationEnd();
    timeEndCalibration = wxMin(timeEndCalibration, timeEndCalibration - params.GetTimeSpanDays());

    return timeEndCalibration;
}

double asMethodCalibrator::GetTimeEndArchive(asParametersScoring &params) const
{
    double timeEndArchive = params.GetArchiveEnd();
    timeEndArchive = wxMin(timeEndArchive, timeEndArchive - params.GetTimeSpanDays());

    return timeEndArchive;
}

bool asMethodCalibrator::PointersShared(asParametersScoring &params, int iStep, int iPtor, int iPre)
{
    if (iStep == 0 && iPtor == 0) {
        return false;
    }

    int prev_step = 0, prev_ptor = 0, prev_dat = 0;
    bool share = false;

    for (prev_step = 0; prev_step <= iStep; prev_step++) {

        int ptor_max = params.GetPredictorsNb(prev_step);
        if (prev_step == iStep) {
            ptor_max = iPtor;
        }

        for (prev_ptor = 0; prev_ptor < ptor_max; prev_ptor++) {
            share = true;

            if (!params.NeedsPreprocessing(iStep, iPtor)) {
                if (!params.GetPredictorDatasetId(iStep, iPtor).IsSameAs(
                        params.GetPredictorDatasetId(prev_step, prev_ptor), false))
                    share = false;
                if (!params.GetPredictorGridType(iStep, iPtor).IsSameAs(
                        params.GetPredictorGridType(prev_step, prev_ptor), false))
                    share = false;
            } else {
                if (!params.GetPreprocessMethod(iStep, iPtor).IsSameAs(params.GetPreprocessMethod(prev_step, prev_ptor),
                                                                       false))
                    share = false;
                if (params.GetPreprocessSize(iStep, iPtor) != params.GetPreprocessSize(prev_step, prev_ptor)) {
                    share = false;
                } else {
                    int preprocessSize = params.GetPreprocessSize(iStep, iPtor);

                    for (int iPre = 0; iPre < preprocessSize; iPre++) {
                        if (!params.GetPreprocessDatasetId(iStep, iPtor, iPre).IsSameAs(
                                params.GetPreprocessDatasetId(prev_step, prev_ptor, iPre), false))
                            share = false;
                        if (!params.GetPreprocessDataId(iStep, iPtor, iPre).IsSameAs(
                                params.GetPreprocessDataId(prev_step, prev_ptor, iPre), false))
                            share = false;
                    }
                }
            }

            if (params.GetPreloadXmin(iStep, iPtor) != params.GetPreloadXmin(prev_step, prev_ptor))
                share = false;
            if (params.GetPreloadXptsnb(iStep, iPtor) != params.GetPreloadXptsnb(prev_step, prev_ptor))
                share = false;
            if (params.GetPredictorXstep(iStep, iPtor) != params.GetPredictorXstep(prev_step, prev_ptor))
                share = false;
            if (params.GetPreloadYmin(iStep, iPtor) != params.GetPreloadYmin(prev_step, prev_ptor))
                share = false;
            if (params.GetPreloadYptsnb(iStep, iPtor) != params.GetPreloadYptsnb(prev_step, prev_ptor))
                share = false;
            if (params.GetPredictorYstep(iStep, iPtor) != params.GetPredictorYstep(prev_step, prev_ptor))
                share = false;
            if (params.GetPredictorFlatAllowed(iStep, iPtor) != params.GetPredictorFlatAllowed(prev_step, prev_ptor))
                share = false;

            vf levels1 = params.GetPreloadLevels(iStep, iPtor);
            vf levels2 = params.GetPreloadLevels(prev_step, prev_ptor);
            if (levels1.size() != levels2.size()) {
                share = false;
            } else {
                for (unsigned int i = 0; i < levels1.size(); i++) {
                    if (levels1[i] != levels2[i])
                        share = false;
                }
            }

            vd hours1 = params.GetPreloadTimeHours(iStep, iPtor);
            vd hours2 = params.GetPreloadTimeHours(prev_step, prev_ptor);
            if (hours1.size() != hours2.size()) {
                share = false;
            } else {
                for (unsigned int i = 0; i < hours1.size(); i++) {
                    if (hours1[i] != hours2[i])
                        share = false;
                }
            }

            bool dataIdFound = false;
            vwxs preloadDataIds = params.GetPreloadDataIds(iStep, iPtor);
            vwxs preloadDataIdsPrev = params.GetPreloadDataIds(prev_step, prev_ptor);
            for (unsigned int i = 0; i < preloadDataIdsPrev.size(); i++) {
                // Vector can be empty in case of preprocessing
                if (preloadDataIds.size() > iPre && preloadDataIdsPrev.size() > i) {
                    wxASSERT(!preloadDataIds[iPre].IsEmpty());
                    wxASSERT(!preloadDataIdsPrev[i].IsEmpty());
                    if (preloadDataIds[iPre].IsSameAs(preloadDataIdsPrev[i])) {
                        prev_dat = i;
                        dataIdFound = true;
                    }
                }
            }
            if (!dataIdFound) {
                share = false;
            }

            if (share)
                break;
        }

        if (share)
            break;
    }

    if (share) {
        wxLogVerbose(_("Share data pointer"));

        vf preloadLevels = params.GetPreloadLevels(iStep, iPtor);
        vd preloadTimeHours = params.GetPreloadTimeHours(iStep, iPtor);
        wxASSERT(preloadLevels.size() > 0);
        wxASSERT(preloadTimeHours.size() > 0);

        m_preloadedArchivePointerCopy[iStep][iPtor][iPre] = true;

        wxASSERT(m_preloadedArchive[prev_step].size() > (unsigned) prev_ptor);
        wxASSERT(m_preloadedArchive[prev_step][prev_ptor].size() > (unsigned) prev_dat);
        wxASSERT(m_preloadedArchive[prev_step][prev_ptor][prev_dat].size() == preloadLevels.size());

        // Load data for every level and every hour
        for (unsigned int iLevel = 0; iLevel < preloadLevels.size(); iLevel++) {
            wxASSERT(m_preloadedArchive[prev_step][prev_ptor][prev_dat][iLevel].size() == preloadTimeHours.size());
            for (unsigned int iHour = 0; iHour < preloadTimeHours.size(); iHour++) {
                // Copy pointer
                m_preloadedArchive[iStep][iPtor][iPre][iLevel][iHour] = m_preloadedArchive[prev_step][prev_ptor][prev_dat][iLevel][iHour];
            }
        }

        params.SetPreloadYptsnb(iStep, iPtor, params.GetPreloadYptsnb(prev_step, prev_ptor));

        return true;
    }

    return false;
}

bool asMethodCalibrator::PreloadDataWithoutPreprocessing(asParametersScoring &params, int iStep, int iPtor, int iPre)
{
    wxLogVerbose(_("Preloading data for predictor %d of step %d."), iPtor, iStep);

    double timeStartData = wxMin(GetTimeStartCalibration(params), GetTimeStartArchive(params));
    double timeEndData = wxMax(GetTimeEndCalibration(params), GetTimeEndArchive(params));

    vwxs preloadDataIds = params.GetPreloadDataIds(iStep, iPtor);
    vf preloadLevels = params.GetPreloadLevels(iStep, iPtor);
    vd preloadTimeHours = params.GetPreloadTimeHours(iStep, iPtor);
    wxASSERT(preloadDataIds.size() > iPre);
    wxASSERT(preloadLevels.size() > 0);
    wxASSERT(preloadTimeHours.size() > 0);

    // Load data for every level and every hour
    for (unsigned int iLevel = 0; iLevel < preloadLevels.size(); iLevel++) {
        for (unsigned int iHour = 0; iHour < preloadTimeHours.size(); iHour++) {
            // Loading the dataset information
            asPredictorArch *predictor = asPredictorArch::GetInstance(params.GetPredictorDatasetId(iStep, iPtor),
                                                                      preloadDataIds[iPre], m_predictorDataDir);
            if (!predictor) {
                return false;
            }

            // Select the number of members for ensemble data.
            if (predictor->IsEnsemble()) {
                predictor->SelectMembers(params.GetPredictorMembersNb(iStep, iPtor));
            }

            // Date array object instantiation for the data loading.
            // The array has the same length than timeArrayArchive, and the predictor dates are aligned with the
            // target dates, but the dates are not the same.
            double ptorStart = timeStartData - double(params.GetTimeShiftDays()) + preloadTimeHours[iHour] / 24.0;

            wxLogDebug("%f - %f + %f = %f", timeStartData, double(params.GetTimeShiftDays()),
                       preloadTimeHours[iHour] / 24.0, ptorStart);
            wxLogDebug("ptorStart = %s", asTime::GetStringTime(ptorStart));
            wxLogDebug("timeStartData = %s", asTime::GetStringTime(timeStartData));
            wxLogDebug("params.GetTimeShiftDays() = %f", double(params.GetTimeShiftDays()));
            wxLogDebug("preloadTimeHours[iHour]/24.0 = %f", preloadTimeHours[iHour] / 24.0);

            double ptorEnd = timeEndData - double(params.GetTimeShiftDays()) + preloadTimeHours[iHour] / 24.0;

            asTimeArray timeArray(ptorStart, ptorEnd, params.GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
            timeArray.Init();

            asGeo geo;
            double yMax = params.GetPreloadYmin(iStep, iPtor) +
                          params.GetPredictorYstep(iStep, iPtor) * (double) (params.GetPreloadYptsnb(iStep, iPtor) - 1);
            if (yMax > geo.GetAxisYmax()) {
                double diff = yMax - geo.GetAxisYmax();
                int removePts = (int) asTools::Round(diff / params.GetPredictorYstep(iStep, iPtor));
                params.SetPreloadYptsnb(iStep, iPtor, params.GetPreloadYptsnb(iStep, iPtor) - removePts);
                wxLogVerbose(_("Adapt Y axis extent according to the maximum allowed (from %.3f to %.3f)."), yMax,
                             yMax - diff);
                wxLogVerbose(_("Remove %d points (%.3f-%.3f)/%.3f."), removePts, yMax, geo.GetAxisYmax(),
                             params.GetPredictorYstep(iStep, iPtor));
            }

            wxASSERT(params.GetPreloadXptsnb(iStep, iPtor) > 0);
            wxASSERT(params.GetPreloadYptsnb(iStep, iPtor) > 0);

            // Area object instantiation
            asGeoAreaCompositeGrid *area = asGeoAreaCompositeGrid::GetInstance(
                    params.GetPredictorGridType(iStep, iPtor), params.GetPreloadXmin(iStep, iPtor),
                    params.GetPreloadXptsnb(iStep, iPtor), params.GetPredictorXstep(iStep, iPtor),
                    params.GetPreloadYmin(iStep, iPtor), params.GetPreloadYptsnb(iStep, iPtor),
                    params.GetPredictorYstep(iStep, iPtor), preloadLevels[iLevel], asNONE,
                    params.GetPredictorFlatAllowed(iStep, iPtor));
            wxASSERT(area);

            // Check the starting dates coherence
            if (predictor->GetOriginalProviderStart() > ptorStart) {
                wxLogError(_("The first year defined in the parameters (%s) is prior to the start date of the data (%s) (in asMethodCalibrator::PreloadData)."),
                           asTime::GetStringTime(ptorStart),
                           asTime::GetStringTime(predictor->GetOriginalProviderStart()));
                wxDELETE(area);
                wxDELETE(predictor);
                return false;
            }

            // Data loading
            wxLogVerbose(_("Loading %s data for level %d, %gh."), preloadDataIds[iPre], (int) preloadLevels[iLevel],
                         preloadTimeHours[iHour]);
            try {
                if (!predictor->Load(area, timeArray)) {
                    wxLogWarning(_("The data (%s for level %d, at %gh) could not be loaded."), preloadDataIds[iPre],
                                 (int) preloadLevels[iLevel], preloadTimeHours[iHour]);
                    wxDELETE(area);
                    wxDELETE(predictor);
                    continue; // The requested data can be missing (e.g. level not available).
                }
            } catch (std::bad_alloc &ba) {
                wxString msg(ba.what(), wxConvUTF8);
                wxLogError(_("Bad allocation in the data preloading: %s"), msg);
                wxDELETE(area);
                wxDELETE(predictor);
                return false;
            } catch (std::exception &e) {
                wxString msg(e.what(), wxConvUTF8);
                wxLogError(_("Exception in the data preloading: %s"), msg);
                wxDELETE(area);
                wxDELETE(predictor);
                return false;
            }
            wxLogVerbose(_("Data loaded."));
            wxDELETE(area);

            m_preloadedArchive[iStep][iPtor][iPre][iLevel][iHour] = predictor;
        }
    }

    return true;
}

bool asMethodCalibrator::PreloadDataWithPreprocessing(asParametersScoring &params, int iStep, int iPtor)
{
    wxLogVerbose(_("Preloading data for predictor preprocessed %d of step %d."), iPtor, iStep);

    double timeStartData = wxMin(GetTimeStartCalibration(params), GetTimeStartArchive(params));
    double timeEndData = wxMax(GetTimeEndCalibration(params), GetTimeEndArchive(params));

    // Check the preprocessing method
    wxString method = params.GetPreprocessMethod(iStep, iPtor);

    // Get the number of sub predictors
    int preprocessSize = params.GetPreprocessSize(iStep, iPtor);

    // Levels and time arrays
    vf preloadLevels = params.GetPreloadLevels(iStep, iPtor);
    vd preloadTimeHours = params.GetPreloadTimeHours(iStep, iPtor);

    // Check on which variable to loop
    unsigned long preloadLevelsSize = preloadLevels.size();
    unsigned long preloadTimeHoursSize = preloadTimeHours.size();
    bool loopOnLevels = true;
    bool loopOnTimeHours = true;

    if (method.IsSameAs("Multiplication") || method.IsSameAs("Multiply") || method.IsSameAs("Addition") ||
        method.IsSameAs("Average")) {
        loopOnLevels = false;
        loopOnTimeHours = false;
        preloadLevelsSize = 1;
        preloadTimeHoursSize = 1;
    } else if (method.IsSameAs("Gradients") || method.IsSameAs("HumidityIndex") || method.IsSameAs("HumidityFlux") ||
               method.IsSameAs("FormerHumidityIndex")) {
        if (preloadLevelsSize == 0) {
            loopOnLevels = false;
            preloadLevelsSize = 1;
        }
        if (preloadTimeHoursSize == 0) {
            loopOnTimeHours = false;
            preloadTimeHoursSize = 1;
        }
    } else {
        wxLogError(_("Preprocessing method unknown in PreloadDataWithPreprocessing."));
        return false;
    }

    wxLogVerbose(_("Preprocessing data (%d predictor(s)) while loading."), preprocessSize);

    // Load data for every level and every hour
    for (unsigned int iLevel = 0; iLevel < preloadLevelsSize; iLevel++) {
        for (unsigned int iHour = 0; iHour < preloadTimeHoursSize; iHour++) {
            std::vector<asPredictorArch *> predictorsPreprocess;

            for (int iPre = 0; iPre < preprocessSize; iPre++) {
                wxLogVerbose(_("Preloading data for predictor %d (preprocess %d) of step %d."), iPtor, iPre, iStep);

                // Get level
                float level;
                if (loopOnLevels) {
                    level = preloadLevels[iLevel];
                } else {
                    level = params.GetPreprocessLevel(iStep, iPtor, iPre);
                }

                // Get time
                double timeHours;
                if (loopOnTimeHours) {
                    timeHours = preloadTimeHours[iHour];
                } else {
                    timeHours = params.GetPreprocessTimeHours(iStep, iPtor, iPre);
                }

                // Correct according to the method
                if (method.IsSameAs("Gradients")) {
                    // Nothing to change
                } else if (method.IsSameAs("HumidityIndex")) {
                    if (iPre == 1)
                        level = 0; // pr_wtr
                } else if (method.IsSameAs("HumidityFlux")) {
                    if (iPre == 2)
                        level = 0; // pr_wtr
                } else if (method.IsSameAs("FormerHumidityIndex")) {
                    if (iPre == 2)
                        level = 0; // pr_wtr
                    if (iPre == 3)
                        level = 0; // pr_wtr
                    if (iPre == 0)
                        timeHours = preloadTimeHours[0];
                    if (iPre == 1)
                        timeHours = preloadTimeHours[1];
                    if (iPre == 2)
                        timeHours = preloadTimeHours[0];
                    if (iPre == 3)
                        timeHours = preloadTimeHours[1];
                }

                // Date array object instantiation for the data loading.
                // The array has the same length than timeArrayArchive, and the predictor dates are aligned
                // with the target dates, but the dates are not the same.
                double ptorStart = timeStartData - double(params.GetTimeShiftDays()) + timeHours / 24.0;
                double ptorEnd = timeEndData - double(params.GetTimeShiftDays()) + timeHours / 24.0;
                asTimeArray timeArray(ptorStart, ptorEnd, params.GetTimeArrayAnalogsTimeStepHours(),
                                      asTimeArray::Simple);
                timeArray.Init();

                // Loading the datasets information
                asPredictorArch *predictorPreprocess = asPredictorArch::GetInstance(
                        params.GetPreprocessDatasetId(iStep, iPtor, iPre),
                        params.GetPreprocessDataId(iStep, iPtor, iPre), m_predictorDataDir);
                if (!predictorPreprocess) {
                    Cleanup(predictorsPreprocess);
                    return false;
                }

                // Select the number of members for ensemble data.
                if (predictorPreprocess->IsEnsemble()) {
                    predictorPreprocess->SelectMembers(params.GetPreprocessMembersNb(iStep, iPtor, iPre));
                }

                asGeo geo;
                double yMax = params.GetPreloadYmin(iStep, iPtor) + params.GetPredictorYstep(iStep, iPtor) *
                                                                    double(params.GetPreloadYptsnb(iStep, iPtor) - 1);
                if (yMax > geo.GetAxisYmax()) {
                    double diff = yMax - geo.GetAxisYmax();
                    int removePts = (int) asTools::Round(diff / params.GetPredictorYstep(iStep, iPtor));
                    params.SetPreloadYptsnb(iStep, iPtor, params.GetPreloadYptsnb(iStep, iPtor) - removePts);
                    wxLogVerbose(_("Adapt Y axis extent according to the maximum allowed (from %.2f to %.2f)."), yMax,
                                 yMax - diff);
                }

                // Area object instantiation
                asGeoAreaCompositeGrid *area = asGeoAreaCompositeGrid::GetInstance(
                        params.GetPredictorGridType(iStep, iPtor), params.GetPreloadXmin(iStep, iPtor),
                        params.GetPreloadXptsnb(iStep, iPtor), params.GetPredictorXstep(iStep, iPtor),
                        params.GetPreloadYmin(iStep, iPtor), params.GetPreloadYptsnb(iStep, iPtor),
                        params.GetPredictorYstep(iStep, iPtor), level, asNONE,
                        params.GetPredictorFlatAllowed(iStep, iPtor));
                wxASSERT(area);

                // Check the starting dates coherence
                if (predictorPreprocess->GetOriginalProviderStart() > ptorStart) {
                    wxLogError(_("The first year defined in the parameters (%s) is prior to the start date of the data (%s) (in asMethodCalibrator::PreloadData, preprocessing)."),
                               asTime::GetStringTime(ptorStart),
                               asTime::GetStringTime(predictorPreprocess->GetOriginalProviderStart()));
                    wxDELETE(area);
                    wxDELETE(predictorPreprocess);
                    Cleanup(predictorsPreprocess);
                    return false;
                }

                // Data loading
                wxLogVerbose(_("Loading %s data for level %d, %gh."), params.GetPreprocessDataId(iStep, iPtor, iPre),
                             (int) level, timeHours);
                if (!predictorPreprocess->Load(area, timeArray)) {
                    wxLogError(_("The data could not be loaded."));
                    wxDELETE(area);
                    wxDELETE(predictorPreprocess);
                    return false;
                }
                wxDELETE(area);
                predictorsPreprocess.push_back(predictorPreprocess);
            }

            wxLogVerbose(_("Preprocessing data."));
            asPredictorArch *predictor = new asPredictorArch(*predictorsPreprocess[0]);

            try {
                if (!asPreprocessor::Preprocess(predictorsPreprocess, params.GetPreprocessMethod(iStep, iPtor),
                                                predictor)) {
                    wxLogError(_("Data preprocessing failed."));
                    wxDELETE(predictor);
                    Cleanup(predictorsPreprocess);
                    return false;
                }
                m_preloadedArchive[iStep][iPtor][0][iLevel][iHour] = predictor;
            } catch (std::bad_alloc &ba) {
                wxString msg(ba.what(), wxConvUTF8);
                wxLogError(_("Bad allocation caught in the data preprocessing: %s"), msg);
                wxDELETE(predictor);
                Cleanup(predictorsPreprocess);
                return false;
            } catch (std::exception &e) {
                wxString msg(e.what(), wxConvUTF8);
                wxLogError(_("Exception in the data preprocessing: %s"), msg);
                wxDELETE(predictor);
                Cleanup(predictorsPreprocess);
                return false;
            }
            Cleanup(predictorsPreprocess);
            wxLogVerbose(_("Preprocessing over."));
        }
    }

    // Fix the criteria if S1
    if (method.IsSameAs("Gradients") && params.GetPredictorCriteria(iStep, iPtor).IsSameAs("S1")) {
        params.SetPredictorCriteria(iStep, iPtor, "S1grads");
    } else if (method.IsSameAs("Gradients") && params.GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1")) {
        params.SetPredictorCriteria(iStep, iPtor, "NS1grads");
    }

    return true;
}

bool asMethodCalibrator::LoadData(std::vector<asPredictor *> &predictors, asParametersScoring &params, int iStep,
                                  double timeStartData, double timeEndData)
{
    try {
        // Loop through every predictor
        for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
            if (!PreloadData(params)) {
                wxLogError(_("Could not preload the data."));
                return false;
            }

            if (params.NeedsPreloading(iStep, iPtor)) {
                if (!ExtractPreloadedData(predictors, params, iStep, iPtor)) {
                    return false;
                }
            } else {
                wxLogVerbose(_("Loading data."));

                if (!params.NeedsPreprocessing(iStep, iPtor)) {
                    if (!ExtractDataWithoutPreprocessing(predictors, params, iStep, iPtor, timeStartData,
                                                         timeEndData)) {
                        return false;
                    }
                } else {
                    if (!ExtractDataWithPreprocessing(predictors, params, iStep, iPtor, timeStartData, timeEndData)) {
                        return false;
                    }
                }

                wxLogVerbose(_("Data loaded"));
            }
        }
    } catch (std::bad_alloc &ba) {
        wxString msg(ba.what(), wxConvUTF8);
        wxLogError(_("Bad allocation in the data loading: %s"), msg);
        return false;
    } catch (asException &e) {
        wxString fullMessage = e.GetFullMessage();
        if (!fullMessage.IsEmpty()) {
            wxLogError(fullMessage);
        }
        wxLogError(_("Failed to load data."));
        return false;
    } catch (std::exception &e) {
        wxString msg(e.what(), wxConvUTF8);
        wxLogError(_("Exception in the data loading: %s"), msg);
        return false;
    }

    return true;
}

bool asMethodCalibrator::ExtractPreloadedData(std::vector<asPredictor *> &predictors, asParametersScoring &params,
                                              int iStep, int iPtor)
{
    wxLogVerbose(_("Using preloaded data."));

    bool doPreprocessGradients = false;

    // Get preload arrays
    vf preloadLevels = params.GetPreloadLevels(iStep, iPtor);
    vd preloadTimeHours = params.GetPreloadTimeHours(iStep, iPtor);
    float level;
    double time;
    int iLevel = 0, iHour = 0, iPre = 0;

    // Get data ID
    vwxs preloadDataIds = params.GetPreloadDataIds(iStep, iPtor);
    for (int i = 0; i < preloadDataIds.size(); i++) {
        if (preloadDataIds[i].IsSameAs(params.GetPredictorDataId(iStep, iPtor))) {
            iPre = i;
        }
    }

    if (!params.NeedsPreprocessing(iStep, iPtor)) {
        wxASSERT(preloadLevels.size() > 0);
        wxASSERT(preloadTimeHours.size() > 0);

        level = params.GetPredictorLevel(iStep, iPtor);
        time = params.GetPredictorTimeHours(iStep, iPtor);

        // Get level and hour indices
        iLevel = asTools::SortedArraySearch(&preloadLevels[0], &preloadLevels[preloadLevels.size() - 1], level);
        iHour = asTools::SortedArraySearch(&preloadTimeHours[0], &preloadTimeHours[preloadTimeHours.size() - 1], time);

        // Force gradients preprocessing anyway.
        if (params.GetPredictorCriteria(iStep, iPtor).IsSameAs("S1")) {
            doPreprocessGradients = true;
            params.SetPredictorCriteria(iStep, iPtor, "S1grads");
        } else if (params.GetPredictorCriteria(iStep, iPtor).IsSameAs("S1grads")) {
            doPreprocessGradients = true;
        } else if (params.GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1")) {
            doPreprocessGradients = true;
            params.SetPredictorCriteria(iStep, iPtor, "NS1grads");
        } else if (params.GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1grads")) {
            doPreprocessGradients = true;
        }
    } else {
        // Correct according to the method
        if (params.GetPreprocessMethod(iStep, iPtor).IsSameAs("Gradients")) {
            level = params.GetPreprocessLevel(iStep, iPtor, 0);
            time = params.GetPreprocessTimeHours(iStep, iPtor, 0);
            if (params.GetPredictorCriteria(iStep, iPtor).IsSameAs("S1") ||
                params.GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1")) {
                wxLogError(_("The criteria value has not been changed after the gradient preprocessing."));
                return false;
            }
        } else if (params.GetPreprocessMethod(iStep, iPtor).IsSameAs("HumidityIndex")) {
            level = params.GetPreprocessLevel(iStep, iPtor, 0);
            time = params.GetPreprocessTimeHours(iStep, iPtor, 0);
        } else if (params.GetPreprocessMethod(iStep, iPtor).IsSameAs("HumidityFlux")) {
            level = params.GetPreprocessLevel(iStep, iPtor, 0);
            time = params.GetPreprocessTimeHours(iStep, iPtor, 0);
        } else if (params.GetPreprocessMethod(iStep, iPtor).IsSameAs("FormerHumidityIndex")) {
            level = params.GetPreprocessLevel(iStep, iPtor, 0);
            time = params.GetPreprocessTimeHours(iStep, iPtor, 0);
        } else {
            level = params.GetPreprocessLevel(iStep, iPtor, 0);
            time = params.GetPreprocessTimeHours(iStep, iPtor, 0);
        }

        // Get level and hour indices
        if (preloadLevels.size() > 0) {
            iLevel = asTools::SortedArraySearch(&preloadLevels[0], &preloadLevels[preloadLevels.size() - 1], level);
        }
        if (preloadTimeHours.size() > 0) {
            iHour = asTools::SortedArraySearch(&preloadTimeHours[0], &preloadTimeHours[preloadTimeHours.size() - 1],
                                               time);
        }
    }

    // Check indices
    if (iLevel == asNOT_FOUND || iLevel == asOUT_OF_RANGE) {
        wxLogError(_("The level (%f) could not be found in the preloaded data."), level);
        return false;
    }
    if (iHour == asNOT_FOUND || iHour == asOUT_OF_RANGE) {
        wxLogError(_("The hour (%d) could not be found in the preloaded data."), (int) time);
        return false;
    }

    // Get data on the desired domain
    wxASSERT((unsigned) iStep < m_preloadedArchive.size());
    wxASSERT((unsigned) iPtor < m_preloadedArchive[iStep].size());
    wxASSERT((unsigned) iPre < m_preloadedArchive[iStep][iPtor].size());
    wxASSERT((unsigned) iLevel < m_preloadedArchive[iStep][iPtor][iPre].size());
    wxASSERT((unsigned) iHour < m_preloadedArchive[iStep][iPtor][iPre][iLevel].size());
    if (m_preloadedArchive[iStep][iPtor][iPre][iLevel][iHour] == NULL) {
        if (!GetRandomValidData(params, iStep, iPtor, iPre)) {
            wxLogError(_("The pointer to preloaded data is null."));
            return false;
        }

        level = params.GetPredictorLevel(iStep, iPtor);
        time = params.GetPredictorTimeHours(iStep, iPtor);
        iLevel = asTools::SortedArraySearch(&preloadLevels[0], &preloadLevels[preloadLevels.size() - 1], level);
        iHour = asTools::SortedArraySearch(&preloadTimeHours[0], &preloadTimeHours[preloadTimeHours.size() - 1], time);
    }
    if (iLevel < 0 || iHour < 0) {
        wxLogError(_("An unexpected error occurred."));
        return false;
    }

    // Copy the data
    wxASSERT(m_preloadedArchive[iStep][iPtor][iPre][iLevel][iHour]);
    asPredictorArch *desiredPredictor = new asPredictorArch(*m_preloadedArchive[iStep][iPtor][iPre][iLevel][iHour]);

    // Area object instantiation
    asGeoAreaCompositeGrid *desiredArea = asGeoAreaCompositeGrid::GetInstance(params.GetPredictorGridType(iStep, iPtor),
                                                                              params.GetPredictorXmin(iStep, iPtor),
                                                                              params.GetPredictorXptsnb(iStep, iPtor),
                                                                              params.GetPredictorXstep(iStep, iPtor),
                                                                              params.GetPredictorYmin(iStep, iPtor),
                                                                              params.GetPredictorYptsnb(iStep, iPtor),
                                                                              params.GetPredictorYstep(iStep, iPtor),
                                                                              params.GetPredictorLevel(iStep, iPtor),
                                                                              asNONE,
                                                                              params.GetPredictorFlatAllowed(iStep,
                                                                                                             iPtor));

    wxASSERT(desiredArea);

    if (!desiredPredictor->ClipToArea(desiredArea)) {
        wxLogError(_("The data could not be extracted (iStep = %d, iPtor = %d, iPre = %d, iLevel = %d, iHour = %d)."),
                   iStep, iPtor, iPre, iLevel, iHour);
        wxDELETE(desiredArea);
        wxDELETE(desiredPredictor);
        return false;
    }
    wxDELETE(desiredArea);

    if (doPreprocessGradients) {
        std::vector<asPredictorArch *> predictorsPreprocess;
        predictorsPreprocess.push_back(desiredPredictor);

        asPredictorArch *newPredictor = new asPredictorArch(*predictorsPreprocess[0]);
        if (!asPreprocessor::Preprocess(predictorsPreprocess, "Gradients", newPredictor)) {
            wxLogError(_("Data preprocessing failed."));
            Cleanup(predictorsPreprocess);
            wxDELETE(newPredictor);
            return false;
        }

        Cleanup(predictorsPreprocess);

        wxASSERT(newPredictor->GetTimeSize() > 0);
        predictors.push_back(newPredictor);
    } else {
        wxASSERT(desiredPredictor->GetTimeSize() > 0);
        predictors.push_back(desiredPredictor);
    }

    return true;
}

bool asMethodCalibrator::ExtractDataWithoutPreprocessing(std::vector<asPredictor *> &predictors,
                                                         asParametersScoring &params, int iStep, int iPtor,
                                                         double timeStartData, double timeEndData)
{
    // Date array object instantiation for the data loading. The array has the same length than timeArrayArchive, and the predictor dates are aligned with the target dates, but the dates are not the same.
    double ptorStart = timeStartData - params.GetTimeShiftDays() + params.GetPredictorTimeHours(iStep, iPtor) / 24.0;
    double ptorEnd = timeEndData - params.GetTimeShiftDays() + params.GetPredictorTimeHours(iStep, iPtor) / 24.0;
    asTimeArray timeArray(ptorStart, ptorEnd, params.GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
    timeArray.Init();

    // Loading the datasets information
    asPredictorArch *predictor = asPredictorArch::GetInstance(params.GetPredictorDatasetId(iStep, iPtor),
                                                              params.GetPredictorDataId(iStep, iPtor),
                                                              m_predictorDataDir);
    if (!predictor) {
        return false;
    }

    // Select the number of members for ensemble data.
    if (predictor->IsEnsemble()) {
        predictor->SelectMembers(params.GetPredictorMembersNb(iStep, iPtor));
    }

    // Area object instantiation
    asGeoAreaCompositeGrid *area = asGeoAreaCompositeGrid::GetInstance(params.GetPredictorGridType(iStep, iPtor),
                                                                       params.GetPredictorXmin(iStep, iPtor),
                                                                       params.GetPredictorXptsnb(iStep, iPtor),
                                                                       params.GetPredictorXstep(iStep, iPtor),
                                                                       params.GetPredictorYmin(iStep, iPtor),
                                                                       params.GetPredictorYptsnb(iStep, iPtor),
                                                                       params.GetPredictorYstep(iStep, iPtor),
                                                                       params.GetPredictorLevel(iStep, iPtor), asNONE,
                                                                       params.GetPredictorFlatAllowed(iStep, iPtor));
    wxASSERT(area);

    // Check the starting dates coherence
    if (predictor->GetOriginalProviderStart() > ptorStart) {
        wxLogError(_("The first year defined in the parameters (%s) is prior to the start date of the data (%s) (in asMethodCalibrator::GetAnalogsDates, no preprocessing)."),
                   asTime::GetStringTime(ptorStart), asTime::GetStringTime(predictor->GetOriginalProviderStart()));
        wxDELETE(area);
        wxDELETE(predictor);
        return false;
    }

    // Data loading
    if (!predictor->Load(area, timeArray)) {
        wxLogError(_("The data could not be loaded."));
        wxDELETE(area);
        wxDELETE(predictor);
        return false;
    }
    wxDELETE(area);
    predictors.push_back(predictor);

    return true;
}

bool asMethodCalibrator::ExtractDataWithPreprocessing(std::vector<asPredictor *> &predictors,
                                                      asParametersScoring &params, int iStep, int iPtor,
                                                      double timeStartData, double timeEndData)
{
    std::vector<asPredictorArch *> predictorsPreprocess;

    int preprocessSize = params.GetPreprocessSize(iStep, iPtor);

    wxLogVerbose(_("Preprocessing data (%d predictor(s)) while loading."), preprocessSize);

    for (int iPre = 0; iPre < preprocessSize; iPre++) {
        // Date array object instantiation for the data loading. The array has the same length than timeArrayArchive, and the predictor dates are aligned with the target dates, but the dates are not the same.
        double ptorStart = timeStartData - double(params.GetTimeShiftDays()) +
                           params.GetPreprocessTimeHours(iStep, iPtor, iPre) / 24.0;
        double ptorEnd = timeEndData - double(params.GetTimeShiftDays()) +
                         params.GetPreprocessTimeHours(iStep, iPtor, iPre) / 24.0;
        asTimeArray timeArray(ptorStart, ptorEnd, params.GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
        timeArray.Init();

        // Loading the dataset information
        asPredictorArch *predictorPreprocess = asPredictorArch::GetInstance(
                params.GetPreprocessDatasetId(iStep, iPtor, iPre), params.GetPreprocessDataId(iStep, iPtor, iPre),
                m_predictorDataDir);
        if (!predictorPreprocess) {
            Cleanup(predictorsPreprocess);
            return false;
        }

        // Select the number of members for ensemble data.
        if (predictorPreprocess->IsEnsemble()) {
            predictorPreprocess->SelectMembers(params.GetPreprocessMembersNb(iStep, iPtor, iPre));
        }

        // Area object instantiation
        asGeoAreaCompositeGrid *area = asGeoAreaCompositeGrid::GetInstance(params.GetPredictorGridType(iStep, iPtor),
                                                                           params.GetPredictorXmin(iStep, iPtor),
                                                                           params.GetPredictorXptsnb(iStep, iPtor),
                                                                           params.GetPredictorXstep(iStep, iPtor),
                                                                           params.GetPredictorYmin(iStep, iPtor),
                                                                           params.GetPredictorYptsnb(iStep, iPtor),
                                                                           params.GetPredictorYstep(iStep, iPtor),
                                                                           params.GetPreprocessLevel(iStep, iPtor,
                                                                                                     iPre), asNONE,
                                                                           params.GetPredictorFlatAllowed(iStep,
                                                                                                          iPtor));
        wxASSERT(area);

        // Check the starting dates coherence
        if (predictorPreprocess->GetOriginalProviderStart() > ptorStart) {
            wxLogError(_("The first year defined in the parameters (%s) is prior to the start date of the data (%s) (in asMethodCalibrator::GetAnalogsDates, preprocessing)."),
                       asTime::GetStringTime(ptorStart),
                       asTime::GetStringTime(predictorPreprocess->GetOriginalProviderStart()));
            wxDELETE(area);
            wxDELETE(predictorPreprocess);
            Cleanup(predictorsPreprocess);
            return false;
        }

        // Data loading
        if (!predictorPreprocess->Load(area, timeArray)) {
            wxLogError(_("The data could not be loaded."));
            wxDELETE(area);
            wxDELETE(predictorPreprocess);
            Cleanup(predictorsPreprocess);
            return false;
        }
        wxDELETE(area);
        predictorsPreprocess.push_back(predictorPreprocess);
    }

    // Fix the criteria if S1
    if (params.GetPreprocessMethod(iStep, iPtor).IsSameAs("Gradients") &&
        params.GetPredictorCriteria(iStep, iPtor).IsSameAs("S1")) {
        params.SetPredictorCriteria(iStep, iPtor, "S1grads");
    } else if (params.GetPreprocessMethod(iStep, iPtor).IsSameAs("Gradients") &&
               params.GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1")) {
        params.SetPredictorCriteria(iStep, iPtor, "NS1grads");
    }

    asPredictorArch *predictor = new asPredictorArch(*predictorsPreprocess[0]);
    if (!asPreprocessor::Preprocess(predictorsPreprocess, params.GetPreprocessMethod(iStep, iPtor), predictor)) {
        wxLogError(_("Data preprocessing failed."));
        Cleanup(predictorsPreprocess);
        wxDELETE(predictor);
        return false;
    }

    Cleanup(predictorsPreprocess);
    predictors.push_back(predictor);

    return true;
}

va1f asMethodCalibrator::GetClimatologyData(asParametersScoring &params)
{
    vi stationIds = params.GetPredictandStationIds();

    // Get start and end dates
    a1d predictandTime = m_predictandDB->GetTime();
    float predictandTimeDays = float(params.GetPredictandTimeHours() / 24.0);
    double timeStart, timeEnd;
    timeStart = wxMax(predictandTime[0], params.GetCalibrationStart());
    timeStart = floor(timeStart) + predictandTimeDays;
    timeEnd = wxMin(predictandTime[predictandTime.size() - 1], params.GetCalibrationEnd());
    timeEnd = floor(timeEnd) + predictandTimeDays;

    if (predictandTime.size() < 1) {
        wxLogError(_("An unexpected error occurred."));
        return va1f(stationIds.size(), a1f(1));
    }

    // Check if data are effectively available for this period
    int indexPredictandTimeStart = asTools::SortedArraySearchCeil(&predictandTime[0],
                                                                  &predictandTime[predictandTime.size() - 1],
                                                                  timeStart);
    int indexPredictandTimeEnd = asTools::SortedArraySearchFloor(&predictandTime[0],
                                                                 &predictandTime[predictandTime.size() - 1], timeEnd);

    if (indexPredictandTimeStart < 0 || indexPredictandTimeEnd < 0) {
        wxLogError(_("An unexpected error occurred."));
        return va1f(stationIds.size(), a1f(1));
    }

    for (int iStat = 0; iStat < (int) stationIds.size(); iStat++) {
        a1f predictandDataNorm = m_predictandDB->GetDataNormalizedStation(stationIds[iStat]);

        while (asTools::IsNaN(predictandDataNorm(indexPredictandTimeStart))) {
            indexPredictandTimeStart++;
        }
        while (asTools::IsNaN(predictandDataNorm(indexPredictandTimeEnd))) {
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
    indexPredictandTimeStart = asTools::SortedArraySearchCeil(&predictandTime[0],
                                                              &predictandTime[predictandTime.size() - 1], timeStart);
    indexPredictandTimeEnd = asTools::SortedArraySearchFloor(&predictandTime[0],
                                                             &predictandTime[predictandTime.size() - 1], timeEnd);

    if (indexPredictandTimeStart < 0 || indexPredictandTimeEnd < 0) {
        wxLogError(_("An unexpected error occurred."));
        return va1f(stationIds.size(), a1f(1));
    }

    // Get index step
    double predictandTimeStep = predictandTime[1] - predictandTime[0];
    double targetTimeStep = params.GetTimeArrayTargetTimeStepHours() / 24.0;
    int indexStep = int(targetTimeStep / predictandTimeStep);

    // Get vector length
    int dataLength = (indexPredictandTimeEnd - indexPredictandTimeStart) / indexStep + 1;

    // Process the climatology score
    va1f climatologyData(stationIds.size(), a1f(dataLength));
    for (int iStat = 0; iStat < (int) stationIds.size(); iStat++) {
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

void asMethodCalibrator::Cleanup(std::vector<asPredictorArch *> predictorsPreprocess)
{
    if (predictorsPreprocess.size() > 0) {
        for (unsigned int i = 0; i < predictorsPreprocess.size(); i++) {
            wxDELETE(predictorsPreprocess[i]);
        }
        predictorsPreprocess.resize(0);
    }
}

void asMethodCalibrator::Cleanup(std::vector<asPredictor *> predictors)
{
    if (predictors.size() > 0) {
        for (unsigned int i = 0; i < predictors.size(); i++) {
            wxDELETE(predictors[i]);
        }
        predictors.resize(0);
    }
}

void asMethodCalibrator::Cleanup(std::vector<asCriteria *> criteria)
{
    if (criteria.size() > 0) {
        for (unsigned int i = 0; i < criteria.size(); i++) {
            wxDELETE(criteria[i]);
        }
        criteria.resize(0);
    }
}

void asMethodCalibrator::DeletePreloadedData()
{
    if (!m_preloaded)
        return;

    for (unsigned int i = 0; i < m_preloadedArchive.size(); i++) {
        for (unsigned int j = 0; j < m_preloadedArchive[i].size(); j++) {
            for (unsigned int k = 0; k < m_preloadedArchive[i][j].size(); k++) {
                if (!m_preloadedArchivePointerCopy[i][j][k]) {
                    for (unsigned int l = 0; l < m_preloadedArchive[i][j][k].size(); l++) {
                        for (unsigned int m = 0; m < m_preloadedArchive[i][j][k][l].size(); m++) {
                            wxDELETE(m_preloadedArchive[i][j][k][l][m]);
                        }
                    }
                }
            }
        }
    }

    m_preloaded = false;
}

bool asMethodCalibrator::GetAnalogsDates(asResultsDates &results, asParametersScoring &params, int iStep,
                                         bool &containsNaNs)
{
    // Initialize the result object
    results.SetCurrentStep(iStep);
    results.Init(params);

    // Archive date array
    asTimeArray timeArrayArchive(GetTimeStartArchive(params), GetTimeEndArchive(params),
                                 params.GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
    if (params.HasValidationPeriod()) // remove validation years
    {
        timeArrayArchive.SetForbiddenYears(params.GetValidationYearsVector());
    }
    timeArrayArchive.Init();

    // Target date array
    asTimeArray timeArrayTarget(GetTimeStartCalibration(params), GetTimeEndCalibration(params),
                                params.GetTimeArrayTargetTimeStepHours(), params.GetTimeArrayTargetMode());

    if (!m_validationMode) {
        if (params.HasValidationPeriod()) // remove validation years
        {
            timeArrayTarget.SetForbiddenYears(params.GetValidationYearsVector());
        }
    }

    if (params.GetTimeArrayTargetMode().CmpNoCase("predictand_thresholds") == 0 ||
        params.GetTimeArrayTargetMode().CmpNoCase("PredictandThresholds") == 0) {
        vi stations = params.GetPredictandStationIds();
        if (stations.size() > 1) {
            wxLogError(_("You cannot use predictand thresholds with the multivariate approach."));
            return false;
        }

        if (!timeArrayTarget.Init(*m_predictandDB, params.GetTimeArrayTargetPredictandSerieName(), stations[0],
                                  params.GetTimeArrayTargetPredictandMinThreshold(),
                                  params.GetTimeArrayTargetPredictandMaxThreshold())) {
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
        timeArrayTarget.KeepOnlyYears(params.GetValidationYearsVector());
    }

    // Data date array
    double timeStartData = wxMin(GetTimeStartCalibration(params), GetTimeStartArchive(params)); // Always Jan 1st
    double timeEndData = wxMax(GetTimeEndCalibration(params), GetTimeEndArchive(params));
    asTimeArray timeArrayData(timeStartData, timeEndData, params.GetTimeArrayAnalogsTimeStepHours(),
                              asTimeArray::Simple);
    timeArrayData.Init();

    // Check on the archive length
    if (timeArrayArchive.GetSize() < 100) {
        wxLogError(_("The time array is not consistent in asMethodCalibrator::GetAnalogsDates: size=%d."),
                   timeArrayArchive.GetSize());
        return false;
    }
    wxLogVerbose(_("Date arrays created."));
    /*
        // Calculate needed memory
        wxLongLong neededMem = 0;
        for(int iPtor=0; iPtor<params.GetPredictorsNb(iStep); iPtor++)
        {
            neededMem += (params.GetPredictorXptsnb(iStep, iPtor))
                         * (params.GetPredictorYptsnb(iStep, iPtor));
        }
        neededMem *= timeArrayArchive.GetSize(); // time dimension
        neededMem *= 4; // to bytes (for floats)
        double neededMemMb = neededMem.ToDouble();
        neededMemMb /= 1048576.0; // to Mb

        // Get available memory
        wxMemorySize freeMemSize = wxGetFreeMemory();
        wxLongLong freeMem = freeMemSize;
        double freeMemMb = freeMem.ToDouble();
        freeMemMb /= 1048576.0; // To Mb

        if(freeMemSize==-1)
        {
            wxLogVerbose(_("Needed memory for data: %.2f Mb (cannot evaluate available memory)"), neededMemMb);
        }
        else
        {
            wxLogVerbose(_("Needed memory for data: %.2f Mb (%.2f Mb available)"), neededMemMb, freeMemMb);
            if(neededMemMb>freeMemMb)
            {
                wxLogError(_("Data cannot fit into available memory."));
                return false;
            }
        }
    */
    // Load the predictor data
    std::vector<asPredictor *> predictors;
    if (!LoadData(predictors, params, iStep, timeStartData, timeEndData)) {
        wxLogError(_("Failed loading predictor data."));
        Cleanup(predictors);
        return false;
    }

    // Create the criterion
    std::vector<asCriteria *> criteria;
    for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
        // Instantiate a score object
        asCriteria *criterion = asCriteria::GetInstance(params.GetPredictorCriteria(iStep, iPtor));
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

bool asMethodCalibrator::GetAnalogsSubDates(asResultsDates &results, asParametersScoring &params,
                                            asResultsDates &anaDates, int iStep, bool &containsNaNs)
{
    // Initialize the result object
    results.SetCurrentStep(iStep);
    results.Init(params);

    // Date array object instantiation for the processor
    wxLogVerbose(_("Creating a date arrays for the processor."));
    double timeStart = params.GetArchiveStart();
    double timeEnd = params.GetArchiveEnd();
    timeEnd = wxMin(timeEnd,
                    timeEnd - params.GetTimeSpanDays()); // Adjust so the predictors search won't overtake the array
    asTimeArray timeArrayArchive(timeStart, timeEnd, params.GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
    timeArrayArchive.Init();
    wxLogVerbose(_("Date arrays created."));

    // Load the predictor data
    std::vector<asPredictor *> predictors;
    if (!LoadData(predictors, params, iStep, timeStart, timeEnd)) {
        wxLogError(_("Failed loading predictor data."));
        Cleanup(predictors);
        return false;
    }

    // Create the score objects
    std::vector<asCriteria *> criteria;
    for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
        wxLogVerbose(_("Creating a criterion object."));
        asCriteria *criterion = asCriteria::GetInstance(params.GetPredictorCriteria(iStep, iPtor));
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

bool asMethodCalibrator::GetAnalogsValues(asResultsValues &results, asParametersScoring &params,
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

bool asMethodCalibrator::GetAnalogsScores(asResultsScores &results, asParametersScoring &params,
                                          asResultsValues &anaValues, int iStep)
{
    // Initialize the result object
    results.SetCurrentStep(iStep);
    results.Init(params);

    // Instantiate a score object
    wxLogVerbose(_("Instantiating a score object"));
    asScore *score = asScore::GetInstance(params.GetScoreName());
    score->SetQuantile(params.GetScoreQuantile());
    score->SetThreshold(params.GetScoreThreshold());

    if (score->UsesClimatology() && m_scoreClimatology.size() == 0) {
        wxLogVerbose(_("Processing the score of the climatology."));

        va1f climatologyData = GetClimatologyData(params);
        vi stationIds = params.GetPredictandStationIds();
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

bool asMethodCalibrator::GetAnalogsTotalScore(asResultsTotalScore &results, asParametersScoring &params,
                                              asResultsScores &anaScores, int iStep)
{
    // Initialize the result object
    results.SetCurrentStep(iStep);
    results.Init(params);

    // Date array object instantiation for the final score
    wxLogVerbose(_("Creating a date array for the final score."));
    double timeStart = params.GetCalibrationStart();
    double timeEnd = params.GetCalibrationEnd() + 1;
    while (timeEnd > params.GetCalibrationEnd() + 0.999) {
        timeEnd -= params.GetTimeArrayTargetTimeStepHours() / 24.0;
    }
    asTimeArray timeArray(timeStart, timeEnd, params.GetTimeArrayTargetTimeStepHours(), params.GetScoreTimeArrayMode());

    // TODO: Add every options for the Init function (generic version)
    //    timeArray.Init(params.GetScoreTimeArrayDate(), params.GetForecastScoreTimeArrayIntervalDays());
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
            rowEnd = asTools::SortedArraySearchFloor(&analogsNbVect[0], &analogsNbVect[analogsNbVect.size() - 1],
                                                     prevAnalogsNb);
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
        if (!GetAnalogsDates(anaDates, params, iStep, containsNaNs))
            return false;
    } else {
        if (!GetAnalogsSubDates(anaDates, params, anaDatesPrevious, iStep, containsNaNs))
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

        if (!GetAnalogsValues(anaValues, params, anaDates, iStep))
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

            if (!GetAnalogsScores(anaScores, params, anaValues, iStep))
                return false;
            if (!GetAnalogsTotalScore(anaScoreFinal, params, anaScores, iStep))
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

bool asMethodCalibrator::SaveDetails(asParametersCalibration &params)
{
    asResultsDates anaDatesPrevious;
    asResultsDates anaDates;
    asResultsValues anaValues;
    asResultsScores anaScores;
    asResultsTotalScore anaScoreFinal;

    // Process every step one after the other
    int stepsNb = params.GetStepsNb();
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

bool asMethodCalibrator::Validate(asParametersCalibration &params)
{
    bool skipValidation = false;
    wxFileConfig::Get()->Read("/Optimizer/SkipValidation", &skipValidation, false);

    if (skipValidation) {
        return true;
    }

    if (!params.HasValidationPeriod()) {
        wxLogWarning("The parameters have no validation period !");
        return false;
    }

    m_validationMode = true;

    asResultsDates anaDatesPrevious;
    asResultsDates anaDates;
    asResultsValues anaValues;
    asResultsScores anaScores;
    asResultsTotalScore anaScoreFinal;

    // Process every step one after the other
    int stepsNb = params.GetStepsNb();
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

bool asMethodCalibrator::GetRandomValidData(asParametersScoring &params, int iStep, int iPtor, int iPre)
{
    vi levels, hours;

    for (int iLevel = 0; iLevel < m_preloadedArchive[iStep][iPtor][iPre].size(); iLevel++) {
        for (int iHour = 0; iHour < m_preloadedArchive[iStep][iPtor][iPre][iLevel].size(); iHour++) {
            if (m_preloadedArchive[iStep][iPtor][iPre][iLevel][iHour] != NULL) {
                levels.push_back(iLevel);
                hours.push_back(iHour);
            }
        }
    }

    wxASSERT(levels.size() == hours.size());

    int randomIndex = asTools::Random(0, levels.size() - 1, 1);
    float newLevel = params.GetPreloadLevels(iStep, iPtor)[levels[randomIndex]];
    double newHour = params.GetPreloadTimeHours(iStep, iPtor)[hours[randomIndex]];

    params.SetPredictorLevel(iStep, iPtor, newLevel);
    params.SetPredictorTimeHours(iStep, iPtor, newHour);

    return true;
}
