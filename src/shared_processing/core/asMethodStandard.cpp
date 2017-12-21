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

#include "asMethodStandard.h"
#include "asPredictorArch.h"
#include "asPreprocessor.h"
#include "asParameters.h"
#include "asThreadPreloadArchiveData.h"
#include "asTimeArray.h"
#include "asGeoAreaCompositeGrid.h"
#include "asCriteria.h"
#include "asPredictorArch.h"


wxDEFINE_EVENT(asEVT_STATUS_STARTING, wxCommandEvent);
wxDEFINE_EVENT(asEVT_STATUS_RUNNING, wxCommandEvent);
wxDEFINE_EVENT(asEVT_STATUS_FAILED, wxCommandEvent);
wxDEFINE_EVENT(asEVT_STATUS_SUCCESS, wxCommandEvent);
wxDEFINE_EVENT(asEVT_STATUS_DOWNLOADING, wxCommandEvent);
wxDEFINE_EVENT(asEVT_STATUS_DOWNLOADED, wxCommandEvent);
wxDEFINE_EVENT(asEVT_STATUS_LOADING, wxCommandEvent);
wxDEFINE_EVENT(asEVT_STATUS_LOADED, wxCommandEvent);
wxDEFINE_EVENT(asEVT_STATUS_SAVING, wxCommandEvent);
wxDEFINE_EVENT(asEVT_STATUS_SAVED, wxCommandEvent);
wxDEFINE_EVENT(asEVT_STATUS_PROCESSING, wxCommandEvent);
wxDEFINE_EVENT(asEVT_STATUS_PROCESSED, wxCommandEvent);


asMethodStandard::asMethodStandard()
        : m_cancel(false),
          m_preloaded(false),
          m_predictandDB(nullptr)
{

}

asMethodStandard::~asMethodStandard()
{
    wxDELETE(m_predictandDB);
}

bool asMethodStandard::Manager()
{
    return false;
}

bool asMethodStandard::LoadPredictandDB(const wxString &predictandDBFilePath)
{
    wxDELETE(m_predictandDB);

    if (predictandDBFilePath.IsEmpty()) {
        if (m_predictandDBFilePath.IsEmpty()) {
            wxLogError(_("There is no predictand database file selected."));
            return false;
        }

        m_predictandDB = asPredictand::GetInstance(m_predictandDBFilePath);
        if (!m_predictandDB)
            return false;

        if (!m_predictandDB->Load(m_predictandDBFilePath)) {
            wxLogError(_("Couldn't load the predictand database."));
            return false;
        }
    } else {
        m_predictandDB = asPredictand::GetInstance(predictandDBFilePath);
        if (!m_predictandDB)
            return false;

        if (!m_predictandDB->Load(predictandDBFilePath)) {
            wxLogError(_("Couldn't load the predictand database."));
            return false;
        }
    }

    if (!m_predictandDB)
        return false;
    wxASSERT(m_predictandDB);

    return true;
}

void asMethodStandard::Cancel()
{
    m_cancel = true;
}

bool asMethodStandard::Preprocess(std::vector<asPredictorArch *> predictors, const wxString &method, asPredictor *result)
{
    std::vector<asPredictor *> ptorsPredictors(predictors.begin(), predictors.end());

    return asPreprocessor::Preprocess(ptorsPredictors, method, result);
}

double asMethodStandard::GetTimeStartArchive(asParameters *params) const
{
    double timeStartArchive = params->GetArchiveStart();
    timeStartArchive += std::abs(params->GetTimeShiftDays());

    return timeStartArchive;
}

double asMethodStandard::GetTimeEndArchive(asParameters *params) const
{
    double timeEndArchive = params->GetArchiveEnd();
    timeEndArchive = wxMin(timeEndArchive, timeEndArchive - params->GetTimeSpanDays());

    return timeEndArchive;
}

void asMethodStandard::InitializePreloadedArchiveDataContainers(asParameters *params)
{
    if (m_preloadedArchive.empty()) {
        m_preloadedArchive.resize((unsigned long) params->GetStepsNb());
        m_preloadedArchivePointerCopy.resize((unsigned long) params->GetStepsNb());
        for (int iStep = 0; iStep < params->GetStepsNb(); iStep++) {
            m_preloadedArchive[iStep].resize((unsigned long) params->GetPredictorsNb(iStep));
            m_preloadedArchivePointerCopy[iStep].resize((unsigned long) params->GetPredictorsNb(iStep));

            for (int iPtor = 0; iPtor < params->GetPredictorsNb(iStep); iPtor++) {

                vwxs preloadDataIds = params->GetPreloadDataIds(iStep, iPtor);
                vf preloadLevels = params->GetPreloadLevels(iStep, iPtor);
                vd preloadTimeHours = params->GetPreloadTimeHours(iStep, iPtor);

                unsigned long preloadDataIdsSize = wxMax(preloadDataIds.size(), 1);
                unsigned long preloadLevelsSize = wxMax(preloadLevels.size(), 1);
                unsigned long preloadTimeHoursSize = wxMax(preloadTimeHours.size(), 1);

                m_preloadedArchivePointerCopy[iStep][iPtor].resize(preloadDataIdsSize);
                m_preloadedArchive[iStep][iPtor].resize(preloadDataIdsSize);

                for (unsigned int iDat = 0; iDat < preloadDataIdsSize; iDat++) {
                    m_preloadedArchivePointerCopy[iStep][iPtor][iDat] = false;
                    m_preloadedArchive[iStep][iPtor][iDat].resize(preloadLevelsSize);

                    // Load data for every level and every hour
                    for (unsigned int iLevel = 0; iLevel < preloadLevelsSize; iLevel++) {
                        m_preloadedArchive[iStep][iPtor][iDat][iLevel].resize(preloadTimeHoursSize);
                        for (unsigned int iHour = 0; iHour < preloadTimeHoursSize; iHour++) {
                            m_preloadedArchive[iStep][iPtor][iDat][iLevel][iHour] = NULL;
                        }
                    }
                }
            }
        }
    }
}

bool asMethodStandard::PreloadArchiveData(asParameters *params)
{
    if (!m_preloaded) {
        // Set preload to true here, so cleanup is made in case of exceptions.
        m_preloaded = true;

        InitializePreloadedArchiveDataContainers(params);

        if (!ProceedToArchiveDataPreloading(params))
            return false;

        if (!CheckArchiveDataIsPreloaded(params))
            return false;
    }

    return true;
}

bool asMethodStandard::ProceedToArchiveDataPreloading(asParameters *params)
{
    bool parallelDataLoad = false;
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Read("/General/ParallelDataLoad", &parallelDataLoad, true);
    ThreadsManager().CritSectionConfig().Leave();

    if (parallelDataLoad) {
        wxLogVerbose(_("Preloading data with threads."));
    }

    for (int iStep = 0; iStep < params->GetStepsNb(); iStep++) {
        for (int iPtor = 0; iPtor < params->GetPredictorsNb(iStep); iPtor++) {
            if (params->NeedsPreloading(iStep, iPtor)) {

                if (params->NeedsPreprocessing(iStep, iPtor)) {
                    if (PointersArchiveDataShared(params, iStep, iPtor, 0)) {
                        continue;
                    }
                    if (!PreloadArchiveDataWithPreprocessing(params, iStep, iPtor)) {
                        return false;
                    }
                } else {
                    for (int i = 0; i < params->GetPredictorDataIdNb(iStep, iPtor); i++) {
                        if (PointersArchiveDataShared(params, iStep, iPtor, i)) {
                            continue;
                        }
                        if (parallelDataLoad) {
                            auto *thread = new asThreadPreloadArchiveData(this, params, iStep, iPtor, i);
                            if (!ThreadsManager().HasFreeThread(thread->GetType())) {
                                ThreadsManager().WaitForFreeThread(thread->GetType());
                            }
                            ThreadsManager().AddThread(thread);
                        } else {
                            if (!PreloadArchiveDataWithoutPreprocessing(params, iStep, iPtor, i)) {
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

bool asMethodStandard::CheckArchiveDataIsPreloaded(const asParameters *params) const
{
    for (int iStep = 0; iStep < params->GetStepsNb(); iStep++) {
        for (int iPtor = 0; iPtor < params->GetPredictorsNb(iStep); iPtor++) {
            if (params->NeedsPreloading(iStep, iPtor)) {
                if (!params->NeedsPreprocessing(iStep, iPtor)) {
                    for (int iPre = 0; iPre < params->GetPredictorDataIdNb(iStep, iPtor); iPre++) {
                        if (!HasPreloadedArchiveData(iStep, iPtor, iPre)) {
                            wxLogError(_("No data was preloaded for step %d, predictor %d and num %d."),
                                       iStep, iPtor, iPre);
                            return false;
                        }
                    }
                }
                if (!HasPreloadedArchiveData(iStep, iPtor)) {
                    wxLogError(_("No data was preloaded for step %d and predictor %d."), iStep, iPtor);
                    return false;
                }
            }
        }
    }

    return true;
}

bool asMethodStandard::HasPreloadedArchiveData(int iStep, int iPtor) const
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

bool asMethodStandard::HasPreloadedArchiveData(int iStep, int iPtor, int iPre) const
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

bool asMethodStandard::PointersArchiveDataShared(asParameters *params, int iStep, int iPtor, int iPre)
{
    if (iStep == 0 && iPtor == 0) {
        return false;
    }

    int prev_step = 0, prev_ptor = 0, prev_dat = 0;
    bool share = false;

    for (prev_step = 0; prev_step <= iStep; prev_step++) {

        int ptor_max = params->GetPredictorsNb(prev_step);
        if (prev_step == iStep) {
            ptor_max = iPtor;
        }

        for (prev_ptor = 0; prev_ptor < ptor_max; prev_ptor++) {
            share = true;

            if (!params->NeedsPreprocessing(iStep, iPtor)) {
                if (!params->GetPredictorDatasetId(iStep, iPtor).IsSameAs(
                        params->GetPredictorDatasetId(prev_step, prev_ptor), false))
                    share = false;
                if (!params->GetPredictorGridType(iStep, iPtor).IsSameAs(
                        params->GetPredictorGridType(prev_step, prev_ptor), false))
                    share = false;
            } else {
                if (!params->GetPreprocessMethod(iStep, iPtor).IsSameAs(params->GetPreprocessMethod(prev_step, prev_ptor),
                                                                       false))
                    share = false;
                if (params->GetPreprocessSize(iStep, iPtor) != params->GetPreprocessSize(prev_step, prev_ptor)) {
                    share = false;
                } else {
                    int preprocessSize = params->GetPreprocessSize(iStep, iPtor);

                    for (int i = 0; i < preprocessSize; i++) {
                        if (!params->GetPreprocessDatasetId(iStep, iPtor, i).IsSameAs(
                                params->GetPreprocessDatasetId(prev_step, prev_ptor, i), false))
                            share = false;
                        if (!params->GetPreprocessDataId(iStep, iPtor, i).IsSameAs(
                                params->GetPreprocessDataId(prev_step, prev_ptor, i), false))
                            share = false;
                    }
                }
            }

            if (params->GetPreloadXmin(iStep, iPtor) != params->GetPreloadXmin(prev_step, prev_ptor))
                share = false;
            if (params->GetPreloadXptsnb(iStep, iPtor) != params->GetPreloadXptsnb(prev_step, prev_ptor))
                share = false;
            if (params->GetPredictorXstep(iStep, iPtor) != params->GetPredictorXstep(prev_step, prev_ptor))
                share = false;
            if (params->GetPreloadYmin(iStep, iPtor) != params->GetPreloadYmin(prev_step, prev_ptor))
                share = false;
            if (params->GetPreloadYptsnb(iStep, iPtor) != params->GetPreloadYptsnb(prev_step, prev_ptor))
                share = false;
            if (params->GetPredictorYstep(iStep, iPtor) != params->GetPredictorYstep(prev_step, prev_ptor))
                share = false;
            if (params->GetPredictorFlatAllowed(iStep, iPtor) != params->GetPredictorFlatAllowed(prev_step, prev_ptor))
                share = false;

            vf levels1 = params->GetPreloadLevels(iStep, iPtor);
            vf levels2 = params->GetPreloadLevels(prev_step, prev_ptor);
            if (levels1.size() != levels2.size()) {
                share = false;
            } else {
                for (unsigned int i = 0; i < levels1.size(); i++) {
                    if (levels1[i] != levels2[i])
                        share = false;
                }
            }

            vd hours1 = params->GetPreloadTimeHours(iStep, iPtor);
            vd hours2 = params->GetPreloadTimeHours(prev_step, prev_ptor);
            if (hours1.size() != hours2.size()) {
                share = false;
            } else {
                for (unsigned int i = 0; i < hours1.size(); i++) {
                    if (hours1[i] != hours2[i])
                        share = false;
                }
            }

            bool dataIdFound = false;
            vwxs preloadDataIds = params->GetPreloadDataIds(iStep, iPtor);
            vwxs preloadDataIdsPrev = params->GetPreloadDataIds(prev_step, prev_ptor);
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

        vf preloadLevels = params->GetPreloadLevels(iStep, iPtor);
        vd preloadTimeHours = params->GetPreloadTimeHours(iStep, iPtor);
        wxASSERT(!preloadLevels.empty());
        wxASSERT(!preloadTimeHours.empty());

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

        params->SetPreloadYptsnb(iStep, iPtor, params->GetPreloadYptsnb(prev_step, prev_ptor));

        return true;
    }

    return false;
}

bool asMethodStandard::PreloadArchiveDataWithoutPreprocessing(asParameters *params, int iStep, int iPtor, int i)
{
    wxLogVerbose(_("Preloading data for predictor %d of step %d."), iPtor, iStep);

    double timeStartData = GetEffectiveArchiveDataStart(params);
    double timeEndData = GetEffectiveArchiveDataEnd(params);

    vwxs preloadDataIds = params->GetPreloadDataIds(iStep, iPtor);
    vf preloadLevels = params->GetPreloadLevels(iStep, iPtor);
    vd preloadTimeHours = params->GetPreloadTimeHours(iStep, iPtor);
    wxASSERT(preloadDataIds.size() > i);
    wxASSERT(!preloadLevels.empty());
    wxASSERT(!preloadTimeHours.empty());

    // Load data for every level and every hour
    for (unsigned int iLevel = 0; iLevel < preloadLevels.size(); iLevel++) {
        for (unsigned int iHour = 0; iHour < preloadTimeHours.size(); iHour++) {
            // Loading the dataset information
            asPredictorArch *predictor = asPredictorArch::GetInstance(params->GetPredictorDatasetId(iStep, iPtor),
                                                                      preloadDataIds[i], m_predictorDataDir);
            if (!predictor) {
                return false;
            }

            // Select the number of members for ensemble data.
            if (predictor->IsEnsemble()) {
                predictor->SelectMembers(params->GetPredictorMembersNb(iStep, iPtor));
            }

            // Date array object instantiation for the data loading.
            // The array has the same length than timeArrayArchive, and the predictor dates are aligned with the
            // target dates, but the dates are not the same.
            double ptorStart = timeStartData - double(params->GetTimeShiftDays()) + preloadTimeHours[iHour] / 24.0;

            wxLogDebug("%f - %f + %f = %f", timeStartData, double(params->GetTimeShiftDays()),
                       preloadTimeHours[iHour] / 24.0, ptorStart);
            wxLogDebug("ptorStart = %s", asTime::GetStringTime(ptorStart));
            wxLogDebug("timeStartData = %s", asTime::GetStringTime(timeStartData));
            wxLogDebug("params->GetTimeShiftDays() = %f", double(params->GetTimeShiftDays()));
            wxLogDebug("preloadTimeHours[iHour]/24.0 = %f", preloadTimeHours[iHour] / 24.0);

            double ptorEnd = timeEndData - double(params->GetTimeShiftDays()) + preloadTimeHours[iHour] / 24.0;

            asTimeArray timeArray(ptorStart, ptorEnd, params->GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
            timeArray.Init();

            asGeo geo;
            double yMax = params->GetPreloadYmin(iStep, iPtor) +
                          params->GetPredictorYstep(iStep, iPtor) * (double) (params->GetPreloadYptsnb(iStep, iPtor) - 1);
            if (yMax > geo.GetAxisYmax()) {
                double diff = yMax - geo.GetAxisYmax();
                int removePts = (int) asTools::Round(diff / params->GetPredictorYstep(iStep, iPtor));
                params->SetPreloadYptsnb(iStep, iPtor, params->GetPreloadYptsnb(iStep, iPtor) - removePts);
                wxLogVerbose(_("Adapt Y axis extent according to the maximum allowed (from %.3f to %.3f)."), yMax,
                             yMax - diff);
                wxLogVerbose(_("Remove %d points (%.3f-%.3f)/%.3f."), removePts, yMax, geo.GetAxisYmax(),
                             params->GetPredictorYstep(iStep, iPtor));
            }

            wxASSERT(params->GetPreloadXptsnb(iStep, iPtor) > 0);
            wxASSERT(params->GetPreloadYptsnb(iStep, iPtor) > 0);

            // Area object instantiation
            asGeoAreaCompositeGrid *area = asGeoAreaCompositeGrid::GetInstance(
                    params->GetPredictorGridType(iStep, iPtor), params->GetPreloadXmin(iStep, iPtor),
                    params->GetPreloadXptsnb(iStep, iPtor), params->GetPredictorXstep(iStep, iPtor),
                    params->GetPreloadYmin(iStep, iPtor), params->GetPreloadYptsnb(iStep, iPtor),
                    params->GetPredictorYstep(iStep, iPtor), preloadLevels[iLevel], asNONE,
                    params->GetPredictorFlatAllowed(iStep, iPtor));
            wxASSERT(area);

            // Check the starting dates coherence
            if (predictor->GetOriginalProviderStart() > ptorStart) {
                wxLogError(_("The first year defined in the parameters (%s) is prior to the start date of the data (%s) (in asMethodCalibrator::PreloadArchiveData)."),
                           asTime::GetStringTime(ptorStart),
                           asTime::GetStringTime(predictor->GetOriginalProviderStart()));
                wxDELETE(area);
                wxDELETE(predictor);
                return false;
            }

            // Data loading
            wxLogVerbose(_("Loading %s data for level %d, %gh."), preloadDataIds[i], (int) preloadLevels[iLevel],
                         preloadTimeHours[iHour]);
            try {
                if (!predictor->Load(area, timeArray)) {
                    wxLogWarning(_("The data (%s for level %d, at %gh) could not be loaded."), preloadDataIds[i],
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

            m_preloadedArchive[iStep][iPtor][i][iLevel][iHour] = predictor;
        }
    }

    return true;
}

bool asMethodStandard::PreloadArchiveDataWithPreprocessing(asParameters *params, int iStep, int iPtor)
{
    wxLogVerbose(_("Preloading data for predictor preprocessed %d of step %d."), iPtor, iStep);

    double timeStartData = GetEffectiveArchiveDataStart(params);
    double timeEndData = GetEffectiveArchiveDataEnd(params);

    // Check the preprocessing method
    wxString method = params->GetPreprocessMethod(iStep, iPtor);

    // Get the number of sub predictors
    int preprocessSize = params->GetPreprocessSize(iStep, iPtor);

    // Levels and time arrays
    vf preloadLevels = params->GetPreloadLevels(iStep, iPtor);
    vd preloadTimeHours = params->GetPreloadTimeHours(iStep, iPtor);

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
        wxLogError(_("Preprocessing method unknown in PreloadArchiveDataWithPreprocessing."));
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
                    level = params->GetPreprocessLevel(iStep, iPtor, iPre);
                }

                // Get time
                double timeHours;
                if (loopOnTimeHours) {
                    timeHours = preloadTimeHours[iHour];
                } else {
                    timeHours = params->GetPreprocessTimeHours(iStep, iPtor, iPre);
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
                double ptorStart = timeStartData - double(params->GetTimeShiftDays()) + timeHours / 24.0;
                double ptorEnd = timeEndData - double(params->GetTimeShiftDays()) + timeHours / 24.0;
                asTimeArray timeArray(ptorStart, ptorEnd, params->GetTimeArrayAnalogsTimeStepHours(),
                                      asTimeArray::Simple);
                timeArray.Init();

                // Loading the datasets information
                asPredictorArch *predictorPreprocess = asPredictorArch::GetInstance(
                        params->GetPreprocessDatasetId(iStep, iPtor, iPre),
                        params->GetPreprocessDataId(iStep, iPtor, iPre), m_predictorDataDir);
                if (!predictorPreprocess) {
                    Cleanup(predictorsPreprocess);
                    return false;
                }

                // Select the number of members for ensemble data.
                if (predictorPreprocess->IsEnsemble()) {
                    predictorPreprocess->SelectMembers(params->GetPreprocessMembersNb(iStep, iPtor, iPre));
                }

                asGeo geo;
                double yMax = params->GetPreloadYmin(iStep, iPtor) + params->GetPredictorYstep(iStep, iPtor) *
                                                                    double(params->GetPreloadYptsnb(iStep, iPtor) - 1);
                if (yMax > geo.GetAxisYmax()) {
                    double diff = yMax - geo.GetAxisYmax();
                    int removePts = (int) asTools::Round(diff / params->GetPredictorYstep(iStep, iPtor));
                    params->SetPreloadYptsnb(iStep, iPtor, params->GetPreloadYptsnb(iStep, iPtor) - removePts);
                    wxLogVerbose(_("Adapt Y axis extent according to the maximum allowed (from %.2f to %.2f)."), yMax,
                                 yMax - diff);
                }

                // Area object instantiation
                asGeoAreaCompositeGrid *area = asGeoAreaCompositeGrid::GetInstance(
                        params->GetPredictorGridType(iStep, iPtor), params->GetPreloadXmin(iStep, iPtor),
                        params->GetPreloadXptsnb(iStep, iPtor), params->GetPredictorXstep(iStep, iPtor),
                        params->GetPreloadYmin(iStep, iPtor), params->GetPreloadYptsnb(iStep, iPtor),
                        params->GetPredictorYstep(iStep, iPtor), level, asNONE,
                        params->GetPredictorFlatAllowed(iStep, iPtor));
                wxASSERT(area);

                // Check the starting dates coherence
                if (predictorPreprocess->GetOriginalProviderStart() > ptorStart) {
                    wxLogError(_("The first year defined in the parameters (%s) is prior to the start date of the data (%s) (in asMethodCalibrator::PreloadArchiveData, preprocessing)."),
                               asTime::GetStringTime(ptorStart),
                               asTime::GetStringTime(predictorPreprocess->GetOriginalProviderStart()));
                    wxDELETE(area);
                    wxDELETE(predictorPreprocess);
                    Cleanup(predictorsPreprocess);
                    return false;
                }

                // Data loading
                wxLogVerbose(_("Loading %s data for level %d, %gh."), params->GetPreprocessDataId(iStep, iPtor, iPre),
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
                if (!Preprocess(predictorsPreprocess, params->GetPreprocessMethod(iStep, iPtor),
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
    if (method.IsSameAs("Gradients") && params->GetPredictorCriteria(iStep, iPtor).IsSameAs("S1")) {
        params->SetPredictorCriteria(iStep, iPtor, "S1grads");
    } else if (method.IsSameAs("Gradients") && params->GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1")) {
        params->SetPredictorCriteria(iStep, iPtor, "NS1grads");
    }

    return true;
}

bool asMethodStandard::LoadArchiveData(std::vector<asPredictor *> &predictors, asParameters *params, int iStep,
                                       double timeStartData, double timeEndData)
{
    try {
        // Loop through every predictor
        for (int iPtor = 0; iPtor < params->GetPredictorsNb(iStep); iPtor++) {
            if (!PreloadArchiveData(params)) {
                wxLogError(_("Could not preload the data."));
                return false;
            }

            if (params->NeedsPreloading(iStep, iPtor)) {
                if (!ExtractPreloadedArchiveData(predictors, params, iStep, iPtor)) {
                    return false;
                }
            } else {
                wxLogVerbose(_("Loading data."));

                if (!params->NeedsPreprocessing(iStep, iPtor)) {
                    if (!ExtractArchiveDataWithoutPreprocessing(predictors, params, iStep, iPtor, timeStartData,
                                                                timeEndData)) {
                        return false;
                    }
                } else {
                    if (!ExtractArchiveDataWithPreprocessing(predictors, params, iStep, iPtor, timeStartData,
                                                             timeEndData)) {
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

bool asMethodStandard::ExtractPreloadedArchiveData(std::vector<asPredictor *> &predictors, asParameters *params,
                                                   int iStep, int iPtor)
{
    wxLogVerbose(_("Using preloaded data."));

    bool doPreprocessGradients = false;

    // Get preload arrays
    vf preloadLevels = params->GetPreloadLevels(iStep, iPtor);
    vd preloadTimeHours = params->GetPreloadTimeHours(iStep, iPtor);
    float level;
    double time;
    int iLevel = 0, iHour = 0, iPre = 0;

    // Get data ID
    vwxs preloadDataIds = params->GetPreloadDataIds(iStep, iPtor);
    for (int i = 0; i < preloadDataIds.size(); i++) {
        if (preloadDataIds[i].IsSameAs(params->GetPredictorDataId(iStep, iPtor))) {
            iPre = i;
        }
    }

    if (!params->NeedsPreprocessing(iStep, iPtor)) {
        wxASSERT(!preloadLevels.empty());
        wxASSERT(!preloadTimeHours.empty());

        level = params->GetPredictorLevel(iStep, iPtor);
        time = params->GetPredictorTimeHours(iStep, iPtor);

        // Get level and hour indices
        iLevel = asTools::SortedArraySearch(&preloadLevels[0], &preloadLevels[preloadLevels.size() - 1], level);
        iHour = asTools::SortedArraySearch(&preloadTimeHours[0], &preloadTimeHours[preloadTimeHours.size() - 1], time);

        // Force gradients preprocessing anyway.
        if (params->GetPredictorCriteria(iStep, iPtor).IsSameAs("S1")) {
            doPreprocessGradients = true;
            params->SetPredictorCriteria(iStep, iPtor, "S1grads");
        } else if (params->GetPredictorCriteria(iStep, iPtor).IsSameAs("S1grads")) {
            doPreprocessGradients = true;
        } else if (params->GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1")) {
            doPreprocessGradients = true;
            params->SetPredictorCriteria(iStep, iPtor, "NS1grads");
        } else if (params->GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1grads")) {
            doPreprocessGradients = true;
        }
    } else {
        // Correct according to the method
        if (params->GetPreprocessMethod(iStep, iPtor).IsSameAs("Gradients")) {
            level = params->GetPreprocessLevel(iStep, iPtor, 0);
            time = params->GetPreprocessTimeHours(iStep, iPtor, 0);
            if (params->GetPredictorCriteria(iStep, iPtor).IsSameAs("S1") ||
                params->GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1")) {
                wxLogError(_("The criteria value has not been changed after the gradient preprocessing."));
                return false;
            }
        } else if (params->GetPreprocessMethod(iStep, iPtor).IsSameAs("HumidityIndex")) {
            level = params->GetPreprocessLevel(iStep, iPtor, 0);
            time = params->GetPreprocessTimeHours(iStep, iPtor, 0);
        } else if (params->GetPreprocessMethod(iStep, iPtor).IsSameAs("HumidityFlux")) {
            level = params->GetPreprocessLevel(iStep, iPtor, 0);
            time = params->GetPreprocessTimeHours(iStep, iPtor, 0);
        } else if (params->GetPreprocessMethod(iStep, iPtor).IsSameAs("FormerHumidityIndex")) {
            level = params->GetPreprocessLevel(iStep, iPtor, 0);
            time = params->GetPreprocessTimeHours(iStep, iPtor, 0);
        } else {
            level = params->GetPreprocessLevel(iStep, iPtor, 0);
            time = params->GetPreprocessTimeHours(iStep, iPtor, 0);
        }

        // Get level and hour indices
        if (!preloadLevels.empty()) {
            iLevel = asTools::SortedArraySearch(&preloadLevels[0], &preloadLevels[preloadLevels.size() - 1], level);
        }
        if (!preloadTimeHours.empty()) {
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

        level = params->GetPredictorLevel(iStep, iPtor);
        time = params->GetPredictorTimeHours(iStep, iPtor);
        iLevel = asTools::SortedArraySearch(&preloadLevels[0], &preloadLevels[preloadLevels.size() - 1], level);
        iHour = asTools::SortedArraySearch(&preloadTimeHours[0], &preloadTimeHours[preloadTimeHours.size() - 1], time);
    }
    if (iLevel < 0 || iHour < 0) {
        wxLogError(_("An unexpected error occurred."));
        return false;
    }

    // Copy the data
    wxASSERT(m_preloadedArchive[iStep][iPtor][iPre][iLevel][iHour]);
    auto *desiredPredictor = new asPredictorArch(*m_preloadedArchive[iStep][iPtor][iPre][iLevel][iHour]);

    // Area object instantiation
    asGeoAreaCompositeGrid *desiredArea = asGeoAreaCompositeGrid::GetInstance(params->GetPredictorGridType(iStep, iPtor),
                                                                              params->GetPredictorXmin(iStep, iPtor),
                                                                              params->GetPredictorXptsnb(iStep, iPtor),
                                                                              params->GetPredictorXstep(iStep, iPtor),
                                                                              params->GetPredictorYmin(iStep, iPtor),
                                                                              params->GetPredictorYptsnb(iStep, iPtor),
                                                                              params->GetPredictorYstep(iStep, iPtor),
                                                                              params->GetPredictorLevel(iStep, iPtor),
                                                                              asNONE,
                                                                              params->GetPredictorFlatAllowed(iStep,
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

        auto *newPredictor = new asPredictorArch(*predictorsPreprocess[0]);
        if (!Preprocess(predictorsPreprocess, "Gradients", newPredictor)) {
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

bool asMethodStandard::ExtractArchiveDataWithoutPreprocessing(std::vector<asPredictor *> &predictors,
                                                              asParameters *params, int iStep, int iPtor,
                                                              double timeStartData, double timeEndData)
{
    // Date array object instantiation for the data loading. The array has the same length than timeArrayArchive, and the predictor dates are aligned with the target dates, but the dates are not the same.
    double ptorStart = timeStartData - params->GetTimeShiftDays() + params->GetPredictorTimeHours(iStep, iPtor) / 24.0;
    double ptorEnd = timeEndData - params->GetTimeShiftDays() + params->GetPredictorTimeHours(iStep, iPtor) / 24.0;
    asTimeArray timeArray(ptorStart, ptorEnd, params->GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
    timeArray.Init();

    // Loading the datasets information
    asPredictorArch *predictor = asPredictorArch::GetInstance(params->GetPredictorDatasetId(iStep, iPtor),
                                                              params->GetPredictorDataId(iStep, iPtor),
                                                              m_predictorDataDir);
    if (!predictor) {
        return false;
    }

    // Select the number of members for ensemble data.
    if (predictor->IsEnsemble()) {
        predictor->SelectMembers(params->GetPredictorMembersNb(iStep, iPtor));
    }

    // Area object instantiation
    asGeoAreaCompositeGrid *area = asGeoAreaCompositeGrid::GetInstance(params->GetPredictorGridType(iStep, iPtor),
                                                                       params->GetPredictorXmin(iStep, iPtor),
                                                                       params->GetPredictorXptsnb(iStep, iPtor),
                                                                       params->GetPredictorXstep(iStep, iPtor),
                                                                       params->GetPredictorYmin(iStep, iPtor),
                                                                       params->GetPredictorYptsnb(iStep, iPtor),
                                                                       params->GetPredictorYstep(iStep, iPtor),
                                                                       params->GetPredictorLevel(iStep, iPtor), asNONE,
                                                                       params->GetPredictorFlatAllowed(iStep, iPtor));
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

bool asMethodStandard::ExtractArchiveDataWithPreprocessing(std::vector<asPredictor *> &predictors,
                                                           asParameters *params, int iStep, int iPtor,
                                                           double timeStartData, double timeEndData)
{
    std::vector<asPredictorArch *> predictorsPreprocess;

    int preprocessSize = params->GetPreprocessSize(iStep, iPtor);

    wxLogVerbose(_("Preprocessing data (%d predictor(s)) while loading."), preprocessSize);

    for (int iPre = 0; iPre < preprocessSize; iPre++) {
        // Date array object instantiation for the data loading. The array has the same length than timeArrayArchive, and the predictor dates are aligned with the target dates, but the dates are not the same.
        double ptorStart = timeStartData - double(params->GetTimeShiftDays()) +
                           params->GetPreprocessTimeHours(iStep, iPtor, iPre) / 24.0;
        double ptorEnd = timeEndData - double(params->GetTimeShiftDays()) +
                         params->GetPreprocessTimeHours(iStep, iPtor, iPre) / 24.0;
        asTimeArray timeArray(ptorStart, ptorEnd, params->GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
        timeArray.Init();

        // Loading the dataset information
        asPredictorArch *predictorPreprocess = asPredictorArch::GetInstance(
                params->GetPreprocessDatasetId(iStep, iPtor, iPre), params->GetPreprocessDataId(iStep, iPtor, iPre),
                m_predictorDataDir);
        if (!predictorPreprocess) {
            Cleanup(predictorsPreprocess);
            return false;
        }

        // Select the number of members for ensemble data.
        if (predictorPreprocess->IsEnsemble()) {
            predictorPreprocess->SelectMembers(params->GetPreprocessMembersNb(iStep, iPtor, iPre));
        }

        // Area object instantiation
        asGeoAreaCompositeGrid *area = asGeoAreaCompositeGrid::GetInstance(params->GetPredictorGridType(iStep, iPtor),
                                                                           params->GetPredictorXmin(iStep, iPtor),
                                                                           params->GetPredictorXptsnb(iStep, iPtor),
                                                                           params->GetPredictorXstep(iStep, iPtor),
                                                                           params->GetPredictorYmin(iStep, iPtor),
                                                                           params->GetPredictorYptsnb(iStep, iPtor),
                                                                           params->GetPredictorYstep(iStep, iPtor),
                                                                           params->GetPreprocessLevel(iStep, iPtor,
                                                                                                     iPre), asNONE,
                                                                           params->GetPredictorFlatAllowed(iStep,
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
    if (params->GetPreprocessMethod(iStep, iPtor).IsSameAs("Gradients") &&
        params->GetPredictorCriteria(iStep, iPtor).IsSameAs("S1")) {
        params->SetPredictorCriteria(iStep, iPtor, "S1grads");
    } else if (params->GetPreprocessMethod(iStep, iPtor).IsSameAs("Gradients") &&
               params->GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1")) {
        params->SetPredictorCriteria(iStep, iPtor, "NS1grads");
    }

    asPredictorArch *predictor = new asPredictorArch(*predictorsPreprocess[0]);
    if (!Preprocess(predictorsPreprocess, params->GetPreprocessMethod(iStep, iPtor), predictor)) {
        wxLogError(_("Data preprocessing failed."));
        Cleanup(predictorsPreprocess);
        wxDELETE(predictor);
        return false;
    }

    Cleanup(predictorsPreprocess);
    predictors.push_back(predictor);

    return true;
}

void asMethodStandard::Cleanup(std::vector<asPredictorArch *> predictorsPreprocess)
{
    if (!predictorsPreprocess.empty()) {
        for (unsigned int i = 0; i < predictorsPreprocess.size(); i++) {
            wxDELETE(predictorsPreprocess[i]);
        }
        predictorsPreprocess.resize(0);
    }
}

void asMethodStandard::Cleanup(std::vector<asPredictor *> predictors)
{
    if (!predictors.empty()) {
        for (unsigned int i = 0; i < predictors.size(); i++) {
            wxDELETE(predictors[i]);
        }
        predictors.resize(0);
    }
}

void asMethodStandard::Cleanup(std::vector<asCriteria *> criteria)
{
    if (!criteria.empty()) {
        for (unsigned int i = 0; i < criteria.size(); i++) {
            wxDELETE(criteria[i]);
        }
        criteria.resize(0);
    }
}

void asMethodStandard::DeletePreloadedArchiveData()
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

bool asMethodStandard::GetRandomValidData(asParameters *params, int iStep, int iPtor, int iPre)
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

    wxASSERT(!levels.empty());
    wxASSERT(!hours.empty());
    wxASSERT(levels.size() == hours.size());

    int randomIndex = asTools::Random(0, levels.size() - 1, 1);
    float newLevel = params->GetPreloadLevels(iStep, iPtor)[levels[randomIndex]];
    double newHour = params->GetPreloadTimeHours(iStep, iPtor)[hours[randomIndex]];

    params->SetPredictorLevel(iStep, iPtor, newLevel);
    params->SetPredictorTimeHours(iStep, iPtor, newHour);

    return true;
}