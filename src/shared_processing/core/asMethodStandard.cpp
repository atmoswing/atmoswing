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

#include "asAreaCompGrid.h"
#include "asCriteria.h"
#include "asParameters.h"
#include "asPredictor.h"
#include "asPreprocessor.h"
#include "asThreadPreloadArchiveData.h"
#include "asTimeArray.h"

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
      m_warnFailedLoadingData(true),
      m_predictandDB(nullptr) {
    ThreadsManager().CritSectionConfig().Enter();
    m_dumpPredictorData = wxFileConfig::Get()->ReadBool("/General/DumpPredictorData", false);
    m_loadFromDumpedData = wxFileConfig::Get()->ReadBool("/General/LoadDumpedData", false);
    ThreadsManager().CritSectionConfig().Leave();
}

asMethodStandard::~asMethodStandard() {
    wxDELETE(m_predictandDB);
}

bool asMethodStandard::Manager() {
    return false;
}

bool asMethodStandard::LoadPredictandDB(const wxString &predictandDBFilePath) {
    wxDELETE(m_predictandDB);

    if (predictandDBFilePath.IsEmpty()) {
        if (m_predictandDBFilePath.IsEmpty()) {
            wxLogError(_("There is no predictand database file selected."));
            return false;
        }

        m_predictandDB = asPredictand::GetInstance(m_predictandDBFilePath);
        if (!m_predictandDB) return false;

        if (!m_predictandDB->Load(m_predictandDBFilePath)) {
            wxLogError(_("Couldn't load the predictand database."));
            return false;
        }
    } else {
        m_predictandDB = asPredictand::GetInstance(predictandDBFilePath);
        if (!m_predictandDB) return false;

        if (!m_predictandDB->Load(predictandDBFilePath)) {
            wxLogError(_("Couldn't load the predictand database."));
            return false;
        }
    }

    if (!m_predictandDB) return false;
    wxASSERT(m_predictandDB);

    return true;
}

void asMethodStandard::Cancel() {
    m_cancel = true;
}

bool asMethodStandard::Preprocess(std::vector<asPredictor *> predictors, const wxString &method, asPredictor *result) {
    std::vector<asPredictor *> ptorsPredictors(predictors.begin(), predictors.end());

    return asPreprocessor::Preprocess(ptorsPredictors, method, result);
}

double asMethodStandard::GetTimeStartArchive(asParameters *params) const {
    return params->GetArchiveStart() + params->GetTimeShiftDays();
}

double asMethodStandard::GetTimeEndArchive(asParameters *params) const {
    return params->GetArchiveEnd() - params->GetTimeSpanDays();
}

void asMethodStandard::InitializePreloadedArchiveDataContainers(asParameters *params) {
    if (m_preloadedArchive.empty()) {
        m_preloadedArchive.resize((long)params->GetStepsNb());
        m_preloadedArchivePointerCopy.resize((long)params->GetStepsNb());
        for (int iStep = 0; iStep < params->GetStepsNb(); iStep++) {
            m_preloadedArchive[iStep].resize((long)params->GetPredictorsNb(iStep));
            m_preloadedArchivePointerCopy[iStep].resize((long)params->GetPredictorsNb(iStep));

            for (int iPtor = 0; iPtor < params->GetPredictorsNb(iStep); iPtor++) {
                vwxs preloadDataIds = params->GetPreloadDataIds(iStep, iPtor);
                vf preloadLevels = params->GetPreloadLevels(iStep, iPtor);
                vd preloadHours = params->GetPreloadHours(iStep, iPtor);

                long preloadDataIdsSize = wxMax(preloadDataIds.size(), 1);
                long preloadLevelsSize = wxMax(preloadLevels.size(), 1);
                long preloadHoursSize = wxMax(preloadHours.size(), 1);

                m_preloadedArchivePointerCopy[iStep][iPtor].resize(preloadDataIdsSize);
                m_preloadedArchive[iStep][iPtor].resize(preloadDataIdsSize);

                for (int iDat = 0; iDat < preloadDataIdsSize; iDat++) {
                    m_preloadedArchivePointerCopy[iStep][iPtor][iDat] = false;
                    m_preloadedArchive[iStep][iPtor][iDat].resize(preloadLevelsSize);

                    // Load data for every level and every hour
                    for (int iLevel = 0; iLevel < preloadLevelsSize; iLevel++) {
                        m_preloadedArchive[iStep][iPtor][iDat][iLevel].resize(preloadHoursSize);
                        for (int iHour = 0; iHour < preloadHoursSize; iHour++) {
                            m_preloadedArchive[iStep][iPtor][iDat][iLevel][iHour] = nullptr;
                        }
                    }
                }
            }
        }
    }
}

bool asMethodStandard::PreloadArchiveData(asParameters *params) {
    if (!m_preloaded) {
        // Set preload to true here, so cleanup is made in case of exceptions.
        m_preloaded = true;

        InitializePreloadedArchiveDataContainers(params);

        if (!ProceedToArchiveDataPreloading(params)) return false;

        if (!CheckArchiveDataIsPreloaded(params)) return false;
    }

    return true;
}

bool asMethodStandard::ProceedToArchiveDataPreloading(asParameters *params) {
    ThreadsManager().CritSectionConfig().Enter();
    bool parallelDataLoad = wxFileConfig::Get()->ReadBool("/General/ParallelDataLoad", false);
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

bool asMethodStandard::CheckArchiveDataIsPreloaded(const asParameters *params) const {
    for (int iStep = 0; iStep < params->GetStepsNb(); iStep++) {
        for (int iPtor = 0; iPtor < params->GetPredictorsNb(iStep); iPtor++) {
            if (params->NeedsPreloading(iStep, iPtor)) {
                if (!params->NeedsPreprocessing(iStep, iPtor)) {
                    for (int iPre = 0; iPre < params->GetPredictorDataIdNb(iStep, iPtor); iPre++) {
                        if (!HasPreloadedArchiveData(iStep, iPtor, iPre)) {
                            wxLogError(_("No data was preloaded for step %d, predictor %d and num %d."), iStep, iPtor,
                                       iPre);
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

bool asMethodStandard::HasPreloadedArchiveData(int iStep, int iPtor) const {
    for (const auto &datPre : m_preloadedArchive[iStep][iPtor]) {
        for (const auto &datLevel : datPre) {
            for (auto datHour : datLevel) {
                if (datHour != nullptr) {
                    return true;
                }
            }
        }
    }

    return false;
}

bool asMethodStandard::HasPreloadedArchiveData(int iStep, int iPtor, int iPre) const {
    for (const auto &datLevel : m_preloadedArchive[iStep][iPtor][iPre]) {
        for (auto datHour : datLevel) {
            if (datHour != nullptr) {
                return true;
            }
        }
    }

    return false;
}

bool asMethodStandard::PointersArchiveDataShared(asParameters *params, int iStep, int iPtor, int iPre) {
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
                if (!params->GetPredictorDatasetId(iStep, iPtor)
                         .IsSameAs(params->GetPredictorDatasetId(prev_step, prev_ptor), false))
                    share = false;
                if (!params->GetPredictorGridType(iStep, iPtor)
                         .IsSameAs(params->GetPredictorGridType(prev_step, prev_ptor), false))
                    share = false;
            } else {
                if (!params->GetPreprocessMethod(iStep, iPtor)
                         .IsSameAs(params->GetPreprocessMethod(prev_step, prev_ptor), false))
                    share = false;
                if (params->GetPreprocessSize(iStep, iPtor) != params->GetPreprocessSize(prev_step, prev_ptor)) {
                    share = false;
                } else {
                    int preprocessSize = params->GetPreprocessSize(iStep, iPtor);

                    for (int i = 0; i < preprocessSize; i++) {
                        if (!params->GetPreprocessDatasetId(iStep, iPtor, i)
                                 .IsSameAs(params->GetPreprocessDatasetId(prev_step, prev_ptor, i), false))
                            share = false;
                        if (!params->GetPreprocessDataId(iStep, iPtor, i)
                                 .IsSameAs(params->GetPreprocessDataId(prev_step, prev_ptor, i), false))
                            share = false;
                    }
                }
            }

            if (params->GetPreloadXmin(iStep, iPtor) != params->GetPreloadXmin(prev_step, prev_ptor)) share = false;
            if (params->GetPreloadXptsnb(iStep, iPtor) != params->GetPreloadXptsnb(prev_step, prev_ptor)) share = false;
            if (params->GetPredictorXstep(iStep, iPtor) != params->GetPredictorXstep(prev_step, prev_ptor))
                share = false;
            if (params->GetPreloadYmin(iStep, iPtor) != params->GetPreloadYmin(prev_step, prev_ptor)) share = false;
            if (params->GetPreloadYptsnb(iStep, iPtor) != params->GetPreloadYptsnb(prev_step, prev_ptor)) share = false;
            if (params->GetPredictorYstep(iStep, iPtor) != params->GetPredictorYstep(prev_step, prev_ptor))
                share = false;
            if (params->GetPredictorFlatAllowed(iStep, iPtor) != params->GetPredictorFlatAllowed(prev_step, prev_ptor))
                share = false;

            vf levels1 = params->GetPreloadLevels(iStep, iPtor);
            vf levels2 = params->GetPreloadLevels(prev_step, prev_ptor);
            if (levels1.size() != levels2.size()) {
                share = false;
            } else {
                for (int i = 0; i < levels1.size(); i++) {
                    if (levels1[i] != levels2[i]) share = false;
                }
            }

            vd hours1 = params->GetPreloadHours(iStep, iPtor);
            vd hours2 = params->GetPreloadHours(prev_step, prev_ptor);
            if (hours1.size() != hours2.size()) {
                share = false;
            } else {
                for (int i = 0; i < hours1.size(); i++) {
                    if (hours1[i] != hours2[i]) share = false;
                }
            }

            bool dataIdFound = false;
            vwxs preloadDataIds = params->GetPreloadDataIds(iStep, iPtor);
            vwxs preloadDataIdsPrev = params->GetPreloadDataIds(prev_step, prev_ptor);
            for (int i = 0; i < preloadDataIdsPrev.size(); i++) {
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

            if (share) break;
        }

        if (share) break;
    }

    if (share) {
        wxLogVerbose(_("Share data pointer"));

        vf preloadLevels = params->GetPreloadLevels(iStep, iPtor);
        vd preloadHours = params->GetPreloadHours(iStep, iPtor);
        wxASSERT(!preloadLevels.empty());
        wxASSERT(!preloadHours.empty());

        m_preloadedArchivePointerCopy[iStep][iPtor][iPre] = true;

        wxASSERT(m_preloadedArchive[prev_step].size() > prev_ptor);
        wxASSERT(m_preloadedArchive[prev_step][prev_ptor].size() > prev_dat);
        wxASSERT(m_preloadedArchive[prev_step][prev_ptor][prev_dat].size() == preloadLevels.size());

        // Load data for every level and every hour
        for (int iLevel = 0; iLevel < preloadLevels.size(); iLevel++) {
            wxASSERT(m_preloadedArchive[prev_step][prev_ptor][prev_dat][iLevel].size() == preloadHours.size());
            for (int iHour = 0; iHour < preloadHours.size(); iHour++) {
                // Copy pointer
                m_preloadedArchive[iStep][iPtor][iPre][iLevel][iHour] =
                    m_preloadedArchive[prev_step][prev_ptor][prev_dat][iLevel][iHour];
            }
        }

        params->SetPreloadYptsnb(iStep, iPtor, params->GetPreloadYptsnb(prev_step, prev_ptor));

        return true;
    }

    return false;
}

bool asMethodStandard::PreloadArchiveDataWithoutPreprocessing(asParameters *params, int iStep, int iPtor, int iDat) {
    wxLogVerbose(_("Preloading data for predictor %d of step %d."), iPtor, iStep);

    double timeStartData = GetEffectiveArchiveDataStart(params);
    double timeEndData = GetEffectiveArchiveDataEnd(params);

    vwxs preloadDataIds = params->GetPreloadDataIds(iStep, iPtor);
    vf preloadLevels = params->GetPreloadLevels(iStep, iPtor);
    vd preloadHours = params->GetPreloadHours(iStep, iPtor);
    wxASSERT(preloadDataIds.size() > iDat);
    wxASSERT(!preloadLevels.empty());
    wxASSERT(!preloadHours.empty());

    int predictorSize = 0;

    // Load data for every level and every hour
    for (int iLevel = 0; iLevel < preloadLevels.size(); iLevel++) {
        for (int iHour = 0; iHour < preloadHours.size(); iHour++) {
            // Loading the dataset information
            asPredictor *predictor = asPredictor::GetInstance(params->GetPredictorDatasetId(iStep, iPtor),
                                                              preloadDataIds[iDat], m_predictorDataDir);
            if (!predictor) {
                return false;
            }

            // Set warning option
            predictor->SetWarnMissingLevels(m_warnFailedLoadingData);

            // Select the number of members for ensemble data.
            if (predictor->IsEnsemble()) {
                predictor->SelectMembers(params->GetPredictorMembersNb(iStep, iPtor));
            }

            // Date array object instantiation for data loading.
            double ptorStart = timeStartData + preloadHours[iHour] / 24.0;
            double ptorEnd = timeEndData + preloadHours[iHour] / 24.0;

            asTimeArray timeArray(ptorStart, ptorEnd, params->GetAnalogsTimeStepHours(),
                                  params->GetTimeArrayAnalogsMode());
            timeArray.Init();

            double yMax =
                params->GetPreloadYmin(iStep, iPtor) +
                params->GetPredictorYstep(iStep, iPtor) * (double)(params->GetPreloadYptsnb(iStep, iPtor) - 1);
            if (yMax > 90) {
                double diff = yMax - 90;
                auto removePts = (int)asRound(diff / params->GetPredictorYstep(iStep, iPtor));
                params->SetPreloadYptsnb(iStep, iPtor, params->GetPreloadYptsnb(iStep, iPtor) - removePts);
                wxLogVerbose(_("Adapt Y axis extent according to the maximum allowed (from %.3f to %.3f)."), yMax,
                             yMax - diff);
                wxLogVerbose(_("Remove %d points (%.3f-%.3f)/%.3f."), removePts, yMax, 90,
                             params->GetPredictorYstep(iStep, iPtor));
            }

            wxASSERT(params->GetPreloadXptsnb(iStep, iPtor) > 0);
            wxASSERT(params->GetPreloadYptsnb(iStep, iPtor) > 0);

            // Area object instantiation
            wxString gridType = params->GetPredictorGridType(iStep, iPtor);
            double xMin = params->GetPreloadXmin(iStep, iPtor);
            int xPtsNb = params->GetPreloadXptsnb(iStep, iPtor);
            double xStep = params->GetPredictorXstep(iStep, iPtor);
            double yMin = params->GetPreloadYmin(iStep, iPtor);
            int yPtsNb = params->GetPreloadYptsnb(iStep, iPtor);
            double yStep = params->GetPredictorYstep(iStep, iPtor);
            int flatAllowed = params->GetPredictorFlatAllowed(iStep, iPtor);
            bool isLatLon = asPredictor::IsLatLon(params->GetPredictorDatasetId(iStep, iPtor));
            asAreaCompGrid *area =
                asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep, flatAllowed, isLatLon);
            wxASSERT(area);
            area->AllowResizeFromData();

            // Set standardize option
            if (params->GetStandardize(iStep, iPtor)) {
                predictor->SetStandardize(true);
            }

            if (m_dumpPredictorData) {
                predictor->SetLevel(preloadLevels[iLevel]);
                predictor->SetTimeArray(timeArray.GetTimeArray());
                if (predictor->DumpFileExists()) {
                    predictor->SetWasDumped(true);
                    m_preloadedArchive[iStep][iPtor][iDat][iLevel][iHour] = predictor;
                    continue;
                }
            }

            if (m_loadFromDumpedData) {
                predictor->SetLevel(preloadLevels[iLevel]);
                predictor->SetTimeArray(timeArray.GetTimeArray());
                if (predictor->DumpFileExists()) {
                    if (!predictor->LoadDumpedData()) {
                        wxLogError(_("Failed loading dumped data."));
                        return false;
                    }
                    m_preloadedArchive[iStep][iPtor][iDat][iLevel][iHour] = predictor;
                    continue;
                }
            }

            // Data loading
            wxLogVerbose(_("Loading %s data for level %d, %gh."), preloadDataIds[iDat], (int)preloadLevels[iLevel],
                         preloadHours[iHour]);
            try {
                if (!predictor->Load(area, timeArray, preloadLevels[iLevel])) {
                    if (m_warnFailedLoadingData) {
                        wxLogWarning(_("The data (%s for level %d, at %gh) could not be loaded."), preloadDataIds[iDat],
                                     (int)preloadLevels[iLevel], preloadHours[iHour]);
                    } else {
                        wxLogVerbose(_("The data (%s for level %d, at %gh) could not be loaded."), preloadDataIds[iDat],
                                     (int)preloadLevels[iLevel], preloadHours[iHour]);
                    }
                    wxDELETE(area);
                    wxDELETE(predictor);
                    continue;  // The requested data can be missing (e.g. level not available).
                }
            } catch (std::bad_alloc &ba) {
                wxString msg(ba.what(), wxConvUTF8);
                wxLogError(_("Bad allocation caught during data preloading: %s"), msg);
                wxDELETE(area);
                wxDELETE(predictor);
                return false;

            } catch (std::exception &e) {
                wxString msg(e.what(), wxConvUTF8);
                wxLogError(_("Exception caught during data preloading: %s"), msg);
                wxDELETE(area);
                wxDELETE(predictor);
                return false;
            }
            wxLogVerbose(_("Data loaded."));
            wxDELETE(area);

            if (predictorSize > 0 && predictorSize != predictor->GetData().size()) {
                wxLogError(_("The preloaded data has a different length than other data series: %d != %d"),
                           (int)predictor->GetData().size(), predictorSize);
                wxDELETE(predictor);
                return false;
            } else {
                predictorSize = predictor->GetData().size();
            }

            // Standardize data
            if (params->GetStandardize(iStep, iPtor) &&
                !predictor->StandardizeData(params->GetStandardizeMean(iStep, iPtor),
                                            params->GetStandardizeSd(iStep, iPtor))) {
                wxLogError(_("Data standardisation has failed."));
                wxFAIL;
                return false;
            }

            if (m_dumpPredictorData || m_loadFromDumpedData) {
                // Dumped files do not exist here.
                if (!predictor->SaveDumpFile()) {
                    return false;
                }
                wxLogMessage(_("File dumbed for %s (level %d, %gh)."), preloadDataIds[iDat], (int)preloadLevels[iLevel],
                             preloadHours[iHour]);
                if (m_dumpPredictorData) {
                    predictor->DumpData();
                }
            }

            m_preloadedArchive[iStep][iPtor][iDat][iLevel][iHour] = predictor;
        }
    }

    return true;
}

bool asMethodStandard::PreloadArchiveDataWithPreprocessing(asParameters *params, int iStep, int iPtor) {
    wxLogVerbose(_("Preloading data for predictor preprocessed %d of step %d."), iPtor, iStep);

    double timeStartData = GetEffectiveArchiveDataStart(params);
    double timeEndData = GetEffectiveArchiveDataEnd(params);

    // Check the preprocessing method
    wxString method = params->GetPreprocessMethod(iStep, iPtor);

    // Get the number of sub predictors
    int preprocessSize = params->GetPreprocessSize(iStep, iPtor);

    // Levels and time arrays
    vf preloadLevels = params->GetPreloadLevels(iStep, iPtor);
    vd preloadHours = params->GetPreloadHours(iStep, iPtor);

    // Check on which variable to loop
    long preloadLevelsSize = preloadLevels.size();
    long preloadHoursSize = preloadHours.size();
    bool loopOnLevels = true;
    bool loopOnHours = true;

    if (method.IsSameAs("Multiplication") || method.IsSameAs("Multiply") || method.IsSameAs("Addition") ||
        method.IsSameAs("Average")) {
        loopOnLevels = false;
        loopOnHours = false;
        preloadLevelsSize = 1;
        preloadHoursSize = 1;
    } else if (params->NeedsGradientPreprocessing(iStep, iPtor) || method.IsSameAs("HumidityIndex") ||
               method.IsSameAs("HumidityFlux") || method.IsSameAs("FormerHumidityIndex")) {
        if (preloadLevelsSize == 0) {
            loopOnLevels = false;
            preloadLevelsSize = 1;
        }
        if (preloadHoursSize == 0) {
            loopOnHours = false;
            preloadHoursSize = 1;
        }
    } else {
        wxLogError(_("Preprocessing method unknown in PreloadArchiveDataWithPreprocessing."));
        return false;
    }

    wxLogVerbose(_("Preprocessing data (%d predictor(s)) while loading."), preprocessSize);

    // Load data for every level and every hour
    for (int iLevel = 0; iLevel < preloadLevelsSize; iLevel++) {
        for (int iHour = 0; iHour < preloadHoursSize; iHour++) {
            std::vector<asPredictor *> predictorsPreprocess;

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
                double hours;
                if (loopOnHours) {
                    hours = preloadHours[iHour];
                } else {
                    hours = params->GetPreprocessHour(iStep, iPtor, iPre);
                }

                // Correct according to the method
                if (params->NeedsGradientPreprocessing(iStep, iPtor)) {
                    // Nothing to change
                } else if (method.IsSameAs("HumidityIndex")) {
                    if (iPre == 1) level = 0;  // pr_wtr
                } else if (method.IsSameAs("HumidityFlux")) {
                    if (iPre == 2) level = 0;  // pr_wtr
                } else if (method.IsSameAs("FormerHumidityIndex")) {
                    if (iPre == 2) level = 0;  // pr_wtr
                    if (iPre == 3) level = 0;  // pr_wtr
                    if (iPre == 0) hours = preloadHours[0];
                    if (iPre == 1) hours = preloadHours[1];
                    if (iPre == 2) hours = preloadHours[0];
                    if (iPre == 3) hours = preloadHours[1];
                }

                // Date array object instantiation for data loading.
                double ptorStart = timeStartData + hours / 24.0;
                double ptorEnd = timeEndData + hours / 24.0;
                asTimeArray timeArray(ptorStart, ptorEnd, params->GetAnalogsTimeStepHours(),
                                      params->GetTimeArrayAnalogsMode());
                timeArray.Init();

                // Loading the datasets information
                asPredictor *predictorPreprocess =
                    asPredictor::GetInstance(params->GetPreprocessDatasetId(iStep, iPtor, iPre),
                                             params->GetPreprocessDataId(iStep, iPtor, iPre), m_predictorDataDir);
                if (!predictorPreprocess) {
                    Cleanup(predictorsPreprocess);
                    return false;
                }

                // Set warning option
                predictorPreprocess->SetWarnMissingLevels(m_warnFailedLoadingData);

                // Select the number of members for ensemble data.
                if (predictorPreprocess->IsEnsemble()) {
                    predictorPreprocess->SelectMembers(params->GetPreprocessMembersNb(iStep, iPtor, iPre));
                }

                double yMax = params->GetPreloadYmin(iStep, iPtor) +
                    params->GetPredictorYstep(iStep, iPtor) * double(params->GetPreloadYptsnb(iStep, iPtor) - 1);

                if (predictorPreprocess->IsLatLon() && yMax > 90) {
                    double diff = yMax - 90;
                    int removePts = (int)asRound(diff / params->GetPredictorYstep(iStep, iPtor));
                    params->SetPreloadYptsnb(iStep, iPtor, params->GetPreloadYptsnb(iStep, iPtor) - removePts);
                    wxLogVerbose(_("Adapt Y axis extent according to the maximum allowed (from %.2f to %.2f)."), yMax,
                                 yMax - diff);
                }

                // Area object instantiation
                wxString gridType = params->GetPredictorGridType(iStep, iPtor);
                double xMin = params->GetPreloadXmin(iStep, iPtor);
                int xPtsNb = params->GetPreloadXptsnb(iStep, iPtor);
                double xStep = params->GetPredictorXstep(iStep, iPtor);
                double yMin = params->GetPreloadYmin(iStep, iPtor);
                int yPtsNb = params->GetPreloadYptsnb(iStep, iPtor);
                double yStep = params->GetPredictorYstep(iStep, iPtor);
                int flatAllowed = params->GetPredictorFlatAllowed(iStep, iPtor);
                bool isLatLon = asPredictor::IsLatLon(params->GetPredictorDatasetId(iStep, iPtor));
                asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep,
                                                                   flatAllowed, isLatLon);
                wxASSERT(area);
                area->AllowResizeFromData();

                // Data loading
                wxLogVerbose(_("Loading %s data for level %d, %gh."), params->GetPreprocessDataId(iStep, iPtor, iPre),
                             (int)level, hours);
                if (!predictorPreprocess->Load(area, timeArray, level)) {
                    wxLogError(_("The data could not be loaded."));
                    wxDELETE(area);
                    wxDELETE(predictorPreprocess);
                    return false;
                }
                wxDELETE(area);
                predictorsPreprocess.push_back(predictorPreprocess);
            }

            wxLogVerbose(_("Preprocessing data."));
            auto *predictor = new asPredictor(*predictorsPreprocess[0]);

            try {
                if (!Preprocess(predictorsPreprocess, params->GetPreprocessMethod(iStep, iPtor), predictor)) {
                    wxLogError(_("Data preprocessing failed."));
                    wxDELETE(predictor);
                    Cleanup(predictorsPreprocess);
                    return false;
                }

                // Standardize data
                if (params->GetStandardize(iStep, iPtor) &&
                    !predictor->StandardizeData(params->GetStandardizeMean(iStep, iPtor), params->GetStandardizeSd(iStep, iPtor))) {
                    wxLogError(_("Data standardisation has failed."));
                    wxFAIL;
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
                wxLogError(_("Exception caught during data preprocessing: %s"), msg);
                wxDELETE(predictor);
                Cleanup(predictorsPreprocess);
                return false;
            }
            Cleanup(predictorsPreprocess);
            wxLogVerbose(_("Preprocessing over."));
        }
    }

    // Fix the criteria if S1
    params->FixCriteriaIfGradientsPreprocessed(iStep, iPtor);

    return true;
}

bool asMethodStandard::LoadArchiveData(std::vector<asPredictor *> &predictors, asParameters *params, int iStep,
                                       double timeStartData, double timeEndData) {
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
                    if (!ExtractArchiveData(predictors, params, iStep, iPtor, timeStartData, timeEndData)) {
                        return false;
                    }
                } else {
                    if (!PreprocessArchiveData(predictors, params, iStep, iPtor, timeStartData, timeEndData)) {
                        return false;
                    }
                }

                wxLogVerbose(_("Data loaded"));
            }
        }
    } catch (std::bad_alloc &ba) {
        wxString msg(ba.what(), wxConvUTF8);
        wxLogError(_("Bad allocation caught during data loading: %s"), msg);
        return false;
    } catch (std::exception &e) {
        wxString msg(e.what(), wxConvUTF8);
        wxLogError(_("Exception caught during data loading: %s"), msg);
        return false;
    }

    return true;
}

bool asMethodStandard::ExtractPreloadedArchiveData(std::vector<asPredictor *> &predictors, asParameters *params,
                                                   int iStep, int iPtor) {
    wxLogVerbose(_("Using preloaded data."));

    bool doPreprocessGradients = false;

    // Get preload arrays
    vf preloadLevels = params->GetPreloadLevels(iStep, iPtor);
    vd preloadHours = params->GetPreloadHours(iStep, iPtor);
    float level;
    double hour;
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
        wxASSERT(!preloadHours.empty());

        level = params->GetPredictorLevel(iStep, iPtor);
        hour = params->GetPredictorHour(iStep, iPtor);

        // Get level and hour indices
        iLevel = asFind(&preloadLevels[0], &preloadLevels[preloadLevels.size() - 1], level);
        iHour = asFind(&preloadHours[0], &preloadHours[preloadHours.size() - 1], hour);

        // Force gradients preprocessing anyway.
        params->ForceUsingGradientsPreprocessing(iStep, iPtor);
        if (params->IsCriteriaUsingGradients(iStep, iPtor)) {
            doPreprocessGradients = true;
        }
    } else {
        // Correct according to the method
        if (params->NeedsGradientPreprocessing(iStep, iPtor)) {
            level = params->GetPreprocessLevel(iStep, iPtor, 0);
            hour = params->GetPreprocessHour(iStep, iPtor, 0);
            if (!params->IsCriteriaUsingGradients(iStep, iPtor)) {
                wxLogError(_("The criteria value has not been changed after the gradient preprocessing."));
                return false;
            }
        } else if (params->GetPreprocessMethod(iStep, iPtor).IsSameAs("HumidityIndex")) {
            level = params->GetPreprocessLevel(iStep, iPtor, 0);
            hour = params->GetPreprocessHour(iStep, iPtor, 0);
        } else if (params->GetPreprocessMethod(iStep, iPtor).IsSameAs("HumidityFlux")) {
            level = params->GetPreprocessLevel(iStep, iPtor, 0);
            hour = params->GetPreprocessHour(iStep, iPtor, 0);
        } else if (params->GetPreprocessMethod(iStep, iPtor).IsSameAs("FormerHumidityIndex")) {
            level = params->GetPreprocessLevel(iStep, iPtor, 0);
            hour = params->GetPreprocessHour(iStep, iPtor, 0);
        } else {
            level = params->GetPreprocessLevel(iStep, iPtor, 0);
            hour = params->GetPreprocessHour(iStep, iPtor, 0);
        }

        // Get level and hour indices
        if (!preloadLevels.empty()) {
            iLevel = asFind(&preloadLevels[0], &preloadLevels[preloadLevels.size() - 1], level);
        }
        if (!preloadHours.empty()) {
            iHour = asFind(&preloadHours[0], &preloadHours[preloadHours.size() - 1], hour);
        }
    }

    // Check indices
    if (iLevel == asNOT_FOUND || iLevel == asOUT_OF_RANGE) {
        wxLogError(_("The level (%f) could not be found in the preloaded data."), level);
        return false;
    }
    if (iHour == asNOT_FOUND || iHour == asOUT_OF_RANGE) {
        wxLogError(_("The hour (%d) could not be found in the preloaded data."), (int)hour);
        return false;
    }

    // Get data on the desired domain
    wxASSERT(iStep < m_preloadedArchive.size());
    wxASSERT(iPtor < m_preloadedArchive[iStep].size());
    wxASSERT(iPre < m_preloadedArchive[iStep][iPtor].size());
    wxASSERT(iLevel < m_preloadedArchive[iStep][iPtor][iPre].size());
    wxASSERT(iHour < m_preloadedArchive[iStep][iPtor][iPre][iLevel].size());
    if (!m_preloadedArchive[iStep][iPtor][iPre][iLevel][iHour]) {
        if (!GetRandomLevelValidData(params, iStep, iPtor, iPre, iHour)) {
            if (!GetRandomValidData(params, iStep, iPtor, iPre)) {
                wxLogError(_("The pointer to preloaded data is null."));
                return false;
            }
        }

        level = params->GetPredictorLevel(iStep, iPtor);
        hour = params->GetPredictorHour(iStep, iPtor);
        iLevel = asFind(&preloadLevels[0], &preloadLevels[preloadLevels.size() - 1], level);
        iHour = asFind(&preloadHours[0], &preloadHours[preloadHours.size() - 1], hour);
    }
    if (iLevel < 0 || iHour < 0) {
        wxLogError(_("An unexpected error occurred."));
        return false;
    }

    // Copy the data
    wxASSERT(m_preloadedArchive[iStep][iPtor][iPre][iLevel][iHour]);
    auto *desiredPredictor = new asPredictor(*m_preloadedArchive[iStep][iPtor][iPre][iLevel][iHour]);

    // Load dumped data
    if (m_dumpPredictorData) {
        if (desiredPredictor->WasDumped()) {
            if (!desiredPredictor->LoadDumpedData()) {
                wxLogError(_("Failed loading dumped data."));
                return false;
            }
        }
    }

    // Check minimum number of points
    asCriteria *criterion = asCriteria::GetInstance(params->GetPredictorCriteria(iStep, iPtor));
    if (params->GetPredictorXptsnb(iStep, iPtor) < criterion->GetMinPointsNb()) {
        params->SetPredictorXptsnb(iStep, iPtor, criterion->GetMinPointsNb());
    }
    if (params->GetPredictorYptsnb(iStep, iPtor) < criterion->GetMinPointsNb()) {
        params->SetPredictorYptsnb(iStep, iPtor, criterion->GetMinPointsNb());
    }
    wxDELETE(criterion);

    // Area object instantiation
    asAreaCompGrid *desiredArea = asAreaCompGrid::GetInstance(params, iStep, iPtor);

    wxASSERT(desiredArea);

    // Check area with data availability
    double xMinShift = 0;
    if (desiredArea->GetXmin() > desiredPredictor->GetXmax()) {
        xMinShift = -360;
        if (desiredArea->GetXmin() + xMinShift > desiredPredictor->GetXmax()) {
            wxLogError(_("An unexpected error occurred."));
            return false;
        }
    }
    if (desiredArea->GetXmin() < desiredPredictor->GetXmin()) {
        xMinShift = 360;
        if (desiredArea->GetXmin() + xMinShift < desiredPredictor->GetXmin()) {
            wxLogError(_("An unexpected error occurred."));
            return false;
        }
    }
    double xMaxShift = xMinShift;
    if (desiredArea->GetXmin() > desiredArea->GetXmax()) {
        xMaxShift += 360;
    }
    if (desiredArea->GetXmax() + xMaxShift > desiredPredictor->GetXmax()) {
        a1d lonAxis = desiredPredictor->GetLonAxis();
        int indexXmin = asFindClosest(&lonAxis[0], &lonAxis[lonAxis.size() - 1], desiredArea->GetXmin() + xMinShift);
        int indexXmax = lonAxis.size() - 1;
        wxASSERT(indexXmin >= 0);
        params->SetPredictorXptsnb(iStep, iPtor, indexXmax - indexXmin + 1);
        wxDELETE(desiredArea);
        desiredArea = asAreaCompGrid::GetInstance(params, iStep, iPtor);
    }
    if (desiredArea->GetYmax() > desiredPredictor->GetYmax()) {
        a1d latAxis = desiredPredictor->GetLatAxis();
        int indexYmin = asFindClosest(&latAxis[0], &latAxis[latAxis.size() - 1], desiredArea->GetYmin());
        int indexYmax = asFindClosest(&latAxis[0], &latAxis[latAxis.size() - 1], desiredPredictor->GetYmax());
        params->SetPredictorYptsnb(iStep, iPtor, std::abs(indexYmax - indexYmin) + 1);
        wxDELETE(desiredArea);
        desiredArea = asAreaCompGrid::GetInstance(params, iStep, iPtor);
    }

    if (!desiredPredictor->ClipToArea(desiredArea)) {
        wxLogError(_("The data could not be extracted (iStep = %d, iPtor = %d, iPre = %d, iLevel = %d, iHour = %d)."),
                   iStep, iPtor, iPre, iLevel, iHour);
        wxDELETE(desiredArea);
        wxDELETE(desiredPredictor);
        return false;
    }
    wxDELETE(desiredArea);

    if (doPreprocessGradients) {
        std::vector<asPredictor *> predictorsPreprocess;
        predictorsPreprocess.push_back(desiredPredictor);

        auto *newPredictor = new asPredictor(*predictorsPreprocess[0]);
        if (!Preprocess(predictorsPreprocess, params->GetPreprocessMethod(iStep, iPtor), newPredictor)) {
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

bool asMethodStandard::ExtractArchiveData(std::vector<asPredictor *> &predictors, asParameters *params, int iStep,
                                          int iPtor, double timeStartData, double timeEndData) {
    // Date array object instantiation for the data loading.
    double ptorStart = timeStartData + params->GetPredictorTimeAsDays(iStep, iPtor);
    double ptorEnd = timeEndData + params->GetPredictorTimeAsDays(iStep, iPtor);
    asTimeArray timeArray(ptorStart, ptorEnd, params->GetAnalogsTimeStepHours(), params->GetTimeArrayAnalogsMode());
    timeArray.Init();

    // Force gradients preprocessing anyway.
    params->ForceUsingGradientsPreprocessing(iStep, iPtor);

    // Loading the datasets information
    asPredictor *predictor = asPredictor::GetInstance(params->GetPredictorDatasetId(iStep, iPtor),
                                                      params->GetPredictorDataId(iStep, iPtor), m_predictorDataDir);
    if (!predictor) {
        return false;
    }

    // Set warning option
    predictor->SetWarnMissingLevels(m_warnFailedLoadingData);

    // Select the number of members for ensemble data.
    if (predictor->IsEnsemble()) {
        predictor->SelectMembers(params->GetPredictorMembersNb(iStep, iPtor));
    }

    // Area object instantiation
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(params, iStep, iPtor);
    wxASSERT(area);

    // Data loading
    if (!predictor->Load(area, timeArray, params->GetPredictorLevel(iStep, iPtor))) {
        wxLogError(_("The data could not be loaded."));
        wxDELETE(area);
        wxDELETE(predictor);
        return false;
    }
    wxDELETE(area);

    // Standardize data
    if (params->GetStandardize(iStep, iPtor) &&
        !predictor->StandardizeData(params->GetStandardizeMean(iStep, iPtor), params->GetStandardizeSd(iStep, iPtor))) {
        wxLogError(_("Data standardisation has failed."));
        wxFAIL;
        return false;
    }

    if (params->IsCriteriaUsingGradients(iStep, iPtor)) {
        std::vector<asPredictor *> predictorsPreprocess;
        predictorsPreprocess.push_back(predictor);

        auto *newPredictor = new asPredictor(*predictorsPreprocess[0]);
        if (!Preprocess(predictorsPreprocess, params->GetPreprocessMethod(iStep, iPtor), newPredictor)) {
            wxLogError(_("Data preprocessing failed."));
            Cleanup(predictorsPreprocess);
            wxDELETE(newPredictor);
            return false;
        }

        Cleanup(predictorsPreprocess);

        wxASSERT(newPredictor->GetTimeSize() > 0);
        predictors.push_back(newPredictor);
    } else {
        wxASSERT(predictor->GetTimeSize() > 0);
        predictors.push_back(predictor);
    }

    return true;
}

bool asMethodStandard::PreprocessArchiveData(std::vector<asPredictor *> &predictors, asParameters *params, int iStep,
                                             int iPtor, double timeStartData, double timeEndData) {
    std::vector<asPredictor *> predictorsPreprocess;

    int preprocessSize = params->GetPreprocessSize(iStep, iPtor);

    wxLogVerbose(_("Preprocessing data (%d predictor(s)) while loading."), preprocessSize);

    for (int iPre = 0; iPre < preprocessSize; iPre++) {
        // Date array object instantiation for data loading.
        double ptorStart = timeStartData + params->GetPreprocessTimeAsDays(iStep, iPtor, iPre);
        double ptorEnd = timeEndData + params->GetPreprocessTimeAsDays(iStep, iPtor, iPre);
        asTimeArray timeArray(ptorStart, ptorEnd, params->GetAnalogsTimeStepHours(), params->GetTimeArrayAnalogsMode());
        timeArray.Init();

        // Loading the dataset information
        asPredictor *predictorPreprocess =
            asPredictor::GetInstance(params->GetPreprocessDatasetId(iStep, iPtor, iPre),
                                     params->GetPreprocessDataId(iStep, iPtor, iPre), m_predictorDataDir);
        if (!predictorPreprocess) {
            Cleanup(predictorsPreprocess);
            return false;
        }

        // Set warning option
        predictorPreprocess->SetWarnMissingLevels(m_warnFailedLoadingData);

        // Select the number of members for ensemble data.
        if (predictorPreprocess->IsEnsemble()) {
            predictorPreprocess->SelectMembers(params->GetPreprocessMembersNb(iStep, iPtor, iPre));
        }

        // Area object instantiation
        asAreaCompGrid *area = asAreaCompGrid::GetInstance(params, iStep, iPtor);
        wxASSERT(area);

        // Data loading
        if (!predictorPreprocess->Load(area, timeArray, params->GetPreprocessLevel(iStep, iPtor, iPre))) {
            wxLogError(_("The data could not be loaded."));
            wxDELETE(area);
            wxDELETE(predictorPreprocess);
            Cleanup(predictorsPreprocess);
            return false;
        }
        wxDELETE(area);
        predictorsPreprocess.push_back(predictorPreprocess);
    }

    // Fix the criteria if S1/S2
    params->FixCriteriaIfGradientsPreprocessed(iStep, iPtor);

    asPredictor *predictor = new asPredictor(*predictorsPreprocess[0]);
    if (!Preprocess(predictorsPreprocess, params->GetPreprocessMethod(iStep, iPtor), predictor)) {
        wxLogError(_("Data preprocessing failed."));
        Cleanup(predictorsPreprocess);
        wxDELETE(predictor);
        return false;
    }

    // Standardize data
    if (params->GetStandardize(iStep, iPtor) &&
        !predictor->StandardizeData(params->GetStandardizeMean(iStep, iPtor), params->GetStandardizeSd(iStep, iPtor))) {
        wxLogError(_("Data standardisation has failed."));
        wxFAIL;
        return false;
    }

    Cleanup(predictorsPreprocess);
    predictors.push_back(predictor);

    return true;
}

void asMethodStandard::Cleanup(std::vector<asPredictor *> predictors) {
    if (!predictors.empty()) {
        for (auto &predictor : predictors) {
            wxDELETE(predictor);
        }
        predictors.resize(0);
    }
}

void asMethodStandard::Cleanup(std::vector<asCriteria *> criteria) {
    if (!criteria.empty()) {
        for (auto &criterion : criteria) {
            wxDELETE(criterion);
        }
        criteria.resize(0);
    }
}

void asMethodStandard::DeletePreloadedArchiveData() {
    if (!m_preloaded) return;

    for (int i = 0; i < m_preloadedArchive.size(); i++) {
        for (int j = 0; j < m_preloadedArchive[i].size(); j++) {
            for (int k = 0; k < m_preloadedArchive[i][j].size(); k++) {
                if (!m_preloadedArchivePointerCopy[i][j][k]) {
                    for (auto &l : m_preloadedArchive[i][j][k]) {
                        for (auto &m : l) {
                            wxDELETE(m);
                        }
                    }
                }
            }
        }
    }

    m_preloaded = false;
}

bool asMethodStandard::GetRandomLevelValidData(asParameters *params, int iStep, int iPtor, int iPre, int iHour) {
    vi levels;

    for (int iLevel = 0; iLevel < m_preloadedArchive[iStep][iPtor][iPre].size(); iLevel++) {
        wxASSERT(m_preloadedArchive[iStep][iPtor][iPre][iLevel].size() > iHour);
        if (m_preloadedArchive[iStep][iPtor][iPre][iLevel][iHour] != nullptr) {
            levels.push_back(iLevel);
        }
    }

    if (levels.empty()) {
        return false;
    }

    int randomIndex = asRandom(0, levels.size() - 1, 1);
    float newLevel = params->GetPreloadLevels(iStep, iPtor)[levels[randomIndex]];

    params->SetPredictorLevel(iStep, iPtor, newLevel);

    return true;
}

bool asMethodStandard::GetRandomValidData(asParameters *params, int iStep, int iPtor, int iPre) {
    vi levels, hours;

    for (int iLevel = 0; iLevel < m_preloadedArchive[iStep][iPtor][iPre].size(); iLevel++) {
        for (int iHour = 0; iHour < m_preloadedArchive[iStep][iPtor][iPre][iLevel].size(); iHour++) {
            if (m_preloadedArchive[iStep][iPtor][iPre][iLevel][iHour] != nullptr) {
                levels.push_back(iLevel);
                hours.push_back(iHour);
            }
        }
    }

    if (levels.empty()) {
        return false;
    }

    wxASSERT(!levels.empty());
    wxASSERT(!hours.empty());
    wxASSERT(levels.size() == hours.size());

    int randomIndex = asRandom(0, levels.size() - 1, 1);
    float newLevel = params->GetPreloadLevels(iStep, iPtor)[levels[randomIndex]];
    double newHour = params->GetPreloadHours(iStep, iPtor)[hours[randomIndex]];

    params->SetPredictorLevel(iStep, iPtor, newLevel);
    params->SetPredictorHour(iStep, iPtor, newHour);

    return true;
}