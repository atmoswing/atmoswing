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

#include "asMethodForecasting.h"

#include "asAreaGrid.h"
#include "asPreprocessor.h"
#include "asProcessor.h"
#include "asResultsDates.h"
#include "asResultsValues.h"
#include "asTimeArray.h"

#ifndef UNIT_TESTING

#include "AtmoSwingAppForecaster.h"

#endif

asMethodForecasting::asMethodForecasting(asBatchForecasts* batchForecasts, wxWindow* parent)
    : asMethodStandard(),
      _batchForecasts(batchForecasts),
      _forecastDate(NAN),
      _parent(parent) {}

asMethodForecasting::~asMethodForecasting() {
    ClearForecasts();
}

void asMethodForecasting::ClearForecasts() {
    _aggregator.ClearArrays();
}

bool asMethodForecasting::Manager() {
    ClearForecasts();
    asPredictorOper::SetDefaultPredictorsUrls();

#if USE_GUI
    if (g_responsive) wxTheApp->Yield();
#endif
    _cancel = false;
    bool hasErrors = false;

    if (isnan(_forecastDate)) {
        wxLogError(_("The date of the forecast has not been defined."));
        return false;
    }

    try {
#if USE_GUI
        // Switch off all leds
        wxCommandEvent eventStart(asEVT_STATUS_STARTING);
        if (_parent != nullptr) {
            _parent->ProcessWindowEvent(eventStart);
        }
#endif

        // Get paths
        wxString forecastParametersDir = _batchForecasts->GetParametersFileDirectory();
        wxString predictandDBDir = _batchForecasts->GetPredictandDBDirectory();

        // Execute the forecasts
        for (int i = 0; i < _batchForecasts->GetForecastsNb(); i++) {
#if USE_GUI
            if (g_responsive) wxTheApp->Yield();
            if (_cancel) return false;

            // Send event
            wxCommandEvent eventRunning(asEVT_STATUS_RUNNING);
            eventRunning.SetInt(i);
            if (_parent != nullptr) {
                _parent->ProcessWindowEvent(eventRunning);
            }

            if (g_responsive) wxTheApp->Yield();
#endif
            asLog::PrintToConsole(asStrF(_("Processing %s... "), _batchForecasts->GetForecastFileName(i)));
            wxLogMessage(_("Processing %s"), _batchForecasts->GetForecastFileName(i));
            fflush(stdout);

            // Load parameters
            _paramsFilePath = forecastParametersDir + DS + _batchForecasts->GetForecastFileName(i);
            asParametersForecast params;
            if (!params.LoadFromFile(_paramsFilePath)) return false;
            params.InitValues();

            _predictandDBFilePath = predictandDBDir + DS + params.GetPredictandDatabase();

#if USE_GUI
            if (g_responsive) wxTheApp->Yield();
#endif

            // Watch
            wxStopWatch sw;

            // Forecast
            if (!Forecast(params)) {
                hasErrors = true;
                asLog::PrintToConsole(_("FAILED!\n"));
                wxLogError(_("The forecast could not be processed"));

#if USE_GUI
                // Send event
                wxCommandEvent eventFailed(asEVT_STATUS_FAILED);
                eventFailed.SetInt(i);
                if (_parent != nullptr) {
                    _parent->ProcessWindowEvent(eventFailed);
                }
#endif
                Cleanup();
            } else {
                // Display processing time
                asLog::PrintToConsole(asStrF(_("done in %.1f sec.\n"), float(sw.Time()) / 1000.0f));
                wxLogMessage(_("Processing of the forecast \"%s\" - \"%s\" took %.1f sec to execute"),
                             params.GetMethodIdDisplay(), params.GetSpecificTagDisplay(), float(sw.Time()) / 1000.0f);
                fflush(stdout);

#if USE_GUI
                // Send event
                wxCommandEvent eventSuccess(asEVT_STATUS_SUCCESS);
                eventSuccess.SetInt(i);
                if (_parent != nullptr) {
                    _parent->ProcessWindowEvent(eventSuccess);
                }
#endif
            }
        }

        // Optional exports
        if (_batchForecasts->HasExports()) {
            if (_batchForecasts->GetExport() == asBatchForecasts::FullXml) {
                if (!_aggregator.ExportSyntheticFullXml(_batchForecasts->GetExportsOutputDirectory())) {
                    wxLogError(_("The export of the synthetic xml failed."));
                }
            } else if (_batchForecasts->GetExport() == asBatchForecasts::SmallCsv) {
                if (!_aggregator.ExportSyntheticSmallCsv(_batchForecasts->GetExportsOutputDirectory())) {
                    wxLogError(_("The export of the synthetic csv failed."));
                }
            } else if (_batchForecasts->GetExport() == asBatchForecasts::CustomCsvFVG) {
                if (!_aggregator.ExportSyntheticCustomCsvFVG(_batchForecasts->GetExportsOutputDirectory())) {
                    wxLogError(_("The export of the synthetic csv failed."));
                }
            }
        }
    } catch (runtime_error& e) {
        wxString msg(e.what(), wxConvUTF8);
        if (!msg.IsEmpty()) {
#if USE_GUI
            if (!g_silentMode) wxMessageBox(msg);
#else
            wxLogError(msg);
#endif
        }
        return false;
    }
#if USE_GUI
#if wxUSE_STATUSBAR
    wxLogStatus(_("Forecasting over."));
#endif
#endif
    Cleanup();

    return !hasErrors;
}

bool asMethodForecasting::Forecast(asParametersForecast& params) {
    // Process every step one after the other
    int stepsNb = params.GetStepsNb();

    // Download real-time predictors
    asResultsForecast resultsCheck;
    resultsCheck.SetForecastsDirectory(_batchForecasts->GetForecastsOutputDirectory());
    bool forecastDateChanged = true;
    while (forecastDateChanged) {
#if USE_GUI
        if (g_responsive) wxTheApp->Yield();
#endif
        if (_cancel) return false;

        // Check if result already exists
        resultsCheck.SetCurrentStep(stepsNb - 1);
        resultsCheck.Init(params, _forecastDate);
        if (resultsCheck.Exists()) {
            wxLogVerbose(_("Forecast already exists."));
            _resultsFilePaths.push_back(resultsCheck.GetFilePath());
            if (_batchForecasts->HasExports()) {
                auto results = new asResultsForecast();
                results->SetFilePath(resultsCheck.GetFilePath());
                results->Load();
                _aggregator.Add(results);
            }
#if USE_GUI
            if (g_responsive) wxTheApp->Yield();
#endif
            return true;
        }

        // Send event
#if USE_GUI
        wxCommandEvent eventDownloading(asEVT_STATUS_DOWNLOADING);
        if (_parent != nullptr) {
            _parent->ProcessWindowEvent(eventDownloading);
        }

        if (g_responsive) wxTheApp->Yield();
#endif

        forecastDateChanged = false;
        for (int iStep = 0; iStep < stepsNb; iStep++) {
#if USE_GUI
            if (g_responsive) wxTheApp->Yield();
#endif
            if (_cancel) return false;
            if (!DownloadRealtimePredictors(params, iStep, forecastDateChanged)) return false;
        }

        // Check again if result already exists (if change in date)
        resultsCheck.Init(params, _forecastDate);
        if (resultsCheck.Exists()) {
            wxLogVerbose(_("Forecast already exists."));
            _resultsFilePaths.push_back(resultsCheck.GetFilePath());
            if (_batchForecasts->HasExports()) {
                auto results = new asResultsForecast();
                results->SetFilePath(resultsCheck.GetFilePath());
                results->Load();
                _aggregator.Add(results);
            }
#if USE_GUI
            if (g_responsive) wxTheApp->Yield();
#endif
            return true;
        }

        // Send event
#if USE_GUI
        wxCommandEvent eventDownloaded(asEVT_STATUS_DOWNLOADED);
        if (_parent != nullptr) {
            _parent->ProcessWindowEvent(eventDownloaded);
        }
#endif
    }

#if USE_GUI
    if (g_responsive) wxTheApp->Yield();
#endif
    if (_cancel) return false;

    if (!LoadPredictandDB(_predictandDBFilePath)) return false;

#if USE_GUI
    if (g_responsive) wxTheApp->Yield();
#endif
    if (_cancel) return false;

    // Resulting object
    auto resultsPrevious = new asResultsForecast();
    auto results = new asResultsForecast();
    results->SetForecastsDirectory(_batchForecasts->GetForecastsOutputDirectory());

    for (int iStep = 0; iStep < stepsNb; iStep++) {
#if USE_GUI
        if (g_responsive) wxTheApp->Yield();
#endif
        if (_cancel) {
            wxDELETE(results);
            wxDELETE(resultsPrevious);
            return false;
        }

        if (iStep == 0) {
            if (!GetAnalogsDates(*results, params, iStep)) {
                wxDELETE(results);
                wxDELETE(resultsPrevious);
                return false;
            }
        } else {
            if (!GetAnalogsSubDates(*results, params, *resultsPrevious, iStep)) {
                wxDELETE(results);
                wxDELETE(resultsPrevious);
                return false;
            }
        }

        // At last get the values
        if (iStep == stepsNb - 1) {
#if USE_GUI
            if (g_responsive) wxTheApp->Yield();
#endif
            if (_cancel) {
                wxDELETE(results);
                wxDELETE(resultsPrevious);
                return false;
            }

            if (!GetAnalogsValues(*results, params, iStep)) {
                wxDELETE(results);
                wxDELETE(resultsPrevious);
                return false;
            }

#if USE_GUI
            // Send event
            wxCommandEvent eventSaving(asEVT_STATUS_SAVING);
            if (_parent != nullptr) {
                _parent->ProcessWindowEvent(eventSaving);
            }
#endif

            try {
                results->Save();
            } catch (runtime_error& e) {
                wxString msg(e.what(), wxConvUTF8);
                wxLogError(_("Exception caught: %s"), msg);
#if USE_GUI
                if (!g_silentMode) wxMessageBox(msg);
#else
                asLog::PrintToConsole(asStrF(_("Exception caught: %s"), msg));
#endif
                if (wxFileExists(results->GetFilePath())) {
                    wxRemoveFile(results->GetFilePath());
                }
                wxDELETE(results);
                wxDELETE(resultsPrevious);
                return false;
            }

#if USE_GUI
            // Send event
            wxCommandEvent eventSaved(asEVT_STATUS_SAVED);
            if (_parent != nullptr) {
                _parent->ProcessWindowEvent(eventSaved);
            }
#endif
        }

        // Keep the analogs dates of the best parameters set
        *resultsPrevious = *results;
    }

    _aggregator.Add(results);
    _resultsFilePaths.push_back(results->GetFilePath());

    wxDELETE(resultsPrevious);

    Cleanup();

    return true;
}

bool asMethodForecasting::DownloadRealtimePredictors(asParametersForecast& params, int iStep,
                                                     bool& forecastDateChanged) {
    for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
#if USE_GUI
        if (g_responsive) wxTheApp->Yield();
#endif
        if (_cancel) return false;

        wxLogVerbose(_("Downloading data."));

#if USE_GUI
        if (g_responsive) wxTheApp->Yield();
#endif

        if (!params.NeedsPreprocessing(iStep, iPtor)) {
            // Instantiate a predictor object
            asPredictorOper* predictorRealtime = asPredictorOper::GetInstance(
                params.GetPredictorRealtimeDatasetId(iStep, iPtor), params.GetPredictorRealtimeDataId(iStep, iPtor));

            if (!predictorRealtime) {
                wxDELETE(predictorRealtime);
                return false;
            }

            predictorRealtime->SetLevel(params.GetPredictorLevel(iStep, iPtor));

            if (!GetFiles(params, predictorRealtime, forecastDateChanged, params.GetPredictorHour(iStep, iPtor))) {
                wxDELETE(predictorRealtime);
                return false;
            }

            wxDELETE(predictorRealtime);
        } else {
            int preprocessSize = params.GetPreprocessSize(iStep, iPtor);

            for (int iPre = 0; iPre < preprocessSize; iPre++) {
#if USE_GUI
                if (g_responsive) wxTheApp->Yield();
#endif
                if (_cancel) return false;

                // Instantiate a predictor object
                asPredictorOper* predictorRealtimePreprocess = asPredictorOper::GetInstance(
                    params.GetPreprocessRealtimeDatasetId(iStep, iPtor, iPre),
                    params.GetPreprocessRealtimeDataId(iStep, iPtor, iPre));

                if (!predictorRealtimePreprocess) {
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }

                predictorRealtimePreprocess->SetLevel(params.GetPreprocessLevel(iStep, iPtor, iPre));

                if (!GetFiles(params, predictorRealtimePreprocess, forecastDateChanged,
                              params.GetPreprocessHour(iStep, iPtor, iPre))) {
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }

                wxDELETE(predictorRealtimePreprocess);
            }
        }

        wxLogVerbose(_("Data downloaded."));
    }

    return true;
}

bool asMethodForecasting::GetFiles(asParametersForecast& params, asPredictorOper* predictorRealtime,
                                   bool& forecastDateChanged, double hour) {
    // Get preferences
    wxConfigBase* pConfig = wxFileConfig::Get();
    long maxPrevStepsNb = pConfig->ReadLong("/Internet/MaxPreviousStepsNb", 5);

    predictorRealtime->SetPredictorsRealtimeDirectory(_batchForecasts->GetPredictorsRealtimeDirectory());

    // Set the desired forecasting date
    _forecastDate = predictorRealtime->SetRunDateInUse(_forecastDate);

    // Check if result already exists
    asResultsForecast resultsCheck;
    resultsCheck.SetForecastsDirectory(_batchForecasts->GetForecastsOutputDirectory());
    resultsCheck.SetCurrentStep(params.GetStepsNb() - 1);
    resultsCheck.Init(params, _forecastDate);
    if (resultsCheck.Exists()) {
        wxLogVerbose(_("Forecast already exists."));
#if USE_GUI
        if (g_responsive) wxTheApp->Yield();
#endif
        return true;
    }

    // Update forecasting date
    if (!predictorRealtime->BuildFilenamesAndUrls(hour, params.GetTargetTimeStepHours(), params.GetLeadTimeNb())) {
        return false;
    }

    // Realtime data downloading
    int counterFails = 0;
    while (true) {
#if USE_GUI
        if (g_responsive) wxTheApp->Yield();
#endif
        if (_cancel) {
            return false;
        }

        if (predictorRealtime->ShouldDownload()) {
            // Download predictor
            int resDownload = predictorRealtime->Download();
            if (resDownload == asSUCCESS) {
                break;
            } else if (resDownload == asFAILED) {
                if (counterFails < maxPrevStepsNb) {
                    // Try to download older data
                    _forecastDate = predictorRealtime->DecrementRunDateInUse();
                    // Check if result already exists
                    resultsCheck.SetCurrentStep(params.GetStepsNb() - 1);
                    resultsCheck.Init(params, _forecastDate);
                    if (resultsCheck.Exists()) {
                        wxLogVerbose(_("Forecast already exists."));
#if USE_GUI
                        if (g_responsive) wxTheApp->Yield();
#endif
                        return true;
                    }
                    forecastDateChanged = true;
                    predictorRealtime->BuildFilenamesAndUrls(hour, params.GetTargetTimeStepHours(),
                                                             params.GetLeadTimeNb());
                    counterFails++;
                } else {
                    wxLogError(_("The maximum attempts is reached to download the real-time predictor."));
                    return false;
                }
            } else {
                // Canceled for example.
                return false;
            }
        } else {
            // Check that files exist
            int countMissing = 0;

            vwxs fileNames = predictorRealtime->GetFileNames();

            for (const auto& fileName : fileNames) {
                wxString filePath = _batchForecasts->GetPredictorsRealtimeDirectory() + DS + fileName;
                if (!wxFileName::FileExists(filePath)) {
                    wxLogError(_("File not found: %s"), filePath);
                    countMissing++;
                }
            }

            if (100 * countMissing / fileNames.size() <= predictorRealtime->GetPercentMissingAllowed()) {
                break;
            }

            wxLogError(_("Directory: %s"), _batchForecasts->GetPredictorsRealtimeDirectory());
            return false;
        }
    }
    _forecastDate = predictorRealtime->GetRunDateInUse();

    return true;
}

bool asMethodForecasting::PreprocessRealtimePredictors(vector<asPredictorOper*> predictors, const wxString& method,
                                                       asPredictor* result) {
    vector<asPredictor*> ptorsPredictors(predictors.begin(), predictors.end());

    return asPreprocessor::Preprocess(ptorsPredictors, method, result);
}

bool asMethodForecasting::GetAnalogsDates(asResultsForecast& results, asParametersForecast& params, int iStep) {
    // Initialize the result object
    results.SetForecastsDirectory(_batchForecasts->GetForecastsOutputDirectory());
    results.SetCurrentStep(iStep);
    results.Init(params, _forecastDate);

    // Archive time array
    double timeStartArchive = params.GetArchiveStart();
    double timeEndArchive = params.GetArchiveEnd() - params.GetTimeSpanDays();
    asTimeArray timeArrayArchive(timeStartArchive, timeEndArchive, params.GetAnalogsTimeStepHours(),
                                 params.GetTimeArrayAnalogsMode());
    timeArrayArchive.Init();

    // Target time array
    vd leadTime = params.GetLeadTimeDaysVector();
    vd tmpTimeArray;
    for (double i : leadTime) {
        if (params.GetTargetTimeStepHours() >= 24) {
            tmpTimeArray.push_back(floor(_forecastDate) + i);
        } else {
            tmpTimeArray.push_back(_forecastDate + i);
        }
    }
    wxASSERT(!tmpTimeArray.empty());
    double timeStartTarget = tmpTimeArray[0];
    double timeEndTarget = tmpTimeArray[tmpTimeArray.size() - 1];

    asTimeArray timeArrayTarget = asTimeArray(tmpTimeArray);
    timeArrayTarget.Init();

    wxASSERT(timeArrayArchive.GetSize() > 100);
    if (!HasEnoughMemory(params, iStep, timeArrayArchive)) {
        return false;
    }

#if USE_GUI
    // Send event
    wxCommandEvent eventLoading(asEVT_STATUS_LOADING);
    if (_parent != nullptr) {
        _parent->ProcessWindowEvent(eventLoading);
    }
#endif

    // Loop through every predictor
    for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
        wxLogVerbose(_("Loading data (step %d, predictor nb %d)."), iStep, iPtor);

#if USE_GUI
        if (g_responsive) wxTheApp->Yield();
#endif

        if (!params.NeedsPreprocessing(iStep, iPtor)) {
            // Date array object instantiation for the data loading.
            double ptorStartArchive = timeStartArchive + params.GetTimeShiftDays() +
                                      params.GetPredictorTimeAsDays(iStep, iPtor);
            double ptorEndArchive = timeEndArchive + params.GetTimeShiftDays() +
                                    params.GetPredictorTimeAsDays(iStep, iPtor);
            asTimeArray timeArrayDataArchive(ptorStartArchive, ptorEndArchive, params.GetAnalogsTimeStepHours(),
                                             params.GetTimeArrayAnalogsMode());
            timeArrayDataArchive.Init();

            double ptorStartTarget = timeStartTarget + params.GetTimeShiftDays() +
                                     params.GetPredictorTimeAsDays(iStep, iPtor);
            double ptorEndTarget = timeEndTarget + params.GetTimeShiftDays() +
                                   params.GetPredictorTimeAsDays(iStep, iPtor);
            asTimeArray timeArrayDataTarget(ptorStartTarget, ptorEndTarget, params.GetTargetTimeStepHours(),
                                            asTimeArray::Simple);
            timeArrayDataTarget.Init();

            // Instantiate an archive predictor object
            asPredictor* predictorArchive = asPredictor::GetInstance(params.GetPredictorArchiveDatasetId(iStep, iPtor),
                                                                     params.GetPredictorArchiveDataId(iStep, iPtor),
                                                                     _batchForecasts->GetPredictorsArchiveDirectory());
            if (!predictorArchive) {
                return false;
            }

            // Set warning option
            predictorArchive->SetWarnMissingLevels(_warnFailedLoadingData);

            // Select the number of members for ensemble data.
            if (predictorArchive->IsEnsemble()) {
                predictorArchive->SelectMembers(params.GetPredictorArchiveMembersNb(iStep, iPtor));
            }

            // Instantiate an realtime predictor object
            asPredictorOper* predictorRealtime = asPredictorOper::GetInstance(
                params.GetPredictorRealtimeDatasetId(iStep, iPtor), params.GetPredictorRealtimeDataId(iStep, iPtor));
            if (!predictorRealtime) {
                wxDELETE(predictorArchive);
                return false;
            }
            predictorRealtime->SetPredictorsRealtimeDirectory(_batchForecasts->GetPredictorsRealtimeDirectory());
            predictorRealtime->SetRunDateInUse(_forecastDate);

            // Select the number of members for ensemble data.
            if (predictorRealtime->IsEnsemble()) {
                predictorRealtime->SelectMembers(params.GetPredictorRealtimeMembersNb(iStep, iPtor));
            }

            // Update
            predictorRealtime->SetLevel(params.GetPredictorLevel(iStep, iPtor));
            if (!predictorRealtime->BuildFilenamesAndUrls(params.GetPredictorHour(iStep, iPtor),
                                                          params.GetTargetTimeStepHours(), params.GetLeadTimeNb())) {
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            // Check time array for real-time data
            vd listTimeArray = predictorRealtime->GetDataDates();
            wxASSERT_MSG(listTimeArray.size() >= timeArrayDataTarget.GetSize(),
                         asStrF("size of listTimeArray = %d, size of timeArrayDataTarget = %d",
                                (int)listTimeArray.size(), (int)timeArrayDataTarget.GetSize()));
            for (int i = 0; i < timeArrayDataTarget.GetSize(); i++) {
                if (listTimeArray[i] != timeArrayDataTarget[i]) {
                    wxLogError(_("The real-time predictor time array is not consistent "
                                 "(listTimeArray[%d](%f)!=timeArrayDataTarget[%d](%f))."),
                               i, listTimeArray[i], i, timeArrayDataTarget[i]);
                    wxLogError(_("It is likely that the lead times you defined go beyond the data availability."));
                    wxDELETE(predictorArchive);
                    wxDELETE(predictorRealtime);
                    return false;
                }
            }

            // Area object instantiation
            asAreaGrid* area = asAreaGrid::GetInstance(&params, iStep, iPtor);

            // Standardize
            if (params.GetStandardize(iStep, iPtor)) {
                wxLogError(_("Data standardization is not yet implemented in operational forecasting."));
                wxDELETE(area);
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            // Archive data loading
            wxLogVerbose(_("Loading archive data."));
            if (!predictorArchive->Load(area, timeArrayDataArchive, params.GetPredictorLevel(iStep, iPtor))) {
                wxLogError(_("Archive data (%s) could not be loaded."), predictorArchive->GetDataId());
                wxDELETE(area);
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            // Standardize data
            if (params.GetArchiveStandardize(iStep, iPtor) &&
                !predictorArchive->StandardizeData(params.GetArchiveStandardizeMean(iStep, iPtor),
                                                   params.GetArchiveStandardizeSd(iStep, iPtor))) {
                wxLogError(_("Data standardisation has failed."));
                wxFAIL;
                wxDELETE(area);
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            wxASSERT(predictorArchive->GetData().size() > 1);
            _storagePredictorsArchive.push_back(predictorArchive);

            // Realtime data loading
            wxLogVerbose(_("Loading forecast data (predictorRealtime->Load)."));
            if (!predictorRealtime->Load(area, timeArrayDataTarget, params.GetPredictorLevel(iStep, iPtor))) {
                wxLogError(_("Real-time data (%s) could not be loaded."), predictorRealtime->GetDataId());
                wxDELETE(area);
                wxDELETE(predictorRealtime);
                return false;
            }

            // Standardize data
            if (params.GetRealtimeStandardize(iStep, iPtor) &&
                !predictorRealtime->StandardizeData(params.GetRealtimeStandardizeMean(iStep, iPtor),
                                                    params.GetRealtimeStandardizeSd(iStep, iPtor))) {
                wxLogError(_("Data standardisation has failed."));
                wxFAIL;
                wxDELETE(area);
                wxDELETE(predictorRealtime);
                return false;
            }

            wxDELETE(area);
            wxASSERT(predictorRealtime->GetData().size() > 1);
            _storagePredictorsRealtime.push_back(predictorRealtime);
        } else {
            int preprocessSize = params.GetPreprocessSize(iStep, iPtor);

            wxLogVerbose(_("Preprocessing data (%d predictor(s)) while loading."), preprocessSize);

            for (int iPre = 0; iPre < preprocessSize; iPre++) {
                wxLogVerbose(_("Loading predictor %d."), iPre);
#if USE_GUI
                if (g_responsive) wxTheApp->Yield();
#endif

                // Date array object instantiation for the data loading.
                // The array has the same length than timeArrayArchive, and the predictor dates are aligned with the
                // target dates, but the dates are not the same.
                double ptorStartArchive = timeStartArchive + params.GetTimeShiftDays() +
                                          params.GetPreprocessTimeAsDays(iStep, iPtor, iPre);
                double ptorEndArchive = timeEndArchive + params.GetTimeShiftDays() +
                                        params.GetPreprocessTimeAsDays(iStep, iPtor, iPre);
                asTimeArray timeArrayDataArchive(ptorStartArchive, ptorEndArchive, params.GetAnalogsTimeStepHours(),
                                                 params.GetTimeArrayAnalogsMode());
                timeArrayDataArchive.Init();

                double ptorStartTarget = timeStartTarget + params.GetTimeShiftDays() +
                                         params.GetPreprocessTimeAsDays(iStep, iPtor, iPre);
                double ptorEndTarget = timeEndTarget + params.GetTimeShiftDays() +
                                       params.GetPreprocessTimeAsDays(iStep, iPtor, iPre);
                asTimeArray timeArrayDataTarget(ptorStartTarget, ptorEndTarget, params.GetTargetTimeStepHours(),
                                                asTimeArray::Simple);
                timeArrayDataTarget.Init();

                // Instantiate an archive predictor object
                asPredictor* predictorArchivePreprocess = asPredictor::GetInstance(
                    params.GetPreprocessArchiveDatasetId(iStep, iPtor, iPre),
                    params.GetPreprocessArchiveDataId(iStep, iPtor, iPre),
                    _batchForecasts->GetPredictorsArchiveDirectory());
                if (!predictorArchivePreprocess) {
                    return false;
                }

                // Set warning option
                predictorArchivePreprocess->SetWarnMissingLevels(_warnFailedLoadingData);

                // Select the number of members for ensemble data.
                if (predictorArchivePreprocess->IsEnsemble()) {
                    predictorArchivePreprocess->SelectMembers(params.GetPreprocessArchiveMembersNb(iStep, iPtor, iPre));
                }

                // Instantiate an realtime predictor object
                asPredictorOper* predictorRealtimePreprocess = asPredictorOper::GetInstance(
                    params.GetPreprocessRealtimeDatasetId(iStep, iPtor, iPre),
                    params.GetPreprocessRealtimeDataId(iStep, iPtor, iPre));
                if (!predictorRealtimePreprocess) {
                    wxDELETE(predictorArchivePreprocess);
                    return false;
                }
                predictorRealtimePreprocess->SetPredictorsRealtimeDirectory(
                    _batchForecasts->GetPredictorsRealtimeDirectory());
                predictorRealtimePreprocess->SetRunDateInUse(_forecastDate);

                // Select the number of members for ensemble data.
                if (predictorRealtimePreprocess->IsEnsemble()) {
                    predictorRealtimePreprocess->SelectMembers(
                        params.GetPreprocessRealtimeMembersNb(iStep, iPtor, iPre));
                }

                // Update
                predictorRealtimePreprocess->SetLevel(params.GetPreprocessLevel(iStep, iPtor, iPre));
                if (!predictorRealtimePreprocess->BuildFilenamesAndUrls(params.GetPreprocessHour(iStep, iPtor, iPre),
                                                                        params.GetTargetTimeStepHours(),
                                                                        params.GetLeadTimeNb())) {
                    wxDELETE(predictorArchivePreprocess);
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }

                // Check time array for real-time data
                vd listTimeArray = predictorRealtimePreprocess->GetDataDates();
                wxASSERT_MSG(listTimeArray.size() >= timeArrayDataTarget.GetSize(),
                             asStrF("size of listTimeArray = %d, size of timeArrayDataTarget = %d",
                                    (int)listTimeArray.size(), (int)timeArrayDataTarget.GetSize()));

                for (int i = 0; i < timeArrayDataTarget.GetSize(); i++) {
                    if (listTimeArray[i] != timeArrayDataTarget[i]) {
                        wxLogError(_("The real-time predictor time array is not consistent "
                                     "(listTimeArray[%d](%f)!=timeArrayDataTarget[%d](%f))."),
                                   i, listTimeArray[i], i, timeArrayDataTarget[i]);
                        wxLogError(_("It is likely that the lead times you defined go beyond the data availability."));
                        wxDELETE(predictorArchivePreprocess);
                        wxDELETE(predictorRealtimePreprocess);
                        return false;
                    }
                }

                // Area object instantiation
                asAreaGrid* area = asAreaGrid::GetInstance(&params, iStep, iPtor);

                // Archive data loading
                wxLogVerbose(_("Loading archive data."));
                if (!predictorArchivePreprocess->Load(area, timeArrayDataArchive,
                                                      params.GetPreprocessLevel(iStep, iPtor, iPre))) {
                    wxLogError(_("Archive data (%s) could not be loaded."), predictorArchivePreprocess->GetDataId());
                    wxDELETE(area);
                    wxDELETE(predictorArchivePreprocess);
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }
                _storagePredictorsArchivePreprocess.push_back(predictorArchivePreprocess);

                // Realtime data loading
                wxLogVerbose(_("Loading forecast data (predictorRealtimePreprocess->Load)."));
                if (!predictorRealtimePreprocess->Load(area, timeArrayDataTarget,
                                                       params.GetPreprocessLevel(iStep, iPtor, iPre))) {
                    wxLogError(_("Real-time data (%s) could not be loaded."), predictorRealtimePreprocess->GetDataId());
                    wxDELETE(area);
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }
                wxDELETE(area);
                _storagePredictorsRealtimePreprocess.push_back(predictorRealtimePreprocess);
            }

            // Fix the criteria if S1
            params.FixCriteriaIfGradientsPreprocessed(iStep, iPtor);

            // Instantiate an archive predictor object
            auto predictorArchive = new asPredictor(*_storagePredictorsArchivePreprocess[0]);
            if (!predictorArchive) {
                return false;
            }

            if (!Preprocess(_storagePredictorsArchivePreprocess, params.GetPreprocessMethod(iStep, iPtor),
                            predictorArchive)) {
                wxLogError(_("Data preprocessing failed."));
                wxDELETE(predictorArchive);
                return false;
            }

            // Instantiate an realtime predictor object
            auto predictorRealtime = new asPredictorOper(*_storagePredictorsRealtimePreprocess[0]);
            if (!predictorRealtime) {
                wxDELETE(predictorArchive);
                return false;
            }
            predictorRealtime->SetPredictorsRealtimeDirectory(_batchForecasts->GetPredictorsRealtimeDirectory());

            if (!PreprocessRealtimePredictors(_storagePredictorsRealtimePreprocess,
                                              params.GetPreprocessMethod(iStep, iPtor), predictorRealtime)) {
                wxLogError(_("Data preprocessing failed."));
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            // Standardize data
            if (params.GetArchiveStandardize(iStep, iPtor) &&
                !predictorArchive->StandardizeData(params.GetArchiveStandardizeMean(iStep, iPtor),
                                                   params.GetArchiveStandardizeSd(iStep, iPtor))) {
                wxLogError(_("Data standardisation has failed."));
                wxFAIL;
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }
            if (params.GetRealtimeStandardize(iStep, iPtor) &&
                !predictorRealtime->StandardizeData(params.GetRealtimeStandardizeMean(iStep, iPtor),
                                                    params.GetRealtimeStandardizeSd(iStep, iPtor))) {
                wxLogError(_("Data standardisation has failed."));
                wxFAIL;
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            wxASSERT(predictorArchive->GetData().size() > 1);
            wxASSERT(predictorRealtime->GetData().size() > 1);

            _storagePredictorsArchive.push_back(predictorArchive);
            _storagePredictorsRealtime.push_back(predictorRealtime);
            DeletePreprocessData();
        }

        wxLogVerbose(_("Data loaded"));

        // Instantiate a score object. _storageCriteria takes ownership of the raw pointer.
        _storageCriteria.push_back(asCriteria::GetInstance(params.GetPredictorCriteria(iStep, iPtor)).release());
    }

#if USE_GUI
    // Send events
    wxCommandEvent eventLoaded(asEVT_STATUS_LOADED);
    wxCommandEvent eventProcessing(asEVT_STATUS_PROCESSING);
    if (_parent != nullptr) {
        _parent->ProcessWindowEvent(eventLoaded);
        _parent->ProcessWindowEvent(eventProcessing);
    }
#endif

    a1d timeArrayTargetVect = timeArrayTarget.GetTimeArray();
    a1d timeArrayTargetVectUnique(1);

    // Loop over the lead times
    for (int iLead = 0; iLead < timeArrayTarget.GetSize(); iLead++) {
        // Set the corresponding analogs number
        params.SetAnalogsNumber(iStep, params.GetAnalogsNumberLeadTime(iStep, iLead));

        // Create a standard analogs dates result object
        asResultsDates anaDates;
        anaDates.SetCurrentStep(iStep);
        anaDates.Init(&params);

        // Time array with one value
        timeArrayTargetVectUnique[0] = timeArrayTarget[iLead];
        asTimeArray timeArrayTargetLeadTime = asTimeArray(timeArrayTargetVectUnique);
        bool containsNaNs = false;

        if (!asProcessor::GetAnalogsDates(_storagePredictorsArchive, _storagePredictorsRealtime, timeArrayArchive,
                                          timeArrayArchive, timeArrayTarget, timeArrayTargetLeadTime, _storageCriteria,
                                          &params, iStep, anaDates, containsNaNs)) {
            wxLogError(_("Failed processing the analogs dates."));
            return false;
        }

        a2f dates = anaDates.GetAnalogsDates();
        wxASSERT(dates.rows() == 1);
        a1f rowDates = dates.row(0);
        results.SetAnalogsDates(iLead, rowDates);

        a2f criteriaVal = anaDates.GetAnalogsCriteria();
        wxASSERT(criteriaVal.rows() == 1);
        a1f rowCriteria = criteriaVal.row(0);
        results.SetAnalogsCriteria(iLead, rowCriteria);
    }

    results.SetTargetDates(timeArrayTarget.GetTimeArray());

    Cleanup();

    return true;
}

bool asMethodForecasting::GetAnalogsSubDates(asResultsForecast& results, asParametersForecast& params,
                                             asResultsForecast& resultsPrev, int iStep) {
    // Initialize the result object
    results.SetForecastsDirectory(_batchForecasts->GetForecastsOutputDirectory());
    results.SetCurrentStep(iStep);
    results.Init(params, _forecastDate);

    // Archive time array
    double timeStartArchive = params.GetArchiveStart();
    double timeEndArchive = params.GetArchiveEnd() - params.GetTimeSpanDays();
    asTimeArray timeArrayArchive(timeStartArchive, timeEndArchive, params.GetAnalogsTimeStepHours(),
                                 params.GetTimeArrayAnalogsMode());
    timeArrayArchive.Init();

    // Target time array
    vd leadTime = params.GetLeadTimeDaysVector();
    vd tmpTimeArray;
    for (double time : leadTime) {
        if (params.GetTargetTimeStepHours() >= 24) {
            tmpTimeArray.push_back(floor(_forecastDate) + time);
        } else {
            tmpTimeArray.push_back(_forecastDate + time);
        }
    }
    wxASSERT(!tmpTimeArray.empty());
    double timeStartTarget = tmpTimeArray[0];
    double timeEndTarget = tmpTimeArray[tmpTimeArray.size() - 1];

    asTimeArray timeArrayTarget = asTimeArray(tmpTimeArray);
    timeArrayTarget.Init();

    // Check archive time array length
    if (timeArrayArchive.GetSize() < 100) {
        wxLogError(_("The time array is not consistent in asMethodForecasting::GetAnalogsDates: size=%d."),
                   timeArrayArchive.GetSize());
        return false;
    }

    if (!HasEnoughMemory(params, iStep, timeArrayArchive)) {
        return false;
    }

#if USE_GUI
    // Send event
    wxCommandEvent eventLoading(asEVT_STATUS_LOADING);
    if (_parent != nullptr) {
        _parent->ProcessWindowEvent(eventLoading);
    }
#endif

    // Loop through every predictor
    for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
        wxLogVerbose(_("Loading data."));

        if (!params.NeedsPreprocessing(iStep, iPtor)) {
            // Date array object instantiation for the data loading.
            double ptorStartArchive = timeStartArchive + params.GetTimeShiftDays() +
                                      params.GetPredictorTimeAsDays(iStep, iPtor);
            double ptorEndArchive = timeEndArchive + params.GetTimeShiftDays() +
                                    params.GetPredictorTimeAsDays(iStep, iPtor);
            asTimeArray timeArrayDataArchive(ptorStartArchive, ptorEndArchive, params.GetAnalogsTimeStepHours(),
                                             params.GetTimeArrayAnalogsMode());
            timeArrayDataArchive.Init();

            double ptorStartTarget = timeStartTarget + params.GetTimeShiftDays() +
                                     params.GetPredictorTimeAsDays(iStep, iPtor);
            double ptorEndTarget = timeEndTarget + params.GetTimeShiftDays() +
                                   params.GetPredictorTimeAsDays(iStep, iPtor);
            asTimeArray timeArrayDataTarget(ptorStartTarget, ptorEndTarget, params.GetTargetTimeStepHours(),
                                            asTimeArray::Simple);
            timeArrayDataTarget.Init();

            // Instantiate an archive predictor object
            asPredictor* predictorArchive = asPredictor::GetInstance(params.GetPredictorArchiveDatasetId(iStep, iPtor),
                                                                     params.GetPredictorArchiveDataId(iStep, iPtor),
                                                                     _batchForecasts->GetPredictorsArchiveDirectory());
            if (!predictorArchive) {
                return false;
            }

            // Set warning option
            predictorArchive->SetWarnMissingLevels(_warnFailedLoadingData);

            // Select the number of members for ensemble data.
            if (predictorArchive->IsEnsemble()) {
                predictorArchive->SelectMembers(params.GetPredictorArchiveMembersNb(iStep, iPtor));
            }

            // Instantiate an realtime predictor object
            asPredictorOper* predictorRealtime = asPredictorOper::GetInstance(
                params.GetPredictorRealtimeDatasetId(iStep, iPtor), params.GetPredictorRealtimeDataId(iStep, iPtor));
            if (!predictorRealtime) {
                wxDELETE(predictorArchive);
                return false;
            }
            predictorRealtime->SetPredictorsRealtimeDirectory(_batchForecasts->GetPredictorsRealtimeDirectory());
            predictorRealtime->SetRunDateInUse(_forecastDate);

            // Select the number of members for ensemble data.
            if (predictorRealtime->IsEnsemble()) {
                predictorRealtime->SelectMembers(params.GetPredictorRealtimeMembersNb(iStep, iPtor));
            }

            // Update
            predictorRealtime->SetLevel(params.GetPredictorLevel(iStep, iPtor));
            if (!predictorRealtime->BuildFilenamesAndUrls(params.GetPredictorHour(iStep, iPtor),
                                                          params.GetTargetTimeStepHours(), params.GetLeadTimeNb())) {
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            // Check time array for real-time data
            vd listTimeArray = predictorRealtime->GetDataDates();
            wxASSERT_MSG(listTimeArray.size() >= timeArrayDataTarget.GetSize(),
                         asStrF("size of listTimeArray = %d, size of timeArrayDataTarget = %d",
                                (int)listTimeArray.size(), (int)timeArrayDataTarget.GetSize()));
            for (int i = 0; i < timeArrayDataTarget.GetSize(); i++) {
                if (listTimeArray[i] != timeArrayDataTarget[i]) {
                    wxLogError(_("The real-time predictor time array is not consistent "
                                 "(listTimeArray[%d](%f)!=timeArrayDataTarget[%d](%f))."),
                               i, listTimeArray[i], i, timeArrayDataTarget[i]);
                    wxLogError(_("It is likely that the lead times you defined go beyond the data availability."));
                    wxDELETE(predictorArchive);
                    wxDELETE(predictorRealtime);
                    return false;
                }
            }

            // Area object instantiation
            asAreaGrid* area = asAreaGrid::GetInstance(&params, iStep, iPtor);

            // Archive data loading
            if (!predictorArchive->Load(area, timeArrayDataArchive, params.GetPredictorLevel(iStep, iPtor))) {
                wxLogError(_("Archive data (%s) could not be loaded."), predictorArchive->GetDataId());
                wxDELETE(area);
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            // Standardize data
            if (params.GetArchiveStandardize(iStep, iPtor) &&
                !predictorArchive->StandardizeData(params.GetArchiveStandardizeMean(iStep, iPtor),
                                                   params.GetArchiveStandardizeSd(iStep, iPtor))) {
                wxLogError(_("Data standardisation has failed."));
                wxFAIL;
                wxDELETE(area);
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            _storagePredictorsArchive.push_back(predictorArchive);

            // Realtime data loading
            if (!predictorRealtime->Load(area, timeArrayDataTarget, params.GetPredictorLevel(iStep, iPtor))) {
                wxLogError(_("Real-time data (%s) could not be loaded."), predictorRealtime->GetDataId());
                wxDELETE(area);
                wxDELETE(predictorRealtime);
                return false;
            }

            // Standardize data
            if (params.GetRealtimeStandardize(iStep, iPtor) &&
                !predictorRealtime->StandardizeData(params.GetRealtimeStandardizeMean(iStep, iPtor),
                                                    params.GetRealtimeStandardizeSd(iStep, iPtor))) {
                wxLogError(_("Data standardisation has failed."));
                wxFAIL;
                wxDELETE(area);
                wxDELETE(predictorRealtime);
                return false;
            }

            wxDELETE(area);
            _storagePredictorsRealtime.push_back(predictorRealtime);
        } else {
            int preprocessSize = params.GetPreprocessSize(iStep, iPtor);

            wxLogVerbose(_("Preprocessing data (%d predictor(s)) while loading."), preprocessSize);

            for (int iPre = 0; iPre < preprocessSize; iPre++) {
                wxLogVerbose(_("Loading predictor %d."), iPre);

                // Date array object instantiation for data loading.
                double ptorStartArchive = timeStartArchive + params.GetTimeShiftDays() +
                                          params.GetPreprocessTimeAsDays(iStep, iPtor, iPre);
                double ptorEndArchive = timeEndArchive + params.GetTimeShiftDays() +
                                        params.GetPreprocessTimeAsDays(iStep, iPtor, iPre);
                asTimeArray timeArrayDataArchive(ptorStartArchive, ptorEndArchive, params.GetAnalogsTimeStepHours(),
                                                 params.GetTimeArrayAnalogsMode());
                timeArrayDataArchive.Init();

                double ptorStartTarget = timeStartTarget + params.GetTimeShiftDays() +
                                         params.GetPreprocessTimeAsDays(iStep, iPtor, iPre);
                double ptorEndTarget = timeEndTarget + params.GetTimeShiftDays() +
                                       params.GetPreprocessTimeAsDays(iStep, iPtor, iPre);
                asTimeArray timeArrayDataTarget(ptorStartTarget, ptorEndTarget, params.GetTargetTimeStepHours(),
                                                asTimeArray::Simple);
                timeArrayDataTarget.Init();

                // Instantiate an archive predictor object
                asPredictor* predictorArchivePreprocess = asPredictor::GetInstance(
                    params.GetPreprocessArchiveDatasetId(iStep, iPtor, iPre),
                    params.GetPreprocessArchiveDataId(iStep, iPtor, iPre),
                    _batchForecasts->GetPredictorsArchiveDirectory());
                if (!predictorArchivePreprocess) {
                    return false;
                }

                // Set warning option
                predictorArchivePreprocess->SetWarnMissingLevels(_warnFailedLoadingData);

                // Select the number of members for ensemble data.
                if (predictorArchivePreprocess->IsEnsemble()) {
                    predictorArchivePreprocess->SelectMembers(params.GetPreprocessArchiveMembersNb(iStep, iPtor, iPre));
                }

                // Instantiate an realtime predictor object
                asPredictorOper* predictorRealtimePreprocess = asPredictorOper::GetInstance(
                    params.GetPreprocessRealtimeDatasetId(iStep, iPtor, iPre),
                    params.GetPreprocessRealtimeDataId(iStep, iPtor, iPre));
                if (!predictorRealtimePreprocess) {
                    wxDELETE(predictorArchivePreprocess);
                    return false;
                }
                predictorRealtimePreprocess->SetPredictorsRealtimeDirectory(
                    _batchForecasts->GetPredictorsRealtimeDirectory());
                predictorRealtimePreprocess->SetRunDateInUse(_forecastDate);

                // Select the number of members for ensemble data.
                if (predictorRealtimePreprocess->IsEnsemble()) {
                    predictorRealtimePreprocess->SelectMembers(
                        params.GetPreprocessRealtimeMembersNb(iStep, iPtor, iPre));
                }

                // Update
                predictorRealtimePreprocess->SetLevel(params.GetPreprocessLevel(iStep, iPtor, iPre));
                if (!predictorRealtimePreprocess->BuildFilenamesAndUrls(params.GetPreprocessHour(iStep, iPtor, iPre),
                                                                        params.GetTargetTimeStepHours(),
                                                                        params.GetLeadTimeNb())) {
                    wxDELETE(predictorArchivePreprocess);
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }

                // Check time array for real-time data
                vd listTimeArray = predictorRealtimePreprocess->GetDataDates();
                wxASSERT_MSG(listTimeArray.size() >= timeArrayDataTarget.GetSize(),
                             asStrF("listTimeArray.size() = %d, timeArrayDataTarget.GetSize() = %d",
                                    (int)listTimeArray.size(), (int)timeArrayDataTarget.GetSize()));
                for (int i = 0; i < timeArrayDataTarget.GetSize(); i++) {
                    if (listTimeArray[i] != timeArrayDataTarget[i]) {
                        wxLogError(_("The real-time predictor time array is not consistent "
                                     "(listTimeArray[%d](%f)!=timeArrayDataTarget[%d](%f))."),
                                   i, listTimeArray[i], i, timeArrayDataTarget[i]);
                        wxLogError(_("It is likely that the lead times you defined go beyond the data availability."));
                        wxDELETE(predictorArchivePreprocess);
                        wxDELETE(predictorRealtimePreprocess);
                        return false;
                    }
                }

                // Area object instantiation
                asAreaGrid* area = asAreaGrid::GetInstance(&params, iStep, iPtor);

                // Archive data loading
                if (!predictorArchivePreprocess->Load(area, timeArrayDataArchive,
                                                      params.GetPreprocessLevel(iStep, iPtor, iPre))) {
                    wxLogError(_("Archive data (%s) could not be loaded."), predictorArchivePreprocess->GetDataId());
                    wxDELETE(area);
                    wxDELETE(predictorArchivePreprocess);
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }
                _storagePredictorsArchivePreprocess.push_back(predictorArchivePreprocess);

                // Realtime data loading
                if (!predictorRealtimePreprocess->Load(area, timeArrayDataTarget,
                                                       params.GetPreprocessLevel(iStep, iPtor, iPre))) {
                    wxLogError(_("Real-time data (%s) could not be loaded."), predictorRealtimePreprocess->GetDataId());
                    wxDELETE(area);
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }
                wxDELETE(area);
                _storagePredictorsRealtimePreprocess.push_back(predictorRealtimePreprocess);

                wxASSERT_MSG(
                    predictorArchivePreprocess->GetLatPtsnb() == predictorRealtimePreprocess->GetLatPtsnb(),
                    asStrF("predictorArchivePreprocess.GetLatPtsnb()=%d, predictorRealtimePreprocess.GetLatPtsnb()=%d",
                           predictorArchivePreprocess->GetLatPtsnb(), predictorRealtimePreprocess->GetLatPtsnb()));
                wxASSERT_MSG(
                    predictorArchivePreprocess->GetLonPtsnb() == predictorRealtimePreprocess->GetLonPtsnb(),
                    asStrF("predictorArchivePreprocess.GetLonPtsnb()=%d, predictorRealtimePreprocess.GetLonPtsnb()=%d",
                           predictorArchivePreprocess->GetLonPtsnb(), predictorRealtimePreprocess->GetLonPtsnb()));
            }

            // Fix the criteria if S1
            params.FixCriteriaIfGradientsPreprocessed(iStep, iPtor);

            // Instantiate an archive predictor object
            auto predictorArchive = new asPredictor(*_storagePredictorsArchivePreprocess[0]);
            if (!predictorArchive) {
                return false;
            }

            if (!Preprocess(_storagePredictorsArchivePreprocess, params.GetPreprocessMethod(iStep, iPtor),
                            predictorArchive)) {
                wxLogError(_("Data preprocessing failed."));
                wxDELETE(predictorArchive);
                return false;
            }

            // Instantiate an realtime predictor object
            auto predictorRealtime = new asPredictorOper(*_storagePredictorsRealtimePreprocess[0]);
            if (!predictorRealtime) {
                wxDELETE(predictorArchive);
                return false;
            }
            predictorRealtime->SetPredictorsRealtimeDirectory(_batchForecasts->GetPredictorsRealtimeDirectory());

            if (!PreprocessRealtimePredictors(_storagePredictorsRealtimePreprocess,
                                              params.GetPreprocessMethod(iStep, iPtor), predictorRealtime)) {
                wxLogError(_("Data preprocessing failed."));
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            // Standardize data
            if (params.GetArchiveStandardize(iStep, iPtor) &&
                !predictorArchive->StandardizeData(params.GetArchiveStandardizeMean(iStep, iPtor),
                                                   params.GetArchiveStandardizeSd(iStep, iPtor))) {
                wxLogError(_("Data standardisation has failed."));
                wxFAIL;
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }
            if (params.GetRealtimeStandardize(iStep, iPtor) &&
                !predictorRealtime->StandardizeData(params.GetRealtimeStandardizeMean(iStep, iPtor),
                                                    params.GetRealtimeStandardizeSd(iStep, iPtor))) {
                wxLogError(_("Data standardisation has failed."));
                wxFAIL;
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            wxASSERT(predictorArchive->GetLatPtsnb() == predictorRealtime->GetLatPtsnb());
            wxASSERT(predictorArchive->GetLonPtsnb() == predictorRealtime->GetLonPtsnb());
            _storagePredictorsArchive.push_back(predictorArchive);
            _storagePredictorsRealtime.push_back(predictorRealtime);
            DeletePreprocessData();
        }

        wxLogVerbose(_("Data loaded"));

        // Instantiate a score object. _storageCriteria takes ownership of the raw pointer.
        wxLogVerbose(_("Creating a criterion object."));
        _storageCriteria.push_back(asCriteria::GetInstance(params.GetPredictorCriteria(iStep, iPtor)).release());
        wxLogVerbose(_("Criterion object created."));
    }

#if USE_GUI
    // Send events
    wxCommandEvent eventLoaded(asEVT_STATUS_LOADED);
    wxCommandEvent eventProcessing(asEVT_STATUS_PROCESSING);
    if (_parent != nullptr) {
        _parent->ProcessWindowEvent(eventLoaded);
        _parent->ProcessWindowEvent(eventProcessing);
    }
#endif

    a1f leadTimes = resultsPrev.GetTargetDates();

    // Loop over the lead times
    for (int iLead = 0; iLead < timeArrayTarget.GetSize(); iLead++) {
        // Create a analogs date object for previous results
        asResultsDates anaDatesPrev;
        anaDatesPrev.SetCurrentStep(iStep - 1);
        anaDatesPrev.Init(&params);

        // Set the corresponding analogs number
        params.SetAnalogsNumber(iStep - 1, params.GetAnalogsNumberLeadTime(iStep - 1, iLead));
        params.SetAnalogsNumber(iStep, params.GetAnalogsNumberLeadTime(iStep, iLead));

        // Create a standard analogs dates result object
        asResultsDates anaDates;
        anaDates.SetCurrentStep(iStep);
        anaDates.Init(&params);

        a1f datesPrev = resultsPrev.GetAnalogsDates(iLead);
        a2f datesPrev2D(1, params.GetAnalogsNumberLeadTime(iStep - 1, iLead));
        datesPrev2D.row(0) = datesPrev;

        a1f criteriaPrev = resultsPrev.GetAnalogsCriteria(iLead);
        a2f criteriaPrev2D(1, params.GetAnalogsNumberLeadTime(iStep - 1, iLead));
        criteriaPrev2D.row(0) = criteriaPrev;

        a1d leadTimeArray(1);
        leadTimeArray[0] = leadTimes[iLead];
        anaDatesPrev.SetTargetDates(leadTimeArray);
        anaDatesPrev.SetAnalogsDates(datesPrev2D);
        anaDatesPrev.SetAnalogsCriteria(criteriaPrev2D);
        bool containsNaNs = false;

        if (!asProcessor::GetAnalogsSubDates(_storagePredictorsArchive, _storagePredictorsRealtime, timeArrayArchive,
                                             timeArrayTarget, anaDatesPrev, _storageCriteria, &params, iStep, anaDates,
                                             containsNaNs)) {
            wxLogError(_("Failed processing the analogs dates."));
            return false;
        }

        a2f dates = anaDates.GetAnalogsDates();
        wxASSERT(dates.rows() == 1);
        a1f rowDates = dates.row(0);
        results.SetAnalogsDates(iLead, rowDates);

        a2f criteriaVal = anaDates.GetAnalogsCriteria();
        wxASSERT(criteriaVal.rows() == 1);
        a1f rowCriteria = criteriaVal.row(0);
        results.SetAnalogsCriteria(iLead, rowCriteria);
    }

    results.SetTargetDates(leadTimes);

    Cleanup();

    return true;
}

bool asMethodForecasting::GetAnalogsValues(asResultsForecast& results, asParametersForecast& params, int iStep) {
    // Initialize the result object
    results.SetForecastsDirectory(_batchForecasts->GetForecastsOutputDirectory());
    results.SetCurrentStep(iStep);
    results.SetPredictandDatasetId(_predictandDB->GetDatasetId());
    results.SetPredictandParameter(_predictandDB->GetDataParameter());
    results.SetPredictandTemporalResolution(_predictandDB->GetDataTemporalResolution());
    results.SetPredictandSpatialAggregation(_predictandDB->GetDataSpatialAggregation());
    results.SetCoordinateSystem(_predictandDB->GetCoordSys());

    // Set the predictands values to the corresponding analog dates
    wxASSERT(_predictandDB);

    // Set the stations IDs and coordinates
    wxASSERT(_predictandDB->GetStationsNb() > 0);
    a1i stationsId = _predictandDB->GetStationsIdArray();
    wxASSERT(stationsId.size() > 0);
    results.SetStationIds(stationsId);
    results.SetStationOfficialIds(_predictandDB->GetStationOfficialIdsArray());
    results.SetStationNames(_predictandDB->GetStationNamesArray());
    results.SetStationHeights(_predictandDB->GetStationHeightsArray());
    results.SetStationXCoords(_predictandDB->GetStationXCoordsArray());
    results.SetStationYCoords(_predictandDB->GetStationYCoordsArray());
    if (_predictandDB->HasReferenceAxis()) {
        a1f refAxis = _predictandDB->GetReferenceAxis();
        results.SetReferenceAxis(refAxis);
        a2f refValues = _predictandDB->GetReferenceValuesArray();
        results.SetReferenceValues(refValues);
    }

    // Set the predictor properties
    vwxs predictorDatasetIdsOper;
    vwxs predictorDatasetIdsArchive;
    vwxs predictorDataIdsOper;
    vwxs predictorDataIdsArchive;
    vf predictorLevels;
    vf predictorHours;
    vf predictorXmin;
    vf predictorXmax;
    vf predictorYmin;
    vf predictorYmax;

    for (int i = 0; i < params.GetStepsNb(); ++i) {
        for (int j = 0; j < params.GetPredictorsNb(i); ++j) {
            auto xMin = float(params.GetPredictorXmin(i, j));
            auto xMax = float(params.GetPredictorXmin(i, j) +
                              params.GetPredictorXstep(i, j) * (params.GetPredictorXptsnb(i, j) - 1));
            auto yMin = float(params.GetPredictorYmin(i, j));
            auto yMax = float(params.GetPredictorYmin(i, j) +
                              params.GetPredictorYstep(i, j) * (params.GetPredictorYptsnb(i, j) - 1));
            if (params.NeedsPreprocessing(i, j)) {
                for (int k = 0; k < params.GetPreprocessSize(i, j); ++k) {
                    predictorDatasetIdsOper.push_back(params.GetPreprocessRealtimeDatasetId(i, j, k));
                    predictorDatasetIdsArchive.push_back(params.GetPreprocessArchiveDatasetId(i, j, k));
                    predictorDataIdsOper.push_back(params.GetPreprocessRealtimeDataId(i, j, k));
                    predictorDataIdsArchive.push_back(params.GetPreprocessArchiveDataId(i, j, k));
                    predictorLevels.push_back(params.GetPreprocessLevel(i, j, k));
                    predictorHours.push_back(float(params.GetPreprocessHour(i, j, k)));
                    predictorXmin.push_back(xMin);
                    predictorXmax.push_back(xMax);
                    predictorYmin.push_back(yMin);
                    predictorYmax.push_back(yMax);
                }
            } else {
                predictorDatasetIdsOper.push_back(params.GetPredictorRealtimeDatasetId(i, j));
                predictorDatasetIdsArchive.push_back(params.GetPredictorArchiveDatasetId(i, j));
                predictorDataIdsOper.push_back(params.GetPredictorRealtimeDataId(i, j));
                predictorDataIdsArchive.push_back(params.GetPredictorArchiveDataId(i, j));
                predictorLevels.push_back(params.GetPredictorLevel(i, j));
                predictorHours.push_back(float(params.GetPredictorHour(i, j)));
                predictorXmin.push_back(xMin);
                predictorXmax.push_back(xMax);
                predictorYmin.push_back(yMin);
                predictorYmax.push_back(yMax);
            }
        }
    }

    results.SetPredictorDatasetIdsOper(predictorDatasetIdsOper);
    results.SetPredictorDatasetIdsArchive(predictorDatasetIdsArchive);
    results.SetPredictorDataIdsOper(predictorDataIdsOper);
    results.SetPredictorDataIdsArchive(predictorDataIdsArchive);
    results.SetPredictorLevels(predictorLevels);
    results.SetPredictorHours(predictorHours);
    results.SetPredictorLonMin(predictorXmin);
    results.SetPredictorLonMax(predictorXmax);
    results.SetPredictorLatMin(predictorYmin);
    results.SetPredictorLatMax(predictorYmax);

    a1f leadTimes = results.GetTargetDates();

    // Loop over the lead times
    for (int iLead = 0; iLead < leadTimes.size(); iLead++) {
        // Set the corresponding analogs number
        params.SetAnalogsNumber(iStep, params.GetAnalogsNumberLeadTime(iStep, iLead));

        asResultsDates anaDates;
        anaDates.SetCurrentStep(iStep);
        anaDates.Init(&params);

        a1f datesPrev = results.GetAnalogsDates(iLead);
        a2f datesPrev2D(1, params.GetAnalogsNumberLeadTime(iStep, iLead));
        datesPrev2D.row(0) = datesPrev;

        a1f criteriaPrev = results.GetAnalogsCriteria(iLead);
        a2f criteriaPrev2D(1, params.GetAnalogsNumberLeadTime(iStep, iLead));
        criteriaPrev2D.row(0) = criteriaPrev;

        a1d leadTimeArray(1);
        leadTimeArray[0] = leadTimes[iLead];
        anaDates.SetTargetDates(leadTimeArray);
        anaDates.SetAnalogsDates(datesPrev2D);
        anaDates.SetAnalogsCriteria(criteriaPrev2D);

        // Process for every station
        for (int iStat = 0; iStat < stationsId.size(); iStat++) {
            vi stationId(1);
            stationId[0] = stationsId[iStat];

            // Set the next station ID
            params.SetPredictandStationIds(stationId);

            // Create a standard analogs values result object
            asResultsValues anaValues = asResultsValues();
            anaValues.SetCurrentStep(iStep);
            anaValues.Init(&params);

            if (!asProcessor::GetAnalogsValues(*_predictandDB, anaDates, &params, anaValues)) {
                wxLogError(_("Failed setting the predictand values to the corresponding analog dates."));
                return false;
            }

            va2f valuesRaw = anaValues.GetAnalogsValuesRaw();
            wxASSERT(valuesRaw[0].rows() == 1);
            a1f rowValuesRaw = valuesRaw[0].row(0);
            results.SetAnalogsValuesRaw(iLead, iStat, rowValuesRaw);

            va2f valuesNorm = anaValues.GetAnalogsValuesNorm();
            wxASSERT(valuesNorm[0].rows() == 1);
            a1f rowValuesNorm = valuesNorm[0].row(0);
            results.SetAnalogsValuesNorm(iLead, iStat, rowValuesNorm);
        }
    }

#if USE_GUI
    // Send event
    wxCommandEvent eventProcessed(asEVT_STATUS_PROCESSED);
    if (_parent != nullptr) {
        _parent->ProcessWindowEvent(eventProcessed);
    }
#endif

    return true;
}

double asMethodForecasting::GetEffectiveArchiveDataStart(asParameters* params) const {
    return GetTimeStartArchive(params);
}

double asMethodForecasting::GetEffectiveArchiveDataEnd(asParameters* params) const {
    return GetTimeEndArchive(params);
}

void asMethodForecasting::Cleanup() {
    DeletePreprocessData();

    if (!_storagePredictorsArchive.empty()) {
        for (auto& predictors : _storagePredictorsArchive) {
            wxDELETE(predictors);
        }
        _storagePredictorsArchive.resize(0);
    }

    if (!_storagePredictorsRealtime.empty()) {
        for (auto& predictors : _storagePredictorsRealtime) {
            wxDELETE(predictors);
        }
        _storagePredictorsRealtime.resize(0);
    }

    if (!_storageCriteria.empty()) {
        for (auto& criteria : _storageCriteria) {
            wxDELETE(criteria);
        }
        _storageCriteria.resize(0);
    }

    // Do not delete preloaded data here !
}

bool asMethodForecasting::HasEnoughMemory(const asParametersForecast& params, int iStep,
                                          const asTimeArray& timeArrayArchive) const {
    wxLongLong neededMem = 0;
    for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
        neededMem += (params.GetPredictorXptsnb(iStep, iPtor)) * (params.GetPredictorYptsnb(iStep, iPtor));
    }
    neededMem *= timeArrayArchive.GetSize();  // time dimension
    neededMem *= 4;                           // to bytes (for floats)
    double neededMemMb = neededMem.ToDouble();
    neededMemMb /= 1048576.0;  // to Mb

    // Get available memory
    wxMemorySize freeMemSize = wxGetFreeMemory();
    wxLongLong freeMem = freeMemSize;
    double freeMemMb = freeMem.ToDouble();
    freeMemMb /= 1048576.0;  // To Mb

    if (freeMemSize < 0) {
        wxLogVerbose(_("Needed memory for data: %.2f Mb (cannot evaluate available memory)"), neededMemMb);
    } else {
        wxLogVerbose(_("Needed memory for data: %.2f Mb (%.2f Mb available)"), neededMemMb, freeMemMb);
        if (neededMemMb > freeMemMb) {
            wxLogError(_("Data cannot fit into available memory."));
            return false;
        }
    }

    return true;
}

void asMethodForecasting::DeletePreprocessData() {
    for (auto& predictors : _storagePredictorsArchivePreprocess) {
        wxDELETE(predictors);
    }
    _storagePredictorsArchivePreprocess.resize(0);

    for (auto& predictors : _storagePredictorsRealtimePreprocess) {
        wxDELETE(predictors);
    }
    _storagePredictorsRealtimePreprocess.resize(0);
}