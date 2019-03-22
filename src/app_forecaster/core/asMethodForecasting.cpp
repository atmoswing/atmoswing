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

#include "asResultsDates.h"
#include "asResultsValues.h"
#include "asTimeArray.h"
#include "asAreaCompGrid.h"
#include "asProcessor.h"
#include "asPreprocessor.h"

#ifndef UNIT_TESTING

#include "AtmoswingAppForecaster.h"

#endif

asMethodForecasting::asMethodForecasting(asBatchForecasts *batchForecasts, wxWindow *parent)
        : asMethodStandard(),
          m_batchForecasts(batchForecasts),
          m_forecastDate(NaNd),
          m_parent(parent)
{

}

asMethodForecasting::~asMethodForecasting()
{
    ClearForecasts();
}

void asMethodForecasting::ClearForecasts()
{
    m_aggregator.ClearArrays();
}

bool asMethodForecasting::Manager()
{
    ClearForecasts();
    asPredictorOper::SetDefaultPredictorsUrls();

#if wxUSE_GUI
    if (g_responsive)
        wxGetApp().Yield();
#endif
    m_cancel = false;

    if (asIsNaN(m_forecastDate)) {
        wxLogError(_("The date of the forecast has not been defined."));
        return false;
    }

    try {
#if wxUSE_GUI
        // Switch off all leds
        wxCommandEvent eventStart(asEVT_STATUS_STARTING);
        if (m_parent != nullptr) {
            m_parent->ProcessWindowEvent(eventStart);
        }
#endif

        // Get paths
        wxString forecastParametersDir = m_batchForecasts->GetParametersFileDirectory();
        wxString predictandDBDir = m_batchForecasts->GetPredictandDBDirectory();

        // Execute the forecasts
        for (int i = 0; i < m_batchForecasts->GetForecastsNb(); i++) {
#if wxUSE_GUI
            if (g_responsive)
                wxGetApp().Yield();
            if (m_cancel)
                return false;

            // Send event
            wxCommandEvent eventRunning(asEVT_STATUS_RUNNING);
            eventRunning.SetInt(i);
            if (m_parent != nullptr) {
                m_parent->ProcessWindowEvent(eventRunning);
            }

            if (g_responsive)
                wxGetApp().Yield();
#endif
            wxPrintf("Processing %s... ", m_batchForecasts->GetForecastFileName(i));
            wxLogMessage("Processing %s", m_batchForecasts->GetForecastFileName(i));
            fflush(stdout);

            // Load parameters
            m_paramsFilePath = forecastParametersDir + DS + m_batchForecasts->GetForecastFileName(i);
            asParametersForecast params;
            if (!params.LoadFromFile(m_paramsFilePath))
                return false;
            params.InitValues();

            m_predictandDBFilePath = predictandDBDir + DS + params.GetPredictandDatabase();

#if wxUSE_GUI
            if (g_responsive)
                wxGetApp().Yield();
#endif

            // Watch
            wxStopWatch sw;

            // Forecast
            if (!Forecast(params)) {
                wxPrintf(_("FAILED!\n"));
                wxLogError(_("The forecast could not be achived"));

#if wxUSE_GUI
                // Send event
                wxCommandEvent eventFailed(asEVT_STATUS_FAILED);
                eventFailed.SetInt(i);
                if (m_parent != nullptr) {
                    m_parent->ProcessWindowEvent(eventFailed);
                }
#endif
            } else {
                // Display processing time
                wxPrintf(_("done in %.1f sec.\n"), static_cast<float>(sw.Time()) / 1000.0f);
                wxLogMessage(_("Processing of the forecast \"%s\" - \"%s\" took %.1f sec to execute"),
                             params.GetMethodIdDisplay(),
                             params.GetSpecificTagDisplay(),
                             static_cast<float>(sw.Time()) / 1000.0f);
                fflush(stdout);

#if wxUSE_GUI
                // Send event
                wxCommandEvent eventSuccess(asEVT_STATUS_SUCCESS);
                eventSuccess.SetInt(i);
                if (m_parent != nullptr) {
                    m_parent->ProcessWindowEvent(eventSuccess);
                }
#endif
            }
        }

        // Optional exports
        if (m_batchForecasts->HasExports()) {
            if (m_batchForecasts->ExportSyntheticXml()) {
                if (!m_aggregator.ExportSyntheticXml(m_batchForecasts->GetExportsOutputDirectory())) {
                    wxLogError(_("The export of the synthetic xml failed."));
                }
            }
        }
    } catch (asException &e) {
        wxString fullMessage = e.GetFullMessage();
        if (!fullMessage.IsEmpty()) {
#if wxUSE_GUI
            if (!g_silentMode)
                wxMessageBox(fullMessage);
#else
            wxLogError(fullMessage);
#endif
        }
        return false;
    }
#if wxUSE_GUI
    #if wxUSE_STATUSBAR
        wxLogStatus(_("Forecasting over."));
    #endif
#endif
    Cleanup();

    return true;
}

bool asMethodForecasting::Forecast(asParametersForecast &params)
{
    // Process every step one after the other
    int stepsNb = params.GetStepsNb();

    // Download real-time predictors
    asResultsForecast resultsCheck;
    resultsCheck.SetForecastsDirectory(m_batchForecasts->GetForecastsOutputDirectory());
    bool forecastDateChanged = true;
    while (forecastDateChanged) {
#if wxUSE_GUI
        if (g_responsive)
            wxGetApp().Yield();
#endif
        if (m_cancel)
            return false;

        // Check if result already exists
        resultsCheck.SetCurrentStep(stepsNb - 1);
        resultsCheck.Init(params, m_forecastDate);
        if (resultsCheck.Exists()) {
            wxLogVerbose(_("Forecast already exists."));
            m_resultsFilePaths.push_back(resultsCheck.GetFilePath());
            if (m_batchForecasts->HasExports()) {
                asResultsForecast *results = new asResultsForecast();
                results->SetFilePath(resultsCheck.GetFilePath());
                results->Load();
                m_aggregator.Add(results);
            }
#if wxUSE_GUI
            if (g_responsive)
                wxGetApp().Yield();
#endif
            return true;
        }

        // Send event
#if wxUSE_GUI
        wxCommandEvent eventDownloading(asEVT_STATUS_DOWNLOADING);
        if (m_parent != nullptr) {
            m_parent->ProcessWindowEvent(eventDownloading);
        }

        if (g_responsive)
            wxGetApp().Yield();
#endif

        forecastDateChanged = false;
        for (int iStep = 0; iStep < stepsNb; iStep++) {
#if wxUSE_GUI
            if (g_responsive)
                wxGetApp().Yield();
#endif
            if (m_cancel)
                return false;
            if (!DownloadRealtimePredictors(params, iStep, forecastDateChanged))
                return false;
        }

        // Check again if result already exists (if change in date)
        resultsCheck.Init(params, m_forecastDate);
        if (resultsCheck.Exists()) {
            wxLogVerbose(_("Forecast already exists."));
            m_resultsFilePaths.push_back(resultsCheck.GetFilePath());
            if (m_batchForecasts->HasExports()) {
                asResultsForecast *results = new asResultsForecast();
                results->SetFilePath(resultsCheck.GetFilePath());
                results->Load();
                m_aggregator.Add(results);
            }
#if wxUSE_GUI
            if (g_responsive)
                wxGetApp().Yield();
#endif
            return true;
        }

        // Send event
#if wxUSE_GUI
        wxCommandEvent eventDownloaded(asEVT_STATUS_DOWNLOADED);
        if (m_parent != nullptr) {
            m_parent->ProcessWindowEvent(eventDownloaded);
        }
#endif
    }

#if wxUSE_GUI
    if (g_responsive)
        wxGetApp().Yield();
#endif
    if (m_cancel)
        return false;

    // Load the Predictand DB
    wxLogVerbose(_("Loading the Predictand DB."));
    if (!LoadPredictandDB(m_predictandDBFilePath))
        return false;
    wxLogVerbose(_("Predictand DB loaded."));

#if wxUSE_GUI
    if (g_responsive)
        wxGetApp().Yield();
#endif
    if (m_cancel)
        return false;

    // Resulting object
    auto *resultsPrevious = new asResultsForecast();
    auto *results = new asResultsForecast();
    results->SetForecastsDirectory(m_batchForecasts->GetForecastsOutputDirectory());

    for (int iStep = 0; iStep < stepsNb; iStep++) {
#if wxUSE_GUI
        if (g_responsive)
            wxGetApp().Yield();
#endif
        if (m_cancel) {
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
#if wxUSE_GUI
            if (g_responsive)
                wxGetApp().Yield();
#endif
            if (m_cancel) {
                wxDELETE(results);
                wxDELETE(resultsPrevious);
                return false;
            }

            if (!GetAnalogsValues(*results, params, iStep)) {
                wxDELETE(results);
                wxDELETE(resultsPrevious);
                return false;
            }

#if wxUSE_GUI
            // Send event
            wxCommandEvent eventSaving(asEVT_STATUS_SAVING);
            if (m_parent != nullptr) {
                m_parent->ProcessWindowEvent(eventSaving);
            }
#endif

            try {
                results->Save();
            } catch (asException &e) {
                wxString fullMessage = e.GetFullMessage();
                if (!fullMessage.IsEmpty()) {
#if wxUSE_GUI
                    if (!g_silentMode)
                        wxMessageBox(fullMessage);
#endif
                }
                wxDELETE(results);
                wxDELETE(resultsPrevious);
                return false;
            }

#if wxUSE_GUI
            // Send event
            wxCommandEvent eventSaved(asEVT_STATUS_SAVED);
            if (m_parent != nullptr) {
                m_parent->ProcessWindowEvent(eventSaved);
            }
#endif
        }

        // Keep the analogs dates of the best parameters set
        *resultsPrevious = *results;
    }

    m_aggregator.Add(results);
    m_resultsFilePaths.push_back(results->GetFilePath());

    wxDELETE(resultsPrevious);

    Cleanup();

    return true;
}

bool asMethodForecasting::DownloadRealtimePredictors(asParametersForecast &params, int iStep, bool &forecastDateChanged)
{
    // Get preferences
    wxConfigBase *pConfig = wxFileConfig::Get();
    long maxPrevStepsNb = pConfig->Read("/Internet/MaxPreviousStepsNb", (long) 5);

    for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
#if wxUSE_GUI
        if (g_responsive)
            wxGetApp().Yield();
#endif
        if (m_cancel)
            return false;

        wxLogVerbose(_("Downloading data."));

#if wxUSE_GUI
        if (g_responsive)
            wxGetApp().Yield();
#endif

        if (!params.NeedsPreprocessing(iStep, iPtor)) {
            // Instanciate a predictor object
            asPredictorOper *predictorRealtime = asPredictorOper::GetInstance(
                    params.GetPredictorRealtimeDatasetId(iStep, iPtor),
                    params.GetPredictorRealtimeDataId(iStep, iPtor));
            if (!predictorRealtime) {
                wxDELETE(predictorRealtime);
                return false;
            }
            predictorRealtime->SetPredictorsRealtimeDirectory(m_batchForecasts->GetPredictorsRealtimeDirectory());

            // Set the desired forecasting date
            m_forecastDate = predictorRealtime->SetRunDateInUse(m_forecastDate);

            // Check if result already exists
            asResultsForecast resultsCheck;
            resultsCheck.SetForecastsDirectory(m_batchForecasts->GetForecastsOutputDirectory());
            resultsCheck.SetCurrentStep(params.GetStepsNb() - 1);
            resultsCheck.Init(params, m_forecastDate);
            if (resultsCheck.Exists()) {
                wxLogVerbose(_("Forecast already exists."));
#if wxUSE_GUI
                if (g_responsive)
                    wxGetApp().Yield();
#endif
                wxDELETE(predictorRealtime);
                return true;
            }

            // Restriction needed
            wxASSERT(params.GetTargetTimeStepHours() > 0);
            predictorRealtime->RestrictTimeArray(params.GetPredictorTimeHours(iStep, iPtor),
                                                 params.GetTargetTimeStepHours(),
                                                 params.GetLeadTimeNb());

            // Update forecasting date
            if (!predictorRealtime->BuildFilenamesUrls()) {
                wxDELETE(predictorRealtime);
                return false;
            }

            // Realtime data downloading
            int counterFails = 0;
            while (true) {
#if wxUSE_GUI
                if (g_responsive)
                    wxGetApp().Yield();
#endif
                if (m_cancel) {
                    wxDELETE(predictorRealtime);
                    return false;
                }

                // Download predictor
                int resDownload = predictorRealtime->Download();
                if (resDownload == asSUCCESS) {
                    break;
                } else if (resDownload == asFAILED) {
                    if (counterFails < maxPrevStepsNb) {
                        // Try to download older data
                        m_forecastDate = predictorRealtime->DecrementRunDateInUse();
                        // Check if result already exists
                        resultsCheck.SetCurrentStep(params.GetStepsNb() - 1);
                        resultsCheck.Init(params, m_forecastDate);
                        if (resultsCheck.Exists()) {
                            wxLogVerbose(_("Forecast already exists."));
#if wxUSE_GUI
                            if (g_responsive)
                                wxGetApp().Yield();
#endif
                            wxDELETE(predictorRealtime);
                            return true;
                        }
                        forecastDateChanged = true;
                        predictorRealtime->BuildFilenamesUrls();
                        counterFails++;
                    } else {
                        wxLogError(_("The maximum attempts is reached to download the real-time predictor. Forecasting failed."));
                        wxDELETE(predictorRealtime);
                        return false;
                    }
                } else {
                    // Canceled for example.
                    wxDELETE(predictorRealtime);
                    return false;
                }
            }
            m_forecastDate = predictorRealtime->GetRunDateInUse();
            wxDELETE(predictorRealtime);
        } else {
            int preprocessSize = params.GetPreprocessSize(iStep, iPtor);

            for (int iPre = 0; iPre < preprocessSize; iPre++) {
#if wxUSE_GUI
                if (g_responsive)
                    wxGetApp().Yield();
#endif
                if (m_cancel)
                    return false;

                // Instanciate a predictor object
                asPredictorOper *predictorRealtimePreprocess = asPredictorOper::GetInstance(
                        params.GetPreprocessRealtimeDatasetId(iStep, iPtor, iPre),
                        params.GetPreprocessRealtimeDataId(iStep, iPtor, iPre));
                if (!predictorRealtimePreprocess) {
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }
                predictorRealtimePreprocess->SetPredictorsRealtimeDirectory(
                        m_batchForecasts->GetPredictorsRealtimeDirectory());

                // Set the desired forecasting date
                m_forecastDate = predictorRealtimePreprocess->SetRunDateInUse(m_forecastDate);

                // Restriction needed
                wxASSERT(params.GetTargetTimeStepHours() > 0);
                predictorRealtimePreprocess->RestrictTimeArray(params.GetPreprocessTimeHours(iStep, iPtor, iPre),
                                                               params.GetTargetTimeStepHours(),
                                                               params.GetLeadTimeNb());

                // Update forecasting date
                if (!predictorRealtimePreprocess->BuildFilenamesUrls()) {
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }

                // Realtime data downloading
                int counterFails = 0;
                while (true) {
#if wxUSE_GUI
                    if (g_responsive)
                        wxGetApp().Yield();
#endif
                    if (m_cancel) {
                        wxDELETE(predictorRealtimePreprocess);
                        return false;
                    }

                    // Download predictor
                    int resDownload = predictorRealtimePreprocess->Download();
                    if (resDownload == asSUCCESS) {
                        break;
                    } else if (resDownload == asFAILED) {
                        if (counterFails < maxPrevStepsNb) {
                            // Try to download older data
                            m_forecastDate = predictorRealtimePreprocess->DecrementRunDateInUse();
                            forecastDateChanged = true;
                            predictorRealtimePreprocess->BuildFilenamesUrls();
                            counterFails++;
                        } else {
                            wxLogError(_("The maximum attempts is reached to download the real-time predictor. Forecasting failed."));
                            wxDELETE(predictorRealtimePreprocess);
                            return false;
                        }
                    } else {
                        // Canceled for example.
                        wxDELETE(predictorRealtimePreprocess);
                        return false;
                    }
                }
                m_forecastDate = predictorRealtimePreprocess->GetRunDateInUse();
                wxDELETE(predictorRealtimePreprocess);
            }
        }

        wxLogVerbose(_("Data downloaded."));
    }

    return true;
}

bool asMethodForecasting::PreprocessRealtimePredictors(std::vector<asPredictorOper *> predictors,
                                                       const wxString &method, asPredictor *result)
{
    std::vector<asPredictor *> ptorsPredictors(predictors.begin(), predictors.end());

    return asPreprocessor::Preprocess(ptorsPredictors, method, result);
}

bool asMethodForecasting::GetAnalogsDates(asResultsForecast &results, asParametersForecast &params, int iStep)
{
    // Initialize the result object
    results.SetForecastsDirectory(m_batchForecasts->GetForecastsOutputDirectory());
    results.SetCurrentStep(iStep);
    results.Init(params, m_forecastDate);

    // Archive time array
    double timeStartArchive = params.GetArchiveStart();
    double timeEndArchive = params.GetArchiveEnd();
    timeEndArchive = wxMin(timeEndArchive, timeEndArchive -
                                           params.GetTimeSpanDays()); // Adjust so the predictors search won't overtake the array
    asTimeArray timeArrayArchive(timeStartArchive, timeEndArchive, params.GetAnalogsTimeStepHours(),
                                 asTimeArray::Simple);
    timeArrayArchive.Init();

    // Get last lead time of the data
    double lastLeadTime = 9999;
    for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
        if (!params.NeedsPreprocessing(iStep, iPtor)) {
            // Instanciate a predictor object
            asPredictorOper *predictorRealtime = asPredictorOper::GetInstance(
                    params.GetPredictorRealtimeDatasetId(iStep, iPtor),
                    params.GetPredictorRealtimeDataId(iStep, iPtor));
            if (!predictorRealtime) {
                return false;
            }

            predictorRealtime->SetRunDateInUse(m_forecastDate);
            lastLeadTime = wxMin(lastLeadTime,
                                 predictorRealtime->GetForecastLeadTimeEnd() / 24.0 - params.GetTimeSpanDays());

            wxDELETE(predictorRealtime);
        } else {
            for (int iPre = 0; iPre < params.GetPreprocessSize(iStep, iPtor); iPre++) {
                // Instanciate a predictor object
                asPredictorOper *predictorRealtimePreprocess = asPredictorOper::GetInstance(
                        params.GetPreprocessRealtimeDatasetId(iStep, iPtor, iPre),
                        params.GetPreprocessRealtimeDataId(iStep, iPtor, iPre));
                if (!predictorRealtimePreprocess) {
                    return false;
                }

                predictorRealtimePreprocess->SetRunDateInUse(m_forecastDate);
                lastLeadTime = wxMin(lastLeadTime, predictorRealtimePreprocess->GetForecastLeadTimeEnd() / 24.0 -
                                                   params.GetTimeSpanDays());

                wxDELETE(predictorRealtimePreprocess);
            }
        }
    }

    // Target time array
    vi leadTime = params.GetLeadTimeDaysVector();
    vd tmpTimeArray;
    for (int i : leadTime) {
        if (i > lastLeadTime)
            break;

        double tmpDate = floor(m_forecastDate) + i;
        tmpTimeArray.push_back(tmpDate);
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
    wxLogVerbose(_("Date array created."));

    // Calculate needed memory
    wxLongLong neededMem = 0;
    for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
        neededMem += (params.GetPredictorXptsnb(iStep, iPtor)) * (params.GetPredictorYptsnb(iStep, iPtor));
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

    if (freeMemSize < 0) {
        wxLogVerbose(_("Needed memory for data: %.2f Mb (cannot evaluate available memory)"), neededMemMb);
    } else {
        wxLogVerbose(_("Needed memory for data: %.2f Mb (%.2f Mb available)"), neededMemMb, freeMemMb);
        if (neededMemMb > freeMemMb) {
            wxLogError(_("Data cannot fit into available memory."));
            return false;
        }
    }

#if wxUSE_GUI
    // Send event
    wxCommandEvent eventLoading(asEVT_STATUS_LOADING);
    if (m_parent != nullptr) {
        m_parent->ProcessWindowEvent(eventLoading);
    }
#endif

    // Loop through every predictor
    for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
        wxLogVerbose(_("Loading data (step %d, predictor nb %d)."), iStep, iPtor);

#if wxUSE_GUI
        if (g_responsive)
            wxGetApp().Yield();
#endif

        if (!params.NeedsPreprocessing(iStep, iPtor)) {
            // Date array object instantiation for the data loading. The array has the same length than timeArrayArchive, and the predictor dates are aligned with the target dates, but the dates are not the same.
            double ptorStartArchive = timeStartArchive - params.GetTimeShiftDays()
                                      + params.GetPredictorTimeHours(iStep, iPtor) / 24.0;
            double ptorEndArchive = timeEndArchive - params.GetTimeShiftDays()
                                    + params.GetPredictorTimeHours(iStep, iPtor) / 24.0;
            asTimeArray timeArrayDataArchive(ptorStartArchive, ptorEndArchive, params.GetAnalogsTimeStepHours(),
                                             params.GetTimeArrayAnalogsMode());
            timeArrayDataArchive.Init();

            double ptorStartTarget = timeStartTarget - params.GetTimeShiftDays()
                                     + params.GetPredictorTimeHours(iStep, iPtor) / 24.0;
            double ptorEndTarget = timeEndTarget - params.GetTimeShiftDays()
                                   + params.GetPredictorTimeHours(iStep, iPtor) / 24.0;
            asTimeArray timeArrayDataTarget(ptorStartTarget, ptorEndTarget, params.GetTargetTimeStepHours(),
                                            asTimeArray::Simple);
            timeArrayDataTarget.Init();

            // Instanciate an archive predictor object
            asPredictor *predictorArchive = asPredictor::GetInstance(
                    params.GetPredictorArchiveDatasetId(iStep, iPtor),
                    params.GetPredictorArchiveDataId(iStep, iPtor),
                    m_batchForecasts->GetPredictorsArchiveDirectory());
            if (!predictorArchive) {
                return false;
            }

            // Select the number of members for ensemble data.
            if (predictorArchive->IsEnsemble()) {
                predictorArchive->SelectMembers(params.GetPredictorArchiveMembersNb(iStep, iPtor));
            }

            // Instanciate an realtime predictor object
            asPredictorOper *predictorRealtime = asPredictorOper::GetInstance(
                    params.GetPredictorRealtimeDatasetId(iStep, iPtor),
                    params.GetPredictorRealtimeDataId(iStep, iPtor));
            if (!predictorRealtime) {
                wxDELETE(predictorArchive);
                return false;
            }
            predictorRealtime->SetPredictorsRealtimeDirectory(m_batchForecasts->GetPredictorsRealtimeDirectory());
            predictorRealtime->SetRunDateInUse(m_forecastDate);

            // Select the number of members for ensemble data.
            if (predictorRealtime->IsEnsemble()) {
                predictorRealtime->SelectMembers(params.GetPredictorRealtimeMembersNb(iStep, iPtor));
            }

            // Restriction needed
            wxASSERT(params.GetTargetTimeStepHours() > 0);
            predictorRealtime->RestrictTimeArray(params.GetPredictorTimeHours(iStep, iPtor),
                                                 params.GetTargetTimeStepHours(),
                                                 params.GetLeadTimeNb());

            // Update
            if (!predictorRealtime->BuildFilenamesUrls()) {
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            // Check time array for real-time data
            vd listTimeArray = predictorRealtime->GetDataDates();
            wxASSERT_MSG(listTimeArray.size() >= (unsigned) timeArrayDataTarget.GetSize(),
                         wxString::Format("size of listTimeArray = %d, size of timeArrayDataTarget = %d",
                                          (int) listTimeArray.size(), (int) timeArrayDataTarget.GetSize()));
            for (int i = 0; i < timeArrayDataTarget.GetSize(); i++) {
                if (listTimeArray[i] != timeArrayDataTarget[i]) {
                    wxLogError(_("The real-time predictor time array is not consistent (listTimeArray[%d](%f)!=timeArrayDataTarget[%d](%f))."),
                               i, listTimeArray[i], i, timeArrayDataTarget[i]);
                    wxLogError(_("It is likely that the lead times you defined go beyond the data availability."));
                    wxDELETE(predictorArchive);
                    wxDELETE(predictorRealtime);
                    return false;
                }
            }

            // Area object instantiation
            asAreaCompGrid *area = asAreaCompGrid::GetInstance(&params, iStep, iPtor);

            // Standardize
            if (params.GetStandardize(iStep, iPtor)) {
                wxLogError(_("Data standardization is not yet implemented in operational forecasting."));
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

            wxASSERT(predictorArchive->GetData().size() > 1);
            m_storagePredictorsArchive.push_back(predictorArchive);

            // Realtime data loading
            wxLogVerbose(_("Loading GCM forecast data."));
            if (!predictorRealtime->Load(area, timeArrayDataTarget, params.GetPredictorLevel(iStep, iPtor))) {
                wxLogError(_("Real-time data (%s) could not be loaded."), predictorRealtime->GetDataId());
                wxDELETE(area);
                wxDELETE(predictorRealtime);
                return false;
            }

            wxDELETE(area);
            wxASSERT(predictorRealtime->GetData().size() > 1);
            m_storagePredictorsRealtime.push_back(predictorRealtime);
        } else {
            int preprocessSize = params.GetPreprocessSize(iStep, iPtor);

            wxLogVerbose(_("Preprocessing data (%d predictor(s)) while loading."), preprocessSize);

            for (int iPre = 0; iPre < preprocessSize; iPre++) {

                wxLogVerbose(_("Loading predictor %d."), iPre);
#if wxUSE_GUI
                if (g_responsive)
                    wxGetApp().Yield();
#endif

                // Date array object instantiation for the data loading. The array has the same length than timeArrayArchive, and the predictor dates are aligned with the target dates, but the dates are not the same.
                double ptorStartArchive = timeStartArchive - params.GetTimeShiftDays() +
                                          params.GetPreprocessTimeHours(iStep, iPtor, iPre) / 24.0;
                double ptorEndArchive = timeEndArchive - params.GetTimeShiftDays() +
                                        params.GetPreprocessTimeHours(iStep, iPtor, iPre) / 24.0;
                asTimeArray timeArrayDataArchive(ptorStartArchive, ptorEndArchive, params.GetAnalogsTimeStepHours(),
                                                 params.GetTimeArrayAnalogsMode());
                timeArrayDataArchive.Init();

                double ptorStartTarget = timeStartTarget - params.GetTimeShiftDays() +
                                         params.GetPreprocessTimeHours(iStep, iPtor, iPre) / 24.0;
                double ptorEndTarget = timeEndTarget - params.GetTimeShiftDays() +
                                       params.GetPreprocessTimeHours(iStep, iPtor, iPre) / 24.0;
                asTimeArray timeArrayDataTarget(ptorStartTarget, ptorEndTarget, params.GetTargetTimeStepHours(), asTimeArray::Simple);
                timeArrayDataTarget.Init();

                // Instanciate an archive predictor object
                asPredictor *predictorArchivePreprocess = asPredictor::GetInstance(
                        params.GetPreprocessArchiveDatasetId(iStep, iPtor, iPre),
                        params.GetPreprocessArchiveDataId(iStep, iPtor, iPre),
                        m_batchForecasts->GetPredictorsArchiveDirectory());
                if (!predictorArchivePreprocess) {
                    return false;
                }

                // Select the number of members for ensemble data.
                if (predictorArchivePreprocess->IsEnsemble()) {
                    predictorArchivePreprocess->SelectMembers(params.GetPreprocessArchiveMembersNb(iStep, iPtor, iPre));
                }

                // Instanciate an realtime predictor object
                asPredictorOper *predictorRealtimePreprocess = asPredictorOper::GetInstance(
                        params.GetPreprocessRealtimeDatasetId(iStep, iPtor, iPre),
                        params.GetPreprocessRealtimeDataId(iStep, iPtor, iPre));
                if (!predictorRealtimePreprocess) {
                    wxDELETE(predictorArchivePreprocess);
                    return false;
                }
                predictorRealtimePreprocess->SetPredictorsRealtimeDirectory(
                        m_batchForecasts->GetPredictorsRealtimeDirectory());
                predictorRealtimePreprocess->SetRunDateInUse(m_forecastDate);

                // Select the number of members for ensemble data.
                if (predictorRealtimePreprocess->IsEnsemble()) {
                    predictorRealtimePreprocess->SelectMembers(
                            params.GetPreprocessRealtimeMembersNb(iStep, iPtor, iPre));
                }

                // Restriction needed
                wxASSERT(params.GetTargetTimeStepHours() > 0);
                predictorRealtimePreprocess->RestrictTimeArray(params.GetPreprocessTimeHours(iStep, iPtor, iPre),
                                                               params.GetTargetTimeStepHours(),
                                                               params.GetLeadTimeNb());

                // Update
                if (!predictorRealtimePreprocess->BuildFilenamesUrls()) {
                    wxDELETE(predictorArchivePreprocess);
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }

                // Check time array for real-time data
                vd listTimeArray = predictorRealtimePreprocess->GetDataDates();
                wxASSERT_MSG(listTimeArray.size() >= (unsigned) timeArrayDataTarget.GetSize(),
                             wxString::Format("size of listTimeArray = %d, size of timeArrayDataTarget = %d",
                                              (int) listTimeArray.size(), (int) timeArrayDataTarget.GetSize()));

                for (int i = 0; i < timeArrayDataTarget.GetSize(); i++) {
                    if (listTimeArray[i] != timeArrayDataTarget[i]) {
                        wxLogError(_("The real-time predictor time array is not consistent (listTimeArray[%d](%f)!=timeArrayDataTarget[%d](%f))."),
                                   i, listTimeArray[i], i, timeArrayDataTarget[i]);
                        wxLogError(_("It is likely that the lead times you defined go beyond the data availability."));
                        wxDELETE(predictorArchivePreprocess);
                        wxDELETE(predictorRealtimePreprocess);
                        return false;
                    }
                }

                // Area object instantiation
                asAreaCompGrid *area = asAreaCompGrid::GetInstance(&params, iStep, iPtor);

                // Archive data loading
                wxLogVerbose(_("Loading archive data."));
                if (!predictorArchivePreprocess->Load(area, timeArrayDataArchive, params.GetPreprocessLevel(iStep, iPtor, iPre))) {
                    wxLogError(_("Archive data (%s) could not be loaded."), predictorArchivePreprocess->GetDataId());
                    wxDELETE(area);
                    wxDELETE(predictorArchivePreprocess);
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }
                m_storagePredictorsArchivePreprocess.push_back(predictorArchivePreprocess);

                // Realtime data loading
                wxLogVerbose(_("Loading forecast data."));
                if (!predictorRealtimePreprocess->Load(area, timeArrayDataTarget, params.GetPreprocessLevel(iStep, iPtor, iPre))) {
                    wxLogError(_("Real-time data (%s) could not be loaded."), predictorRealtimePreprocess->GetDataId());
                    wxDELETE(area);
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }
                wxDELETE(area);
                m_storagePredictorsRealtimePreprocess.push_back(predictorRealtimePreprocess);
            }

            // Fix the criteria if S1
            params.FixCriteriaIfGradientsPreprocessed(iStep, iPtor);

            // Instanciate an archive predictor object
            asPredictor *predictorArchive = new asPredictor(*m_storagePredictorsArchivePreprocess[0]);
            if (!predictorArchive) {
                return false;
            }

            if (!Preprocess(m_storagePredictorsArchivePreprocess, params.GetPreprocessMethod(iStep, iPtor),
                            predictorArchive)) {
                wxLogError(_("Data preprocessing failed."));
                wxDELETE(predictorArchive);
                return false;
            }

            // Instanciate an realtime predictor object
            asPredictorOper *predictorRealtime = new asPredictorOper(*m_storagePredictorsRealtimePreprocess[0]);
            if (!predictorRealtime) {
                wxDELETE(predictorArchive);
                return false;
            }
            predictorRealtime->SetPredictorsRealtimeDirectory(m_batchForecasts->GetPredictorsRealtimeDirectory());

            if (!PreprocessRealtimePredictors(m_storagePredictorsRealtimePreprocess,
                                              params.GetPreprocessMethod(iStep, iPtor),
                                              predictorRealtime)) {
                wxLogError(_("Data preprocessing failed."));
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            // Standardize
            if (params.GetStandardize(iStep, iPtor)) {
                wxLogError(_("Data standardization is not yet implemented in operational forecasting."));
                return false;
            }

            wxASSERT(predictorArchive->GetData().size() > 1);
            wxASSERT(predictorRealtime->GetData().size() > 1);

            m_storagePredictorsArchive.push_back(predictorArchive);
            m_storagePredictorsRealtime.push_back(predictorRealtime);
            DeletePreprocessData();
        }

        wxLogVerbose(_("Data loaded"));

        // Instantiate a score object
        wxLogVerbose(_("Creating a criterion object."));
        asCriteria *criterion = asCriteria::GetInstance(params.GetPredictorCriteria(iStep, iPtor));
        if (criterion->NeedsDataRange()) {
            wxASSERT(m_storagePredictorsArchive.size() > iPtor);
            wxASSERT(m_storagePredictorsArchive[iPtor]);
            criterion->SetDataRange(m_storagePredictorsArchive[iPtor]);
        }
        m_storageCriteria.push_back(criterion);
        wxLogVerbose(_("Criterion object created."));

    }

#if wxUSE_GUI
    // Send events
    wxCommandEvent eventLoaded(asEVT_STATUS_LOADED);
    wxCommandEvent eventProcessing(asEVT_STATUS_PROCESSING);
    if (m_parent != nullptr) {
        m_parent->ProcessWindowEvent(eventLoaded);
        m_parent->ProcessWindowEvent(eventProcessing);
    }
#endif

    // Send data and criteria to processor
    wxLogVerbose(_("Start processing the comparison."));

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

        if (!asProcessor::GetAnalogsDates(m_storagePredictorsArchive, m_storagePredictorsRealtime, timeArrayArchive,
                                          timeArrayArchive, timeArrayTarget, timeArrayTargetLeadTime, m_storageCriteria,
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

bool asMethodForecasting::GetAnalogsSubDates(asResultsForecast &results, asParametersForecast &params,
                                             asResultsForecast &resultsPrev, int iStep)
{
    // Initialize the result object
    results.SetForecastsDirectory(m_batchForecasts->GetForecastsOutputDirectory());
    results.SetCurrentStep(iStep);
    results.Init(params, m_forecastDate);

    // Date array object instantiation for the processor
    wxLogVerbose(_("Creating a date arrays for the processor."));

    // Archive time array
    double timeStartArchive = params.GetArchiveStart();
    double timeEndArchive = params.GetArchiveEnd();
    timeEndArchive = wxMin(timeEndArchive, timeEndArchive -
                                           params.GetTimeSpanDays()); // Adjust so the predictors search won't overtake the array
    asTimeArray timeArrayArchive(timeStartArchive, timeEndArchive, params.GetAnalogsTimeStepHours(),
                                 asTimeArray::Simple);
    timeArrayArchive.Init();

    // Get last lead time of the data
    double lastLeadTime = 9999;
    for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
        if (!params.NeedsPreprocessing(iStep, iPtor)) {
            // Instanciate a predictor object
            asPredictorOper *predictorRealtime = asPredictorOper::GetInstance(
                    params.GetPredictorRealtimeDatasetId(iStep, iPtor),
                    params.GetPredictorRealtimeDataId(iStep, iPtor));
            if (!predictorRealtime) {
                return false;
            }
            predictorRealtime->SetPredictorsRealtimeDirectory(m_batchForecasts->GetPredictorsRealtimeDirectory());
            predictorRealtime->SetRunDateInUse(m_forecastDate);
            lastLeadTime = wxMin(lastLeadTime,
                                 predictorRealtime->GetForecastLeadTimeEnd() / 24.0 - params.GetTimeSpanDays());

            wxDELETE(predictorRealtime);
        } else {
            for (int iPre = 0; iPre < params.GetPreprocessSize(iStep, iPtor); iPre++) {
                // Instanciate a predictor object
                asPredictorOper *predictorRealtimePreprocess = asPredictorOper::GetInstance(
                        params.GetPreprocessRealtimeDatasetId(iStep, iPtor, iPre),
                        params.GetPreprocessRealtimeDataId(iStep, iPtor, iPre));
                if (!predictorRealtimePreprocess) {
                    return false;
                }
                predictorRealtimePreprocess->SetPredictorsRealtimeDirectory(
                        m_batchForecasts->GetPredictorsRealtimeDirectory());
                predictorRealtimePreprocess->SetRunDateInUse(m_forecastDate);
                lastLeadTime = wxMin(lastLeadTime, predictorRealtimePreprocess->GetForecastLeadTimeEnd() / 24.0 -
                                                   params.GetTimeSpanDays());

                wxDELETE(predictorRealtimePreprocess);
            }
        }
    }

    // Target time array
    vi leadTime = params.GetLeadTimeDaysVector();
    vd tmpTimeArray;
    for (unsigned int i = 0; i < leadTime.size(); i++) {
        if (leadTime[i] > lastLeadTime)
            break;

        double tmpDate = floor(m_forecastDate) + leadTime[i];
        tmpTimeArray.push_back(tmpDate);
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
    wxLogVerbose(_("Date array created."));

    // Calculate needed memory
    wxLongLong neededMem = 0;
    for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
        neededMem += (params.GetPredictorXptsnb(iStep, iPtor)) * (params.GetPredictorYptsnb(iStep, iPtor));
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

    if (freeMemSize < 0) {
        wxLogVerbose(_("Needed memory for data: %.2f Mb (cannot evaluate available memory)"), neededMemMb);
    } else {
        wxLogVerbose(_("Needed memory for data: %.2f Mb (%.2f Mb available)"), neededMemMb, freeMemMb);
        if (neededMemMb > freeMemMb) {
            wxLogError(_("Data cannot fit into available memory."));
            return false;
        }
    }

#if wxUSE_GUI
    // Send event
    wxCommandEvent eventLoading(asEVT_STATUS_LOADING);
    if (m_parent != nullptr) {
        m_parent->ProcessWindowEvent(eventLoading);
    }
#endif

    // Loop through every predictor
    for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
        wxLogVerbose(_("Loading data."));

        if (!params.NeedsPreprocessing(iStep, iPtor)) {
            // Date array object instantiation for the data loading. The array has the same length than timeArrayArchive, and the predictor dates are aligned with the target dates, but the dates are not the same.
            double ptorStartArchive =
                    timeStartArchive - params.GetTimeShiftDays() + params.GetPredictorTimeHours(iStep, iPtor) / 24.0;
            double ptorEndArchive =
                    timeEndArchive - params.GetTimeShiftDays() + params.GetPredictorTimeHours(iStep, iPtor) / 24.0;
            asTimeArray timeArrayDataArchive(ptorStartArchive, ptorEndArchive, params.GetAnalogsTimeStepHours(),
                                             params.GetTimeArrayAnalogsMode());
            timeArrayDataArchive.Init();

            double ptorStartTarget =
                    timeStartTarget - params.GetTimeShiftDays() + params.GetPredictorTimeHours(iStep, iPtor) / 24.0;
            double ptorEndTarget =
                    timeEndTarget - params.GetTimeShiftDays() + params.GetPredictorTimeHours(iStep, iPtor) / 24.0;
            asTimeArray timeArrayDataTarget(ptorStartTarget, ptorEndTarget, params.GetTargetTimeStepHours(),
                                            asTimeArray::Simple);
            timeArrayDataTarget.Init();

            // Instanciate an archive predictor object
            asPredictor *predictorArchive = asPredictor::GetInstance(
                    params.GetPredictorArchiveDatasetId(iStep, iPtor), params.GetPredictorArchiveDataId(iStep, iPtor),
                    m_batchForecasts->GetPredictorsArchiveDirectory());
            if (!predictorArchive) {
                return false;
            }

            // Select the number of members for ensemble data.
            if (predictorArchive->IsEnsemble()) {
                predictorArchive->SelectMembers(params.GetPredictorArchiveMembersNb(iStep, iPtor));
            }

            // Instanciate an realtime predictor object
            asPredictorOper *predictorRealtime = asPredictorOper::GetInstance(
                    params.GetPredictorRealtimeDatasetId(iStep, iPtor),
                    params.GetPredictorRealtimeDataId(iStep, iPtor));
            if (!predictorRealtime) {
                wxDELETE(predictorArchive);
                return false;
            }
            predictorRealtime->SetPredictorsRealtimeDirectory(m_batchForecasts->GetPredictorsRealtimeDirectory());
            predictorRealtime->SetRunDateInUse(m_forecastDate);

            // Select the number of members for ensemble data.
            if (predictorRealtime->IsEnsemble()) {
                predictorRealtime->SelectMembers(params.GetPredictorRealtimeMembersNb(iStep, iPtor));
            }

            // Restriction needed
            wxASSERT(params.GetTargetTimeStepHours() > 0);
            predictorRealtime->RestrictTimeArray(params.GetPredictorTimeHours(iStep, iPtor),
                                                 params.GetTargetTimeStepHours(),
                                                 params.GetLeadTimeNb());

            // Update
            if (!predictorRealtime->BuildFilenamesUrls()) {
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            // Check time array for real-time data
            vd listTimeArray = predictorRealtime->GetDataDates();
            wxASSERT_MSG(listTimeArray.size() >= (unsigned) timeArrayDataTarget.GetSize(),
                         wxString::Format("size of listTimeArray = %d, size of timeArrayDataTarget = %d",
                                          (int) listTimeArray.size(), (int) timeArrayDataTarget.GetSize()));
            for (unsigned int i = 0; i < (unsigned) timeArrayDataTarget.GetSize(); i++) {
                if (listTimeArray[i] != timeArrayDataTarget[i]) {
                    wxLogError(_("The real-time predictor time array is not consistent (listTimeArray[%d](%f)!=timeArrayDataTarget[%d](%f))."),
                               i, listTimeArray[i], i, timeArrayDataTarget[i]);
                    wxLogError(_("It is likely that the lead times you defined go beyond the data availability."));
                    wxDELETE(predictorArchive);
                    wxDELETE(predictorRealtime);
                    return false;
                }
            }

            // Area object instantiation
            asAreaCompGrid *area = asAreaCompGrid::GetInstance(&params, iStep, iPtor);

            // Standardize
            if (params.GetStandardize(iStep, iPtor)) {
                wxLogError(_("Data standardization is not yet implemented in operational forecasting."));
                return false;
            }

            // Archive data loading
            if (!predictorArchive->Load(area, timeArrayDataArchive, params.GetPredictorLevel(iStep, iPtor))) {
                wxLogError(_("Archive data (%s) could not be loaded."), predictorArchive->GetDataId());
                wxDELETE(area);
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }
            m_storagePredictorsArchive.push_back(predictorArchive);

            // Realtime data loading
            if (!predictorRealtime->Load(area, timeArrayDataTarget, params.GetPredictorLevel(iStep, iPtor))) {
                wxLogError(_("Real-time data (%s) could not be loaded."), predictorRealtime->GetDataId());
                wxDELETE(area);
                wxDELETE(predictorRealtime);
                return false;
            }
            wxDELETE(area);
            m_storagePredictorsRealtime.push_back(predictorRealtime);
        } else {
            int preprocessSize = params.GetPreprocessSize(iStep, iPtor);

            wxLogVerbose(_("Preprocessing data (%d predictor(s)) while loading."), preprocessSize);

            for (int iPre = 0; iPre < preprocessSize; iPre++) {

                wxLogVerbose(_("Loading predictor %d."), iPre);

                // Date array object instantiation for the data loading. The array has the same length than timeArrayArchive, and the predictor dates are aligned with the target dates, but the dates are not the same.
                double ptorStartArchive = timeStartArchive - params.GetTimeShiftDays() +
                                          params.GetPreprocessTimeHours(iStep, iPtor, iPre) / 24.0;
                double ptorEndArchive = timeEndArchive - params.GetTimeShiftDays() +
                                        params.GetPreprocessTimeHours(iStep, iPtor, iPre) / 24.0;
                asTimeArray timeArrayDataArchive(ptorStartArchive, ptorEndArchive, params.GetAnalogsTimeStepHours(),
                                                 params.GetTimeArrayAnalogsMode());
                timeArrayDataArchive.Init();

                double ptorStartTarget = timeStartTarget - params.GetTimeShiftDays() +
                                         params.GetPreprocessTimeHours(iStep, iPtor, iPre) / 24.0;
                double ptorEndTarget = timeEndTarget - params.GetTimeShiftDays() +
                                       params.GetPreprocessTimeHours(iStep, iPtor, iPre) / 24.0;
                asTimeArray timeArrayDataTarget(ptorStartTarget, ptorEndTarget, params.GetTargetTimeStepHours(), asTimeArray::Simple);
                timeArrayDataTarget.Init();

                // Instanciate an archive predictor object
                asPredictor *predictorArchivePreprocess = asPredictor::GetInstance(
                        params.GetPreprocessArchiveDatasetId(iStep, iPtor, iPre),
                        params.GetPreprocessArchiveDataId(iStep, iPtor, iPre),
                        m_batchForecasts->GetPredictorsArchiveDirectory());
                if (!predictorArchivePreprocess) {
                    return false;
                }

                // Select the number of members for ensemble data.
                if (predictorArchivePreprocess->IsEnsemble()) {
                    predictorArchivePreprocess->SelectMembers(params.GetPreprocessArchiveMembersNb(iStep, iPtor, iPre));
                }

                // Instanciate an realtime predictor object
                asPredictorOper *predictorRealtimePreprocess = asPredictorOper::GetInstance(
                        params.GetPreprocessRealtimeDatasetId(iStep, iPtor, iPre),
                        params.GetPreprocessRealtimeDataId(iStep, iPtor, iPre));
                if (!predictorRealtimePreprocess) {
                    wxDELETE(predictorArchivePreprocess);
                    return false;
                }
                predictorRealtimePreprocess->SetPredictorsRealtimeDirectory(
                        m_batchForecasts->GetPredictorsRealtimeDirectory());
                predictorRealtimePreprocess->SetRunDateInUse(m_forecastDate);

                // Select the number of members for ensemble data.
                if (predictorRealtimePreprocess->IsEnsemble()) {
                    predictorRealtimePreprocess->SelectMembers(
                            params.GetPreprocessRealtimeMembersNb(iStep, iPtor, iPre));
                }

                // Restriction needed
                wxASSERT(params.GetTargetTimeStepHours() > 0);
                predictorRealtimePreprocess->RestrictTimeArray(params.GetPreprocessTimeHours(iStep, iPtor, iPre),
                                                               params.GetTargetTimeStepHours(),
                                                               params.GetLeadTimeNb());

                // Update
                if (!predictorRealtimePreprocess->BuildFilenamesUrls()) {
                    wxDELETE(predictorArchivePreprocess);
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }

                // Check time array for real-time data
                vd listTimeArray = predictorRealtimePreprocess->GetDataDates();
                wxASSERT_MSG(listTimeArray.size() >= (unsigned) timeArrayDataTarget.GetSize(),
                             wxString::Format("listTimeArray.size() = %d, timeArrayDataTarget.GetSize() = %d",
                                              (int) listTimeArray.size(), (int) timeArrayDataTarget.GetSize()));
                for (unsigned int i = 0; i < (unsigned) timeArrayDataTarget.GetSize(); i++) {
                    if (listTimeArray[i] != timeArrayDataTarget[i]) {
                        wxLogError(_("The real-time predictor time array is not consistent (listTimeArray[%d](%f)!=timeArrayDataTarget[%d](%f))."),
                                   i, listTimeArray[i], i, timeArrayDataTarget[i]);
                        wxLogError(_("It is likely that the lead times you defined go beyond the data availability."));
                        wxDELETE(predictorArchivePreprocess);
                        wxDELETE(predictorRealtimePreprocess);
                        return false;
                    }
                }

                // Area object instantiation
                asAreaCompGrid *area = asAreaCompGrid::GetInstance(&params, iStep, iPtor);

                // Archive data loading
                if (!predictorArchivePreprocess->Load(area, timeArrayDataArchive, params.GetPreprocessLevel(iStep, iPtor, iPre))) {
                    wxLogError(_("Archive data (%s) could not be loaded."), predictorArchivePreprocess->GetDataId());
                    wxDELETE(area);
                    wxDELETE(predictorArchivePreprocess);
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }
                m_storagePredictorsArchivePreprocess.push_back(predictorArchivePreprocess);

                // Realtime data loading
                if (!predictorRealtimePreprocess->Load(area, timeArrayDataTarget, params.GetPreprocessLevel(iStep, iPtor, iPre))) {
                    wxLogError(_("Real-time data (%s) could not be loaded."), predictorRealtimePreprocess->GetDataId());
                    wxDELETE(area);
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }
                wxDELETE(area);
                m_storagePredictorsRealtimePreprocess.push_back(predictorRealtimePreprocess);

                wxASSERT_MSG(predictorArchivePreprocess->GetLatPtsnb() == predictorRealtimePreprocess->GetLatPtsnb(),
                             wxString::Format(
                                     "predictorArchivePreprocess.GetLatPtsnb()=%d, predictorRealtimePreprocess.GetLatPtsnb()=%d",
                                     predictorArchivePreprocess->GetLatPtsnb(),
                                     predictorRealtimePreprocess->GetLatPtsnb()));
                wxASSERT_MSG(predictorArchivePreprocess->GetLonPtsnb() == predictorRealtimePreprocess->GetLonPtsnb(),
                             wxString::Format(
                                     "predictorArchivePreprocess.GetLonPtsnb()=%d, predictorRealtimePreprocess.GetLonPtsnb()=%d",
                                     predictorArchivePreprocess->GetLonPtsnb(),
                                     predictorRealtimePreprocess->GetLonPtsnb()));
            }

            // Fix the criteria if S1
            params.FixCriteriaIfGradientsPreprocessed(iStep, iPtor);

            // Instantiate an archive predictor object
            asPredictor *predictorArchive = new asPredictor(*m_storagePredictorsArchivePreprocess[0]);
            if (!predictorArchive) {
                return false;
            }

            if (!Preprocess(m_storagePredictorsArchivePreprocess, params.GetPreprocessMethod(iStep, iPtor),
                            predictorArchive)) {
                wxLogError(_("Data preprocessing failed."));
                wxDELETE(predictorArchive);
                return false;
            }

            // Instantiate an realtime predictor object
            asPredictorOper *predictorRealtime = new asPredictorOper(*m_storagePredictorsRealtimePreprocess[0]);
            if (!predictorRealtime) {
                wxDELETE(predictorArchive);
                return false;
            }
            predictorRealtime->SetPredictorsRealtimeDirectory(m_batchForecasts->GetPredictorsRealtimeDirectory());

            if (!PreprocessRealtimePredictors(m_storagePredictorsRealtimePreprocess,
                                              params.GetPreprocessMethod(iStep, iPtor), predictorRealtime)) {
                wxLogError(_("Data preprocessing failed."));
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            // Standardize
            if (params.GetStandardize(iStep, iPtor)) {
                wxLogError(_("Data standardization is not yet implemented in operational forecasting."));
                return false;
            }

            wxASSERT(predictorArchive->GetLatPtsnb() == predictorRealtime->GetLatPtsnb());
            wxASSERT(predictorArchive->GetLonPtsnb() == predictorRealtime->GetLonPtsnb());
            m_storagePredictorsArchive.push_back(predictorArchive);
            m_storagePredictorsRealtime.push_back(predictorRealtime);
            DeletePreprocessData();
        }

        wxLogVerbose(_("Data loaded"));

        // Instantiate a score object
        wxLogVerbose(_("Creating a criterion object."));
        asCriteria *criterion = asCriteria::GetInstance(params.GetPredictorCriteria(iStep, iPtor));
        if (criterion->NeedsDataRange()) {
            wxASSERT(m_storagePredictorsArchive.size() > iPtor);
            wxASSERT(m_storagePredictorsArchive[iPtor]);
            criterion->SetDataRange(m_storagePredictorsArchive[iPtor]);
        }
        m_storageCriteria.push_back(criterion);
        wxLogVerbose(_("Criterion object created."));

    }

#if wxUSE_GUI
    // Send events
    wxCommandEvent eventLoaded(asEVT_STATUS_LOADED);
    wxCommandEvent eventProcessing(asEVT_STATUS_PROCESSING);
    if (m_parent != nullptr) {
        m_parent->ProcessWindowEvent(eventLoaded);
        m_parent->ProcessWindowEvent(eventProcessing);
    }
#endif

    // Send data and criteria to processor
    wxLogVerbose(_("Start processing the comparison."));

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

        if (!asProcessor::GetAnalogsSubDates(m_storagePredictorsArchive, m_storagePredictorsRealtime, timeArrayArchive,
                                             timeArrayTarget, anaDatesPrev, m_storageCriteria, &params, iStep, anaDates,
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

bool asMethodForecasting::GetAnalogsValues(asResultsForecast &results, asParametersForecast &params, int iStep)
{
    // Initialize the result object
    results.SetForecastsDirectory(m_batchForecasts->GetForecastsOutputDirectory());
    results.SetCurrentStep(iStep);
    results.SetPredictandDatasetId(m_predictandDB->GetDatasetId());
    results.SetPredictandParameter(m_predictandDB->GetDataParameter());
    results.SetPredictandTemporalResolution(m_predictandDB->GetDataTemporalResolution());
    results.SetPredictandSpatialAggregation(m_predictandDB->GetDataSpatialAggregation());

    // Set the predictands values to the corresponding analog dates
    wxASSERT(m_predictandDB);

    // Extract the stations IDs and coordinates
    wxASSERT(m_predictandDB->GetStationsNb() > 0);
    a1i stationsId = m_predictandDB->GetStationsIdArray();
    wxASSERT(stationsId.size() > 0);
    results.SetStationIds(stationsId);
    results.SetStationOfficialIds(m_predictandDB->GetStationOfficialIdsArray());
    results.SetStationNames(m_predictandDB->GetStationNamesArray());
    results.SetStationHeights(m_predictandDB->GetStationHeightsArray());
    results.SetStationXCoords(m_predictandDB->GetStationXCoordsArray());
    results.SetStationYCoords(m_predictandDB->GetStationYCoordsArray());
    a1f refAxis = m_predictandDB->GetReferenceAxis();
    results.SetReferenceAxis(refAxis);
    a2f refValues = m_predictandDB->GetReferenceValuesArray();
    results.SetReferenceValues(refValues);

    a1f leadTimes = results.GetTargetDates();

    wxLogVerbose(_("Start setting the predictand values to the corresponding analog dates."));

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

            if (!asProcessor::GetAnalogsValues(*m_predictandDB, anaDates, &params, anaValues)) {
                wxLogError(_("Failed setting the predictand values to the corresponding analog dates."));
                return false;
            }

            va2f valuesRaw = anaValues.GetAnalogsValuesRaw();
            wxASSERT(valuesRaw[0].rows() == 1);
            a1f rowValuesRaw = valuesRaw[0].row(0);
            results.SetAnalogsValuesRaw(iLead, iStat, rowValuesRaw);
        }
    }

#if wxUSE_GUI
    // Send event
    wxCommandEvent eventProcessed(asEVT_STATUS_PROCESSED);
    if (m_parent != nullptr) {
        m_parent->ProcessWindowEvent(eventProcessed);
    }
#endif

    wxLogVerbose(_("Predictands association over."));

    return true;
}

double asMethodForecasting::GetEffectiveArchiveDataStart(asParameters *params) const
{
    return GetTimeStartArchive(params);
}

double asMethodForecasting::GetEffectiveArchiveDataEnd(asParameters *params) const
{
    return GetTimeEndArchive(params);
}

void asMethodForecasting::Cleanup()
{
    DeletePreprocessData();

    if (!m_storagePredictorsArchive.empty()) {
        for (unsigned int i = 0; i < m_storagePredictorsArchive.size(); i++) {
            wxDELETE(m_storagePredictorsArchive[i]);
        }
        m_storagePredictorsArchive.resize(0);
    }

    if (!m_storagePredictorsRealtime.empty()) {
        for (unsigned int i = 0; i < m_storagePredictorsRealtime.size(); i++) {
            wxDELETE(m_storagePredictorsRealtime[i]);
        }
        m_storagePredictorsRealtime.resize(0);
    }

    if (!m_storageCriteria.empty()) {
        for (unsigned int i = 0; i < m_storageCriteria.size(); i++) {
            wxDELETE(m_storageCriteria[i]);
        }
        m_storageCriteria.resize(0);
    }

    // Do not delete preloaded data here !
}

void asMethodForecasting::DeletePreprocessData()
{
    for (unsigned int i = 0; i < m_storagePredictorsArchivePreprocess.size(); i++) {
        wxDELETE(m_storagePredictorsArchivePreprocess[i]);
    }
    m_storagePredictorsArchivePreprocess.resize(0);

    for (unsigned int i = 0; i < m_storagePredictorsRealtimePreprocess.size(); i++) {
        wxDELETE(m_storagePredictorsRealtimePreprocess[i]);
    }
    m_storagePredictorsRealtimePreprocess.resize(0);
}