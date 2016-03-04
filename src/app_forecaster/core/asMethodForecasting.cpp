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
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 */

#include "asMethodForecasting.h"

#include "asDataPredictand.h"
#include "asResultsAnalogsDates.h"
#include "asResultsAnalogsValues.h"
#include "asPredictorCriteria.h"
#include "asTimeArray.h"
#include "asGeoAreaCompositeGrid.h"
#include "asProcessor.h"
#include "asPreprocessor.h"
#ifndef UNIT_TESTING
    #include "AtmoswingAppForecaster.h"
#endif

asMethodForecasting::asMethodForecasting(asBatchForecasts* batchForecasts, wxWindow* parent)
:
asMethodStandard()
{
    m_batchForecasts = batchForecasts;
    m_forecastDate = NaNDouble;
    m_paramsFilePath = wxEmptyString;
    m_predictandDBFilePath = wxEmptyString;
    m_parent = parent;
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

    #if wxUSE_GUI
        if (g_responsive) wxGetApp().Yield();
    #endif
    m_cancel = false;

    if(asTools::IsNaN(m_forecastDate))
    {
        asLogError(_("The date of the forecast has not been defined."));
        return false;
    }

    try
    {
        #if wxUSE_GUI
            // Switch off all leds
            wxCommandEvent eventStart (asEVT_STATUS_STARTING);
            if (m_parent != NULL) {
                m_parent->ProcessWindowEvent(eventStart);
            }
        #endif

        // Get paths
        wxString forecastParametersDir = m_batchForecasts->GetParametersFileDirectory();
        wxString predictandDBDir = m_batchForecasts->GetPredictandDBDirectory();

        // Execute the forecasts
        for (int i=0; i<m_batchForecasts->GetForecastsNb(); i++)
        {
            #if wxUSE_GUI
                if (g_responsive) wxGetApp().Yield();
                if (m_cancel) return false;

                // Send event
                wxCommandEvent eventRunning (asEVT_STATUS_RUNNING);
                eventRunning.SetInt(i);
                if (m_parent != NULL) {
                    m_parent->ProcessWindowEvent(eventRunning);
                }

                if (g_responsive) wxGetApp().Yield();
            #endif

            // Load parameters
            m_paramsFilePath = forecastParametersDir + DS + m_batchForecasts->GetForecastFileName(i);
            asParametersForecast params;
            if(!params.LoadFromFile(m_paramsFilePath)) return false;
            params.InitValues();

            m_predictandDBFilePath = predictandDBDir + DS + params.GetPredictandDatabase();

            #if wxUSE_GUI
                if (g_responsive) wxGetApp().Yield();
            #endif

            // Watch
            wxStopWatch sw;

            // Forecast
            if(!Forecast(params))
            {
                asLogError(_("The forecast could not be achived"));

                #if wxUSE_GUI
                    // Send event
                    wxCommandEvent eventFailed (asEVT_STATUS_FAILED);
                    eventFailed.SetInt(i);
                    if (m_parent != NULL) {
                        m_parent->ProcessWindowEvent(eventFailed);
                    }
                #endif
            }
            else
            {
                // Display processing time
                asLogMessageImportant(wxString::Format(_("Processing of the forecast \"%s\" - \"%s\" took %ld ms to execute"), params.GetMethodIdDisplay(), params.GetSpecificTagDisplay(), sw.Time()));

                #if wxUSE_GUI
                    // Send event
                    wxCommandEvent eventSuccess (asEVT_STATUS_SUCCESS);
                    eventSuccess.SetInt(i);
                    if (m_parent != NULL) {
                        m_parent->ProcessWindowEvent(eventSuccess);
                    }
                #endif
            }
        }

        // Optional exports
        if (m_batchForecasts->HasExports())
        {
            if (m_batchForecasts->ExportSyntheticXml())
            {
                if (!m_aggregator.ExportSyntheticXml(m_batchForecasts->GetExportsOutputDirectory()))
                {
                    asLogError(_("The export of the synthetic xml failed."));
                }
            }
        }
    }
    catch(asException& e)
    {
        wxString fullMessage = e.GetFullMessage();
        if (!fullMessage.IsEmpty())
        {
            #if wxUSE_GUI
                if (!g_silentMode)
                    wxMessageBox(fullMessage);
            #else
                asLogError(fullMessage);
            #endif
        }
        return false;
    }

    asLogState(_("Forecasting over."));
    Cleanup();

    return true;
}

bool asMethodForecasting::Forecast(asParametersForecast &params)
{
    // Process every step one after the other
    int stepsNb = params.GetStepsNb();

    // Download real-time predictors
    asResultsAnalogsForecast resultsCheck;
    resultsCheck.SetForecastsDirectory(m_batchForecasts->GetForecastsOutputDirectory());
    bool forecastDateChanged = true;
    while(forecastDateChanged)
    {
        #if wxUSE_GUI
            if (g_responsive) wxGetApp().Yield();
        #endif
        if (m_cancel) return false;

        // Check if result already exists
        resultsCheck.SetCurrentStep(stepsNb-1);
        resultsCheck.Init(params, m_forecastDate);
        if (resultsCheck.Exists())
        {
            asLogMessage(_("Forecast already exists."));
            m_resultsFilePaths.push_back(resultsCheck.GetFilePath());
			if (m_batchForecasts->HasExports()) {
				asResultsAnalogsForecast * results = new asResultsAnalogsForecast();
				results->Load(resultsCheck.GetFilePath());
				m_aggregator.Add(results);
			}
            #if wxUSE_GUI
                if (g_responsive) wxGetApp().Yield();
            #endif
            return true;
        }

        // Send event
        #if wxUSE_GUI
            wxCommandEvent eventDownloading (asEVT_STATUS_DOWNLOADING);
            if (m_parent != NULL) {
                m_parent->ProcessWindowEvent(eventDownloading);
            }

            if (g_responsive) wxGetApp().Yield();
        #endif

        forecastDateChanged = false;
        for (int i_step=0; i_step<stepsNb; i_step++)
        {
            #if wxUSE_GUI
                if (g_responsive) wxGetApp().Yield();
            #endif
            if (m_cancel) return false;
            if(!DownloadRealtimePredictors(params, i_step, forecastDateChanged)) return false;
        }

        // Check again if result already exists (if change in date)
        resultsCheck.Init(params, m_forecastDate);
        if (resultsCheck.Exists())
        {
            asLogMessage(_("Forecast already exists."));
            m_resultsFilePaths.push_back(resultsCheck.GetFilePath());
			if (m_batchForecasts->HasExports()) {
				asResultsAnalogsForecast * results = new asResultsAnalogsForecast();
				results->Load(resultsCheck.GetFilePath());
				m_aggregator.Add(results);
			}
            #if wxUSE_GUI
                if (g_responsive) wxGetApp().Yield();
            #endif
            return true;
        }

        // Send event
        #if wxUSE_GUI
            wxCommandEvent eventDownloaded (asEVT_STATUS_DOWNLOADED);
            if (m_parent != NULL) {
                m_parent->ProcessWindowEvent(eventDownloaded);
            }
        #endif
    }

    #if wxUSE_GUI
        if (g_responsive) wxGetApp().Yield();
    #endif
    if (m_cancel) return false;

    // Load the Predictand DB
    asLogMessage(_("Loading the Predictand DB."));
    if(!LoadPredictandDB(m_predictandDBFilePath)) return false;
    asLogMessage(_("Predictand DB loaded."));

   #if wxUSE_GUI
        if (g_responsive) wxGetApp().Yield();
    #endif
    if (m_cancel) return false;

    // Resulting object
    asResultsAnalogsForecast * resultsPrevious = new asResultsAnalogsForecast();
    asResultsAnalogsForecast * results = new asResultsAnalogsForecast();
    results->SetForecastsDirectory(m_batchForecasts->GetForecastsOutputDirectory());

    for (int i_step=0; i_step<stepsNb; i_step++)
    {
        #if wxUSE_GUI
            if (g_responsive) wxGetApp().Yield();
        #endif
        if (m_cancel) {
            wxDELETE(results);
            wxDELETE(resultsPrevious);
            return false;
        }

        if (i_step==0)
        {
            if(!GetAnalogsDates(*results, params, i_step)) {
                wxDELETE(results);
                wxDELETE(resultsPrevious);
                return false;
            }
        }
        else
        {
            if(!GetAnalogsSubDates(*results, params, *resultsPrevious, i_step)) {
                wxDELETE(results);
                wxDELETE(resultsPrevious);
                return false;
            }
        }

        // At last get the values
        if (i_step==stepsNb-1)
        {
            #if wxUSE_GUI
                if (g_responsive) wxGetApp().Yield();
            #endif
            if (m_cancel) {
                wxDELETE(results);
                wxDELETE(resultsPrevious);
                return false;
            }

            if(!GetAnalogsValues(*results, params, i_step)) {
                wxDELETE(results);
                wxDELETE(resultsPrevious);
                return false;
            }

            #if wxUSE_GUI
                // Send event
                wxCommandEvent eventSaving (asEVT_STATUS_SAVING);
                if (m_parent != NULL) {
                    m_parent->ProcessWindowEvent(eventSaving);
                }
            #endif

            try
            {
                results->Save();
            }
            catch(asException& e)
            {
                wxString fullMessage = e.GetFullMessage();
                if (!fullMessage.IsEmpty())
                {
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
                wxCommandEvent eventSaved (asEVT_STATUS_SAVED);
                if (m_parent != NULL) {
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

bool asMethodForecasting::DownloadRealtimePredictors(asParametersForecast &params, int i_step, bool &forecastDateChanged)
{
    // Get preferences
    wxConfigBase *pConfig = wxFileConfig::Get();
    long maxPrevStepsNbDef = 5;
    long maxPrevStepsNb = pConfig->Read("/Internet/MaxPreviousStepsNb", maxPrevStepsNbDef);

    for(int i_ptor=0;i_ptor<params.GetPredictorsNb(i_step);i_ptor++)
    {
        #if wxUSE_GUI
            if (g_responsive) wxGetApp().Yield();
        #endif
        if (m_cancel) return false;

        asLogMessage(_("Downloading data."));

        #if wxUSE_GUI
            if (g_responsive) wxGetApp().Yield();
        #endif

        if(!params.NeedsPreprocessing(i_step, i_ptor))
        {
            // Instanciate a predictor object
            asDataPredictorRealtime* predictorRealtime = asDataPredictorRealtime::GetInstance(params.GetPredictorRealtimeDatasetId(i_step, i_ptor), params.GetPredictorRealtimeDataId(i_step, i_ptor));
            if (!predictorRealtime)
            {
                wxDELETE(predictorRealtime);
                return false;
            }
            predictorRealtime->SetPredictorsRealtimeDirectory(m_batchForecasts->GetPredictorsRealtimeDirectory());

            // Set the desired forecasting date
            m_forecastDate = predictorRealtime->SetRunDateInUse(m_forecastDate);

            // Check if result already exists
            asResultsAnalogsForecast resultsCheck;
            resultsCheck.SetForecastsDirectory(m_batchForecasts->GetForecastsOutputDirectory());
            resultsCheck.SetCurrentStep(params.GetStepsNb()-1);
            resultsCheck.Init(params, m_forecastDate);
            if (resultsCheck.Exists())
            {
                asLogMessage(_("Forecast already exists."));
                #if wxUSE_GUI
                    if (g_responsive) wxGetApp().Yield();
                #endif
                wxDELETE(predictorRealtime);
                return true;
            }

            // Restriction needed
            wxASSERT(params.GetTimeArrayTargetTimeStepHours()>0);
            predictorRealtime->RestrictTimeArray(params.GetPredictorTimeHours(i_step, i_ptor), params.GetTimeArrayTargetTimeStepHours());

            // Update forecasting date
            if(!predictorRealtime->BuildFilenamesUrls())
            {
                wxDELETE(predictorRealtime);
                return false;
            }

            // Realtime data downloading
            int counterFails = 0;
            while (true)
            {
                #if wxUSE_GUI
                    if (g_responsive) wxGetApp().Yield();
                #endif
                if (m_cancel)
                {
                    wxDELETE(predictorRealtime);
                    return false;
                }

                // Download predictor
                int resDownload = predictorRealtime->Download();
                if (resDownload==asSUCCESS)
                {
                    break;
                }
                else if (resDownload==asFAILED)
                {
                    if (counterFails<maxPrevStepsNb)
                    {
                        // Try to download older data
                        m_forecastDate = predictorRealtime->DecrementRunDateInUse();
                        // Check if result already exists
                        resultsCheck.SetCurrentStep(params.GetStepsNb()-1);
                        resultsCheck.Init(params, m_forecastDate);
                        if (resultsCheck.Exists())
                        {
                            asLogMessage(_("Forecast already exists."));
                            #if wxUSE_GUI
                                if (g_responsive) wxGetApp().Yield();
                            #endif
                            wxDELETE(predictorRealtime);
                            return true;
                        }
                        forecastDateChanged = true;
                        predictorRealtime->BuildFilenamesUrls();
                        counterFails++;
                    }
                    else
                    {
                        asLogError(_("The maximum attempts is reached to download the real-time predictor. Forecasting failed."));
                        wxDELETE(predictorRealtime);
                        return false;
                    }
                }
                else
                {
                    // Canceled for example.
                    wxDELETE(predictorRealtime);
                    return false;
                }
            }
            m_forecastDate = predictorRealtime->GetRunDateInUse();
            wxDELETE(predictorRealtime);
        }
        else
        {
            int preprocessSize = params.GetPreprocessSize(i_step, i_ptor);

            for (int i_prepro=0; i_prepro<preprocessSize; i_prepro++)
            {
                #if wxUSE_GUI
                    if (g_responsive) wxGetApp().Yield();
                #endif
                if (m_cancel) return false;

                // Instanciate a predictor object
                asDataPredictorRealtime* predictorRealtimePreprocess = asDataPredictorRealtime::GetInstance(params.GetPreprocessRealtimeDatasetId(i_step, i_ptor, i_prepro), params.GetPreprocessRealtimeDataId(i_step, i_ptor, i_prepro));
                if (!predictorRealtimePreprocess)
                {
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }
                predictorRealtimePreprocess->SetPredictorsRealtimeDirectory(m_batchForecasts->GetPredictorsRealtimeDirectory());

                // Set the desired forecasting date
                m_forecastDate = predictorRealtimePreprocess->SetRunDateInUse(m_forecastDate);

                // Restriction needed
                wxASSERT(params.GetTimeArrayTargetTimeStepHours()>0);
                predictorRealtimePreprocess->RestrictTimeArray(params.GetPreprocessTimeHours(i_step, i_ptor, i_prepro), params.GetTimeArrayTargetTimeStepHours());

                // Update forecasting date
                if(!predictorRealtimePreprocess->BuildFilenamesUrls())
                {
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }

                // Realtime data downloading
                int counterFails = 0;
                while (true)
                {
                    #if wxUSE_GUI
                        if (g_responsive) wxGetApp().Yield();
                    #endif
                    if (m_cancel)
                    {
                        wxDELETE(predictorRealtimePreprocess);
                        return false;
                    }

                    // Download predictor
                    int resDownload = predictorRealtimePreprocess->Download();
                    if (resDownload==asSUCCESS)
                    {
                        break;
                    }
                    else if (resDownload==asFAILED)
                    {
                        if (counterFails<maxPrevStepsNb)
                        {
                            // Try to download older data
                            m_forecastDate = predictorRealtimePreprocess->DecrementRunDateInUse();
                            forecastDateChanged = true;
                            predictorRealtimePreprocess->BuildFilenamesUrls();
                            counterFails++;
                        }
                        else
                        {
                            asLogError(_("The maximum attempts is reached to download the real-time predictor. Forecasting failed."));
                            wxDELETE(predictorRealtimePreprocess);
                            return false;
                        }
                    }
                    else
                    {
                        // Canceled for example.
                        wxDELETE(predictorRealtimePreprocess);
                        return false;
                    }
                }
                m_forecastDate = predictorRealtimePreprocess->GetRunDateInUse();
                wxDELETE(predictorRealtimePreprocess);
            }
        }

        asLogMessage(_("Data downloaded."));
    }

    return true;
}

bool asMethodForecasting::GetAnalogsDates(asResultsAnalogsForecast &results, asParametersForecast &params, int i_step)
{
    // Get preferences
    int linAlgebraMethod = (int)(wxFileConfig::Get()->Read("/Processing/LinAlgebra", (long)asLIN_ALGEBRA_NOVAR));

    // Initialize the result object
    results.SetForecastsDirectory(m_batchForecasts->GetForecastsOutputDirectory());
    results.SetCurrentStep(i_step);
    results.Init(params, m_forecastDate);

    // Archive time array
    double timeStartArchive = params.GetArchiveStart(); 
    double timeEndArchive = params.GetArchiveEnd();
    timeEndArchive = wxMin(timeEndArchive, timeEndArchive-params.GetTimeSpanDays()); // Adjust so the predictors search won't overtake the array
    asTimeArray timeArrayArchive(timeStartArchive, timeEndArchive, params.GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
    timeArrayArchive.Init();

    // Get last lead time of the data
    double lastLeadTime = 9999;
    for(int i_ptor=0;i_ptor<params.GetPredictorsNb(i_step);i_ptor++)
    {
        if(!params.NeedsPreprocessing(i_step, i_ptor))
        {
            // Instanciate a predictor object
            asDataPredictorRealtime* predictorRealtime = asDataPredictorRealtime::GetInstance(params.GetPredictorRealtimeDatasetId(i_step, i_ptor), params.GetPredictorRealtimeDataId(i_step, i_ptor));
            if (!predictorRealtime)
            {
                return false;
            }

            predictorRealtime->SetRunDateInUse(m_forecastDate);
            lastLeadTime = wxMin(lastLeadTime, predictorRealtime->GetForecastLeadTimeEnd()/24.0 - params.GetTimeSpanDays());

            wxDELETE(predictorRealtime);
        }
        else
        {
            for (int i_prepro=0; i_prepro<params.GetPreprocessSize(i_step, i_ptor); i_prepro++)
            {
                // Instanciate a predictor object
                asDataPredictorRealtime* predictorRealtimePreprocess = asDataPredictorRealtime::GetInstance(params.GetPreprocessRealtimeDatasetId(i_step, i_ptor, i_prepro), params.GetPreprocessRealtimeDataId(i_step, i_ptor, i_prepro));
                if (!predictorRealtimePreprocess)
                {
                    return false;
                }

                predictorRealtimePreprocess->SetRunDateInUse(m_forecastDate);
                lastLeadTime = wxMin(lastLeadTime, predictorRealtimePreprocess->GetForecastLeadTimeEnd()/24.0 - params.GetTimeSpanDays());

                wxDELETE(predictorRealtimePreprocess);
            }
        }
    }

    // Target time array
    VectorInt leadTime = params.GetLeadTimeDaysVector();
    VectorDouble tmpTimeArray;
    for (unsigned int i=0; i<leadTime.size(); i++)
    {
        if (leadTime[i]>lastLeadTime) break;

        double tmpDate = floor(m_forecastDate)+leadTime[i];
        tmpTimeArray.push_back(tmpDate);
    }
    wxASSERT(tmpTimeArray.size()>0);
    double timeStartTarget = tmpTimeArray[0];
    double timeEndTarget = tmpTimeArray[tmpTimeArray.size()-1];

    asTimeArray timeArrayTarget = asTimeArray(tmpTimeArray);
    timeArrayTarget.Init();

    // Check archive time array length
    if(timeArrayArchive.GetSize()<100)
    {
        asLogError(wxString::Format(_("The time array is not consistent in asMethodForecasting::GetAnalogsDates: size=%d."),timeArrayArchive.GetSize()));
        return false;
    }
    asLogMessage(_("Date array created."));

    // Calculate needed memory
    wxLongLong neededMem = 0;
    for(int i_ptor=0;i_ptor<params.GetPredictorsNb(i_step);i_ptor++)
    {
        neededMem += (params.GetPredictorXptsnb(i_step, i_ptor))
                    * (params.GetPredictorYptsnb(i_step, i_ptor));
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
        asLogMessage(wxString::Format(_("Needed memory for data: %.2f Mb (cannot evaluate available memory)"), neededMemMb));
    }
    else
    {
        asLogMessage(wxString::Format(_("Needed memory for data: %.2f Mb (%.2f Mb available)"), neededMemMb, freeMemMb));
        if(neededMemMb>freeMemMb)
        {
            asLogError(_("Data cannot fit into available memory."));
            return false;
        }
    }

    #if wxUSE_GUI
        // Send event
        wxCommandEvent eventLoading (asEVT_STATUS_LOADING);
        if (m_parent != NULL) {
            m_parent->ProcessWindowEvent(eventLoading);
        }
    #endif

    // Loop through every predictor
    for(int i_ptor=0;i_ptor<params.GetPredictorsNb(i_step);i_ptor++)
    {
        asLogMessage(wxString::Format(_("Loading data (step %d, predictor nb %d)."), i_step, i_ptor));

        #if wxUSE_GUI
            if (g_responsive) wxGetApp().Yield();
        #endif

        if(!params.NeedsPreprocessing(i_step, i_ptor))
        {
            // Date array object instantiation for the data loading. The array has the same length than timeArrayArchive, and the predictor dates are aligned with the target dates, but the dates are not the same.
            double ptorStartArchive = timeStartArchive-params.GetTimeShiftDays()+params.GetPredictorTimeHours(i_step, i_ptor)/24.0;
            double ptorEndArchive = timeEndArchive-params.GetTimeShiftDays()+params.GetPredictorTimeHours(i_step, i_ptor)/24.0;
            asTimeArray timeArrayDataArchive(ptorStartArchive, ptorEndArchive, params.GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
            timeArrayDataArchive.Init();

            double ptorStartTarget = timeStartTarget-params.GetTimeShiftDays()+params.GetPredictorTimeHours(i_step, i_ptor)/24.0;
            double ptorEndTarget = timeEndTarget-params.GetTimeShiftDays()+params.GetPredictorTimeHours(i_step, i_ptor)/24.0;
            asTimeArray timeArrayDataTarget(ptorStartTarget, ptorEndTarget, params.GetTimeArrayTargetTimeStepHours(), asTimeArray::Simple);
            timeArrayDataTarget.Init();

            // Instanciate an archive predictor object
            asDataPredictorArchive* predictorArchive = asDataPredictorArchive::GetInstance(params.GetPredictorArchiveDatasetId(i_step, i_ptor), params.GetPredictorArchiveDataId(i_step, i_ptor), m_batchForecasts->GetPredictorsArchiveDirectory());
            if (!predictorArchive)
            {
                return false;
            }

            // Instanciate an realtime predictor object
            asDataPredictorRealtime* predictorRealtime = asDataPredictorRealtime::GetInstance(params.GetPredictorRealtimeDatasetId(i_step, i_ptor), params.GetPredictorRealtimeDataId(i_step, i_ptor));
            if (!predictorRealtime)
            {
                wxDELETE(predictorArchive);
                return false;
            }
            predictorRealtime->SetPredictorsRealtimeDirectory(m_batchForecasts->GetPredictorsRealtimeDirectory());
            predictorRealtime->SetRunDateInUse(m_forecastDate);

            // Restriction needed
            wxASSERT(params.GetTimeArrayTargetTimeStepHours()>0);
            predictorRealtime->RestrictTimeArray(params.GetPredictorTimeHours(i_step, i_ptor), params.GetTimeArrayTargetTimeStepHours());

            // Update
            if(!predictorRealtime->BuildFilenamesUrls())
            {
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            // Check time array for real-time data
            VectorDouble listTimeArray = predictorRealtime->GetDataDates();
            wxASSERT_MSG(listTimeArray.size()>=(unsigned)timeArrayDataTarget.GetSize(), wxString::Format("size of listTimeArray = %d, size of timeArrayDataTarget = %d", (int)listTimeArray.size(), (int)timeArrayDataTarget.GetSize()));
            for (int i=0; i<timeArrayDataTarget.GetSize(); i++)
            {
                if(listTimeArray[i]!=timeArrayDataTarget[i])
                {
                    asLogError(wxString::Format(_("The real-time predictor time array is not consistent (listTimeArray[%d](%f)!=timeArrayDataTarget[%d](%f))."), i, listTimeArray[i], i, timeArrayDataTarget[i]));
                    asLogError(_("It is likely that the lead times you defined go beyond the data availability."));
                    wxDELETE(predictorArchive);
                    wxDELETE(predictorRealtime);
                    return false;
                }
            }

            // Area object instantiation
            asGeoAreaCompositeGrid* area = asGeoAreaCompositeGrid::GetInstance(params.GetPredictorGridType(i_step, i_ptor), 
                params.GetPredictorXmin(i_step, i_ptor), 
                params.GetPredictorXptsnb(i_step, i_ptor), 
                params.GetPredictorXstep(i_step, i_ptor), 
                params.GetPredictorYmin(i_step, i_ptor), 
                params.GetPredictorYptsnb(i_step, i_ptor), 
                params.GetPredictorYstep(i_step, i_ptor), 
                params.GetPredictorLevel(i_step, i_ptor), 
                asNONE, 
                params.GetPredictorFlatAllowed(i_step, i_ptor));

            // Check the starting dates coherence
            if (predictorArchive->GetOriginalProviderStart()>ptorStartArchive)
            {
                asLogError(wxString::Format(_("The first year defined in the parameters (%s) is prior to the start date of the data (%s)."), asTime::GetStringTime(ptorStartArchive), asTime::GetStringTime(predictorArchive->GetOriginalProviderStart())));
                wxDELETE(area);
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            // Archive data loading
            asLogMessage(_("Loading archive data."));
            if(!predictorArchive->Load(area, timeArrayDataArchive))
            {
                asLogError(_("Archive data could not be loaded."));
                wxDELETE(area);
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            wxASSERT(predictorArchive->GetData().size()>1);
            m_storagePredictorsArchive.push_back(predictorArchive);

            // Realtime data loading
            asLogMessage(_("Loading GCM forecast data."));
            if(!predictorRealtime->Load(area, timeArrayDataTarget))
            {
                asLogError(_("Real-time data could not be loaded."));
                wxDELETE(area);
                wxDELETE(predictorRealtime);
                return false;
            }
            
            wxDELETE(area);
            wxASSERT(predictorRealtime->GetData().size()>1);
            m_storagePredictorsRealtime.push_back(predictorRealtime);
        }
        else
        {
            int preprocessSize = params.GetPreprocessSize(i_step, i_ptor);

            asLogMessage(wxString::Format(_("Preprocessing data (%d predictor(s)) while loading."), preprocessSize));

            for (int i_prepro=0; i_prepro<preprocessSize; i_prepro++)
            {
                #if wxUSE_GUI
                    if (g_responsive) wxGetApp().Yield();
                #endif

                // Date array object instantiation for the data loading. The array has the same length than timeArrayArchive, and the predictor dates are aligned with the target dates, but the dates are not the same.
                double ptorStartArchive = timeStartArchive-params.GetTimeShiftDays()+params.GetPreprocessTimeHours(i_step, i_ptor, i_prepro)/24.0;
                double ptorEndArchive = timeEndArchive-params.GetTimeShiftDays()+params.GetPreprocessTimeHours(i_step, i_ptor, i_prepro)/24.0;
                asTimeArray timeArrayDataArchive(ptorStartArchive, ptorEndArchive, params.GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
                timeArrayDataArchive.Init();

                double ptorStartTarget = timeStartTarget-params.GetTimeShiftDays()+params.GetPreprocessTimeHours(i_step, i_ptor, i_prepro)/24.0;
                double ptorEndTarget = timeEndTarget-params.GetTimeShiftDays()+params.GetPreprocessTimeHours(i_step, i_ptor, i_prepro)/24.0;
                asTimeArray timeArrayDataTarget(ptorStartTarget, ptorEndTarget, params.GetTimeArrayTargetTimeStepHours(), asTimeArray::Simple);
                timeArrayDataTarget.Init();

                // Instanciate an archive predictor object
                asDataPredictorArchive* predictorArchivePreprocess = asDataPredictorArchive::GetInstance(params.GetPreprocessArchiveDatasetId(i_step, i_ptor, i_prepro), params.GetPreprocessArchiveDataId(i_step, i_ptor, i_prepro), m_batchForecasts->GetPredictorsArchiveDirectory());
                if (!predictorArchivePreprocess)
                {
                    return false;
                }

                // Instanciate an realtime predictor object
                asDataPredictorRealtime* predictorRealtimePreprocess = asDataPredictorRealtime::GetInstance(params.GetPreprocessRealtimeDatasetId(i_step, i_ptor, i_prepro), params.GetPreprocessRealtimeDataId(i_step, i_ptor, i_prepro));
                if (!predictorRealtimePreprocess)
                {
                    wxDELETE(predictorArchivePreprocess);
                    return false;
                }
                predictorRealtimePreprocess->SetPredictorsRealtimeDirectory(m_batchForecasts->GetPredictorsRealtimeDirectory());
                predictorRealtimePreprocess->SetRunDateInUse(m_forecastDate);

                // Restriction needed
                wxASSERT(params.GetTimeArrayTargetTimeStepHours()>0);
                predictorRealtimePreprocess->RestrictTimeArray(params.GetPreprocessTimeHours(i_step, i_ptor, i_prepro), params.GetTimeArrayTargetTimeStepHours());

                // Update
                if(!predictorRealtimePreprocess->BuildFilenamesUrls())
                {
                    wxDELETE(predictorArchivePreprocess);
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }

                // Check time array for real-time data
                VectorDouble listTimeArray = predictorRealtimePreprocess->GetDataDates();
                wxASSERT_MSG(listTimeArray.size()>=(unsigned)timeArrayDataTarget.GetSize(), wxString::Format("size of listTimeArray = %d, size of timeArrayDataTarget = %d", (int)listTimeArray.size(), (int)timeArrayDataTarget.GetSize()));

                for (int i=0; i<timeArrayDataTarget.GetSize(); i++)
                {
                    if(listTimeArray[i]!=timeArrayDataTarget[i])
                    {
                        asLogError(wxString::Format(_("The real-time predictor time array is not consistent (listTimeArray[%d](%f)!=timeArrayDataTarget[%d](%f))."), i, listTimeArray[i], i, timeArrayDataTarget[i]));
                        asLogError(_("It is likely that the lead times you defined go beyond the data availability."));
                        wxDELETE(predictorArchivePreprocess);
                        wxDELETE(predictorRealtimePreprocess);
                        return false;
                    }
                }

                // Area object instantiation
                asGeoAreaCompositeGrid* area = asGeoAreaCompositeGrid::GetInstance(params.GetPredictorGridType(i_step, i_ptor), params.GetPredictorXmin(i_step, i_ptor), params.GetPredictorXptsnb(i_step, i_ptor), params.GetPredictorXstep(i_step, i_ptor), params.GetPredictorYmin(i_step, i_ptor), params.GetPredictorYptsnb(i_step, i_ptor), params.GetPredictorYstep(i_step, i_ptor), params.GetPreprocessLevel(i_step, i_ptor, i_prepro), asNONE, params.GetPredictorFlatAllowed(i_step, i_ptor));

                // Check the starting dates coherence
                if (predictorArchivePreprocess->GetOriginalProviderStart()>ptorStartArchive)
                {
                    asLogError(wxString::Format(_("The first year defined in the parameters (%s) is prior to the start date of the data (%s)."), asTime::GetStringTime(ptorStartArchive), asTime::GetStringTime(predictorArchivePreprocess->GetOriginalProviderStart())));
                    wxDELETE(area);
                    wxDELETE(predictorArchivePreprocess);
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }

                // Archive data loading
                asLogMessage(_("Loading archive data."));
                if(!predictorArchivePreprocess->Load(area, timeArrayDataArchive))
                {
                    asLogError(_("Archive data could not be loaded."));
                    wxDELETE(area);
                    wxDELETE(predictorArchivePreprocess);
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }
                m_storagePredictorsArchivePreprocess.push_back(predictorArchivePreprocess);

                // Realtime data loading
                asLogMessage(_("Loading forecast data."));
                if(!predictorRealtimePreprocess->Load(area, timeArrayDataTarget))
                {
                    asLogError(_("Real-time data could not be loaded."));
                    wxDELETE(area);
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }
                wxDELETE(area);
                m_storagePredictorsRealtimePreprocess.push_back(predictorRealtimePreprocess);
            }

            // Fix the criteria if S1
            if(params.GetPredictorCriteria(i_step, i_ptor).IsSameAs("S1"))
            {
                params.SetPredictorCriteria(i_step, i_ptor, "S1grads");
            }

            // Instanciate an archive predictor object
            asDataPredictorArchive* predictorArchive = new asDataPredictorArchive(*m_storagePredictorsArchivePreprocess[0]);
            if (!predictorArchive)
            {
                return false;
            }

            if(!asPreprocessor::Preprocess(m_storagePredictorsArchivePreprocess, params.GetPreprocessMethod(i_step, i_ptor), predictorArchive))
            {
                asLogError(_("Data preprocessing failed."));
                wxDELETE(predictorArchive);
                return false;
            }

            // Instanciate an realtime predictor object
            asDataPredictorRealtime* predictorRealtime = new asDataPredictorRealtime(*m_storagePredictorsRealtimePreprocess[0]);
            if (!predictorRealtime)
            {
                wxDELETE(predictorArchive);
                return false;
            }
            predictorRealtime->SetPredictorsRealtimeDirectory(m_batchForecasts->GetPredictorsRealtimeDirectory());

            if(!asPreprocessor::Preprocess(m_storagePredictorsRealtimePreprocess, params.GetPreprocessMethod(i_step, i_ptor), predictorRealtime))
            {
                asLogError(_("Data preprocessing failed."));
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            wxASSERT(predictorArchive->GetData().size()>1);
            wxASSERT(predictorRealtime->GetData().size()>1);

            m_storagePredictorsArchive.push_back(predictorArchive);
            m_storagePredictorsRealtime.push_back(predictorRealtime);
            DeletePreprocessData();
        }

        asLogMessage(_("Data loaded"));

        // Instantiate a score object
        asLogMessage(_("Creating a criterion object."));
        asPredictorCriteria* criterion = asPredictorCriteria::GetInstance(params.GetPredictorCriteria(i_step, i_ptor), linAlgebraMethod);
        m_storageCriteria.push_back(criterion);
        asLogMessage(_("Criterion object created."));

    }

    #if wxUSE_GUI
        // Send events
        wxCommandEvent eventLoaded (asEVT_STATUS_LOADED);
        wxCommandEvent eventProcessing (asEVT_STATUS_PROCESSING);
        if (m_parent != NULL) {
            m_parent->ProcessWindowEvent(eventLoaded);
            m_parent->ProcessWindowEvent(eventProcessing);
        }
    #endif

    // Send data and criteria to processor
    asLogMessage(_("Start processing the comparison."));

    Array1DDouble timeArrayTargetVect = timeArrayTarget.GetTimeArray();
    Array1DDouble timeArrayTargetVectUnique(1);

    // Loop over the lead times
    for (int i_leadtime=0; i_leadtime<timeArrayTarget.GetSize(); i_leadtime++)
    {
        // Set the corresponding analogs number
        params.SetAnalogsNumber(i_step, params.GetAnalogsNumberLeadTime(i_step, i_leadtime));

        // Create a standard analogs dates result object
        asResultsAnalogsDates anaDates;
        anaDates.SetCurrentStep(i_step);
        anaDates.Init(params);

        // Time array with one value
        timeArrayTargetVectUnique[0] = timeArrayTarget[i_leadtime];
        asTimeArray timeArrayTargetLeadTime = asTimeArray(timeArrayTargetVectUnique);
        bool containsNaNs = false;

        if(!asProcessor::GetAnalogsDates(m_storagePredictorsArchive, m_storagePredictorsRealtime,
                                         timeArrayArchive, timeArrayArchive, timeArrayTarget, timeArrayTargetLeadTime,
                                         m_storageCriteria, params, i_step, anaDates, containsNaNs))
        {
            asLogError(_("Failed processing the analogs dates."));
            return false;
        }

        Array2DFloat dates = anaDates.GetAnalogsDates();
        wxASSERT(dates.rows()==1);
        Array1DFloat rowDates = dates.row(0);
        results.SetAnalogsDates(i_leadtime, rowDates);

        Array2DFloat criteriaVal = anaDates.GetAnalogsCriteria();
        wxASSERT(criteriaVal.rows()==1);
        Array1DFloat rowCriteria = criteriaVal.row(0);
        results.SetAnalogsCriteria(i_leadtime, rowCriteria);
    }

    results.SetTargetDates(timeArrayTarget.GetTimeArray());

    Cleanup();

    return true;
}

bool asMethodForecasting::GetAnalogsSubDates(asResultsAnalogsForecast &results, asParametersForecast &params, asResultsAnalogsForecast &resultsPrev, int i_step)
{
    // Get the linear algebra method
    int linAlgebraMethod = (int)(wxFileConfig::Get()->Read("/Processing/LinAlgebra", (long)asLIN_ALGEBRA_NOVAR));

    // Initialize the result object
    results.SetForecastsDirectory(m_batchForecasts->GetForecastsOutputDirectory());
    results.SetCurrentStep(i_step);
    results.Init(params, m_forecastDate);

    // Date array object instantiation for the processor
    asLogMessage(_("Creating a date arrays for the processor."));

    // Archive time array
    double timeStartArchive = params.GetArchiveStart();
    double timeEndArchive = params.GetArchiveEnd();
    timeEndArchive = wxMin(timeEndArchive, timeEndArchive-params.GetTimeSpanDays()); // Adjust so the predictors search won't overtake the array
    asTimeArray timeArrayArchive(timeStartArchive, timeEndArchive, params.GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
    timeArrayArchive.Init();

    // Get last lead time of the data
    double lastLeadTime = 9999;
    for(int i_ptor=0;i_ptor<params.GetPredictorsNb(i_step);i_ptor++)
    {
        if(!params.NeedsPreprocessing(i_step, i_ptor))
        {
            // Instanciate a predictor object
            asDataPredictorRealtime* predictorRealtime = asDataPredictorRealtime::GetInstance(params.GetPredictorRealtimeDatasetId(i_step, i_ptor), params.GetPredictorRealtimeDataId(i_step, i_ptor));
            if (!predictorRealtime)
            {
                return false;
            }
            predictorRealtime->SetPredictorsRealtimeDirectory(m_batchForecasts->GetPredictorsRealtimeDirectory());
            predictorRealtime->SetRunDateInUse(m_forecastDate);
            lastLeadTime = wxMin(lastLeadTime, predictorRealtime->GetForecastLeadTimeEnd()/24.0 - params.GetTimeSpanDays());

            wxDELETE(predictorRealtime);
        }
        else
        {
            for (int i_prepro=0; i_prepro<params.GetPreprocessSize(i_step, i_ptor); i_prepro++)
            {
                // Instanciate a predictor object
                asDataPredictorRealtime* predictorRealtimePreprocess = asDataPredictorRealtime::GetInstance(params.GetPreprocessRealtimeDatasetId(i_step, i_ptor, i_prepro), params.GetPreprocessRealtimeDataId(i_step, i_ptor, i_prepro));
                if (!predictorRealtimePreprocess)
                {
                    return false;
                }
                predictorRealtimePreprocess->SetPredictorsRealtimeDirectory(m_batchForecasts->GetPredictorsRealtimeDirectory());
                predictorRealtimePreprocess->SetRunDateInUse(m_forecastDate);
                lastLeadTime = wxMin(lastLeadTime, predictorRealtimePreprocess->GetForecastLeadTimeEnd()/24.0 - params.GetTimeSpanDays());

                wxDELETE(predictorRealtimePreprocess);
            }
        }
    }

    // Target time array
    VectorInt leadTime = params.GetLeadTimeDaysVector();
    VectorDouble tmpTimeArray;
    for (unsigned int i=0; i<leadTime.size(); i++)
    {
        if (leadTime[i]>lastLeadTime) break;

        double tmpDate = floor(m_forecastDate)+leadTime[i];
        tmpTimeArray.push_back(tmpDate);
    }
    wxASSERT(tmpTimeArray.size()>0);
    double timeStartTarget = tmpTimeArray[0];
    double timeEndTarget = tmpTimeArray[tmpTimeArray.size()-1];

    asTimeArray timeArrayTarget = asTimeArray(tmpTimeArray);
    timeArrayTarget.Init();

    // Check archive time array length
    if(timeArrayArchive.GetSize()<100)
    {
        asLogError(wxString::Format(_("The time array is not consistent in asMethodForecasting::GetAnalogsDates: size=%d."),timeArrayArchive.GetSize()));
        return false;
    }
    asLogMessage(_("Date array created."));

    // Calculate needed memory
    wxLongLong neededMem = 0;
    for(int i_ptor=0;i_ptor<params.GetPredictorsNb(i_step);i_ptor++)
    {
        neededMem += (params.GetPredictorXptsnb(i_step, i_ptor))
                    * (params.GetPredictorYptsnb(i_step, i_ptor));
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
        asLogMessage(wxString::Format(_("Needed memory for data: %.2f Mb (cannot evaluate available memory)"), neededMemMb));
    }
    else
    {
        asLogMessage(wxString::Format(_("Needed memory for data: %.2f Mb (%.2f Mb available)"), neededMemMb, freeMemMb));
        if(neededMemMb>freeMemMb)
        {
            asLogError(_("Data cannot fit into available memory."));
            return false;
        }
    }

    #if wxUSE_GUI
        // Send event
        wxCommandEvent eventLoading (asEVT_STATUS_LOADING);
        if (m_parent != NULL) {
            m_parent->ProcessWindowEvent(eventLoading);
        }
    #endif

    // Loop through every predictor
    for(int i_ptor=0;i_ptor<params.GetPredictorsNb(i_step);i_ptor++)
    {
        asLogMessage(_("Loading data."));

        if(!params.NeedsPreprocessing(i_step, i_ptor))
        {
            // Date array object instantiation for the data loading. The array has the same length than timeArrayArchive, and the predictor dates are aligned with the target dates, but the dates are not the same.
            double ptorStartArchive = timeStartArchive-params.GetTimeShiftDays()+params.GetPredictorTimeHours(i_step, i_ptor)/24.0;
            double ptorEndArchive = timeEndArchive-params.GetTimeShiftDays()+params.GetPredictorTimeHours(i_step, i_ptor)/24.0;
            asTimeArray timeArrayDataArchive(ptorStartArchive, ptorEndArchive, params.GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
            timeArrayDataArchive.Init();

            double ptorStartTarget = timeStartTarget-params.GetTimeShiftDays()+params.GetPredictorTimeHours(i_step, i_ptor)/24.0;
            double ptorEndTarget = timeEndTarget-params.GetTimeShiftDays()+params.GetPredictorTimeHours(i_step, i_ptor)/24.0;
            asTimeArray timeArrayDataTarget(ptorStartTarget, ptorEndTarget, params.GetTimeArrayTargetTimeStepHours(), asTimeArray::Simple);
            timeArrayDataTarget.Init();

            // Instanciate an archive predictor object
            asDataPredictorArchive* predictorArchive = asDataPredictorArchive::GetInstance(params.GetPredictorArchiveDatasetId(i_step, i_ptor), params.GetPredictorArchiveDataId(i_step, i_ptor), m_batchForecasts->GetPredictorsArchiveDirectory());
            if (!predictorArchive)
            {
                return false;
            }

            // Instanciate an realtime predictor object
            asDataPredictorRealtime* predictorRealtime = asDataPredictorRealtime::GetInstance(params.GetPredictorRealtimeDatasetId(i_step, i_ptor), params.GetPredictorRealtimeDataId(i_step, i_ptor));
            if (!predictorRealtime)
            {
                wxDELETE(predictorArchive);
                return false;
            }
            predictorRealtime->SetPredictorsRealtimeDirectory(m_batchForecasts->GetPredictorsRealtimeDirectory());
            predictorRealtime->SetRunDateInUse(m_forecastDate);

            // Restriction needed
            wxASSERT(params.GetTimeArrayTargetTimeStepHours()>0);
            predictorRealtime->RestrictTimeArray(params.GetPredictorTimeHours(i_step, i_ptor), params.GetTimeArrayTargetTimeStepHours());

            // Update
            if(!predictorRealtime->BuildFilenamesUrls())
            {
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            // Check time array for real-time data
            VectorDouble listTimeArray = predictorRealtime->GetDataDates();
            wxASSERT_MSG(listTimeArray.size()>=(unsigned)timeArrayDataTarget.GetSize(), wxString::Format("size of listTimeArray = %d, size of timeArrayDataTarget = %d", (int)listTimeArray.size(), (int)timeArrayDataTarget.GetSize()));
            for (unsigned int i=0; i<(unsigned)timeArrayDataTarget.GetSize(); i++)
            {
                if(listTimeArray[i]!=timeArrayDataTarget[i])
                {
                    asLogError(wxString::Format(_("The real-time predictor time array is not consistent (listTimeArray[%d](%f)!=timeArrayDataTarget[%d](%f))."), i, listTimeArray[i], i, timeArrayDataTarget[i]));
                    asLogError(_("It is likely that the lead times you defined go beyond the data availability."));
                    wxDELETE(predictorArchive);
                    wxDELETE(predictorRealtime);
                    return false;
                }
            }

            // Area object instantiation
            asGeoAreaCompositeGrid* area = asGeoAreaCompositeGrid::GetInstance(params.GetPredictorGridType(i_step, i_ptor), params.GetPredictorXmin(i_step, i_ptor), params.GetPredictorXptsnb(i_step, i_ptor), params.GetPredictorXstep(i_step, i_ptor), params.GetPredictorYmin(i_step, i_ptor), params.GetPredictorYptsnb(i_step, i_ptor), params.GetPredictorYstep(i_step, i_ptor), params.GetPredictorLevel(i_step, i_ptor), asNONE, params.GetPredictorFlatAllowed(i_step, i_ptor));

            // Check the starting dates coherence
            if (predictorArchive->GetOriginalProviderStart()>ptorStartArchive)
            {
                asLogError(wxString::Format(_("The first year defined in the parameters (%s) is prior to the start date of the data (%s)."), asTime::GetStringTime(ptorStartArchive), asTime::GetStringTime(predictorArchive->GetOriginalProviderStart())));
                wxDELETE(area);
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            // Archive data loading
            if(!predictorArchive->Load(area, timeArrayDataArchive))
            {
                asLogError(_("Archive data could not be loaded."));
                wxDELETE(area);
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }
            m_storagePredictorsArchive.push_back(predictorArchive);

            // Realtime data loading
            if(!predictorRealtime->Load(area, timeArrayDataTarget))
            {
                asLogError(_("Real-time data could not be loaded."));
                wxDELETE(area);
                wxDELETE(predictorRealtime);
                return false;
            }
            wxDELETE(area);
            m_storagePredictorsRealtime.push_back(predictorRealtime);
        }
        else
        {
            int preprocessSize = params.GetPreprocessSize(i_step, i_ptor);

            asLogMessage(wxString::Format(_("Preprocessing data (%d predictor(s)) while loading."), preprocessSize));

            for (int i_prepro=0; i_prepro<preprocessSize; i_prepro++)
            {
                // Date array object instantiation for the data loading. The array has the same length than timeArrayArchive, and the predictor dates are aligned with the target dates, but the dates are not the same.
                double ptorStartArchive = timeStartArchive-params.GetTimeShiftDays()+params.GetPreprocessTimeHours(i_step, i_ptor, i_prepro)/24.0;
                double ptorEndArchive = timeEndArchive-params.GetTimeShiftDays()+params.GetPreprocessTimeHours(i_step, i_ptor, i_prepro)/24.0;
                asTimeArray timeArrayDataArchive(ptorStartArchive, ptorEndArchive, params.GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
                timeArrayDataArchive.Init();

                double ptorStartTarget = timeStartTarget-params.GetTimeShiftDays()+params.GetPreprocessTimeHours(i_step, i_ptor, i_prepro)/24.0;
                double ptorEndTarget = timeEndTarget-params.GetTimeShiftDays()+params.GetPreprocessTimeHours(i_step, i_ptor, i_prepro)/24.0;
                asTimeArray timeArrayDataTarget(ptorStartTarget, ptorEndTarget, params.GetTimeArrayTargetTimeStepHours(), asTimeArray::Simple);
                timeArrayDataTarget.Init();

                // Instanciate an archive predictor object
                asDataPredictorArchive* predictorArchivePreprocess = asDataPredictorArchive::GetInstance(params.GetPreprocessArchiveDatasetId(i_step, i_ptor, i_prepro), params.GetPreprocessArchiveDataId(i_step, i_ptor, i_prepro), m_batchForecasts->GetPredictorsArchiveDirectory());
                if (!predictorArchivePreprocess)
                {
                    return false;
                }

                // Instanciate an realtime predictor object
                asDataPredictorRealtime* predictorRealtimePreprocess = asDataPredictorRealtime::GetInstance(params.GetPreprocessRealtimeDatasetId(i_step, i_ptor, i_prepro), params.GetPreprocessRealtimeDataId(i_step, i_ptor, i_prepro));
                if (!predictorRealtimePreprocess)
                {
                    wxDELETE(predictorArchivePreprocess);
                    return false;
                }
                predictorRealtimePreprocess->SetPredictorsRealtimeDirectory(m_batchForecasts->GetPredictorsRealtimeDirectory());
                predictorRealtimePreprocess->SetRunDateInUse(m_forecastDate);

                // Restriction needed
                wxASSERT(params.GetTimeArrayTargetTimeStepHours()>0);
                predictorRealtimePreprocess->RestrictTimeArray(params.GetPreprocessTimeHours(i_step, i_ptor, i_prepro), params.GetTimeArrayTargetTimeStepHours());

                // Update
                if(!predictorRealtimePreprocess->BuildFilenamesUrls())
                {
                    wxDELETE(predictorArchivePreprocess);
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }

                // Check time array for real-time data
                VectorDouble listTimeArray = predictorRealtimePreprocess->GetDataDates();
                wxASSERT_MSG(listTimeArray.size()>=(unsigned)timeArrayDataTarget.GetSize(), wxString::Format("listTimeArray.size() = %d, timeArrayDataTarget.GetSize() = %d", (int)listTimeArray.size(), (int)timeArrayDataTarget.GetSize()));
                for (unsigned int i=0; i<(unsigned)timeArrayDataTarget.GetSize(); i++)
                {
                    if(listTimeArray[i]!=timeArrayDataTarget[i])
                    {
                        asLogError(wxString::Format(_("The real-time predictor time array is not consistent (listTimeArray[%d](%f)!=timeArrayDataTarget[%d](%f))."), i, listTimeArray[i], i, timeArrayDataTarget[i]));
                        asLogError(_("It is likely that the lead times you defined go beyond the data availability."));
                        wxDELETE(predictorArchivePreprocess);
                        wxDELETE(predictorRealtimePreprocess);
                        return false;
                    }
                }

                // Area object instantiation
                asGeoAreaCompositeGrid* area = asGeoAreaCompositeGrid::GetInstance(params.GetPredictorGridType(i_step, i_ptor), params.GetPredictorXmin(i_step, i_ptor), params.GetPredictorXptsnb(i_step, i_ptor), params.GetPredictorXstep(i_step, i_ptor), params.GetPredictorYmin(i_step, i_ptor), params.GetPredictorYptsnb(i_step, i_ptor), params.GetPredictorYstep(i_step, i_ptor), params.GetPreprocessLevel(i_step, i_ptor, i_prepro), asNONE, params.GetPredictorFlatAllowed(i_step, i_ptor));

                // Check the starting dates coherence
                if (predictorArchivePreprocess->GetOriginalProviderStart()>ptorStartArchive)
                {
                    asLogError(wxString::Format(_("The first year defined in the parameters (%s) is prior to the start date of the data (%s)."), asTime::GetStringTime(ptorStartArchive), asTime::GetStringTime(predictorArchivePreprocess->GetOriginalProviderStart())));
                    wxDELETE(area);
                    wxDELETE(predictorArchivePreprocess);
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }

                // Archive data loading
                if(!predictorArchivePreprocess->Load(area, timeArrayDataArchive))
                {
                    asLogError(_("Archive data could not be loaded."));
                    wxDELETE(area);
                    wxDELETE(predictorArchivePreprocess);
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }
                m_storagePredictorsArchivePreprocess.push_back(predictorArchivePreprocess);

                // Realtime data loading
                if(!predictorRealtimePreprocess->Load(area, timeArrayDataTarget))
                {
                    asLogError(_("Real-time data could not be loaded."));
                    wxDELETE(area);
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }
                wxDELETE(area);
                m_storagePredictorsRealtimePreprocess.push_back(predictorRealtimePreprocess);

                wxASSERT_MSG(predictorArchivePreprocess->GetLatPtsnb()==predictorRealtimePreprocess->GetLatPtsnb(), wxString::Format("predictorArchivePreprocess.GetLatPtsnb()=%d, predictorRealtimePreprocess.GetLatPtsnb()=%d",predictorArchivePreprocess->GetLatPtsnb(), predictorRealtimePreprocess->GetLatPtsnb()));
                wxASSERT_MSG(predictorArchivePreprocess->GetLonPtsnb()==predictorRealtimePreprocess->GetLonPtsnb(), wxString::Format("predictorArchivePreprocess.GetLonPtsnb()=%d, predictorRealtimePreprocess.GetLonPtsnb()=%d",predictorArchivePreprocess->GetLonPtsnb(), predictorRealtimePreprocess->GetLonPtsnb()));
            }

            // Fix the criteria if S1
            if(params.GetPredictorCriteria(i_step, i_ptor).IsSameAs("S1"))
            {
                params.SetPredictorCriteria(i_step, i_ptor, "S1grads");
            }
            
            // Instanciate an archive predictor object
            asDataPredictorArchive* predictorArchive = new asDataPredictorArchive(*m_storagePredictorsArchivePreprocess[0]);
            if (!predictorArchive)
            {
                return false;
            }

            if(!asPreprocessor::Preprocess(m_storagePredictorsArchivePreprocess, params.GetPreprocessMethod(i_step, i_ptor), predictorArchive))
            {
                asLogError(_("Data preprocessing failed."));
                wxDELETE(predictorArchive);
                return false;
            }

            // Instanciate an realtime predictor object
            asDataPredictorRealtime* predictorRealtime = new asDataPredictorRealtime(*m_storagePredictorsRealtimePreprocess[0]);
            if (!predictorRealtime)
            {
                wxDELETE(predictorArchive);
                return false;
            }
            predictorRealtime->SetPredictorsRealtimeDirectory(m_batchForecasts->GetPredictorsRealtimeDirectory());

            if(!asPreprocessor::Preprocess(m_storagePredictorsRealtimePreprocess, params.GetPreprocessMethod(i_step, i_ptor), predictorRealtime))
            {
                asLogError(_("Data preprocessing failed."));
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            wxASSERT(predictorArchive->GetLatPtsnb()==predictorRealtime->GetLatPtsnb());
            wxASSERT(predictorArchive->GetLonPtsnb()==predictorRealtime->GetLonPtsnb());
            m_storagePredictorsArchive.push_back(predictorArchive);
            m_storagePredictorsRealtime.push_back(predictorRealtime);
            DeletePreprocessData();
        }

        asLogMessage(_("Data loaded"));

        // Instantiate a score object
        asLogMessage(_("Creating a criterion object."));
        asPredictorCriteria* criterion = asPredictorCriteria::GetInstance(params.GetPredictorCriteria(i_step, i_ptor), linAlgebraMethod);
        m_storageCriteria.push_back(criterion);
        asLogMessage(_("Criterion object created."));

    }

    #if wxUSE_GUI
        // Send events
        wxCommandEvent eventLoaded (asEVT_STATUS_LOADED);
        wxCommandEvent eventProcessing (asEVT_STATUS_PROCESSING);
        if (m_parent != NULL) {
            m_parent->ProcessWindowEvent(eventLoaded);
            m_parent->ProcessWindowEvent(eventProcessing);
        }
    #endif

    // Send data and criteria to processor
    asLogMessage(_("Start processing the comparison."));

    Array1DFloat leadTimes = resultsPrev.GetTargetDates();

    // Loop over the lead times
    for (int i_leadtime=0; i_leadtime<timeArrayTarget.GetSize(); i_leadtime++)
    {
        // Create a analogs date object for previous results
        asResultsAnalogsDates anaDatesPrev;
        anaDatesPrev.SetCurrentStep(i_step-1);
        anaDatesPrev.Init(params);

        // Set the corresponding analogs number
        params.SetAnalogsNumber(i_step-1, params.GetAnalogsNumberLeadTime(i_step-1, i_leadtime));
        params.SetAnalogsNumber(i_step, params.GetAnalogsNumberLeadTime(i_step, i_leadtime));

        // Create a standard analogs dates result object
        asResultsAnalogsDates anaDates;
        anaDates.SetCurrentStep(i_step);
        anaDates.Init(params);

        Array1DFloat datesPrev = resultsPrev.GetAnalogsDates(i_leadtime);
        Array2DFloat datesPrev2D(1, params.GetAnalogsNumberLeadTime(i_step-1, i_leadtime));
        datesPrev2D.row(0) = datesPrev;

        Array1DFloat criteriaPrev = resultsPrev.GetAnalogsCriteria(i_leadtime);
        Array2DFloat criteriaPrev2D(1, params.GetAnalogsNumberLeadTime(i_step-1, i_leadtime));
        criteriaPrev2D.row(0) = criteriaPrev;

        Array1DDouble leadTimeArray(1);
        leadTimeArray[0] = leadTimes[i_leadtime];
        anaDatesPrev.SetTargetDates(leadTimeArray);
        anaDatesPrev.SetAnalogsDates(datesPrev2D);
        anaDatesPrev.SetAnalogsCriteria(criteriaPrev2D);
        bool containsNaNs = false;

        if(!asProcessor::GetAnalogsSubDates(m_storagePredictorsArchive, m_storagePredictorsRealtime, timeArrayArchive, timeArrayTarget, anaDatesPrev, m_storageCriteria, params, i_step, anaDates, containsNaNs))
        {
            asLogError(_("Failed processing the analogs dates."));
            return false;
        }

        Array2DFloat dates = anaDates.GetAnalogsDates();
        wxASSERT(dates.rows()==1);
        Array1DFloat rowDates = dates.row(0);
        results.SetAnalogsDates(i_leadtime, rowDates);

        Array2DFloat criteriaVal = anaDates.GetAnalogsCriteria();
        wxASSERT(criteriaVal.rows()==1);
        Array1DFloat rowCriteria = criteriaVal.row(0);
        results.SetAnalogsCriteria(i_leadtime, rowCriteria);
    }

    results.SetTargetDates(leadTimes);

    Cleanup();

    return true;
}

bool asMethodForecasting::GetAnalogsValues(asResultsAnalogsForecast &results, asParametersForecast &params, int i_step)
{
    // Initialize the result object
    results.SetForecastsDirectory(m_batchForecasts->GetForecastsOutputDirectory());
    results.SetCurrentStep(i_step);
    results.SetPredictandDatasetId(m_predictandDB->GetDatasetId());
    results.SetPredictandParameter(m_predictandDB->GetDataParameter());
    results.SetPredictandTemporalResolution(m_predictandDB->GetDataTemporalResolution());
    results.SetPredictandSpatialAggregation(m_predictandDB->GetDataSpatialAggregation());

    // Set the predictands values to the corresponding analog dates
    wxASSERT(m_predictandDB);

    // Extract the stations IDs and coordinates
    wxASSERT(m_predictandDB->GetStationsNb()>0);
    Array1DInt stationsId = m_predictandDB->GetStationsIdArray();
    wxASSERT(stationsId.size()>0);
    results.SetStationIds(stationsId);
    results.SetStationOfficialIds(m_predictandDB->GetStationOfficialIdsArray());
    results.SetStationNames(m_predictandDB->GetStationNamesArray());
    results.SetStationHeights(m_predictandDB->GetStationHeightsArray());
    results.SetStationXCoords(m_predictandDB->GetStationXCoordsArray());
    results.SetStationYCoords(m_predictandDB->GetStationYCoordsArray());
    Array1DFloat refAxis = m_predictandDB->GetReferenceAxis();
    results.SetReferenceAxis(refAxis);
    Array2DFloat refValues = m_predictandDB->GetReferenceValuesArray();
    results.SetReferenceValues(refValues);

    Array1DFloat leadTimes = results.GetTargetDates();

    asLogMessage(_("Start setting the predictand values to the corresponding analog dates."));

    // Loop over the lead times
    for (int i_leadtime=0; i_leadtime<leadTimes.size(); i_leadtime++)
    {
        // Set the corresponding analogs number
        params.SetAnalogsNumber(i_step, params.GetAnalogsNumberLeadTime(i_step, i_leadtime));

        asResultsAnalogsDates anaDates;
        anaDates.SetCurrentStep(i_step);
        anaDates.Init(params);

        Array1DFloat datesPrev = results.GetAnalogsDates(i_leadtime);
        Array2DFloat datesPrev2D(1, params.GetAnalogsNumberLeadTime(i_step, i_leadtime));
        datesPrev2D.row(0) = datesPrev;

        Array1DFloat criteriaPrev = results.GetAnalogsCriteria(i_leadtime);
        Array2DFloat criteriaPrev2D(1, params.GetAnalogsNumberLeadTime(i_step, i_leadtime));
        criteriaPrev2D.row(0) = criteriaPrev;

        Array1DDouble leadTimeArray(1);
        leadTimeArray[0] = leadTimes[i_leadtime];
        anaDates.SetTargetDates(leadTimeArray);
        anaDates.SetAnalogsDates(datesPrev2D);
        anaDates.SetAnalogsCriteria(criteriaPrev2D);

        // Process for every station
        for (int i_stat=0; i_stat<stationsId.size(); i_stat++)
        {
            VectorInt stationId(1);
            stationId[0] = stationsId[i_stat];

            // Set the next station ID
            params.SetPredictandStationIds(stationId);

            // Create a standard analogs values result object
            asResultsAnalogsValues anaValues = asResultsAnalogsValues();
            anaValues.SetCurrentStep(i_step);
            anaValues.Init(params);

            if(!asProcessor::GetAnalogsValues(*m_predictandDB, anaDates, params, anaValues))
            {
                asLogError(_("Failed setting the predictand values to the corresponding analog dates."));
                return false;
            }

            VArray2DFloat valuesGross = anaValues.GetAnalogsValuesGross();
            wxASSERT(valuesGross[0].rows()==1);
            Array1DFloat rowValuesGross = valuesGross[0].row(0);
            results.SetAnalogsValuesGross(i_leadtime, i_stat, rowValuesGross);
        }
    }

    #if wxUSE_GUI
        // Send event
        wxCommandEvent eventProcessed (asEVT_STATUS_PROCESSED);
        if (m_parent != NULL) {
            m_parent->ProcessWindowEvent(eventProcessed);
        }
    #endif

    asLogMessage(_("Predictands association over."));

    return true;
}

void asMethodForecasting::Cleanup()
{
    DeletePreprocessData();

    if (m_storagePredictorsArchive.size()>0)
    {
        for (unsigned int i=0; i<m_storagePredictorsArchive.size(); i++)
        {
            wxDELETE(m_storagePredictorsArchive[i]);
        }
        m_storagePredictorsArchive.resize(0);
    }

    if (m_storagePredictorsRealtime.size()>0)
    {
        for (unsigned int i=0; i<m_storagePredictorsRealtime.size(); i++)
        {
            wxDELETE(m_storagePredictorsRealtime[i]);
        }
        m_storagePredictorsRealtime.resize(0);
    }

    if (m_storageCriteria.size()>0)
    {
        for (unsigned int i=0; i<m_storageCriteria.size(); i++)
        {
            wxDELETE(m_storageCriteria[i]);
        }
        m_storageCriteria.resize(0);
    }

    // Do not delete preloaded data here !
}

void asMethodForecasting::DeletePreprocessData()
{
    for (unsigned int i=0; i<m_storagePredictorsArchivePreprocess.size(); i++)
    {
        wxDELETE(m_storagePredictorsArchivePreprocess[i]);
    }
    m_storagePredictorsArchivePreprocess.resize(0);
    
    for (unsigned int i=0; i<m_storagePredictorsRealtimePreprocess.size(); i++)
    {
        wxDELETE(m_storagePredictorsRealtimePreprocess[i]);
    }
    m_storagePredictorsRealtimePreprocess.resize(0);
}