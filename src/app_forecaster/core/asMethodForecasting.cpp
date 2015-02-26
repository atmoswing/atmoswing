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
#include "asResultsAnalogsForecast.h"
#include "asResultsAnalogsForecastAggregator.h"
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
    m_BatchForecasts = batchForecasts;
    m_ForecastDate = NaNDouble;
    m_ParamsFilePath = wxEmptyString;
    m_PredictandDBFilePath = wxEmptyString;
    m_Parent = parent;
}

asMethodForecasting::~asMethodForecasting()
{
    
}

bool asMethodForecasting::Manager()
{
    #if wxUSE_GUI
        if (g_Responsive) wxGetApp().Yield();
    #endif
    m_Cancel = false;

    if(asTools::IsNaN(m_ForecastDate))
    {
        asLogError(_("The date of the forecast has not been defined."));
        return false;
    }

    try
    {
        #if wxUSE_GUI
            // Switch off all leds
            wxCommandEvent eventStart (asEVT_STATUS_STARTING);
            if (m_Parent != NULL) {
                m_Parent->ProcessWindowEvent(eventStart);
            }
        #endif

        // Get paths
        wxString forecastParametersDir = m_BatchForecasts->GetParametersFileDirectory();
        wxString predictandDBDir = m_BatchForecasts->GetPredictandDBDirectory();

        // Execute the forecasts
        for (int i=0; i<m_BatchForecasts->GetForecastsNb(); i++)
        {
            #if wxUSE_GUI
                if (g_Responsive) wxGetApp().Yield();
                if (m_Cancel) return false;

                // Send event
                wxCommandEvent eventRunning (asEVT_STATUS_RUNNING);
                eventRunning.SetInt(i);
                if (m_Parent != NULL) {
                    m_Parent->ProcessWindowEvent(eventRunning);
                }

                if (g_Responsive) wxGetApp().Yield();
            #endif

            // Load parameters
            m_ParamsFilePath = forecastParametersDir + DS + m_BatchForecasts->GetForecastFileName(i);
            asParametersForecast params;
            if(!params.LoadFromFile(m_ParamsFilePath)) return false;
            params.InitValues();

            m_PredictandDBFilePath = predictandDBDir + DS + params.GetPredictandDatabase();

            #if wxUSE_GUI
                if (g_Responsive) wxGetApp().Yield();
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
                    if (m_Parent != NULL) {
                        m_Parent->ProcessWindowEvent(eventFailed);
                    }
                #endif
            }
            else
            {
                // Display processing time
                asLogMessageImportant(wxString::Format(_("Processing of the forecast %s - %s took %ldms to execute"), params.GetMethodId().c_str(), params.GetSpecificTag().c_str(), sw.Time()));

                #if wxUSE_GUI
                    // Send event
                    wxCommandEvent eventSuccess (asEVT_STATUS_SUCCESS);
                    eventSuccess.SetInt(i);
                    if (m_Parent != NULL) {
                        m_Parent->ProcessWindowEvent(eventSuccess);
                    }
                #endif
            }
        }

        // Optional exports
//        if (m_BatchForecasts->HasExports())
        {
            // Reload results
            asResultsAnalogsForecastAggregator aggregator;
            for (int i=0; i<m_ResultsFilePaths.size(); i++)
            {
                asResultsAnalogsForecast * forecast = new asResultsAnalogsForecast();
                forecast->Load(m_ResultsFilePaths[i]);
                aggregator.Add(forecast);
            }
            
//            if (m_BatchForecasts->ExportSyntheticXml())
            {
                if (!aggregator.ExportSyntheticXml())
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
                if (!g_SilentMode)
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
    resultsCheck.SetForecastsDirectory(m_BatchForecasts->GetForecastsOutputDirectory());
    bool forecastDateChanged = true;
    while(forecastDateChanged)
    {
        #if wxUSE_GUI
            if (g_Responsive) wxGetApp().Yield();
        #endif
        if (m_Cancel) return false;

        // Check if result already exists
        resultsCheck.SetCurrentStep(stepsNb-1);
        resultsCheck.Init(params, m_ForecastDate);
        if (resultsCheck.Exists())
        {
            asLogMessage(_("Forecast already exists."));
            m_ResultsFilePaths.push_back(resultsCheck.GetFilePath());
            #if wxUSE_GUI
                if (g_Responsive) wxGetApp().Yield();
            #endif
            return true;
        }

        // Send event
        #if wxUSE_GUI
            wxCommandEvent eventDownloading (asEVT_STATUS_DOWNLOADING);
            if (m_Parent != NULL) {
                m_Parent->ProcessWindowEvent(eventDownloading);
            }

            if (g_Responsive) wxGetApp().Yield();
        #endif

        forecastDateChanged = false;
        for (int i_step=0; i_step<stepsNb; i_step++)
        {
            #if wxUSE_GUI
                if (g_Responsive) wxGetApp().Yield();
            #endif
            if (m_Cancel) return false;
            if(!DownloadRealtimePredictors(params, i_step, forecastDateChanged)) return false;
        }

        // Check if result already exists
        resultsCheck.Init(params, m_ForecastDate);
        if (resultsCheck.Exists())
        {
            asLogMessage(_("Forecast already exists."));
            m_ResultsFilePaths.push_back(resultsCheck.GetFilePath());
            #if wxUSE_GUI
                if (g_Responsive) wxGetApp().Yield();
            #endif
            return true;
        }

        // Send event
        #if wxUSE_GUI
            wxCommandEvent eventDownloaded (asEVT_STATUS_DOWNLOADED);
            if (m_Parent != NULL) {
                m_Parent->ProcessWindowEvent(eventDownloaded);
            }
        #endif
    }

    #if wxUSE_GUI
        if (g_Responsive) wxGetApp().Yield();
    #endif
    if (m_Cancel) return false;

    // Load the Predictand DB
    asLogMessage(_("Loading the Predictand DB."));
    if(!LoadPredictandDB(m_PredictandDBFilePath)) return false;
    asLogMessage(_("Predictand DB loaded."));

   #if wxUSE_GUI
        if (g_Responsive) wxGetApp().Yield();
    #endif
    if (m_Cancel) return false;

    // Resulting object
    asResultsAnalogsForecast resultsPrevious;
    asResultsAnalogsForecast results;
    results.SetForecastsDirectory(m_BatchForecasts->GetForecastsOutputDirectory());

    for (int i_step=0; i_step<stepsNb; i_step++)
    {
        #if wxUSE_GUI
            if (g_Responsive) wxGetApp().Yield();
        #endif
        if (m_Cancel) return false;

        if (i_step==0)
        {
            if(!GetAnalogsDates(results, params, i_step)) return false;
        }
        else
        {
            if(!GetAnalogsSubDates(results, params, resultsPrevious, i_step)) return false;
        }

        // At last get the values
        if (i_step==stepsNb-1)
        {
            #if wxUSE_GUI
                if (g_Responsive) wxGetApp().Yield();
            #endif
            if (m_Cancel) return false;

            if(!GetAnalogsValues(results, params, i_step)) return false;

            #if wxUSE_GUI
                // Send event
                wxCommandEvent eventSaving (asEVT_STATUS_SAVING);
                if (m_Parent != NULL) {
                    m_Parent->ProcessWindowEvent(eventSaving);
                }
            #endif

            try
            {
                results.Save();
            }
            catch(asException& e)
            {
                wxString fullMessage = e.GetFullMessage();
                if (!fullMessage.IsEmpty())
                {
                    #if wxUSE_GUI
                        if (!g_SilentMode)
                            wxMessageBox(fullMessage);
                    #endif
                }
                return false;
            }

            #if wxUSE_GUI
                // Send event
                wxCommandEvent eventSaved (asEVT_STATUS_SAVED);
                if (m_Parent != NULL) {
                    m_Parent->ProcessWindowEvent(eventSaved);
                }
            #endif
        }

        // Keep the analogs dates of the best parameters set
        resultsPrevious = results;
    }

    m_ResultsFilePaths.push_back(results.GetFilePath());

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
            if (g_Responsive) wxGetApp().Yield();
        #endif
        if (m_Cancel) return false;

        asLogMessage(_("Downloading data."));

        #if wxUSE_GUI
            if (g_Responsive) wxGetApp().Yield();
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
            predictorRealtime->SetPredictorsRealtimeDirectory(m_BatchForecasts->GetPredictorsRealtimeDirectory());

            // Set the desired forecasting date
            m_ForecastDate = predictorRealtime->SetRunDateInUse(m_ForecastDate);

            // Check if result already exists
            asResultsAnalogsForecast resultsCheck;
            resultsCheck.SetForecastsDirectory(m_BatchForecasts->GetForecastsOutputDirectory());
            resultsCheck.SetCurrentStep(params.GetStepsNb()-1);
            resultsCheck.Init(params, m_ForecastDate);
            if (resultsCheck.Exists())
            {
                asLogMessage(_("Forecast already exists."));
                #if wxUSE_GUI
                    if (g_Responsive) wxGetApp().Yield();
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
                    if (g_Responsive) wxGetApp().Yield();
                #endif
                if (m_Cancel)
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
                        m_ForecastDate = predictorRealtime->DecrementRunDateInUse();
                        // Check if result already exists
                        resultsCheck.SetCurrentStep(params.GetStepsNb()-1);
                        resultsCheck.Init(params, m_ForecastDate);
                        if (resultsCheck.Exists())
                        {
                            asLogMessage(_("Forecast already exists."));
                            #if wxUSE_GUI
                                if (g_Responsive) wxGetApp().Yield();
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
            m_ForecastDate = predictorRealtime->GetRunDateInUse();
            wxDELETE(predictorRealtime);
        }
        else
        {
            int preprocessSize = params.GetPreprocessSize(i_step, i_ptor);

            for (int i_prepro=0; i_prepro<preprocessSize; i_prepro++)
            {
                #if wxUSE_GUI
                    if (g_Responsive) wxGetApp().Yield();
                #endif
                if (m_Cancel) return false;

                // Instanciate a predictor object
                asDataPredictorRealtime* predictorRealtimePreprocess = asDataPredictorRealtime::GetInstance(params.GetPreprocessRealtimeDatasetId(i_step, i_ptor, i_prepro), params.GetPreprocessRealtimeDataId(i_step, i_ptor, i_prepro));
                if (!predictorRealtimePreprocess)
                {
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }
                predictorRealtimePreprocess->SetPredictorsRealtimeDirectory(m_BatchForecasts->GetPredictorsRealtimeDirectory());

                // Set the desired forecasting date
                m_ForecastDate = predictorRealtimePreprocess->SetRunDateInUse(m_ForecastDate);

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
                        if (g_Responsive) wxGetApp().Yield();
                    #endif
                    if (m_Cancel)
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
                            m_ForecastDate = predictorRealtimePreprocess->DecrementRunDateInUse();
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
                m_ForecastDate = predictorRealtimePreprocess->GetRunDateInUse();
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
    results.SetForecastsDirectory(m_BatchForecasts->GetForecastsOutputDirectory());
    results.SetCurrentStep(i_step);
    results.Init(params, m_ForecastDate);

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

            predictorRealtime->SetRunDateInUse(m_ForecastDate);
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

                predictorRealtimePreprocess->SetRunDateInUse(m_ForecastDate);
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

        double tmpDate = floor(m_ForecastDate)+leadTime[i];
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
        if (m_Parent != NULL) {
            m_Parent->ProcessWindowEvent(eventLoading);
        }
    #endif

    // Loop through every predictor
    for(int i_ptor=0;i_ptor<params.GetPredictorsNb(i_step);i_ptor++)
    {
        asLogMessage(wxString::Format(_("Loading data (step %d, predictor nb %d)."), i_step, i_ptor));

        #if wxUSE_GUI
            if (g_Responsive) wxGetApp().Yield();
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
            asDataPredictorArchive* predictorArchive = asDataPredictorArchive::GetInstance(params.GetPredictorArchiveDatasetId(i_step, i_ptor), params.GetPredictorArchiveDataId(i_step, i_ptor), m_BatchForecasts->GetPredictorsArchiveDirectory());
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
            predictorRealtime->SetPredictorsRealtimeDirectory(m_BatchForecasts->GetPredictorsRealtimeDirectory());
            predictorRealtime->SetRunDateInUse(m_ForecastDate);

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
            m_StoragePredictorsArchive.push_back(predictorArchive);

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
            m_StoragePredictorsRealtime.push_back(predictorRealtime);
        }
        else
        {
            int preprocessSize = params.GetPreprocessSize(i_step, i_ptor);

            asLogMessage(wxString::Format(_("Preprocessing data (%d predictor(s)) while loading."), preprocessSize));

            for (int i_prepro=0; i_prepro<preprocessSize; i_prepro++)
            {
                #if wxUSE_GUI
                    if (g_Responsive) wxGetApp().Yield();
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
                asDataPredictorArchive* predictorArchivePreprocess = asDataPredictorArchive::GetInstance(params.GetPreprocessArchiveDatasetId(i_step, i_ptor, i_prepro), params.GetPreprocessArchiveDataId(i_step, i_ptor, i_prepro), m_BatchForecasts->GetPredictorsArchiveDirectory());
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
                predictorRealtimePreprocess->SetPredictorsRealtimeDirectory(m_BatchForecasts->GetPredictorsRealtimeDirectory());
                predictorRealtimePreprocess->SetRunDateInUse(m_ForecastDate);

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
                m_StoragePredictorsArchivePreprocess.push_back(predictorArchivePreprocess);

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
                m_StoragePredictorsRealtimePreprocess.push_back(predictorRealtimePreprocess);
            }

            // Fix the criteria if S1
            if(params.GetPredictorCriteria(i_step, i_ptor).IsSameAs("S1"))
            {
                params.SetPredictorCriteria(i_step, i_ptor, "S1grads");
            }

            // Instanciate an archive predictor object
            asDataPredictorArchive* predictorArchive = new asDataPredictorArchive(*m_StoragePredictorsArchivePreprocess[0]);
            if (!predictorArchive)
            {
                return false;
            }

            if(!asPreprocessor::Preprocess(m_StoragePredictorsArchivePreprocess, params.GetPreprocessMethod(i_step, i_ptor), predictorArchive))
            {
                asLogError(_("Data preprocessing failed."));
                wxDELETE(predictorArchive);
                return false;
            }

            // Instanciate an realtime predictor object
            asDataPredictorRealtime* predictorRealtime = new asDataPredictorRealtime(*m_StoragePredictorsRealtimePreprocess[0]);
            if (!predictorRealtime)
            {
                wxDELETE(predictorArchive);
                return false;
            }
            predictorRealtime->SetPredictorsRealtimeDirectory(m_BatchForecasts->GetPredictorsRealtimeDirectory());

            if(!asPreprocessor::Preprocess(m_StoragePredictorsRealtimePreprocess, params.GetPreprocessMethod(i_step, i_ptor), predictorRealtime))
            {
                asLogError(_("Data preprocessing failed."));
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            wxASSERT(predictorArchive->GetData().size()>1);
            wxASSERT(predictorRealtime->GetData().size()>1);

            m_StoragePredictorsArchive.push_back(predictorArchive);
            m_StoragePredictorsRealtime.push_back(predictorRealtime);
            DeletePreprocessData();
        }

        asLogMessage(_("Data loaded"));

        // Instantiate a score object
        asLogMessage(_("Creating a criterion object."));
        asPredictorCriteria* criterion = asPredictorCriteria::GetInstance(params.GetPredictorCriteria(i_step, i_ptor), linAlgebraMethod);
        m_StorageCriteria.push_back(criterion);
        asLogMessage(_("Criterion object created."));

    }

    #if wxUSE_GUI
        // Send events
        wxCommandEvent eventLoaded (asEVT_STATUS_LOADED);
        wxCommandEvent eventProcessing (asEVT_STATUS_PROCESSING);
        if (m_Parent != NULL) {
            m_Parent->ProcessWindowEvent(eventLoaded);
            m_Parent->ProcessWindowEvent(eventProcessing);
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

        if(!asProcessor::GetAnalogsDates(m_StoragePredictorsArchive, m_StoragePredictorsRealtime,
                                         timeArrayArchive, timeArrayArchive, timeArrayTarget, timeArrayTargetLeadTime,
                                         m_StorageCriteria, params, i_step, anaDates, containsNaNs))
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
    results.SetForecastsDirectory(m_BatchForecasts->GetForecastsOutputDirectory());
    results.SetCurrentStep(i_step);
    results.Init(params, m_ForecastDate);

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
            predictorRealtime->SetPredictorsRealtimeDirectory(m_BatchForecasts->GetPredictorsRealtimeDirectory());
            predictorRealtime->SetRunDateInUse(m_ForecastDate);
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
                predictorRealtimePreprocess->SetPredictorsRealtimeDirectory(m_BatchForecasts->GetPredictorsRealtimeDirectory());
                predictorRealtimePreprocess->SetRunDateInUse(m_ForecastDate);
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

        double tmpDate = floor(m_ForecastDate)+leadTime[i];
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
        if (m_Parent != NULL) {
            m_Parent->ProcessWindowEvent(eventLoading);
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
            asDataPredictorArchive* predictorArchive = asDataPredictorArchive::GetInstance(params.GetPredictorArchiveDatasetId(i_step, i_ptor), params.GetPredictorArchiveDataId(i_step, i_ptor), m_BatchForecasts->GetPredictorsArchiveDirectory());
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
            predictorRealtime->SetPredictorsRealtimeDirectory(m_BatchForecasts->GetPredictorsRealtimeDirectory());
            predictorRealtime->SetRunDateInUse(m_ForecastDate);

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
            m_StoragePredictorsArchive.push_back(predictorArchive);

            // Realtime data loading
            if(!predictorRealtime->Load(area, timeArrayDataTarget))
            {
                asLogError(_("Real-time data could not be loaded."));
                wxDELETE(area);
                wxDELETE(predictorRealtime);
                return false;
            }
            wxDELETE(area);
            m_StoragePredictorsRealtime.push_back(predictorRealtime);
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
                asDataPredictorArchive* predictorArchivePreprocess = asDataPredictorArchive::GetInstance(params.GetPreprocessArchiveDatasetId(i_step, i_ptor, i_prepro), params.GetPreprocessArchiveDataId(i_step, i_ptor, i_prepro), m_BatchForecasts->GetPredictorsArchiveDirectory());
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
                predictorRealtimePreprocess->SetPredictorsRealtimeDirectory(m_BatchForecasts->GetPredictorsRealtimeDirectory());
                predictorRealtimePreprocess->SetRunDateInUse(m_ForecastDate);

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
                m_StoragePredictorsArchivePreprocess.push_back(predictorArchivePreprocess);

                // Realtime data loading
                if(!predictorRealtimePreprocess->Load(area, timeArrayDataTarget))
                {
                    asLogError(_("Real-time data could not be loaded."));
                    wxDELETE(area);
                    wxDELETE(predictorRealtimePreprocess);
                    return false;
                }
                wxDELETE(area);
                m_StoragePredictorsRealtimePreprocess.push_back(predictorRealtimePreprocess);

                wxASSERT_MSG(predictorArchivePreprocess->GetLatPtsnb()==predictorRealtimePreprocess->GetLatPtsnb(), wxString::Format("predictorArchivePreprocess.GetLatPtsnb()=%d, predictorRealtimePreprocess.GetLatPtsnb()=%d",predictorArchivePreprocess->GetLatPtsnb(), predictorRealtimePreprocess->GetLatPtsnb()));
                wxASSERT_MSG(predictorArchivePreprocess->GetLonPtsnb()==predictorRealtimePreprocess->GetLonPtsnb(), wxString::Format("predictorArchivePreprocess.GetLonPtsnb()=%d, predictorRealtimePreprocess.GetLonPtsnb()=%d",predictorArchivePreprocess->GetLonPtsnb(), predictorRealtimePreprocess->GetLonPtsnb()));
            }

            // Fix the criteria if S1
            if(params.GetPredictorCriteria(i_step, i_ptor).IsSameAs("S1"))
            {
                params.SetPredictorCriteria(i_step, i_ptor, "S1grads");
            }
            
            // Instanciate an archive predictor object
            asDataPredictorArchive* predictorArchive = new asDataPredictorArchive(*m_StoragePredictorsArchivePreprocess[0]);
            if (!predictorArchive)
            {
                return false;
            }

            if(!asPreprocessor::Preprocess(m_StoragePredictorsArchivePreprocess, params.GetPreprocessMethod(i_step, i_ptor), predictorArchive))
            {
                asLogError(_("Data preprocessing failed."));
                wxDELETE(predictorArchive);
                return false;
            }

            // Instanciate an realtime predictor object
            asDataPredictorRealtime* predictorRealtime = new asDataPredictorRealtime(*m_StoragePredictorsRealtimePreprocess[0]);
            if (!predictorRealtime)
            {
                wxDELETE(predictorArchive);
                return false;
            }
            predictorRealtime->SetPredictorsRealtimeDirectory(m_BatchForecasts->GetPredictorsRealtimeDirectory());

            if(!asPreprocessor::Preprocess(m_StoragePredictorsRealtimePreprocess, params.GetPreprocessMethod(i_step, i_ptor), predictorRealtime))
            {
                asLogError(_("Data preprocessing failed."));
                wxDELETE(predictorArchive);
                wxDELETE(predictorRealtime);
                return false;
            }

            wxASSERT(predictorArchive->GetLatPtsnb()==predictorRealtime->GetLatPtsnb());
            wxASSERT(predictorArchive->GetLonPtsnb()==predictorRealtime->GetLonPtsnb());
            m_StoragePredictorsArchive.push_back(predictorArchive);
            m_StoragePredictorsRealtime.push_back(predictorRealtime);
            DeletePreprocessData();
        }

        asLogMessage(_("Data loaded"));

        // Instantiate a score object
        asLogMessage(_("Creating a criterion object."));
        asPredictorCriteria* criterion = asPredictorCriteria::GetInstance(params.GetPredictorCriteria(i_step, i_ptor), linAlgebraMethod);
        m_StorageCriteria.push_back(criterion);
        asLogMessage(_("Criterion object created."));

    }

    #if wxUSE_GUI
        // Send events
        wxCommandEvent eventLoaded (asEVT_STATUS_LOADED);
        wxCommandEvent eventProcessing (asEVT_STATUS_PROCESSING);
        if (m_Parent != NULL) {
            m_Parent->ProcessWindowEvent(eventLoaded);
            m_Parent->ProcessWindowEvent(eventProcessing);
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

        if(!asProcessor::GetAnalogsSubDates(m_StoragePredictorsArchive, m_StoragePredictorsRealtime, timeArrayArchive, timeArrayTarget, anaDatesPrev, m_StorageCriteria, params, i_step, anaDates, containsNaNs))
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
    results.SetForecastsDirectory(m_BatchForecasts->GetForecastsOutputDirectory());
    results.SetCurrentStep(i_step);
    results.SetPredictandDatasetId(m_PredictandDB->GetDatasetId());
    results.SetPredictandParameter(m_PredictandDB->GetDataParameter());
    results.SetPredictandTemporalResolution(m_PredictandDB->GetDataTemporalResolution());
    results.SetPredictandSpatialAggregation(m_PredictandDB->GetDataSpatialAggregation());

    // Set the predictands values to the corresponding analog dates
    wxASSERT(m_PredictandDB);

    // Extract the stations IDs and coordinates
    wxASSERT(m_PredictandDB->GetStationsNb()>0);
    Array1DInt stationsId = m_PredictandDB->GetStationsIdArray();
    wxASSERT(stationsId.size()>0);
    results.SetStationIds(stationsId);
    results.SetStationOfficialIds(m_PredictandDB->GetStationOfficialIdsArray());
    results.SetStationNames(m_PredictandDB->GetStationNamesArray());
    results.SetStationHeights(m_PredictandDB->GetStationHeightsArray());
    results.SetStationXCoords(m_PredictandDB->GetStationXCoordsArray());
    results.SetStationYCoords(m_PredictandDB->GetStationYCoordsArray());
    Array1DFloat refAxis = m_PredictandDB->GetReferenceAxis();
    results.SetReferenceAxis(refAxis);
    Array2DFloat refValues = m_PredictandDB->GetReferenceValuesArray();
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

            if(!asProcessor::GetAnalogsValues(*m_PredictandDB, anaDates, params, anaValues))
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
        if (m_Parent != NULL) {
            m_Parent->ProcessWindowEvent(eventProcessed);
        }
    #endif

    asLogMessage(_("Predictands association over."));

    return true;
}

void asMethodForecasting::Cleanup()
{
    DeletePreprocessData();

    if (m_StoragePredictorsArchive.size()>0)
    {
        for (unsigned int i=0; i<m_StoragePredictorsArchive.size(); i++)
        {
            wxDELETE(m_StoragePredictorsArchive[i]);
        }
        m_StoragePredictorsArchive.resize(0);
    }

    if (m_StoragePredictorsRealtime.size()>0)
    {
        for (unsigned int i=0; i<m_StoragePredictorsRealtime.size(); i++)
        {
            wxDELETE(m_StoragePredictorsRealtime[i]);
        }
        m_StoragePredictorsRealtime.resize(0);
    }

    if (m_StorageCriteria.size()>0)
    {
        for (unsigned int i=0; i<m_StorageCriteria.size(); i++)
        {
            wxDELETE(m_StorageCriteria[i]);
        }
        m_StorageCriteria.resize(0);
    }

    // Do not delete preloaded data here !
}

void asMethodForecasting::DeletePreprocessData()
{
    for (unsigned int i=0; i<m_StoragePredictorsArchivePreprocess.size(); i++)
    {
        wxDELETE(m_StoragePredictorsArchivePreprocess[i]);
    }
    m_StoragePredictorsArchivePreprocess.resize(0);
    
    for (unsigned int i=0; i<m_StoragePredictorsRealtimePreprocess.size(); i++)
    {
        wxDELETE(m_StoragePredictorsRealtimePreprocess[i]);
    }
    m_StoragePredictorsRealtimePreprocess.resize(0);
}