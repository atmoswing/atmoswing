/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
#include "asMethodForecasting.h"

#include "asCatalogPredictorsRealtime.h"
#include "asCatalogPredictorsArchive.h"
#include "asDataPredictorRealtime.h"
#include "asDataPredictorArchive.h"
#include "asDataPredictand.h"
#include "asFileForecastingModels.h"
#include "asResultsAnalogsDates.h"
#include "asResultsAnalogsValues.h"
#include "asResultsAnalogsForecast.h"
#include "asPredictorCriteria.h"
#include "asTimeArray.h"
#include "asGeoAreaCompositeGrid.h"
#include "asProcessor.h"
#include "asPreprocessor.h"
#ifndef UNIT_TESTING
    #include "AtmoswingAppForecaster.h"
#endif

asMethodForecasting::asMethodForecasting(wxWindow* parent)
:
asMethodStandard()
{
    m_ForecastDate = NaNDouble;
    m_ModelName = wxEmptyString;
    m_ParamsFilePath = wxEmptyString;
    m_PredictandDBFilePath = wxEmptyString;
    m_PredictorsArchiveDir = wxEmptyString;
    m_Parent = parent;
}

asMethodForecasting::~asMethodForecasting()
{
    //dtor
}

bool asMethodForecasting::Manager()
{
	#if wxUSE_GUI
		if (g_Responsive) wxGetApp().Yield();
	#endif
    m_Cancel = false;

    wxConfigBase *pConfig = wxFileConfig::Get();

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

        // Get models list file
        wxString filePath = asConfig::GetDefaultUserConfigDir() + "CurrentForecastingModelsList.xml";
        if (!wxFileName::FileExists(filePath))
        {
            asLogError(_("The current forecasting models list could not be found."));
            return false;
        }

        asFileForecastingModels file(filePath, asFile::ReadOnly);
        if(!file.Open())
        {
            asLogError(_("Cannot open the models list file."));
            return false;
        }

        // Parse the file
        if(!file.GoToRootElement()) return false;
        if(!file.GoToFirstNodeWithPath("ModelsList")) return false;
        if(!file.GoToFirstNodeWithPath("Model")) return false;

        int counter = 0;
        while(true)
        {
			#if wxUSE_GUI
				if (g_Responsive) wxGetApp().Yield();
				if (m_Cancel) return false;

				// Send event
				wxCommandEvent eventRunning (asEVT_STATUS_RUNNING);
				eventRunning.SetInt(counter);
				if (m_Parent != NULL) {
					m_Parent->ProcessWindowEvent(eventRunning);
				}

				if (g_Responsive) wxGetApp().Yield();
			#endif

            // Set the content to data members
            wxString dirConfig = asConfig::GetDataDir()+"config"+DS;
            wxString dirData = asConfig::GetDataDir()+"data"+DS;
            wxString archivePredictorsDir = pConfig->Read("/StandardPaths/ArchivePredictorsDir", dirData+"predictors");
            wxString forecastParametersDir = pConfig->Read("/StandardPaths/ForecastParametersDir", dirConfig);
            wxString predictandDBDir = pConfig->Read("/StandardPaths/DataPredictandDBDir", dirData+"predictands");

            m_ModelName = file.GetThisElementAttributeValueText("name");
            m_ParamsFilePath = forecastParametersDir + DS + file.GetFirstElementAttributeValueText("ParametersFileName", "value");
            m_PredictandDBFilePath = predictandDBDir + DS + file.GetFirstElementAttributeValueText("PredictandDB", "value");
            m_PredictorsArchiveDir = file.GetFirstElementAttributeValueText("PredictorsArchiveDir", "value");
            if (m_PredictorsArchiveDir.IsEmpty())
            {
                m_PredictorsArchiveDir = archivePredictorsDir;
            }

            // Load parameters
            asParametersForecast params;
            if(!params.LoadFromFile(m_ParamsFilePath)) return false;
            params.InitValues();

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
					eventFailed.SetInt(counter);
					if (m_Parent != NULL) {
						m_Parent->ProcessWindowEvent(eventFailed);
					}
				#endif
            }
            else
            {
                // Display processing time
                asLogMessageImportant(wxString::Format(_("Processing of the model %s took %ldms to execute"), m_ModelName.c_str(), sw.Time()));

				#if wxUSE_GUI
					// Send event
					wxCommandEvent eventSuccess (asEVT_STATUS_SUCCESS);
					eventSuccess.SetInt(counter);
					if (m_Parent != NULL) {
						m_Parent->ProcessWindowEvent(eventSuccess);
					}
				#endif
            }

            // Find the next model
            bool result = file.GoToNextSameNode();
            if (!result) break;

            counter++;
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

    return true;
}

bool asMethodForecasting::Forecast(asParametersForecast &params)
{
    // Process every step one after the other
    int stepsNb = params.GetStepsNb();

    // Download real-time predictors
    asResultsAnalogsForecast resultsCheck(m_ModelName);
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
    asResultsAnalogsForecast resultsPrevious(m_ModelName);
    asResultsAnalogsForecast results(m_ModelName);

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

    return true;
}

bool asMethodForecasting::DownloadRealtimePredictors(asParametersForecast &params, int i_step, bool &forecastDateChanged)
{
    // Get preferences
    wxConfigBase *pConfig = wxFileConfig::Get();
    long maxPrevStepsNbDef = 5;
    long maxPrevStepsNb = pConfig->Read("/Internet/MaxPreviousStepsNb", maxPrevStepsNbDef);

    // Catalog
    asCatalogPredictorsRealtime catalogRealtime = asCatalogPredictorsRealtime();

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
            // Loading the datasets information
            if(!catalogRealtime.Load(params.GetPredictorRealtimeDatasetId(i_step, i_ptor), params.GetPredictorRealtimeDataId(i_step, i_ptor))) return false;

            // Set the desired forecasting date
            m_ForecastDate = catalogRealtime.SetRunDateInUse(m_ForecastDate);

            // Check if result already exists
            asResultsAnalogsForecast resultsCheck(m_ModelName);
            resultsCheck.SetCurrentStep(params.GetStepsNb()-1);
            resultsCheck.Init(params, m_ForecastDate);
            if (resultsCheck.Exists())
            {
                asLogMessage(_("Forecast already exists."));
                #if wxUSE_GUI
                    if (g_Responsive) wxGetApp().Yield();
                #endif
                return true;
            }

            // Restriction needed
            wxASSERT(params.GetTimeArrayTargetTimeStepHours()>0);
            catalogRealtime.RestrictTimeArray(params.GetPredictorDTimeHours(i_step, i_ptor), params.GetTimeArrayTargetTimeStepHours());

            // Update forecasting date
            if(!catalogRealtime.BuildFilenamesUrls()) return false;

            // Realtime data downloading
            asDataPredictorRealtime predictorRealtime(catalogRealtime);
            int counterFails = 0;
            while (true)
            {
                #if wxUSE_GUI
                    if (g_Responsive) wxGetApp().Yield();
                #endif
                if (m_Cancel) return false;

                // Download predictor
                int resDownload = predictorRealtime.Download(catalogRealtime);
                if (resDownload==asSUCCESS)
                {
                    break;
                }
                else if (resDownload==asFAILED)
                {
                    if (counterFails<maxPrevStepsNb)
                    {
                        // Try to download older data
                        m_ForecastDate = catalogRealtime.DecrementRunDateInUse();
                        // Check if result already exists
                        resultsCheck.SetCurrentStep(params.GetStepsNb()-1);
                        resultsCheck.Init(params, m_ForecastDate);
                        if (resultsCheck.Exists())
                        {
                            asLogMessage(_("Forecast already exists."));
                            #if wxUSE_GUI
                                if (g_Responsive) wxGetApp().Yield();
                            #endif
                            return true;
                        }
                        forecastDateChanged = true;
                        catalogRealtime.BuildFilenamesUrls();
                        counterFails++;
                    }
                    else
                    {
                        asLogError(_("The maximum attempts is reached to download the real-time predictor. Forecasting failed."));
                        return false;
                    }
                }
                else
                {
                    // Canceled for example.
                    return false;
                }
            }
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

                // Loading the datasets information
                if(!catalogRealtime.Load(params.GetPreprocessRealtimeDatasetId(i_step, i_ptor, i_prepro), params.GetPreprocessRealtimeDataId(i_step, i_ptor, i_prepro))) return false;

                // Set the desired forecasting date
                m_ForecastDate = catalogRealtime.SetRunDateInUse(m_ForecastDate);

                // Restriction needed
                wxASSERT(params.GetTimeArrayTargetTimeStepHours()>0);
                catalogRealtime.RestrictTimeArray(params.GetPreprocessDTimeHours(i_step, i_ptor, i_prepro), params.GetTimeArrayTargetTimeStepHours());

                // Update forecasting date
                if(!catalogRealtime.BuildFilenamesUrls()) return false;

                // Realtime data downloading
                asDataPredictorRealtime predictorRealtimePreprocess(catalogRealtime);
                int counterFails = 0;
                while (true)
                {
                    #if wxUSE_GUI
                        if (g_Responsive) wxGetApp().Yield();
                    #endif
                    if (m_Cancel) return false;

                    // Download predictor
                    int resDownload = predictorRealtimePreprocess.Download(catalogRealtime);
                    if (resDownload==asSUCCESS)
                    {
                        break;
                    }
                    else if (resDownload==asFAILED)
                    {
                        if (counterFails<maxPrevStepsNb)
                        {
                            // Try to download older data
                            m_ForecastDate = catalogRealtime.DecrementRunDateInUse();
                            forecastDateChanged = true;
                            catalogRealtime.BuildFilenamesUrls();
                            counterFails++;
                        }
                        else
                        {
                            asLogError(_("The maximum attempts is reached to download the real-time predictor. Forecasting failed."));
                            return false;
                        }
                    }
                    else
                    {
                        // Canceled for example.
                        return false;
                    }
                }
            }
        }

        asLogMessage(_("Data downloaded."));
    }

    m_ForecastDate = catalogRealtime.GetRunDateInUse();

    return true;
}

bool asMethodForecasting::GetAnalogsDates(asResultsAnalogsForecast &results, asParametersForecast &params, int i_step)
{
    // Get preferences
    int linAlgebraMethod = (int)(wxFileConfig::Get()->Read("/ProcessingOptions/ProcessingLinAlgebra", (long)asCOEFF_NOVAR));

    // Catalogs
    asCatalogPredictorsArchive catalogArchive = asCatalogPredictorsArchive();
    asCatalogPredictorsRealtime catalogRealtime = asCatalogPredictorsRealtime();

    // Initialize the result object
    results.SetCurrentStep(i_step);
    results.Init(params, m_ForecastDate);

    // Create the vectors to put the data in
    std::vector < asDataPredictor > predictorsArchive;
    std::vector < asDataPredictor > predictorsRealtime;
    std::vector < asPredictorCriteria* > criteria;

    // Date array object instantiation for the processor
    asLogMessage(_("Creating date arrays for the processor."));

    // Archive time array
    double timeStartArchive = asTime::GetMJD(params.GetArchiveYearStart(),1,1); // Always Jan 1st
    double timeEndArchive = asTime::GetMJD(params.GetArchiveYearEnd(),12,31);
    timeEndArchive = wxMin(timeEndArchive, timeEndArchive-params.GetTimeSpanDays()); // Adjust so the predictors search won't overtake the array
    asTimeArray timeArrayArchive(timeStartArchive, timeEndArchive, params.GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
    timeArrayArchive.Init();

    // Get last lead time of the data
    double lastLeadTime = 9999;
    for(int i_ptor=0;i_ptor<params.GetPredictorsNb(i_step);i_ptor++)
    {
        if(!params.NeedsPreprocessing(i_step, i_ptor))
        {
            if(!catalogRealtime.Load(params.GetPredictorRealtimeDatasetId(i_step, i_ptor), params.GetPredictorRealtimeDataId(i_step, i_ptor))) return false;
            catalogRealtime.SetRunDateInUse(m_ForecastDate);
            lastLeadTime = wxMin(lastLeadTime, catalogRealtime.GetForecastLeadTimeEnd()/24.0 - params.GetTimeSpanDays());
        }
        else
        {
            for (int i_prepro=0; i_prepro<params.GetPreprocessSize(i_step, i_ptor); i_prepro++)
            {
                if(!catalogRealtime.Load(params.GetPreprocessRealtimeDatasetId(i_step, i_ptor, i_prepro), params.GetPreprocessRealtimeDataId(i_step, i_ptor, i_prepro))) return false;
                catalogRealtime.SetRunDateInUse(m_ForecastDate);
                lastLeadTime = wxMin(lastLeadTime, catalogRealtime.GetForecastLeadTimeEnd()/24.0 - params.GetTimeSpanDays());
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
        neededMem += (params.GetPredictorUptsnb(i_step, i_ptor))
                    * (params.GetPredictorVptsnb(i_step, i_ptor));
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
            double ptorStartArchive = timeStartArchive-params.GetTimeShiftDays()+params.GetPredictorDTimeDays(i_step, i_ptor);
            double ptorEndArchive = timeEndArchive-params.GetTimeShiftDays()+params.GetPredictorDTimeDays(i_step, i_ptor);
            asTimeArray timeArrayDataArchive(ptorStartArchive, ptorEndArchive, params.GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
            timeArrayDataArchive.Init();

            double ptorStartTarget = timeStartTarget-params.GetTimeShiftDays()+params.GetPredictorDTimeDays(i_step, i_ptor);
            double ptorEndTarget = timeEndTarget-params.GetTimeShiftDays()+params.GetPredictorDTimeDays(i_step, i_ptor);
            asTimeArray timeArrayDataTarget(ptorStartTarget, ptorEndTarget, params.GetTimeArrayTargetTimeStepHours(), asTimeArray::Simple);
            timeArrayDataTarget.Init();

            // Loading the datasets information
            if(!catalogArchive.Load(params.GetPredictorArchiveDatasetId(i_step, i_ptor), params.GetPredictorArchiveDataId(i_step, i_ptor)))
            {
                asPredictorCriteria::DeleteArray(criteria);
                return false;
            }
            //catalogRealtime.SetRestrictDownloads(false); // Reinitilize the download restriction
            if(!catalogRealtime.Load(params.GetPredictorRealtimeDatasetId(i_step, i_ptor), params.GetPredictorRealtimeDataId(i_step, i_ptor)))
            {
                asPredictorCriteria::DeleteArray(criteria);
                return false;
            }
            catalogRealtime.SetRunDateInUse(m_ForecastDate);

            // Restriction needed
            wxASSERT(params.GetTimeArrayTargetTimeStepHours()>0);
            catalogRealtime.RestrictTimeArray(params.GetPredictorDTimeHours(i_step, i_ptor), params.GetTimeArrayTargetTimeStepHours());

            // Update
            if(!catalogRealtime.BuildFilenamesUrls())
            {
                asPredictorCriteria::DeleteArray(criteria);
                return false;
            }

            // Check time array for real-time data
            VectorDouble catalogTimeArray = catalogRealtime.GetDataDates();
            wxASSERT_MSG(catalogTimeArray.size()>=(unsigned)timeArrayDataTarget.GetSize(), wxString::Format("size of catalogTimeArray = %d, size of timeArrayDataTarget = %d", (int)catalogTimeArray.size(), (int)timeArrayDataTarget.GetSize()));
            for (int i=0; i<timeArrayDataTarget.GetSize(); i++)
            {
                if(catalogTimeArray[i]!=timeArrayDataTarget[i])
                {
                    asLogError(wxString::Format(_("The real-time predictor time array is not consistent (catalogTimeArray[%d](%f)!=timeArrayDataTarget[%d](%f))."), i, catalogTimeArray[i], i, timeArrayDataTarget[i]));
                    asPredictorCriteria::DeleteArray(criteria);
                    return false;
                }
            }

            // Area object instantiation
            wxASSERT(catalogArchive.GetCoordSys()==catalogRealtime.GetCoordSys());
            asGeoAreaCompositeGrid* area = asGeoAreaCompositeGrid::GetInstance(catalogArchive.GetCoordSys(), params.GetPredictorGridType(i_step, i_ptor), params.GetPredictorUmin(i_step, i_ptor), params.GetPredictorUptsnb(i_step, i_ptor), params.GetPredictorUstep(i_step, i_ptor), params.GetPredictorVmin(i_step, i_ptor), params.GetPredictorVptsnb(i_step, i_ptor), params.GetPredictorVstep(i_step, i_ptor), params.GetPredictorLevel(i_step, i_ptor), asNONE, params.GetPredictorFlatAllowed(i_step, i_ptor));

            // Check the starting dates coherence
            if (catalogArchive.GetStart()>ptorStartArchive)
            {
                asLogError(wxString::Format(_("The first year defined in the parameters (%s) is prior to the start date of the catalog (%s)."), asTime::GetStringTime(ptorStartArchive), asTime::GetStringTime(catalogArchive.GetStart())));
                wxDELETE(area);
                asPredictorCriteria::DeleteArray(criteria);
                return false;
            }

            // Archive data loading
            asLogMessage(_("Loading archive data."));
            asDataPredictorArchive predictorArchive(catalogArchive);
            if(!predictorArchive.Load(area, timeArrayDataArchive, m_PredictorsArchiveDir))
            {
                asLogError(_("Archive data could not be loaded."));
                wxDELETE(area);
                asPredictorCriteria::DeleteArray(criteria);
                return false;
            }
            predictorsArchive.push_back(predictorArchive);

            // Realtime data loading
            asLogMessage(_("Loading GCM forecast data."));
            asDataPredictorRealtime predictorRealtime(catalogRealtime);
            if(!predictorRealtime.Load(area, timeArrayDataTarget))
            {
                asLogError(_("Real-time data could not be loaded."));
                wxDELETE(area);
                asPredictorCriteria::DeleteArray(criteria);
                return false;
            }
            wxDELETE(area);
            predictorsRealtime.push_back(predictorRealtime);
        }
        else
        {
            int preprocessSize = params.GetPreprocessSize(i_step, i_ptor);
            std::vector < asDataPredictorArchive > predictorsArchivePreprocess;
            std::vector < asDataPredictorRealtime > predictorsRealtimePreprocess;

            asLogMessage(wxString::Format(_("Preprocessing data (%d predictor(s)) while loading."), preprocessSize));

            for (int i_prepro=0; i_prepro<preprocessSize; i_prepro++)
            {
                #if wxUSE_GUI
                    if (g_Responsive) wxGetApp().Yield();
                #endif

                // Date array object instantiation for the data loading. The array has the same length than timeArrayArchive, and the predictor dates are aligned with the target dates, but the dates are not the same.
                double ptorStartArchive = timeStartArchive-params.GetTimeShiftDays()+params.GetPreprocessDTimeDays(i_step, i_ptor, i_prepro);
                double ptorEndArchive = timeEndArchive-params.GetTimeShiftDays()+params.GetPreprocessDTimeDays(i_step, i_ptor, i_prepro);
                asTimeArray timeArrayDataArchive(ptorStartArchive, ptorEndArchive, params.GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
                timeArrayDataArchive.Init();

                double ptorStartTarget = timeStartTarget-params.GetTimeShiftDays()+params.GetPreprocessDTimeDays(i_step, i_ptor, i_prepro);
                double ptorEndTarget = timeEndTarget-params.GetTimeShiftDays()+params.GetPreprocessDTimeDays(i_step, i_ptor, i_prepro);
                asTimeArray timeArrayDataTarget(ptorStartTarget, ptorEndTarget, params.GetTimeArrayTargetTimeStepHours(), asTimeArray::Simple);
                timeArrayDataTarget.Init();

                // Loading the datasets information
                if(!catalogArchive.Load(params.GetPreprocessArchiveDatasetId(i_step, i_ptor, i_prepro), params.GetPreprocessArchiveDataId(i_step, i_ptor, i_prepro)))
                {
                    asPredictorCriteria::DeleteArray(criteria);
                    return false;
                }
                if(!catalogRealtime.Load(params.GetPreprocessRealtimeDatasetId(i_step, i_ptor, i_prepro), params.GetPreprocessRealtimeDataId(i_step, i_ptor, i_prepro)))
                {
                    asPredictorCriteria::DeleteArray(criteria);
                    return false;
                }
                catalogRealtime.SetRunDateInUse(m_ForecastDate);

                // Restriction needed
                wxASSERT(params.GetTimeArrayTargetTimeStepHours()>0);
                catalogRealtime.RestrictTimeArray(params.GetPreprocessDTimeHours(i_step, i_ptor, i_prepro), params.GetTimeArrayTargetTimeStepHours());

                // Update
                if(!catalogRealtime.BuildFilenamesUrls())
                {
                    asPredictorCriteria::DeleteArray(criteria);
                    return false;
                }

                // Check time array for real-time data
                VectorDouble catalogTimeArray = catalogRealtime.GetDataDates();
                wxASSERT(catalogTimeArray.size()==timeArrayDataTarget.GetSize());
                for (unsigned int i=0; i<catalogTimeArray.size(); i++)
                {
                    if(catalogTimeArray[i]!=timeArrayDataTarget[i])
                    {
                        asLogError(wxString::Format(_("The real-time predictor time array is not consistent (catalogTimeArray[%d](%f)!=timeArrayDataTarget[%d](%f))."), i, catalogTimeArray[i], i, timeArrayDataTarget[i]));
                        asPredictorCriteria::DeleteArray(criteria);
                        return false;
                    }
                }

                // Area object instantiation
                wxASSERT(catalogArchive.GetCoordSys()==catalogRealtime.GetCoordSys());
                asGeoAreaCompositeGrid* area = asGeoAreaCompositeGrid::GetInstance(catalogArchive.GetCoordSys(), params.GetPredictorGridType(i_step, i_ptor), params.GetPredictorUmin(i_step, i_ptor), params.GetPredictorUptsnb(i_step, i_ptor), params.GetPredictorUstep(i_step, i_ptor), params.GetPredictorVmin(i_step, i_ptor), params.GetPredictorVptsnb(i_step, i_ptor), params.GetPredictorVstep(i_step, i_ptor), params.GetPreprocessLevel(i_step, i_ptor, i_prepro), asNONE, params.GetPredictorFlatAllowed(i_step, i_ptor));

                // Check the starting dates coherence
                if (catalogArchive.GetStart()>ptorStartArchive)
                {
                    asLogError(wxString::Format(_("The first year defined in the parameters (%s) is prior to the start date of the catalog (%s)."), asTime::GetStringTime(ptorStartArchive), asTime::GetStringTime(catalogArchive.GetStart())));
                    wxDELETE(area);
                    asPredictorCriteria::DeleteArray(criteria);
                    return false;
                }

                // Archive data loading
                asLogMessage(_("Loading archive data."));
                asDataPredictorArchive predictorArchivePreprocess(catalogArchive);
                if(!predictorArchivePreprocess.Load(area, timeArrayDataArchive, m_PredictorsArchiveDir))
                {
                    asLogError(_("Archive data could not be loaded."));
                    wxDELETE(area);
                    asPredictorCriteria::DeleteArray(criteria);
                    return false;
                }
                predictorsArchivePreprocess.push_back(predictorArchivePreprocess);

                // Realtime data loading
                asLogMessage(_("Loading GCM forecast data."));
                asDataPredictorRealtime predictorRealtimePreprocess(catalogRealtime);
                if(!predictorRealtimePreprocess.Load(area, timeArrayDataTarget))
                {
                    asLogError(_("Real-time data could not be loaded."));
                    wxDELETE(area);
                    asPredictorCriteria::DeleteArray(criteria);
                    return false;
                }
                wxDELETE(area);
                predictorsRealtimePreprocess.push_back(predictorRealtimePreprocess);
            }

            // Fix the criteria if S1
            if(params.GetPredictorCriteria(i_step, i_ptor).IsSameAs("S1"))
            {
                params.SetPredictorCriteria(i_step, i_ptor, "S1grads");
            }

            asDataPredictorArchive predictorArchive(predictorsArchivePreprocess[0]);
            if(!asPreprocessor::Preprocess(predictorsArchivePreprocess, params.GetPreprocessMethod(i_step, i_ptor), &predictorArchive))
            {
               asLogError(_("Data preprocessing failed."));
               return false;
            }

            asDataPredictorRealtime predictorRealtime(predictorsRealtimePreprocess[0]);
            if(!asPreprocessor::Preprocess(predictorsRealtimePreprocess, params.GetPreprocessMethod(i_step, i_ptor), &predictorRealtime))
            {
               asLogError(_("Data preprocessing failed."));
               return false;
            }

            predictorsArchive.push_back(predictorArchive);
            predictorsRealtime.push_back(predictorRealtime);
        }

        asLogMessage(_("Data loaded"));

        // Instantiate a score object
        asLogMessage(_("Creating a criterion object."));
        asPredictorCriteria* criterion = asPredictorCriteria::GetInstance(params.GetPredictorCriteria(i_step, i_ptor), linAlgebraMethod);
        criteria.push_back(criterion);
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

        if(!asProcessor::GetAnalogsDates(predictorsArchive, predictorsRealtime,
                                         timeArrayArchive, timeArrayArchive, timeArrayTarget, timeArrayTargetLeadTime,
                                         criteria, params, i_step, anaDates, containsNaNs))
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

    asPredictorCriteria::DeleteArray(criteria);

    return true;
}

bool asMethodForecasting::GetAnalogsSubDates(asResultsAnalogsForecast &results, asParametersForecast &params, asResultsAnalogsForecast &resultsPrev, int i_step)
{
    // Get the linear algebra method
    int linAlgebraMethod = (int)(wxFileConfig::Get()->Read("/ProcessingOptions/ProcessingLinAlgebra", (long)asCOEFF_NOVAR));

    // Catalog
    asCatalogPredictorsArchive catalogArchive = asCatalogPredictorsArchive();
    asCatalogPredictorsRealtime catalogRealtime = asCatalogPredictorsRealtime();

    // Initialize the result object
    results.SetCurrentStep(i_step);
    results.Init(params, m_ForecastDate);

    // Create the vectors to put the data in
    std::vector < asDataPredictor > predictorsArchive;
    std::vector < asDataPredictor > predictorsRealtime;
    std::vector < asPredictorCriteria* > criteria;

    // Date array object instantiation for the processor
    asLogMessage(_("Creating a date arrays for the processor."));

    // Archive time array
    double timeStartArchive = asTime::GetMJD(params.GetArchiveYearStart(),1,1); // Always Jan 1st
    double timeEndArchive = asTime::GetMJD(params.GetArchiveYearEnd(),12,31);
    timeEndArchive = wxMin(timeEndArchive, timeEndArchive-params.GetTimeSpanDays()); // Adjust so the predictors search won't overtake the array
    asTimeArray timeArrayArchive(timeStartArchive, timeEndArchive, params.GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
    timeArrayArchive.Init();

    // Get last lead time of the data
    double lastLeadTime = 9999;
    for(int i_ptor=0;i_ptor<params.GetPredictorsNb(i_step);i_ptor++)
    {
        if(!params.NeedsPreprocessing(i_step, i_ptor))
        {
            if(!catalogRealtime.Load(params.GetPredictorRealtimeDatasetId(i_step, i_ptor), params.GetPredictorRealtimeDataId(i_step, i_ptor))) return false;
            catalogRealtime.SetRunDateInUse(m_ForecastDate);
            lastLeadTime = wxMin(lastLeadTime, catalogRealtime.GetForecastLeadTimeEnd()/24.0 - params.GetTimeSpanDays());
        }
        else
        {
            for (int i_prepro=0; i_prepro<params.GetPreprocessSize(i_step, i_ptor); i_prepro++)
            {
                if(!catalogRealtime.Load(params.GetPreprocessRealtimeDatasetId(i_step, i_ptor, i_prepro), params.GetPreprocessRealtimeDataId(i_step, i_ptor, i_prepro))) return false;
                catalogRealtime.SetRunDateInUse(m_ForecastDate);
                lastLeadTime = wxMin(lastLeadTime, catalogRealtime.GetForecastLeadTimeEnd()/24.0 - params.GetTimeSpanDays());
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
        neededMem += (params.GetPredictorUptsnb(i_step, i_ptor))
                    * (params.GetPredictorVptsnb(i_step, i_ptor));
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
            double ptorStartArchive = timeStartArchive-params.GetTimeShiftDays()+params.GetPredictorDTimeDays(i_step, i_ptor);
            double ptorEndArchive = timeEndArchive-params.GetTimeShiftDays()+params.GetPredictorDTimeDays(i_step, i_ptor);
            asTimeArray timeArrayDataArchive(ptorStartArchive, ptorEndArchive, params.GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
            timeArrayDataArchive.Init();

            double ptorStartTarget = timeStartTarget-params.GetTimeShiftDays()+params.GetPredictorDTimeDays(i_step, i_ptor);
            double ptorEndTarget = timeEndTarget-params.GetTimeShiftDays()+params.GetPredictorDTimeDays(i_step, i_ptor);
            asTimeArray timeArrayDataTarget(ptorStartTarget, ptorEndTarget, params.GetTimeArrayTargetTimeStepHours(), asTimeArray::Simple);
            timeArrayDataTarget.Init();

            // Loading the datasets information
            if(!catalogArchive.Load(params.GetPredictorArchiveDatasetId(i_step, i_ptor), params.GetPredictorArchiveDataId(i_step, i_ptor)))
            {
                asPredictorCriteria::DeleteArray(criteria);
                return false;
            }
            if(!catalogRealtime.Load(params.GetPredictorRealtimeDatasetId(i_step, i_ptor), params.GetPredictorRealtimeDataId(i_step, i_ptor)))
            {
                asPredictorCriteria::DeleteArray(criteria);
                return false;
            }
            catalogRealtime.SetRunDateInUse(m_ForecastDate);

            // Restriction needed
            wxASSERT(params.GetTimeArrayTargetTimeStepHours()>0);
            catalogRealtime.RestrictTimeArray(params.GetPredictorDTimeHours(i_step, i_ptor), params.GetTimeArrayTargetTimeStepHours());

            // Update
            if(!catalogRealtime.BuildFilenamesUrls())
            {
                asPredictorCriteria::DeleteArray(criteria);
                return false;
            }

            // Check time array for real-time data
            VectorDouble catalogTimeArray = catalogRealtime.GetDataDates();
            wxASSERT_MSG(catalogTimeArray.size()>=(unsigned)timeArrayDataTarget.GetSize(), wxString::Format("size of catalogTimeArray = %d, size of timeArrayDataTarget = %d", (int)catalogTimeArray.size(), (int)timeArrayDataTarget.GetSize()));
            for (unsigned int i=0; i<(unsigned)timeArrayDataTarget.GetSize(); i++)
            {
                if(catalogTimeArray[i]!=timeArrayDataTarget[i])
                {
                    asLogError(wxString::Format(_("The real-time predictor time array is not consistent (catalogTimeArray[%d](%f)!=timeArrayDataTarget[%d](%f))."), i, catalogTimeArray[i], i, timeArrayDataTarget[i]));
                    asPredictorCriteria::DeleteArray(criteria);
                    return false;
                }
            }

            // Area object instantiation
            wxASSERT(catalogArchive.GetCoordSys()==catalogRealtime.GetCoordSys());
            asGeoAreaCompositeGrid* area = asGeoAreaCompositeGrid::GetInstance(catalogArchive.GetCoordSys(), params.GetPredictorGridType(i_step, i_ptor), params.GetPredictorUmin(i_step, i_ptor), params.GetPredictorUptsnb(i_step, i_ptor), params.GetPredictorUstep(i_step, i_ptor), params.GetPredictorVmin(i_step, i_ptor), params.GetPredictorVptsnb(i_step, i_ptor), params.GetPredictorVstep(i_step, i_ptor), params.GetPredictorLevel(i_step, i_ptor), asNONE, params.GetPredictorFlatAllowed(i_step, i_ptor));

            // Check the starting dates coherence
            if (catalogArchive.GetStart()>ptorStartArchive)
            {
                asLogError(wxString::Format(_("The first year defined in the parameters (%s) is prior to the start date of the catalog (%s)."), asTime::GetStringTime(ptorStartArchive), asTime::GetStringTime(catalogArchive.GetStart())));
                wxDELETE(area);
                asPredictorCriteria::DeleteArray(criteria);
                return false;
            }

            // Archive data loading
            asDataPredictorArchive predictorArchive(catalogArchive);
            if(!predictorArchive.Load(area, timeArrayDataArchive, m_PredictorsArchiveDir))
            {
                asLogError(_("Archive data could not be loaded."));
                wxDELETE(area);
                asPredictorCriteria::DeleteArray(criteria);
                return false;
            }
            predictorsArchive.push_back(predictorArchive);

            // Realtime data loading
            asDataPredictorRealtime predictorRealtime(catalogRealtime);
            if(!predictorRealtime.Load(area, timeArrayDataTarget))
            {
                asLogError(_("Real-time data could not be loaded."));
                wxDELETE(area);
                asPredictorCriteria::DeleteArray(criteria);
                return false;
            }
            wxDELETE(area);
            predictorsRealtime.push_back(predictorRealtime);
        }
        else
        {
            int preprocessSize = params.GetPreprocessSize(i_step, i_ptor);
            std::vector < asDataPredictorArchive > predictorsArchivePreprocess;
            std::vector < asDataPredictorRealtime > predictorsRealtimePreprocess;

            asLogMessage(wxString::Format(_("Preprocessing data (%d predictor(s)) while loading."), preprocessSize));

            for (int i_prepro=0; i_prepro<preprocessSize; i_prepro++)
            {
                // Date array object instantiation for the data loading. The array has the same length than timeArrayArchive, and the predictor dates are aligned with the target dates, but the dates are not the same.
                double ptorStartArchive = timeStartArchive-params.GetTimeShiftDays()+params.GetPreprocessDTimeDays(i_step, i_ptor, i_prepro);
                double ptorEndArchive = timeEndArchive-params.GetTimeShiftDays()+params.GetPreprocessDTimeDays(i_step, i_ptor, i_prepro);
                asTimeArray timeArrayDataArchive(ptorStartArchive, ptorEndArchive, params.GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
                timeArrayDataArchive.Init();

                double ptorStartTarget = timeStartTarget-params.GetTimeShiftDays()+params.GetPreprocessDTimeDays(i_step, i_ptor, i_prepro);
                double ptorEndTarget = timeEndTarget-params.GetTimeShiftDays()+params.GetPreprocessDTimeDays(i_step, i_ptor, i_prepro);
                asTimeArray timeArrayDataTarget(ptorStartTarget, ptorEndTarget, params.GetTimeArrayTargetTimeStepHours(), asTimeArray::Simple);
                timeArrayDataTarget.Init();

                // Loading the datasets information
                if(!catalogArchive.Load(params.GetPreprocessArchiveDatasetId(i_step, i_ptor, i_prepro), params.GetPreprocessArchiveDataId(i_step, i_ptor, i_prepro)))
                {
                    asPredictorCriteria::DeleteArray(criteria);
                    return false;
                }
                if(!catalogRealtime.Load(params.GetPreprocessRealtimeDatasetId(i_step, i_ptor, i_prepro), params.GetPreprocessRealtimeDataId(i_step, i_ptor, i_prepro)))
                {
                    asPredictorCriteria::DeleteArray(criteria);
                    return false;
                }
                catalogRealtime.SetRunDateInUse(m_ForecastDate);

                // Restriction needed
                wxASSERT(params.GetTimeArrayTargetTimeStepHours()>0);
                catalogRealtime.RestrictTimeArray(params.GetPreprocessDTimeHours(i_step, i_ptor, i_prepro), params.GetTimeArrayTargetTimeStepHours());

                // Update
                if(!catalogRealtime.BuildFilenamesUrls())
                {
                    asPredictorCriteria::DeleteArray(criteria);
                    return false;
                }

                // Check time array for real-time data
                VectorDouble catalogTimeArray = catalogRealtime.GetDataDates();
                wxASSERT_MSG(catalogTimeArray.size()>=(unsigned)timeArrayDataTarget.GetSize(), wxString::Format("catalogTimeArray.size() = %d, timeArrayDataTarget.GetSize() = %d", (int)catalogTimeArray.size(), (int)timeArrayDataTarget.GetSize()));
                for (unsigned int i=0; i<(unsigned)timeArrayDataTarget.GetSize(); i++)
                {
                    if(catalogTimeArray[i]!=timeArrayDataTarget[i])
                    {
                        asLogError(wxString::Format(_("The real-time predictor time array is not consistent (catalogTimeArray[%d](%f)!=timeArrayDataTarget[%d](%f))."), i, catalogTimeArray[i], i, timeArrayDataTarget[i]));
                        asPredictorCriteria::DeleteArray(criteria);
                        return false;
                    }
                }

                // Area object instantiation
                wxASSERT(catalogArchive.GetCoordSys()==catalogRealtime.GetCoordSys());
                asGeoAreaCompositeGrid* area = asGeoAreaCompositeGrid::GetInstance(catalogArchive.GetCoordSys(), params.GetPredictorGridType(i_step, i_ptor), params.GetPredictorUmin(i_step, i_ptor), params.GetPredictorUptsnb(i_step, i_ptor), params.GetPredictorUstep(i_step, i_ptor), params.GetPredictorVmin(i_step, i_ptor), params.GetPredictorVptsnb(i_step, i_ptor), params.GetPredictorVstep(i_step, i_ptor), params.GetPreprocessLevel(i_step, i_ptor, i_prepro), asNONE, params.GetPredictorFlatAllowed(i_step, i_ptor));

                // Check the starting dates coherence
                if (catalogArchive.GetStart()>ptorStartArchive)
                {
                    asLogError(wxString::Format(_("The first year defined in the parameters (%s) is prior to the start date of the catalog (%s)."), asTime::GetStringTime(ptorStartArchive), asTime::GetStringTime(catalogArchive.GetStart())));
                    wxDELETE(area);
                    asPredictorCriteria::DeleteArray(criteria);
                    return false;
                }

                // Archive data loading
                asDataPredictorArchive predictorArchivePreprocess(catalogArchive);
                if(!predictorArchivePreprocess.Load(area, timeArrayDataArchive, m_PredictorsArchiveDir))
                {
                    asLogError(_("Archive data could not be loaded."));
                    wxDELETE(area);
                    asPredictorCriteria::DeleteArray(criteria);
                    return false;
                }
                predictorsArchivePreprocess.push_back(predictorArchivePreprocess);

                // Realtime data loading
                asDataPredictorRealtime predictorRealtimePreprocess(catalogRealtime);
                if(!predictorRealtimePreprocess.Load(area, timeArrayDataTarget))
                {
                    asLogError(_("Real-time data could not be loaded."));
                    wxDELETE(area);
                    asPredictorCriteria::DeleteArray(criteria);
                    return false;
                }
                wxDELETE(area);
                predictorsRealtimePreprocess.push_back(predictorRealtimePreprocess);

                wxASSERT_MSG(predictorArchivePreprocess.GetLatPtsnb()==predictorRealtimePreprocess.GetLatPtsnb(), wxString::Format("predictorArchivePreprocess.GetLatPtsnb()=%d, predictorRealtimePreprocess.GetLatPtsnb()=%d",predictorArchivePreprocess.GetLatPtsnb(), predictorRealtimePreprocess.GetLatPtsnb()));
                wxASSERT_MSG(predictorArchivePreprocess.GetLonPtsnb()==predictorRealtimePreprocess.GetLonPtsnb(), wxString::Format("predictorArchivePreprocess.GetLonPtsnb()=%d, predictorRealtimePreprocess.GetLonPtsnb()=%d",predictorArchivePreprocess.GetLonPtsnb(), predictorRealtimePreprocess.GetLonPtsnb()));
            }

            // Fix the criteria if S1
            if(params.GetPredictorCriteria(i_step, i_ptor).IsSameAs("S1"))
            {
                params.SetPredictorCriteria(i_step, i_ptor, "S1grads");
            }

            asDataPredictorArchive predictorArchive(predictorsArchivePreprocess[0]);;
            if(!asPreprocessor::Preprocess(predictorsArchivePreprocess, params.GetPreprocessMethod(i_step, i_ptor), &predictorArchive))
            {
               asLogError(_("Data preprocessing failed."));
               return false;
            }

            asDataPredictorRealtime predictorRealtime(predictorsRealtimePreprocess[0]);;
            if(!asPreprocessor::Preprocess(predictorsRealtimePreprocess, params.GetPreprocessMethod(i_step, i_ptor), &predictorRealtime))
            {
               asLogError(_("Data preprocessing failed."));
               return false;
            }

            wxASSERT(predictorArchive.GetLatPtsnb()==predictorRealtime.GetLatPtsnb());
            wxASSERT(predictorArchive.GetLonPtsnb()==predictorRealtime.GetLonPtsnb());
            predictorsArchive.push_back(predictorArchive);
            predictorsRealtime.push_back(predictorRealtime);
        }

        asLogMessage(_("Data loaded"));

        // Instantiate a score object
        asLogMessage(_("Creating a criterion object."));
        asPredictorCriteria* criterion = asPredictorCriteria::GetInstance(params.GetPredictorCriteria(i_step, i_ptor), linAlgebraMethod);
        criteria.push_back(criterion);
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

        if(!asProcessor::GetAnalogsSubDates(predictorsArchive, predictorsRealtime, timeArrayArchive, timeArrayTarget, anaDatesPrev, criteria, params, i_step, anaDates, containsNaNs))
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

    asPredictorCriteria::DeleteArray(criteria);

    return true;
}

bool asMethodForecasting::GetAnalogsValues(asResultsAnalogsForecast &results, asParametersForecast &params, int i_step)
{
    // Initialize the result object
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
    results.SetStationsIds(stationsId);
    results.SetStationsNames(m_PredictandDB->GetStationsNameArray());
    results.SetStationsHeights(m_PredictandDB->GetStationsHeightArray());
    results.SetStationsLat(m_PredictandDB->GetStationsLatArray());
    results.SetStationsLon(m_PredictandDB->GetStationsLonArray());
    results.SetStationsLocCoordU(m_PredictandDB->GetStationsLocCoordUArray());
    results.SetStationsLocCoordV(m_PredictandDB->GetStationsLocCoordVArray());
    results.SetReferenceAxis(m_PredictandDB->GetReferenceAxis());
    results.SetReferenceValues(m_PredictandDB->GetReferenceValuesArray());

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
            int stationId = stationsId[i_stat];

            // Set the next station ID
            params.SetPredictandStationId(stationId);

            // Create a standard analogs values result object
            asResultsAnalogsValues anaValues = asResultsAnalogsValues();
            anaValues.SetCurrentStep(i_step);
            anaValues.Init(params);

            if(!asProcessor::GetAnalogsValues(*m_PredictandDB, anaDates, params, anaValues))
            {
                asLogError(_("Failed setting the predictand values to the corresponding analog dates."));
                return false;
            }

            Array2DFloat valuesGross = anaValues.GetAnalogsValuesGross();
            wxASSERT(valuesGross.rows()==1);
            Array1DFloat rowValuesGross = valuesGross.row(0);
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

