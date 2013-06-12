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
 
#include "asProcessor.h"

#include <asTimeArray.h>
#include <asParameters.h>
#include <asPreprocessor.h>
#include <asPredictorCriteria.h>
#include <asDataPredictorArchive.h>
#include <asDataPredictorRealtime.h>
#include <asDataPredictand.h>
#include <asResultsAnalogsDates.h>
#include <asResultsAnalogsValues.h>
//#include <asDialogProgressBar.h>
#include <asThreadProcessorGetAnalogsDates.h>
#include <asThreadProcessorGetAnalogsSubDates.h>
#ifndef UNIT_TESTING
	#include <AtmoswingAppForecaster.h>
#endif

bool asProcessor::GetAnalogsDates(std::vector < asDataPredictor > &predictorsArchive,
                                  std::vector < asDataPredictor > &predictorsTarget,
                                  asTimeArray &timeArrayArchiveData,
                                  asTimeArray &timeArrayArchiveSelection,
                                  asTimeArray &timeArrayTargetData,
                                  asTimeArray &timeArrayTargetSelection,
                                  std::vector < asPredictorCriteria* > criteria,
                                  asParameters &params,
                                  int step,
                                  asResultsAnalogsDates &results,
                                  bool &containsNaNs)
{
	#if wxUSE_GUI
		wxGetApp().Yield();
	#endif

    // Get the processing method
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    long defaultMethod = (long)asINSERT;
    int method = (int)(pConfig->Read("/ProcessingOptions/ProcessingMethod", defaultMethod));
    bool allowMultithreading;
    pConfig->Read("/Standard/AllowMultithreading", &allowMultithreading, false);

    // Check options compatibility
    if (!allowMultithreading && method==asMULTITHREADS)
    {
        method = asINSERT;
    }

    // Check available threads
    if (method==asMULTITHREADS)
    {
        int threadsNb = ThreadsManager().GetAvailableThreadsNb();
        if (threadsNb<2)
        {
            method = asINSERT;
        }
    }

    ThreadsManager().CritSectionConfig().Leave();

    // Check the step
    if (step>0)
    {
        asLogError(_("The Analogs SubDates method must be called for this step."));
        return false;
    }

    // Watch
    wxStopWatch sw;

    // Extract some data
    Array1DDouble timeTargetSelection = timeArrayTargetSelection.GetTimeArray();
    int timeTargetSelectionSize = timeTargetSelection.size();
    bool isasc = (criteria[0]->GetOrder()==Asc);
    int predictorsNb = params.GetPredictorsNb(step);
    wxASSERT(predictorsArchive.size()>0);
    wxASSERT(predictorsArchive.size()==predictorsNb);

    // Check analogs number. Correct if superior to the time serie
    int analogsNb = params.GetAnalogsNumber(step);
    if(analogsNb>timeArrayArchiveSelection.GetSize())
    {
        asLogError(_("The given analog number is superior to the time serie."));
        return false;
    }

    // Matrices containers
    VpArray2DFloat vTargData = VpArray2DFloat(predictorsNb);
    VpArray2DFloat vArchData = VpArray2DFloat(predictorsNb);
    Array1DInt vRowsNb(predictorsNb);
    Array1DInt vColsNb(predictorsNb);

    for (int i_ptor=0; i_ptor<predictorsNb; i_ptor++)
    {
        vRowsNb[i_ptor] = predictorsArchive[i_ptor].GetData()[0].rows();
        vColsNb[i_ptor] = predictorsArchive[i_ptor].GetData()[0].cols();

        // Check criteria ordering
        if(isasc != (criteria[i_ptor]->GetOrder()==Asc))
        {
            asLogError(_("You cannot combine criteria that are ascendant and descendant."));
            return false;
        }
    }

    // Containers for final results
    Array2DFloat finalAnalogsCriteria(timeTargetSelectionSize, analogsNb);
    Array2DFloat finalAnalogsDates(timeTargetSelectionSize, analogsNb);

    // The progress bar
    wxString dialogmessage = _("Processing the data comparison.\n");
	#if wxUSE_GUI
		asDialogProgressBar ProgressBar(dialogmessage, timeTargetSelectionSize);
	#endif

    switch (method)
    {

        case (asMULTITHREADS):
        {
            bool enableMessageBox = false;
            if (Log().IsMessageBoxOnErrorEnabled()) enableMessageBox = true;
            Log().DisableMessageBoxOnError();

			#if wxUSE_GUI
				if (g_Responsive) wxGetApp().Yield();
			#endif

            // Get threads number
            int threadsNb = ThreadsManager().GetAvailableThreadsNb();

            // Adapt to the number of targets
            if (2*threadsNb>timeTargetSelectionSize)
            {
                threadsNb = 1;
            }

            // Create and give data
            int start = 0, end = -1;
            int threadType = -1;
            std::vector < bool* > vContainsNaNs;
            for (int i_threads=0; i_threads<threadsNb; i_threads++)
            {
                bool* flag = new bool;
                *flag = false;
                vContainsNaNs.push_back(flag);
                start = end+1;
                end = ceil(((float)(i_threads+1)*(float)(timeTargetSelectionSize-1)/(float)threadsNb));
                wxASSERT_MSG(end>=start, wxString::Format("start = %d, end = %d, timeTargetSelectionSize = %d", start, end, timeTargetSelectionSize));

                asThreadProcessorGetAnalogsDates* thread = new asThreadProcessorGetAnalogsDates(&predictorsArchive, &predictorsTarget,
                                                                                                &timeArrayArchiveData, &timeArrayArchiveSelection,
                                                                                                &timeArrayTargetData, &timeArrayTargetSelection,
                                                                                                criteria, params, step,
                                                                                                vTargData, vArchData, vRowsNb, vColsNb,
                                                                                                start, end, &finalAnalogsCriteria, &finalAnalogsDates,
                                                                                                flag);
                threadType = thread->GetType();

                ThreadsManager().AddThread(thread);
            }

            // Wait until all done
            ThreadsManager().Wait(threadType);

            for (unsigned int i_threads=0; i_threads<vContainsNaNs.size()-1; i_threads++)
            {
                if(*vContainsNaNs[i_threads])
                {
                    containsNaNs = true;
                }
                wxDELETE(vContainsNaNs[i_threads]);
            }
            if (containsNaNs)
            {
                return true;
            }

            if (enableMessageBox) Log().EnableMessageBoxOnError();

            break;
        }

        case (asINSERT):
        {
            // Extract some data
            Array1DDouble timeArchiveData = timeArrayArchiveData.GetTimeArray();
            int timeArchiveDataSize = timeArchiveData.size();
            wxASSERT(timeArchiveDataSize>0);
            wxASSERT(predictorsArchive.size()>0);
            wxASSERT(predictorsArchive[0].GetData().size()>0);
            wxASSERT_MSG(timeArchiveDataSize==predictorsArchive[0].GetData().size(), wxString::Format("timeArchiveDataSize = %d, predictorsArchive[0].GetData().size() = %d", timeArchiveDataSize, (int)predictorsArchive[0].GetData().size()));
            Array1DDouble timeTargetData = timeArrayTargetData.GetTimeArray();
            int timeTargetDataSize = timeTargetData.size();
            wxASSERT(timeTargetDataSize==predictorsTarget[0].GetData().size());

            // Containers for daily results
            Array1DFloat ScoreArrayOneDay(analogsNb);
            Array1DFloat DateArrayOneDay(analogsNb);

            // Some other variables
            float tmpscore, thisscore;
            int counter = 0;
            int i_timeTarg, i_timeArch, i_timeTargRelative, i_timeArchRelative;

            // DateArray object instantiation. There is one array for all the predictors, as they are aligned, so it picks the predictors we are interested in, but which didn't take place at the same time.
            asTimeArray dateArrayArchiveSelection(timeArrayArchiveSelection.GetStart(), timeArrayArchiveSelection.GetEnd(), params.GetTimeArrayAnalogsTimeStepHours(), params.GetTimeArrayAnalogsMode());
            if(timeArrayArchiveSelection.HasForbiddenYears())
            {
                dateArrayArchiveSelection.SetForbiddenYears(timeArrayArchiveSelection.GetForbiddenYears());
            }

            // Reset the index start target
            int i_timeTargStart = 0;

            // Loop through every timestep as target data
            for (int i_dateTarg=0; i_dateTarg<timeTargetSelectionSize; i_dateTarg++)
            {
				#if wxUSE_GUI
					if (g_Responsive) wxGetApp().Yield();

					// Update the progress bar
					wxString updatedialogmessage = dialogmessage + wxString::Format(_("Processing: %d / %d time steps"), i_dateTarg+1, timeTargetSelectionSize);
					if(!ProgressBar.Update(i_dateTarg, updatedialogmessage))
					{
						asLogMessage(_("The process has been canceled by the user."));
						return false;
					}
				#endif

                // Check if the next data is the following. If not, search for it in the array.
                if(timeTargetDataSize>i_timeTargStart+1 && abs(timeTargetSelection[i_dateTarg]-timeTargetData[i_timeTargStart+1])<0.01)
                {
                    i_timeTargRelative = 1;
                } else {
                    i_timeTargRelative = asTools::SortedArraySearch(&timeTargetData[i_timeTargStart], &timeTargetData[timeTargetDataSize-1], timeTargetSelection[i_dateTarg], 0.01);
                }

                // Check if a row was found
                if (i_timeTargRelative!=asNOT_FOUND && i_timeTargRelative!=asOUT_OF_RANGE)
                {
                    // Convert the relative index into an absolute index
                    i_timeTarg = i_timeTargRelative+i_timeTargStart;
                    i_timeTargStart = i_timeTarg;

                    // Extract target data
                    for (int i_ptor=0; i_ptor<predictorsNb; i_ptor++)
                    {
                        vTargData[i_ptor] = &predictorsTarget[i_ptor].GetData()[i_timeTarg];
                    }
                    // DateArray object initialization.
                    dateArrayArchiveSelection.Init(timeTargetSelection[i_dateTarg], params.GetTimeArrayAnalogsIntervalDays(), params.GetTimeArrayAnalogsExcludeDays());

                    // Counter representing the current index
                    counter = 0;
                    int NaNcounter = 0;

                    // Reset the index start target
                    int i_timeArchStart = 0;

                    // Loop through the datearray for candidate data
                    for (int i_dateArch=0; i_dateArch<dateArrayArchiveSelection.GetSize(); i_dateArch++)
                    {
                        // Check if the next data is the following. If not, search for it in the array.
                        if(timeArchiveDataSize>i_timeArchStart+1 && abs(dateArrayArchiveSelection[i_dateArch]-timeArchiveData[i_timeArchStart+1])<0.01)
                        {
                            i_timeArchRelative = 1;
                        } else {
                            i_timeArchRelative = asTools::SortedArraySearch(&timeArchiveData[i_timeArchStart], &timeArchiveData[timeArchiveDataSize-1], dateArrayArchiveSelection[i_dateArch], 0.01);
                        }

                        // Check if a row was found
                        if (i_timeArchRelative!=asNOT_FOUND && i_timeArchRelative!=asOUT_OF_RANGE)
                        {
                            // Convert the relative index into an absolute index
                            i_timeArch = i_timeArchRelative+i_timeArchStart;
                            i_timeArchStart = i_timeArch;

                            // Process the criteria
                            thisscore = 0;
                            for (int i_ptor=0; i_ptor<predictorsNb; i_ptor++)
                            {
                                // Get data
                                vArchData[i_ptor] = &predictorsArchive[i_ptor].GetData()[i_timeArch];

                                // Assess the criteria
                                wxASSERT(criteria.size()>(unsigned)i_ptor);
                                wxASSERT(vTargData[i_ptor]);
                                wxASSERT(vArchData[i_ptor]);
                                tmpscore = criteria[i_ptor]->Assess(*vTargData[i_ptor], *vArchData[i_ptor], vRowsNb[i_ptor], vColsNb[i_ptor]);

                                // Weight and add the score
                                thisscore += tmpscore * params.GetPredictorWeight(step, i_ptor);
                            }
                            if (asTools::IsNaN(thisscore))
                            {
                                NaNcounter++;
                                if (NaNcounter>=10)
                                {
                                    containsNaNs = true;
                                    return true;
                                }
                            }

                            // Check if the array is already full
                            if (counter>analogsNb-1)
                            {
                                if (isasc)
                                {
                                    if (thisscore<ScoreArrayOneDay[analogsNb-1])
                                    {
                                        asTools::SortedArraysInsert(&ScoreArrayOneDay[0], &ScoreArrayOneDay[analogsNb-1], &DateArrayOneDay[0], &DateArrayOneDay[analogsNb-1], Asc, thisscore, (float)timeArchiveData[i_timeArch]);
                                    }
                                } else {
                                    if (thisscore>ScoreArrayOneDay[analogsNb-1])
                                    {
                                        asTools::SortedArraysInsert(&ScoreArrayOneDay[0], &ScoreArrayOneDay[analogsNb-1], &DateArrayOneDay[0], &DateArrayOneDay[analogsNb-1], Desc, thisscore, (float)timeArchiveData[i_timeArch]);
                                    }
                                }
                            }
                            else if (counter<analogsNb-1)
                            {
                                // Add score and date to the vectors
                                ScoreArrayOneDay[counter] = thisscore;
                                DateArrayOneDay[counter] = (float)timeArchiveData[i_timeArch];
                            }
                            else if (counter==analogsNb-1)
                            {
                                // Add score and date to the vectors
                                ScoreArrayOneDay[counter] = thisscore;
                                DateArrayOneDay[counter] = (float)timeArchiveData[i_timeArch];

                                // Sort both scores and dates arrays
                                if (isasc)
                                {
                                    asTools::SortArrays(&ScoreArrayOneDay[0], &ScoreArrayOneDay[analogsNb-1], &DateArrayOneDay[0], &DateArrayOneDay[analogsNb-1], Asc);
                                } else {
                                    asTools::SortArrays(&ScoreArrayOneDay[0], &ScoreArrayOneDay[analogsNb-1], &DateArrayOneDay[0], &DateArrayOneDay[analogsNb-1], Desc);
                                }
                            }

                            counter++;
                        }
                        else
                        {
                            asLogError(_("The date was not found in the array (Analogs subdates fct, insert option). That should not happen."));
                        }
                    }

                    // Check that the number of occurences are larger than the desired analogs number. If not, set a warning
                    if (counter>analogsNb)
                    {
                        // Copy results
                        finalAnalogsCriteria.row(i_dateTarg) = ScoreArrayOneDay.transpose();
                        finalAnalogsDates.row(i_dateTarg) = DateArrayOneDay.transpose();
                    }
                    else
                    {
                        asLogWarning(_("There is not enough available data to satisfy the number of analogs."));
                    }
                }
            }

            break;
        }

        case (asFULL_ARRAY):
        {
            // Extract some data
            Array1DDouble timeArchiveData = timeArrayArchiveData.GetTimeArray();
            int timeArchiveDataSize = timeArchiveData.size();
            wxASSERT(timeArchiveDataSize==predictorsArchive[0].GetData().size());
            Array1DDouble timeTargetData = timeArrayTargetData.GetTimeArray();
            int timeTargetDataSize = timeTargetData.size();
            wxASSERT(timeTargetDataSize==predictorsTarget[0].GetData().size());

            // Containers for daily results
            Array1DFloat ScoreArrayOneDay(timeArrayArchiveSelection.GetSize());
            Array1DFloat DateArrayOneDay(timeArrayArchiveSelection.GetSize());

            // Some other variables
            float tmpscore, thisscore;
            int counter = 0;
            int i_timeTarg, i_timeArch, i_timeTargRelative, i_timeArchRelative;

            // DateArray object instantiation. There is one array for all the predictors, as they are aligned, so it picks the predictors we are interested in, but which didn't take place at the same time.
            asTimeArray dateArrayArchiveSelection(timeArrayArchiveSelection.GetStart(), timeArrayArchiveSelection.GetEnd(), params.GetTimeArrayAnalogsTimeStepHours(), params.GetTimeArrayAnalogsMode());
            if(timeArrayArchiveSelection.HasForbiddenYears())
            {
                dateArrayArchiveSelection.SetForbiddenYears(timeArrayArchiveSelection.GetForbiddenYears());
            }

            // Reset the index start target
            int i_timeTargStart = 0;

            // Loop through every timestep as target data
            for (int i_dateTarg=0; i_dateTarg<timeTargetSelectionSize; i_dateTarg++)
            {
				#if wxUSE_GUI
					if (g_Responsive) wxGetApp().Yield();

					// Update the progress bar
					wxString updatedialogmessage = dialogmessage + wxString::Format(_("Processing: %d / %d time steps"), (int)i_dateTarg, timeTargetSelectionSize-1);
					if(!ProgressBar.Update(i_dateTarg, updatedialogmessage))
					{
						asLogMessage(_("The process has been canceled by the user."));
						return false;
					}
				#endif

                // Check if the next data is the following. If not, search for it in the array.
                if(timeTargetDataSize>i_timeTargStart+1 && abs(timeTargetSelection[i_dateTarg]-timeTargetData[i_timeTargStart+1])<0.01)
                {
                    i_timeTargRelative = 1;
                } else {
                    i_timeTargRelative = asTools::SortedArraySearch(&timeTargetData[i_timeTargStart], &timeTargetData[timeTargetDataSize-1], timeTargetSelection[i_dateTarg], 0.01);
                }

                // Check if a row was found
                if (i_timeTargRelative!=asNOT_FOUND && i_timeTargRelative!=asOUT_OF_RANGE)
                {
                    // Convert the relative index into an absolute index
                    i_timeTarg = i_timeTargRelative+i_timeTargStart;
                    i_timeTargStart = i_timeTarg;

                    // Extract target data
                    for (int i_ptor=0; i_ptor<predictorsNb; i_ptor++)
                    {
                        vTargData[i_ptor] = &predictorsTarget[i_ptor].GetData()[i_timeTarg];
                    }
                    // DateArray initialization
                    dateArrayArchiveSelection.Init(timeTargetSelection[i_dateTarg], params.GetTimeArrayAnalogsIntervalDays(), params.GetTimeArrayAnalogsExcludeDays());

                    // Counter representing the current index
                    counter = 0;
                    int NaNcounter = 0;

                    // Reset the index start target
                    int i_timeArchStart = 0;

                    // Loop through the dateArrayArchiveSelection for candidate data
                    for (int i_dateArch=0; i_dateArch<dateArrayArchiveSelection.GetSize(); i_dateArch++)
                    {
                        // Check if the next data is the following. If not, search for it in the array.
                        if(timeArchiveDataSize>i_timeArchStart+1 && abs(dateArrayArchiveSelection[i_dateArch]-timeArchiveData[i_timeArchStart+1])<0.01)
                        {
                            i_timeArchRelative = 1;
                        } else {
                            i_timeArchRelative = asTools::SortedArraySearch(&timeArchiveData[i_timeArchStart], &timeArchiveData[timeArchiveDataSize-1], dateArrayArchiveSelection[i_dateArch], 0.01);
                        }

                        // Check if a row was found
                        if (i_timeArchRelative!=asNOT_FOUND && i_timeArchRelative!=asOUT_OF_RANGE)
                        {
                            // Convert the relative index into an absolute index
                            i_timeArch = i_timeArchRelative+i_timeArchStart;
                            i_timeArchStart = i_timeArch;

                            // Process the criteria
                            thisscore = 0;
                            for (int i_ptor=0; i_ptor<predictorsNb; i_ptor++)
                            {
                                // Get data
                                vArchData[i_ptor] = &predictorsArchive[i_ptor].GetData()[i_timeArch];

                                // Assess the criteria
                                wxASSERT(criteria.size()>(unsigned)i_ptor);
                                tmpscore = criteria[i_ptor]->Assess(*vTargData[i_ptor], *vArchData[i_ptor], vRowsNb[i_ptor], vColsNb[i_ptor]);

                                // Weight and add the score
                                thisscore += tmpscore * params.GetPredictorWeight(step, i_ptor);
                            }
                            if (asTools::IsNaN(thisscore))
                            {
                                NaNcounter++;
                                if (NaNcounter>=10)
                                {
                                    containsNaNs = true;
                                    return true;
                                }
                            }

                            // Store in the result array
                            ScoreArrayOneDay[counter] = thisscore;
                            DateArrayOneDay[counter] = (float)timeArchiveData[i_timeArch];
                            counter++;
                        }
                        else
                        {
                            asLogError(_("The date was not found in the array (Analogs dates fct, full array option). That should not happen."));
                        }
                    }

                    // Check that the number of occurences are larger than the desired analogs number. If not, set a warning
                    if (counter>=analogsNb)
                    {
                        // Sort both scores and dates arrays
                        if (isasc)
                        {
                            asTools::SortArrays(&ScoreArrayOneDay[0], &ScoreArrayOneDay[counter-1], &DateArrayOneDay[0], &DateArrayOneDay[counter-1], Asc);
                        } else {
                            asTools::SortArrays(&ScoreArrayOneDay[0], &ScoreArrayOneDay[counter-1], &DateArrayOneDay[0], &DateArrayOneDay[counter-1], Desc);
                        }

                        // Copy results
                        finalAnalogsCriteria.row(i_dateTarg) = ScoreArrayOneDay.head(analogsNb).transpose();
                        finalAnalogsDates.row(i_dateTarg) = DateArrayOneDay.head(analogsNb).transpose();
                    }
                    else
                    {
                        asLogWarning(_("There is not enough available data to satisfy the number of analogs"));
                        asLogWarning(wxString::Format(_("Analogs number (%d) > counter (%d), date array size (%d) with %d days intervals."), analogsNb, counter, dateArrayArchiveSelection.GetSize(), params.GetTimeArrayAnalogsIntervalDays()));
                    }
                }
            }

            break;
        }

        default:
            asThrowException(_("The processing method is not correctly defined."));
    }

	#if wxUSE_GUI
		ProgressBar.Destroy();
		if (g_Responsive) wxGetApp().Yield();
	#endif

    // Copy results to the resulting object
    results.SetTargetDates(timeTargetSelection);
    results.SetAnalogsCriteria(finalAnalogsCriteria);
    results.SetAnalogsDates(finalAnalogsDates);

    // Display the time the function took
    asLogMessage(wxString::Format(_("The function asProcessor::GetAnalogsDates took %ldms to execute"), sw.Time()));

    return true;
}

bool asProcessor::GetAnalogsSubDates(std::vector < asDataPredictor > &predictorsArchive,
                                     std::vector < asDataPredictor > &predictorsTarget,
                                     asTimeArray &timeArrayArchiveData,
                                     asTimeArray &timeArrayTargetData,
                                     asResultsAnalogsDates &anaDates,
                                     std::vector < asPredictorCriteria* > criteria,
                                     asParameters &params,
                                     int step,
                                     asResultsAnalogsDates &results,
                                     bool &containsNaNs)
{
	#if wxUSE_GUI
		wxGetApp().Yield();
	#endif

    // Get the processing method
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    long defaultMethod = (long)asINSERT;
    int method = (int)(pConfig->Read("/ProcessingOptions/ProcessingMethod", defaultMethod));
    bool allowMultithreading;
    pConfig->Read("/Standard/AllowMultithreading", &allowMultithreading, false);

    // Check options compatibility
    if (!allowMultithreading && method==asMULTITHREADS)
    {
        method = asINSERT;
    }

    // Check available threads
    if (method==asMULTITHREADS)
    {
        int threadsNb = ThreadsManager().GetAvailableThreadsNb();
        if (threadsNb<2)
        {
            method = asINSERT;
        }
    }

    ThreadsManager().CritSectionConfig().Leave();

    // Check the step
    if (step==0)
    {
        asLogError(_("The  Analogs Dates method cannot be called first."));
        return false;
    }

    // Watch
    wxStopWatch sw;

    // Extract some data
    Array1DDouble timeArchiveData = timeArrayArchiveData.GetTimeArray();
    int timeArchiveDataSize = timeArchiveData.size();
    Array1DDouble timeTargetData = timeArrayTargetData.GetTimeArray();
    int timeTargetDataSize = timeTargetData.size();
    Array1DFloat timeTargetSelection = anaDates.GetTargetDates();
    int timeTargetSelectionSize = timeTargetSelection.size();
    Array2DFloat analogsDates = anaDates.GetAnalogsDates();
    bool isasc = (criteria[0]->GetOrder()==Asc);
    int predictorsNb = params.GetPredictorsNb(step);

    // Check the analogs number. Correct if superior to the time serie
    int analogsNb = params.GetAnalogsNumber(step);
    int analogsNbPrevious = params.GetAnalogsNumber(step-1);
    if(analogsNb>analogsNbPrevious)
    {
        asLogError(wxString::Format(_("The given analog number (%d) is superior to the previous step (%d)."), analogsNb, analogsNbPrevious));
        return false;
    }

    // Matrices containers
    VpArray2DFloat vTargData = VpArray2DFloat(predictorsNb);
    VpArray2DFloat vArchData = VpArray2DFloat(predictorsNb);
    Array1DInt vRowsNb(predictorsNb);
    Array1DInt vColsNb(predictorsNb);

    for (int i_ptor=0; i_ptor<predictorsNb; i_ptor++)
    {
        vRowsNb[i_ptor] = predictorsArchive[i_ptor].GetData()[0].rows();
        vColsNb[i_ptor] = predictorsArchive[i_ptor].GetData()[0].cols();

        // Check criteria ordering
        if(isasc != (criteria[i_ptor]->GetOrder()==Asc))
        {
            asLogError(_("You cannot combine criteria that are ascendant and descendant."));
            return false;
        }
    }

    // Containers for daily results
    Array1DFloat currentAnalogsDates(analogsNbPrevious);

    // Containers for final results
    Array2DFloat finalAnalogsCriteria(timeTargetSelectionSize, analogsNb);
    Array2DFloat finalAnalogsDates(timeTargetSelectionSize, analogsNb);

    // Some other variables
    float tmpscore, thisscore;
    int i_timeArch, i_timeTarg, counter = 0;

	#if wxUSE_GUI
		// The progress bar
		wxString dialogmessage = _("Processing the data comparison.\n");
		asDialogProgressBar ProgressBar(dialogmessage, timeTargetSelectionSize);
	#endif

    switch (method)
    {
        case (asMULTITHREADS):
        {
            bool enableMessageBox = false;
            if (Log().IsMessageBoxOnErrorEnabled()) enableMessageBox = true;
            Log().DisableMessageBoxOnError();

			#if wxUSE_GUI
				if (g_Responsive) wxGetApp().Yield();
			#endif

            // Get threads number
            int threadsNb = ThreadsManager().GetAvailableThreadsNb();

            // Adapt to the number of targets
            if (2*threadsNb>timeTargetSelectionSize)
            {
                threadsNb = 1;
            }

            // Create and give data
            int start = 0, end = -1;
            int threadType = -1;
            std::vector < bool* > vContainsNaNs;
            for (int i_threads=0; i_threads<threadsNb; i_threads++)
            {
                bool* flag = new bool;
                *flag = false;
                vContainsNaNs.push_back(flag);
                start = end+1;
                end = ceil(((float)(i_threads+1)*(float)(timeTargetSelectionSize-1)/(float)threadsNb));
                wxASSERT_MSG(end>=start, wxString::Format("start = %d, end = %d, timeTargetSelectionSize = %d", start, end, timeTargetSelectionSize));

                asThreadProcessorGetAnalogsSubDates* thread = new asThreadProcessorGetAnalogsSubDates(&predictorsArchive, &predictorsTarget,
                                                                                                        &timeArrayArchiveData,
                                                                                                        &timeArrayTargetData,
                                                                                                        &timeTargetSelection,
                                                                                                        criteria, params, step,
                                                                                                        vTargData, vArchData, vRowsNb, vColsNb,
                                                                                                        start, end,
                                                                                                        &finalAnalogsCriteria, &finalAnalogsDates,
                                                                                                        &analogsDates, flag);
                threadType = thread->GetType();

                ThreadsManager().AddThread(thread);
            }

            // Wait until all done
            ThreadsManager().Wait(threadType);

            for (unsigned int i_threads=0; i_threads<vContainsNaNs.size()-1; i_threads++)
            {
                if(*vContainsNaNs[i_threads])
                {
                    containsNaNs = true;
                }
                wxDELETE(vContainsNaNs[i_threads]);
            }
            if (containsNaNs)
            {
                return true;
            }

            if (enableMessageBox) Log().EnableMessageBoxOnError();

            wxASSERT(finalAnalogsDates(0,0)>0);
            wxASSERT(finalAnalogsDates(1,1)>0);

            break;
        }

        case (asFULL_ARRAY): // Not implemented
        case (asINSERT):
        {
            // Containers for daily results
            Array1DFloat ScoreArrayOneDay(analogsNb);
            Array1DFloat DateArrayOneDay(analogsNb);

            // Loop through every timestep as target data
            for (int i_anadates=0; i_anadates<timeTargetSelectionSize; i_anadates++)
            {
				#if wxUSE_GUI
					if (g_Responsive) wxGetApp().Yield();

					// Update the progress bar
					wxString updatedialogmessage = dialogmessage + wxString::Format(_("Processing: %d / %d time steps"), i_anadates+1, timeTargetSelectionSize);
					if(!ProgressBar.Update(i_anadates, updatedialogmessage))
					{
						asLogMessage(_("The process has been canceled by the user."));
						return false;
					}
				#endif

                i_timeTarg = asTools::SortedArraySearch(&timeTargetData[0], &timeTargetData[timeTargetDataSize-1], timeTargetSelection[i_anadates], 0.01);
                wxASSERT_MSG(i_timeTarg>=0, wxString::Format(_("Looking for %s in betwwen %s and %s."), asTime::GetStringTime(timeTargetSelection[i_anadates], "DD.MM.YYYY hh:mm").c_str(),
                                                             asTime::GetStringTime(timeTargetData[0], "DD.MM.YYYY hh:mm").c_str(),
                                                             asTime::GetStringTime(timeTargetData[timeTargetDataSize-1], "DD.MM.YYYY hh:mm").c_str()));

                // Extract target data
                for (int i_ptor=0; i_ptor<predictorsNb; i_ptor++)
                {
                    vTargData[i_ptor] = &predictorsTarget[i_ptor].GetData()[i_timeTarg];
                }

                // Get dates
// TODO (phorton#1#): Check if the dates are really consistent between the steps !!
                currentAnalogsDates = analogsDates.row(i_anadates);

                // Counter representing the current index
                counter = 0;
                int NaNcounter = 0;

                // Loop through the previous analogs for candidate data
                for (int i_prevanalogs=0; i_prevanalogs<analogsNbPrevious; i_prevanalogs++)
                {
                    // Find row in the predictor time array
                    i_timeArch = asTools::SortedArraySearch(&timeArchiveData[0], &timeArchiveData[timeArchiveDataSize-1], currentAnalogsDates[i_prevanalogs], 0.01);

                    // Check if a row was found
                    if (i_timeArch!=asNOT_FOUND && i_timeArch!=asOUT_OF_RANGE)
                    {
                        // Process the criteria
                        thisscore = 0;
                        for (int i_ptor=0; i_ptor<predictorsNb; i_ptor++)
                        {
                            // Get data
                            vArchData[i_ptor] = &predictorsArchive[i_ptor].GetData()[i_timeArch];

                            // Assess the criteria
                            wxASSERT(criteria.size()>(unsigned)i_ptor);
                            wxASSERT(vTargData[i_ptor]);
                            wxASSERT(vArchData[i_ptor]);
                            wxASSERT(timeArchiveData.size()>i_timeArch);
                            wxASSERT_MSG(vArchData[i_ptor]->size()==vTargData[i_ptor]->size(), wxString::Format("%s (%d th element) in archive, %s (%d th element) in target: vArchData size = %d, vTargData size = %d",
                                                                                                                asTime::GetStringTime(timeArchiveData[i_timeArch], "DD.MM.YYYY hh:mm").c_str(), i_timeArch,
                                                                                                                asTime::GetStringTime(timeTargetData[i_timeTarg], "DD.MM.YYYY hh:mm").c_str(), i_timeTarg,
                                                                                                                (int)vArchData[i_ptor]->size(), (int)vTargData[i_ptor]->size()));
                            tmpscore = criteria[i_ptor]->Assess(*vTargData[i_ptor], *vArchData[i_ptor], vRowsNb[i_ptor], vColsNb[i_ptor]);

                            /*
                            // For debugging
                            wxLogMessage("timeTarget = %s",asTime::GetStringTime(timeTargetSelection[i_anadates]).c_str());
                            wxLogMessage("timeCandidate = %s",asTime::GetStringTime(currentAnalogsDates[i_prevanalogs]).c_str());
                            for (int i=0; i<vRowsNb[i_ptor]; i++ )
                            {
                                for (int j=0; j<vColsNb[i_ptor]; j++ )
                                {
                                    wxLogMessage("TargData(%d,%d) = %f", i, j, (*vTargData[i_ptor])(i,j));
                                    wxLogMessage("ArchData(%d,%d) = %f", i, j, (*vArchData[i_ptor])(i,j));
                                }
                            }
                            wxLogMessage("tmpscore=%f",tmpscore);
                            */

                            // Weight and add the score
                            thisscore += tmpscore * params.GetPredictorWeight(step, i_ptor);
                        }
                        if (asTools::IsNaN(thisscore))
                        {
                            NaNcounter++;
                            if (NaNcounter>=10)
                            {
                                containsNaNs = true;
                                return true;
                            }
                        }

                        // Check if the array is already full
                        if (counter>analogsNb-1)
                        {
                            if (isasc)
                            {
                                if (thisscore<ScoreArrayOneDay[analogsNb-1])
                                {
                                    asTools::SortedArraysInsert(&ScoreArrayOneDay[0], &ScoreArrayOneDay[analogsNb-1], &DateArrayOneDay[0], &DateArrayOneDay[analogsNb-1], Asc, thisscore, (float)timeArchiveData[i_timeArch]);
                                }
                            } else {
                                if (thisscore>ScoreArrayOneDay[analogsNb-1])
                                {
                                    asTools::SortedArraysInsert(&ScoreArrayOneDay[0], &ScoreArrayOneDay[analogsNb-1], &DateArrayOneDay[0], &DateArrayOneDay[analogsNb-1], Desc, thisscore, (float)timeArchiveData[i_timeArch]);
                                }
                            }
                        }
                        else if (counter<analogsNb-1)
                        {
                            // Add score and date to the vectors
                            ScoreArrayOneDay[counter] = thisscore;
                            DateArrayOneDay[counter] = (float)timeArchiveData[i_timeArch];
                        }
                        else if (counter==analogsNb-1)
                        {
                            // Add score and date to the vectors
                            ScoreArrayOneDay[counter] = thisscore;
                            DateArrayOneDay[counter] = (float)timeArchiveData[i_timeArch];

                            // Sort both scores and dates arrays
                            if (isasc)
                            {
                                asTools::SortArrays(&ScoreArrayOneDay[0], &ScoreArrayOneDay[analogsNb-1], &DateArrayOneDay[0], &DateArrayOneDay[analogsNb-1], Asc);
                            } else {
                                asTools::SortArrays(&ScoreArrayOneDay[0], &ScoreArrayOneDay[analogsNb-1], &DateArrayOneDay[0], &DateArrayOneDay[analogsNb-1], Desc);
                            }
                        }

                        counter++;
                    }
                    else
                    {
                        asLogError(_("The date was not found in the array (Analogs subdates fct). That should not happen."));
                        return false;
                    }
                }

                // Check that the number of occurences are larger than the desired analogs number. If not, set a warning
                if (counter>=analogsNb)
                {
                    // Copy results
                    finalAnalogsCriteria.row(i_anadates) = ScoreArrayOneDay.head(analogsNb).transpose();
                    finalAnalogsDates.row(i_anadates) = DateArrayOneDay.head(analogsNb).transpose();
                }
                else
                {
                    asLogWarning(_("There is not enough available data to satisfy the number of analogs"));
                    asLogWarning(wxString::Format(_("Analogs number (%d) > counter (%d)"), analogsNb, counter));
                }
            }

            break;
        }

        default:
            asThrowException(_("The processing method is not correctly defined."));
    }

	#if wxUSE_GUI
		ProgressBar.Destroy();
	#endif

    // Copy results to the resulting object
    results.SetTargetDates(timeTargetSelection);
    results.SetAnalogsCriteria(finalAnalogsCriteria);
    results.SetAnalogsDates(finalAnalogsDates);

    // Display the time the function took
    asLogMessage(wxString::Format(_("The function asProcessor::GetAnalogsSubDates took %ldms to execute"), sw.Time()));

    return true;
}

bool asProcessor::GetAnalogsValues(asDataPredictand &predictand,
                                   asResultsAnalogsDates &anaDates,
                                   asParameters &params,
                                   asResultsAnalogsValues &results)
{
    // Watch
    //wxStopWatch sw;

    // Extract Data
    Array1DFloat timeTargetSelection = anaDates.GetTargetDates();
    Array2DFloat analogsDates = anaDates.GetAnalogsDates();
    Array2DFloat analogsCriteria = anaDates.GetAnalogsCriteria();
    Array1DFloat predictandDataNorm = predictand.GetDataNormalizedStation(params.GetPredictandStationId());
    Array1DFloat predictandDataGross = predictand.GetDataGrossStation(params.GetPredictandStationId());
    Array1DDouble predictandTime = predictand.GetTime();
    int predictandTimeLength = predictand.GetTimeLength();
    int timeTargetSelectionLength = timeTargetSelection.size();
    int analogsNb = analogsDates.cols();

    wxASSERT(timeTargetSelectionLength>0);

    // Correct the time arrays to account for predictand time and not predictors time
    for (int i_time=0; i_time<timeTargetSelectionLength; i_time++)
    {
        timeTargetSelection[i_time] -= params.GetTimeShiftDays();

        for (int i_analog=0; i_analog<analogsNb; i_analog++)
        {
            analogsDates(i_time, i_analog) -= params.GetTimeShiftDays();
        }
    }

    // Get start and end dates
    double timeStart, timeEnd;
    timeStart = wxMax(predictandTime[0],asTime::GetMJD(params.GetArchiveYearStart(),1,1));
    timeEnd = wxMin(predictandTime[predictandTimeLength-1],asTime::GetMJD(params.GetArchiveYearEnd(),12,31));

    // Check if data are effectively available for this period
    int indexPredictandTimeStart = asTools::SortedArraySearchCeil(&predictandTime[0],&predictandTime[predictandTimeLength-1],timeStart);
    int indexPredictandTimeEnd = asTools::SortedArraySearchFloor(&predictandTime[0],&predictandTime[predictandTimeLength-1],timeEnd);
    while (asTools::IsNaN(predictandDataNorm(indexPredictandTimeStart)))
    {
        indexPredictandTimeStart++;
    }
    while (asTools::IsNaN(predictandDataNorm(indexPredictandTimeEnd)))
    {
        indexPredictandTimeEnd--;
    }
    timeStart = predictandTime[indexPredictandTimeStart];
    timeEnd = predictandTime[indexPredictandTimeEnd];
    if (timeEnd<=timeStart)
    {
        return false;
    }

    // Get start and end indices for the analogs dates
    double timeStartTarg = wxMax(timeStart, (double)timeTargetSelection[0]);
    double timeEndTarg = wxMin(timeEnd, (double)timeTargetSelection[timeTargetSelectionLength-1]);
    int indexTargDatesStart = asTools::SortedArraySearchCeil(&timeTargetSelection[0],&timeTargetSelection[timeTargetSelectionLength-1],timeStartTarg);
    int indexTargDatesEnd = asTools::SortedArraySearchFloor(&timeTargetSelection[0],&timeTargetSelection[timeTargetSelectionLength-1],timeEndTarg);
    int targTimeLength = 0;
    bool ignoreTargetValues = true;
    if( indexTargDatesStart==asNOT_FOUND || indexTargDatesStart==asOUT_OF_RANGE || indexTargDatesEnd==asNOT_FOUND || indexTargDatesEnd==asOUT_OF_RANGE)
    {
        // In case of real forecasting
        ignoreTargetValues = true;
        indexTargDatesStart = 0;
        indexTargDatesEnd = timeTargetSelectionLength-1;
        targTimeLength = timeTargetSelectionLength;
    }
    else
    {
        targTimeLength = indexTargDatesEnd-indexTargDatesStart+1;
        ignoreTargetValues = false;
    }

    // Some variables
    float currentAnalogDate, currentTargetDate;
    float predictandValueNorm, predictandValueGross;
    int predictandIndex;
    float predictandDTimeDays = (float)params.GetPredictandDTimeDays();

    // Resize containers
    wxASSERT(targTimeLength>0);
    wxASSERT(analogsNb>0);
    Array2DFloat finalAnalogValuesNorm(targTimeLength, analogsNb);
    Array2DFloat finalAnalogValuesGross(targTimeLength, analogsNb);
    Array2DFloat finalAnalogCriteria(targTimeLength, analogsNb);
    Array1DFloat finalTargetDates(targTimeLength);
    Array1DFloat finalTargetValuesNorm(targTimeLength);
    Array1DFloat finalTargetValuesGross(targTimeLength);

	#if wxUSE_GUI
		if (g_Responsive) wxGetApp().Yield();
	#endif

    // Get predictand values
    for (int i_targdate=indexTargDatesStart; i_targdate<=indexTargDatesEnd; i_targdate++)
    {
        wxASSERT(i_targdate>=0);
        int i_targdatenew = i_targdate-indexTargDatesStart;
        currentTargetDate = timeTargetSelection(i_targdate);
        finalTargetDates[i_targdatenew] = currentTargetDate;
        predictandIndex = asTools::SortedArraySearchClosest(&predictandTime[0],&predictandTime[predictandTimeLength-1],currentTargetDate+predictandDTimeDays);
        if( ignoreTargetValues | (predictandIndex==asOUT_OF_RANGE) | (predictandIndex==asNOT_FOUND) )
        {
            finalTargetValuesNorm(i_targdatenew) = NaNFloat;
            finalTargetValuesGross(i_targdatenew) = NaNFloat;
        }
        else
        {
            predictandValueNorm = predictandDataNorm(predictandIndex);
            predictandValueGross = predictandDataGross(predictandIndex);
            finalTargetValuesNorm(i_targdatenew) = predictandValueNorm;
            finalTargetValuesGross(i_targdatenew) = predictandValueGross;
        }

        for (int i_anadate=0; i_anadate<analogsNb; i_anadate++)
        {
            currentAnalogDate = analogsDates(i_targdate,i_anadate);

            if (!asTools::IsNaN(currentAnalogDate))
            {
                // Check that the date is in the range
                if ((currentAnalogDate>=timeStart) && (currentAnalogDate<=timeEnd))
                {
                    predictandIndex = asTools::SortedArraySearchClosest(&predictandTime[0],&predictandTime[predictandTime.size()-1],currentAnalogDate+predictandDTimeDays);
                    if( (predictandIndex==asOUT_OF_RANGE) | (predictandIndex==asNOT_FOUND))
                    {
                        wxString currDate = asTime::GetStringTime(currentAnalogDate+predictandDTimeDays);
                        wxString startDate = asTime::GetStringTime(predictandTime[0]);
                        wxString endDate = asTime::GetStringTime(predictandTime[predictandTime.size()-1]);
                        asLogWarning(wxString::Format(_("The current analog date (%s) was not found in the predictand time array (%s-%s)."), currDate.c_str(), startDate.c_str(), endDate.c_str()));
                        finalAnalogValuesNorm(i_targdatenew,i_anadate) = NaNFloat;
                        finalAnalogValuesGross(i_targdatenew,i_anadate) = NaNFloat;
                    }
                    else
                    {
                        predictandValueNorm = predictandDataNorm(predictandIndex);
                        predictandValueGross = predictandDataGross(predictandIndex);
                        wxASSERT(!asTools::IsNaN(predictandValueNorm));
                        wxASSERT(!asTools::IsNaN(predictandValueGross));
                        wxASSERT(predictandValueNorm<10000);
                        finalAnalogValuesNorm(i_targdatenew,i_anadate) = predictandValueNorm;
                        finalAnalogValuesGross(i_targdatenew,i_anadate) = predictandValueGross;
                    }
                } else {
                    asLogError(wxString::Format(_("The current analog date (%s) is outside of the allowed period (%s-%s))."), asTime::GetStringTime(currentAnalogDate, "DD.MM.YYYY").c_str(), asTime::GetStringTime(timeStart, "DD.MM.YYYY").c_str(), asTime::GetStringTime(timeEnd, "DD.MM.YYYY").c_str()));
                    finalAnalogValuesNorm(i_targdatenew,i_anadate) = NaNFloat;
                    finalAnalogValuesGross(i_targdatenew,i_anadate) = NaNFloat;
                }
                finalAnalogCriteria(i_targdatenew,i_anadate) = analogsCriteria(i_targdate,i_anadate);

            } else {
                asLogError(_("The current analog date is a NaN."));
                finalAnalogCriteria(i_targdatenew,i_anadate) = NaNFloat;
            }
        }

        wxASSERT(!asTools::HasNaN(&finalAnalogValuesNorm(i_targdatenew,0), &finalAnalogValuesNorm(i_targdatenew,analogsNb-1)));
        wxASSERT(!asTools::HasNaN(&finalAnalogValuesGross(i_targdatenew,0), &finalAnalogValuesGross(i_targdatenew,analogsNb-1)));
        wxASSERT(!asTools::HasNaN(&finalAnalogCriteria(i_targdatenew,0), &finalAnalogCriteria(i_targdatenew,analogsNb-1)));
    }

    //wxASSERT(!asTools::HasNaN(&finalTargetValuesNorm(0), &finalTargetValuesNorm(finalTargetValuesNorm.size()-1)));
    //wxASSERT(!asTools::HasNaN(&finalTargetValuesGross(0), &finalTargetValuesGross(finalTargetValuesGross.size()-1)));

    // Copy results to the resulting object
    results.SetAnalogsValuesNorm(finalAnalogValuesNorm);
    results.SetAnalogsValuesGross(finalAnalogValuesGross);
    results.SetAnalogsCriteria(finalAnalogCriteria);
    results.SetTargetDates(finalTargetDates);
    results.SetTargetValuesNorm(finalTargetValuesNorm);
    results.SetTargetValuesGross(finalTargetValuesGross);

    // Display the time the function took
    //asLogMessage(wxString::Format(_("The function asProcessor::GetAnalogsValues took %ldms to execute"), sw.Time()));

    return true;
}
