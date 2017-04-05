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

#include "asProcessor.h"

#include <asTimeArray.h>
#include <asParameters.h>
#include <asPreprocessor.h>
#include <asPredictorCriteria.h>
#include <asDataPredictorArchive.h>
#include <asResultsAnalogsDates.h>
#include <asResultsAnalogsValues.h>
#include <asThreadProcessorGetAnalogsDates.h>
#include <asThreadProcessorGetAnalogsSubDates.h>

#ifdef APP_FORECASTER
#include <AtmoswingAppForecaster.h>
#endif
#ifdef APP_OPTIMIZER
#include <AtmoswingAppOptimizer.h>
#endif
#ifdef USE_CUDA
#include <asProcessorCuda.cuh>
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

bool asProcessor::GetAnalogsDates(std::vector<asDataPredictor *> predictorsArchive,
                                  std::vector<asDataPredictor *> predictorsTarget, asTimeArray &timeArrayArchiveData,
                                  asTimeArray &timeArrayArchiveSelection, asTimeArray &timeArrayTargetData,
                                  asTimeArray &timeArrayTargetSelection, std::vector<asPredictorCriteria *> criteria,
                                  asParameters &params, int step, asResultsAnalogsDates &results, bool &containsNaNs)
{
    // Get the processing method
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    long defaultMethod = (long) asMULTITHREADS;
    int method = (int) (pConfig->Read("/Processing/Method", defaultMethod));
    bool allowMultithreading;
    pConfig->Read("/Processing/AllowMultithreading", &allowMultithreading, true);
    bool parallelEvaluations;
    pConfig->Read("/Optimizer/ParallelEvaluations", &parallelEvaluations, true);

    // Check options compatibility
    if (!allowMultithreading && method == asMULTITHREADS) {
        method = asINSERT;
    }

    // Check available threads
    if (method == asMULTITHREADS) {
        int threadsNb = ThreadsManager().GetAvailableThreadsNb();
        if (threadsNb < 2) {
            method = asINSERT;
        }
    }

    ThreadsManager().CritSectionConfig().Leave();

    // Check the step
    if (step > 0) {
        wxLogError(_("The Analogs SubDates method must be called for this step."));
        return false;
    }

    // Watch
    wxStopWatch sw;

    // Extract some data
    Array1DDouble timeTargetSelection = timeArrayTargetSelection.GetTimeArray();
    int timeTargetSelectionSize = (int) timeTargetSelection.size();
    wxASSERT(criteria[0]);
    bool isasc = (criteria[0]->GetOrder() == Asc);
    unsigned int predictorsNb = (unsigned int) params.GetPredictorsNb(step);
    unsigned int membersNb = (unsigned int) predictorsTarget[0]->GetData()[0].size();

    wxASSERT(predictorsArchive.size() > 0);
    wxASSERT_MSG((int) predictorsArchive.size() == predictorsNb,
                 wxString::Format("predictorsArchive.size() = %d, predictorsNb = %d", (int) predictorsArchive.size(),
                                  predictorsNb));

    // Check analogs number. Correct if superior to the time serie
    int analogsNb = params.GetAnalogsNumber(step);
    if (analogsNb > timeArrayArchiveSelection.GetSize()) {
        wxLogError(_("The given analog number is superior to the time serie."));
        return false;
    }

    // Matrices containers
    VpArray2DFloat vTargData = VpArray2DFloat(predictorsNb);
    VpArray2DFloat vArchData = VpArray2DFloat(predictorsNb);
    Array1DInt vRowsNb(predictorsNb);
    Array1DInt vColsNb(predictorsNb);

    for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
        wxASSERT((int) predictorsArchive.size() > iPtor);
        wxASSERT(predictorsArchive[iPtor]);
        wxASSERT(predictorsArchive[iPtor]->GetData().size() > 0);
        wxASSERT(vRowsNb.size() > iPtor);
        wxASSERT(vColsNb.size() > iPtor);
        vRowsNb[iPtor] = (int) predictorsArchive[iPtor]->GetData()[0][0].rows();
        vColsNb[iPtor] = (int) predictorsArchive[iPtor]->GetData()[0][0].cols();

        if (predictorsTarget[iPtor]->GetData()[0].size() != membersNb) {
            wxLogError(_("All variables must contain the same number of members."));
            return false;
        }

        // Check criteria ordering
        if (isasc != (criteria[iPtor]->GetOrder() == Asc)) {
            wxLogError(_("You cannot combine criteria that are ascendant and descendant."));
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

    switch (method) {

        case (asMULTITHREADS): {

            // Get threads number
            int threadsNb = ThreadsManager().GetAvailableThreadsNb();

            // Adapt to the number of targets
            if (2 * threadsNb > timeTargetSelectionSize) {
                threadsNb = 1;
            }

            // Create and give data
            int end = -1;
            int threadType = -1;
            std::vector<bool *> vContainsNaNs;
            for (int iThread = 0; iThread < threadsNb; iThread++) {
                bool *flag = new bool;
                *flag = false;
                vContainsNaNs.push_back(flag);
                int start = end + 1;
                end = ceil(((float) (iThread + 1) * (float) (timeTargetSelectionSize - 1) / (float) threadsNb));
                wxASSERT_MSG(end >= start,
                             wxString::Format("start = %d, end = %d, timeTargetSelectionSize = %d", start, end,
                                              timeTargetSelectionSize));

                asThreadProcessorGetAnalogsDates *thread = new asThreadProcessorGetAnalogsDates(predictorsArchive,
                                                                                                predictorsTarget,
                                                                                                &timeArrayArchiveData,
                                                                                                &timeArrayArchiveSelection,
                                                                                                &timeArrayTargetData,
                                                                                                &timeArrayTargetSelection,
                                                                                                criteria, params, step,
                                                                                                vTargData, vArchData,
                                                                                                vRowsNb, vColsNb, start,
                                                                                                end,
                                                                                                &finalAnalogsCriteria,
                                                                                                &finalAnalogsDates,
                                                                                                flag);
                threadType = thread->GetType();

                ThreadsManager().AddThread(thread);
            }

            // Wait until all done
            ThreadsManager().Wait(threadType);

            // Flush logs
            if (!parallelEvaluations)
                wxLog::FlushActive();

            for (unsigned int i = 0; i < vContainsNaNs.size(); i++) {
                if (*vContainsNaNs[i]) {
                    containsNaNs = true;
                }
                wxDELETE(vContainsNaNs[i]);
            }
            if (containsNaNs) {
                wxLogWarning(_("NaNs were found in the criteria values."));
            }

            break;
        }

#ifdef USE_CUDA
        case (asCUDA): // Based on the asFULL_ARRAY method
        {
            // Check criteria compatibility
            for (int iPtor=0; iPtor<predictorsNb; iPtor++)
            {
                if(criteria[iPtor]->GetType()!=criteria[0]->GetType())
                {
                    wxLogError(_("For CUDA implementation, every predictors in the same analogy level must share the same criterion."));
                    return false;
                }
            }

            switch(criteria[0]->GetType())
            {
                case (asPredictorCriteria::S1grads):
                    break;
                default:
                    wxLogError(_("The %s criteria is not yet implemented for CUDA."), criteria[0]->GetName());
                    return false;
            }

            // To minimize the data copy, we only allow 1 dataset
            if (predictorsArchive[0] != predictorsTarget[0])
            {
                wxLogError(_("The CUDA implementation is only available in calibration (prefect prog)."), criteria[0]->GetName());
                return false;
            }

            // Extract some data
            Array1DDouble timeArchiveData = timeArrayArchiveData.GetTimeArray();
            int timeArchiveDataSize = timeArchiveData.size();
            wxASSERT(timeArchiveDataSize==predictorsArchive[0]->GetData().size());
            Array1DDouble timeTargetData = timeArrayTargetData.GetTimeArray();
            int timeTargetDataSize = timeTargetData.size();
            wxASSERT(timeTargetDataSize==predictorsTarget[0]->GetData().size());

            // Storage for data pointers
            std::vector < float* > vpData(predictorsNb);
            std::vector < std::vector < float* > > vvpData(timeArchiveDataSize);

            // Copy predictor data
            for (int iTime=0; iTime<timeArchiveDataSize; iTime++)
            {
                for (int iPtor=0; iPtor<predictorsNb; iPtor++)
                {
                    vpData[iPtor] = predictorsArchive[iPtor]->GetData()[iTime].data();
                }
                vvpData[iTime] = vpData;
            }

            // DateArray object instantiation. There is one array for all the predictors, as they are aligned, so it picks the predictors we are interested in, but which didn't take place at the same time.
            asTimeArray dateArrayArchiveSelection(timeArrayArchiveSelection.GetStart(), timeArrayArchiveSelection.GetEnd(), params.GetTimeArrayAnalogsTimeStepHours(), params.GetTimeArrayAnalogsMode());
            if(timeArrayArchiveSelection.HasForbiddenYears())
            {
                dateArrayArchiveSelection.SetForbiddenYears(timeArrayArchiveSelection.GetForbiddenYears());
            }

            // Containers for the results
            std::vector < VectorFloat > resultingCriteria(timeTargetSelectionSize);
            std::vector < VectorFloat > resultingDates(timeTargetSelectionSize);
            //std::fill(resultingDates.begin(), resultingDates.end(), NaNFloat);

            // Containers for daily results
            Array1DFloat ScoreArrayOneDay(analogsNb);
            Array1DFloat DateArrayOneDay(analogsNb);

            // Containers for the indices
            VectorInt lengths(timeTargetSelectionSize);
            VectorInt indicesTarg(timeTargetSelectionSize);
            std::vector < VectorInt > indicesArch(timeTargetSelectionSize);

            // Constant data
            VectorFloat weights(predictorsNb);
            VectorInt colsNb(predictorsNb);
            VectorInt rowsNb(predictorsNb);

            for (int iPtor=0; iPtor<predictorsNb; iPtor++)
            {
                weights[iPtor] = params.GetPredictorWeight(step, iPtor);
                colsNb[iPtor] = vRowsNb[iPtor];
                rowsNb[iPtor] = vColsNb[iPtor];
            }

            // Some other variables
            int counter = 0;
            int iTimeTarg, iTimeArch, iTimeTargRelative, iTimeArchRelative;

            // Reset the index start target
            int iTimeTargStart = 0;

            /* First we find the dates */

            // Loop through every timestep as target data
            for (int iDateTarg=0; iDateTarg<timeTargetSelectionSize; iDateTarg++)
            {
                // Check if the next data is the following. If not, search for it in the array.
                if(timeTargetDataSize>iTimeTargStart+1 && std::abs(timeTargetSelection[iDateTarg]-timeTargetData[iTimeTargStart+1])<0.01)
                {
                    iTimeTargRelative = 1;
                } else {
                    iTimeTargRelative = asTools::SortedArraySearch(&timeTargetData[iTimeTargStart], &timeTargetData[timeTargetDataSize-1], timeTargetSelection[iDateTarg], 0.01);
                }

                // Check if a row was found
                if (iTimeTargRelative!=asNOT_FOUND && iTimeTargRelative!=asOUT_OF_RANGE)
                {
                    // Convert the relative index into an absolute index
                    iTimeTarg = iTimeTargRelative+iTimeTargStart;
                    iTimeTargStart = iTimeTarg;

                    // Keep the index
                    indicesTarg[iDateTarg] = iTimeTarg;

                    // DateArray initialization
                    dateArrayArchiveSelection.Init(timeTargetSelection[iDateTarg], params.GetTimeArrayAnalogsIntervalDays(), params.GetTimeArrayAnalogsExcludeDays());

                    // Counter representing the current index
                    counter = 0;

                    // Reset the index start target
                    int iTimeArchStart = 0;

                    // Get a new container for variable vectors
                    VectorInt currentIndices(dateArrayArchiveSelection.GetSize());
                    VectorFloat currentDates(dateArrayArchiveSelection.GetSize());

                    // Loop through the dateArrayArchiveSelection for candidate data
                    for (int iDateArch=0; iDateArch<dateArrayArchiveSelection.GetSize(); iDateArch++)
                    {
                        // Check if the next data is the following. If not, search for it in the array.
                        if(timeArchiveDataSize>iTimeArchStart+1 && std::abs(dateArrayArchiveSelection[iDateArch]-timeArchiveData[iTimeArchStart+1])<0.01)
                        {
                            iTimeArchRelative = 1;
                        } else {
                            iTimeArchRelative = asTools::SortedArraySearch(&timeArchiveData[iTimeArchStart], &timeArchiveData[timeArchiveDataSize-1], dateArrayArchiveSelection[iDateArch], 0.01);
                        }

                        // Check if a row was found
                        if (iTimeArchRelative!=asNOT_FOUND && iTimeArchRelative!=asOUT_OF_RANGE)
                        {
                            // Convert the relative index into an absolute index
                            iTimeArch = iTimeArchRelative+iTimeArchStart;
                            iTimeArchStart = iTimeArch;

                            // Store the index and the date
                            currentDates[counter] = (float)timeArchiveData[iTimeArch];
                            currentIndices[counter] = iTimeArch;
                            counter++;
                        }
                        else
                        {
                            wxLogError(_("The date was not found in the array (Analogs dates fct, CUDA option). That should not happen."));
                        }
                    }

                    // Keep the indices
                    lengths[iDateTarg] = counter;
                    indicesArch[iDateTarg] = currentIndices;
                    resultingDates[iDateTarg] = currentDates;
                }
            }

            /* Then we process on GPU */

            if(asProcessorCuda::ProcessCriteria(vvpData, indicesTarg, indicesArch, resultingCriteria, lengths, colsNb, rowsNb, weights))
            {
                /* If succeeded, we work on the outputs */

                for (int iDateTarg=0; iDateTarg<timeTargetSelectionSize; iDateTarg++)
                {
                    std::vector < float > vectCriteria = resultingCriteria[iDateTarg];
                    std::vector < float > vectDates = resultingDates[iDateTarg];

                    int vectCriteriaSize = vectCriteria.size();
                    int resCounter = 0;

                    for(int iDateArch=0; iDateArch<vectCriteriaSize; iDateArch++)
                    {
#ifdef _DEBUG
                            if (asTools::IsNaN(vectCriteria[iDateArch]))
                            {
                                containsNaNs = true;
                                wxLogWarning(_("NaNs were found in the criteria values."));
                                wxLogWarning(_("Target date: %s, archive date: %s."),asTime::GetStringTime(timeTargetSelection[iDateTarg]) , asTime::GetStringTime(DateArrayOneDay[iDateArch]));
                            }
#endif

                        // Check if the array is already full
                        if (resCounter>analogsNb-1)
                        {
                            if (isasc)
                            {
                                if (vectCriteria[iDateArch]<ScoreArrayOneDay[analogsNb-1])
                                {
                                    asTools::SortedArraysInsert(&ScoreArrayOneDay[0], &ScoreArrayOneDay[analogsNb-1], &DateArrayOneDay[0], &DateArrayOneDay[analogsNb-1], Asc, vectCriteria[iDateArch], vectDates[iDateArch]);
                                }
                            } else {
                                if (vectCriteria[iDateArch]>ScoreArrayOneDay[analogsNb-1])
                                {
                                    asTools::SortedArraysInsert(&ScoreArrayOneDay[0], &ScoreArrayOneDay[analogsNb-1], &DateArrayOneDay[0], &DateArrayOneDay[analogsNb-1], Desc, vectCriteria[iDateArch], vectDates[iDateArch]);
                                }
                            }
                        }
                        else if (resCounter<analogsNb-1)
                        {
                            // Add score and date to the vectors
                            ScoreArrayOneDay[resCounter] = vectCriteria[iDateArch];
                            DateArrayOneDay[resCounter] = vectDates[iDateArch];
                        }
                        else if (resCounter==analogsNb-1)
                        {
                            // Add score and date to the vectors
                            ScoreArrayOneDay[resCounter] = vectCriteria[iDateArch];
                            DateArrayOneDay[resCounter] = vectDates[iDateArch];

                            // Sort both scores and dates arrays
                            if (isasc)
                            {
                                asTools::SortArrays(&ScoreArrayOneDay[0], &ScoreArrayOneDay[analogsNb-1], &DateArrayOneDay[0], &DateArrayOneDay[analogsNb-1], Asc);
                            } else {
                                asTools::SortArrays(&ScoreArrayOneDay[0], &ScoreArrayOneDay[analogsNb-1], &DateArrayOneDay[0], &DateArrayOneDay[analogsNb-1], Desc);
                            }
                        }

                        resCounter++;
                    }

                    // Check that the number of occurences are larger than the desired analogs number. If not, set a warning
                    if (resCounter>=analogsNb)
                    {
                        // Copy results
                        finalAnalogsCriteria.row(iDateTarg) = ScoreArrayOneDay.head(analogsNb).transpose();
                        finalAnalogsDates.row(iDateTarg) = DateArrayOneDay.head(analogsNb).transpose();
                    }
                    else
                    {
                        wxLogWarning(_("There is not enough available data to satisfy the number of analogs"));
                        wxLogWarning(_("Analogs number (%d) > vectCriteriaSize (%d), date array size (%d) with %d days intervals."), analogsNb, vectCriteriaSize, dateArrayArchiveSelection.GetSize(), params.GetTimeArrayAnalogsIntervalDays());
                    }
                }

                // cudaDeviceReset must be called before exiting in order for profiling and
                // tracing tools such as Nsight and Visual Profiler to show complete traces.
                cudaDeviceReset();

                break;
            }

            /* Else we continue on asINSERT */
        }
#endif

        case (asINSERT): {
            // Extract some data
            Array1DDouble timeArchiveData = timeArrayArchiveData.GetTimeArray();
            int timeArchiveDataSize = timeArchiveData.size();
            wxASSERT(timeArchiveDataSize > 0);
            wxASSERT(predictorsArchive.size() > 0);
            wxASSERT(predictorsArchive[0]->GetData().size() > 0);
            wxASSERT_MSG(timeArchiveDataSize == (int) predictorsArchive[0]->GetData().size(),
                         wxString::Format("timeArchiveDataSize = %d, predictorsArchive[0].GetData().size() = %d",
                                          timeArchiveDataSize, (int) predictorsArchive[0]->GetData().size()));
            Array1DDouble timeTargetData = timeArrayTargetData.GetTimeArray();
            int timeTargetDataSize = timeTargetData.size();
            wxASSERT(predictorsTarget[0]);
            wxASSERT(timeTargetDataSize == (int) predictorsTarget[0]->GetData().size());

            // Containers for daily results
            Array1DFloat ScoreArrayOneDay(analogsNb);
            Array1DFloat DateArrayOneDay(analogsNb);

            // Some other variables
            float tmpscore, thisscore;
            int counter = 0;
            int iTimeTarg, iTimeArch, iTimeTargRelative, iTimeArchRelative;

            // DateArray object instantiation. There is one array for all the predictors, as they are aligned, so it picks the predictors we are interested in, but which didn't take place at the same time.
            asTimeArray dateArrayArchiveSelection(timeArrayArchiveSelection.GetStart(),
                                                  timeArrayArchiveSelection.GetEnd(),
                                                  params.GetTimeArrayAnalogsTimeStepHours(),
                                                  params.GetTimeArrayAnalogsMode());
            if (timeArrayArchiveSelection.HasForbiddenYears()) {
                dateArrayArchiveSelection.SetForbiddenYears(timeArrayArchiveSelection.GetForbiddenYears());
            }

            // Reset the index start target
            int iTimeTargStart = 0;

            // Loop through every timestep as target data
            for (int iDateTarg = 0; iDateTarg < timeTargetSelectionSize; iDateTarg++) {
                // Check if the next data is the following. If not, search for it in the array.
                if (timeTargetDataSize > iTimeTargStart + 1 &&
                    std::abs(timeTargetSelection[iDateTarg] - timeTargetData[iTimeTargStart + 1]) < 0.01) {
                    iTimeTargRelative = 1;
                } else {
                    iTimeTargRelative = asTools::SortedArraySearch(&timeTargetData[iTimeTargStart],
                                                                    &timeTargetData[timeTargetDataSize - 1],
                                                                    timeTargetSelection[iDateTarg], 0.01);
                }

                // Check if a row was found
                if (iTimeTargRelative != asNOT_FOUND && iTimeTargRelative != asOUT_OF_RANGE) {
                    // Convert the relative index into an absolute index
                    iTimeTarg = iTimeTargRelative + iTimeTargStart;
                    iTimeTargStart = iTimeTarg;

                    // DateArray object initialization.
                    dateArrayArchiveSelection.Init(timeTargetSelection[iDateTarg],
                                                   params.GetTimeArrayAnalogsIntervalDays(),
                                                   params.GetTimeArrayAnalogsExcludeDays());

                    // Counter representing the current index
                    counter = 0;

                    // Loop over the members
                    for (int iMem = 0; iMem < membersNb; ++iMem) {

                        // Extract target data
                        for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                            vTargData[iPtor] = &predictorsTarget[iPtor]->GetData()[iTimeTarg][iMem];
                        }

                        // Reset the index start target
                        int iTimeArchStart = 0;

                        // Loop through the datearray for candidate data
                        for (int iDateArch = 0; iDateArch < dateArrayArchiveSelection.GetSize(); iDateArch++) {
                            // Check if the next data is the following. If not, search for it in the array.
                            if (timeArchiveDataSize > iTimeArchStart + 1 &&
                                std::abs(dateArrayArchiveSelection[iDateArch] - timeArchiveData[iTimeArchStart + 1]) <
                                0.01) {
                                iTimeArchRelative = 1;
                            } else {
                                iTimeArchRelative = asTools::SortedArraySearch(&timeArchiveData[iTimeArchStart],
                                                                                &timeArchiveData[timeArchiveDataSize - 1],
                                                                                dateArrayArchiveSelection[iDateArch],
                                                                                0.01);
                            }

                            // Check if a row was found
                            if (iTimeArchRelative != asNOT_FOUND && iTimeArchRelative != asOUT_OF_RANGE) {
                                // Convert the relative index into an absolute index
                                iTimeArch = iTimeArchRelative + iTimeArchStart;
                                iTimeArchStart = iTimeArch;

                                // Process the criteria
                                thisscore = 0;
                                for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                                    // Get data
                                    vArchData[iPtor] = &predictorsArchive[iPtor]->GetData()[iTimeArch][iMem];

                                    // Assess the criteria
                                    wxASSERT(criteria.size() > (unsigned) iPtor);
                                    wxASSERT(vTargData[iPtor]);
                                    wxASSERT(vArchData[iPtor]);
                                    tmpscore = criteria[iPtor]->Assess(*vTargData[iPtor], *vArchData[iPtor],
                                                                        vRowsNb[iPtor], vColsNb[iPtor]);

                                    // Weight and add the score
                                    thisscore += tmpscore * params.GetPredictorWeight(step, iPtor);

                                }
                                if (asTools::IsNaN(thisscore)) {
                                    containsNaNs = true;
                                    wxLogWarning(_("NaNs were found in the criteria values."));
                                    wxLogWarning(_("Target date: %s, archive date: %s."),
                                                 asTime::GetStringTime(timeTargetSelection[iDateTarg]),
                                                 asTime::GetStringTime(dateArrayArchiveSelection[iDateArch]));
                                }

                                // Check if the array is already full
                                if (counter > analogsNb - 1) {
                                    if (isasc) {
                                        if (thisscore < ScoreArrayOneDay[analogsNb - 1]) {
                                            asTools::SortedArraysInsert(&ScoreArrayOneDay[0],
                                                                        &ScoreArrayOneDay[analogsNb - 1],
                                                                        &DateArrayOneDay[0],
                                                                        &DateArrayOneDay[analogsNb - 1], Asc, thisscore,
                                                                        (float) timeArchiveData[iTimeArch]);
                                        }
                                    } else {
                                        if (thisscore > ScoreArrayOneDay[analogsNb - 1]) {
                                            asTools::SortedArraysInsert(&ScoreArrayOneDay[0],
                                                                        &ScoreArrayOneDay[analogsNb - 1],
                                                                        &DateArrayOneDay[0],
                                                                        &DateArrayOneDay[analogsNb - 1], Desc,
                                                                        thisscore, (float) timeArchiveData[iTimeArch]);
                                        }
                                    }
                                } else if (counter < analogsNb - 1) {
                                    // Add score and date to the vectors
                                    ScoreArrayOneDay[counter] = thisscore;
                                    DateArrayOneDay[counter] = (float) timeArchiveData[iTimeArch];
                                } else if (counter == analogsNb - 1) {
                                    // Add score and date to the vectors
                                    ScoreArrayOneDay[counter] = thisscore;
                                    DateArrayOneDay[counter] = (float) timeArchiveData[iTimeArch];

                                    // Sort both scores and dates arrays
                                    if (isasc) {
                                        asTools::SortArrays(&ScoreArrayOneDay[0], &ScoreArrayOneDay[analogsNb - 1],
                                                            &DateArrayOneDay[0], &DateArrayOneDay[analogsNb - 1], Asc);
                                    } else {
                                        asTools::SortArrays(&ScoreArrayOneDay[0], &ScoreArrayOneDay[analogsNb - 1],
                                                            &DateArrayOneDay[0], &DateArrayOneDay[analogsNb - 1], Desc);
                                    }
                                }

                                counter++;
                            } else {
                                wxLogError(_("The candidate (%s) was not found in the array (%s - %s) (Target date: %s)."),
                                           asTime::GetStringTime(dateArrayArchiveSelection[iDateArch]),
                                           asTime::GetStringTime(timeArchiveData[iTimeArchStart]),
                                           asTime::GetStringTime(timeArchiveData[timeArchiveDataSize - 1]),
                                           asTime::GetStringTime(timeTargetSelection[iDateTarg]));
                            }
                        }
                    }

                    // Check that the number of occurences are larger than the desired analogs number. If not, set a warning
                    if (counter > analogsNb) {
                        // Copy results
                        finalAnalogsCriteria.row(iDateTarg) = ScoreArrayOneDay.transpose();
                        finalAnalogsDates.row(iDateTarg) = DateArrayOneDay.transpose();
                    } else {
                        wxLogWarning(_("There is not enough available data to satisfy the number of analogs."));
                    }
                }
            }

            break;
        }

        case (asFULL_ARRAY): {
            // Extract some data
            Array1DDouble timeArchiveData = timeArrayArchiveData.GetTimeArray();
            int timeArchiveDataSize = timeArchiveData.size();
            wxASSERT(timeArchiveDataSize == (int) predictorsArchive[0]->GetData().size());
            Array1DDouble timeTargetData = timeArrayTargetData.GetTimeArray();
            int timeTargetDataSize = timeTargetData.size();
            wxASSERT(timeTargetDataSize == (int) predictorsTarget[0]->GetData().size());

            // Containers for daily results
            Array1DFloat ScoreArrayOneDay(timeArrayArchiveSelection.GetSize());
            Array1DFloat DateArrayOneDay(timeArrayArchiveSelection.GetSize());

            // DateArray object instantiation. There is one array for all the predictors, as they are aligned, so it picks the predictors we are interested in, but which didn't take place at the same time.
            asTimeArray dateArrayArchiveSelection(timeArrayArchiveSelection.GetStart(),
                                                  timeArrayArchiveSelection.GetEnd(),
                                                  params.GetTimeArrayAnalogsTimeStepHours(),
                                                  params.GetTimeArrayAnalogsMode());
            if (timeArrayArchiveSelection.HasForbiddenYears()) {
                dateArrayArchiveSelection.SetForbiddenYears(timeArrayArchiveSelection.GetForbiddenYears());
            }

            // Reset the index start target
            int iTimeTargStart = 0;

            // Loop through every timestep as target data
            for (int iDateTarg = 0; iDateTarg < timeTargetSelectionSize; iDateTarg++) {
                int iTimeTargRelative;

                // Check if the next data is the following. If not, search for it in the array.
                if (timeTargetDataSize > iTimeTargStart + 1 &&
                    std::abs(timeTargetSelection[iDateTarg] - timeTargetData[iTimeTargStart + 1]) < 0.01) {
                    iTimeTargRelative = 1;
                } else {
                    iTimeTargRelative = asTools::SortedArraySearch(&timeTargetData[iTimeTargStart],
                                                                    &timeTargetData[timeTargetDataSize - 1],
                                                                    timeTargetSelection[iDateTarg], 0.01);
                }

                // Check if a row was found
                if (iTimeTargRelative != asNOT_FOUND && iTimeTargRelative != asOUT_OF_RANGE) {
                    // Convert the relative index into an absolute index
                    int iTimeTarg = iTimeTargRelative + iTimeTargStart;
                    iTimeTargStart = iTimeTarg;

                    // DateArray initialization
                    dateArrayArchiveSelection.Init(timeTargetSelection[iDateTarg],
                                                   params.GetTimeArrayAnalogsIntervalDays(),
                                                   params.GetTimeArrayAnalogsExcludeDays());

                    // Counter representing the current index
                    int counter = 0;

                    // Loop over the members
                    for (int iMem = 0; iMem < membersNb; ++iMem) {

                        // Extract target data
                        for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                            vTargData[iPtor] = &predictorsTarget[iPtor]->GetData()[iTimeTarg][iMem];
                        }

                        // Reset the index start target
                        int iTimeArchStart = 0;

                        // Loop through the dateArrayArchiveSelection for candidate data
                        for (int iDateArch = 0; iDateArch < dateArrayArchiveSelection.GetSize(); iDateArch++) {
                            int iTimeArchRelative;

                            // Check if the next data is the following. If not, search for it in the array.
                            if (timeArchiveDataSize > iTimeArchStart + 1 &&
                                std::abs(dateArrayArchiveSelection[iDateArch] - timeArchiveData[iTimeArchStart + 1]) <
                                0.01) {
                                iTimeArchRelative = 1;
                            } else {
                                iTimeArchRelative = asTools::SortedArraySearch(&timeArchiveData[iTimeArchStart],
                                                                                &timeArchiveData[timeArchiveDataSize - 1],
                                                                                dateArrayArchiveSelection[iDateArch],
                                                                                0.01);
                            }

                            // Check if a row was found
                            if (iTimeArchRelative != asNOT_FOUND && iTimeArchRelative != asOUT_OF_RANGE) {
                                // Convert the relative index into an absolute index
                                int iTimeArch = iTimeArchRelative + iTimeArchStart;
                                iTimeArchStart = iTimeArch;

                                // Process the criteria
                                float thisscore = 0;
                                for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                                    // Get data
                                    vArchData[iPtor] = &predictorsArchive[iPtor]->GetData()[iTimeArch][iMem];

                                    // Assess the criteria
                                    wxASSERT(criteria.size() > (unsigned) iPtor);
                                    float tmpscore = criteria[iPtor]->Assess(*vTargData[iPtor], *vArchData[iPtor],
                                                                              vRowsNb[iPtor], vColsNb[iPtor]);

                                    // Weight and add the score
                                    thisscore += tmpscore * params.GetPredictorWeight(step, iPtor);
                                }
                                if (asTools::IsNaN(thisscore)) {
                                    containsNaNs = true;
                                    wxLogWarning(_("NaNs were found in the criteria values."));
                                    wxLogWarning(_("Target date: %s, archive date: %s."),
                                                 asTime::GetStringTime(timeTargetSelection[iDateTarg]),
                                                 asTime::GetStringTime(dateArrayArchiveSelection[iDateArch]));
                                }

                                // Store in the result array
                                ScoreArrayOneDay[counter] = thisscore;
                                DateArrayOneDay[counter] = (float) timeArchiveData[iTimeArch];
                                counter++;
                            } else {
                                wxLogError(_("The date was not found in the array (Analogs dates fct, full array option). That should not happen."));
                            }
                        }
                    }

                    // Check that the number of occurences are larger than the desired analogs number. If not, set a warning
                    if (counter >= analogsNb) {
                        // Sort both scores and dates arrays
                        if (isasc) {
                            asTools::SortArrays(&ScoreArrayOneDay[0], &ScoreArrayOneDay[counter - 1],
                                                &DateArrayOneDay[0], &DateArrayOneDay[counter - 1], Asc);
                        } else {
                            asTools::SortArrays(&ScoreArrayOneDay[0], &ScoreArrayOneDay[counter - 1],
                                                &DateArrayOneDay[0], &DateArrayOneDay[counter - 1], Desc);
                        }

                        // Copy results
                        finalAnalogsCriteria.row(iDateTarg) = ScoreArrayOneDay.head(analogsNb).transpose();
                        finalAnalogsDates.row(iDateTarg) = DateArrayOneDay.head(analogsNb).transpose();
                    } else {
                        wxLogWarning(_("There is not enough available data to satisfy the number of analogs"));
                        wxLogWarning(_("Analogs number (%d) > counter (%d), date array size (%d) with %d days intervals."),
                                     analogsNb, counter, dateArrayArchiveSelection.GetSize(),
                                     params.GetTimeArrayAnalogsIntervalDays());
                    }
                }
            }

            break;
        }

        default:
            asThrowException(_("The processing method is not correctly defined."));
    }

    // Copy results to the resulting object
    results.SetTargetDates(timeTargetSelection);
    results.SetAnalogsCriteria(finalAnalogsCriteria);
    results.SetAnalogsDates(finalAnalogsDates);

    // Display the time the function took
    wxLogVerbose(_("The function asProcessor::GetAnalogsDates took %.3f s to execute"), float(sw.Time())/1000.0f);

    return true;
}

bool asProcessor::GetAnalogsSubDates(std::vector<asDataPredictor *> predictorsArchive,
                                     std::vector<asDataPredictor *> predictorsTarget, asTimeArray &timeArrayArchiveData,
                                     asTimeArray &timeArrayTargetData, asResultsAnalogsDates &anaDates,
                                     std::vector<asPredictorCriteria *> criteria, asParameters &params, int step,
                                     asResultsAnalogsDates &results, bool &containsNaNs)
{
    // Get the processing method
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    long defaultMethod = (long) asMULTITHREADS;
    int method = (int) (pConfig->Read("/Processing/Method", defaultMethod));
    bool allowMultithreading;
    pConfig->Read("/Processing/AllowMultithreading", &allowMultithreading, true);
    bool parallelEvaluations;
    pConfig->Read("/Optimizer/ParallelEvaluations", &parallelEvaluations, true);

    // Check options compatibility
    if (!allowMultithreading && method == asMULTITHREADS) {
        method = asINSERT;
    }

    // Check available threads
    if (method == asMULTITHREADS) {
        int threadsNb = ThreadsManager().GetAvailableThreadsNb();
        if (threadsNb < 2) {
            method = asINSERT;
        }
    }

    ThreadsManager().CritSectionConfig().Leave();

    // Check the step
    if (step == 0) {
        wxLogError(_("The  Analogs Dates method cannot be called first."));
        return false;
    }

    // Watch
    wxStopWatch sw;

    // Extract some data
    Array1DDouble timeArchiveData = timeArrayArchiveData.GetTimeArray();
    unsigned int timeArchiveDataSize = (unsigned int) timeArchiveData.size();
    wxASSERT(timeArchiveDataSize > 0);
    Array1DDouble timeTargetData = timeArrayTargetData.GetTimeArray();
    unsigned int timeTargetDataSize = (unsigned int) timeTargetData.size();
    wxASSERT(timeTargetDataSize > 0);
    Array1DFloat timeTargetSelection = anaDates.GetTargetDates();
    unsigned int timeTargetSelectionSize = (unsigned int) timeTargetSelection.size();
    wxASSERT(timeTargetSelectionSize > 0);
    Array2DFloat analogsDates = anaDates.GetAnalogsDates();
    bool isasc = (criteria[0]->GetOrder() == Asc);
    unsigned int predictorsNb = (unsigned int) params.GetPredictorsNb(step);
    wxASSERT(predictorsNb > 0);
    unsigned int membersNb = (unsigned int) predictorsTarget[0]->GetData()[0].size();

    // Check the analogs number. Correct if superior to the time serie
    int analogsNb = params.GetAnalogsNumber(step);
    int analogsNbPrevious = params.GetAnalogsNumber(step - 1);
    if (analogsNb > analogsNbPrevious) {
        wxLogError(_("The given analog number (%d) is superior to the previous step (%d)."), analogsNb,
                   analogsNbPrevious);
        return false;
    }

    // Matrices containers
    VpArray2DFloat vTargData = VpArray2DFloat(predictorsNb);
    VpArray2DFloat vArchData = VpArray2DFloat(predictorsNb);
    Array1DInt vRowsNb(predictorsNb);
    Array1DInt vColsNb(predictorsNb);

    for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
        vRowsNb[iPtor] = (int) predictorsArchive[iPtor]->GetData()[0][0].rows();
        vColsNb[iPtor] = (int) predictorsArchive[iPtor]->GetData()[0][0].cols();

        if (predictorsTarget[iPtor]->GetData()[0].size() != membersNb) {
            wxLogError(_("All variables must contain the same number of members."));
            return false;
        }

        // Check criteria ordering
        if (isasc != (criteria[iPtor]->GetOrder() == Asc)) {
            wxLogError(_("You cannot combine criteria that are ascendant and descendant."));
            return false;
        }
    }

    // Containers for daily results
    Array1DFloat currentAnalogsDates(analogsNbPrevious);

    // Containers for final results
    Array2DFloat finalAnalogsCriteria(timeTargetSelectionSize, analogsNb);
    Array2DFloat finalAnalogsDates(timeTargetSelectionSize, analogsNb);

#if wxUSE_GUI
    // The progress bar
    wxString dialogmessage = _("Processing the data comparison.\n");
    asDialogProgressBar ProgressBar(dialogmessage, timeTargetSelectionSize);
#endif

    switch (method) {
        case (asMULTITHREADS): {

            // Get threads number
            int threadsNb = ThreadsManager().GetAvailableThreadsNb();

            // Adapt to the number of targets
            if (2 * threadsNb > timeTargetSelectionSize) {
                threadsNb = 1;
            }

            // Create and give data
            int end = -1;
            int threadType = -1;
            std::vector<bool *> vContainsNaNs;
            for (int iThread = 0; iThread < threadsNb; iThread++) {
                bool *flag = new bool;
                *flag = false;
                vContainsNaNs.push_back(flag);
                int start = end + 1;
                end = ceil(((float) (iThread + 1) * (float) (timeTargetSelectionSize - 1) / (float) threadsNb));
                wxASSERT_MSG(end >= start,
                             wxString::Format("start = %d, end = %d, timeTargetSelectionSize = %d", start, end,
                                              timeTargetSelectionSize));

                asThreadProcessorGetAnalogsSubDates *thread = new asThreadProcessorGetAnalogsSubDates(predictorsArchive,
                                                                                                      predictorsTarget,
                                                                                                      &timeArrayArchiveData,
                                                                                                      &timeArrayTargetData,
                                                                                                      &timeTargetSelection,
                                                                                                      criteria, params,
                                                                                                      step, vTargData,
                                                                                                      vArchData,
                                                                                                      vRowsNb, vColsNb,
                                                                                                      start, end,
                                                                                                      &finalAnalogsCriteria,
                                                                                                      &finalAnalogsDates,
                                                                                                      &analogsDates,
                                                                                                      flag);
                threadType = thread->GetType();

                ThreadsManager().AddThread(thread);
            }

            // Wait until all done
            ThreadsManager().Wait(threadType);

            // Flush logs
            if (!parallelEvaluations)
                wxLog::FlushActive();

            for (unsigned int i = 0; i < vContainsNaNs.size(); i++) {
                if (*vContainsNaNs[i]) {
                    containsNaNs = true;
                }
                wxDELETE(vContainsNaNs[i]);
            }
            if (containsNaNs) {
                wxLogWarning(_("NaNs were found in the criteria values."));
            }

            wxASSERT(finalAnalogsDates(0, 0) > 0);
            wxASSERT(finalAnalogsDates(0, 1) > 0);

            break;
        }

        case (asFULL_ARRAY): // Not implemented
        case (asINSERT): {
            // Containers for daily results
            Array1DFloat ScoreArrayOneDay(analogsNb);
            Array1DFloat DateArrayOneDay(analogsNb);

            // Loop through every timestep as target data
            for (int iAnalogDate = 0; iAnalogDate < timeTargetSelectionSize; iAnalogDate++) {
                int iTimeTarg = asTools::SortedArraySearch(&timeTargetData[0], &timeTargetData[timeTargetDataSize - 1],
                                                            timeTargetSelection[iAnalogDate], 0.01);
                wxASSERT_MSG(iTimeTarg >= 0, wxString::Format(_("Looking for %s in betwwen %s and %s."),
                                                               asTime::GetStringTime(timeTargetSelection[iAnalogDate],
                                                                                     "DD.MM.YYYY hh:mm"),
                                                               asTime::GetStringTime(timeTargetData[0],
                                                                                     "DD.MM.YYYY hh:mm"),
                                                               asTime::GetStringTime(
                                                                       timeTargetData[timeTargetDataSize - 1],
                                                                       "DD.MM.YYYY hh:mm")));


                // Get dates
                currentAnalogsDates = analogsDates.row(iAnalogDate);

                // Counter representing the current index
                int counter = 0;

                // Loop over the members
                for (int iMem = 0; iMem < membersNb; ++iMem) {

                    // Extract target data
                    for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                        vTargData[iPtor] = &predictorsTarget[iPtor]->GetData()[iTimeTarg][iMem];
                    }

                    // Loop through the previous analogs for candidate data
                    for (int iPrevAnalog = 0; iPrevAnalog < analogsNbPrevious; iPrevAnalog++) {
                        // Find row in the predictor time array
                        int iTimeArch = asTools::SortedArraySearch(&timeArchiveData[0],
                                                                    &timeArchiveData[timeArchiveDataSize - 1],
                                                                    currentAnalogsDates[iPrevAnalog], 0.01);

                        // Check if a row was found
                        if (iTimeArch != asNOT_FOUND && iTimeArch != asOUT_OF_RANGE) {
                            // Process the criteria
                            float thisscore = 0;
                            for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                                // Get data
                                vArchData[iPtor] = &predictorsArchive[iPtor]->GetData()[iTimeArch][iMem];

                                // Assess the criteria
                                wxASSERT(criteria.size() > (unsigned) iPtor);
                                wxASSERT(vTargData[iPtor]);
                                wxASSERT(vArchData[iPtor]);
                                wxASSERT(timeArchiveData.size() > iTimeArch);
                                wxASSERT_MSG(vArchData[iPtor]->size() == vTargData[iPtor]->size(), wxString::Format(
                                        "%s (%d th element) in archive, %s (%d th element) in target: vArchData size = %d, vTargData size = %d",
                                        asTime::GetStringTime(timeArchiveData[iTimeArch], "DD.MM.YYYY hh:mm"),
                                        iTimeArch,
                                        asTime::GetStringTime(timeTargetData[iTimeTarg], "DD.MM.YYYY hh:mm"),
                                        iTimeTarg, (int) vArchData[iPtor]->size(), (int) vTargData[iPtor]->size()));
                                float tmpscore = criteria[iPtor]->Assess(*vTargData[iPtor], *vArchData[iPtor],
                                                                          vRowsNb[iPtor], vColsNb[iPtor]);

                                // Weight and add the score
                                thisscore += tmpscore * params.GetPredictorWeight(step, iPtor);
                            }
                            if (asTools::IsNaN(thisscore)) {
                                containsNaNs = true;
                                wxLogWarning(_("NaNs were found in the criteria values."));
                                wxLogWarning(_("Target date: %s, archive date: %s."),
                                             asTime::GetStringTime(timeTargetData[iTimeTarg]),
                                             asTime::GetStringTime(timeArchiveData[iTimeArch]));
                            }

                            // Check if the array is already full
                            if (counter > analogsNb - 1) {
                                if (isasc) {
                                    if (thisscore < ScoreArrayOneDay[analogsNb - 1]) {
                                        asTools::SortedArraysInsert(&ScoreArrayOneDay[0],
                                                                    &ScoreArrayOneDay[analogsNb - 1],
                                                                    &DateArrayOneDay[0],
                                                                    &DateArrayOneDay[analogsNb - 1], Asc, thisscore,
                                                                    (float) timeArchiveData[iTimeArch]);
                                    }
                                } else {
                                    if (thisscore > ScoreArrayOneDay[analogsNb - 1]) {
                                        asTools::SortedArraysInsert(&ScoreArrayOneDay[0],
                                                                    &ScoreArrayOneDay[analogsNb - 1],
                                                                    &DateArrayOneDay[0],
                                                                    &DateArrayOneDay[analogsNb - 1], Desc, thisscore,
                                                                    (float) timeArchiveData[iTimeArch]);
                                    }
                                }
                            } else if (counter < analogsNb - 1) {
                                // Add score and date to the vectors
                                ScoreArrayOneDay[counter] = thisscore;
                                DateArrayOneDay[counter] = (float) timeArchiveData[iTimeArch];
                            } else if (counter == analogsNb - 1) {
                                // Add score and date to the vectors
                                ScoreArrayOneDay[counter] = thisscore;
                                DateArrayOneDay[counter] = (float) timeArchiveData[iTimeArch];

                                // Sort both scores and dates arrays
                                if (isasc) {
                                    asTools::SortArrays(&ScoreArrayOneDay[0], &ScoreArrayOneDay[analogsNb - 1],
                                                        &DateArrayOneDay[0], &DateArrayOneDay[analogsNb - 1], Asc);
                                } else {
                                    asTools::SortArrays(&ScoreArrayOneDay[0], &ScoreArrayOneDay[analogsNb - 1],
                                                        &DateArrayOneDay[0], &DateArrayOneDay[analogsNb - 1], Desc);
                                }
                            }

                            counter++;
                        } else {
                            wxLogError(_("The date was not found in the array (Analogs subdates fct). That should not happen."));
                            return false;
                        }
                    }
                }

                // Check that the number of occurences are larger than the desired analogs number. If not, set a warning
                if (counter >= analogsNb) {
                    // Copy results
                    finalAnalogsCriteria.row(iAnalogDate) = ScoreArrayOneDay.head(analogsNb).transpose();
                    finalAnalogsDates.row(iAnalogDate) = DateArrayOneDay.head(analogsNb).transpose();
                } else {
                    wxLogWarning(_("There is not enough available data to satisfy the number of analogs"));
                    wxLogWarning(_("Analogs number (%d) > counter (%d)"), analogsNb, counter);
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
    wxASSERT(timeTargetSelection.size() > 0);
    wxASSERT(finalAnalogsCriteria.size() > 0);
    wxASSERT(finalAnalogsDates.size() > 0);
    results.SetTargetDates(timeTargetSelection);
    results.SetAnalogsCriteria(finalAnalogsCriteria);
    results.SetAnalogsDates(finalAnalogsDates);

    // Display the time the function took
    wxLogVerbose(_("The function asProcessor::GetAnalogsSubDates took %.3f s to execute"), float(sw.Time())/1000);

    return true;
}

bool asProcessor::GetAnalogsValues(asDataPredictand &predictand, asResultsAnalogsDates &anaDates, asParameters &params,
                                   asResultsAnalogsValues &results)
{
    // Extract Data
    Array1DFloat timeTargetSelection = anaDates.GetTargetDates();
    Array2DFloat analogsDates = anaDates.GetAnalogsDates();
    Array2DFloat analogsCriteria = anaDates.GetAnalogsCriteria();
    Array1DDouble predictandTime = predictand.GetTime();
    VectorInt stations = params.GetPredictandStationIds();
    int stationsNb = (int) stations.size();
    VArray1DFloat predictandDataNorm((unsigned long) stationsNb);
    VArray1DFloat predictandDataGross((unsigned long) stationsNb);
    for (int iStat = 0; iStat < stationsNb; iStat++) {
        predictandDataNorm[iStat] = predictand.GetDataNormalizedStation(stations[iStat]);
        predictandDataGross[iStat] = predictand.GetDataGrossStation(stations[iStat]);
    }

    int predictandTimeLength = predictand.GetTimeLength();
    int timeTargetSelectionLength = (int) timeTargetSelection.size();
    int analogsNb = (int) analogsDates.cols();

    wxASSERT(timeTargetSelectionLength > 0);

    // Correct the time arrays to account for predictand time and not predictors time
    for (int iTime = 0; iTime < timeTargetSelectionLength; iTime++) {
        timeTargetSelection[iTime] -= params.GetTimeShiftDays();

        for (int iAnalog = 0; iAnalog < analogsNb; iAnalog++) {
            analogsDates(iTime, iAnalog) -= params.GetTimeShiftDays();
        }
    }

    // Get start and end dates
    double timeStart, timeEnd;
    timeStart = wxMax(predictandTime[0], params.GetArchiveStart());
    timeEnd = wxMin(predictandTime[predictandTimeLength - 1], params.GetArchiveEnd());

    // Check if data are effectively available for this period
    int indexPredictandTimeStart = asTools::SortedArraySearchCeil(&predictandTime[0],
                                                                  &predictandTime[predictandTimeLength - 1], timeStart);
    int indexPredictandTimeEnd = asTools::SortedArraySearchFloor(&predictandTime[0],
                                                                 &predictandTime[predictandTimeLength - 1], timeEnd);
    for (int iStat = 0; iStat < (int) stations.size(); iStat++) {
        while (asTools::IsNaN(predictandDataNorm[iStat](indexPredictandTimeStart))) {
            indexPredictandTimeStart++;
        }
        while (asTools::IsNaN(predictandDataNorm[iStat](indexPredictandTimeEnd))) {
            indexPredictandTimeEnd--;
        }
    }
    timeStart = predictandTime[indexPredictandTimeStart];
    timeEnd = predictandTime[indexPredictandTimeEnd];
    if (timeEnd <= timeStart) {
        return false;
    }

    // Get start and end indices for the analogs dates
    double timeStartTarg = wxMax(timeStart, (double) timeTargetSelection[0]);
    double timeEndTarg = wxMin(timeEnd, (double) timeTargetSelection[timeTargetSelectionLength - 1]);
    int indexTargDatesStart = asTools::SortedArraySearchCeil(&timeTargetSelection[0],
                                                             &timeTargetSelection[timeTargetSelectionLength - 1],
                                                             timeStartTarg);
    int indexTargDatesEnd = asTools::SortedArraySearchFloor(&timeTargetSelection[0],
                                                            &timeTargetSelection[timeTargetSelectionLength - 1],
                                                            timeEndTarg);
    int targTimeLength = 0;
    bool ignoreTargetValues;
    if (indexTargDatesStart == asNOT_FOUND || indexTargDatesStart == asOUT_OF_RANGE ||
        indexTargDatesEnd == asNOT_FOUND || indexTargDatesEnd == asOUT_OF_RANGE) {
        // In case of real forecasting
        ignoreTargetValues = true;
        indexTargDatesStart = 0;
        indexTargDatesEnd = timeTargetSelectionLength - 1;
        targTimeLength = timeTargetSelectionLength;
    } else {
        targTimeLength = indexTargDatesEnd - indexTargDatesStart + 1;
        ignoreTargetValues = false;
    }

    // Some variables
    float predictandTimeDays = params.GetPredictandTimeHours() / 24.0;

    // Resize containers
    wxASSERT(targTimeLength > 0);
    wxASSERT(analogsNb > 0);
    VArray2DFloat finalAnalogValuesNorm(stationsNb, Array2DFloat(targTimeLength, analogsNb));
    VArray2DFloat finalAnalogValuesGross(stationsNb, Array2DFloat(targTimeLength, analogsNb));
    Array2DFloat finalAnalogCriteria(targTimeLength, analogsNb);
    Array1DFloat finalTargetDates(targTimeLength);
    VArray1DFloat finalTargetValuesNorm(stationsNb, Array1DFloat(targTimeLength));
    VArray1DFloat finalTargetValuesGross(stationsNb, Array1DFloat(targTimeLength));

    // Get predictand values
    for (int iTargetDate = indexTargDatesStart; iTargetDate <= indexTargDatesEnd; iTargetDate++) {
        wxASSERT(iTargetDate >= 0);
        int iTargetDatenew = iTargetDate - indexTargDatesStart;
        float currentTargetDate = timeTargetSelection(iTargetDate);
        finalTargetDates[iTargetDatenew] = currentTargetDate;
        int predictandIndex = asTools::SortedArraySearchClosest(&predictandTime[0],
                                                                &predictandTime[predictandTimeLength - 1],
                                                                currentTargetDate + predictandTimeDays,
                                                                asHIDE_WARNINGS);
        if (ignoreTargetValues | (predictandIndex == asOUT_OF_RANGE) | (predictandIndex == asNOT_FOUND)) {
            for (int iStat = 0; iStat < (int) stations.size(); iStat++) {
                finalTargetValuesNorm[iStat](iTargetDatenew) = NaNFloat;
                finalTargetValuesGross[iStat](iTargetDatenew) = NaNFloat;
            }
        } else {
            for (int iStat = 0; iStat < (int) stations.size(); iStat++) {
                finalTargetValuesNorm[iStat](iTargetDatenew) = predictandDataNorm[iStat](predictandIndex);
                finalTargetValuesGross[iStat](iTargetDatenew) = predictandDataGross[iStat](predictandIndex);
            }
        }

        for (int iAnalogDate = 0; iAnalogDate < analogsNb; iAnalogDate++) {
            float currentAnalogDate = analogsDates(iTargetDate, iAnalogDate);

            if (!asTools::IsNaN(currentAnalogDate)) {
                // Check that the date is in the range
                if ((currentAnalogDate >= timeStart) && (currentAnalogDate <= timeEnd)) {
                    predictandIndex = asTools::SortedArraySearchClosest(&predictandTime[0],
                                                                        &predictandTime[predictandTime.size() - 1],
                                                                        currentAnalogDate + predictandTimeDays);
                    if ((predictandIndex == asOUT_OF_RANGE) | (predictandIndex == asNOT_FOUND)) {
                        wxString currDate = asTime::GetStringTime(currentAnalogDate + predictandTimeDays);
                        wxString startDate = asTime::GetStringTime(predictandTime[0]);
                        wxString endDate = asTime::GetStringTime(predictandTime[predictandTime.size() - 1]);
                        wxLogWarning(_("The current analog date (%s) was not found in the predictand time array (%s-%s)."),
                                     currDate, startDate, endDate);
                        for (int iStat = 0; iStat < (int) stations.size(); iStat++) {
                            finalAnalogValuesNorm[iStat](iTargetDatenew, iAnalogDate) = NaNFloat;
                            finalAnalogValuesGross[iStat](iTargetDatenew, iAnalogDate) = NaNFloat;
                        }
                    } else {
                        for (int iStat = 0; iStat < (int) stations.size(); iStat++) {
                            wxASSERT(!asTools::IsNaN(predictandDataNorm[iStat](predictandIndex)));
                            wxASSERT(!asTools::IsNaN(predictandDataGross[iStat](predictandIndex)));
                            wxASSERT(predictandDataNorm[iStat](predictandIndex) < 10000);
                            finalAnalogValuesNorm[iStat](iTargetDatenew, iAnalogDate) = predictandDataNorm[iStat](
                                    predictandIndex);
                            finalAnalogValuesGross[iStat](iTargetDatenew, iAnalogDate) = predictandDataGross[iStat](
                                    predictandIndex);
                        }
                    }
                } else {
                    wxLogError(_("The current analog date (%s) is outside of the allowed period (%s-%s))."),
                               asTime::GetStringTime(currentAnalogDate, "DD.MM.YYYY"),
                               asTime::GetStringTime(timeStart, "DD.MM.YYYY"),
                               asTime::GetStringTime(timeEnd, "DD.MM.YYYY"));
                    for (int iStat = 0; iStat < (int) stations.size(); iStat++) {
                        finalAnalogValuesNorm[iStat](iTargetDatenew, iAnalogDate) = NaNFloat;
                        finalAnalogValuesGross[iStat](iTargetDatenew, iAnalogDate) = NaNFloat;
                    }
                }
                finalAnalogCriteria(iTargetDatenew, iAnalogDate) = analogsCriteria(iTargetDate, iAnalogDate);

            } else {
                wxLogError(_("The current analog date is a NaN."));
                finalAnalogCriteria(iTargetDatenew, iAnalogDate) = NaNFloat;
            }
        }

#ifndef UNIT_TESTING
#ifdef _DEBUG
        for (int iStat = 0; iStat < (int) stations.size(); iStat++) {
            wxASSERT(!asTools::HasNaN(&finalAnalogValuesNorm[iStat](iTargetDatenew, 0),
                                      &finalAnalogValuesNorm[iStat](iTargetDatenew, analogsNb - 1)));
            wxASSERT(!asTools::HasNaN(&finalAnalogValuesGross[iStat](iTargetDatenew, 0),
                                      &finalAnalogValuesGross[iStat](iTargetDatenew, analogsNb - 1)));
        }
        wxASSERT(!asTools::HasNaN(&finalAnalogCriteria(iTargetDatenew, 0),
                                  &finalAnalogCriteria(iTargetDatenew, analogsNb - 1)));
#endif
#endif
    }

    // Copy results to the resulting object
    results.SetAnalogsValuesNorm(finalAnalogValuesNorm);
    results.SetAnalogsValuesGross(finalAnalogValuesGross);
    results.SetAnalogsCriteria(finalAnalogCriteria);
    results.SetTargetDates(finalTargetDates);
    results.SetTargetValuesNorm(finalTargetValuesNorm);
    results.SetTargetValuesGross(finalTargetValuesGross);

    return true;
}
