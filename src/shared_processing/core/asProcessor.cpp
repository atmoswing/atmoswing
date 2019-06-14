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
#include <asCriteria.h>
#include <asPredictor.h>
#include <asResultsDates.h>
#include <asResultsValues.h>
#include <asThreadGetAnalogsDates.h>
#include <asThreadGetAnalogsSubDates.h>

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

bool asProcessor::GetAnalogsDates(std::vector<asPredictor *> predictorsArchive,
                                  std::vector<asPredictor *> predictorsTarget, asTimeArray &timeArrayArchiveData,
                                  asTimeArray &timeArrayArchiveSelection, asTimeArray &timeArrayTargetData,
                                  asTimeArray &timeArrayTargetSelection, std::vector<asCriteria *> criteria,
                                  asParameters *params, int step, asResultsDates &results, bool &containsNaNs)
{
    // Get the processing method
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    int method = pConfig->Read("/Processing/Method", (long) asMULTITHREADS);
    bool allowMultithreading = pConfig->ReadBool("/Processing/AllowMultithreading", true);
    bool parallelEvaluations = pConfig->ReadBool("/Processing/ParallelEvaluations", true);
    bool allowDuplicateDates = pConfig->ReadBool("/Processing/AllowDuplicateDates", true);

    // Check options compatibility
    if (!allowMultithreading && method == asMULTITHREADS) {
        method = asSTANDARD;
    }

    // Check available threads
    if (method == asMULTITHREADS) {
        int threadsNb = ThreadsManager().GetAvailableThreadsNb();
        if (threadsNb < 2) {
            method = asSTANDARD;
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
    a1d timeTargetSelection = timeArrayTargetSelection.GetTimeArray();
    int timeTargetSelectionSize = (int) timeTargetSelection.size();
    wxASSERT(criteria[0]);
    bool isAsc = (criteria[0]->GetOrder() == Asc);
    int predictorsNb = params->GetPredictorsNb(step);
    int membersNb = predictorsTarget[0]->GetData()[0].size();

    wxASSERT(!predictorsArchive.empty());
    wxASSERT((int) predictorsArchive.size() == predictorsNb);

    // Check analogs number. Correct if superior to the time serie
    int analogsNb = params->GetAnalogsNumber(step);
    if (analogsNb > timeArrayArchiveSelection.GetSize() * predictorsArchive[0]->GetMembersNb()) {
        wxLogError(_("The given analog number is superior to the time series."));
        return false;
    }

    // Matrices containers
    vpa2f vTargData = vpa2f(predictorsNb);
    vpa2f vArchData = vpa2f(predictorsNb);
    a1i vRowsNb(predictorsNb);
    a1i vColsNb(predictorsNb);

    for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
        wxASSERT((int) predictorsArchive.size() > iPtor);
        wxASSERT(predictorsArchive[iPtor]);
        wxASSERT(!predictorsArchive[iPtor]->GetData().empty());
        wxASSERT(vRowsNb.size() > iPtor);
        wxASSERT(vColsNb.size() > iPtor);
        vRowsNb[iPtor] = (int) predictorsArchive[iPtor]->GetData()[0][0].rows();
        vColsNb[iPtor] = (int) predictorsArchive[iPtor]->GetData()[0][0].cols();

        if (predictorsTarget[iPtor]->GetData()[0].size() != membersNb) {
            wxLogError(_("All variables must contain the same number of members."));
            return false;
        }

        // Check criteria ordering
        if (isAsc != (criteria[iPtor]->GetOrder() == Asc)) {
            wxLogError(_("You cannot combine criteria that are ascendant and descendant."));
            return false;
        }

        // Check for NaNs
        criteria[iPtor]->CheckNaNs(predictorsArchive[iPtor], predictorsTarget[iPtor]);
    }

    // Containers for final results
    a2f finalAnalogsCriteria(timeTargetSelectionSize, analogsNb);
    a2f finalAnalogsDates(timeTargetSelectionSize, analogsNb);

    // The progress bar
    wxString dialogmessage = _("Processing the data comparison.\n");
#if wxUSE_GUI
    asDialogProgressBar ProgressBar(dialogmessage, timeTargetSelectionSize);
#endif

    switch (method) {

#ifdef USE_CUDA
        case (asCUDA): {
            // Check criteria compatibility
            for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                if (!criteria[iPtor]->GetName().IsSameAs(criteria[0]->GetName())) {
                    wxLogError(_("For CUDA implementation, every predictors in the same analogy level must share the same criterion."));
                    return false;
                }
            }

            // Check no members
            for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                if (predictorsArchive[iPtor]->GetMembersNb() > 1) {
                    wxLogError(_("No support for ensemble datasets in CUDA yet."));
                    return false;
                }
            }

            if (criteria[0]->GetName().IsSameAs("S1grads")) {
                // ok
            } else {
                wxLogError(_("The %s criteria is not yet implemented for CUDA."), criteria[0]->GetName());
                return false;
            }

            // To minimize the data copy, we only allow 1 dataset
            if (predictorsArchive[0] != predictorsTarget[0]) {
                wxLogError(_("The CUDA implementation is only available in calibration (prefect prog)."),
                           criteria[0]->GetName());
                return false;
            }

            // Extract some data
            a1d timeArchiveData = timeArrayArchiveData.GetTimeArray();
            int timeArchiveDataSize = timeArchiveData.size();
            wxASSERT(timeArchiveDataSize == predictorsArchive[0]->GetData().size());
            a1d timeTargetData = timeArrayTargetData.GetTimeArray();
            int timeTargetDataSize = timeTargetData.size();
            wxASSERT(timeTargetDataSize == predictorsTarget[0]->GetData().size());

            // Storage for data pointers
            std::vector<float *> vpData(predictorsNb);
            std::vector<std::vector<float *> > vvpData(timeArchiveDataSize);

            // Copy predictor data
            for (int iTime = 0; iTime < timeArchiveDataSize; iTime++) {
                for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                    vpData[iPtor] = predictorsArchive[iPtor]->GetData()[iTime][0].data();
                }
                vvpData[iTime] = vpData;
            }

            // DateArray object instantiation. There is one array for all the predictors, as they are aligned, so it
            // picks the predictors we are interested in, but which didn't take place at the same time.
            asTimeArray dateArrayArchiveSelection(timeArrayArchiveSelection.GetStart(),
                                                  timeArrayArchiveSelection.GetEnd(), params->GetAnalogsTimeStepHours(),
                                                  params->GetTimeArrayAnalogsMode());
            if (timeArrayArchiveSelection.HasForbiddenYears()) {
                dateArrayArchiveSelection.SetForbiddenYears(timeArrayArchiveSelection.GetForbiddenYears());
            }

            // Containers for the results
            std::vector<vf> resultingCriteria(timeTargetSelectionSize);
            std::vector<vf> resultingDates(timeTargetSelectionSize);
            //std::fill(resultingDates.begin(), resultingDates.end(), NaNf);

            // Containers for daily results
            a1f scoreArrayOneDay(analogsNb);
            scoreArrayOneDay.fill(NaNf);
            a1f dateArrayOneDay(analogsNb);
            dateArrayOneDay.fill(NaNf);

            // Containers for the indices
            vi nbArchCandidates(timeTargetSelectionSize);
            vi indicesTarg(timeTargetSelectionSize);
            std::vector<vi> indicesArch(timeTargetSelectionSize);

            // Constant data
            vf weights(predictorsNb);
            vi colsNb(predictorsNb);
            vi rowsNb(predictorsNb);

            for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                weights[iPtor] = params->GetPredictorWeight(step, iPtor);
                colsNb[iPtor] = vColsNb[iPtor];
                rowsNb[iPtor] = vRowsNb[iPtor];
            }

            // Some other variables
            int counter = 0;
            int iTimeTarg, iTimeArch, iTimeTargRelative, iTimeArchRelative;

            // Reset the index start target
            int iTimeTargStart = 0;

            /* First we find the dates */

            // Loop through every timestep as target data
            for (int iDateTarg = 0; iDateTarg < timeTargetSelectionSize; iDateTarg++) {
                // Check if the next data is the following. If not, search for it in the array.
                if (timeTargetDataSize > iTimeTargStart + 1 &&
                    std::abs(timeTargetSelection[iDateTarg] - timeTargetData[iTimeTargStart + 1]) < 0.01) {
                    iTimeTargRelative = 1;
                } else {
                    iTimeTargRelative = asFind(&timeTargetData[iTimeTargStart],
                                               &timeTargetData[timeTargetDataSize - 1],
                                               timeTargetSelection[iDateTarg], 0.01);
                }

                // Check if a row was found
                if (iTimeTargRelative != asNOT_FOUND && iTimeTargRelative != asOUT_OF_RANGE) {
                    // Convert the relative index into an absolute index
                    iTimeTarg = iTimeTargRelative + iTimeTargStart;
                    iTimeTargStart = iTimeTarg;

                    // Keep the index
                    indicesTarg[iDateTarg] = iTimeTarg;

                    // DateArray initialization
                    dateArrayArchiveSelection.Init(timeTargetSelection[iDateTarg], params->GetAnalogsIntervalDays(),
                                                   params->GetAnalogsExcludeDays());

                    // Counter representing the current index
                    counter = 0;

                    // Reset the index start target
                    int iTimeArchStart = 0;

                    // Get a new container for variable vectors
                    vi currentIndices(dateArrayArchiveSelection.GetSize());
                    vf currentDates(dateArrayArchiveSelection.GetSize());

                    // Loop through the dateArrayArchiveSelection for candidate data
                    for (int iDateArch = 0; iDateArch < dateArrayArchiveSelection.GetSize(); iDateArch++) {
                        // Check if the next data is the following. If not, search for it in the array.
                        if (timeArchiveDataSize > iTimeArchStart + 1 &&
                                std::abs(dateArrayArchiveSelection[iDateArch] - timeArchiveData[iTimeArchStart + 1]) < 0.01) {
                            iTimeArchRelative = 1;
                        } else {
                            iTimeArchRelative = asFind(&timeArchiveData[iTimeArchStart],
                                                       &timeArchiveData[timeArchiveDataSize - 1],
                                                       dateArrayArchiveSelection[iDateArch], 0.01);
                        }

                        // Check if a row was found
                        if (iTimeArchRelative != asNOT_FOUND && iTimeArchRelative != asOUT_OF_RANGE) {
                            // Convert the relative index into an absolute index
                            iTimeArch = iTimeArchRelative + iTimeArchStart;
                            iTimeArchStart = iTimeArch;

                            // Store the index and the date
                            currentDates[counter] = (float) timeArchiveData[iTimeArch];
                            currentIndices[counter] = iTimeArch;
                            counter++;
                        } else {
                            wxLogError(_("The date was not found in the array (Analogs dates fct, CUDA option). That should not happen."));
                        }
                    }

                    // Keep the indices
                    nbArchCandidates[iDateTarg] = counter;
                    indicesArch[iDateTarg] = currentIndices;
                    resultingDates[iDateTarg] = currentDates;
                }
            }

            /* Then we process on GPU */

            if (asProcessorCuda::ProcessCriteria(vvpData, indicesTarg, indicesArch, resultingCriteria,
                                                 nbArchCandidates, colsNb, rowsNb, weights)) {
                /* If succeeded, we work on the outputs */

                for (int iDateTarg = 0; iDateTarg < timeTargetSelectionSize; iDateTarg++) {
                    std::vector<float> vectCriteria = resultingCriteria[iDateTarg];
                    std::vector<float> vectDates = resultingDates[iDateTarg];

                    int vectCriteriaSize = vectCriteria.size();
                    int resCounter = 0;

                    scoreArrayOneDay.fill(NaNf);
                    dateArrayOneDay.fill(NaNf);

                    for (int iDateArch = 0; iDateArch < vectCriteriaSize; iDateArch++) {
#ifdef _DEBUG
                        if (asIsNaN(vectCriteria[iDateArch]))
                        {
                            containsNaNs = true;
                            wxLogWarning(_("NaNs were found in the criteria values."));
                            wxLogWarning(_("Target date: %s, archive date: %s."),asTime::GetStringTime(timeTargetSelection[iDateTarg]) , asTime::GetStringTime(dateArrayOneDay[iDateArch]));
                        }
#endif

                        // Check if the array is already full
                        if (resCounter > analogsNb - 1) {
                            if (isAsc) {
                                if (vectCriteria[iDateArch] < scoreArrayOneDay[analogsNb - 1]) {
                                    asArraysInsert(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1],
                                                   &dateArrayOneDay[0], &dateArrayOneDay[analogsNb - 1],
                                                   Asc, vectCriteria[iDateArch], vectDates[iDateArch]);
                                }
                            } else {
                                if (vectCriteria[iDateArch] > scoreArrayOneDay[analogsNb - 1]) {
                                    asArraysInsert(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1],
                                                   &dateArrayOneDay[0], &dateArrayOneDay[analogsNb - 1],
                                                   Desc, vectCriteria[iDateArch], vectDates[iDateArch]);
                                }
                            }
                        } else if (resCounter < analogsNb - 1) {
                            // Add score and date to the vectors
                            scoreArrayOneDay[resCounter] = vectCriteria[iDateArch];
                            dateArrayOneDay[resCounter] = vectDates[iDateArch];
                        } else if (resCounter == analogsNb - 1) {
                            // Add score and date to the vectors
                            scoreArrayOneDay[resCounter] = vectCriteria[iDateArch];
                            dateArrayOneDay[resCounter] = vectDates[iDateArch];

                            // Sort both scores and dates arrays
                            if (isAsc) {
                                asSortArrays(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1],
                                             &dateArrayOneDay[0], &dateArrayOneDay[analogsNb - 1], Asc);
                            } else {
                                asSortArrays(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1],
                                             &dateArrayOneDay[0], &dateArrayOneDay[analogsNb - 1], Desc);
                            }
                        }

                        resCounter++;
                    }

                    if (resCounter < analogsNb) {
                        wxLogWarning(_("There is not enough available data to satisfy the number of analogs."));
                        wxLogWarning(_("Analogs number (%d) > vectCriteriaSize (%d), date array size (%d) with %d days intervals."),
                                     analogsNb, vectCriteriaSize, dateArrayArchiveSelection.GetSize(),
                                     params->GetAnalogsIntervalDays());
                    }

                    finalAnalogsCriteria.row(iDateTarg) = scoreArrayOneDay.head(analogsNb).transpose();
                    finalAnalogsDates.row(iDateTarg) = dateArrayOneDay.head(analogsNb).transpose();
                }

                // cudaDeviceReset must be called before exiting in order for profiling and
                // tracing tools such as Nsight and Visual Profiler to show complete traces.
                cudaDeviceReset();

                break;
            }

            /* Else we continue on asMULTITHREADS */
        }
#endif

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
            std::vector<bool *> vSuccess;
            for (int iThread = 0; iThread < threadsNb; iThread++) {
                bool *flag = new bool;
                *flag = false;
                vContainsNaNs.push_back(flag);
                bool *success = new bool;
                *success = false;
                vSuccess.push_back(success);
                int start = end + 1;
                end = ceil(((float) (iThread + 1) * (float) (timeTargetSelectionSize - 1) / (float) threadsNb));
                wxASSERT(end >= start);

                asThreadGetAnalogsDates *thread = new asThreadGetAnalogsDates(predictorsArchive, predictorsTarget,
                                                                              &timeArrayArchiveData,
                                                                              &timeArrayArchiveSelection,
                                                                              &timeArrayTargetData,
                                                                              &timeArrayTargetSelection, criteria,
                                                                              params, step, vTargData, vArchData,
                                                                              vRowsNb, vColsNb, start, end,
                                                                              &finalAnalogsCriteria, &finalAnalogsDates,
                                                                              flag, allowDuplicateDates, success);
                threadType = thread->GetType();

                ThreadsManager().AddThread(thread);
            }

            // Wait until all done
            ThreadsManager().Wait(threadType);

            // Flush logs
            if (!parallelEvaluations)
                wxLog::FlushActive();

            for (auto &containsNaN : vContainsNaNs) {
                if (*containsNaN) {
                    containsNaNs = true;
                }
                wxDELETE(containsNaN);
            }
            if (containsNaNs) {
                wxLogWarning(_("NaNs were found in the criteria values."));
            }

            bool failed = false;
            for (auto &success : vSuccess) {
                if (!*success) {
                    failed = true;
                }
                wxDELETE(success);
            }
            if (failed) {
                wxLogError(_("Errors were found during extraction of the analog dates."));
                return false;
            }

            break;
        }

        case (asSTANDARD): {
            // Extract some data
            a1d timeArchiveData = timeArrayArchiveData.GetTimeArray();
            wxASSERT(timeArchiveData.size() > 0);
            wxASSERT(!predictorsArchive.empty());
            wxASSERT(!predictorsArchive[0]->GetData().empty());
            wxASSERT(timeArchiveData.size() == predictorsArchive[0]->GetData().size());
            if (timeArchiveData.size() != predictorsArchive[0]->GetData().size()) {
                wxLogError(_("The size of the time array and the archive data are not equal (%d != %d)."),
                           (int) timeArchiveData.size(), (int) predictorsArchive[0]->GetData().size());
                wxLogError(_("Time array starts on %s and ends on %s."), asTime::GetStringTime(timeArchiveData[0], ISOdateTime),
                           asTime::GetStringTime(timeArchiveData[timeArchiveData.size() - 1], ISOdateTime));
                return false;
            }
            a1d timeTargetData = timeArrayTargetData.GetTimeArray();
            wxASSERT(predictorsTarget[0]);
            wxASSERT(timeTargetData.size() == predictorsTarget[0]->GetData().size());
            if (timeTargetData.size() != predictorsTarget[0]->GetData().size()) {
                wxLogError(_("The size of the time array and the target data are not equal (%d != %d)."),
                           (int) timeTargetData.size(), (int) predictorsTarget[0]->GetData().size());
                wxLogError(_("Time array starts on %s and ends on %s."), asTime::GetStringTime(timeTargetData[0], ISOdateTime),
                           asTime::GetStringTime(timeTargetData[timeTargetData.size() - 1], ISOdateTime));
                return false;
            }

            // Containers for daily results
            a1f scoreArrayOneDay(analogsNb);
            scoreArrayOneDay.fill(NaNf);
            a1f dateArrayOneDay(analogsNb);
            dateArrayOneDay.fill(NaNf);

            // DateArray object instantiation. There is one array for all the predictors, as they are aligned, so it picks the predictors we are interested in, but which didn't take place at the same time.
            asTimeArray dateArrayArchiveSelection(timeArrayArchiveSelection.GetStart(),
                                                  timeArrayArchiveSelection.GetEnd(), params->GetAnalogsTimeStepHours(),
                                                  params->GetTimeArrayAnalogsMode());
            if (timeArrayArchiveSelection.HasForbiddenYears()) {
                dateArrayArchiveSelection.SetForbiddenYears(timeArrayArchiveSelection.GetForbiddenYears());
            }

            // Reset the index start target
            int iTimeTargStart = 0;

            // Loop through every timestep as target data
            for (int iDateTarg = 0; iDateTarg < timeTargetSelectionSize; iDateTarg++) {

                int iTimeTargRelative = FindNextDate(timeTargetSelection, timeTargetData, iTimeTargStart, iDateTarg);

                // Check if a row was found
                if (iTimeTargRelative == asNOT_FOUND || iTimeTargRelative == asOUT_OF_RANGE) {
                    continue;
                }

                // Convert the relative index into an absolute index
                int iTimeTarg = iTimeTargRelative + iTimeTargStart;
                iTimeTargStart = iTimeTarg;

                // DateArray object initialization.
                dateArrayArchiveSelection.Init(timeTargetSelection[iDateTarg], params->GetAnalogsIntervalDays(),
                                               params->GetAnalogsExcludeDays());

                // Counter representing the current index
                int counter = 0;

                scoreArrayOneDay.fill(NaNf);
                dateArrayOneDay.fill(NaNf);

                // Loop over the members
                for (int iMem = 0; iMem < membersNb; ++iMem) {

                    // Extract target data
                    for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                        vTargData[iPtor] = &predictorsTarget[iPtor]->GetData()[iTimeTarg][iMem];
                    }

                    // Reset the index start target
                    int iTimeArchStart = 0;

                    // Loop through the date array for candidate data
                    for (int iDateArch = 0; iDateArch < dateArrayArchiveSelection.GetSize(); iDateArch++) {

                        int iTimeArchRelative = FindNextDate(dateArrayArchiveSelection, timeArchiveData,
                                                             iTimeArchStart, iDateArch);

                        // Check if a row was found
                        if (iTimeArchRelative == asNOT_FOUND || iTimeArchRelative == asOUT_OF_RANGE) {
                            wxLogError(_("The candidate (%s) was not found in the array (%s - %s) (Target date: %s)."),
                                       asTime::GetStringTime(dateArrayArchiveSelection[iDateArch]),
                                       asTime::GetStringTime(timeArchiveData[iTimeArchStart]),
                                       asTime::GetStringTime(timeArchiveData[timeArchiveData.size() - 1]),
                                       asTime::GetStringTime(timeTargetSelection[iDateTarg]));
                            continue;
                        }

                        // Convert the relative index into an absolute index
                        int iTimeArch = iTimeArchRelative + iTimeArchStart;
                        iTimeArchStart = iTimeArch;

                        // Process the criteria
                        float thisScore = 0;
                        for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                            // Get data
                            vArchData[iPtor] = &predictorsArchive[iPtor]->GetData()[iTimeArch][iMem];

                            // Assess the criteria
                            wxASSERT(criteria.size() > iPtor);
                            wxASSERT(vTargData[iPtor]);
                            wxASSERT(vArchData[iPtor]);
                            float tmpScore = criteria[iPtor]->Assess(*vTargData[iPtor], *vArchData[iPtor],
                                                                     vRowsNb[iPtor], vColsNb[iPtor]);

                            // Weight and add the score
                            thisScore += tmpScore * params->GetPredictorWeight(step, iPtor);

                            if (asIsNaN(tmpScore)) {
                                containsNaNs = true;
                                wxLogWarning(_("NaNs were found in the criteria values (%s/%s)."),
                                             predictorsArchive[iPtor]->GetProduct(),
                                             predictorsArchive[iPtor]->GetDataId());
                                wxLogWarning(_("Target date: %s, archive date: %s."),
                                             asTime::GetStringTime(timeTargetSelection[iDateTarg]),
                                             asTime::GetStringTime(dateArrayArchiveSelection[iDateArch]));
                            }
                        }

                        // Avoid duplicate analog dates
                        if (!allowDuplicateDates && iMem > 0) {
                            if (counter <= analogsNb - 1) {
                                wxFAIL;
                                wxLogError(_("It should not happen that the array of analogue dates is not full when adding members."));
                                return false;
                            }
                            InsertInArraysNoDuplicate(isAsc, analogsNb, (float) timeArchiveData[iTimeArch],
                                                      thisScore, scoreArrayOneDay, dateArrayOneDay);
                        } else {
                            InsertInArrays(isAsc, analogsNb, (float) timeArchiveData[iTimeArch], thisScore,
                                           counter, scoreArrayOneDay, dateArrayOneDay);
                        }

                        counter++;
                    }
                }

                if (counter < analogsNb) {
                    wxLogWarning(_("There is not enough available data to satisfy the number of analogs (in asProcessor::GetAnalogsDates)."));
                }

                // Copy results
                finalAnalogsCriteria.row(iDateTarg) = scoreArrayOneDay.transpose();
                finalAnalogsDates.row(iDateTarg) = dateArrayOneDay.transpose();

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
    wxLogVerbose(_("The function asProcessor::GetAnalogsDates took %.3f s to execute"), float(sw.Time()) / 1000.0f);

    return true;
}

int asProcessor::FindNextDate(asTimeArray &dateArray, a1d &timeData, int iTimeStart, int iDate)
{
    // Check if the next data is the following. If not, search for it in the array.
    if (timeData.size() > iTimeStart + 1 && std::abs(dateArray[iDate] - timeData[iTimeStart + 1]) < 0.01) {
        return 1;
    }

    return asFind(&timeData[iTimeStart], &timeData[timeData.size() - 1], dateArray[iDate], 0.01);
}

int asProcessor::FindNextDate(a1d &dateArray, a1d &timeData, int iTimeStart, int iDate)
{
    // Check if the next data is the following. If not, search for it in the array.
    if (timeData.size() > iTimeStart + 1 && std::abs(dateArray[iDate] - timeData[iTimeStart + 1]) < 0.01) {
        return 1;
    }

    return asFind(&timeData[iTimeStart], &timeData[timeData.size() - 1], dateArray[iDate], 0.01);
}

void asProcessor::InsertInArrays(bool isAsc, int analogsNb, float analogDate, float score, int counter,
                                 a1f &scoreArrayOneDay, a1f &dateArrayOneDay)
{
    // Check if the array is already full
    if (counter > analogsNb - 1) {
        if (isAsc) {
            if (score < scoreArrayOneDay[analogsNb - 1]) {
                asArraysInsert(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1], &dateArrayOneDay[0],
                               &dateArrayOneDay[analogsNb - 1], Asc, score, analogDate);
            }
        } else {
            if (score > scoreArrayOneDay[analogsNb - 1]) {
                asArraysInsert(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1], &dateArrayOneDay[0],
                               &dateArrayOneDay[analogsNb - 1], Desc, score, analogDate);
            }
        }
    } else if (counter < analogsNb - 1) {
        // Add score and date to the vectors
        scoreArrayOneDay[counter] = score;
        dateArrayOneDay[counter] = analogDate;
    } else if (counter == analogsNb - 1) {
        // Add score and date to the vectors
        scoreArrayOneDay[counter] = score;
        dateArrayOneDay[counter] = analogDate;

        // Sort both scores and dates arrays
        if (isAsc) {
            asSortArrays(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1], &dateArrayOneDay[0],
                                &dateArrayOneDay[analogsNb - 1], Asc);
        } else {
            asSortArrays(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1], &dateArrayOneDay[0],
                                &dateArrayOneDay[analogsNb - 1], Desc);
        }
    }
}

void asProcessor::InsertInArraysNoDuplicate(bool isAsc, int analogsNb, float analogDate, float score,
                                            a1f &scoreArrayOneDay, a1f &dateArrayOneDay)
{

    if (isAsc) {
        if (score >= scoreArrayOneDay[analogsNb - 1]) {
            return;
        }

        // Look for duplicate analogue date
        for (int i = 0; i < analogsNb; ++i) {
            if (dateArrayOneDay[i] == analogDate) {
                if (score < scoreArrayOneDay[i]) {
                    dateArrayOneDay[i] = analogDate;
                    scoreArrayOneDay[i] = score;
                    asSortArrays(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1], &dateArrayOneDay[0],
                                 &dateArrayOneDay[analogsNb - 1], Asc);
                }
                return;
            }
        }

        asArraysInsert(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1], &dateArrayOneDay[0],
                       &dateArrayOneDay[analogsNb - 1], Asc, score, analogDate);

    } else {
        if (score <= scoreArrayOneDay[analogsNb - 1]) {
            return;
        }

        // Look for duplicate analogue date
        for (int i = 0; i < analogsNb; ++i) {
            if (dateArrayOneDay[i] == analogDate) {
                if (score > scoreArrayOneDay[i]) {
                    dateArrayOneDay[i] = analogDate;
                    scoreArrayOneDay[i] = score;
                    asSortArrays(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1], &dateArrayOneDay[0],
                                 &dateArrayOneDay[analogsNb - 1], Desc);
                }
                return;
            }
        }

        asArraysInsert(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1], &dateArrayOneDay[0],
                       &dateArrayOneDay[analogsNb - 1], Desc, score, analogDate);

    }

}

bool asProcessor::GetAnalogsSubDates(std::vector<asPredictor *> predictorsArchive,
                                     std::vector<asPredictor *> predictorsTarget, asTimeArray &timeArrayArchiveData,
                                     asTimeArray &timeArrayTargetData, asResultsDates &anaDates,
                                     std::vector<asCriteria *> criteria, asParameters *params, int step,
                                     asResultsDates &results, bool &containsNaNs)
{
    // Get the processing method
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    int method = pConfig->Read("/Processing/Method", (long) asMULTITHREADS);
    bool allowMultithreading = pConfig->ReadBool("/Processing/AllowMultithreading", true);
    bool parallelEvaluations = pConfig->ReadBool("/Processing/ParallelEvaluations", true);

    // Check options compatibility
    if (!allowMultithreading && method == asMULTITHREADS) {
        method = asSTANDARD;
    }

    // Check available threads
    if (method == asMULTITHREADS) {
        int threadsNb = ThreadsManager().GetAvailableThreadsNb();
        if (threadsNb < 2) {
            method = asSTANDARD;
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
    a1d timeArchiveData = timeArrayArchiveData.GetTimeArray();
    auto timeArchiveDataSize = timeArchiveData.size();
    wxASSERT(timeArchiveDataSize > 0);
    a1d timeTargetData = timeArrayTargetData.GetTimeArray();
    auto timeTargetDataSize = timeTargetData.size();
    wxASSERT(timeTargetDataSize > 0);
    a1f timeTargetSelection = anaDates.GetTargetDates();
    auto timeTargetSelectionSize = timeTargetSelection.size();
    wxASSERT(timeTargetSelectionSize > 0);
    a2f analogsDates = anaDates.GetAnalogsDates();
    bool isasc = (criteria[0]->GetOrder() == Asc);
    auto predictorsNb = params->GetPredictorsNb(step);
    wxASSERT(predictorsNb > 0);
    auto membersNb = predictorsTarget[0]->GetData()[0].size();

    // Check the analogs number. Correct if superior to the time serie
    int analogsNb = params->GetAnalogsNumber(step);
    int analogsNbPrevious = params->GetAnalogsNumber(step - 1);
    if (analogsNb > analogsNbPrevious) {
        wxLogError(_("The given analog number (%d) is superior to the previous step (%d)."), analogsNb,
                   analogsNbPrevious);
        return false;
    }

    // Matrices containers
    vpa2f vTargData = vpa2f(predictorsNb);
    vpa2f vArchData = vpa2f(predictorsNb);
    a1i vRowsNb(predictorsNb);
    a1i vColsNb(predictorsNb);

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

        // Check for NaNs
        criteria[iPtor]->CheckNaNs(predictorsArchive[iPtor], predictorsTarget[iPtor]);
    }

    // Containers for daily results
    a1f currentAnalogsDates(analogsNbPrevious);

    // Containers for final results
    a2f finalAnalogsCriteria(timeTargetSelectionSize, analogsNb);
    a2f finalAnalogsDates(timeTargetSelectionSize, analogsNb);

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
            std::vector<bool *> vSuccess;
            for (int iThread = 0; iThread < threadsNb; iThread++) {
                bool *flag = new bool;
                *flag = false;
                vContainsNaNs.push_back(flag);
                bool *success = new bool;
                *success = false;
                vSuccess.push_back(success);
                int start = end + 1;
                end = ceil(((float) (iThread + 1) * (float) (timeTargetSelectionSize - 1) / (float) threadsNb));
                wxASSERT_MSG(end >= start,
                             wxString::Format("start = %d, end = %d, timeTargetSelectionSize = %d", start, end,
                                              timeTargetSelectionSize));

                asThreadGetAnalogsSubDates *thread = new asThreadGetAnalogsSubDates(predictorsArchive, predictorsTarget,
                                                                                    &timeArrayArchiveData,
                                                                                    &timeArrayTargetData,
                                                                                    &timeTargetSelection, criteria,
                                                                                    params, step, vTargData, vArchData,
                                                                                    vRowsNb, vColsNb, start, end,
                                                                                    &finalAnalogsCriteria,
                                                                                    &finalAnalogsDates, &analogsDates,
                                                                                    flag, success);
                threadType = thread->GetType();

                ThreadsManager().AddThread(thread);
            }

            // Wait until all done
            ThreadsManager().Wait(threadType);

            // Flush logs
            if (!parallelEvaluations)
                wxLog::FlushActive();

            for (auto &containsNaN : vContainsNaNs) {
                if (*containsNaN) {
                    containsNaNs = true;
                }
                wxDELETE(containsNaN);
            }
            if (containsNaNs) {
                wxLogWarning(_("NaNs were found in the criteria values."));
            }

            bool failed = false;
            for (auto &success : vSuccess) {
                if (!*success) {
                    failed = true;
                }
                wxDELETE(success);
            }
            if (failed) {
                wxFAIL;
                wxLogError(_("Errors were found during extraction of the analog dates."));
            }

            wxASSERT(finalAnalogsDates(0, 0) > 0);
            wxASSERT(finalAnalogsDates(0, 1) > 0);

            break;
        }

        case (asSTANDARD): {
            // Containers for daily results
            a1f scoreArrayOneDay(analogsNb);
            scoreArrayOneDay.fill(NaNf);
            a1f dateArrayOneDay(analogsNb);
            dateArrayOneDay.fill(NaNf);

            // Loop through every timestep as target data
            for (int iAnalogDate = 0; iAnalogDate < timeTargetSelectionSize; iAnalogDate++) {
                int iTimeTarg = asFind(&timeTargetData[0], &timeTargetData[timeTargetDataSize - 1],
                                       timeTargetSelection[iAnalogDate], 0.01);
                wxASSERT_MSG(iTimeTarg >= 0, wxString::Format(_("Looking for %s in betwwen %s and %s."),
                                                              asTime::GetStringTime(timeTargetSelection[iAnalogDate],
                                                                                    "DD.MM.YYYY hh:mm"),
                                                              asTime::GetStringTime(timeTargetData[0],
                                                                                    "DD.MM.YYYY hh:mm"),
                                                              asTime::GetStringTime(
                                                                      timeTargetData[timeTargetDataSize - 1],
                                                                      "DD.MM.YYYY hh:mm")));

                if (iTimeTarg < 0) {
                    wxLogError(_("An unexpected error occurred."));
                    return false;
                }

                // Get dates
                currentAnalogsDates = analogsDates.row(iAnalogDate);

                // Counter representing the current index
                int counter = 0;

                scoreArrayOneDay.fill(NaNf);
                dateArrayOneDay.fill(NaNf);

                // Loop over the members
                for (int iMem = 0; iMem < membersNb; ++iMem) {

                    // Extract target data
                    for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                        vTargData[iPtor] = &predictorsTarget[iPtor]->GetData()[iTimeTarg][iMem];
                    }

                    // Loop through the previous analogs for candidate data
                    for (int iPrevAnalog = 0; iPrevAnalog < analogsNbPrevious; iPrevAnalog++) {
                        // Find row in the predictor time array
                        int iTimeArch = asFind(&timeArchiveData[0], &timeArchiveData[timeArchiveDataSize - 1],
                                               currentAnalogsDates[iPrevAnalog], 0.01);

                        // Check if a row was found
                        if (iTimeArch != asNOT_FOUND && iTimeArch != asOUT_OF_RANGE) {
                            // Process the criteria
                            float thisscore = 0;
                            for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                                // Get data
                                vArchData[iPtor] = &predictorsArchive[iPtor]->GetData()[iTimeArch][iMem];

                                // Assess the criteria
                                wxASSERT(criteria.size() > iPtor);
                                wxASSERT(vTargData[iPtor]);
                                wxASSERT(vArchData[iPtor]);
                                wxASSERT(timeArchiveData.size() > iTimeArch);
                                wxASSERT_MSG(vArchData[iPtor]->size() == vTargData[iPtor]->size(), wxString::Format(
                                        "%s (%d th element) in archive, %s (%d th element) in target: vArchData size = %d, vTargData size = %d",
                                        asTime::GetStringTime(timeArchiveData[iTimeArch], "DD.MM.YYYY hh:mm"),
                                        iTimeArch, asTime::GetStringTime(timeTargetData[iTimeTarg], "DD.MM.YYYY hh:mm"),
                                        iTimeTarg, (int) vArchData[iPtor]->size(), (int) vTargData[iPtor]->size()));
                                float tmpscore = criteria[iPtor]->Assess(*vTargData[iPtor], *vArchData[iPtor],
                                                                         vRowsNb[iPtor], vColsNb[iPtor]);

                                // Weight and add the score
                                thisscore += tmpscore * params->GetPredictorWeight(step, iPtor);

                                if (asIsNaN(tmpscore)) {
                                    containsNaNs = true;
                                    wxLogWarning(_("NaNs were found in the criteria values (%s/%s)."),
                                                 predictorsArchive[iPtor]->GetProduct(),
                                                 predictorsArchive[iPtor]->GetDataId());
                                    wxLogWarning(_("Target date: %s, archive date: %s."),
                                                 asTime::GetStringTime(timeTargetData[iTimeTarg]),
                                                 asTime::GetStringTime(timeArchiveData[iTimeArch]));
                                }
                            }

                            // Check if the array is already full
                            if (counter > analogsNb - 1) {
                                if (isasc) {
                                    if (thisscore < scoreArrayOneDay[analogsNb - 1]) {
                                        asArraysInsert(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1],
                                                       &dateArrayOneDay[0], &dateArrayOneDay[analogsNb - 1], Asc,
                                                       thisscore, (float) timeArchiveData[iTimeArch]);
                                    }
                                } else {
                                    if (thisscore > scoreArrayOneDay[analogsNb - 1]) {
                                        asArraysInsert(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1],
                                                       &dateArrayOneDay[0], &dateArrayOneDay[analogsNb - 1], Desc,
                                                       thisscore, (float) timeArchiveData[iTimeArch]);
                                    }
                                }
                            } else if (counter < analogsNb - 1) {
                                // Add score and date to the vectors
                                scoreArrayOneDay[counter] = thisscore;
                                dateArrayOneDay[counter] = (float) timeArchiveData[iTimeArch];
                            } else if (counter == analogsNb - 1) {
                                // Add score and date to the vectors
                                scoreArrayOneDay[counter] = thisscore;
                                dateArrayOneDay[counter] = (float) timeArchiveData[iTimeArch];

                                // Sort both scores and dates arrays
                                if (isasc) {
                                    asSortArrays(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1],
                                                        &dateArrayOneDay[0], &dateArrayOneDay[analogsNb - 1], Asc);
                                } else {
                                    asSortArrays(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1],
                                                        &dateArrayOneDay[0], &dateArrayOneDay[analogsNb - 1], Desc);
                                }
                            }

                            counter++;
                        } else {
                            wxLogError(_("The date was not found in the array (Analogs subdates fct). That should not happen."));
                            return false;
                        }
                    }
                }

                if (counter < analogsNb) {
                    wxLogWarning(_("There is not enough available data to satisfy the number of analogs (in asProcessor::GetAnalogsSubDates)."));
                    wxLogWarning(_("Analogs number (%d) > counter (%d)"), analogsNb, counter);
                }

                // Copy results
                finalAnalogsCriteria.row(iAnalogDate) = scoreArrayOneDay.transpose();
                finalAnalogsDates.row(iAnalogDate) = dateArrayOneDay.transpose();
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
    wxLogVerbose(_("The function asProcessor::GetAnalogsSubDates took %.3f s to execute"), float(sw.Time()) / 1000);

    return true;
}

bool asProcessor::GetAnalogsValues(asPredictand &predictand, asResultsDates &anaDates, asParameters *params,
                                   asResultsValues &results)
{
    // Extract Data
    a1f timeTargetSelection = anaDates.GetTargetDates();
    a2f analogsDates = anaDates.GetAnalogsDates();
    a2f analogsCriteria = anaDates.GetAnalogsCriteria();
    a1d predictandTime = predictand.GetTime();
    vi stations = params->GetPredictandStationIds();
    int stationsNb = (int) stations.size();
    va1f predictandDataNorm((long) stationsNb);
    va1f predictandDataRaw((long) stationsNb);
    for (int iStat = 0; iStat < stationsNb; iStat++) {
        predictandDataNorm[iStat] = predictand.GetDataNormalizedStation(stations[iStat]);
        predictandDataRaw[iStat] = predictand.GetDataRawStation(stations[iStat]);
    }

    int predictandTimeLength = predictand.GetTimeLength();
    int timeTargetSelectionLength = (int) timeTargetSelection.size();
    int analogsNb = (int) analogsDates.cols();

    wxASSERT(timeTargetSelectionLength > 0);

    // Correct the time arrays to account for predictand time and not predictors time
    if (params->GetTimeShiftDays() != 0) {
        for (int iTime = 0; iTime < timeTargetSelectionLength; iTime++) {
            timeTargetSelection[iTime] += params->GetTimeShiftDays();

            for (int iAnalog = 0; iAnalog < analogsNb; iAnalog++) {
                analogsDates(iTime, iAnalog) += params->GetTimeShiftDays();
            }
        }
    }

    // Get start and end dates
    double timeStart, timeEnd;
    timeStart = wxMax(predictandTime[0], params->GetArchiveStart());
    timeEnd = wxMin(predictandTime[predictandTimeLength - 1], params->GetArchiveEnd());

    // Check if data are effectively available for this period
    int indexPredictandTimeStart = asFindCeil(&predictandTime[0], &predictandTime[predictandTimeLength - 1], timeStart);
    int indexPredictandTimeEnd = asFindFloor(&predictandTime[0], &predictandTime[predictandTimeLength - 1], timeEnd);

    if (indexPredictandTimeStart < 0 || indexPredictandTimeEnd < 0) {
        wxLogError(_("An unexpected error occurred."));
        return false;
    }

    for (int iStat = 0; iStat < (int) stations.size(); iStat++) {
        while (asIsNaN(predictandDataNorm[iStat](indexPredictandTimeStart))) {
            indexPredictandTimeStart++;
        }
        while (asIsNaN(predictandDataNorm[iStat](indexPredictandTimeEnd))) {
            indexPredictandTimeEnd--;
        }
    }

    if (indexPredictandTimeEnd < 0) {
        wxLogError(_("An unexpected error occurred."));
        return false;
    }

    timeStart = predictandTime[indexPredictandTimeStart];
    timeEnd = predictandTime[indexPredictandTimeEnd];
    if (timeEnd <= timeStart) {
        wxLogError(_("An unexpected error occurred."));
        return false;
    }

    // Get start and end indices for the analogs dates
    double timeStartTarg = wxMax(timeStart, (double) timeTargetSelection[0]);
    double timeEndTarg = wxMin(timeEnd, (double) timeTargetSelection[timeTargetSelectionLength - 1]);
    int indexTargDatesStart = asFindCeil(&timeTargetSelection[0], &timeTargetSelection[timeTargetSelectionLength - 1],
                                         timeStartTarg);
    int indexTargDatesEnd = asFindFloor(&timeTargetSelection[0], &timeTargetSelection[timeTargetSelectionLength - 1],
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
    float predictandTimeDays = params->GetPredictandTimeHours() / 24.0;

    // Resize containers
    wxASSERT(targTimeLength > 0);
    wxASSERT(analogsNb > 0);
    va2f finalAnalogValuesNorm(stationsNb, a2f(targTimeLength, analogsNb));
    va2f finalAnalogValuesRaw(stationsNb, a2f(targTimeLength, analogsNb));
    a2f finalAnalogCriteria(targTimeLength, analogsNb);
    a1f finalTargetDates(targTimeLength);
    va1f finalTargetValuesNorm(stationsNb, a1f(targTimeLength));
    va1f finalTargetValuesRaw(stationsNb, a1f(targTimeLength));

    // Get predictand values
    for (int iTargetDate = indexTargDatesStart; iTargetDate <= indexTargDatesEnd; iTargetDate++) {
        wxASSERT(iTargetDate >= 0);
        int iTargetDatenew = iTargetDate - indexTargDatesStart;
        float currentTargetDate = timeTargetSelection(iTargetDate);
        finalTargetDates[iTargetDatenew] = currentTargetDate;
        int predictandIndex = asFindClosest(&predictandTime[0], &predictandTime[predictandTimeLength - 1],
                                            currentTargetDate + predictandTimeDays, asHIDE_WARNINGS);
        if (ignoreTargetValues || (predictandIndex == asOUT_OF_RANGE) || (predictandIndex == asNOT_FOUND)) {
            for (int iStat = 0; iStat < (int) stations.size(); iStat++) {
                finalTargetValuesNorm[iStat](iTargetDatenew) = NaNf;
                finalTargetValuesRaw[iStat](iTargetDatenew) = NaNf;
            }
        } else {
            for (int iStat = 0; iStat < (int) stations.size(); iStat++) {
                finalTargetValuesNorm[iStat](iTargetDatenew) = predictandDataNorm[iStat](predictandIndex);
                finalTargetValuesRaw[iStat](iTargetDatenew) = predictandDataRaw[iStat](predictandIndex);
            }
        }

        for (int iAnalogDate = 0; iAnalogDate < analogsNb; iAnalogDate++) {
            float currentAnalogDate = analogsDates(iTargetDate, iAnalogDate);

            if (!asIsNaN(currentAnalogDate)) {
                // Check that the date is in the range
                if ((currentAnalogDate >= timeStart) && (currentAnalogDate <= timeEnd)) {
                    predictandIndex = asFindClosest(&predictandTime[0], &predictandTime[predictandTime.size() - 1],
                                                    currentAnalogDate + predictandTimeDays);
                    if ((predictandIndex == asOUT_OF_RANGE) || (predictandIndex == asNOT_FOUND)) {
                        wxString currDate = asTime::GetStringTime(currentAnalogDate + predictandTimeDays);
                        wxString startDate = asTime::GetStringTime(predictandTime[0]);
                        wxString endDate = asTime::GetStringTime(predictandTime[predictandTime.size() - 1]);
                        wxLogWarning(_("The current analog date (%s) was not found in the predictand time array (%s-%s)."),
                                     currDate, startDate, endDate);
                        for (int iStat = 0; iStat < (int) stations.size(); iStat++) {
                            finalAnalogValuesNorm[iStat](iTargetDatenew, iAnalogDate) = NaNf;
                            finalAnalogValuesRaw[iStat](iTargetDatenew, iAnalogDate) = NaNf;
                        }
                    } else {
                        for (int iStat = 0; iStat < (int) stations.size(); iStat++) {
                            finalAnalogValuesNorm[iStat](iTargetDatenew, iAnalogDate) = predictandDataNorm[iStat](
                                    predictandIndex);
                            finalAnalogValuesRaw[iStat](iTargetDatenew, iAnalogDate) = predictandDataRaw[iStat](
                                    predictandIndex);
                        }
                    }
                } else {
                    wxLogError(_("The current analog date (%s) is outside of the allowed period (%s-%s))."),
                               asTime::GetStringTime(currentAnalogDate, "DD.MM.YYYY"),
                               asTime::GetStringTime(timeStart, "DD.MM.YYYY"),
                               asTime::GetStringTime(timeEnd, "DD.MM.YYYY"));
                    for (int iStat = 0; iStat < (int) stations.size(); iStat++) {
                        finalAnalogValuesNorm[iStat](iTargetDatenew, iAnalogDate) = NaNf;
                        finalAnalogValuesRaw[iStat](iTargetDatenew, iAnalogDate) = NaNf;
                    }
                }
                finalAnalogCriteria(iTargetDatenew, iAnalogDate) = analogsCriteria(iTargetDate, iAnalogDate);

            } else {
                wxLogError(_("The current analog date is a NaN."));
                finalAnalogCriteria(iTargetDatenew, iAnalogDate) = NaNf;
            }
        }

#ifndef UNIT_TESTING
#ifdef _DEBUG
        wxASSERT(!asHasNaN(&finalAnalogCriteria(iTargetDatenew, 0),
                           &finalAnalogCriteria(iTargetDatenew, analogsNb - 1)));
#endif
#endif
    }

    // Copy results to the resulting object
    results.SetAnalogsValuesNorm(finalAnalogValuesNorm);
    results.SetAnalogsValuesRaw(finalAnalogValuesRaw);
    results.SetAnalogsCriteria(finalAnalogCriteria);
    results.SetTargetDates(finalTargetDates);
    results.SetTargetValuesNorm(finalTargetValuesNorm);
    results.SetTargetValuesRaw(finalTargetValuesRaw);

    return true;
}
