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
 * Portions Copyright 2019 Pascal Horton, University of Bern.
 */

#include "asProcessor.h"

#include "asCriteria.h"
#include "asParameters.h"
#include "asPredictor.h"
#include "asPreprocessor.h"
#include "asResultsDates.h"
#include "asResultsValues.h"
#include "asThreadGetAnalogsDates.h"
#include "asThreadGetAnalogsSubDates.h"
#include "asTimeArray.h"

#ifdef APP_FORECASTER
#include "AtmoswingAppForecaster.h"
#endif
#ifdef APP_OPTIMIZER
#include "AtmoswingAppOptimizer.h"
#endif
#ifdef USE_CUDA
#include "asProcessorCuda.cuh"
#endif

bool asProcessor::GetAnalogsDates(std::vector<asPredictor*> predictorsArchive,
                                  std::vector<asPredictor*> predictorsTarget, asTimeArray& timeArrayArchiveData,
                                  asTimeArray& timeArrayArchiveSelection, asTimeArray& timeArrayTargetData,
                                  asTimeArray& timeArrayTargetSelection, std::vector<asCriteria*> criteria,
                                  asParameters* params, int step, asResultsDates& results, bool& containsNaNs) {
    // Get the processing method
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase* pConfig = wxFileConfig::Get();
    int method = pConfig->Read("/Processing/Method", (long)asMULTITHREADS);
    bool allowMultithreading = pConfig->ReadBool("/Processing/AllowMultithreading", true);
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
    int timeTargetSelectionSize = (int)timeTargetSelection.size();
    wxASSERT(criteria[0]);
    bool isAsc = (criteria[0]->GetOrder() == Asc);
    int predictorsNb = params->GetPredictorsNb(step);
    int membersNb = predictorsTarget[0]->GetData()[0].size();

    wxASSERT(!predictorsArchive.empty());
    wxASSERT((int)predictorsArchive.size() == predictorsNb);

    // Check analogs number. Correct if superior to the time serie
    int analogsNb = params->GetAnalogsNumber(step);
    if (analogsNb > timeArrayArchiveSelection.GetSize() * predictorsArchive[0]->GetMembersNb()) {
        wxLogError(_("The given analog number is superior to the time series."));
        return false;
    }

    // Matrices containers
    a1i vRowsNb(predictorsNb);
    a1i vColsNb(predictorsNb);

    for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
        wxASSERT((int)predictorsArchive.size() > iPtor);
        wxASSERT(predictorsArchive[iPtor]);
        wxASSERT(!predictorsArchive[iPtor]->GetData().empty());
        wxASSERT(vRowsNb.size() > iPtor);
        wxASSERT(vColsNb.size() > iPtor);
        vRowsNb[iPtor] = (int)predictorsArchive[iPtor]->GetData()[0][0].rows();
        vColsNb[iPtor] = (int)predictorsArchive[iPtor]->GetData()[0][0].cols();

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

#if USE_GUI
    // The progress bar
    asDialogProgressBar ProgressBar(_("Processing the data comparison."), timeTargetSelectionSize);
#endif

    switch (method) {
#ifdef USE_CUDA
        case (asCUDA): {
            // Check no members
            for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                if (predictorsArchive[iPtor]->GetMembersNb() > 1) {
                    wxLogError(_("No support for ensemble datasets in CUDA yet."));
                    return false;
                }
            }

            // To minimize the data copy, we only allow 1 dataset
            if (predictorsArchive[0] != predictorsTarget[0]) {
                wxLogError(_("The CUDA implementation is only available in calibration (prefect prog)."));
                return false;
            }

            // Extract time arrays
            a1d timeArchData = timeArrayArchiveData.GetTimeArray();
            if (!CheckArchiveTimeArray(predictorsArchive, timeArchData)) return false;
            a1d timeTargData = timeArrayTargetData.GetTimeArray();
            if (!CheckTargetTimeArray(predictorsTarget, timeTargData)) return false;

            // Constant data
            vf weights(predictorsNb);
            vi colsNb(predictorsNb);
            vi rowsNb(predictorsNb);
            std::vector<CudaCriteria> crit(predictorsNb);

            for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                weights[iPtor] = params->GetPredictorWeight(step, iPtor);
                colsNb[iPtor] = vColsNb[iPtor];
                rowsNb[iPtor] = vRowsNb[iPtor];
                if (criteria[iPtor]->GetName().IsSameAs("S1grads")) {
                    crit[iPtor] = S1grads;
                } else if (criteria[iPtor]->GetName().IsSameAs("S0")) {
                    crit[iPtor] = S0;
                } else if (criteria[iPtor]->GetName().IsSameAs("S2grads")) {
                    crit[iPtor] = S2grads;
                } else if (criteria[iPtor]->GetName().IsSameAs("RMSE")) {
                    crit[iPtor] = RMSE;
                } else if (criteria[iPtor]->GetName().IsSameAs("MD")) {
                    crit[iPtor] = MD;
                } else if (criteria[iPtor]->GetName().IsSameAs("RSE")) {
                    crit[iPtor] = RSE;
                } else if (criteria[iPtor]->GetName().IsSameAs("SAD")) {
                    crit[iPtor] = SAD;
                } else if (criteria[iPtor]->GetName().IsSameAs("DMV")) {
                    crit[iPtor] = DMV;
                } else if (criteria[iPtor]->GetName().IsSameAs("DSD")) {
                    crit[iPtor] = DSD;
                } else {
                    wxLogError(_("The %s criteria is not yet implemented for CUDA."), criteria[iPtor]->GetName());
                    return false;
                }
            }

            // Get max predictor size
            long totDataSize = 0;
            for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                int ptsNb = vColsNb[iPtor] * vRowsNb[iPtor];
                totDataSize += timeArchData.size() * ptsNb;
            }

            // Alloc space for predictor data
            float *hData, *dData = nullptr;
            hData = static_cast<float*>(malloc(totDataSize * sizeof(float)));
            checkCudaErrors(cudaMalloc((void**)&dData, totDataSize * sizeof(float)));

            // Copy predictor data to the host array
            vl ptorStart(predictorsNb);
            long pStart = 0;
            for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                ptorStart[iPtor] = pStart;
                int ptsNb = vColsNb[iPtor] * vRowsNb[iPtor];

                // Copy data in the new arrays
                for (int iTime = 0; iTime < timeArchData.size(); iTime++) {
                    float* pData = predictorsArchive[iPtor]->GetData()[iTime][0].data();
                    for (int iPt = 0; iPt < ptsNb; iPt++) {
                        hData[ptorStart[iPtor] + iTime * ptsNb + iPt] = pData[iPt];
                        pStart++;
                    }
                }
            }
            wxASSERT(totDataSize == pStart);

            // Copy the data to the device
            checkCudaErrors(cudaMemcpy(dData, hData, totDataSize * sizeof(float), cudaMemcpyHostToDevice));

            // DateArray object instantiation. There is one array for all the predictors, as they are aligned, so it
            // picks the predictors we are interested in, but which didn't take place at the same time.
            asTimeArray datesArchiveSlt(timeArrayArchiveSelection.GetStart(), timeArrayArchiveSelection.GetEnd(),
                                        params->GetAnalogsTimeStepHours(), params->GetTimeArrayAnalogsMode());
            if (timeArrayArchiveSelection.HasForbiddenYears()) {
                datesArchiveSlt.SetForbiddenYears(timeArrayArchiveSelection.GetForbiddenYears());
            }

            // Containers for daily results
            a1f scoreArrayOneDay(analogsNb);
            a1f dateArrayOneDay(analogsNb);

            // Get typical number of candidates
            datesArchiveSlt.Init(timeTargetSelection[0], params->GetAnalogsIntervalDays(),
                                 params->GetAnalogsExcludeDays());

            // Alloc space for indices
            int maxCandNb = int(datesArchiveSlt.GetSize() * 1.2);  // 1.2 as margin
            int *indicesArch, *dIdxArch = nullptr;
            checkCudaErrors(cudaMallocHost((void**)&indicesArch, nStreams * maxCandNb * sizeof(int)));
            checkCudaErrors(cudaMalloc((void**)&dIdxArch, nStreams * maxCandNb * sizeof(int)));

            // Get a new container for variable vectors
            float* currentDates;
            currentDates = static_cast<float*>(malloc(nStreams * maxCandNb * sizeof(float)));

            // Alloc space for results
            float *hRes, *dRes = nullptr;
            checkCudaErrors(cudaMallocHost((void**)&hRes, nStreams * maxCandNb * sizeof(float)));
            checkCudaErrors(cudaMalloc((void**)&dRes, nStreams * maxCandNb * sizeof(float)));

            // Number of candidates per stream
            vi nbCandidates(nStreams);

            // Reset the index start target
            int iTimeTargStart = 0;

            // Init streams
            cudaStream_t streams[nStreams];
            for (auto& stream : streams) checkCudaErrors(cudaStreamCreate(&stream));

            // Extract indices
            for (int i = 0; i < timeTargetSelectionSize + nStreams / 2; i++) {
                // Prepare date arrays and start on kernel
                if (i < timeTargetSelectionSize) {
                    int iDateTarg = i;
                    int streamId = iDateTarg % nStreams;
                    int offset = streamId * maxCandNb;

                    int iTimeTargRelative = FindNextDate(timeTargetSelection, timeTargData, iTimeTargStart, iDateTarg);

                    // Check if a row was found
                    if (iTimeTargRelative == asNOT_FOUND || iTimeTargRelative == asOUT_OF_RANGE) {
                        continue;
                    }

                    // Convert the relative index into an absolute index
                    int iTimeTarg = iTimeTargRelative + iTimeTargStart;
                    iTimeTargStart = iTimeTarg;

                    // DateArray initialization
                    datesArchiveSlt.Init(timeTargetSelection[iDateTarg], params->GetAnalogsIntervalDays(),
                                         params->GetAnalogsExcludeDays());

                    // Counter representing the current index
                    int counter = 0;

                    // Reset the index start target
                    int iTimeArchStart = 0;
                    int iTimeArchRelative;

                    // Loop through the datesArchiveSlt for candidate data
                    for (int iDateArch = 0; iDateArch < datesArchiveSlt.GetSize(); iDateArch++) {
                        // Check if the next data is the following. If not, search for it in the array.
                        if (timeArchData.size() > iTimeArchStart + 1 &&
                            std::abs(datesArchiveSlt[iDateArch] - timeArchData[iTimeArchStart + 1]) < 0.01) {
                            iTimeArchRelative = 1;
                        } else {
                            iTimeArchRelative = asFind(&timeArchData[iTimeArchStart],
                                                       &timeArchData[timeArchData.size() - 1],
                                                       datesArchiveSlt[iDateArch], 0.01);
                        }

                        // Check if a row was found
                        if (iTimeArchRelative == asNOT_FOUND || iTimeArchRelative == asOUT_OF_RANGE) {
                            wxLogError(_("The candidate (%s) was not found in the array (%s - %s) (Target date: %s)."),
                                       asTime::GetStringTime(datesArchiveSlt[iDateArch]),
                                       asTime::GetStringTime(timeArchData[iTimeArchStart]),
                                       asTime::GetStringTime(timeArchData[timeArchData.size() - 1]),
                                       asTime::GetStringTime(timeTargetSelection[iDateTarg]));
                            continue;
                        }

                        // Convert the relative index into an absolute index
                        int iTimeArch = iTimeArchRelative + iTimeArchStart;
                        iTimeArchStart = iTimeArch;

                        // Store the index and the date
                        currentDates[offset + counter] = (float)timeArchData[iTimeArch];
                        indicesArch[offset + counter] = iTimeArch;
                        counter++;
                    }
                    int nbCand = counter;

                    // Copy to device
                    checkCudaErrors(cudaMemcpyAsync(&dIdxArch[offset], &indicesArch[offset], nbCand * sizeof(int),
                                                    cudaMemcpyHostToDevice, streams[streamId]));

                    // Doing the work on GPU
                    checkCudaErrors(cudaMemsetAsync(&dRes[offset], 0, maxCandNb * sizeof(float), streams[streamId]));
                    asProcessorCuda::ProcessCriteria(dData, ptorStart, iTimeTarg, dIdxArch, dRes, nbCand, colsNb,
                                                     rowsNb, weights, crit, streams[streamId], offset);

                    nbCandidates[streamId] = nbCand;
                }

                // Postprocess the results
                if (i >= nStreams / 2) {
                    int iDateTarg = i - nStreams / 2;
                    int streamId = iDateTarg % nStreams;
                    int offset = streamId * maxCandNb;

                    // Check for any error from the kernel
                    checkCudaErrors(cudaGetLastError());

                    // Copy the resulting array from the device
                    checkCudaErrors(cudaMemcpyAsync(&hRes[offset], &dRes[offset],
                                                    nbCandidates[streamId] * sizeof(float), cudaMemcpyDeviceToHost,
                                                    streams[streamId]));

                    checkCudaErrors(cudaStreamSynchronize(streams[streamId]));

                    // Sort and store results
                    int resCounter = 0;
                    scoreArrayOneDay.fill(NaNf);
                    dateArrayOneDay.fill(NaNf);

                    for (int iDateArch = 0; iDateArch < nbCandidates[streamId]; iDateArch++) {
#ifdef _DEBUG
                        if (asIsNaN(hRes[offset + iDateArch])) {
                            containsNaNs = true;
                            wxLogWarning(_("NaNs were found in the criteria values."));
                            wxLogWarning(_("Target date: %s, archive date: %s."),
                                         asTime::GetStringTime(timeTargetSelection[iDateTarg]),
                                         asTime::GetStringTime(dateArrayOneDay[iDateArch]));
                        }
#endif

                        InsertInArrays(isAsc, analogsNb, currentDates[offset + iDateArch], hRes[offset + iDateArch],
                                       resCounter, scoreArrayOneDay, dateArrayOneDay);

                        resCounter++;
                    }

                    if (resCounter < analogsNb) {
                        wxLogWarning(_("There is not enough available data to satisfy the number of analogs."));
                        wxLogWarning(_("Analogs number (%d) > resCounter (%d), date array size (%d) with "
                                       "%d days intervals."),
                                     analogsNb, resCounter, datesArchiveSlt.GetSize(),
                                     params->GetAnalogsIntervalDays());
                    }

                    finalAnalogsCriteria.row(iDateTarg) = scoreArrayOneDay.head(analogsNb).transpose();
                    finalAnalogsDates.row(iDateTarg) = dateArrayOneDay.head(analogsNb).transpose();
                }
            }

            free(hData);
            checkCudaErrors(cudaFree(dData));
            checkCudaErrors(cudaFreeHost(hRes));
            checkCudaErrors(cudaFree(dRes));
            checkCudaErrors(cudaFreeHost(indicesArch));
            checkCudaErrors(cudaFree(dIdxArch));
            free(currentDates);

            for (auto& stream : streams) cudaStreamDestroy(stream);

            break;
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
            std::vector<bool*> vContainsNaNs;
            std::vector<bool*> vSuccess;
            for (int iThread = 0; iThread < threadsNb; iThread++) {
                bool* flag = new bool;
                *flag = false;
                vContainsNaNs.push_back(flag);
                bool* success = new bool;
                *success = false;
                vSuccess.push_back(success);
                int start = end + 1;
                end = ceil(((float)(iThread + 1) * (float)(timeTargetSelectionSize - 1) / (float)threadsNb));
                wxASSERT(end >= start);

                auto* thread = new asThreadGetAnalogsDates(
                    predictorsArchive, predictorsTarget, &timeArrayArchiveData, &timeArrayArchiveSelection,
                    &timeArrayTargetData, &timeArrayTargetSelection, criteria, params, step, vRowsNb, vColsNb, start,
                    end, &finalAnalogsCriteria, &finalAnalogsDates, flag, allowDuplicateDates, success);

                threadType = thread->GetType();

                ThreadsManager().AddThread(thread);
            }

            // Wait until all done
            ThreadsManager().Wait(threadType);

            // Flush logs
            wxLog::FlushActive();

            for (auto& containsNaN : vContainsNaNs) {
                if (*containsNaN) {
                    containsNaNs = true;
                }
                wxDELETE(containsNaN);
            }
            if (containsNaNs) {
                wxLogWarning(_("NaNs were found in the criteria values."));
            }

            bool failed = false;
            for (auto& success : vSuccess) {
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
            vpa2f vTargData = vpa2f(predictorsNb);
            vpa2f vArchData = vpa2f(predictorsNb);

            // Extract some data
            a1d timeArchiveData = timeArrayArchiveData.GetTimeArray();
            if (!CheckArchiveTimeArray(predictorsArchive, timeArchiveData)) return false;
            a1d timeTargetData = timeArrayTargetData.GetTimeArray();
            if (!CheckTargetTimeArray(predictorsTarget, timeTargetData)) return false;

            // Containers for daily results
            a1f scoreArrayOneDay(analogsNb);
            a1f dateArrayOneDay(analogsNb);

            // DateArray object instantiation. There is one array for all the predictors, as they are aligned,
            // so it picks the predictors we are interested in, but which didn't take place at the same time.
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
                        int iTimeArchRelative = FindNextDate(dateArrayArchiveSelection, timeArchiveData, iTimeArchStart,
                                                             iDateArch);

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

                        if (asIsNaN(thisScore)) {
                            continue;
                        }

                        // Avoid duplicate analog dates
                        if (!allowDuplicateDates && iMem > 0) {
                            if (counter <= analogsNb - 1) {
                                wxFAIL;
                                wxLogError(
                                    _("It should not happen that the array of analogue dates is "
                                      "not full when adding members."));
                                return false;
                            }
                            InsertInArraysNoDuplicate(isAsc, analogsNb, (float)timeArchiveData[iTimeArch], thisScore,
                                                      scoreArrayOneDay, dateArrayOneDay);
                        } else {
                            InsertInArrays(isAsc, analogsNb, (float)timeArchiveData[iTimeArch], thisScore, counter,
                                           scoreArrayOneDay, dateArrayOneDay);
                        }

                        counter++;
                    }
                }

                // Copy results
                finalAnalogsCriteria.row(iDateTarg) = scoreArrayOneDay.transpose();
                finalAnalogsDates.row(iDateTarg) = dateArrayOneDay.transpose();
            }

            break;
        }

        default:
            asThrow(_("The processing method is not correctly defined."));
    }

    // Copy results to the resulting object
    results.SetTargetDates(timeTargetSelection);
    results.SetAnalogsCriteria(finalAnalogsCriteria);
    results.SetAnalogsDates(finalAnalogsDates);

    // Display the time the function took
    wxLogVerbose(_("The function asProcessor::GetAnalogsDates took %.3f s to execute"), float(sw.Time()) / 1000.0f);

    return true;
}

bool asProcessor::CheckTargetTimeArray(const std::vector<asPredictor*>& predictorsTarget, const a1d& timeTargetData) {
    wxASSERT(predictorsTarget[0]);
    wxASSERT(timeTargetData.size() == predictorsTarget[0]->GetData().size());
    if ((size_t)timeTargetData.size() != predictorsTarget[0]->GetData().size()) {
        wxLogError(_("The size of the time array and the target data are not equal (%d != %d)."),
                   (int)timeTargetData.size(), (int)predictorsTarget[0]->GetData().size());
        wxLogError(_("Time array starts on %s and ends on %s."), asTime::GetStringTime(timeTargetData[0], ISOdateTime),
                   asTime::GetStringTime(timeTargetData[timeTargetData.size() - 1], ISOdateTime));
        return false;
    }

    return true;
}

bool asProcessor::CheckArchiveTimeArray(const std::vector<asPredictor*>& predictorsArchive,
                                        const a1d& timeArchiveData) {
    wxASSERT(timeArchiveData.size() > 0);
    wxASSERT(!predictorsArchive.empty());
    wxASSERT(!predictorsArchive[0]->GetData().empty());
    wxASSERT(timeArchiveData.size() == predictorsArchive[0]->GetData().size());
    if ((size_t)timeArchiveData.size() != predictorsArchive[0]->GetData().size()) {
        wxLogError(_("The size of the time array and the archive data are not equal (%d != %d)."),
                   (int)timeArchiveData.size(), (int)predictorsArchive[0]->GetData().size());
        wxLogError(_("Time array starts on %s and ends on %s."), asTime::GetStringTime(timeArchiveData[0], ISOdateTime),
                   asTime::GetStringTime(timeArchiveData[timeArchiveData.size() - 1], ISOdateTime));
        return false;
    }

    return true;
}

int asProcessor::FindNextDate(asTimeArray& dateArray, a1d& timeData, int iTimeStart, int iDate) {
    // Check if the next data is the following. If not, search for it in the array.
    if (timeData.size() > iTimeStart + 1 && std::abs(dateArray[iDate] - timeData[iTimeStart + 1]) < 0.01) {
        return 1;
    }

    return asFind(&timeData[iTimeStart], &timeData[timeData.size() - 1], dateArray[iDate], 0.01);
}

int asProcessor::FindNextDate(a1d& dateArray, a1d& timeData, int iTimeStart, int iDate) {
    // Check if the next data is the following. If not, search for it in the array.
    if (timeData.size() > iTimeStart + 1 && std::abs(dateArray[iDate] - timeData[iTimeStart + 1]) < 0.01) {
        return 1;
    }

    return asFind(&timeData[iTimeStart], &timeData[timeData.size() - 1], dateArray[iDate], 0.01);
}

void asProcessor::InsertInArrays(bool isAsc, int analogsNb, float analogDate, float score, int counter,
                                 a1f& scoreArrayOneDay, a1f& dateArrayOneDay) {
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
    } else {
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
                                            a1f& scoreArrayOneDay, a1f& dateArrayOneDay) {
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

bool asProcessor::GetAnalogsSubDates(std::vector<asPredictor*> predictorsArchive,
                                     std::vector<asPredictor*> predictorsTarget, asTimeArray& timeArrayArchiveData,
                                     asTimeArray& timeArrayTargetData, asResultsDates& anaDates,
                                     std::vector<asCriteria*> criteria, asParameters* params, int step,
                                     asResultsDates& results, bool& containsNaNs) {
    // Get the processing method
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase* pConfig = wxFileConfig::Get();
    int method = pConfig->Read("/Processing/Method", (long)asMULTITHREADS);
    bool allowMultithreading = pConfig->ReadBool("/Processing/AllowMultithreading", true);

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
    bool isAsc = (criteria[0]->GetOrder() == Asc);
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
        vRowsNb[iPtor] = (int)predictorsArchive[iPtor]->GetData()[0][0].rows();
        vColsNb[iPtor] = (int)predictorsArchive[iPtor]->GetData()[0][0].cols();

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

    // Containers for daily results
    a1f currentAnalogsDates(analogsNbPrevious);

    // Containers for final results
    a2f finalAnalogsCriteria(timeTargetSelectionSize, analogsNb);
    a2f finalAnalogsDates(timeTargetSelectionSize, analogsNb);

#if USE_GUI
    // The progress bar
    asDialogProgressBar ProgressBar(_("Processing the data comparison."), timeTargetSelectionSize);
#endif

    switch (method) {
#ifdef USE_CUDA
        case (asCUDA): {
            // Check no members
            for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                if (predictorsArchive[iPtor]->GetMembersNb() > 1) {
                    wxLogError(_("No support for ensemble datasets in CUDA yet."));
                    return false;
                }
            }

            // To minimize the data copy, we only allow 1 dataset
            if (predictorsArchive[0] != predictorsTarget[0]) {
                wxLogError(_("The CUDA implementation is only available in calibration (prefect prog)."));
                return false;
            }

            // Extract time arrays
            a1d timeArchData = timeArrayArchiveData.GetTimeArray();

            // Constant data
            vf weights(predictorsNb);
            vi colsNb(predictorsNb);
            vi rowsNb(predictorsNb);
            std::vector<CudaCriteria> crit(predictorsNb);

            for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                weights[iPtor] = params->GetPredictorWeight(step, iPtor);
                colsNb[iPtor] = vColsNb[iPtor];
                rowsNb[iPtor] = vRowsNb[iPtor];
                if (criteria[iPtor]->GetName().IsSameAs("S1grads")) {
                    crit[iPtor] = S1grads;
                } else if (criteria[iPtor]->GetName().IsSameAs("S0")) {
                    crit[iPtor] = S0;
                } else if (criteria[iPtor]->GetName().IsSameAs("S2grads")) {
                    crit[iPtor] = S2grads;
                } else if (criteria[iPtor]->GetName().IsSameAs("RMSE")) {
                    crit[iPtor] = RMSE;
                } else if (criteria[iPtor]->GetName().IsSameAs("MD")) {
                    crit[iPtor] = MD;
                } else if (criteria[iPtor]->GetName().IsSameAs("RSE")) {
                    crit[iPtor] = RSE;
                } else if (criteria[iPtor]->GetName().IsSameAs("SAD")) {
                    crit[iPtor] = SAD;
                } else if (criteria[iPtor]->GetName().IsSameAs("DMV")) {
                    crit[iPtor] = DMV;
                } else if (criteria[iPtor]->GetName().IsSameAs("DSD")) {
                    crit[iPtor] = DSD;
                } else {
                    wxLogError(_("The %s criteria is not yet implemented for CUDA."), criteria[iPtor]->GetName());
                    return false;
                }
            }

            // Get max predictor size
            long totDataSize = 0;
            for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                int ptsNb = vColsNb[iPtor] * vRowsNb[iPtor];
                totDataSize += timeArchData.size() * ptsNb;
            }

            // Alloc space for predictor data
            float *hData, *dData = nullptr;
            hData = static_cast<float*>(malloc(totDataSize * sizeof(float)));
            checkCudaErrors(cudaMalloc((void**)&dData, totDataSize * sizeof(float)));

            // Copy predictor data to the host array
            vl ptorStart(predictorsNb);
            long pStart = 0;
            for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                ptorStart[iPtor] = pStart;
                int ptsNb = vColsNb[iPtor] * vRowsNb[iPtor];

                // Copy data in the new arrays
                for (int iTime = 0; iTime < timeArchData.size(); iTime++) {
                    float* pData = predictorsArchive[iPtor]->GetData()[iTime][0].data();
                    for (int iPt = 0; iPt < ptsNb; iPt++) {
                        hData[ptorStart[iPtor] + iTime * ptsNb + iPt] = pData[iPt];
                        pStart++;
                    }
                }
            }
            wxASSERT(totDataSize == pStart);

            // Copy the data to the device
            checkCudaErrors(cudaMemcpy(dData, hData, totDataSize * sizeof(float), cudaMemcpyHostToDevice));

            // Containers for daily results
            a1f scoreArrayOneDay(analogsNb);
            a1f dateArrayOneDay(analogsNb);

            // Alloc space for indices
            int maxCandNb = int(analogsDates.cols());
            int *indicesArch, *dIdxArch = nullptr;
            ;
            checkCudaErrors(cudaMallocHost((void**)&indicesArch, nStreams * maxCandNb * sizeof(int)));
            checkCudaErrors(cudaMalloc((void**)&dIdxArch, nStreams * maxCandNb * sizeof(int)));

            // Get a new container for variable vectors
            float* currentDates;
            currentDates = static_cast<float*>(malloc(nStreams * maxCandNb * sizeof(float)));

            // Alloc space for results
            float *hRes, *dRes = nullptr;
            checkCudaErrors(cudaMallocHost((void**)&hRes, nStreams * maxCandNb * sizeof(float)));
            checkCudaErrors(cudaMalloc((void**)&dRes, nStreams * maxCandNb * sizeof(float)));

            // Number of candidates per stream
            vi nbCandidates(nStreams);

            // Init streams
            cudaStream_t streams[nStreams];
            for (auto& stream : streams) checkCudaErrors(cudaStreamCreate(&stream));

            // Loop through every timestep as target data
            for (int i = 0; i < timeTargetSelectionSize + nStreams / 2; i++) {
                // Prepare date arrays and start on kernel
                if (i < timeTargetSelectionSize) {
                    int iDateTarg = i;
                    int streamId = iDateTarg % nStreams;
                    int offset = streamId * maxCandNb;

                    int iTimeTarg = asFind(&timeTargetData[0], &timeTargetData[timeTargetDataSize - 1],
                                           timeTargetSelection[i], 0.01);
                    wxASSERT_MSG(iTimeTarg >= 0,
                                 asStrF(_("Looking for %s in betwwen %s and %s."),
                                                  asTime::GetStringTime(timeTargetSelection[i], "DD.MM.YYYY hh:mm"),
                                                  asTime::GetStringTime(timeTargetData[0], "DD.MM.YYYY hh:mm"),
                                                  asTime::GetStringTime(timeTargetData[timeTargetDataSize - 1],
                                                                        "DD.MM.YYYY hh:mm")));

                    // Get dates
                    currentAnalogsDates = analogsDates.row(i);

                    // Loop through the previous analogs for candidate data
                    for (int iPrevAnalog = 0; iPrevAnalog < analogsNbPrevious; iPrevAnalog++) {
                        // Find row in the predictor time array
                        int iTimeArch = asFind(&timeArchiveData[0], &timeArchiveData[timeArchiveDataSize - 1],
                                               currentAnalogsDates[iPrevAnalog], 0.01);

                        // Check if a row was found
                        if (iTimeArch == asNOT_FOUND || iTimeArch == asOUT_OF_RANGE) {
                            wxLogError(_(
                                "The date was not found in the array (Analogs subdates fct). That should not happen."));

                            return false;
                        }

                        // Store the index and the date
                        currentDates[offset + iPrevAnalog] = (float)timeArchData[iTimeArch];
                        indicesArch[offset + iPrevAnalog] = iTimeArch;
                    }

                    int nbCand = analogsNbPrevious;

                    // Copy to device
                    checkCudaErrors(cudaMemcpyAsync(&dIdxArch[offset], &indicesArch[offset], nbCand * sizeof(int),
                                                    cudaMemcpyHostToDevice, streams[streamId]));

                    // Doing the work on GPU
                    checkCudaErrors(cudaMemsetAsync(&dRes[offset], 0, maxCandNb * sizeof(float), streams[streamId]));
                    asProcessorCuda::ProcessCriteria(dData, ptorStart, iTimeTarg, dIdxArch, dRes, nbCand, colsNb,
                                                     rowsNb, weights, crit, streams[streamId], offset);

                    nbCandidates[streamId] = nbCand;
                }

                // Postprocess the results
                if (i >= nStreams / 2) {
                    int iDateTarg = i - nStreams / 2;
                    int streamId = iDateTarg % nStreams;
                    int offset = streamId * maxCandNb;

                    // Check for any error from the kernel
                    checkCudaErrors(cudaGetLastError());

                    // Copy the resulting array from the device
                    checkCudaErrors(cudaMemcpyAsync(&hRes[offset], &dRes[offset],
                                                    nbCandidates[streamId] * sizeof(float), cudaMemcpyDeviceToHost,
                                                    streams[streamId]));

                    checkCudaErrors(cudaStreamSynchronize(streams[streamId]));

                    // Sort and store results
                    int resCounter = 0;
                    scoreArrayOneDay.fill(NaNf);
                    dateArrayOneDay.fill(NaNf);

                    for (int iDateArch = 0; iDateArch < nbCandidates[streamId]; iDateArch++) {
#ifdef _DEBUG
                        if (asIsNaN(hRes[offset + iDateArch])) {
                            containsNaNs = true;
                            wxLogWarning(_("NaNs were found in the criteria values."));
                            wxLogWarning(_("Target date: %s, archive date: %s."),
                                         asTime::GetStringTime(timeTargetSelection[iDateTarg]),
                                         asTime::GetStringTime(dateArrayOneDay[iDateArch]));
                        }
#endif

                        InsertInArrays(isAsc, analogsNb, currentDates[offset + iDateArch], hRes[offset + iDateArch],
                                       resCounter, scoreArrayOneDay, dateArrayOneDay);

                        resCounter++;
                    }

                    finalAnalogsCriteria.row(iDateTarg) = scoreArrayOneDay.head(analogsNb).transpose();
                    finalAnalogsDates.row(iDateTarg) = dateArrayOneDay.head(analogsNb).transpose();
                }
            }

            free(hData);
            checkCudaErrors(cudaFree(dData));
            checkCudaErrors(cudaFreeHost(hRes));
            checkCudaErrors(cudaFree(dRes));
            checkCudaErrors(cudaFreeHost(indicesArch));
            checkCudaErrors(cudaFree(dIdxArch));
            free(currentDates);

            for (auto& stream : streams) cudaStreamDestroy(stream);

            break;
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
            std::vector<bool*> vContainsNaNs;
            std::vector<bool*> vSuccess;
            for (int iThread = 0; iThread < threadsNb; iThread++) {
                bool* flag = new bool;
                *flag = false;
                vContainsNaNs.push_back(flag);
                bool* success = new bool;
                *success = false;
                vSuccess.push_back(success);
                int start = end + 1;
                end = ceil(((float)(iThread + 1) * (float)(timeTargetSelectionSize - 1) / (float)threadsNb));
                wxASSERT_MSG(end >= start, asStrF("start = %d, end = %d, timeTargetSelectionSize = %d", start, end,
                                                  timeTargetSelectionSize));

                asThreadGetAnalogsSubDates* thread = new asThreadGetAnalogsSubDates(
                    predictorsArchive, predictorsTarget, &timeArrayArchiveData, &timeArrayTargetData,
                    &timeTargetSelection, criteria, params, step, vTargData, vArchData, vRowsNb, vColsNb, start, end,
                    &finalAnalogsCriteria, &finalAnalogsDates, &analogsDates, flag, success);
                threadType = thread->GetType();

                ThreadsManager().AddThread(thread);
            }

            // Wait until all done
            ThreadsManager().Wait(threadType);

            // Flush logs
            wxLog::FlushActive();

            for (auto& containsNaN : vContainsNaNs) {
                if (*containsNaN) {
                    containsNaNs = true;
                }
                wxDELETE(containsNaN);
            }
            if (containsNaNs) {
                wxLogWarning(_("NaNs were found in the criteria values."));
            }

            bool failed = false;
            for (auto& success : vSuccess) {
                if (!*success) {
                    failed = true;
                }
                wxDELETE(success);
            }
            if (failed) {
                wxFAIL;
                wxLogError(_("Errors were found during extraction of the analog dates."));
            }

            break;
        }

        case (asSTANDARD): {
            // Containers for daily results
            a1f scoreArrayOneDay(analogsNb);
            a1f dateArrayOneDay(analogsNb);

            // Loop through every timestep as target data
            for (int iAnalogDate = 0; iAnalogDate < timeTargetSelectionSize; iAnalogDate++) {
                int iTimeTarg = asFind(&timeTargetData[0], &timeTargetData[timeTargetDataSize - 1],
                                       timeTargetSelection[iAnalogDate], 0.01);
                wxASSERT_MSG(iTimeTarg >= 0,
                             asStrF(_("Looking for %s in betwwen %s and %s."),
                                    asTime::GetStringTime(timeTargetSelection[iAnalogDate], "DD.MM.YYYY hh:mm"),
                                    asTime::GetStringTime(timeTargetData[0], "DD.MM.YYYY hh:mm"),
                                    asTime::GetStringTime(timeTargetData[timeTargetDataSize - 1], "DD.MM.YYYY hh:mm")));

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
                        if (iTimeArch == asNOT_FOUND || iTimeArch == asOUT_OF_RANGE) {
                            wxLogError(_("The date was not found in the array (Analogs subdates fct)."));
                            return false;
                        }

                        // Process the criteria
                        float thisScore = 0;
                        for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                            // Get data
                            vArchData[iPtor] = &predictorsArchive[iPtor]->GetData()[iTimeArch][iMem];

                            // Assess the criteria
                            wxASSERT(criteria.size() > iPtor);
                            wxASSERT(vTargData[iPtor]);
                            wxASSERT(vArchData[iPtor]);
                            wxASSERT(timeArchiveData.size() > iTimeArch);
                            wxASSERT_MSG(
                                vArchData[iPtor]->size() == vTargData[iPtor]->size(),
                                asStrF("%s (%d th element) in archive, %s (%d th element) in target: vArchData size = "
                                       "%d, vTargData size = %d",
                                       asTime::GetStringTime(timeArchiveData[iTimeArch], "DD.MM.YYYY hh:mm"), iTimeArch,
                                       asTime::GetStringTime(timeTargetData[iTimeTarg], "DD.MM.YYYY hh:mm"), iTimeTarg,
                                       (int)vArchData[iPtor]->size(), (int)vTargData[iPtor]->size()));
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
                                             asTime::GetStringTime(timeTargetData[iTimeTarg]),
                                             asTime::GetStringTime(timeArchiveData[iTimeArch]));
                            }
                        }

                        if (asIsNaN(thisScore)) {
                            continue;
                        }

                        // Check if the array is already full
                        if (counter > analogsNb - 1) {
                            if (isAsc) {
                                if (thisScore < scoreArrayOneDay[analogsNb - 1]) {
                                    asArraysInsert(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1],
                                                   &dateArrayOneDay[0], &dateArrayOneDay[analogsNb - 1], Asc, thisScore,
                                                   (float)timeArchiveData[iTimeArch]);
                                }
                            } else {
                                if (thisScore > scoreArrayOneDay[analogsNb - 1]) {
                                    asArraysInsert(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1],
                                                   &dateArrayOneDay[0], &dateArrayOneDay[analogsNb - 1], Desc,
                                                   thisScore, (float)timeArchiveData[iTimeArch]);
                                }
                            }
                        } else if (counter < analogsNb - 1) {
                            // Add score and date to the vectors
                            scoreArrayOneDay[counter] = thisScore;
                            dateArrayOneDay[counter] = (float)timeArchiveData[iTimeArch];
                        } else if (counter == analogsNb - 1) {
                            // Add score and date to the vectors
                            scoreArrayOneDay[counter] = thisScore;
                            dateArrayOneDay[counter] = (float)timeArchiveData[iTimeArch];

                            // Sort both scores and dates arrays
                            if (isAsc) {
                                asSortArrays(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1],
                                             &dateArrayOneDay[0], &dateArrayOneDay[analogsNb - 1], Asc);
                            } else {
                                asSortArrays(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1],
                                             &dateArrayOneDay[0], &dateArrayOneDay[analogsNb - 1], Desc);
                            }
                        }

                        counter++;
                    }
                }

                // Copy results
                finalAnalogsCriteria.row(iAnalogDate) = scoreArrayOneDay.transpose();
                finalAnalogsDates.row(iAnalogDate) = dateArrayOneDay.transpose();
            }

            break;
        }

        default:
            asThrow(_("The processing method is not correctly defined."));
    }

#if USE_GUI
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

bool asProcessor::GetAnalogsValues(asPredictand& predictand, asResultsDates& anaDates, asParameters* params,
                                   asResultsValues& results) {
    // Extract Data
    a1f timeTargetSelection = anaDates.GetTargetDates();
    a2f analogsDates = anaDates.GetAnalogsDates();
    a2f analogsCriteria = anaDates.GetAnalogsCriteria();
    a1d predictandTime = predictand.GetTime();
    vi stations = params->GetPredictandStationIds();
    int stationsNb = (int)stations.size();
    va1f predictandDataNorm((long)stationsNb);
    va1f predictandDataRaw((long)stationsNb);
    for (int iStat = 0; iStat < stationsNb; iStat++) {
        predictandDataNorm[iStat] = predictand.GetDataNormalizedStation(stations[iStat]);
        predictandDataRaw[iStat] = predictand.GetDataRawStation(stations[iStat]);
    }

    int predictandTimeLength = predictand.GetTimeLength();
    int timeTargetSelectionLength = (int)timeTargetSelection.size();
    int analogsNb = (int)analogsDates.cols();

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

    for (int iStat = 0; iStat < (int)stations.size(); iStat++) {
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
    double timeStartTarg = wxMax(timeStart, (double)timeTargetSelection[0]);
    double timeEndTarg = wxMin(timeEnd, (double)timeTargetSelection[timeTargetSelectionLength - 1]);
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
    va2f finalAnalogValuesNorm(stationsNb, a2f::Ones(targTimeLength, analogsNb) * NaNf);
    va2f finalAnalogValuesRaw(stationsNb, a2f::Ones(targTimeLength, analogsNb) * NaNf);
    a2f finalAnalogCriteria = a2f::Ones(targTimeLength, analogsNb) * NaNf;
    a1f finalTargetDates = a1f::Ones(targTimeLength) * NaNf;
    va1f finalTargetValuesNorm(stationsNb, a1f::Ones(targTimeLength) * NaNf);
    va1f finalTargetValuesRaw(stationsNb, a1f::Ones(targTimeLength) * NaNf);

    // Get predictand values
    for (int iTargetDate = indexTargDatesStart; iTargetDate <= indexTargDatesEnd; iTargetDate++) {
        wxASSERT(iTargetDate >= 0);
        int iTargetDatenew = iTargetDate - indexTargDatesStart;
        float currentTargetDate = timeTargetSelection(iTargetDate);
        finalTargetDates[iTargetDatenew] = currentTargetDate;
        int predictandIndex = asFindClosest(&predictandTime[0], &predictandTime[predictandTimeLength - 1],
                                            currentTargetDate + predictandTimeDays, asHIDE_WARNINGS);
        if (!ignoreTargetValues && predictandIndex != asOUT_OF_RANGE && predictandIndex != asNOT_FOUND) {
            for (int iStat = 0; iStat < (int)stations.size(); iStat++) {
                finalTargetValuesNorm[iStat](iTargetDatenew) = predictandDataNorm[iStat](predictandIndex);
                finalTargetValuesRaw[iStat](iTargetDatenew) = predictandDataRaw[iStat](predictandIndex);
            }
        }

        for (int iAnalogDate = 0; iAnalogDate < analogsNb; iAnalogDate++) {
            float currentAnalogDate = analogsDates(iTargetDate, iAnalogDate);

            if (asIsNaN(currentAnalogDate)) {
                continue;
            }

            // Check that the date is in the range
            if (currentAnalogDate < timeStart || currentAnalogDate > timeEnd) {
                wxLogError(_("The current analog date (%s) is outside of the allowed period (%s-%s))."),
                           asTime::GetStringTime(currentAnalogDate, "DD.MM.YYYY"),
                           asTime::GetStringTime(timeStart, "DD.MM.YYYY"),
                           asTime::GetStringTime(timeEnd, "DD.MM.YYYY"));
                continue;
            }

            predictandIndex = asFindClosest(&predictandTime[0], &predictandTime[predictandTime.size() - 1],
                                            currentAnalogDate + predictandTimeDays);
            if ((predictandIndex == asOUT_OF_RANGE) || (predictandIndex == asNOT_FOUND)) {
                wxString currDate = asTime::GetStringTime(currentAnalogDate + predictandTimeDays);
                wxString startDate = asTime::GetStringTime(predictandTime[0]);
                wxString endDate = asTime::GetStringTime(predictandTime[predictandTime.size() - 1]);
                wxLogWarning(_("The current analog date (%s) was not found in the "
                               "predictand time array (%s-%s)."),
                             currDate, startDate, endDate);
                continue;
            }

            for (int iStat = 0; iStat < (int)stations.size(); iStat++) {
                finalAnalogValuesNorm[iStat](iTargetDatenew, iAnalogDate) = predictandDataNorm[iStat](predictandIndex);
                finalAnalogValuesRaw[iStat](iTargetDatenew, iAnalogDate) = predictandDataRaw[iStat](predictandIndex);
            }

            finalAnalogCriteria(iTargetDatenew, iAnalogDate) = analogsCriteria(iTargetDate, iAnalogDate);
        }
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
