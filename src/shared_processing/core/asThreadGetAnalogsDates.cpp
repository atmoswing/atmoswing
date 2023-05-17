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

#include "asThreadGetAnalogsDates.h"

#include <utility>

#include "asCriteria.h"
#include "asPredictor.h"
#include "asProcessor.h"
#include "asTimeArray.h"

asThreadGetAnalogsDates::asThreadGetAnalogsDates(
    vector<asPredictor*> predictorsArchive, vector<asPredictor*> predictorsTarget, asTimeArray* timeArrayArchiveData,
    asTimeArray* timeArrayArchiveSelection, asTimeArray* timeArrayTargetData, asTimeArray* timeArrayTargetSelection,
    vector<asCriteria*> criteria, asParameters* params, int step, a1i& vRowsNb, a1i& vColsNb, int start, int end,
    a2f* finalAnalogsCriteria, a2f* finalAnalogsDates, bool* containsNaNs, bool allowDuplicateDates, bool* success)
    : asThread(asThread::ProcessorGetAnalogsDates),
      m_pPredictorsArchive(std::move(predictorsArchive)),
      m_pPredictorsTarget(std::move(predictorsTarget)),
      m_pTimeArrayArchiveData(timeArrayArchiveData),
      m_pTimeArrayArchiveSelection(timeArrayArchiveSelection),
      m_pTimeArrayTargetData(timeArrayTargetData),
      m_pTimeArrayTargetSelection(timeArrayTargetSelection),
      m_criteria(std::move(criteria)),
      m_params(params),
      m_vTargData(vpa2f(vRowsNb.size())),
      m_vArchData(vpa2f(vRowsNb.size())),
      m_vRowsNb(vRowsNb),
      m_vColsNb(vColsNb),
      m_pFinalAnalogsCriteria(finalAnalogsCriteria),
      m_pFinalAnalogsDates(finalAnalogsDates),
      m_allowDuplicateDates(allowDuplicateDates),
      m_success(success) {
    m_step = step;
    m_start = start;
    m_end = end;
    m_pContainsNaNs = containsNaNs;

    wxASSERT_MSG(m_end < timeArrayTargetSelection->GetSize(),
                 _("The given time array end is superior to the time array size."));
    wxASSERT_MSG(m_end != timeArrayTargetSelection->GetSize() - 2,
                 asStrF(_("The given time array end is missing its last value (end=%d, size=%d)."), m_end,
                        (int)timeArrayTargetSelection->GetSize()));
}

asThreadGetAnalogsDates::~asThreadGetAnalogsDates() {}

wxThread::ExitCode asThreadGetAnalogsDates::Entry() {
    // Extract time arrays
    a1d timeArchiveData = m_pTimeArrayArchiveData->GetTimeArray();
    a1d timeTargetData = m_pTimeArrayTargetData->GetTimeArray();
    a1d timeTargetSelection = m_pTimeArrayTargetSelection->GetTimeArray();

    // Some other variables
    int iTimeTarg, iTimeArch;
    int predictorsNb = m_params->GetPredictorsNb(m_step);
    auto membersNb = (m_pPredictorsTarget)[0]->GetData()[0].size();
    int analogsNb = m_params->GetAnalogsNumber(m_step);
    bool isAsc = (m_criteria[0]->GetOrder() == Asc);

    wxASSERT(m_end < timeTargetSelection.size());
    wxASSERT(timeArchiveData.size() == (m_pPredictorsArchive)[0]->GetData().size());
    wxASSERT(timeTargetData.size() <= (m_pPredictorsTarget)[0]->GetData().size());
    wxASSERT(membersNb == (m_pPredictorsArchive)[0]->GetData()[0].size());

    // Containers for daily results
    a1f scoreArrayOneDay(analogsNb);
    scoreArrayOneDay.fill(NAN);
    a1f dateArrayOneDay(analogsNb);
    dateArrayOneDay.fill(NAN);

    // DateArray object instantiation. There is one array for all the predictors, as they are aligned, so it picks
    // the predictors we are interested in, but which didn't take place at the same time.
    asTimeArray dateArrayArchiveSelection(m_pTimeArrayArchiveSelection->GetStart(),
                                          m_pTimeArrayArchiveSelection->GetEnd(), m_params->GetAnalogsTimeStepHours(),
                                          m_params->GetTimeArrayAnalogsMode());
    dateArrayArchiveSelection.SetForbiddenYears(m_pTimeArrayArchiveSelection->GetForbiddenYears());

    // Reset the index start target
    int iTimeTargStart = 0;

    // Loop through every timestep as target data
    for (int iDateTarg = m_start; iDateTarg <= m_end; iDateTarg++) {
        int iTimeTargRelative = asProcessor::FindNextDate(timeTargetSelection, timeTargetData, iTimeTargStart,
                                                          iDateTarg);

        // Check if a row was found
        if (iTimeTargRelative != asNOT_FOUND && iTimeTargRelative != asOUT_OF_RANGE) {
            // Convert the relative index into an absolute index
            iTimeTarg = iTimeTargRelative + iTimeTargStart;
            iTimeTargStart = iTimeTarg;

            // DateArray object initialization.
            dateArrayArchiveSelection.Init(timeTargetSelection[iDateTarg], m_params->GetAnalogsIntervalDays(),
                                           m_params->GetAnalogsExcludeDays());

            // Counter representing the current index
            int counter = 0;

            scoreArrayOneDay.fill(NAN);
            dateArrayOneDay.fill(NAN);

            // Loop over the members
            for (int iMem = 0; iMem < membersNb; ++iMem) {
                // Extract target data
                for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                    m_vTargData[iPtor] = &(m_pPredictorsTarget)[iPtor]->GetData()[iTimeTarg][iMem];
                }

                // Reset the index start target
                int iTimeArchStart = 0;

                // Loop through the datearray for candidate data
                for (int iDateArch = 0; iDateArch < dateArrayArchiveSelection.GetSize(); iDateArch++) {
                    int iTimeArchRelative = asProcessor::FindNextDate(dateArrayArchiveSelection, timeArchiveData,
                                                                      iTimeArchStart, iDateArch);

                    // Check if a row was found
                    if (iTimeArchRelative != asNOT_FOUND && iTimeArchRelative != asOUT_OF_RANGE) {
                        // Convert the relative index into an absolute index
                        iTimeArch = iTimeArchRelative + iTimeArchStart;
                        iTimeArchStart = iTimeArch;

                        // Process the criteria
                        float thisScore = 0;
                        for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                            // Get data
                            m_vArchData[iPtor] = &(m_pPredictorsArchive)[iPtor]->GetData()[iTimeArch][iMem];

                            // Assess the criteria
                            wxASSERT(m_criteria.size() > iPtor);
                            float tmpScore = m_criteria[iPtor]->Assess(*m_vTargData[iPtor], *m_vArchData[iPtor],
                                                                       m_vRowsNb[iPtor], m_vColsNb[iPtor]);

                            // Weight and add the score
                            thisScore += tmpScore * m_params->GetPredictorWeight(m_step, iPtor);
                        }
                        if (isnan(thisScore)) {
                            *m_pContainsNaNs = true;
                            continue;
                        }

                        // Avoid duplicate analog dates
                        if (!m_allowDuplicateDates && iMem > 0) {
                            if (counter <= analogsNb - 1) {
                                wxFAIL;
                                wxLogError(
                                    _("It should not happen that the array of "
                                      "analogue dates is not full when adding members."));
                                *m_success = false;
                                return (wxThread::ExitCode)-1;
                            }
                            asProcessor::InsertInArraysNoDuplicate(isAsc, analogsNb, (float)timeArchiveData[iTimeArch],
                                                                   thisScore, scoreArrayOneDay, dateArrayOneDay);
                        } else {
                            asProcessor::InsertInArrays(isAsc, analogsNb, (float)timeArchiveData[iTimeArch], thisScore,
                                                        counter, scoreArrayOneDay, dateArrayOneDay);
                        }

                        counter++;
                    } else {
                        wxLogError(
                            _("The date was not found in the array (Analogs Dates fct, "
                              "multithreaded option). That should not happen."));
                        wxLogError(_("Start: %g, end: %g, desired value: %g."), timeArchiveData[iTimeArchStart],
                                   timeArchiveData[timeArchiveData.size() - 1], dateArrayArchiveSelection[iDateArch]);
                        *m_success = false;
                        return (wxThread::ExitCode)-1;
                    }
                }
            }

            // Copy results
            m_pFinalAnalogsCriteria->row(iDateTarg) = scoreArrayOneDay.transpose();
            m_pFinalAnalogsDates->row(iDateTarg) = dateArrayOneDay.transpose();
        }
    }

    *m_success = true;

    return (wxThread::ExitCode)0;
}
