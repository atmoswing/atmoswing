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

#include "asThreadGetAnalogsSubDates.h"

#include <utility>

#include "asCriteria.h"
#include "asPredictor.h"
#include "asTimeArray.h"

asThreadGetAnalogsSubDates::asThreadGetAnalogsSubDates(
    std::vector<asPredictor *> predictorsArchive, std::vector<asPredictor *> predictorsTarget,
    asTimeArray *timeArrayArchiveData, asTimeArray *timeArrayTargetData, a1f *timeTargetSelection,
    std::vector<asCriteria *> criteria, asParameters *params, int step, vpa2f &vTargData, vpa2f &vArchData,
    a1i &vRowsNb, a1i &vColsNb, int start, int end, a2f *finalAnalogsCriteria, a2f *finalAnalogsDates,
    a2f *previousAnalogsDates, bool *containsNaNs, bool *success)
    : asThread(asThread::ProcessorGetAnalogsDates),
      m_pPredictorsArchive(std::move(predictorsArchive)),
      m_pPredictorsTarget(std::move(predictorsTarget)),
      m_pTimeArrayArchiveData(timeArrayArchiveData),
      m_pTimeArrayTargetData(timeArrayTargetData),
      m_pTimeTargetSelection(timeTargetSelection),
      m_criteria(std::move(criteria)),
      m_params(params),
      m_vTargData(vTargData),
      m_vArchData(vArchData),
      m_vRowsNb(vRowsNb),
      m_vColsNb(vColsNb),
      m_pFinalAnalogsCriteria(finalAnalogsCriteria),
      m_pFinalAnalogsDates(finalAnalogsDates),
      m_pPreviousAnalogsDates(previousAnalogsDates),
      m_success(success) {
    m_step = step;
    m_start = start;
    m_end = end;
    m_pContainsNaNs = containsNaNs;

    wxASSERT_MSG(m_end < m_pTimeTargetSelection->size(),
                 _("The given time array end is superior to the time array size."));
    wxASSERT_MSG(m_end != m_pTimeTargetSelection->size() - 2,
                 wxString::Format(_("The given time array end is missing its last value (end=%d, size=%d)."), m_end,
                                  (int)m_pTimeTargetSelection->size()));
}

asThreadGetAnalogsSubDates::~asThreadGetAnalogsSubDates() {}

wxThread::ExitCode asThreadGetAnalogsSubDates::Entry() {
    // Extract time arrays
    a1d timeArchiveData = m_pTimeArrayArchiveData->GetTimeArray();
    a1d timeTargetData = m_pTimeArrayTargetData->GetTimeArray();

    // Some other variables
    float tmpscore, thisscore;
    int timeArchiveDataSize = timeArchiveData.size();
    int timeTargetDataSize = timeTargetData.size();
    int predictorsNb = m_params->GetPredictorsNb(m_step);
    int membersNb = (m_pPredictorsTarget)[0]->GetData()[0].size();
    int analogsNbPrevious = m_params->GetAnalogsNumber(m_step - 1);
    int analogsNb = m_params->GetAnalogsNumber(m_step);
    bool isasc = (m_criteria[0]->GetOrder() == Asc);

    wxASSERT(m_end < m_pTimeTargetSelection->size());
    wxASSERT(timeArchiveDataSize == (int)(m_pPredictorsArchive)[0]->GetData().size());
    wxASSERT(timeTargetDataSize == (int)(m_pPredictorsTarget)[0]->GetData().size());
    wxASSERT(membersNb == (m_pPredictorsArchive)[0]->GetData()[0].size());

    // Containers for daily results
    a1f currentAnalogsDates(analogsNbPrevious);
    a1f scoreArrayOneDay(analogsNb);
    scoreArrayOneDay.fill(NaNf);
    a1f dateArrayOneDay(analogsNb);
    dateArrayOneDay.fill(NaNf);

    // Loop through every timestep as target data
    // Former, but disabled: for (int iDateTarg=m_start; !ThreadsManager().Cancelled() && (iDateTarg<=m_end);
    // iDateTarg++)
    for (int iDateTarg = m_start; iDateTarg <= m_end; iDateTarg++) {
        int iTimeTarg = asFind(&timeTargetData[0], &timeTargetData[timeTargetDataSize - 1],
                               (double)m_pTimeTargetSelection->coeff(iDateTarg), 0.01);
        wxASSERT(m_pTimeTargetSelection->coeff(iDateTarg) > 0);
        wxASSERT(iTimeTarg >= 0);
        if (iTimeTarg < 0) {
            wxLogError(_("An unexpected error occurred."));
            *m_success = false;
            return (wxThread::ExitCode)-1;
        }

        // Get dates
        currentAnalogsDates = m_pPreviousAnalogsDates->row(iDateTarg);

        // Counter representing the current index
        int counter = 0;

        scoreArrayOneDay.fill(NaNf);
        dateArrayOneDay.fill(NaNf);

        // Loop over the members
        for (int iMem = 0; iMem < membersNb; ++iMem) {
            // Extract target data
            for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                m_vTargData[iPtor] = &(m_pPredictorsTarget)[iPtor]->GetData()[iTimeTarg][iMem];
            }

            // Loop through the previous analogs for candidate data
            for (int iPrevAnalog = 0; iPrevAnalog < analogsNbPrevious; iPrevAnalog++) {

                if (asIsNaN(currentAnalogsDates[iPrevAnalog])) {
                    *m_pContainsNaNs = true;
                    continue;
                }

                // Find row in the predictor time array
                int iTimeArch = asFind(&timeArchiveData[0], &timeArchiveData[timeArchiveDataSize - 1],
                                       currentAnalogsDates[iPrevAnalog], 0.01);
                wxASSERT(iTimeArch >= 0);
                if (iTimeArch < 0) {
                    wxLogError(_("An unexpected error occurred."));
                    *m_success = false;
                    return (wxThread::ExitCode)-1;
                }

                // Check if a row was found
                if (iTimeArch != asNOT_FOUND && iTimeArch != asOUT_OF_RANGE) {
                    // Process the criteria
                    thisscore = 0;
                    for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                        // Get data
                        m_vArchData[iPtor] = &(m_pPredictorsArchive)[iPtor]->GetData()[iTimeArch][iMem];

                        // Assess the criteria
                        wxASSERT(m_criteria.size() > iPtor);
                        tmpscore = m_criteria[iPtor]->Assess(*m_vTargData[iPtor], *m_vArchData[iPtor], m_vRowsNb[iPtor],
                                                             m_vColsNb[iPtor]);

                        // Weight and add the score
                        thisscore += tmpscore * m_params->GetPredictorWeight(m_step, iPtor);
                    }
                    if (asIsNaN(thisscore)) {
                        *m_pContainsNaNs = true;
                        continue;
                    }

                    // Check if the array is already full
                    if (counter > analogsNb - 1) {
                        if (isasc) {
                            if (thisscore < scoreArrayOneDay[analogsNb - 1]) {
                                asArraysInsert(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1],
                                               &dateArrayOneDay[0], &dateArrayOneDay[analogsNb - 1], Asc, thisscore,
                                               (float)timeArchiveData[iTimeArch]);
                            }
                        } else {
                            if (thisscore > scoreArrayOneDay[analogsNb - 1]) {
                                asArraysInsert(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1],
                                               &dateArrayOneDay[0], &dateArrayOneDay[analogsNb - 1], Desc, thisscore,
                                               (float)timeArchiveData[iTimeArch]);
                            }
                        }
                    } else if (counter < analogsNb - 1) {
                        // Add score and date to the vectors
                        scoreArrayOneDay[counter] = thisscore;
                        dateArrayOneDay[counter] = (float)timeArchiveData[iTimeArch];
                    } else if (counter == analogsNb - 1) {
                        // Add score and date to the vectors
                        scoreArrayOneDay[counter] = thisscore;
                        dateArrayOneDay[counter] = (float)timeArchiveData[iTimeArch];

                        // Sort both scores and dates arrays
                        if (isasc) {
                            asSortArrays(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1], &dateArrayOneDay[0],
                                         &dateArrayOneDay[analogsNb - 1], Asc);
                        } else {
                            asSortArrays(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1], &dateArrayOneDay[0],
                                         &dateArrayOneDay[analogsNb - 1], Desc);
                        }
                    }

                    counter++;
                } else {
                    wxLogError(_("The date was not found in the array (Analogs subdates fct). "
                        "That should not happen."));
                    *m_success = false;
                    return (wxThread::ExitCode)-1;
                }
            }
        }

        // Copy results
        m_pFinalAnalogsCriteria->row(iDateTarg) = scoreArrayOneDay.head(analogsNb).transpose();
        m_pFinalAnalogsDates->row(iDateTarg) = dateArrayOneDay.head(analogsNb).transpose();
    }

    *m_success = true;

    return (wxThread::ExitCode)0;
}
