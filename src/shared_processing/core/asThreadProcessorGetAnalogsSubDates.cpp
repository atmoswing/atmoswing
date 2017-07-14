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

#include "asThreadProcessorGetAnalogsSubDates.h"

#include <asDataPredictor.h>
#include <asPredictorCriteria.h>
#include <asTimeArray.h>


asThreadProcessorGetAnalogsSubDates::asThreadProcessorGetAnalogsSubDates(
        std::vector<asDataPredictor *> predictorsArchive, std::vector<asDataPredictor *> predictorsTarget,
        asTimeArray *timeArrayArchiveData, asTimeArray *timeArrayTargetData, a1f *timeTargetSelection,
        std::vector<asPredictorCriteria *> criteria, asParameters &params, int step, vpa2f &vTargData,
        vpa2f &vArchData, a1i &vRowsNb, a1i &vColsNb, int start, int end,
        a2f *finalAnalogsCriteria, a2f *finalAnalogsDates, a2f *previousAnalogsDates,
        bool *containsNaNs)
        : asThread(),
          m_pPredictorsArchive(predictorsArchive),
          m_pPredictorsTarget(predictorsTarget),
          m_pTimeArrayArchiveData(timeArrayArchiveData),
          m_pTimeArrayTargetData(timeArrayTargetData),
          m_pTimeTargetSelection(timeTargetSelection),
          m_criteria(criteria),
          m_params(params),
          m_vTargData(vTargData),
          m_vArchData(vArchData),
          m_vRowsNb(vRowsNb),
          m_vColsNb(vColsNb),
          m_pFinalAnalogsCriteria(finalAnalogsCriteria),
          m_pFinalAnalogsDates(finalAnalogsDates),
          m_pPreviousAnalogsDates(previousAnalogsDates)
{
    m_type = asThread::ProcessorGetAnalogsDates;
    m_step = step;
    m_start = start;
    m_end = end;
    m_pContainsNaNs = containsNaNs;

    wxASSERT_MSG(m_end < m_pTimeTargetSelection->size(),
                 _("The given time array end is superior to the time array size."));
    wxASSERT_MSG(m_end != m_pTimeTargetSelection->size() - 2,
                 wxString::Format(_("The given time array end is missing its last value (end=%d, size=%d)."), m_end,
                                  (int) m_pTimeTargetSelection->size()));
}

asThreadProcessorGetAnalogsSubDates::~asThreadProcessorGetAnalogsSubDates()
{

}

wxThread::ExitCode asThreadProcessorGetAnalogsSubDates::Entry()
{
    // Extract time arrays
    a1d timeArchiveData = m_pTimeArrayArchiveData->GetTimeArray();
    a1d timeTargetData = m_pTimeArrayTargetData->GetTimeArray();

    // Some other variables
    float tmpscore, thisscore;
    int timeArchiveDataSize = timeArchiveData.size();
    int timeTargetDataSize = timeTargetData.size();
    int predictorsNb = m_params.GetPredictorsNb(m_step);
    unsigned int membersNb = (unsigned int) (m_pPredictorsTarget)[0]->GetData()[0].size();
    int analogsNbPrevious = m_params.GetAnalogsNumber(m_step - 1);
    int analogsNb = m_params.GetAnalogsNumber(m_step);
    bool isasc = (m_criteria[0]->GetOrder() == Asc);

    wxASSERT(m_end < m_pTimeTargetSelection->size());
    wxASSERT(timeArchiveDataSize == (int) (m_pPredictorsArchive)[0]->GetData().size());
    wxASSERT(timeTargetDataSize == (int) (m_pPredictorsTarget)[0]->GetData().size());
    wxASSERT(membersNb == (unsigned int) (m_pPredictorsArchive)[0]->GetData()[0].size());

    // Containers for daily results
    a1f currentAnalogsDates(analogsNbPrevious);
    a1f ScoreArrayOneDay(analogsNb);
    a1f DateArrayOneDay(analogsNb);

    // Loop through every timestep as target data
    // Former, but disabled: for (int iDateTarg=m_start; !ThreadsManager().Cancelled() && (iDateTarg<=m_end); iDateTarg++)
    for (int iDateTarg = m_start; iDateTarg <= m_end; iDateTarg++) {
        int iTimeTarg = asTools::SortedArraySearch(&timeTargetData[0], &timeTargetData[timeTargetDataSize - 1],
                                                    (double) m_pTimeTargetSelection->coeff(iDateTarg), 0.01);
        wxASSERT(m_pTimeTargetSelection->coeff(iDateTarg) > 0);
        wxASSERT(iTimeTarg >= 0);

        // Get dates
        currentAnalogsDates = m_pPreviousAnalogsDates->row(iDateTarg);

        // Counter representing the current index
        int counter = 0;

        // Loop over the members
        for (int iMem = 0; iMem < membersNb; ++iMem) {

            // Extract target data
            for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                m_vTargData[iPtor] = &(m_pPredictorsTarget)[iPtor]->GetData()[iTimeTarg][iMem];
            }

            // Loop through the previous analogs for candidate data
            for (int iPrevAnalog = 0; iPrevAnalog < analogsNbPrevious; iPrevAnalog++) {
                // Find row in the predictor time array
                int iTimeArch = asTools::SortedArraySearch(&timeArchiveData[0],
                                                            &timeArchiveData[timeArchiveDataSize - 1],
                                                            currentAnalogsDates[iPrevAnalog], 0.01);
                wxASSERT(iTimeArch >= 0);
                if (iTimeArch < 0) {
                    wxLogError(_("An unexpected error occurred."));
                    return (wxThread::ExitCode) 1;
                }

                // Check if a row was found
                if (iTimeArch != asNOT_FOUND && iTimeArch != asOUT_OF_RANGE) {
                    // Process the criteria
                    thisscore = 0;
                    for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                        // Get data
                        m_vArchData[iPtor] = &(m_pPredictorsArchive)[iPtor]->GetData()[iTimeArch][iMem];

                        // Assess the criteria
                        wxASSERT(m_criteria.size() > (unsigned) iPtor);
                        tmpscore = m_criteria[iPtor]->Assess(*m_vTargData[iPtor], *m_vArchData[iPtor],
                                                              m_vRowsNb[iPtor], m_vColsNb[iPtor]);

                        // Weight and add the score
                        thisscore += tmpscore * m_params.GetPredictorWeight(m_step, iPtor);
                    }
                    if (asTools::IsNaN(thisscore)) {
                        *m_pContainsNaNs = true;
                    }

                    // Check if the array is already full
                    if (counter > analogsNb - 1) {
                        if (isasc) {
                            if (thisscore < ScoreArrayOneDay[analogsNb - 1]) {
                                asTools::SortedArraysInsert(&ScoreArrayOneDay[0], &ScoreArrayOneDay[analogsNb - 1],
                                                            &DateArrayOneDay[0], &DateArrayOneDay[analogsNb - 1], Asc,
                                                            thisscore, (float) timeArchiveData[iTimeArch]);
                            }
                        } else {
                            if (thisscore > ScoreArrayOneDay[analogsNb - 1]) {
                                asTools::SortedArraysInsert(&ScoreArrayOneDay[0], &ScoreArrayOneDay[analogsNb - 1],
                                                            &DateArrayOneDay[0], &DateArrayOneDay[analogsNb - 1], Desc,
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
                    wxLogError(_("The date was not found in the array (Analogs subdates fct). That should not happen."));
                    return (wxThread::ExitCode) 1;
                }
            }
        }

        // Check that the number of occurrences are larger than the desired analogs number. If not, set a warning
        if (counter >= analogsNb) {
            // Copy results
            m_pFinalAnalogsCriteria->row(iDateTarg) = ScoreArrayOneDay.head(analogsNb).transpose();
            m_pFinalAnalogsDates->row(iDateTarg) = DateArrayOneDay.head(analogsNb).transpose();
        } else {
            wxLogWarning(_("There is not enough available data to satisfy the number of analogs"));
            wxLogWarning(_("Analogs number (%d) > counter (%d)"), analogsNb, counter);
            return (wxThread::ExitCode) 1;
        }
    }

    return (wxThread::ExitCode) 0;
}
