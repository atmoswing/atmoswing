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

#include <asPredictor.h>
#include <asCriteria.h>
#include <asTimeArray.h>


asThreadGetAnalogsDates::asThreadGetAnalogsDates(std::vector<asPredictor *> predictorsArchive,
                                                 std::vector<asPredictor *> predictorsTarget,
                                                 asTimeArray *timeArrayArchiveData,
                                                 asTimeArray *timeArrayArchiveSelection,
                                                 asTimeArray *timeArrayTargetData,
                                                 asTimeArray *timeArrayTargetSelection,
                                                 std::vector<asCriteria *> criteria, asParameters &params,
                                                 int step, vpa2f &vTargData, vpa2f &vArchData, a1i &vRowsNb,
                                                 a1i &vColsNb, int start, int end, a2f *finalAnalogsCriteria,
                                                 a2f *finalAnalogsDates, bool *containsNaNs)
        : asThread(),
          m_pPredictorsArchive(predictorsArchive),
          m_pPredictorsTarget(predictorsTarget),
          m_pTimeArrayArchiveData(timeArrayArchiveData),
          m_pTimeArrayArchiveSelection(timeArrayArchiveSelection),
          m_pTimeArrayTargetData(timeArrayTargetData),
          m_pTimeArrayTargetSelection(timeArrayTargetSelection),
          m_criteria(criteria),
          m_params(params),
          m_vTargData(vTargData),
          m_vArchData(vArchData),
          m_vRowsNb(vRowsNb),
          m_vColsNb(vColsNb),
          m_pFinalAnalogsCriteria(finalAnalogsCriteria),
          m_pFinalAnalogsDates(finalAnalogsDates)
{
    m_type = asThread::ProcessorGetAnalogsDates;
    m_step = step;
    m_start = start;
    m_end = end;
    m_pContainsNaNs = containsNaNs;

    wxASSERT_MSG(m_end < timeArrayTargetSelection->GetSize(),
                 _("The given time array end is superior to the time array size."));
    wxASSERT_MSG(m_end != timeArrayTargetSelection->GetSize() - 2,
                 wxString::Format(_("The given time array end is missing its last value (end=%d, size=%d)."), m_end,
                                  (int) timeArrayTargetSelection->GetSize()));
}

asThreadGetAnalogsDates::~asThreadGetAnalogsDates()
{

}

wxThread::ExitCode asThreadGetAnalogsDates::Entry()
{
    // Extract time arrays
    a1d timeArchiveData = m_pTimeArrayArchiveData->GetTimeArray();
    a1d timeTargetData = m_pTimeArrayTargetData->GetTimeArray();
    a1d timeTargetSelection = m_pTimeArrayTargetSelection->GetTimeArray();

    // Some other variables
    float tmpscore, thisscore;
    int counter = 0;
    int iTimeTarg, iTimeArch, iTimeTargRelative, iTimeArchRelative;
    int timeArchiveDataSize = timeArchiveData.size();
    int timeTargetDataSize = timeTargetData.size();
    int predictorsNb = m_params.GetPredictorsNb(m_step);
    unsigned int membersNb = (unsigned int) (m_pPredictorsTarget)[0]->GetData()[0].size();
    int analogsNb = m_params.GetAnalogsNumber(m_step);
    bool isasc = (m_criteria[0]->GetOrder() == Asc);

    wxASSERT(m_end < timeTargetSelection.size());
    wxASSERT(timeArchiveDataSize == (int) (m_pPredictorsArchive)[0]->GetData().size());
    wxASSERT(timeTargetDataSize == (int) (m_pPredictorsTarget)[0]->GetData().size());
    wxASSERT(membersNb == (unsigned int) (m_pPredictorsArchive)[0]->GetData()[0].size());

    // Containers for daily results
    a1f ScoreArrayOneDay(analogsNb);
    a1f DateArrayOneDay(analogsNb);

    // DateArray object instantiation. There is one array for all the predictors, as they are aligned, so it picks the predictors we are interested in, but which didn't take place at the same time.
    asTimeArray dateArrayArchiveSelection(m_pTimeArrayArchiveSelection->GetStart(),
                                          m_pTimeArrayArchiveSelection->GetEnd(),
                                          m_params.GetTimeArrayAnalogsTimeStepHours(),
                                          m_params.GetTimeArrayAnalogsMode());
    dateArrayArchiveSelection.SetForbiddenYears(m_pTimeArrayArchiveSelection->GetForbiddenYears());

    // Reset the index start target
    int iTimeTargStart = 0;

    // Loop through every timestep as target data
    // Former, but disabled: for (int iDateTarg=m_start; !ThreadsManager().Cancelled() && (iDateTarg<=m_end); iDateTarg++)
    for (int iDateTarg = m_start; iDateTarg <= m_end; iDateTarg++) {
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
            dateArrayArchiveSelection.Init(timeTargetSelection[iDateTarg], m_params.GetTimeArrayAnalogsIntervalDays(),
                                           m_params.GetTimeArrayAnalogsExcludeDays());
            dateArrayArchiveSelection.Init(timeTargetSelection[iDateTarg], m_params.GetTimeArrayAnalogsIntervalDays(),
                                           m_params.GetTimeArrayAnalogsExcludeDays());

            // Counter representing the current index
            counter = 0;

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
                    // Check if the next data is the following. If not, search for it in the array.
                    wxASSERT(timeArchiveData.size() > iTimeArchStart + 1);
                    wxASSERT(dateArrayArchiveSelection.GetSize() > iDateArch);
                    if (timeArchiveDataSize > iTimeArchStart + 1 &&
                        std::abs(dateArrayArchiveSelection[iDateArch] - timeArchiveData[iTimeArchStart + 1]) < 0.01) {
                        iTimeArchRelative = 1;
                    } else {
                        iTimeArchRelative = asTools::SortedArraySearch(&timeArchiveData[iTimeArchStart],
                                                                       &timeArchiveData[timeArchiveDataSize - 1],
                                                                       dateArrayArchiveSelection[iDateArch], 0.01);
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
                                                                &DateArrayOneDay[0], &DateArrayOneDay[analogsNb - 1],
                                                                Asc, thisscore, (float) timeArchiveData[iTimeArch]);
                                }
                            } else {
                                if (thisscore > ScoreArrayOneDay[analogsNb - 1]) {
                                    asTools::SortedArraysInsert(&ScoreArrayOneDay[0], &ScoreArrayOneDay[analogsNb - 1],
                                                                &DateArrayOneDay[0], &DateArrayOneDay[analogsNb - 1],
                                                                Desc, thisscore, (float) timeArchiveData[iTimeArch]);
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
                        wxLogError(_("The date was not found in the array (Analogs Dates fct, multithreaded option). That should not happen."));
                        wxLogError(_("Start: %g, end: %g, desired value: %g."), timeArchiveData[iTimeArchStart],
                                   timeArchiveData[timeArchiveDataSize - 1], dateArrayArchiveSelection[iDateArch]);
                        return (wxThread::ExitCode) 1;
                    }
                }
            }

            // Check that the number of occurences are larger than the desired analogs number. If not, set a warning
            if (counter >= analogsNb) {
                // Copy results
                m_pFinalAnalogsCriteria->row(iDateTarg) = ScoreArrayOneDay.transpose();
                m_pFinalAnalogsDates->row(iDateTarg) = DateArrayOneDay.transpose();
            } else {
                wxLogError(_("There is not enough available data to satisfy the number of analogs:"));
                wxLogError(_("   Analogs number (%d) > counter (%d), date array size (%d) with %d days intervals."),
                           analogsNb, counter, dateArrayArchiveSelection.GetSize(),
                           m_params.GetTimeArrayAnalogsIntervalDays());
                wxLogError(_("   Date array start (%.0f), date array end (%.0f), timestep (%.0f), dateTarg (%.0f)."),
                           timeArchiveData[0], timeArchiveData[timeArchiveDataSize - 1],
                           m_params.GetTimeArrayAnalogsTimeStepHours(), timeTargetSelection[iTimeTarg]);
                return (wxThread::ExitCode) 1;
            }
        }
    }

    return (wxThread::ExitCode) 0;
}
