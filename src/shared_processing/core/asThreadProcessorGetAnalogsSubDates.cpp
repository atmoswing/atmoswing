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


asThreadProcessorGetAnalogsSubDates::asThreadProcessorGetAnalogsSubDates(std::vector < asDataPredictor* > predictorsArchive,
                                                                         std::vector < asDataPredictor* > predictorsTarget,
                                                                         asTimeArray* timeArrayArchiveData,
                                                                         asTimeArray* timeArrayTargetData,
                                                                         Array1DFloat* timeTargetSelection,
                                                                         std::vector < asPredictorCriteria* > criteria,
                                                                         asParameters &params, int step,
                                                                         VpArray2DFloat &vTargData,
                                                                         VpArray2DFloat &vArchData,
                                                                         Array1DInt &vRowsNb, Array1DInt &vColsNb,
                                                                         int start, int end,
                                                                         Array2DFloat* finalAnalogsCriteria,
                                                                         Array2DFloat* finalAnalogsDates,
                                                                         Array2DFloat* previousAnalogsDates,
                                                                         bool* containsNaNs)
:
asThread(),
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
    m_status = Initializing;
    m_type = asThread::ProcessorGetAnalogsDates;
    m_step = step;
    m_start = start;
    m_end = end;
    m_pContainsNaNs = containsNaNs;

    wxASSERT_MSG(m_end<m_pTimeTargetSelection->size(), _("The given time array end is superior to the time array size."));
    wxASSERT_MSG(m_end!=m_pTimeTargetSelection->size()-2, wxString::Format(_("The given time array end is missing its last value (end=%d, size=%d)."), m_end, (int)m_pTimeTargetSelection->size()));

    m_status = Waiting;
}

asThreadProcessorGetAnalogsSubDates::~asThreadProcessorGetAnalogsSubDates()
{

}

wxThread::ExitCode asThreadProcessorGetAnalogsSubDates::Entry()
{
    m_status = Working;

    // Extract time arrays
    Array1DDouble timeArchiveData = m_pTimeArrayArchiveData->GetTimeArray();
    Array1DDouble timeTargetData = m_pTimeArrayTargetData->GetTimeArray();

    // Some other variables
    float tmpscore, thisscore;
    int timeArchiveDataSize = timeArchiveData.size();
    int timeTargetDataSize = timeTargetData.size();
    int predictorsNb = m_params.GetPredictorsNb(m_step);
    int analogsNbPrevious = m_params.GetAnalogsNumber(m_step-1);
    int analogsNb = m_params.GetAnalogsNumber(m_step);
    bool isasc = (m_criteria[0]->GetOrder()==Asc);

    wxASSERT(m_end<m_pTimeTargetSelection->size());
    wxASSERT(timeArchiveDataSize==(int)(m_pPredictorsArchive)[0]->GetData().size());
    wxASSERT(timeTargetDataSize==(int)(m_pPredictorsTarget)[0]->GetData().size());

    // Containers for daily results
    Array1DFloat currentAnalogsDates(analogsNbPrevious);
    Array1DFloat ScoreArrayOneDay(analogsNb);
    Array1DFloat DateArrayOneDay(analogsNb);

    // Loop through every timestep as target data
    // Former, but disabled: for (int i_dateTarg=m_start; !ThreadsManager().Cancelled() && (i_dateTarg<=m_end); i_dateTarg++)
    for (int i_dateTarg=m_start; i_dateTarg<=m_end; i_dateTarg++)
    {
        int i_timeTarg = asTools::SortedArraySearch(&timeTargetData[0], &timeTargetData[timeTargetDataSize-1], (double)m_pTimeTargetSelection->coeff(i_dateTarg), 0.01);
        wxASSERT(m_pTimeTargetSelection->coeff(i_dateTarg)>0);
        wxASSERT(i_timeTarg>=0);

        // Extract target data
        for (int i_ptor=0; i_ptor<predictorsNb; i_ptor++)
        {
            m_vTargData[i_ptor] = &(m_pPredictorsTarget)[i_ptor]->GetData()[i_timeTarg];
        }

        // Get dates
// TODO (phorton#1#): Check if the dates are really consistent between the steps !!
        currentAnalogsDates = m_pPreviousAnalogsDates->row(i_dateTarg);

        // Counter representing the current index
        int counter = 0;

        // Loop through the previous analogs for candidate data
        for (int i_prevAnalogs=0; i_prevAnalogs<analogsNbPrevious; i_prevAnalogs++)
        {
            // Find row in the predictor time array
            int i_timeArch = asTools::SortedArraySearch(&timeArchiveData[0], &timeArchiveData[timeArchiveDataSize-1], currentAnalogsDates[i_prevAnalogs], 0.01);
            wxASSERT(i_timeArch>=0);

            // Check if a row was found
            if (i_timeArch!=asNOT_FOUND && i_timeArch!=asOUT_OF_RANGE)
            {
                // Process the criteria
                thisscore = 0;
                for (int i_ptor=0; i_ptor<predictorsNb; i_ptor++)
                {
                    // Get data
                    m_vArchData[i_ptor] = &(m_pPredictorsArchive)[i_ptor]->GetData()[i_timeArch];

                    // Assess the criteria
                    wxASSERT(m_criteria.size()>(unsigned)i_ptor);
                    tmpscore = m_criteria[i_ptor]->Assess(*m_vTargData[i_ptor], *m_vArchData[i_ptor], m_vRowsNb[i_ptor], m_vColsNb[i_ptor]);

                    // Weight and add the score
                    thisscore += tmpscore * m_params.GetPredictorWeight(m_step, i_ptor);
                }
                if (asTools::IsNaN(thisscore))
                {
                    *m_pContainsNaNs = true;
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
                return (wxThread::ExitCode)1;
            }
        }

        // Check that the number of occurences are larger than the desired analogs number. If not, set a warning
        if (counter>=analogsNb)
        {
            // Copy results
            m_pFinalAnalogsCriteria->row(i_dateTarg) = ScoreArrayOneDay.head(analogsNb).transpose();
            m_pFinalAnalogsDates->row(i_dateTarg) = DateArrayOneDay.head(analogsNb).transpose();
        }
        else
        {
            asLogWarning(_("There is not enough available data to satisfy the number of analogs"));
            asLogWarning(wxString::Format(_("Analogs number (%d) > counter (%d)"), analogsNb, counter));
            return (wxThread::ExitCode)1;
        }
    }

    m_status = Done;

    return (wxThread::ExitCode)0;
}
