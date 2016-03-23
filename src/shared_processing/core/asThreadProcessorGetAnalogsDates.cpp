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

#include "asThreadProcessorGetAnalogsDates.h"

#include <asDataPredictor.h>
#include <asPredictorCriteria.h>
#include <asTimeArray.h>


asThreadProcessorGetAnalogsDates::asThreadProcessorGetAnalogsDates(std::vector < asDataPredictor* > predictorsArchive,
                                                                   std::vector < asDataPredictor* > predictorsTarget,
                                                                   asTimeArray* timeArrayArchiveData,
                                                                   asTimeArray* timeArrayArchiveSelection,
                                                                   asTimeArray* timeArrayTargetData,
                                                                   asTimeArray* timeArrayTargetSelection,
                                                                   std::vector < asPredictorCriteria* > criteria,
                                                                   asParameters &params, int step,
                                                                   VpArray2DFloat &vTargData,
                                                                   VpArray2DFloat &vArchData,
                                                                   Array1DInt &vRowsNb, Array1DInt &vColsNb,
                                                                   int start, int end,
                                                                   Array2DFloat* finalAnalogsCriteria, Array2DFloat* finalAnalogsDates,
                                                                   bool* containsNaNs)
:
asThread(),
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
    m_status = Initializing;
    m_type = asThread::ProcessorGetAnalogsDates;
    m_step = step;
    m_start = start;
    m_end = end;
    m_pContainsNaNs = containsNaNs;

    wxASSERT_MSG(m_end<timeArrayTargetSelection->GetSize(), _("The given time array end is superior to the time array size."));
    wxASSERT_MSG(m_end!=timeArrayTargetSelection->GetSize()-2, wxString::Format(_("The given time array end is missing its last value (end=%d, size=%d)."), m_end, (int)timeArrayTargetSelection->GetSize()));

    m_status = Waiting;
}

asThreadProcessorGetAnalogsDates::~asThreadProcessorGetAnalogsDates()
{

}

wxThread::ExitCode asThreadProcessorGetAnalogsDates::Entry()
{
    m_status = Working;

    // Extract time arrays
    Array1DDouble timeArchiveData = m_pTimeArrayArchiveData->GetTimeArray();
    Array1DDouble timeTargetData = m_pTimeArrayTargetData->GetTimeArray();
    Array1DDouble timeTargetSelection = m_pTimeArrayTargetSelection->GetTimeArray();

    // Some other variables
    float tmpscore, thisscore;
    int counter = 0;
    int i_timeTarg, i_timeArch, i_timeTargRelative, i_timeArchRelative;
    int timeArchiveDataSize = timeArchiveData.size();
    int timeTargetDataSize = timeTargetData.size();
    int predictorsNb = m_params.GetPredictorsNb(m_step);
    int analogsNb = m_params.GetAnalogsNumber(m_step);
    bool isasc = (m_criteria[0]->GetOrder()==Asc);

    wxASSERT(m_end<timeTargetSelection.size());
    wxASSERT(timeArchiveDataSize==(int)(m_pPredictorsArchive)[0]->GetData().size());
    wxASSERT(timeTargetDataSize==(int)(m_pPredictorsTarget)[0]->GetData().size());

    // Containers for daily results
    Array1DFloat ScoreArrayOneDay(analogsNb);
    Array1DFloat DateArrayOneDay(analogsNb);

    // DateArray object instantiation. There is one array for all the predictors, as they are aligned, so it picks the predictors we are interested in, but which didn't take place at the same time.
    asTimeArray dateArrayArchiveSelection(m_pTimeArrayArchiveSelection->GetStart(), m_pTimeArrayArchiveSelection->GetEnd(), m_params.GetTimeArrayAnalogsTimeStepHours(), m_params.GetTimeArrayAnalogsMode());
    dateArrayArchiveSelection.SetForbiddenYears(m_pTimeArrayArchiveSelection->GetForbiddenYears());

    // Reset the index start target
    int i_timeTargStart = 0;

    // Loop through every timestep as target data
    // Former, but disabled: for (int i_dateTarg=m_start; !ThreadsManager().Cancelled() && (i_dateTarg<=m_end); i_dateTarg++)
    for (int i_dateTarg=m_start; i_dateTarg<=m_end; i_dateTarg++)
    {
        // Check if the next data is the following. If not, search for it in the array.
        if(timeTargetDataSize>i_timeTargStart+1 && std::abs(timeTargetSelection[i_dateTarg]-timeTargetData[i_timeTargStart+1])<0.01)
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
                m_vTargData[i_ptor] = &(m_pPredictorsTarget)[i_ptor]->GetData()[i_timeTarg];
            }

            // DateArray object initialization.
            dateArrayArchiveSelection.Init(timeTargetSelection[i_dateTarg], m_params.GetTimeArrayAnalogsIntervalDays(), m_params.GetTimeArrayAnalogsExcludeDays());
            dateArrayArchiveSelection.Init(timeTargetSelection[i_dateTarg], m_params.GetTimeArrayAnalogsIntervalDays(), m_params.GetTimeArrayAnalogsExcludeDays());

            // Counter representing the current index
            counter = 0;

            // Reset the index start target
            int i_timeArchStart = 0;

            // Loop through the datearray for candidate data
            for (int i_dateArch=0; i_dateArch<dateArrayArchiveSelection.GetSize(); i_dateArch++)
            {
                // Check if the next data is the following. If not, search for it in the array.
                wxASSERT(timeArchiveData.size()>i_timeArchStart+1);
                wxASSERT(dateArrayArchiveSelection.GetSize()>i_dateArch);
                if(timeArchiveDataSize>i_timeArchStart+1 && std::abs(dateArrayArchiveSelection[i_dateArch]-timeArchiveData[i_timeArchStart+1])<0.01)
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

                    /* Option to remove analogs within the same day (temporal moving winow)
                    or within a given interval. Decreases the quality of the forecast, so
                    the option is disabled. Only implemented here, not in the asProcessor's
                    other options.

                    bool restrictDate = true;
                    int restrictDateMode = 1;

                    if (restrictDate)
                    {
                        if (counter>0)
                        {
                            int indexSameDate = asNOT_FOUND;

                            // 1. removes +- 18h
                            if (restrictDateMode==1)
                            {
                                indexSameDate = asTools::SortedArraySearch(&DateArrayOneDay[0], &DateArrayOneDay[analogsNb-1], (float)timeArchiveData[i_timeArch], (float)0.9);
                            }
                            // 2. removes the date
                            else if (restrictDateMode==2)
                            {
                                float currVal = floor((float)timeArchiveData[i_timeArch]);
                                indexSameDate = asTools::SortedArraySearch(&DateArrayOneDay[0], &DateArrayOneDay[analogsNb-1], currVal, (float)0.01);
                            }
                            else
                            {
                                asLogError(_("Date restriction mode not correctly defined."));
                                return (wxThread::ExitCode)1;
                            }

                            if (indexSameDate!=asNOT_FOUND && indexSameDate!=asOUT_OF_RANGE)
                            {
                                if(ScoreArrayOneDay[indexSameDate]<thisscore)
                                {
                                    // Add score and date to the vectors
                                    ScoreArrayOneDay[indexSameDate] = thisscore;
                                    DateArrayOneDay[indexSameDate] = (float)timeArchiveData[i_timeArch];

                                    // Sort both scores and dates arrays
                                    if (counter>=analogsNb-1)
                                    {
                                        if (isasc)
                                        {
                                            asTools::SortArrays(&ScoreArrayOneDay[0], &ScoreArrayOneDay[analogsNb-1], &DateArrayOneDay[0], &DateArrayOneDay[analogsNb-1], Asc);
                                        } else {
                                            asTools::SortArrays(&ScoreArrayOneDay[0], &ScoreArrayOneDay[analogsNb-1], &DateArrayOneDay[0], &DateArrayOneDay[analogsNb-1], Desc);
                                        }
                                    }
                                }
                            }
                        }
                    }*/

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
                    asLogError(_("The date was not found in the array (Analogs Dates fct, multithreaded option). That should not happen."));
                    asLogError(wxString::Format(_("Start: %g, end: %g, desired value: %g."),
                                                timeArchiveData[i_timeArchStart], timeArchiveData[timeArchiveDataSize-1], dateArrayArchiveSelection[i_dateArch]));
                    return (wxThread::ExitCode)1;
                }
            }

            // Check that the number of occurences are larger than the desired analogs number. If not, set a warning
            if (counter>=analogsNb)
            {
                // Copy results
                m_pFinalAnalogsCriteria->row(i_dateTarg) = ScoreArrayOneDay.transpose();
                m_pFinalAnalogsDates->row(i_dateTarg) = DateArrayOneDay.transpose();
            }
            else
            {
                asLogError(_("There is not enough available data to satisfy the number of analogs:"));
                asLogError(wxString::Format(_("   Analogs number (%d) > counter (%d), date array size (%d) with %d days intervals."),
                                            analogsNb, counter, dateArrayArchiveSelection.GetSize(), m_params.GetTimeArrayAnalogsIntervalDays()));
                asLogError(wxString::Format(_("   Date array start (%.0f), date array end (%.0f), timestep (%.0f), dateTarg (%.0f)."),
                                            timeArchiveData[0], timeArchiveData[timeArchiveDataSize-1], m_params.GetTimeArrayAnalogsTimeStepHours(),
                                            timeTargetSelection[i_timeTarg] ));
                return (wxThread::ExitCode)1;
            }
        }
    }

    m_status = Done;

    return (wxThread::ExitCode)0;
}
