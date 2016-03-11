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
 
#include "asForecastScoreFinalMean.h"

asForecastScoreFinalMean::asForecastScoreFinalMean(Period period)
:
asForecastScoreFinal(period)
{

}

asForecastScoreFinalMean::asForecastScoreFinalMean(const wxString& periodString)
:
asForecastScoreFinal(periodString)
{

}

asForecastScoreFinalMean::~asForecastScoreFinalMean()
{
    //dtor
}

float asForecastScoreFinalMean::Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray)
{
    wxASSERT(targetDates.rows()>1);
    wxASSERT(forecastScores.rows()>1);

    switch (m_period)
    {
        case (asForecastScoreFinal::Total):
        {
            int targetDatesLength = targetDates.rows();

            // Loop through the targetDates
            float score = 0, divisor = 0;

            for (int i_time=0; i_time<targetDatesLength; i_time++)
            {
                if(!asTools::IsNaN(forecastScores(i_time)))
                {
                    score += forecastScores(i_time);
                    divisor++;
                }
            }

            return (score/divisor);
        }

        case (asForecastScoreFinal::SpecificPeriod):
        {
            int targetDatesLength = targetDates.rows();
            int timeArrayLength = timeArray.GetSize();

            // Get first and last common days
            double FirstDay = wxMax((double)targetDates[0], timeArray.GetFirst());
            double LastDay = wxMin((double)targetDates[targetDatesLength-1], timeArray.GetLast());
            Array1DDouble DateTime = timeArray.GetTimeArray();
            int IndexStart = asTools::SortedArraySearchClosest(&DateTime(0), &DateTime(timeArrayLength-1), FirstDay);
            int IndexEnd = asTools::SortedArraySearchClosest(&DateTime(0), &DateTime(timeArrayLength-1), LastDay);

            // Loop through the timeArray
            float score = 0, divisor = 0;

            for (int i_time=IndexStart; i_time<=IndexEnd; i_time++)
            {
                int indexCurrent = asTools::SortedArraySearchClosest(&targetDates(0), &targetDates(targetDatesLength-1), DateTime(i_time));
                if((indexCurrent!=asNOT_FOUND) & (indexCurrent!=asOUT_OF_RANGE))
                {
                    if(!asTools::IsNaN(forecastScores(indexCurrent)))
                    {
                        score += forecastScores(indexCurrent);
                        divisor++;
                    }
                }
            }

            return (score/divisor);
        }

        default:
        {
            asThrowException(_("Period not yet implemented in asForecastScoreFinalMean."));
        }
    }
}
