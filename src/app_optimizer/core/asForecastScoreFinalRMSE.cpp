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

#include "asForecastScoreFinalRMSE.h"

asForecastScoreFinalRMSE::asForecastScoreFinalRMSE(Period period)
        : asForecastScoreFinal(period)
{

}

asForecastScoreFinalRMSE::asForecastScoreFinalRMSE(const wxString &periodString)
        : asForecastScoreFinal(periodString)
{

}

asForecastScoreFinalRMSE::~asForecastScoreFinalRMSE()
{
    //dtor
}

float asForecastScoreFinalRMSE::Assess(const a1f &targetDates, const a1f &forecastScores, const asTimeArray &timeArray) const
{
    wxASSERT(targetDates.rows() > 1);
    wxASSERT(forecastScores.rows() > 1);

    switch (m_period) {
        case (asForecastScoreFinal::Total): {
            int targetDatesLength = targetDates.rows();

            // Loop through the targetDates
            float score = 0, divisor = 0;

            for (int iTime = 0; iTime < targetDatesLength; iTime++) {
                if (!asTools::IsNaN(forecastScores(iTime))) {
                    score += forecastScores(iTime);
                    divisor++;
                }
            }

            score = sqrt(score / divisor);
            return score;
        }

        case (asForecastScoreFinal::SpecificPeriod): {
            int targetDatesLength = targetDates.rows();
            int timeArrayLength = timeArray.GetSize();

            // Get first and last common days
            double FirstDay = wxMax((double) targetDates[0], timeArray.GetFirst());
            double LastDay = wxMin((double) targetDates[targetDatesLength - 1], timeArray.GetLast());
            a1d DateTime = timeArray.GetTimeArray();
            int IndexStart = asTools::SortedArraySearchClosest(&DateTime(0), &DateTime(timeArrayLength - 1), FirstDay);
            int IndexEnd = asTools::SortedArraySearchClosest(&DateTime(0), &DateTime(timeArrayLength - 1), LastDay);

            // Loop through the timeArray
            float score = 0, divisor = 0;

            for (int iTime = IndexStart; iTime <= IndexEnd; iTime++) {
                int indexCurrent = asTools::SortedArraySearchClosest(&targetDates(0),
                                                                     &targetDates(targetDatesLength - 1),
                                                                     DateTime(iTime));
                if ((indexCurrent != asNOT_FOUND) & (indexCurrent != asOUT_OF_RANGE)) {
                    if (!asTools::IsNaN(forecastScores(indexCurrent))) {
                        score += forecastScores(indexCurrent);
                        divisor++;
                    }
                }
            }

            score = sqrt(score / divisor);
            return score;
        }

        default: {
            asThrowException(_("Period not yet implemented in asForecastScoreFinalRMSE."));
        }
    }
}
