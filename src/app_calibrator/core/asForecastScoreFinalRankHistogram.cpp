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
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2014 Pascal Horton, Terr@num.
 */
 
#include "asForecastScoreFinalRankHistogram.h"

asForecastScoreFinalRankHistogram::asForecastScoreFinalRankHistogram(Period period)
:
asForecastScoreFinal(period)
{
    m_SingleValue = false;
}

asForecastScoreFinalRankHistogram::asForecastScoreFinalRankHistogram(const wxString& periodString)
:
asForecastScoreFinal(periodString)
{
    m_SingleValue = false;
}

asForecastScoreFinalRankHistogram::~asForecastScoreFinalRankHistogram()
{
    //dtor
}

float asForecastScoreFinalRankHistogram::Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray)
{
    asLogError(_("The rank histogram cannot provide a single score value !"));
    return NaNFloat;
}

Array1DFloat asForecastScoreFinalRankHistogram::AssessOnArray(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray)
{
    wxASSERT(targetDates.rows()>1);
    wxASSERT(forecastScores.rows()>1);
    wxASSERT(m_RanksNb>1);

    Array1DInt histogram = Array1DInt::Zero(m_RanksNb);
    int countTot = 0;

    switch (m_Period)
    {
        case (asForecastScoreFinal::Total):
        {
            for (int i=0; i<forecastScores.size(); i++)
            {
                countTot++;

                int rank = (int)asTools::Round(forecastScores[i]);
                wxASSERT(rank<=m_RanksNb);
                histogram[rank-1]++;
            }
            break;
        }

        default:
        {
            asThrowException(_("Period not yet implemented in asForecastScoreFinalRankHistogram."));
        }
    }

    // Process percentages
    Array1DFloat histogramPercent = Array1DFloat::Zero(m_RanksNb);

    for (int i=0; i<m_RanksNb; i++)
    {
        histogramPercent[i] = float(100*histogram[i])/float(countTot);
    }

    return histogramPercent;
}
