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
 * Portions Copyright 2014-2015 Pascal Horton, Terranum.
 */

#include "asForecastScoreFinalRankHistogramReliability.h"

asForecastScoreFinalRankHistogramReliability::asForecastScoreFinalRankHistogramReliability(Period period)
:
asForecastScoreFinal(period)
{

}

asForecastScoreFinalRankHistogramReliability::asForecastScoreFinalRankHistogramReliability(const wxString& periodString)
:
asForecastScoreFinal(periodString)
{

}

asForecastScoreFinalRankHistogramReliability::~asForecastScoreFinalRankHistogramReliability()
{
    //dtor
}

float asForecastScoreFinalRankHistogramReliability::Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray)
{
    asLogWarning(_("Calling asForecastScoreFinalRankHistogramReliability::Assess means it doesn't do bootstraping."));

    wxASSERT(targetDates.rows()>1);
    wxASSERT(forecastScores.rows()>1);
    wxASSERT(m_ranksNb>1);

    Array1DInt histogram = Array1DInt::Zero(m_ranksNb);

    switch (m_period)
    {
        case (asForecastScoreFinal::Total):
        {
            for (int i=0; i<forecastScores.size(); i++)
            {
                int rank = (int)asTools::Round(forecastScores[i]);
                wxASSERT(rank<=m_ranksNb);
                histogram[rank-1]++;
            }
            break;
        }

        default:
        {
            asThrowException(_("Period not yet implemented in asForecastScoreFinalRankHistogramReliability."));
        }
    }

    // Reference: Candille G., Talagrand O., 2005. Evaluation of probabilistic prediction
    // systems for a scalar variable. Q. J. R. Meteorol. Soc. 131, p. 2131-2150
    float delta = 0;
    float delta_rel = float(forecastScores.size()*(m_ranksNb-1))/float(m_ranksNb);
    for (int i=0; i<m_ranksNb; i++)
    {
        delta += pow(float(histogram[i]) - float(forecastScores.size())/float(m_ranksNb), 2.0f);
    }

    float reliability = delta/delta_rel;

    return reliability;
}

float asForecastScoreFinalRankHistogramReliability::AssessOnBootstrap(Array1DFloat &histogramPercent, int forecastScoresSize)
{
    wxASSERT(m_ranksNb>1);

    Array1DFloat histogramReal;
    histogramReal = forecastScoresSize*histogramPercent/100.0f;

    // Reference: Candille G., Talagrand O., 2005. Evaluation of probabilistic prediction
    // systems for a scalar variable. Q. J. R. Meteorol. Soc. 131, p. 2131-2150
    float delta = 0;
    float delta_rel = float(forecastScoresSize*(m_ranksNb-1))/float(m_ranksNb);
    for (int i=0; i<m_ranksNb; i++)
    {
        delta += pow(float(histogramReal[i]) - float(forecastScoresSize)/float(m_ranksNb), 2.0f);
    }

    float reliability = delta/delta_rel;

    return reliability;
}
