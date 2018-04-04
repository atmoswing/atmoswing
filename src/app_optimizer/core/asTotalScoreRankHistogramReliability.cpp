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

#include "asTotalScoreRankHistogramReliability.h"

asTotalScoreRankHistogramReliability::asTotalScoreRankHistogramReliability(Period period)
        : asTotalScore(period)
{

}

asTotalScoreRankHistogramReliability::asTotalScoreRankHistogramReliability(const wxString &periodString)
        : asTotalScore(periodString)
{

}

asTotalScoreRankHistogramReliability::~asTotalScoreRankHistogramReliability()
{
    //dtor
}

float asTotalScoreRankHistogramReliability::Assess(const a1f &targetDates, const a1f &scores,
                                                   const asTimeArray &timeArray) const
{
    wxLogWarning(_("Calling asTotalScoreRankHistogramReliability::Assess means it doesn't do bootstraping."));

    wxASSERT(targetDates.rows() > 1);
    wxASSERT(scores.rows() > 1);
    wxASSERT(m_ranksNb > 1);

    a1i histogram = a1i::Zero(m_ranksNb);

    switch (m_period) {
        case (asTotalScore::Total): {
            for (int i = 0; i < scores.size(); i++) {
                int rank = (int) asRound(scores[i]);
                wxASSERT(rank <= m_ranksNb);
                histogram[rank - 1]++;
            }
            break;
        }

        default: {
            asThrowException(_("Period not yet implemented in asTotalScoreRankHistogramReliability."));
        }
    }

    // Reference: Candille G., Talagrand O., 2005. Evaluation of probabilistic prediction
    // systems for a scalar variable. Q. J. R. Meteorol. Soc. 131, p. 2131-2150
    float delta = 0;
    float delta_rel = static_cast<float>(scores.size() * (m_ranksNb - 1)) / static_cast<float>(m_ranksNb);
    for (int i = 0; i < m_ranksNb; i++) {
        delta += pow(static_cast<float>(histogram[i]) - static_cast<float>(scores.size()) / static_cast<float>(m_ranksNb), 2.0f);
    }

    float reliability = delta / delta_rel;

    return reliability;
}

float asTotalScoreRankHistogramReliability::AssessOnBootstrap(a1f &histogramPercent, int scoresSize) const
{
    wxASSERT(m_ranksNb > 1);

    a1f histogramReal;
    histogramReal = scoresSize * histogramPercent / 100.0f;

    // Reference: Candille G., Talagrand O., 2005. Evaluation of probabilistic prediction
    // systems for a scalar variable. Q. J. R. Meteorol. Soc. 131, p. 2131-2150
    float delta = 0;
    float delta_rel = static_cast<float>(scoresSize * (m_ranksNb - 1)) / static_cast<float>(m_ranksNb);
    for (int i = 0; i < m_ranksNb; i++) {
        delta += pow(static_cast<float>(histogramReal[i]) - static_cast<float>(scoresSize) / static_cast<float>(m_ranksNb), 2.0f);
    }

    float reliability = delta / delta_rel;

    return reliability;
}
