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

#include "asTotalScoreRankHistogram.h"

asTotalScoreRankHistogram::asTotalScoreRankHistogram(const wxString& periodString)
    : asTotalScore(periodString) {
    m_singleValue = false;
}

float asTotalScoreRankHistogram::Assess(const a1f& targetDates, const a1f& scores, const asTimeArray& timeArray) const {
    wxLogError(_("The rank histogram cannot provide a single score value !"));
    return NaNf;
}

a1f asTotalScoreRankHistogram::AssessOnArray(const a1f& targetDates, const a1f& scores,
                                             const asTimeArray& timeArray) const {
    wxASSERT(targetDates.rows() > 1);
    wxASSERT(scores.rows() > 1);
    wxASSERT(m_ranksNb > 1);

    a1i histogram = a1i::Zero(m_ranksNb);
    int countTot = 0;

    switch (m_period) {
        case (asTotalScore::Total): {
            for (int i = 0; i < scores.size(); i++) {
                countTot++;

                int rank = (int)asRound(scores[i]);
                wxASSERT(rank <= m_ranksNb);
                histogram[rank - 1]++;
            }
            break;
        }

        default: {
            asThrowException(_("Period not yet implemented in asTotalScoreRankHistogram."));
        }
    }

    // Process percentages
    a1f histogramPercent = a1f::Zero(m_ranksNb);

    if (countTot <= 0) {
        wxLogError(_("Error processing the final rank histogram."));
        return histogramPercent;
    }

    for (int i = 0; i < m_ranksNb; i++) {
        histogramPercent[i] = float(100 * histogram[i]) / float(countTot);
    }

    return histogramPercent;
}
