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

#include "asTotalScoreMean.h"

asTotalScoreMean::asTotalScoreMean(const wxString& periodString)
    : asTotalScore(periodString) {}

float asTotalScoreMean::Assess(const a1f& targetDates, const a1f& scores, const asTimeArray& timeArray) const {
    wxASSERT(targetDates.rows() > 1);
    wxASSERT(scores.rows() > 1);

    switch (m_period) {
        case (asTotalScore::Total): {
            int targetDatesLength = targetDates.rows();

            // Loop through the targetDates
            float score = 0, divisor = 0;

            for (int iTime = 0; iTime < targetDatesLength; iTime++) {
                if (!asIsNaN(scores(iTime))) {
                    score += scores(iTime);
                    divisor++;
                }
            }

            return (score / divisor);
        }

        case (asTotalScore::SpecificPeriod): {
            int targetDatesLength = targetDates.rows();
            int timeArrayLength = timeArray.GetSize();

            // Get first and last common days
            double firstDay = wxMax((double)targetDates[0], timeArray.GetFirst());
            double lastDay = wxMin((double)targetDates[targetDatesLength - 1], timeArray.GetLast());
            a1d dateTime = timeArray.GetTimeArray();
            int indexStart = asFindClosest(&dateTime(0), &dateTime(timeArrayLength - 1), firstDay);
            int indexEnd = asFindClosest(&dateTime(0), &dateTime(timeArrayLength - 1), lastDay);

            // Loop through the timeArray
            float score = 0, divisor = 0;

            for (int iTime = indexStart; iTime <= indexEnd; iTime++) {
                if (iTime < 0) {
                    wxLogError(_("Error processing the final mean score."));
                    return NaNf;
                }
                int indexCurrent = asFindClosest(&targetDates(0), &targetDates(targetDatesLength - 1), dateTime(iTime));
                if ((indexCurrent != asNOT_FOUND) & (indexCurrent != asOUT_OF_RANGE)) {
                    if (!asIsNaN(scores(indexCurrent))) {
                        score += scores(indexCurrent);
                        divisor++;
                    }
                }
            }

            return (score / divisor);
        }

        default: {
            throw runtime_error(_("Period not yet implemented in asTotalScoreMean."));
        }
    }
}
