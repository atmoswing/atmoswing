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

#include "asTotalScorePSS.h"

asTotalScorePSS::asTotalScorePSS(const wxString& periodString)
    : asTotalScore(periodString) {}

float asTotalScorePSS::Assess(const a1f& targetDates, const a1f& scores, const asTimeArray& timeArray) const {
    wxASSERT(targetDates.rows() > 1);
    wxASSERT(scores.rows() > 1);

    int countA = 0, countB = 0, countC = 0, countD = 0, countTot = 0;

    switch (m_period) {
        case (asTotalScore::Total): {
            for (int i = 0; i < scores.size(); i++) {
                countTot++;
                if (scores[i] == 1) {
                    countA++;
                } else if (scores[i] == 2) {
                    countB++;
                } else if (scores[i] == 3) {
                    countC++;
                } else if (scores[i] == 4) {
                    countD++;
                } else {
                    wxLogError(_("The PSS score (%f) is not an authorized value."), scores[i]);
                    return NAN;
                }
            }
            break;
        }

        default: {
            throw runtime_error(_("Period not yet implemented in asTotalScorePSS."));
        }
    }

    float score;

    if (countTot > 0) {
        float a = (float)countA;
        float b = (float)countB;
        float c = (float)countC;
        float d = (float)countD;
        if (((a + c) * (b + d)) > 0) {
            score = (a * d - b * c) / ((a + c) * (b + d));
        } else {
            score = 0;
        }
    } else {
        score = NAN;
    }

    return score;
}
