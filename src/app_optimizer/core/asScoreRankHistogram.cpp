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

#include "asScoreRankHistogram.h"

asScoreRankHistogram::asScoreRankHistogram()
    : asScore(asScore::RankHistogram, _("Rank Histogram"), _("Verification Rank Histogram (Talagrand Diagram)"), Asc,
              NAN, NAN) {}

asScoreRankHistogram::~asScoreRankHistogram() {}

float asScoreRankHistogram::Assess(float obs, const a1f& values, int nbElements) const {
    wxASSERT(values.size() > 1);
    wxASSERT(nbElements > 0);

    // Check inputs
    if (!CheckObservedValue(obs)) {
        return NAN;
    }
    if (!CheckVectorLength(values, nbElements)) {
        wxLogWarning(_("Problems in a vector length."));
        return NAN;
    }

    // Create the container to sort the data
    a1f x = values;

    // NaNs are not allowed as it messes up the ranks
    if (asHasNaN(&x[0], &x[nbElements - 1]) || isnan(obs)) {
        wxLogError(_("NaNs were found in the Rank Histogram processing function. Cannot continue."));
        return NAN;
    }

    // Sort the forcast array
    asSortArray(&x[0], &x[nbElements - 1], Asc);

    // Get rank
    if (obs < x[0]) {
        return 1;
    } else if (obs > x[nbElements - 1]) {
        return nbElements + 1;
    } else {
        // Check if exact value can be found
        int indExact = asFind(&x[0], &x[nbElements - 1], obs, 0.0, asHIDE_WARNINGS);

        if (indExact != asOUT_OF_RANGE && indExact != asNOT_FOUND) {
            // If the exact value was found in the analogs
            // See: Hamill, T.M., and S.J. Colucci, 1997. Verification of Eta–RSM short-range ensemble
            // forecasts. Monthly Weather Review, 125, 1312-1327.
            // Hamill, T.M., and S.J. Colucci, 1998. Evaluation of Eta–RSM ensemble probabilistic
            // precipitation forecasts. Monthly Weather Review, 126, 711-724.

            // Find first occurrence
            int indFirst = 0;
            while (indFirst < nbElements && x[indFirst] < obs) {
                indFirst++;
            }

            // Count the number of same values
            int m = 1;
            while (indFirst + m < nbElements && x[indFirst + m] == obs) {
                m++;
            }

            // Generate uniform random deviates
            float verif = asRandom(0.0, 1.0);
            a1f rand(m);
            for (int i = 0; i < m; i++) {
                rand[i] = asRandom(0.0, 1.0);
            }

            // Assign rank
            if (m == 1) {
                if (verif < rand[0]) {
                    return indFirst + 1;
                } else {
                    return indFirst + 2;
                }
            } else {
                asSortArray(&rand[0], &rand[m - 1], Asc);
                int subIndex;

                if (verif < rand[0]) {
                    subIndex = 0;
                } else if (verif > rand[m - 1]) {
                    subIndex = m;
                } else {
                    subIndex = 1 + asFindFloor(&rand[0], &rand[m - 1], verif);
                }

                return indFirst + 1 + subIndex;
            }

        } else {
            // Indices for the left
            int indLeft = asFindFloor(&x[0], &x[nbElements - 1], obs);
            wxASSERT(indLeft >= 0);

            return indLeft + 2;  // as the indices are 0-based + element on the left side
        }
    }
}

bool asScoreRankHistogram::ProcessScoreClimatology(const a1f& refVals, const a1f& climatologyData) {
    return true;
}
