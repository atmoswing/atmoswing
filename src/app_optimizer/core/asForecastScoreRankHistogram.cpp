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

#include "asForecastScoreRankHistogram.h"

asForecastScoreRankHistogram::asForecastScoreRankHistogram()
        : asForecastScore()
{
    m_score = asForecastScore::RankHistogram;
    m_name = _("Rank Histogram");
    m_fullName = _("Verification Rank Histogram (Talagrand Diagram)");
    m_scaleBest = NaNf;
    m_scaleWorst = NaNf;
}

asForecastScoreRankHistogram::~asForecastScoreRankHistogram()
{
    //dtor
}

float asForecastScoreRankHistogram::Assess(float ObservedVal, const a1f &ForcastVals, int nbElements) const
{
    wxASSERT(ForcastVals.size() > 1);
    wxASSERT(nbElements > 0);

    // Create the container to sort the data
    a1f x = ForcastVals;

    // NaNs are not allowed as it messes up the ranks
    if (asTools::HasNaN(&x[0], &x[nbElements - 1]) || asTools::IsNaN(ObservedVal)) {
        wxLogError(_("NaNs were found in the Rank Histogram processing function. Cannot continue."));
        return NaNf;
    }

    // Sort the forcast array
    asTools::SortArray(&x[0], &x[nbElements - 1], Asc);

    // Get rank
    if (ObservedVal < x[0]) {
        return 1;
    } else if (ObservedVal > x[nbElements - 1]) {
        return nbElements + 1;
    } else {
        // Check if exact value can be found
        int indExact = asTools::SortedArraySearch(&x[0], &x[nbElements - 1], ObservedVal, 0.0, asHIDE_WARNINGS);

        if (indExact != asOUT_OF_RANGE && indExact != asNOT_FOUND) {
            // If the exact value was found in the analogs
            // See: Hamill, T.M., and S.J. Colucci, 1997. Verification of Eta–RSM short-range ensemble
            // forecasts. Monthly Weather Review, 125, 1312-1327.
            // Hamill, T.M., and S.J. Colucci, 1998. Evaluation of Eta–RSM ensemble probabilistic
            // precipitation forecasts. Monthly Weather Review, 126, 711-724.

            // Find first occurrence
            int indFirst = 0;
            while (indFirst < nbElements && x[indFirst] < ObservedVal) {
                indFirst++;
            }

            // Count the number of same values
            int m = 1;
            while (indFirst + m < nbElements && x[indFirst + m] == ObservedVal) {
                m++;
            }

            // Generate uniform random deviates
            float verif = asTools::Random(0.0, 1.0);
            a1f rand(m);
            for (int i = 0; i < m; i++) {
                rand[i] = asTools::Random(0.0, 1.0);
            }

            // Assign rank
            if (m == 1) {
                if (verif < rand[0]) {
                    return indFirst + 1;
                } else {
                    return indFirst + 2;
                }
            } else {
                asTools::SortArray(&rand[0], &rand[m - 1], Asc);
                int subIndex;

                if (verif < rand[0]) {
                    subIndex = 0;
                } else if (verif > rand[m - 1]) {
                    subIndex = m;
                } else {
                    subIndex = 1 + asTools::SortedArraySearchFloor(&rand[0], &rand[m - 1], verif);
                }

                return indFirst + 1 + subIndex;
            }

        } else {
            // Indices for the left
            int indLeft = asTools::SortedArraySearchFloor(&x[0], &x[nbElements - 1], ObservedVal);
            wxASSERT(indLeft >= 0);

            return indLeft + 2; // as the indices are 0-based + element on the left side
        }
    }
}

bool asForecastScoreRankHistogram::ProcessScoreClimatology(const a1f &refVals,
                                                           const a1f &climatologyData)
{
    return true;
}
