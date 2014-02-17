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
 
#include "asForecastScoreRankHistogram.h"

asForecastScoreRankHistogram::asForecastScoreRankHistogram()
:
asForecastScore()
{
    m_Score = asForecastScore::RankHistogram;
    m_Name = _("Rank Histogram");
    m_FullName = _("Verification Rank Histogram (Talagrand Diagram)");
    m_Order = NoOrder;
    m_ScaleBest = NaNFloat;
    m_ScaleWorst = NaNFloat;
}

asForecastScoreRankHistogram::~asForecastScoreRankHistogram()
{
    //dtor
}

float asForecastScoreRankHistogram::Assess(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements)
{
    wxASSERT(ForcastVals.size()>1);
    wxASSERT(nbElements>0);

    // Create the container to sort the data
    Array1DFloat x = ForcastVals;

    // NaNs are not allowed as it messes up the ranks
    if (asTools::HasNaN(&x[0], &x[nbElements-1]) || asTools::IsNaN(ObservedVal)) {
        asLogError(_("NaNs were found in the Rank Histogram processing function. Cannot continue."));
        return NaNFloat;
    }

    // Sort the forcast array
    asTools::SortArray(&x[0], &x[nbElements-1], Asc);

    // Get rank
    if (ObservedVal<x[0]) {
        return 1;
    }
    else if (ObservedVal>x[nbElements-1]) {
        return nbElements+1;
    }
    else {
        // Indices for the left and right part
        int indLeft = asTools::SortedArraySearchFloor(&x[0], &x[nbElements-1], ObservedVal);
        int indRight = asTools::SortedArraySearchCeil(&x[0], &x[nbElements-1], ObservedVal);
        wxASSERT(indLeft>=0);
        wxASSERT(indRight>=0);
        wxASSERT(indLeft<=indRight);

        int rankVal = indLeft+1; // as the indices are 0-based

        // If the exact value was found in the analogs
        // See: Hamill, T.M., and S.J. Colucci, 1997. Verification of Eta�RSM short-range ensemble
        // forecasts. Monthly Weather Review, 125, 1312�1327.
        // Hamill, T.M., and S.J. Colucci, 1998. Evaluation of Eta�RSM ensemble probabilistic
        // precipitation forecasts. Monthly Weather Review, 126, 711�724.
        if (x[indLeft]==ObservedVal) {

            // Count the number of same values
            int m=1;
            while (indLeft+m<nbElements && x[indLeft+m]==ObservedVal) {
                m++;
            }

            // Generate uniform random deviates
            float verif = asTools::Random(0.0, 1.0);
            Array1DFloat rand(m);
            for (int i=0; i<m; i++) {
                rand[i] = asTools::Random(0.0, 1.0);
            }

            // Assign rank
            if (m==1) {
                if (verif<rand[0])
                {
                    return rankVal;
                }
                else
                {
                    return rankVal+1;
                }
            }
            else {
                asTools::SortArray(&rand[0], &rand[m-1], Asc);
                int subIndex;

                if (verif<rand[0]) {
                    subIndex = 0;
                }
                else if (verif>rand[m-1]) {
                    subIndex = m;
                }
                else {
                    subIndex = asTools::SortedArraySearchFloor(&rand[0], &rand[m-1], verif);
                }

                return rankVal+subIndex;
            }
        }

        return rankVal+1;
    }

}

bool asForecastScoreRankHistogram::ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData)
{
    return true;
}
