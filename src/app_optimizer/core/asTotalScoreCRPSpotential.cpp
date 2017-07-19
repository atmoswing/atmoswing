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
 * Portions Copyright 2014 Renaud Marty, DREAL.
 */

#include "asTotalScoreCRPSpotential.h"

asTotalScoreCRPSpotential::asTotalScoreCRPSpotential(Period period)
        : asTotalScore(period)
{
    m_has2DArrayArgument = true;
}

asTotalScoreCRPSpotential::asTotalScoreCRPSpotential(const wxString &periodString)
        : asTotalScore(periodString)
{
    m_has2DArrayArgument = true;
}

asTotalScoreCRPSpotential::~asTotalScoreCRPSpotential()
{
    //dtor
}

float asTotalScoreCRPSpotential::Assess(const a1f &targetDates, const a2f &scores, const asTimeArray &timeArray) const
{
    wxASSERT(targetDates.rows() > 1);
    wxASSERT(scores.rows() > 1);
    wxASSERT(scores.cols() > 1);

    // Process average on every column
    a1f means = scores.colwise().mean();

    // Extract corresponding arrays
    int binsNbs = means.size() / 3;
    a1f alpha = means.segment(0, binsNbs);
    a1f beta = means.segment(binsNbs, binsNbs);
    a1f g = means.segment(2 * binsNbs, binsNbs);

    // Compute o (coefficent-wise operations)
    a1f o = beta / (alpha + beta);
    wxASSERT(o.size() == alpha.size());

    // Create the p array (coefficent-wise operations)
    a1f p = a1f::LinSpaced(binsNbs, 0, binsNbs - 1);
    p = p / (binsNbs - 1);

    // Compute CRPS potential
    float potential = 0;

    for (int i = 0; i < binsNbs; i++) {
        if (!asTools::IsNaN(g[i]) && !asTools::IsInf(g[i]) && !asTools::IsNaN(o[i]) && !asTools::IsInf(o[i])) {
            potential += g[i] * o[i] * (1 - o[i]);
        }
    }

    return potential;
}
