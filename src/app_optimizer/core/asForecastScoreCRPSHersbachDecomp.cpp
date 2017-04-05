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

#include "asForecastScoreCRPSHersbachDecomp.h"

asForecastScoreCRPSHersbachDecomp::asForecastScoreCRPSHersbachDecomp()
        : asForecastScore()
{
    m_score = asForecastScore::CRPSHersbachDecomp;
    m_name = _("CRPS Hersbach decomposition");
    m_fullName = _("Hersbach decomposition of the Continuous Ranked Probability Score (Hersbach, 2000)");
    m_order = Asc;
    m_scaleBest = 0;
    m_scaleWorst = NaNf;
    m_singleValue = false;
}

asForecastScoreCRPSHersbachDecomp::~asForecastScoreCRPSHersbachDecomp()
{
    //dtor
}

float asForecastScoreCRPSHersbachDecomp::Assess(float ObservedVal, const a1f &ForcastVals, int nbElements) const
{
    wxLogError(_("The Hersbach decomposition of the CRPS cannot provide a single score value !"));
    return NaNf;
}

a1f asForecastScoreCRPSHersbachDecomp::AssessOnArray(float ObservedVal, const a1f &ForcastVals,
                                                              int nbElements) const
{
    wxASSERT(ForcastVals.size() > 1);
    wxASSERT(nbElements > 0);

    // Check the element numbers vs vector length and the observed value
    if (!CheckInputs(ObservedVal, ForcastVals, nbElements)) {
        wxLogWarning(_("The inputs are not conform in the CRPS Hersbach decomposition function"));
        return a2f();
    }

    // Create the container to sort the data
    a1f x = ForcastVals;

    // NaNs are not allowed as it messes up the ranks
    if (asTools::HasNaN(&x[0], &x[nbElements - 1]) || asTools::IsNaN(ObservedVal)) {
        wxLogError(_("NaNs were found in the CRPS Hersbach decomposition processing function. Cannot continue."));
        return a2f();
    }

    // Sort the forcast array
    asTools::SortArray(&x[0], &x[nbElements - 1], Asc);

    // Containers
    int binsNbs = nbElements + 1;
    a1f alpha = a1f::Zero(binsNbs);
    a1f beta = a1f::Zero(binsNbs);
    a1f g = a1f::Zero(binsNbs);

    // Predictive sampling completed by 0 and N+1 elements
    int binsNbsExtra = nbElements + 2;
    a1f z = a1f::Zero(binsNbsExtra);
    z[0] = x[0];
    z.segment(1, nbElements) = x;
    z[binsNbsExtra - 1] = x[nbElements - 1];

    if (ObservedVal < z[0]) {
        z[0] = ObservedVal;
    }

    if (ObservedVal > z[binsNbsExtra - 1]) {
        z[binsNbsExtra - 1] = ObservedVal;
    }

    // Loop on bins (Hersbach, Eq 26)
    for (int k = 0; k < binsNbs; k++) {
        g[k] = z[k + 1] - z[k];
        if (ObservedVal > z(k + 1)) {
            alpha[k] = g[k];
            beta[k] = 0;
        } else if ((ObservedVal <= z[k + 1]) && (ObservedVal >= z[k])) {
            alpha[k] = ObservedVal - z[k];
            beta[k] = z[k + 1] - ObservedVal;
        } else {
            alpha[k] = 0;
            beta[k] = g[k];
        }
    }

    // Outliers cases (Hersbach, Eq 27)
    if (ObservedVal == z[0]) {
        alpha = a1f::Zero(binsNbs);
        beta[0] = z[1] - ObservedVal;
    } else if (ObservedVal == z[binsNbsExtra - 1]) {
        alpha[binsNbs - 1] = ObservedVal - z[binsNbs - 1];
        beta = a1f::Zero(binsNbs);
    }

    // Concatenate the results
    a1f result(3 * binsNbs);
    result.segment(0, binsNbs) = alpha;
    result.segment(binsNbs, binsNbs) = beta;
    result.segment(2 * binsNbs, binsNbs) = g;

    return result;
}

bool asForecastScoreCRPSHersbachDecomp::ProcessScoreClimatology(const a1f &refVals,
                                                                const a1f &climatologyData)
{
    return true;
}
