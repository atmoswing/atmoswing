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

#include "asForecastScoreDF0.h"

asForecastScoreDF0::asForecastScoreDF0()
        : asForecastScore()
{
    m_score = asForecastScore::DF0;
    m_name = _("Difference of F(0)");
    m_fullName = _("Absolute difference of the frequency of null precipitations.");
    m_order = Asc;
    m_scaleBest = 0;
    m_scaleWorst = NaNf;
}

asForecastScoreDF0::~asForecastScoreDF0()
{
    //dtor
}

float asForecastScoreDF0::Assess(float ObservedVal, const a1f &ForcastVals, int nbElements) const
{
    wxASSERT(ForcastVals.size() > 1);
    wxASSERT(nbElements > 0);

    // Check the element numbers vs vector length and the observed value
    if (!CheckInputs(ObservedVal, ForcastVals, nbElements)) {
        wxLogWarning(_("The inputs are not conform in the DF0 processing function"));
        return NaNf;
    }

    // Create the container to sort the data
    a1f x(nbElements);
    float xObs = ObservedVal;

    // Remove the NaNs and copy content
    int nbForecasts = CleanNans(ForcastVals, x, nbElements);
    if (nbForecasts == asNOT_FOUND) {
        wxLogWarning(_("Only NaNs as inputs in the DF0 processing function."));
        return NaNf;
    } else if (nbForecasts <= 2) {
        wxLogWarning(_("Not enough elements to process the DF0."));
        return NaNf;
    }

    // Sort the forcast array
    asTools::SortArray(&x[0], &x[nbForecasts - 1], Asc);

    float score = 0;

    // Cumulative frequency
    a1f F = asTools::GetCumulativeFrequency(nbForecasts);

    // Identify the last 0
    int indexLastZero = -1;
    for (int i = 0; i < nbElements; i++) {
        if (x[i] == 0) {
            indexLastZero = i;
        }
    }

    // Display F(0)analog
    bool dispF0 = false;
    if (dispF0) {
        if (indexLastZero >= 0) {
            wxLogWarning("%f", F(indexLastZero));
        } else {
            wxLogWarning("%d", 0);
        }
    }

    // Find FxObs, fix xObs and integrate beyond limits
    float FxObs;
    if (xObs > 0.0) // If precipitation
    {
        FxObs = 1;
    } else {
        FxObs = 0;
    }

    score = std::abs((1.0f - F(indexLastZero)) - FxObs);

    return score;
}

bool asForecastScoreDF0::ProcessScoreClimatology(const a1f &refVals, const a1f &climatologyData)
{
    return true;
}
