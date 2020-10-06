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

#include "asScoreCRPSEP.h"

asScoreCRPSEP::asScoreCRPSEP()
    : asScore(asScore::CRPSEP, _("CRPS Exact Primitive"), _("Continuous Ranked Probability Score exact solution"), Asc,
              0, NaNf) {}

asScoreCRPSEP::~asScoreCRPSEP() {}

float asScoreCRPSEP::Assess(float obs, const a1f &values, int nbElements) const {
    wxASSERT(values.size() > 1);
    wxASSERT(nbElements > 0);

    // Check inputs
    if (!CheckObservedValue(obs)) {
        return NaNf;
    }
    if (!CheckVectorLength(values, nbElements)) {
        wxLogWarning(_("Problems in a vector length."));
        return NaNf;
    }

    // Create the container to sort the data
    a1f x(nbElements);
    float xObs = obs;

    // Remove the NaNs and copy content
    int nbPredict = CleanNans(values, x, nbElements);
    if (nbPredict == asNOT_FOUND) {
        wxLogWarning(_("Only NaNs as inputs in the CRPS processing function"));
        return NaNf;
    }

    // Sort the forcast array
    asSortArray(&x[0], &x[nbPredict - 1], Asc);

    float crps = 0;

    // Cumulative frequency
    a1f F = asGetCumulativeFrequency(nbPredict);

    float df, dVal;

    // Indices for the left and right part (according to xObs) of the distribution
    int indLeftStart = 0;
    int indLeftEnd = 0;
    int indRightStart = nbPredict - 1;
    int indRightEnd = nbPredict - 1;

    // Find FxObs, fix xObs and integrate beyond limits
    if (xObs <= x[0])  // If xObs before the distribution
    {
        indRightStart = 0;
        crps += x[indRightStart] - xObs;
    } else if (xObs > x[nbPredict - 1])  // If xObs after the distribution
    {
        indLeftEnd = nbPredict - 1;
        crps += xObs - x[indLeftEnd];
    } else  // If xObs inside the distribution
    {
        indLeftEnd = asFindFloor(&x[0], &x[nbPredict - 1], xObs);
        if ((indLeftEnd != nbPredict - 1) & (indLeftEnd != asNOT_FOUND) & (indLeftEnd != asOUT_OF_RANGE)) {
            indRightStart = indLeftEnd + 1;
            float FxObs;
            if (x(indRightStart) == x(indLeftEnd)) {
                FxObs = (F(indLeftEnd) + F(indRightStart)) * 0.5;
            } else {
                FxObs = F(indLeftEnd) + (F(indRightStart) - F(indLeftEnd)) * (xObs - x(indLeftEnd)) /
                                            (x(indRightStart) - x(indLeftEnd));
            }

            // Integrate the CRPS around FxObs
            // First part - from x(indLeftEnd) to xobs
            df = FxObs - F(indLeftEnd);
            dVal = xObs - x(indLeftEnd);
            if (dVal != 0) {
                float a = df / dVal;
                float b = -x(indLeftEnd) * a + F(indLeftEnd);
                crps += (a * a / 3) * (xObs * xObs * xObs - x(indLeftEnd) * x(indLeftEnd) * x(indLeftEnd)) +
                        (a * b) * (xObs * xObs - x(indLeftEnd) * x(indLeftEnd)) + (b * b) * (xObs - x(indLeftEnd));
            }

            // Second part - from xobs to x(indRightStart)
            df = F(indRightStart) - FxObs;
            dVal = x(indRightStart) - xObs;
            if (dVal != 0) {
                float a = -df / dVal;
                float b = -xObs * (-a) + FxObs;
                b = 1 - b;
                crps += (a * a / 3) * (x(indRightStart) * x(indRightStart) * x(indRightStart) - xObs * xObs * xObs) +
                        (a * b) * (x(indRightStart) * x(indRightStart) - xObs * xObs) +
                        (b * b) * (x(indRightStart) - xObs);
            }
        }
    }

    // Integrate on the left part
    for (int i = indLeftStart; i < indLeftEnd; i++) {
        df = F(i + 1) - F(i);
        dVal = x(i + 1) - x(i);
        if (dVal != 0) {
            // Build a line y=ax+b
            float a = df / dVal;
            float b = -x(i) * a + F(i);

            // CRPS after integration with H=0
            crps += (a * a / 3) * (x(i + 1) * x(i + 1) * x(i + 1) - x(i) * x(i) * x(i)) +
                    (a * b) * (x(i + 1) * x(i + 1) - x(i) * x(i)) + (b * b) * (x(i + 1) - x(i));
        }
    }

    // Integrate on the right part
    for (int i = indRightStart; i < indRightEnd; i++) {
        df = F(i + 1) - F(i);
        dVal = x(i + 1) - x(i);
        if (dVal != 0) {
            // Build a line y=ax+b and switch it (a -> -a & b -> 1-b) to easily integrate
            float a = -df / dVal;
            float b = -x(i) * (-a) + F(i);
            b = 1 - b;

            // CRPS after integration with H=0 as we switched the axis
            crps += (a * a / 3) * (x(i + 1) * x(i + 1) * x(i + 1) - x(i) * x(i) * x(i)) +
                    (a * b) * (x(i + 1) * x(i + 1) - x(i) * x(i)) + (b * b) * (x(i + 1) - x(i));
        }
    }

    return crps;
}

bool asScoreCRPSEP::ProcessScoreClimatology(const a1f &refVals, const a1f &climatologyData) {
    return true;
}
