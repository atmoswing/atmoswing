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
 * Portions Copyright 2016-2017 Pascal Horton, University of Bern.
 */

#include "asScoreCRPSAR.h"

asScoreCRPSAR::asScoreCRPSAR()
    : asScore(asScore::CRPSAR, _("CRPS Approx Rectangle"),
              _("Continuous Ranked Probability Score approximation with the rectangle method"), Asc, 0, NAN) {}

float asScoreCRPSAR::Assess(float obs, const a1f& values, int nbElements) const {
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
    a1f x(nbElements);
    float x0 = obs;

    // Remove the NaNs and copy content
    int n = CleanNans(values, x, nbElements);
    if (n == asNOT_FOUND) {
        wxLogWarning(_("Only NaNs as inputs in the CRPS processing function."));
        return NAN;
    } else if (n <= 2) {
        wxLogWarning(_("Not enough elements to process the CRPS."));
        return NAN;
    }

    // Sort the forcast array
    asSortArray(&x[0], &x[n - 1], Asc);

    float crps = 0;

    // Cumulative frequency
    a1f Fx = asGetCumulativeFrequency(n);

    // Add rectangle on right side if observed value is on the right of the distribution
    if (x0 > x[n - 1]) {
        crps += x0 - x[n - 1];
    }

    // Add rectangle on the left side if observed value is on the left of the distribution
    if (x0 < x[0]) {
        crps += x[0] - x0;
    }

    // Integrate the distribution
    if (n > 1) {
        for (int i = 0; i < n - 1; i++) {
            if (x[i] < x0) {
                // Left of the observed value
                if (x[i + 1] <= x0) {
                    // Next value also left side of observed value
                    crps += (x[i + 1] - x[i]) * (Fx[i] * Fx[i] + Fx[i + 1] * Fx[i + 1]) / 2;
                } else {
                    // Observation in between 2 values
                    float F0 = (Fx[i + 1] - Fx[i]) * (x0 - x[i]) / (x[i + 1] - x[i]) + Fx[i];
                    crps += (x0 - x[i]) * (F0 * F0 + Fx[i] * Fx[i]) / 2;
                    crps += (x[i + 1] - x0) * ((F0 - 1) * (F0 - 1) + (Fx[i + 1] - 1) * (Fx[i + 1] - 1)) / 2;
                }
            } else {
                // Right of the observed value
                crps += (x[i + 1] - x[i]) * ((Fx[i] - 1) * (Fx[i] - 1) + (Fx[i + 1] - 1) * (Fx[i + 1] - 1)) / 2;
            }
        }
    }

    return crps;
}

bool asScoreCRPSAR::ProcessScoreClimatology(const a1f& refVals, const a1f& climatologyData) {
    return true;
}
