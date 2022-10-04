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

#include "asScoreDF0.h"

asScoreDF0::asScoreDF0()
    : asScore(asScore::DF0, _("Difference of F(0)"), _("Absolute difference of the frequency of null precipitations."),
              Asc, 0, NaNf) {}

float asScoreDF0::Assess(float obs, const a1f& values, int nbElements) const {
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
        wxLogWarning(_("Only NaNs as inputs in the DF0 processing function."));
        return NaNf;
    } else if (nbPredict <= 2) {
        wxLogWarning(_("Not enough elements to process the DF0."));
        return NaNf;
    }

    // Sort the forcast array
    asSortArray(&x[0], &x[nbPredict - 1], Asc);

    // Cumulative frequency
    a1f F = asGetCumulativeFrequency(nbPredict);

    // Identify the last 0
    int indexLastZero = -1;
    for (int i = 0; i < nbElements; i++) {
        if (x[i] == 0) {
            indexLastZero = i;
        }
    }

    // Find FxObs, fix xObs and integrate beyond limits
    float FxObs;
    if (xObs > 0.0)  // If precipitation
    {
        FxObs = 1;
    } else {
        FxObs = 0;
    }

    if (indexLastZero < 0) {
        wxLogError(_("Error processing DF0 score."));
        return NaNf;
    }

    return std::abs((1.0f - F(indexLastZero)) - FxObs);
}

bool asScoreDF0::ProcessScoreClimatology(const a1f& refVals, const a1f& climatologyData) {
    return true;
}
