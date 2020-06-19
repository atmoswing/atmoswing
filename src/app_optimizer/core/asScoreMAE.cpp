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

#include "asScoreMAE.h"

asScoreMAE::asScoreMAE() : asScore(asScore::MAE, _("Mean absolute error"), _("Mean absolute error"), Asc, 0, NaNf) {}

float asScoreMAE::Assess(float obs, const a1f &values, int nbElements) const {
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

    // Remove the NaNs and copy content
    int nbPredict = CleanNans(values, x, nbElements);
    if (nbPredict == asNOT_FOUND) {
        wxLogWarning(_("Only NaNs as inputs in the CRPS processing function."));
        return NaNf;
    } else if (nbPredict <= 2) {
        wxLogWarning(_("Not enough elements to process the MAE."));
        return NaNf;
    }

    a1f cleanValues = x.head(nbPredict);
    float value = 0;

    if (m_onMean) {
        value = cleanValues.mean();
    } else {
        // Get value for quantile
        wxASSERT(!asIsNaN(m_quantile));
        wxASSERT(m_quantile > 0);
        wxASSERT(m_quantile < 1);
        value = asGetValueForQuantile(cleanValues, m_quantile);
    }

    float score = std::abs(obs - value);

    return score;
}

bool asScoreMAE::ProcessScoreClimatology(const a1f &refVals, const a1f &climatologyData) {
    return true;
}
