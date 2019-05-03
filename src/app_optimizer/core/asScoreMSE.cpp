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
 * Portions Copyright 2019 Pascal Horton, University of Bern.
 */

#include "asScoreMSE.h"

asScoreMSE::asScoreMSE()
        : asScore(asScore::MSE, _("Mean square error"), _("Mean square error"), Asc, 0, NaNf)
{

}

float asScoreMSE::Assess(float obs, const a1f &values, int nbElements) const
{
    wxASSERT(values.size() > 1);
    wxASSERT(nbElements > 0);
    wxASSERT(!asIsNaN(m_quantile));
    wxASSERT(m_quantile > 0);
    wxASSERT(m_quantile < 1);

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
        wxLogWarning(_("Only NaNs as inputs in the MSE processing function."));
        return NaNf;
    } else if (nbPredict <= 2) {
        wxLogWarning(_("Not enough elements to process the MSE."));
        return NaNf;
    }

    a1f cleanValues = x.head(nbPredict);

    // Get value for quantile
    float xQuantile = asGetValueForQuantile(cleanValues, m_quantile);

    float score = (obs - xQuantile) * (obs - xQuantile);

    return score;
}

bool asScoreMSE::ProcessScoreClimatology(const a1f &refVals, const a1f &climatologyData)
{
    return true;
}

