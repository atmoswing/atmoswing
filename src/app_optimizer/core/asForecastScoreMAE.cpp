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

#include "asForecastScoreMAE.h"

asForecastScoreMAE::asForecastScoreMAE()
        : asForecastScore()
{
    m_score = asForecastScore::MAE;
    m_name = _("Mean absolute error");
    m_fullName = _("Mean absolute error");
    m_order = Asc;
    m_scaleBest = 0;
    m_scaleWorst = NaNf;
}

asForecastScoreMAE::~asForecastScoreMAE()
{
    //dtor
}

float asForecastScoreMAE::Assess(float ObservedVal, const a1f &ForcastVals, int nbElements) const
{
    wxASSERT(ForcastVals.size() > 1);
    wxASSERT(nbElements > 0);
    wxASSERT(!asTools::IsNaN(m_quantile));
    wxASSERT(m_quantile > 0);
    wxASSERT(m_quantile < 1);

    // Create the container to sort the data
    a1f x(nbElements);

    // Remove the NaNs and copy content
    int nbForecasts = CleanNans(ForcastVals, x, nbElements);
    if (nbForecasts == asNOT_FOUND) {
        wxLogWarning(_("Only NaNs as inputs in the CRPS processing function."));
        return NaNf;
    } else if (nbForecasts <= 2) {
        wxLogWarning(_("Not enough elements to process the CRPS."));
        return NaNf;
    }

    a1f cleanValues = x.head(nbForecasts);

    // Get value for quantile
    float xQuantile = asTools::GetValueForQuantile(cleanValues, m_quantile);

    float score = std::abs(ObservedVal - xQuantile);

    return score;
}

bool asForecastScoreMAE::ProcessScoreClimatology(const a1f &refVals, const a1f &climatologyData)
{
    return true;
}
