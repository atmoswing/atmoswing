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

#include "asForecastScoreRMSE.h"

asForecastScoreRMSE::asForecastScoreRMSE()
        : asForecastScore()
{
    m_score = asForecastScore::RMSE;
    m_name = _("Root mean square error");
    m_fullName = _("Root mean square error");
    m_order = Asc;
    m_scaleBest = 0;
    m_scaleWorst = NaNFloat;
}

asForecastScoreRMSE::~asForecastScoreRMSE()
{
    //dtor
}

float asForecastScoreRMSE::Assess(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements) const
{
    wxASSERT(ForcastVals.size() > 1);
    wxASSERT(nbElements > 0);
    wxASSERT(!asTools::IsNaN(m_quantile));
    wxASSERT(m_quantile > 0);
    wxASSERT(m_quantile < 1);

    // Create the container to sort the data
    Array1DFloat x(nbElements);

    // Remove the NaNs and copy content
    int nbForecasts = CleanNans(ForcastVals, x, nbElements);
    if (nbForecasts == asNOT_FOUND) {
        asLogWarning(_("Only NaNs as inputs in the CRPS processing function."));
        return NaNFloat;
    } else if (nbForecasts <= 2) {
        asLogWarning(_("Not enough elements to process the CRPS."));
        return NaNFloat;
    }

    Array1DFloat cleanValues = x.head(nbForecasts);

    // Get value for quantile
    float xQuantile = asTools::GetValueForQuantile(cleanValues, m_quantile);

    float score = (ObservedVal - xQuantile) * (ObservedVal - xQuantile);

    return score;
}

bool asForecastScoreRMSE::ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData)
{
    return true;
}

