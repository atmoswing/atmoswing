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

#include "asForecastScoreSEEPS.h"

asForecastScoreSEEPS::asForecastScoreSEEPS()
        : asForecastScore()
{
    m_score = asForecastScore::SEEPS;
    m_name = _("Stable equitable error in probability space");
    m_fullName = _("Stable equitable error in probability space");
    m_scaleBest = NaNFloat;
    m_scaleWorst = NaNFloat;
    m_p1 = NaNFloat;
    m_p3 = NaNFloat;
    m_thresNull = 0.2f;
    m_thresHigh = NaNFloat;
    m_usesClimatology = true;
}

asForecastScoreSEEPS::~asForecastScoreSEEPS()
{
    //dtor
}

float asForecastScoreSEEPS::Assess(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements) const
{
    wxASSERT(ForcastVals.size() > 1);
    wxASSERT(nbElements > 0);
    wxASSERT(!asTools::IsNaN(m_p1));
    wxASSERT(!asTools::IsNaN(m_p3));
    wxASSERT(!asTools::IsNaN(m_thresHigh));
    wxASSERT(!asTools::IsNaN(m_quantile));
    wxASSERT(m_quantile > 0);
    wxASSERT(m_quantile < 1);

    // Create the container to sort the data
    Array1DFloat x(nbElements);

    // Remove the NaNs and copy content
    int nbForecasts = CleanNans(ForcastVals, x, nbElements);
    if (nbForecasts == asNOT_FOUND) {
        wxLogWarning(_("Only NaNs as inputs in the CRPS processing function."));
        return NaNFloat;
    } else if (nbForecasts <= 2) {
        wxLogWarning(_("Not enough elements to process the CRPS."));
        return NaNFloat;
    }

    Array1DFloat cleanValues = x.head(nbForecasts);

    // Get value for quantile
    float fcstV = asTools::GetValueForQuantile(cleanValues, m_quantile);
    float obsV = ObservedVal;

    float score = 0;

    // Forecasted 1, observed 1
    if (fcstV <= m_thresNull && obsV <= m_thresNull) {
        score = 0.0f;
    }
        // Forecasted 2, observed 1
    else if (fcstV <= m_thresNull && obsV > m_thresNull && obsV <= m_thresHigh) {
        score = 0.5 * (1.0 / (1.0 - m_p1));
    }
        // Forecasted 3, observed 1
    else if (fcstV <= m_thresNull && obsV > m_thresHigh) {
        score = 0.5 * ((1.0 / m_p3) + (1.0 / 1.0 - m_p1));
    }
        // Forecasted 1, observed 2
    else if (fcstV > m_thresNull && fcstV <= m_thresHigh && obsV <= m_thresNull) {
        score = 0.5 * (1.0 / m_p1);
    }
        // Forecasted 2, observed 2
    else if (fcstV > m_thresNull && fcstV <= m_thresHigh && obsV > m_thresNull && obsV <= m_thresHigh) {
        score = 0.0f;
    }
        // Forecasted 3, observed 2
    else if (fcstV > m_thresNull && fcstV <= m_thresHigh && obsV > m_thresHigh) {
        score = 0.5 * (1.0 / m_p3);
    }
        // Forecasted 1, observed 3
    else if (fcstV > m_thresHigh && obsV <= m_thresNull) {
        score = 0.5 * ((1.0 / m_p1) + (1.0 / (1.0 - m_p3)));
    }
        // Forecasted 2, observed 3
    else if (fcstV > m_thresHigh && obsV > m_thresNull && obsV <= m_thresHigh) {
        score = 0.5 * (1.0 / (1.0 - m_p3));
    }
        // Forecasted 3, observed 3
    else if (fcstV > m_thresHigh && obsV > m_thresHigh) {
        score = 0.0f;
    }

    return score;
}

bool asForecastScoreSEEPS::ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData)
{
    wxASSERT(!asTools::HasNaN(&refVals[0], &refVals[refVals.size() - 1]));
    wxASSERT(!asTools::HasNaN(&climatologyData[0], &climatologyData[climatologyData.size() - 1]));

    Array1DFloat climatologyDataSorted = climatologyData;
    asTools::SortArray(&climatologyDataSorted[0], &climatologyDataSorted[climatologyDataSorted.size() - 1], Asc);

    // Find first value above the lower threshold
    int rowAboveThreshold1 = asTools::SortedArraySearchFloor(&climatologyDataSorted[0],
                                                             &climatologyDataSorted[climatologyDataSorted.size() - 1],
                                                             m_threshold);
    while (climatologyDataSorted[rowAboveThreshold1] <= m_threshold) {
        rowAboveThreshold1++;
    }

    // Process probability (without processing the empirical frequencies...). Do not substract 1 because it is in 0 basis.
    m_p1 = (float) rowAboveThreshold1 / (float) climatologyDataSorted.size();

    m_p3 = (1 - m_p1) / 3.0f;

    int indexThreshold2 = climatologyDataSorted.size() * (m_p1 + 2 * m_p3);
    m_thresHigh = climatologyDataSorted[indexThreshold2];

    return true;
}
