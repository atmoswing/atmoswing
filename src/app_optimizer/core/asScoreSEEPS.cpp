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

#include "asScoreSEEPS.h"

asScoreSEEPS::asScoreSEEPS()
    : asScore(asScore::SEEPS, _("Stable equitable error in probability space"),
              _("Stable equitable error in probability space"), Asc, NaNf, NaNf, true),
      m_p1(NaNf),
      m_p3(NaNf),
      m_thresNull(0.2f),
      m_thresHigh(NaNf) {}

asScoreSEEPS::~asScoreSEEPS() {}

float asScoreSEEPS::Assess(float obs, const a1f &values, int nbElements) const {
    wxASSERT(values.size() > 1);
    wxASSERT(nbElements > 0);
    wxASSERT(!asIsNaN(m_p1));
    wxASSERT(!asIsNaN(m_p3));
    wxASSERT(!asIsNaN(m_thresHigh));

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
        wxLogWarning(_("Not enough elements to process SEEPS."));
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

    float score = 0;

    // Forecasted 1, observed 1
    if (value <= m_thresNull && obs <= m_thresNull) {
        score = 0.0f;
    }
    // Forecasted 2, observed 1
    else if (value <= m_thresNull && obs > m_thresNull && obs <= m_thresHigh) {
        score = 0.5 * (1.0 / (1.0 - m_p1));
    }
    // Forecasted 3, observed 1
    else if (value <= m_thresNull && obs > m_thresHigh) {
        score = 0.5 * ((1.0 / m_p3) + (1.0 / 1.0 - m_p1));
    }
    // Forecasted 1, observed 2
    else if (value > m_thresNull && value <= m_thresHigh && obs <= m_thresNull) {
        score = 0.5 * (1.0 / m_p1);
    }
    // Forecasted 2, observed 2
    else if (value > m_thresNull && value <= m_thresHigh && obs > m_thresNull && obs <= m_thresHigh) {
        score = 0.0f;
    }
    // Forecasted 3, observed 2
    else if (value > m_thresNull && value <= m_thresHigh && obs > m_thresHigh) {
        score = 0.5 * (1.0 / m_p3);
    }
    // Forecasted 1, observed 3
    else if (value > m_thresHigh && obs <= m_thresNull) {
        score = 0.5 * ((1.0 / m_p1) + (1.0 / (1.0 - m_p3)));
    }
    // Forecasted 2, observed 3
    else if (value > m_thresHigh && obs > m_thresNull && obs <= m_thresHigh) {
        score = 0.5 * (1.0 / (1.0 - m_p3));
    }
    // Forecasted 3, observed 3
    else if (value > m_thresHigh && obs > m_thresHigh) {
        score = 0.0f;
    }

    return score;
}

bool asScoreSEEPS::ProcessScoreClimatology(const a1f &refVals, const a1f &climData) {
    wxASSERT(!asHasNaN(&refVals[0], &refVals[refVals.size() - 1]));
    wxASSERT(!asHasNaN(&climData[0], &climData[climData.size() - 1]));

    a1f climDataSorted = climData;
    asSortArray(&climDataSorted[0], &climDataSorted[climDataSorted.size() - 1], Asc);

    // Find first value above the lower threshold
    int rowAboveThreshold1 = asFindFloor(&climDataSorted[0], &climDataSorted[climDataSorted.size() - 1], m_threshold);
    if (rowAboveThreshold1 < 0) {
        wxLogError(_("Error processing the SEEPS climatology score."));
        return false;
    }

    while (climDataSorted[rowAboveThreshold1] <= m_threshold) {
        rowAboveThreshold1++;
    }

    // Process probability (without processing the empirical frequencies...). Do not substract 1 because it is in 0
    // basis.
    m_p1 = (float)rowAboveThreshold1 / (float)climDataSorted.size();

    m_p3 = (1 - m_p1) / 3.0f;

    int indexThreshold2 = climDataSorted.size() * (m_p1 + 2 * m_p3);
    m_thresHigh = climDataSorted[indexThreshold2];

    return true;
}
