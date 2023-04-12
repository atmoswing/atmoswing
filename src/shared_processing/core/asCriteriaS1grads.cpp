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

#include "asCriteriaS1grads.h"

asCriteriaS1grads::asCriteriaS1grads()
    : asCriteria("S1grads", _("Teweles-Wobus on gradients"), Asc) {
    m_minPointsNb = 2;
    m_scaleWorst = 200;
    m_canUseInline = true;
}

asCriteriaS1grads::~asCriteriaS1grads() = default;

float asCriteriaS1grads::Assess(const a2f& refData, const a2f& evalData, int rowsNb, int colsNb) const {
    wxASSERT(refData.rows() == evalData.rows());
    wxASSERT(refData.cols() == evalData.cols());
    wxASSERT(refData.rows() > 0);
    wxASSERT(refData.cols() > 0);

    float dividend = 0, divisor = 0;

    // Note here that the actual gradient data do not fill the entire data blocks,
    // but the rest being 0-filled, we can simplify the sum calculation !

    if (!m_checkNaNs || (!refData.hasNaN() && !evalData.hasNaN())) {
        dividend = ((refData - evalData).abs()).sum();
        divisor = (refData.abs().max(evalData.abs())).sum();

    } else {
        a2f refDataCorr = (!evalData.isNaN() && !refData.isNaN()).select(refData, 0);
        a2f evalDataCorr = (!evalData.isNaN() && !refData.isNaN()).select(evalData, 0);

        dividend = ((refDataCorr - evalDataCorr).abs()).sum();
        divisor = (refDataCorr.abs().max(evalDataCorr.abs())).sum();
    }

    if (divisor > 0) {
        return 100.0f * (dividend / divisor);  // Can be NaN
    } else {
        if (dividend == 0) {
            wxLogVerbose(_("Both dividend and divisor are equal to zero in the predictor criteria."));
            return m_scaleWorst;
        } else if (isnan(divisor) || isnan(dividend)) {
            return NAN;
        } else {
            return m_scaleWorst;
        }
    }
}
