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

#include "asIncludes.h"

asCriteriaS1grads::asCriteriaS1grads()
    : asCriteria("S1grads", _("Teweles-Wobus on gradients"), Asc) {
    _minPointsNb = 2;
    _scaleWorst = 200;
    _canUseInline = true;
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

    if (!_checkNaNs || (!refData.hasNaN() && !evalData.hasNaN())) {
        dividend = ((refData - evalData).abs()).sum();
        divisor = (refData.abs().max(evalData.abs())).sum();

    } else {
        // Single pass, skipping pairs with NaNs (equivalent to zeroing them in both sums)
        const float* r = refData.data();
        const float* e = evalData.data();
        const auto n = refData.size();
        for (Eigen::Index i = 0; i < n; ++i) {
            if (std::isnan(r[i]) || std::isnan(e[i])) continue;
            dividend += std::fabs(r[i] - e[i]);
            divisor += std::max(std::fabs(r[i]), std::fabs(e[i]));
        }
    }

    if (divisor > 0) {
        return 100.0f * (dividend / divisor);  // Can be NaN
    } else {
        if (dividend == 0) {
            wxLogVerbose(_("Both dividend and divisor are equal to zero in the predictor criteria."));
            return _scaleWorst;
        } else if (isnan(divisor) || isnan(dividend)) {
            return NAN;
        } else {
            return _scaleWorst;
        }
    }
}
