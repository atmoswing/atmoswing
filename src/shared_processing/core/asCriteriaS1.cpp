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

#include "asCriteriaS1.h"

#include "asIncludes.h"

asCriteriaS1::asCriteriaS1()
    : asCriteria("S1", _("Teweles-Wobus score"), Asc) {
    _minPointsNb = 2;
    _scaleWorst = 200;
    _canUseInline = false;
}

asCriteriaS1::~asCriteriaS1() = default;

float asCriteriaS1::Assess(const a2f& refData, const a2f& evalData, int rowsNb, int colsNb) const {
    wxASSERT(refData.rows() == evalData.rows());
    wxASSERT(refData.cols() == evalData.cols());
    wxASSERT(refData.rows() == rowsNb);
    wxASSERT(refData.cols() == colsNb);
    wxASSERT(refData.rows() > 1);
    wxASSERT(refData.cols() > 1);

    if (_checkNaNs && (refData.hasNaN() || evalData.hasNaN())) {
        wxLogWarning(_("NaNs are not handled in with S1 without preprocessing."));
        return NAN;
    }

    // Teweles-Wobus score computed in a single fused pass per gradient direction: each
    // gradient is evaluated once and contributes to both the dividend and the divisor.
    // The data is row-major and contiguous, so both passes run on packet-sized chunks.
    using A4 = Eigen::Array4f;
    using M4 = Eigen::Map<const Eigen::Array4f>;
    const float* r = refData.data();
    const float* e = evalData.data();
    const auto n = (Eigen::Index)rowsNb * colsNb;

    float dividend = 0, divisor = 0;

    // Column gradients: data[i+1] - data[i] over the flattened array (n-1 terms),
    // then remove the spurious terms spanning a row boundary.
    {
        A4 dAcc = A4::Zero(), vAcc = A4::Zero();
        Eigen::Index i = 0;
        for (; i + 4 <= n - 1; i += 4) {
            A4 refGrad = M4(r + i + 1) - M4(r + i);
            A4 evalGrad = M4(e + i + 1) - M4(e + i);
            dAcc += (refGrad - evalGrad).abs();
            vAcc += refGrad.abs().max(evalGrad.abs());
        }
        float d = dAcc.sum(), v = vAcc.sum();
        for (; i < n - 1; ++i) {
            float refGrad = r[i + 1] - r[i];
            float evalGrad = e[i + 1] - e[i];
            d += std::fabs(refGrad - evalGrad);
            v += std::max(std::fabs(refGrad), std::fabs(evalGrad));
        }
        for (int row = 1; row < rowsNb; ++row) {
            Eigen::Index k = (Eigen::Index)row * colsNb - 1;
            float refGrad = r[k + 1] - r[k];
            float evalGrad = e[k + 1] - e[k];
            d -= std::fabs(refGrad - evalGrad);
            v -= std::max(std::fabs(refGrad), std::fabs(evalGrad));
        }
        dividend += d;
        divisor += v;
    }

    // Row gradients: data[i+cols] - data[i], contiguous over (rows-1)*cols terms.
    {
        const Eigen::Index m = n - colsNb;
        A4 dAcc = A4::Zero(), vAcc = A4::Zero();
        Eigen::Index i = 0;
        for (; i + 4 <= m; i += 4) {
            A4 refGrad = M4(r + i + colsNb) - M4(r + i);
            A4 evalGrad = M4(e + i + colsNb) - M4(e + i);
            dAcc += (refGrad - evalGrad).abs();
            vAcc += refGrad.abs().max(evalGrad.abs());
        }
        float d = dAcc.sum(), v = vAcc.sum();
        for (; i < m; ++i) {
            float refGrad = r[i + colsNb] - r[i];
            float evalGrad = e[i + colsNb] - e[i];
            d += std::fabs(refGrad - evalGrad);
            v += std::max(std::fabs(refGrad), std::fabs(evalGrad));
        }
        dividend += d;
        divisor += v;
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
