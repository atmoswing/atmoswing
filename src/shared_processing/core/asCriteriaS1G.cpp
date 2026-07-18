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
 * Portions Copyright 2019 Pascal Horton, University of Bern.
 */

#include "asCriteriaS1G.h"

#include "asIncludes.h"

namespace {

// Gaussian weights flattened to match the linearized gradient passes below.
// wCol holds the column-gradient weights at the flattened positions, with zeros at the
// row-wrap positions (so no correction pass is needed); wRow holds the row-gradient
// weights. Cached per thread as criteria instances are shared between worker threads
// and the dimensions are constant within a processing run.
struct GaussWeightsCache {
    int rows = -1;
    int cols = -1;
    a1f wCol;  // size rows*cols - 1
    a1f wRow;  // size (rows-1)*cols

    void Update(int rowsNb, int colsNb) {
        if (rows == rowsNb && cols == colsNb) return;
        rows = rowsNb;
        cols = colsNb;

        a2f gaussCol = asCriteria::GetGauss2D(rowsNb, colsNb - 1);
        a2f gaussRow = asCriteria::GetGauss2D(rowsNb - 1, colsNb);

        wCol = a1f::Zero((Eigen::Index)rowsNb * colsNb - 1);
        for (int i = 0; i < rowsNb; ++i) {
            for (int j = 0; j < colsNb - 1; ++j) {
                wCol[(Eigen::Index)i * colsNb + j] = gaussCol(i, j);
            }
        }

        wRow = a1f((Eigen::Index)(rowsNb - 1) * colsNb);
        for (int i = 0; i < rowsNb - 1; ++i) {
            for (int j = 0; j < colsNb; ++j) {
                wRow[(Eigen::Index)i * colsNb + j] = gaussRow(i, j);
            }
        }
    }
};

thread_local GaussWeightsCache g_gaussCache;

}  // namespace

asCriteriaS1G::asCriteriaS1G()
    : asCriteria("S1", _("Teweles-Wobus score with a Gaussian weighting"), Asc) {
    _minPointsNb = 2;
    _scaleWorst = 200;
    _canUseInline = false;
}

asCriteriaS1G::~asCriteriaS1G() = default;

float asCriteriaS1G::Assess(const a2f& refData, const a2f& evalData, int rowsNb, int colsNb) const {
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

    g_gaussCache.Update(rowsNb, colsNb);

    // Weighted Teweles-Wobus: same fused single-pass structure as asCriteriaS1, with the
    // Gaussian weights applied per gradient cell in both the dividend and the divisor.
    using A4 = Eigen::Array4f;
    using M4 = Eigen::Map<const Eigen::Array4f>;
    const float* r = refData.data();
    const float* e = evalData.data();
    const auto n = (Eigen::Index)rowsNb * colsNb;

    float dividend = 0, divisor = 0;

    // Column gradients: data[i+1] - data[i] over the flattened array (n-1 terms). The
    // terms spanning a row boundary have a zero weight, so no correction is needed.
    {
        const float* w = g_gaussCache.wCol.data();
        A4 dAcc = A4::Zero(), vAcc = A4::Zero();
        Eigen::Index i = 0;
        for (; i + 4 <= n - 1; i += 4) {
            A4 refGrad = M4(r + i + 1) - M4(r + i);
            A4 evalGrad = M4(e + i + 1) - M4(e + i);
            A4 weight = M4(w + i);
            dAcc += weight * (refGrad - evalGrad).abs();
            vAcc += weight * refGrad.abs().max(evalGrad.abs());
        }
        float d = dAcc.sum(), v = vAcc.sum();
        for (; i < n - 1; ++i) {
            float refGrad = r[i + 1] - r[i];
            float evalGrad = e[i + 1] - e[i];
            d += w[i] * std::fabs(refGrad - evalGrad);
            v += w[i] * std::max(std::fabs(refGrad), std::fabs(evalGrad));
        }
        dividend += d;
        divisor += v;
    }

    // Row gradients: data[i+cols] - data[i], contiguous over (rows-1)*cols terms.
    {
        const float* w = g_gaussCache.wRow.data();
        const Eigen::Index m = n - colsNb;
        A4 dAcc = A4::Zero(), vAcc = A4::Zero();
        Eigen::Index i = 0;
        for (; i + 4 <= m; i += 4) {
            A4 refGrad = M4(r + i + colsNb) - M4(r + i);
            A4 evalGrad = M4(e + i + colsNb) - M4(e + i);
            A4 weight = M4(w + i);
            dAcc += weight * (refGrad - evalGrad).abs();
            vAcc += weight * refGrad.abs().max(evalGrad.abs());
        }
        float d = dAcc.sum(), v = vAcc.sum();
        for (; i < m; ++i) {
            float refGrad = r[i + colsNb] - r[i];
            float evalGrad = e[i + colsNb] - e[i];
            d += w[i] * std::fabs(refGrad - evalGrad);
            v += w[i] * std::max(std::fabs(refGrad), std::fabs(evalGrad));
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
