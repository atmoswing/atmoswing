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

asCriteriaS1G::asCriteriaS1G()
        : asCriteria("S1", _("Teweles-Wobus score with a Gaussian weighting"), Asc)
{
    m_minPointsNb = 2;
    m_scaleWorst = 200;
    m_canUseInline = false;
}

asCriteriaS1G::~asCriteriaS1G() = default;

float asCriteriaS1G::Assess(const a2f &refData, const a2f &evalData, int rowsNb, int colsNb) const
{
    wxASSERT(refData.rows() == evalData.rows());
    wxASSERT(refData.cols() == evalData.cols());
    wxASSERT(refData.rows() == rowsNb);
    wxASSERT(refData.cols() == colsNb);
    wxASSERT(refData.rows() > 1);
    wxASSERT(refData.cols() > 1);

    if (m_checkNaNs && (refData.hasNaN() || evalData.hasNaN())) {
        wxLogWarning(_("NaNs are not handled in with S1 without preprocessing."));
        return NaNf;
    }

    float dividend = 0, divisor = 0;

    a2f g1 = GetGauss2D(rowsNb, colsNb - 1);
    a2f g2 = GetGauss2D(rowsNb - 1, colsNb);

    dividend = (g1 * ((refData.topRightCorner(rowsNb, colsNb - 1) - refData.topLeftCorner(rowsNb, colsNb - 1)) -
                 (evalData.topRightCorner(rowsNb, colsNb - 1) - evalData.topLeftCorner(rowsNb, colsNb - 1))).abs()).sum() +
               (g1 * ((refData.bottomLeftCorner(rowsNb - 1, colsNb) - refData.topLeftCorner(rowsNb - 1, colsNb)) -
                 (evalData.bottomLeftCorner(rowsNb - 1, colsNb) - evalData.topLeftCorner(rowsNb - 1, colsNb))).abs()).sum();

    divisor = (g2 * (refData.topRightCorner(rowsNb, colsNb - 1) - refData.topLeftCorner(rowsNb, colsNb - 1)).abs().max(
               (evalData.topRightCorner(rowsNb, colsNb - 1) - evalData.topLeftCorner(rowsNb, colsNb - 1)).abs())).sum() +
              (g2 * (refData.bottomLeftCorner(rowsNb - 1, colsNb) - refData.topLeftCorner(rowsNb - 1, colsNb)).abs().max(
               (evalData.bottomLeftCorner(rowsNb - 1, colsNb) - evalData.topLeftCorner(rowsNb - 1, colsNb)).abs())).sum();

    if (divisor > 0) {
        return 100.0f * (dividend / divisor); // Can be NaN
    } else {
        if (dividend == 0) {
            wxLogVerbose(_("Both dividend and divisor are equal to zero in the predictor criteria."));
            return m_scaleBest;
        } else {
            return m_scaleWorst;
        }
    }

}
