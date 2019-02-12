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
 * Portions Copyright 2018 Pascal Horton, University of Bern.
 */

#include "asCriteriaS2.h"

asCriteriaS2::asCriteriaS2()
        : asCriteria("S2", _("Derivative of Teweles-Wobus score"), Asc)
{
    m_scaleBest = 0;
    m_scaleWorst = Inff;
    m_canUseInline = false;
}

asCriteriaS2::~asCriteriaS2() = default;

float asCriteriaS2::Assess(const a2f &refData, const a2f &evalData, int rowsNb, int colsNb) const
{
    wxASSERT(refData.rows() == evalData.rows());
    wxASSERT(refData.cols() == evalData.cols());
    wxASSERT(refData.rows() > 2);
    wxASSERT(refData.cols() > 2);

    if (m_checkNaNs && (refData.hasNaN() || evalData.hasNaN())) {
        wxLogWarning(_("NaNs are not handled in with S2 without preprocessing."));
        return NaNf;
    }

    float dividend = 0, divisor = 0;

    a2f RefGradCols1(rowsNb, colsNb - 1);
    a2f RefGradRows1(rowsNb - 1, colsNb);
    a2f EvalGradCols1(evalData.rows(), evalData.cols() - 1);
    a2f EvalGradRows1(evalData.rows() - 1, evalData.cols());

    RefGradCols1 = (refData.topRightCorner(rowsNb, colsNb - 1) - refData.topLeftCorner(rowsNb, colsNb - 1));
    RefGradRows1 = (refData.bottomLeftCorner(rowsNb - 1, colsNb) - refData.topLeftCorner(rowsNb - 1, colsNb));
    EvalGradCols1 = (evalData.topRightCorner(evalData.rows(), evalData.cols() - 1) -
                     evalData.topLeftCorner(evalData.rows(), evalData.cols() - 1));
    EvalGradRows1 = (evalData.bottomLeftCorner(evalData.rows() - 1, evalData.cols()) -
                     evalData.topLeftCorner(evalData.rows() - 1, evalData.cols()));

    a2f RefGradCols2(rowsNb, colsNb - 2);
    a2f RefGradRows2(rowsNb - 2, colsNb);
    a2f EvalGradCols2(evalData.rows(), evalData.cols() - 2);
    a2f EvalGradRows2(evalData.rows() - 2, evalData.cols());

    RefGradCols2 = (RefGradCols1.topRightCorner(rowsNb, colsNb - 2) - RefGradCols1.topLeftCorner(rowsNb, colsNb - 2));
    RefGradRows2 = (RefGradRows1.bottomLeftCorner(rowsNb - 2, colsNb) - RefGradRows1.topLeftCorner(rowsNb - 2, colsNb));
    EvalGradCols2 = (EvalGradCols1.topRightCorner(evalData.rows(), evalData.cols() - 2) -
                     EvalGradCols1.topLeftCorner(evalData.rows(), evalData.cols() - 2));
    EvalGradRows2 = (EvalGradRows1.bottomLeftCorner(evalData.rows() - 2, evalData.cols()) -
                     EvalGradRows1.topLeftCorner(evalData.rows() - 2, evalData.cols()));

    dividend = ((RefGradCols2 - EvalGradCols2).abs()).sum() + ((RefGradRows2 - EvalGradRows2).abs()).sum();
    divisor = (RefGradCols2.abs().max(EvalGradCols2.abs())).sum() + (RefGradRows2.abs().max(EvalGradRows2.abs())).sum();

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
