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

asCriteriaS1::asCriteriaS1()
        : asCriteria(asCriteria::S1, "S1", _("Teweles-Wobus score"), Asc)
{
    m_scaleBest = 0;
    m_scaleWorst = 200;
    m_canUseInline = false;
}

asCriteriaS1::~asCriteriaS1()
{
    //dtor
}

float asCriteriaS1::Assess(const a2f &refData, const a2f &evalData, int rowsNb, int colsNb) const
{
    wxASSERT(refData.rows() == evalData.rows());
    wxASSERT(refData.cols() == evalData.cols());
    wxASSERT(refData.rows() > 1);
    wxASSERT(refData.cols() > 1);

    float dividend = 0, divisor = 0;

    dividend = (((refData.topRightCorner(rowsNb, colsNb - 1) - refData.topLeftCorner(rowsNb, colsNb - 1)) -
                 (evalData.topRightCorner(evalData.rows(), evalData.cols() - 1) -
                  evalData.topLeftCorner(evalData.rows(), evalData.cols() - 1))).abs()).sum() +
               (((refData.bottomLeftCorner(rowsNb - 1, colsNb) - refData.topLeftCorner(rowsNb - 1, colsNb)) -
                 (evalData.bottomLeftCorner(evalData.rows() - 1, evalData.cols()) -
                  evalData.topLeftCorner(evalData.rows() - 1, evalData.cols()))).abs()).sum();

    divisor = ((refData.topRightCorner(rowsNb, colsNb - 1) - refData.topLeftCorner(rowsNb, colsNb - 1)).abs().max(
            (evalData.topRightCorner(evalData.rows(), evalData.cols() - 1) -
             evalData.topLeftCorner(evalData.rows(), evalData.cols() - 1)).abs())).sum() +
              ((refData.bottomLeftCorner(rowsNb - 1, colsNb) - refData.topLeftCorner(rowsNb - 1, colsNb)).abs().max(
                      (evalData.bottomLeftCorner(evalData.rows() - 1, evalData.cols()) -
                       evalData.topLeftCorner(evalData.rows() - 1, evalData.cols())).abs())).sum();


    /* More readable version
    Array2DFloat RefGradCols(rowsNb, colsNb - 1);
    Array2DFloat RefGradRows(rowsNb - 1, colsNb);
    Array2DFloat EvalGradCols(evalData.rows(), evalData.cols() - 1);
    Array2DFloat EvalGradRows(evalData.rows() - 1, evalData.cols());

    RefGradCols = (refData.topRightCorner(rowsNb, colsNb - 1) - refData.topLeftCorner(rowsNb, colsNb - 1));
    RefGradRows = (refData.bottomLeftCorner(rowsNb - 1, colsNb) - refData.topLeftCorner(rowsNb - 1, colsNb));
    EvalGradCols = (evalData.topRightCorner(evalData.rows(), evalData.cols() - 1) -
                    evalData.topLeftCorner(evalData.rows(), evalData.cols() - 1));
    EvalGradRows = (evalData.bottomLeftCorner(evalData.rows() - 1, evalData.cols()) -
                    evalData.topLeftCorner(evalData.rows() - 1, evalData.cols()));

    dividend = ((RefGradCols - EvalGradCols).abs()).sum() + ((RefGradRows - EvalGradRows).abs()).sum();
    divisor = (RefGradCols.abs().max(EvalGradCols.abs())).sum() +
              (RefGradRows.abs().max(EvalGradRows.abs())).sum();

    */

    if (divisor > 0) {
        return 100.0f * (dividend / divisor); // Can be NaN
    } else {
        if (dividend == 0) {
            wxLogVerbose(_("Both dividend and divisor are equal to zero in the predictor criteria."));
            return m_scaleBest;
        } else {
            return NaNf;
        }
    }

}
