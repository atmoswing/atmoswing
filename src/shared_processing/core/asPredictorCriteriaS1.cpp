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
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 */
 
#include "asPredictorCriteriaS1.h"

asPredictorCriteriaS1::asPredictorCriteriaS1(int linAlgebraMethod)
:
asPredictorCriteria(linAlgebraMethod)
{
    m_Criteria = asPredictorCriteria::S1;
    m_Name = "S1";
    m_FullName = _("Teweles-Wobus");
    m_Order = Asc;
    m_ScaleBest = 0;
    m_ScaleWorst = 200;
}

asPredictorCriteriaS1::~asPredictorCriteriaS1()
{
    //dtor
}

float asPredictorCriteriaS1::Assess(const Array2DFloat &refData, const Array2DFloat &evalData, int rowsNb, int colsNb)
{
    wxASSERT_MSG(refData.rows()==evalData.rows(), wxString::Format("refData.rows()=%d, evalData.rows()=%d", (int)refData.rows(), (int)evalData.rows()));
    wxASSERT_MSG(refData.cols()==evalData.cols(), wxString::Format("refData.cols()=%d, evalData.cols()=%d", (int)refData.cols(), (int)evalData.cols()));
    wxASSERT_MSG(refData.rows()>0, wxString::Format("refData.rows()=%d", (int)refData.rows()));
    wxASSERT_MSG(refData.cols()>0, wxString::Format("refData.cols()=%d", (int)refData.cols()));

    if (refData.rows()<1) asThrowException(_("The number of rows of the data is null in the S1 criteria processing."));
    if (refData.cols()<1) asThrowException(_("The number of cols of the data is null in the S1 criteria processing."));

    if (rowsNb==0 || colsNb==0)
    {
        rowsNb = refData.rows();
        colsNb = refData.cols();
    }

    float dividend = 0, divisor = 0;

    switch (m_LinAlgebraMethod)
    {
        case (asLIN_ALGEBRA_NOVAR):
        {

            dividend = (((refData.topRightCorner(rowsNb,colsNb-1) - refData.topLeftCorner(rowsNb,colsNb-1))
                            -(evalData.topRightCorner(evalData.rows(),evalData.cols()-1) - evalData.topLeftCorner(evalData.rows(),evalData.cols()-1))).abs()).sum()
                        + (((refData.bottomLeftCorner(rowsNb-1,colsNb) - refData.topLeftCorner(rowsNb-1,colsNb))
                            -(evalData.bottomLeftCorner(evalData.rows()-1,evalData.cols()) - evalData.topLeftCorner(evalData.rows()-1,evalData.cols()))).abs()).sum();

            divisor = ((refData.topRightCorner(rowsNb,colsNb-1) - refData.topLeftCorner(rowsNb,colsNb-1)).abs().max((evalData.topRightCorner(evalData.rows(),evalData.cols()-1) - evalData.topLeftCorner(evalData.rows(),evalData.cols()-1)).abs())).sum()
                        + ((refData.bottomLeftCorner(rowsNb-1,colsNb) - refData.topLeftCorner(rowsNb-1,colsNb)).abs().max((evalData.bottomLeftCorner(evalData.rows()-1,evalData.cols()) - evalData.topLeftCorner(evalData.rows()-1,evalData.cols())).abs())).sum();

            break;
        }

        case (asLIN_ALGEBRA):
        {
            Array2DFloat RefGradCols(rowsNb,colsNb-1);
            Array2DFloat RefGradRows(rowsNb-1,colsNb);
            Array2DFloat EvalGradCols(evalData.rows(),evalData.cols()-1);
            Array2DFloat EvalGradRows(evalData.rows()-1,evalData.cols());

            RefGradCols = (refData.topRightCorner(rowsNb,colsNb-1) - refData.topLeftCorner(rowsNb,colsNb-1));
            RefGradRows = (refData.bottomLeftCorner(rowsNb-1,colsNb) - refData.topLeftCorner(rowsNb-1,colsNb));
            EvalGradCols = (evalData.topRightCorner(evalData.rows(),evalData.cols()-1) - evalData.topLeftCorner(evalData.rows(),evalData.cols()-1));
            EvalGradRows = (evalData.bottomLeftCorner(evalData.rows()-1,evalData.cols()) - evalData.topLeftCorner(evalData.rows()-1,evalData.cols()));

            dividend = ((RefGradCols-EvalGradCols).abs()).sum() + ((RefGradRows-EvalGradRows).abs()).sum();
            divisor = (RefGradCols.abs().max(EvalGradCols.abs())).sum() + (RefGradRows.abs().max(EvalGradRows.abs())).sum();

            break;
        }

        case (asCOEFF_NOVAR):
        {
            for (int i=0; i<rowsNb-1; i++)
            {
                for (int j=0; j<colsNb; j++)
                {
                    dividend += abs((refData(i+1,j)-refData(i,j))-(evalData(i+1,j)-evalData(i,j)));
                    divisor += wxMax(abs((refData(i+1,j)-refData(i,j))),abs((evalData(i+1,j)-evalData(i,j))));
                }
            }

            for (int i=0; i<rowsNb; i++)
            {
                for (int j=0; j<colsNb-1; j++)
                {
                    dividend += abs((refData(i,j+1)-refData(i,j))-(evalData(i,j+1)-evalData(i,j)));
                    divisor += wxMax(abs((refData(i,j+1)-refData(i,j))), abs((evalData(i,j+1)-evalData(i,j))));
                }
            }

            break;
        }

        case (asCOEFF):
        {
            float refGradCols, refGradRows, evalGradCols, evalGradRows;

            for (int i=0; i<rowsNb-1; i++)
            {
                for (int j=0; j<colsNb; j++)
                {
                    refGradRows = refData(i+1,j)-refData(i,j);
                    evalGradRows = evalData(i+1,j)-evalData(i,j);

                    dividend += abs(refGradRows-evalGradRows);
                    divisor += wxMax(abs(refGradRows),abs(evalGradRows));
                }
            }

            for (int i=0; i<rowsNb; i++)
            {
                for (int j=0; j<colsNb-1; j++)
                {
                    refGradCols = refData(i,j+1)-refData(i,j);
                    evalGradCols = evalData(i,j+1)-evalData(i,j);

                    dividend += abs(refGradCols-evalGradCols);
                    divisor += wxMax(abs(refGradCols), abs(evalGradCols));
                }
            }

            break;
        }

        default:
        {
            asLogError(_("The calculation method was not correcty set"));
            return NaNFloat;
        }
    }

    if (asTools::IsNaN(dividend) || asTools::IsNaN(divisor))
    {
        asLogWarning(_("NaNs were found in the data."));
        return NaNFloat;
    }

    if (divisor>0)
    {
        wxASSERT_MSG(dividend>=0, wxString::Format("dividend = %f", dividend));
        wxASSERT(divisor>0);

        float val = (float)100*(dividend/divisor);

        wxASSERT(val<=m_ScaleWorst);
        wxASSERT(val>=m_ScaleBest);

        return val;
    }else {
        if (dividend==0)
        {
            asLogWarning(_("Both dividend and divisor are equal to zero in the predictor criteria."));
            return m_ScaleBest;
        }
        else
        {
            asLogWarning(_("Division by zero in the predictor criteria."));
            return NaNFloat;
        }
    }

    return asNOT_VALID;

}
