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
 
 #include "asPredictorCriteriaS1grads.h"

asPredictorCriteriaS1grads::asPredictorCriteriaS1grads(int linAlgebraMethod)
:
asPredictorCriteria(linAlgebraMethod)
{
    m_Criteria = asPredictorCriteria::S1grads;
    m_Name = "S1grads";
    m_FullName = _("Teweles-Wobus on gradients");
    m_Order = Asc;
    m_ScaleBest = 0;
    m_ScaleWorst = 200;
}

asPredictorCriteriaS1grads::~asPredictorCriteriaS1grads()
{
    //dtor
}

float asPredictorCriteriaS1grads::Assess(const Array2DFloat &refData, const Array2DFloat &evalData, int rowsNb, int colsNb)
{
    wxASSERT_MSG(refData.rows()==evalData.rows(), wxString::Format("refData.rows()=%d, evalData.rows()=%d", (int)refData.rows(), (int)evalData.rows()));
    wxASSERT_MSG(refData.cols()==evalData.cols(), wxString::Format("refData.cols()=%d, evalData.cols()=%d", (int)refData.cols(), (int)evalData.cols()));

    if (rowsNb==0 || colsNb==0)
    {
        rowsNb = refData.rows();
        colsNb = refData.cols();
    }

    float dividend = 0, divisor = 0;
    wxASSERT(rowsNb==1);

    switch (m_LinAlgebraMethod)
    {
        case (asLIN_ALGEBRA_NOVAR):
        case (asLIN_ALGEBRA):
        {
            dividend = ((refData-evalData).abs()).sum();
            divisor = (refData.abs().max(evalData.abs())).sum();

            break;
        }

        case (asCOEFF_NOVAR):
        case (asCOEFF):
        {
            for (int i=0; i<colsNb; i++)
            {
                dividend += abs(refData(0,i)-evalData(0,i));
                divisor += wxMax(abs(refData(0,i)),abs(evalData(0,i)));
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
        wxASSERT(dividend>=0);
        wxASSERT(divisor>0);

        float val = 100.0f*(dividend/divisor);

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
