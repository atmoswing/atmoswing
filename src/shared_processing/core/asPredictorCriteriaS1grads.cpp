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
    m_criteria = asPredictorCriteria::S1grads;
    m_name = "S1grads";
    m_fullName = _("Teweles-Wobus on gradients");
    m_order = Asc;
    m_scaleBest = 0;
    m_scaleWorst = 200;
    m_canUseInline = true;
}

asPredictorCriteriaS1grads::~asPredictorCriteriaS1grads()
{
    //dtor
}

float asPredictorCriteriaS1grads::Assess(const Array2DFloat &refData, const Array2DFloat &evalData, int rowsNb, int colsNb)
{
    wxASSERT_MSG(refData.rows()==evalData.rows(), wxString::Format("refData.rows()=%d, evalData.rows()=%d", (int)refData.rows(), (int)evalData.rows()));
    wxASSERT_MSG(refData.cols()==evalData.cols(), wxString::Format("refData.cols()=%d, evalData.cols()=%d", (int)refData.cols(), (int)evalData.cols()));
    wxASSERT_MSG(refData.rows()>0, wxString::Format("refData.rows()=%d", (int)refData.rows()));
    wxASSERT_MSG(refData.cols()>0, wxString::Format("refData.cols()=%d", (int)refData.cols()));

    #ifdef _DEBUG
        if (refData.rows()<1) asThrowException(_("The number of rows of the data is null in the S1grads criteria processing."));
        if (refData.cols()<1) asThrowException(_("The number of cols of the data is null in the S1grads criteria processing."));
    #endif

    float dividend = 0, divisor = 0;

    // Note here that the actual gradient data do not fill the entire data blocks,
    // but the rest being 0-filled, we can simplify the sum calculation !

    switch (m_linAlgebraMethod)
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
            for (int i=0; i<rowsNb; i++)
            {
                for (int j=0; j<colsNb; j++)
                {
                    dividend += abs(refData(i,j)-evalData(i,j));
                    divisor += wxMax(abs(refData(i,j)),abs(evalData(i,j)));
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

    if (divisor>0)
    {
        return 100.0f*(dividend/divisor); // Can be NaN
    }else {
        if (dividend==0)
        {
            asLogWarning(_("Both dividend and divisor are equal to zero in the predictor criteria."));
            return m_scaleBest;
        }
        else
        {
            return NaNFloat;
        }
    }

}
