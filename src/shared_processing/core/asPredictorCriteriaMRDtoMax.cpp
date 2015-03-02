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
 
#include "asPredictorCriteriaMRDtoMax.h"

asPredictorCriteriaMRDtoMax::asPredictorCriteriaMRDtoMax(int linAlgebraMethod)
:
asPredictorCriteria(linAlgebraMethod)
{
    m_criteria = asPredictorCriteria::MRDtoMax;
    m_name = "MRDtoMax";
    m_fullName = _("Mean Relative Differences to the Maximum");
    m_order = Asc;
    m_scaleBest = 0;
    m_scaleWorst = NaNFloat;
    m_canUseInline = true;
}

asPredictorCriteriaMRDtoMax::~asPredictorCriteriaMRDtoMax()
{
    //dtor
}

float asPredictorCriteriaMRDtoMax::Assess(const Array2DFloat &refData, const Array2DFloat &evalData, int rowsNb, int colsNb)
{
    wxASSERT_MSG(refData.rows()==evalData.rows(), wxString::Format("refData.rows()=%d, evalData.rows()=%d", (int)refData.rows(), (int)evalData.rows()));
    wxASSERT_MSG(refData.cols()==evalData.cols(), wxString::Format("refData.cols()=%d, evalData.cols()=%d", (int)refData.cols(), (int)evalData.cols()));

    float rd = 0;

    switch (m_linAlgebraMethod)
    {
        case (asLIN_ALGEBRA_NOVAR): // Not implemented yet
        case (asLIN_ALGEBRA): // Not implemented yet
        case (asCOEFF_NOVAR):
        {
            for (int i=0; i<rowsNb; i++)
            {
                for (int j=0; j<colsNb; j++)
                {
                    if(wxMax(abs(evalData(i,j)), abs(refData(i,j)))>0)
                    {
                        rd += abs(evalData(i,j) - refData(i,j))/wxMax(abs(evalData(i,j)), abs(refData(i,j)));
                    }
                    else
                    {
                        if (abs(evalData(i,j) - refData(i,j))!=0)
                        {
                            asLogWarning(_("Division by zero in the predictor criteria."));
                            return NaNFloat;
                        }
                    }
                }
            }

            break;
        }

        case (asCOEFF):
        {
            float dividend = 0, divisor = 0;

            for (int i=0; i<rowsNb; i++)
            {
                for (int j=0; j<colsNb; j++)
                {
                    dividend = abs(evalData(i,j) - refData(i,j));
                    divisor = wxMax(abs(evalData(i,j)), abs(refData(i,j)));

                    if(divisor>0)
                    {
                        rd += dividend/divisor;
                    }
                    else
                    {
                        if (dividend!=0)
                        {
                            asLogWarning(_("Division by zero in the predictor criteria."));
                            return NaNFloat;
                        }
                    }
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

    return rd/refData.size();

}
