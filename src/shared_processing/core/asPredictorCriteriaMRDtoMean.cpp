/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
#include "asPredictorCriteriaMRDtoMean.h"

asPredictorCriteriaMRDtoMean::asPredictorCriteriaMRDtoMean(int linAlgebraMethod)
:
asPredictorCriteria(linAlgebraMethod)
{
    m_Criteria = asPredictorCriteria::MRDtoMean;
    m_Name = "MRDtoMean";
    m_FullName = _("Mean Relative Differences to the Mean");
    m_Order = Asc;
    m_ScaleBest = 0;
    m_ScaleWorst = NaNFloat;
}

asPredictorCriteriaMRDtoMean::~asPredictorCriteriaMRDtoMean()
{
    //dtor
}

float asPredictorCriteriaMRDtoMean::Assess(const Array2DFloat &refData, const Array2DFloat &evalData, int rowsNb, int colsNb)
{
    wxASSERT_MSG(refData.rows()==evalData.rows(), wxString::Format("refData.rows()=%d, evalData.rows()=%d", (int)refData.rows(), (int)evalData.rows()));
    wxASSERT_MSG(refData.cols()==evalData.cols(), wxString::Format("refData.cols()=%d, evalData.cols()=%d", (int)refData.cols(), (int)evalData.cols()));

    if (rowsNb==0 || colsNb==0)
    {
        rowsNb = refData.rows();
        colsNb = refData.cols();
    }

    float rd = 0;

    switch (m_LinAlgebraMethod)
    {
        case (asLIN_ALGEBRA_NOVAR): // Not implemented yet
        case (asLIN_ALGEBRA): // Not implemented yet
        case (asCOEFF_NOVAR):
        {
            for (int i=0; i<rowsNb; i++)
            {
                for (int j=0; j<colsNb; j++)
                {
                    if(abs(evalData(i,j)+refData(i,j))>0)
                    {
                        rd += abs(evalData(i,j) - refData(i,j)) / (abs(evalData(i,j) + refData(i,j))*0.5);
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

            // Faster in the order cols then rows than the opposite
            for (int i=0; i<rowsNb; i++)
            {
                for (int j=0; j<colsNb; j++)
                {
                    dividend = abs(evalData(i,j) - refData(i,j));
                    divisor = abs(evalData(i,j) + refData(i,j))*0.5;

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
