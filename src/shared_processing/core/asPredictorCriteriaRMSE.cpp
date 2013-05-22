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
 
#include "asPredictorCriteriaRMSE.h"

asPredictorCriteriaRMSE::asPredictorCriteriaRMSE(int linAlgebraMethod)
:
asPredictorCriteria(linAlgebraMethod)
{
    m_Criteria = asPredictorCriteria::RMSE;
    m_Name = "RMSE";
    m_FullName = _("Root Mean Square Error");
    m_Order = Asc;
    m_ScaleBest = 0;
    m_ScaleWorst = NaNFloat;
}

asPredictorCriteriaRMSE::~asPredictorCriteriaRMSE()
{
    //dtor
}

float asPredictorCriteriaRMSE::Assess(const Array2DFloat &refData, const Array2DFloat &evalData, int rowsNb, int colsNb)
{
    wxASSERT_MSG(refData.rows()==evalData.rows(), wxString::Format("refData.rows()=%d, evalData.rows()=%d", (int)refData.rows(), (int)evalData.rows()));
    wxASSERT_MSG(refData.cols()==evalData.cols(), wxString::Format("refData.cols()=%d, evalData.cols()=%d", (int)refData.cols(), (int)evalData.cols()));

    if (rowsNb==0 || colsNb==0)
    {
        rowsNb = refData.rows();
        colsNb = refData.cols();
    }

    wxASSERT(refData.rows()==rowsNb);
    wxASSERT(refData.cols()==colsNb);
    wxASSERT(evalData.rows()==rowsNb);
    wxASSERT(evalData.cols()==colsNb);

    float mse = 0;

    switch (m_LinAlgebraMethod)
    {
        case (asLIN_ALGEBRA_NOVAR):
        case (asLIN_ALGEBRA):
        {
            mse += (evalData-refData).pow(2).sum();
            break;
        }

        case (asCOEFF_NOVAR):
        case (asCOEFF):
        {
            for (int i=0; i<rowsNb; i++)
            {
                for (int j=0; j<colsNb; j++)
                {
                    mse += (evalData(i,j) - refData(i,j)) * (evalData(i,j) - refData(i,j));
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

    wxASSERT(mse>=0);
    wxASSERT(refData.size()>0);

    mse /= (float)refData.size();

    wxASSERT(mse>=0);
    wxASSERT_MSG(sqrt(mse)>=m_ScaleBest, _("The criteria is below the lower limit..."));

    return sqrt(mse);
}
