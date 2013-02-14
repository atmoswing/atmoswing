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
 
#include "asPredictorCriteriaRSE.h"

asPredictorCriteriaRSE::asPredictorCriteriaRSE(int linAlgebraMethod)
:
asPredictorCriteria(linAlgebraMethod)
{
    m_Criteria = asPredictorCriteria::RSE;
    m_Name = "RSE";
    m_FullName = _("Root Squared Error");
    m_Order = Asc;
    m_ScaleBest = 0;
    m_ScaleWorst = NaNFloat;
}

asPredictorCriteriaRSE::~asPredictorCriteriaRSE()
{
    //dtor
}

float asPredictorCriteriaRSE::Assess(const Array2DFloat &refData, const Array2DFloat &evalData, int rowsNb, int colsNb)
{
    wxASSERT_MSG(refData.rows()==evalData.rows(), wxString::Format("refData.rows()=%d, evalData.rows()=%d", (int)refData.rows(), (int)evalData.rows()));
    wxASSERT_MSG(refData.cols()==evalData.cols(), wxString::Format("refData.cols()=%d, evalData.cols()=%d", (int)refData.cols(), (int)evalData.cols()));

    if (rowsNb==0 || colsNb==0)
    {
        rowsNb = refData.rows();
        colsNb = refData.cols();
    }

    float se = 0;

    switch (m_LinAlgebraMethod)
    {
        case (asLIN_ALGEBRA_NOVAR):
        case (asLIN_ALGEBRA):
        {
            se += (evalData-refData).pow(2).sum();
            break;
        }

        case (asCOEFF_NOVAR):
        case (asCOEFF):
        {
            for (int i=0; i<rowsNb; i++)
            {
                for (int j=0; j<colsNb; j++)
                {
                    se += (evalData(i,j) - refData(i,j)) * (evalData(i,j) - refData(i,j));
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

    wxASSERT(se>=0);
    wxASSERT_MSG(sqrt(se)>=m_ScaleBest, wxString::Format(_("The criteria (%g) is below the lower limit (%g)..."), sqrt(se), m_ScaleBest));

    return sqrt(se);

}
