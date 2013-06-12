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
 
#include "asPredictorCriteriaS1weights.h"

asPredictorCriteriaS1weights::asPredictorCriteriaS1weights(int linAlgebraMethod)
:
asPredictorCriteria(linAlgebraMethod)
{
    m_Criteria = asPredictorCriteria::S1weights;
    m_Name = "S1weights";
    m_FullName = _("Teweles-Wobus");
    m_Order = Asc;
    m_ScaleBest = 0;
    m_ScaleWorst = 200;
    m_wU = 0.5;
    m_wV = 0.5;

    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Read("/Test/S1weights/wU", &m_wU, 0.5);
    pConfig->Read("/Test/S1weights/wV", &m_wV, 0.5);
    if (m_wU+m_wV!=1)
    {
        if (m_wU+m_wV==0)
        {
            asLogError(_("The sum of the weights is null. Weights are corrected to 0.5."));
            m_wU = 0.5;
            m_wV = 0.5;
        }
        m_wU = m_wU/(m_wU+m_wV);
        m_wV = m_wV/(m_wU+m_wV);
    }
}

asPredictorCriteriaS1weights::~asPredictorCriteriaS1weights()
{
    //dtor
}

float asPredictorCriteriaS1weights::Assess(const Array2DFloat &refData, const Array2DFloat &evalData, int rowsNb, int colsNb)
{
    wxASSERT_MSG(refData.rows()==evalData.rows(), wxString::Format("refData.rows()=%d, evalData.rows()=%d", (int)refData.rows(), (int)evalData.rows()));
    wxASSERT_MSG(refData.cols()==evalData.cols(), wxString::Format("refData.cols()=%d, evalData.cols()=%d", (int)refData.cols(), (int)evalData.cols()));

    if (rowsNb==0 || colsNb==0)
    {
        rowsNb = refData.rows();
        colsNb = refData.cols();
    }

    float dividend = 0, divisor = 0;

    switch (m_LinAlgebraMethod)
    {
        case (asLIN_ALGEBRA_NOVAR):
        case (asLIN_ALGEBRA):
        case (asCOEFF_NOVAR):
        case (asCOEFF):
        {
            float refGradCols, refGradRows, evalGradCols, evalGradRows;

            for (int i=0; i<rowsNb-1; i++)
            {
                for (int j=0; j<colsNb; j++)
                {
                    refGradRows = m_wV*(refData(i+1,j)-refData(i,j));
                    evalGradRows = m_wV*(evalData(i+1,j)-evalData(i,j));

                    dividend += abs(refGradRows-evalGradRows);
                    divisor += wxMax(abs(refGradRows),abs(evalGradRows));
/*
                    // Without weighting the denominator:

                    refGradRows = (refData(i+1,j)-refData(i,j));
                    evalGradRows = (evalData(i+1,j)-evalData(i,j));

                    dividend += m_wV*abs(refGradRows-evalGradRows);
                    divisor += wxMax(abs(refGradRows),abs(evalGradRows));*/
                }
            }

            for (int i=0; i<rowsNb; i++)
            {
                for (int j=0; j<colsNb-1; j++)
                {
                    refGradCols = m_wU*(refData(i,j+1)-refData(i,j));
                    evalGradCols = m_wU*(evalData(i,j+1)-evalData(i,j));

                    dividend += abs(refGradCols-evalGradCols);
                    divisor += wxMax(abs(refGradCols), abs(evalGradCols));
/*
                    // Without weighting the denominator:

                    refGradCols = (refData(i,j+1)-refData(i,j));
                    evalGradCols = (evalData(i,j+1)-evalData(i,j));

                    dividend += m_wU*abs(refGradCols-evalGradCols);
                    divisor += wxMax(abs(refGradCols), abs(evalGradCols));*/
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
        wxASSERT(dividend>=0);
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
