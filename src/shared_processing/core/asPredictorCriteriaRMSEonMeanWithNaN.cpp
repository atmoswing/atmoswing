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
 
#include "asPredictorCriteriaRMSEonMeanWithNaN.h"

asPredictorCriteriaRMSEonMeanWithNaN::asPredictorCriteriaRMSEonMeanWithNaN(int linAlgebraMethod)
:
asPredictorCriteria(linAlgebraMethod)
{
    m_Criteria = asPredictorCriteria::RMSEwithNaN;
    m_Name = "RMSEonMeanWithNaN";
    m_FullName = _("Root Mean Square Error on the mean value of the grid, with NaNs management");
    m_Order = Asc;
    m_ScaleBest = 0;
    m_ScaleWorst = NaNFloat;
}

asPredictorCriteriaRMSEonMeanWithNaN::~asPredictorCriteriaRMSEonMeanWithNaN()
{
    //dtor
}

float asPredictorCriteriaRMSEonMeanWithNaN::Assess(const Array2DFloat &refData, const Array2DFloat &evalData, int rowsNb, int colsNb)
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

    float mse = 0, evalMean = 0, refMean = 0;
    float finalsize = (float)refData.size();

    switch (m_LinAlgebraMethod)
    {
        case (asLIN_ALGEBRA_NOVAR):
        case (asLIN_ALGEBRA):
        case (asCOEFF_NOVAR):
        case (asCOEFF):
        {
            for (int i=0; i<rowsNb; i++)
            {
                for (int j=0; j<colsNb; j++)
                {
                    if (!asTools::IsNaN(evalData(i,j)) && !asTools::IsNaN(refData(i,j)))
                    {
                        evalMean += evalData(i,j);
                        refMean += refData(i,j);
                    }
                    else
                    {
                        finalsize--;
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

    wxASSERT(mse>=0);
    wxASSERT(refData.size()>0);

    if (finalsize==0)
    {
        asLogMessage(_("Only NaNs in the criteria calculation."));
        return NaNFloat;
    }

    evalMean /= finalsize;
    refMean /= finalsize;

    mse += (evalMean - refMean) * (evalMean - refMean);

    if (asTools::IsNaN(mse))
    {
        return NaNFloat;
    }

    wxASSERT(mse>=0);
    wxASSERT_MSG(sqrt(mse)>=m_ScaleBest, _("The criteria is below the lower limit..."));

    return sqrt(mse);
}
