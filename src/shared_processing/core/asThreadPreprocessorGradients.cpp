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

#include "asThreadPreprocessorGradients.h"

#include <asDataPredictor.h>


asThreadPreprocessorGradients::asThreadPreprocessorGradients(VArray2DFloat* gradients, std::vector < asDataPredictor* > predictors, int start, int end)
:
asThread(), 
m_pGradients(gradients), 
m_pPredictors(predictors)
{
    m_Status = Initializing;

    m_Type = asThread::PreprocessorGradients;
    
    m_Start = start;
    m_End = end;

    m_Status = Waiting;
}

asThreadPreprocessorGradients::~asThreadPreprocessorGradients()
{

}

wxThread::ExitCode asThreadPreprocessorGradients::Entry()
{
    m_Status = Working;

    int rowsNb = m_pPredictors[0]->GetLatPtsnb();
    int colsNb = m_pPredictors[0]->GetLonPtsnb();
    int timeSize = m_pPredictors[0]->GetSizeTime();

    Array1DFloat tmpgrad = Array1DFloat::Constant((rowsNb-1)*colsNb+rowsNb*(colsNb-1), NaNFloat);

    for (int i_time=0; i_time<timeSize; i_time++)
    {
        int counter=0;

        // Vertical gradients
        for (int i_row=0; i_row<rowsNb-1; i_row++)
        {
            for (int i_col=0; i_col<colsNb; i_col++)
            {
                tmpgrad(counter) = m_pPredictors[0]->GetData()[i_time](i_row+1,i_col)-m_pPredictors[0]->GetData()[i_time](i_row,i_col);
                counter++;
            }
        }

        // Horizontal gradients
        for (int i_row=0; i_row<rowsNb; i_row++)
        {
            for (int i_col=0; i_col<colsNb-1; i_col++)
            {
                // The matrix is transposed to be coherent with the dimensions
                tmpgrad(counter) = m_pPredictors[0]->GetData()[i_time](i_row,i_col+1)-m_pPredictors[0]->GetData()[i_time](i_row,i_col);
                counter++;
            }
        }

        //gradients.push_back(tmpgrad.transpose());
        (*m_pGradients)[i_time] = tmpgrad.transpose();
    }

    m_Status = Done;

    return 0;
}
