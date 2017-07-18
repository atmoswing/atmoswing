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
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#include "asThreadPreprocessGradients.h"

#include <asPredictor.h>


asThreadPreprocessGradients::asThreadPreprocessGradients(vva2f *gradients,
                                                             std::vector<asPredictor *> predictors, int start,
                                                             int end)
        : asThread(),
          m_pGradients(gradients),
          m_pPredictors(predictors)
{
    m_type = asThread::PreprocessorGradients;
    m_start = start;
    m_end = end;
}

asThreadPreprocessGradients::~asThreadPreprocessGradients()
{

}

wxThread::ExitCode asThreadPreprocessGradients::Entry()
{
    int rowsNb = m_pPredictors[0]->GetLatPtsnb();
    int colsNb = m_pPredictors[0]->GetLonPtsnb();
    int timeSize = m_pPredictors[0]->GetTimeSize();
    int membersNb = m_pPredictors[0]->GetMembersNb();

    a1f tmpgrad = a1f::Constant((rowsNb - 1) * colsNb + rowsNb * (colsNb - 1), NaNf);

    for (int iTime = 0; iTime < timeSize; iTime++) {
        int counter = 0;
        for (int iMem = 0; iMem < membersNb; iMem++) {

            // Vertical gradients
            for (int iRow = 0; iRow < rowsNb - 1; iRow++) {
                for (int iCol = 0; iCol < colsNb; iCol++) {
                    tmpgrad(counter) = m_pPredictors[0]->GetData()[iTime][iMem](iRow + 1, iCol) -
                                       m_pPredictors[0]->GetData()[iTime][iMem](iRow, iCol);
                    counter++;
                }
            }

            // Horizontal gradients
            for (int iRow = 0; iRow < rowsNb; iRow++) {
                for (int iCol = 0; iCol < colsNb - 1; iCol++) {
                    // The matrix is transposed to be coherent with the dimensions
                    tmpgrad(counter) = m_pPredictors[0]->GetData()[iTime][iMem](iRow, iCol + 1) -
                                       m_pPredictors[0]->GetData()[iTime][iMem](iRow, iCol);
                    counter++;
                }
            }

            (*m_pGradients)[iTime][iMem] = tmpgrad.transpose();
        }
    }

    return (wxThread::ExitCode) 0;
}
