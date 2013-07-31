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
 
#include "asThreadPreprocessorGradients.h"

#include <asDataPredictor.h>


asThreadPreprocessorGradients::asThreadPreprocessorGradients(VArray2DFloat* gradients, std::vector < asDataPredictor >* predictors, int start, int end)
:
asThread()
{
    m_Status = Initializing;

    m_Type = asThread::PreprocessorGradients;

    m_pPredictors = predictors;
    m_pGradients = gradients;
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

    int rowsNb = (*m_pPredictors)[0].GetLatPtsnb();
    int colsNb = (*m_pPredictors)[0].GetLonPtsnb();
    int timeSize = (*m_pPredictors)[0].GetSizeTime();

    Array1DFloat tmpgrad = Array1DFloat::Constant((rowsNb-1)*colsNb+rowsNb*(colsNb-1), NaNFloat);

    for (int i_time=0; i_time<timeSize; i_time++)
    {
        int counter=0;

        // Vertical gradients
        for (int i_row=0; i_row<rowsNb-1; i_row++)
        {
            for (int i_col=0; i_col<colsNb; i_col++)
            {
                tmpgrad(counter) = (*m_pPredictors)[0].GetData()[i_time](i_row+1,i_col)-(*m_pPredictors)[0].GetData()[i_time](i_row,i_col);
                counter++;
            }
        }

        // Horizontal gradients
        for (int i_row=0; i_row<rowsNb; i_row++)
        {
            for (int i_col=0; i_col<colsNb-1; i_col++)
            {
                // The matrix is transposed to be coherent with the dimensions
                tmpgrad(counter) = (*m_pPredictors)[0].GetData()[i_time](i_row,i_col+1)-(*m_pPredictors)[0].GetData()[i_time](i_row,i_col);
                counter++;
            }
        }

        //gradients.push_back(tmpgrad.transpose());
        (*m_pGradients)[i_time] = tmpgrad.transpose();
    }

    m_Status = Done;

    return 0;
}
