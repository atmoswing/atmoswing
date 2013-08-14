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
 
#ifndef asThreadPreprocessorGradients_H
#define asThreadPreprocessorGradients_H

#include <asThread.h>
#include <asIncludes.h>

class asDataPredictor;


class asThreadPreprocessorGradients: public asThread
{
public:
    /** Default constructor */
    asThreadPreprocessorGradients(VArray2DFloat* gradients, std::vector < asDataPredictor >* predictors, int start, int end);
    /** Default destructor */
    virtual ~asThreadPreprocessorGradients();

    virtual ExitCode Entry();


protected:
private:
    std::vector < asDataPredictor >* m_pPredictors;
    VArray2DFloat* m_pGradients;
    int m_Start;
    int m_End;

};

#endif // asThreadPreprocessorGradients_H
