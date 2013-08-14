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
 
#ifndef ASPREDICTORCRITERIAS1weights_H
#define ASPREDICTORCRITERIAS1weights_H

#include <asIncludes.h>
#include <asPredictorCriteria.h>

class asPredictorCriteriaS1weights: public asPredictorCriteria
{
public:

    /** Default constructor
     * \param criteria The chosen criteria
     */
    asPredictorCriteriaS1weights(int linAlgebraMethod=asCOEFF_NOVAR);

    /** Default destructor */
    ~asPredictorCriteriaS1weights();

    /** Process the Criteria
     * \param refData The target day
     * \param evalData The day to assess
     * \return The Criteria
     */
    float Assess(const Array2DFloat &refData, const Array2DFloat &evalData, int rowsNb=0, int colsNb=0);


protected:

private:
    float m_wU;
    float m_wV;

};

#endif
