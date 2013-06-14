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
 
#include "asPredictorCriteria.h"
#include "asPredictorCriteriaMD.h"
#include "asPredictorCriteriaMRDtoMax.h"
#include "asPredictorCriteriaMRDtoMean.h"
#include "asPredictorCriteriaRMSE.h"
#include "asPredictorCriteriaRMSEwithNaN.h"
#include "asPredictorCriteriaRMSEonMeanWithNaN.h"
#include "asPredictorCriteriaRSE.h"
#include "asPredictorCriteriaS1.h"
#include "asPredictorCriteriaS1grads.h"
#include "asPredictorCriteriaS1weights.h"
#include "asPredictorCriteriaSAD.h"


asPredictorCriteria::asPredictorCriteria(int linAlgebraMethod)
{
    m_LinAlgebraMethod = linAlgebraMethod;
}

asPredictorCriteria::~asPredictorCriteria()
{
    //dtor
}

asPredictorCriteria* asPredictorCriteria::GetInstance(Criteria criteriaEnum, int linAlgebraMethod)
{
    switch (criteriaEnum)
    {
        case (S1):
        {
            asPredictorCriteria* criteria = new asPredictorCriteriaS1(linAlgebraMethod);
            return criteria;
        }
        case (S1grads):
        {
            asPredictorCriteria* criteria = new asPredictorCriteriaS1grads(linAlgebraMethod);
            return criteria;
        }
        case (S1weights):
        {
            asPredictorCriteria* criteria = new asPredictorCriteriaS1weights(linAlgebraMethod);
            return criteria;
        }
        case (SAD):
        {
            asPredictorCriteria* criteria = new asPredictorCriteriaSAD(linAlgebraMethod);
            return criteria;
        }
        case (MD):
        {
            asPredictorCriteria* criteria = new asPredictorCriteriaMD(linAlgebraMethod);
            return criteria;
        }
        case (MRDtoMax):
        {
            asPredictorCriteria* criteria = new asPredictorCriteriaMRDtoMax(linAlgebraMethod);
            return criteria;
        }
        case (MRDtoMean):
        {
            asPredictorCriteria* criteria = new asPredictorCriteriaMRDtoMean(linAlgebraMethod);
            return criteria;
        }
        case (RMSE):
        {
            asPredictorCriteria* criteria = new asPredictorCriteriaRMSE(linAlgebraMethod);
            return criteria;
        }
        case (RMSEwithNaN):
        {
            asPredictorCriteria* criteria = new asPredictorCriteriaRMSEwithNaN(linAlgebraMethod);
            return criteria;
        }
        case (RMSEonMeanWithNaN):
        {
            asPredictorCriteria* criteria = new asPredictorCriteriaRMSEonMeanWithNaN(linAlgebraMethod);
            return criteria;
        }
        case (RSE):
        {
            asPredictorCriteria* criteria = new asPredictorCriteriaRSE(linAlgebraMethod);
            return criteria;
        }
        default:
        {
            asLogError(_("The predictor criteria was not correctly defined."));
            asPredictorCriteria* criteria = new asPredictorCriteriaSAD(linAlgebraMethod);
            return criteria;
        }
    }
    return NULL;

}

asPredictorCriteria* asPredictorCriteria::GetInstance(const wxString &criteriaString, int linAlgebraMethod)
{
    if (criteriaString.CmpNoCase("S1")==0)
    {
        asPredictorCriteria* criteria = new asPredictorCriteriaS1(linAlgebraMethod);
        return criteria;
    }
    else if (criteriaString.CmpNoCase("S1grads")==0)
    {
        asPredictorCriteria* criteria = new asPredictorCriteriaS1grads(linAlgebraMethod);
        return criteria;
    }
    else if (criteriaString.CmpNoCase("S1weights")==0)
    {
        asPredictorCriteria* criteria = new asPredictorCriteriaS1weights(linAlgebraMethod);
        return criteria;
    }
    else if (criteriaString.CmpNoCase("SAD")==0)
    {
        asPredictorCriteria* criteria = new asPredictorCriteriaSAD(linAlgebraMethod);
        return criteria;
    }
    else if (criteriaString.CmpNoCase("MD")==0)
    {
        asPredictorCriteria* criteria = new asPredictorCriteriaMD(linAlgebraMethod);
        return criteria;
    }
    else if (criteriaString.CmpNoCase("MRDtoMax")==0)
    {
        asPredictorCriteria* criteria = new asPredictorCriteriaMRDtoMax(linAlgebraMethod);
        return criteria;
    }
    else if (criteriaString.CmpNoCase("MRDtoMean")==0)
    {
        asPredictorCriteria* criteria = new asPredictorCriteriaMRDtoMean(linAlgebraMethod);
        return criteria;
    }
    else if (criteriaString.CmpNoCase("RMSE")==0)
    {
        asPredictorCriteria* criteria = new asPredictorCriteriaRMSE(linAlgebraMethod);
        return criteria;
    }
    else if (criteriaString.CmpNoCase("RMSEwithNaN")==0)
    {
        asPredictorCriteria* criteria = new asPredictorCriteriaRMSEwithNaN(linAlgebraMethod);
        return criteria;
    }
    else if (criteriaString.CmpNoCase("RMSEonMeanWithNaN")==0)
    {
        asPredictorCriteria* criteria = new asPredictorCriteriaRMSEonMeanWithNaN(linAlgebraMethod);
        return criteria;
    }
    else if (criteriaString.CmpNoCase("RSE")==0)
    {
        asPredictorCriteria* criteria = new asPredictorCriteriaRSE(linAlgebraMethod);
        return criteria;
    }
    else
    {
        asLogError(_("The predictor criteria was not correctly defined."));
        asPredictorCriteria* criteria = new asPredictorCriteriaSAD(linAlgebraMethod);
        return criteria;
    }
    asLogError(_("The predictor criteria was not correctly defined."));
    return NULL;
}

void asPredictorCriteria::DeleteArray(std::vector < asPredictorCriteria* > criteria)
{
    if (criteria.size()==0) return;

    for (int i=0; (unsigned)i<criteria.size(); i++)
    {
        wxDELETE(criteria[i]);
    }
}
