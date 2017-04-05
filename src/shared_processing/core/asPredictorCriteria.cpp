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
 */

#include "asPredictorCriteria.h"
#include "asPredictorCriteriaMD.h"
#include "asPredictorCriteriaNMD.h"
#include "asPredictorCriteriaMRDtoMax.h"
#include "asPredictorCriteriaMRDtoMean.h"
#include "asPredictorCriteriaRMSE.h"
#include "asPredictorCriteriaNRMSE.h"
#include "asPredictorCriteriaRMSEwithNaN.h"
#include "asPredictorCriteriaRMSEonMeanWithNaN.h"
#include "asPredictorCriteriaRSE.h"
#include "asPredictorCriteriaS1.h"
#include "asPredictorCriteriaNS1.h"
#include "asPredictorCriteriaS1grads.h"
#include "asPredictorCriteriaNS1grads.h"
#include "asPredictorCriteriaSAD.h"
#include "asDataPredictor.h"


asPredictorCriteria::asPredictorCriteria(int linAlgebraMethod)
{
    m_linAlgebraMethod = linAlgebraMethod;
    m_canUseInline = false;
    m_criteria = Undefined;
    m_scaleBest = NaNf;
    m_scaleWorst = NaNf;
    m_dataMin = NaNf;
    m_dataMax = NaNf;
    m_needsDataRange = false;
}

asPredictorCriteria::~asPredictorCriteria()
{
    //dtor
}

asPredictorCriteria *asPredictorCriteria::GetInstance(Criteria criteriaEnum, int linAlgebraMethod)
{
    switch (criteriaEnum) {
        case (S1): {
            asPredictorCriteria *criteria = new asPredictorCriteriaS1(linAlgebraMethod);
            return criteria;
        }
        case (NS1): {
            asPredictorCriteria *criteria = new asPredictorCriteriaNS1(linAlgebraMethod);
            return criteria;
        }
        case (S1grads): {
            asPredictorCriteria *criteria = new asPredictorCriteriaS1grads(linAlgebraMethod);
            return criteria;
        }
        case (NS1grads): {
            asPredictorCriteria *criteria = new asPredictorCriteriaNS1grads(linAlgebraMethod);
            return criteria;
        }
        case (SAD): {
            asPredictorCriteria *criteria = new asPredictorCriteriaSAD(linAlgebraMethod);
            return criteria;
        }
        case (MD): {
            asPredictorCriteria *criteria = new asPredictorCriteriaMD(linAlgebraMethod);
            return criteria;
        }
        case (NMD): {
            asPredictorCriteria *criteria = new asPredictorCriteriaNMD(linAlgebraMethod);
            return criteria;
        }
        case (MRDtoMax): {
            asPredictorCriteria *criteria = new asPredictorCriteriaMRDtoMax(linAlgebraMethod);
            return criteria;
        }
        case (MRDtoMean): {
            asPredictorCriteria *criteria = new asPredictorCriteriaMRDtoMean(linAlgebraMethod);
            return criteria;
        }
        case (RMSE): {
            asPredictorCriteria *criteria = new asPredictorCriteriaRMSE(linAlgebraMethod);
            return criteria;
        }
        case (NRMSE): {
            asPredictorCriteria *criteria = new asPredictorCriteriaNRMSE(linAlgebraMethod);
            return criteria;
        }
        case (RMSEwithNaN): {
            asPredictorCriteria *criteria = new asPredictorCriteriaRMSEwithNaN(linAlgebraMethod);
            return criteria;
        }
        case (RMSEonMeanWithNaN): {
            asPredictorCriteria *criteria = new asPredictorCriteriaRMSEonMeanWithNaN(linAlgebraMethod);
            return criteria;
        }
        case (RSE): {
            asPredictorCriteria *criteria = new asPredictorCriteriaRSE(linAlgebraMethod);
            return criteria;
        }
        default: {
            wxLogError(_("The predictor criteria was not correctly defined."));
            asPredictorCriteria *criteria = new asPredictorCriteriaSAD(linAlgebraMethod);
            return criteria;
        }
    }
}

asPredictorCriteria *asPredictorCriteria::GetInstance(const wxString &criteriaString, int linAlgebraMethod)
{
    if (criteriaString.CmpNoCase("S1") == 0) {
        asPredictorCriteria *criteria = new asPredictorCriteriaS1(linAlgebraMethod);
        return criteria;
    } else if (criteriaString.CmpNoCase("NS1") == 0) {
        asPredictorCriteria *criteria = new asPredictorCriteriaNS1(linAlgebraMethod);
        return criteria;
    } else if (criteriaString.CmpNoCase("S1grads") == 0) {
        asPredictorCriteria *criteria = new asPredictorCriteriaS1grads(linAlgebraMethod);
        return criteria;
    } else if (criteriaString.CmpNoCase("NS1grads") == 0) {
        asPredictorCriteria *criteria = new asPredictorCriteriaNS1grads(linAlgebraMethod);
        return criteria;
    } else if (criteriaString.CmpNoCase("SAD") == 0) {
        asPredictorCriteria *criteria = new asPredictorCriteriaSAD(linAlgebraMethod);
        return criteria;
    } else if (criteriaString.CmpNoCase("MD") == 0) {
        asPredictorCriteria *criteria = new asPredictorCriteriaMD(linAlgebraMethod);
        return criteria;
    } else if (criteriaString.CmpNoCase("NMD") == 0) {
        asPredictorCriteria *criteria = new asPredictorCriteriaNMD(linAlgebraMethod);
        return criteria;
    } else if (criteriaString.CmpNoCase("MRDtoMax") == 0) {
        asPredictorCriteria *criteria = new asPredictorCriteriaMRDtoMax(linAlgebraMethod);
        return criteria;
    } else if (criteriaString.CmpNoCase("MRDtoMean") == 0) {
        asPredictorCriteria *criteria = new asPredictorCriteriaMRDtoMean(linAlgebraMethod);
        return criteria;
    } else if (criteriaString.CmpNoCase("RMSE") == 0) {
        asPredictorCriteria *criteria = new asPredictorCriteriaRMSE(linAlgebraMethod);
        return criteria;
    } else if (criteriaString.CmpNoCase("NRMSE") == 0) {
        asPredictorCriteria *criteria = new asPredictorCriteriaNRMSE(linAlgebraMethod);
        return criteria;
    } else if (criteriaString.CmpNoCase("RMSEwithNaN") == 0) {
        asPredictorCriteria *criteria = new asPredictorCriteriaRMSEwithNaN(linAlgebraMethod);
        return criteria;
    } else if (criteriaString.CmpNoCase("RMSEonMeanWithNaN") == 0) {
        asPredictorCriteria *criteria = new asPredictorCriteriaRMSEonMeanWithNaN(linAlgebraMethod);
        return criteria;
    } else if (criteriaString.CmpNoCase("RSE") == 0) {
        asPredictorCriteria *criteria = new asPredictorCriteriaRSE(linAlgebraMethod);
        return criteria;
    } else {
        wxLogError(_("The predictor criteria was not correctly defined."));
        asPredictorCriteria *criteria = new asPredictorCriteriaSAD(linAlgebraMethod);
        return criteria;
    }
}

void asPredictorCriteria::SetDataRange(const asDataPredictor *data)
{
    m_dataMin = data->GetMinValue();
    m_dataMax = data->GetMaxValue();
}

void asPredictorCriteria::SetDataRange(float minValue, float maxValue)
{
    m_dataMin = minValue;
    m_dataMax = maxValue;
}