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

#include "asCriteria.h"
#include "asCriteriaMD.h"
#include "asCriteriaNMD.h"
#include "asCriteriaMRDtoMax.h"
#include "asCriteriaMRDtoMean.h"
#include "asCriteriaRMSE.h"
#include "asCriteriaNRMSE.h"
#include "asCriteriaRMSEwithNaN.h"
#include "asCriteriaRMSEonMeanWithNaN.h"
#include "asCriteriaRSE.h"
#include "asCriteriaS1.h"
#include "asCriteriaNS1.h"
#include "asCriteriaS1grads.h"
#include "asCriteriaNS1grads.h"
#include "asCriteriaSAD.h"
#include "asPredictor.h"


asCriteria::asCriteria()
        : m_criteria(CriteriaUndefined),
          m_name(wxEmptyString),
          m_fullName(wxEmptyString),
          m_order(Asc),
          m_needsDataRange(false),
          m_dataMin(NaNf),
          m_dataMax(NaNf),
          m_scaleBest(NaNf),
          m_scaleWorst(NaNf),
          m_canUseInline(false)
{

}

asCriteria::~asCriteria()
{
    //dtor
}

asCriteria *asCriteria::GetInstance(Criteria criteriaEnum)
{
    switch (criteriaEnum) {
        case (S1): {
            asCriteria *criteria = new asCriteriaS1();
            return criteria;
        }
        case (NS1): {
            asCriteria *criteria = new asCriteriaNS1();
            return criteria;
        }
        case (S1grads): {
            asCriteria *criteria = new asCriteriaS1grads();
            return criteria;
        }
        case (NS1grads): {
            asCriteria *criteria = new asCriteriaNS1grads();
            return criteria;
        }
        case (SAD): {
            asCriteria *criteria = new asCriteriaSAD();
            return criteria;
        }
        case (MD): {
            asCriteria *criteria = new asCriteriaMD();
            return criteria;
        }
        case (NMD): {
            asCriteria *criteria = new asCriteriaNMD();
            return criteria;
        }
        case (MRDtoMax): {
            asCriteria *criteria = new asCriteriaMRDtoMax();
            return criteria;
        }
        case (MRDtoMean): {
            asCriteria *criteria = new asCriteriaMRDtoMean();
            return criteria;
        }
        case (RMSE): {
            asCriteria *criteria = new asCriteriaRMSE();
            return criteria;
        }
        case (NRMSE): {
            asCriteria *criteria = new asCriteriaNRMSE();
            return criteria;
        }
        case (RMSEwithNaN): {
            asCriteria *criteria = new asCriteriaRMSEwithNaN();
            return criteria;
        }
        case (RMSEonMeanWithNaN): {
            asCriteria *criteria = new asCriteriaRMSEonMeanWithNaN();
            return criteria;
        }
        case (RSE): {
            asCriteria *criteria = new asCriteriaRSE();
            return criteria;
        }
        default: {
            wxLogError(_("The predictor criteria was not correctly defined."));
            asCriteria *criteria = new asCriteriaSAD();
            return criteria;
        }
    }
}

asCriteria *asCriteria::GetInstance(const wxString &criteriaString)
{
    if (criteriaString.CmpNoCase("S1") == 0) {
        asCriteria *criteria = new asCriteriaS1();
        return criteria;
    } else if (criteriaString.CmpNoCase("NS1") == 0) {
        asCriteria *criteria = new asCriteriaNS1();
        return criteria;
    } else if (criteriaString.CmpNoCase("S1grads") == 0) {
        asCriteria *criteria = new asCriteriaS1grads();
        return criteria;
    } else if (criteriaString.CmpNoCase("NS1grads") == 0) {
        asCriteria *criteria = new asCriteriaNS1grads();
        return criteria;
    } else if (criteriaString.CmpNoCase("SAD") == 0) {
        asCriteria *criteria = new asCriteriaSAD();
        return criteria;
    } else if (criteriaString.CmpNoCase("MD") == 0) {
        asCriteria *criteria = new asCriteriaMD();
        return criteria;
    } else if (criteriaString.CmpNoCase("NMD") == 0) {
        asCriteria *criteria = new asCriteriaNMD();
        return criteria;
    } else if (criteriaString.CmpNoCase("MRDtoMax") == 0) {
        asCriteria *criteria = new asCriteriaMRDtoMax();
        return criteria;
    } else if (criteriaString.CmpNoCase("MRDtoMean") == 0) {
        asCriteria *criteria = new asCriteriaMRDtoMean();
        return criteria;
    } else if (criteriaString.CmpNoCase("RMSE") == 0) {
        asCriteria *criteria = new asCriteriaRMSE();
        return criteria;
    } else if (criteriaString.CmpNoCase("NRMSE") == 0) {
        asCriteria *criteria = new asCriteriaNRMSE();
        return criteria;
    } else if (criteriaString.CmpNoCase("RMSEwithNaN") == 0) {
        asCriteria *criteria = new asCriteriaRMSEwithNaN();
        return criteria;
    } else if (criteriaString.CmpNoCase("RMSEonMeanWithNaN") == 0) {
        asCriteria *criteria = new asCriteriaRMSEonMeanWithNaN();
        return criteria;
    } else if (criteriaString.CmpNoCase("RSE") == 0) {
        asCriteria *criteria = new asCriteriaRSE();
        return criteria;
    } else {
        wxLogError(_("The predictor criteria was not correctly defined."));
        asCriteria *criteria = new asCriteriaSAD();
        return criteria;
    }
}

void asCriteria::SetDataRange(const asPredictor *data)
{
    m_dataMin = data->GetMinValue();
    m_dataMax = data->GetMaxValue();
}

void asCriteria::SetDataRange(float minValue, float maxValue)
{
    m_dataMin = minValue;
    m_dataMax = maxValue;
}