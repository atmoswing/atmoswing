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
#include "asCriteriaRMSE.h"
#include "asCriteriaNRMSE.h"
#include "asCriteriaRSE.h"
#include "asCriteriaS0.h"
#include "asCriteriaS1.h"
#include "asCriteriaNS1.h"
#include "asCriteriaS1G.h"
#include "asCriteriaS1grads.h"
#include "asCriteriaNS1grads.h"
#include "asCriteriaS2.h"
#include "asCriteriaS2grads.h"
#include "asCriteriaNS2.h"
#include "asCriteriaSAD.h"
#include "asCriteriaDSD.h"
#include "asCriteriaNDSD.h"
#include "asCriteriaDMV.h"
#include "asCriteriaNDMV.h"
#include "asPredictor.h"


asCriteria::asCriteria(const wxString &name, const wxString &fullname, Order order)
        : m_name(name),
          m_fullName(fullname),
          m_order(order),
          m_minPointsNb(1),
          m_needsDataRange(false),
          m_dataMin(NaNf),
          m_dataMax(NaNf),
          m_scaleBest(0),
          m_scaleWorst(Inff),
          m_canUseInline(false),
          m_checkNaNs(true)
{

}

asCriteria::~asCriteria() = default;

asCriteria *asCriteria::GetInstance(const wxString &criteriaString)
{
    if (criteriaString.CmpNoCase("S1") == 0 || criteriaString.CmpNoCase("S1s") == 0) {
        // Teweles-Wobus
        asCriteria *criteria = new asCriteriaS1();
        return criteria;
    } else if (criteriaString.CmpNoCase("NS1") == 0 || criteriaString.CmpNoCase("NS1s") == 0) {
        // Normalized Teweles-Wobus
        asCriteria *criteria = new asCriteriaNS1();
        return criteria;
    } else if (criteriaString.CmpNoCase("S1G") == 0 || criteriaString.CmpNoCase("S1sG") == 0) {
        // Teweles-Wobus with Gaussian weights
		asCriteria *criteria = new asCriteriaS1G();
        return criteria;
    } else if (criteriaString.CmpNoCase("S1grads") == 0) {
        // Teweles-Wobus on gradients
        asCriteria *criteria = new asCriteriaS1grads();
        return criteria;
    } else if (criteriaString.CmpNoCase("NS1grads") == 0) {
        // Normalized Teweles-Wobus on gradients
        asCriteria *criteria = new asCriteriaNS1grads();
        return criteria;
    } else if (criteriaString.CmpNoCase("S2") == 0 || criteriaString.CmpNoCase("S2s") == 0) {
        // Derivative of Teweles-Wobus
        asCriteria *criteria = new asCriteriaS2();
        return criteria;
    } else if (criteriaString.CmpNoCase("NS2") == 0 || criteriaString.CmpNoCase("NS2s") == 0) {
        // Normalized derivative of Teweles-Wobus
        asCriteria *criteria = new asCriteriaNS2();
        return criteria;
    } else if (criteriaString.CmpNoCase("S2grads") == 0) {
        // Derivative of Teweles-Wobus on gradients
        asCriteria *criteria = new asCriteriaS2grads();
        return criteria;
    } else if (criteriaString.CmpNoCase("NS2grads") == 0) {
        // Normalized derivative of Teweles-Wobus on gradients
        asCriteria *criteria = new asCriteriaNS1grads();
        return criteria;
    } else if (criteriaString.CmpNoCase("S0") == 0) {
        // Teweles-Wobus on raw data
        asCriteria *criteria = new asCriteriaS0();
        return criteria;
    } else if (criteriaString.CmpNoCase("SAD") == 0) {
        // Sum of absolute differences
        asCriteria *criteria = new asCriteriaSAD();
        return criteria;
    } else if (criteriaString.CmpNoCase("MD") == 0) {
        // Mean absolute difference
        asCriteria *criteria = new asCriteriaMD();
        return criteria;
    } else if (criteriaString.CmpNoCase("NMD") == 0) {
        // Normalized Mean difference
        asCriteria *criteria = new asCriteriaNMD();
        return criteria;
    } else if (criteriaString.CmpNoCase("RMSE") == 0) {
        // Root mean square error
        asCriteria *criteria = new asCriteriaRMSE();
        return criteria;
    } else if (criteriaString.CmpNoCase("NRMSE") == 0) {
        // Normalized Root mean square error (min-max approach)
        asCriteria *criteria = new asCriteriaNRMSE();
        return criteria;
    } else if (criteriaString.CmpNoCase("RSE") == 0) {
        // Root square error (According to Bontron. Should not be used !)
        asCriteria *criteria = new asCriteriaRSE();
        return criteria;
    } else if (criteriaString.CmpNoCase("DMV") == 0) {
        // Difference in mean value (nonspatial)
        asCriteria *criteria = new asCriteriaDMV();
        return criteria;
    } else if (criteriaString.CmpNoCase("NDMV") == 0) {
        // Normalized difference in mean value (nonspatial)
        asCriteria *criteria = new asCriteriaNDMV();
        return criteria;
    } else if (criteriaString.CmpNoCase("DSD") == 0) {
        // Difference in standard deviation (nonspatial)
        asCriteria *criteria = new asCriteriaDSD();
        return criteria;
    } else if (criteriaString.CmpNoCase("NDSD") == 0) {
        // Normalized difference in standard deviation (nonspatial)
        asCriteria *criteria = new asCriteriaNDSD();
        return criteria;
    } else {
        wxLogError(_("The predictor criteria was not correctly defined (%s)."), criteriaString);
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

void asCriteria::CheckNaNs(const asPredictor *ptor1, const asPredictor *ptor2)
{
    if (!ptor1->HasNaN() && !ptor1->HasNaN()) {
        m_checkNaNs = false;
    }
}

a2f asCriteria::GetGauss2D(int nY, int nX)
{
    float A = 1.0;
    auto x0 = (nX + 1.0f) / 2.0f;
    auto y0 = (nY + 1.0f) / 2.0f;

    float a = 1 / (0.5f * std::pow(x0, 2));
    float c = 1 / (0.5f * std::pow(y0, 2));

    a2f X = Eigen::RowVectorXf::LinSpaced(nX, 1, nX).replicate(nY, 1);
    a2f Y = Eigen::VectorXf::LinSpaced(nY, 1, nY).replicate(1, nX);

    return A * (-(a * (X - x0).pow(2) + c * (Y - y0).pow(2))).exp();
}