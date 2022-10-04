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
 * Portions Copyright 2016-2021 Pascal Horton, University of Bern.
 */

#include "asCriteria.h"

#include "asCriteriaDMV.h"
#include "asCriteriaDSD.h"
#include "asCriteriaMD.h"
#include "asCriteriaRMSE.h"
#include "asCriteriaRSE.h"
#include "asCriteriaS0.h"
#include "asCriteriaS0obs.h"
#include "asCriteriaS1.h"
#include "asCriteriaS1G.h"
#include "asCriteriaS1grads.h"
#include "asCriteriaS1obs.h"
#include "asCriteriaS2.h"
#include "asCriteriaS2grads.h"
#include "asCriteriaSAD.h"
#include "asPredictor.h"

asCriteria::asCriteria(const wxString& name, const wxString& fullname, Order order)
    : m_name(name),
      m_fullName(fullname),
      m_order(order),
      m_minPointsNb(1),
      m_scaleBest(0),
      m_scaleWorst(Inff),
      m_canUseInline(false),
      m_checkNaNs(true) {}

asCriteria::~asCriteria() = default;

asCriteria* asCriteria::GetInstance(const wxString& criteriaString) {
    if (criteriaString.CmpNoCase("S1") == 0 || criteriaString.CmpNoCase("S1s") == 0) {
        // Teweles-Wobus
        return new asCriteriaS1();
    } else if (criteriaString.CmpNoCase("S1obs") == 0) {
        // Teweles-Wobus with division by the observation
        return new asCriteriaS1obs();
    } else if (criteriaString.CmpNoCase("S1G") == 0 || criteriaString.CmpNoCase("S1sG") == 0) {
        // Teweles-Wobus with Gaussian weights
        return new asCriteriaS1G();
    } else if (criteriaString.CmpNoCase("S1grads") == 0) {
        // Teweles-Wobus on gradients
        return new asCriteriaS1grads();
    } else if (criteriaString.CmpNoCase("S2") == 0 || criteriaString.CmpNoCase("S2s") == 0) {
        // Derivative of Teweles-Wobus
        return new asCriteriaS2();
    } else if (criteriaString.CmpNoCase("S2grads") == 0) {
        // Derivative of Teweles-Wobus on gradients
        return new asCriteriaS2grads();
    } else if (criteriaString.CmpNoCase("S0") == 0) {
        // Teweles-Wobus on raw data
        return new asCriteriaS0();
    } else if (criteriaString.CmpNoCase("S0obs") == 0) {
        // Teweles-Wobus on raw data with division by the observation
        return new asCriteriaS0obs();
    } else if (criteriaString.CmpNoCase("SAD") == 0) {
        // Sum of absolute differences
        return new asCriteriaSAD();
    } else if (criteriaString.CmpNoCase("MD") == 0) {
        // Mean absolute difference
        return new asCriteriaMD();
    } else if (criteriaString.CmpNoCase("RMSE") == 0) {
        // Root mean square error
        return new asCriteriaRMSE();
    } else if (criteriaString.CmpNoCase("RSE") == 0) {
        // Root square error (According to Bontron. Should not be used !)
        return new asCriteriaRSE();
    } else if (criteriaString.CmpNoCase("DMV") == 0) {
        // Difference in mean value (nonspatial)
        return new asCriteriaDMV();
    } else if (criteriaString.CmpNoCase("DSD") == 0) {
        // Difference in standard deviation (nonspatial)
        return new asCriteriaDSD();
    } else {
        wxLogError(_("The predictor criteria was not correctly defined (%s)."), criteriaString);
        return nullptr;
    }
}

void asCriteria::CheckNaNs(const asPredictor* ptor1, const asPredictor* ptor2) {
    if (wxFileConfig::Get()->ReadBool("/General/SkipNansCheck", false)) {
        m_checkNaNs = false;
        return;
    }

    if (!ptor1->HasNaN() && !ptor1->HasNaN()) {
        m_checkNaNs = false;
    }
}

a2f asCriteria::GetGauss2D(int nY, int nX) {
    float A = 1.0;
    auto x0 = (nX + 1.0f) / 2.0f;
    auto y0 = (nY + 1.0f) / 2.0f;

    float a = 1 / (0.5f * std::pow(x0, 2));
    float c = 1 / (0.5f * std::pow(y0, 2));

    a2f X = Eigen::RowVectorXf::LinSpaced(nX, 1, nX).replicate(nY, 1);
    a2f Y = Eigen::VectorXf::LinSpaced(nY, 1, nY).replicate(1, nX);

    return A * (-(a * (X - x0).pow(2) + c * (Y - y0).pow(2))).exp();
}