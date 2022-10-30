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

#include "asCriteriaRSE.h"

asCriteriaRSE::asCriteriaRSE()
    : asCriteria("RSE", _("Root Squared Error"), Asc) {
    m_canUseInline = true;
}

asCriteriaRSE::~asCriteriaRSE() = default;

float asCriteriaRSE::Assess(const a2f& refData, const a2f& evalData, int rowsNb, int colsNb) const {
    wxASSERT(refData.rows() == evalData.rows());
    wxASSERT(refData.cols() == evalData.cols());

    float se = 0;

    if (!m_checkNaNs || (!refData.hasNaN() && !evalData.hasNaN())) {
        se = (evalData - refData).pow(2).sum();

    } else {
        a2f diff = evalData - refData;

        int size = (!diff.isNaN()).count();
        if (size == 0) {
            wxLogVerbose(_("Only NaNs in the RSE criteria calculation."));
            return NaNf;
        }

        se = ((diff.isNaN()).select(0, diff)).pow(2).sum();
    }

    return std::sqrt(se);
}
