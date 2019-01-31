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
 * Portions Copyright 2016 Pascal Horton, University of Bern.
 */

#include "asCriteriaNRMSE.h"

asCriteriaNRMSE::asCriteriaNRMSE()
        : asCriteriaRMSE()
{
    m_name = "NRMSE";
    m_fullName = _("Normalized Root Mean Square Error");
    m_needsDataRange = true;
}

asCriteriaNRMSE::~asCriteriaNRMSE()
{
    //dtor
}

float asCriteriaNRMSE::Assess(const a2f &refData, const a2f &evalData, int rowsNb, int colsNb) const
{
    if (m_dataMax == m_dataMin) {
        return 0;
    }

    return asCriteriaRMSE::Assess(refData, evalData, rowsNb, colsNb) / (m_dataMax - m_dataMin);
}
