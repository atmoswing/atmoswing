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

#include "asPredictorCriteriaNS1.h"

asPredictorCriteriaNS1::asPredictorCriteriaNS1(int linAlgebraMethod)
        : asPredictorCriteriaS1(linAlgebraMethod)
{
    m_criteria = asPredictorCriteria::NS1;
    m_name = "NS1";
    m_fullName = _("Normalized Teweles-Wobus");
    m_order = Asc;
}

asPredictorCriteriaNS1::~asPredictorCriteriaNS1()
{
    //dtor
}

float asPredictorCriteriaNS1::Assess(const Array2DFloat &refData, const Array2DFloat &evalData, int rowsNb, int colsNb) const
{
    return asPredictorCriteriaS1::Assess(refData, evalData, rowsNb, colsNb) / m_scaleWorst;
}
