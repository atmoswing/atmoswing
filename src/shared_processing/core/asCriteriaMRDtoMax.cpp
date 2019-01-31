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

#include "asCriteriaMRDtoMax.h"

asCriteriaMRDtoMax::asCriteriaMRDtoMax()
        : asCriteria("MRDtoMax", _("Mean Relative Differences to the Maximum"), Asc)
{
    m_scaleBest = 0;
    m_scaleWorst = NaNf;
    m_canUseInline = true;
}

asCriteriaMRDtoMax::~asCriteriaMRDtoMax()
{
    //dtor
}

float asCriteriaMRDtoMax::Assess(const a2f &refData, const a2f &evalData, int rowsNb, int colsNb) const
{
    wxASSERT(refData.rows() == evalData.rows());
    wxASSERT(refData.cols() == evalData.cols());

    float rd = 0;

    for (int i = 0; i < rowsNb; i++) {
        for (int j = 0; j < colsNb; j++) {
            if (wxMax(std::abs(evalData(i, j)), std::abs(refData(i, j))) > 0) {
                rd += std::abs(evalData(i, j) - refData(i, j)) /
                      wxMax(std::abs(evalData(i, j)), std::abs(refData(i, j)));
            } else {
                if (std::abs(evalData(i, j) - refData(i, j)) != 0) {
                    wxLogWarning(_("Division by zero in the predictor criteria."));
                    return NaNf;
                }
            }
        }
    }

    return rd / (float) refData.size();

}
