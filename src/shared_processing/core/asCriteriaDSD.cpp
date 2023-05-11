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
 * Portions Copyright 2018 Pascal Horton, University of Bern.
 */

#include "asCriteriaDSD.h"

asCriteriaDSD::asCriteriaDSD()
    : asCriteria("DSD", _("Difference in standard deviation (nonspatial)"), Asc) {
    m_minPointsNb = 2;
    m_canUseInline = true;
}

asCriteriaDSD::~asCriteriaDSD() = default;

float asCriteriaDSD::Assess(const a2f& refData, const a2f& evalData, int rowsNb, int colsNb) const {
    wxASSERT(refData.rows() == evalData.rows());
    wxASSERT(refData.cols() == evalData.cols());
    wxASSERT(refData.rows() == rowsNb);
    wxASSERT(refData.cols() == colsNb);
    wxASSERT(evalData.rows() == rowsNb);
    wxASSERT(evalData.cols() == colsNb);

    if (!m_checkNaNs || (!refData.hasNaN() && !evalData.hasNaN())) {
        float refStdDev = std::sqrt((refData - refData.mean()).square().sum() / (float)(refData.size() - 1));
        float evalStdDev = std::sqrt((evalData - evalData.mean()).square().sum() / (float)(evalData.size() - 1));

        return std::fabs(refStdDev - evalStdDev);

    } else {
        int size = (!evalData.isNaN() && !refData.isNaN()).count();
        if (size <= 1) {
            wxLogVerbose(_("Not enough data to process the DSD score."));
            return NAN;
        }

        float refMean = ((!evalData.isNaN() && !refData.isNaN()).select(refData, 0)).sum() / float(size);
        float evalMean = ((!evalData.isNaN() && !refData.isNaN()).select(evalData, 0)).sum() / float(size);

        a2f refDataDiff = (!evalData.isNaN() && !refData.isNaN()).select(refData - refMean, 0);
        a2f evalDataDiff = (!evalData.isNaN() && !refData.isNaN()).select(evalData - evalMean, 0);

        float refStdDev = std::sqrt((refDataDiff).square().sum() / (float)(size - 1));
        float evalStdDev = std::sqrt((evalDataDiff).square().sum() / (float)(size - 1));

        return std::fabs(refStdDev - evalStdDev);
    }
}
