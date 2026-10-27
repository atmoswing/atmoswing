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

#include "asCriteriaMD.h"

#include "asIncludes.h"

asCriteriaMD::asCriteriaMD()
    : asCriteria("MD", _("Mean Absolute Difference"), Asc) {
    _canUseInline = true;
}

asCriteriaMD::~asCriteriaMD() = default;

float asCriteriaMD::Assess(const ra2f& refData, const ra2f& evalData, int rowsNb, int colsNb) const {
    wxASSERT(refData.rows() == evalData.rows());
    wxASSERT(refData.cols() == evalData.cols());

    if (!_checkNaNs || (!refData.hasNaN() && !evalData.hasNaN())) {
        return (evalData - refData).abs().sum() / (float)refData.size();

    } else {
        // Single pass: sum the absolute differences and count the valid pairs at once
        const float* r = refData.data();
        const float* e = evalData.data();
        const auto n = refData.size();
        float sad = 0;
        int size = 0;
        for (Eigen::Index i = 0; i < n; ++i) {
            float diff = e[i] - r[i];
            if (std::isnan(diff)) continue;
            sad += std::fabs(diff);
            size++;
        }

        if (size == 0) {
            wxLogVerbose(_("Only NaNs in the MD criteria calculation."));
            return NAN;
        }

        return sad / (float)size;
    }
}
