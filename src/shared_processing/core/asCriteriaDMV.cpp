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

#include "asCriteriaDMV.h"

#include "asIncludes.h"

asCriteriaDMV::asCriteriaDMV()
    : asCriteria("DMV", _("Absolute difference in mean value (nonspatial)"), Asc) {
    _canUseInline = true;
}

asCriteriaDMV::~asCriteriaDMV() = default;

float asCriteriaDMV::Assess(const ra2f& refData, const ra2f& evalData, int rowsNb, int colsNb) const {
    wxASSERT(refData.rows() == evalData.rows());
    wxASSERT(refData.cols() == evalData.cols());
    wxASSERT(refData.rows() == rowsNb);
    wxASSERT(refData.cols() == colsNb);
    wxASSERT(evalData.rows() == rowsNb);
    wxASSERT(evalData.cols() == colsNb);

    if (!_checkNaNs || (!refData.hasNaN() && !evalData.hasNaN())) {
        return std::fabs(refData.mean() - evalData.mean());

    } else {
        // Single pass: sum both fields and count the valid pairs at once
        const float* r = refData.data();
        const float* e = evalData.data();
        const auto n = refData.size();
        float refSum = 0, evalSum = 0;
        int size = 0;
        for (Eigen::Index i = 0; i < n; ++i) {
            if (std::isnan(r[i]) || std::isnan(e[i])) continue;
            refSum += r[i];
            evalSum += e[i];
            size++;
        }

        if (size == 0) {
            wxLogVerbose(_("Only NaNs in the DMV criteria calculation."));
            return NAN;
        }

        return std::fabs(refSum / float(size) - evalSum / float(size));
    }
}
