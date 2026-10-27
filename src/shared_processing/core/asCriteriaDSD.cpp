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

#include "asIncludes.h"

asCriteriaDSD::asCriteriaDSD()
    : asCriteria("DSD", _("Difference in standard deviation (nonspatial)"), Asc) {
    _minPointsNb = 2;
    _canUseInline = true;
}

asCriteriaDSD::~asCriteriaDSD() = default;

float asCriteriaDSD::Assess(const ra2f& refData, const ra2f& evalData, int rowsNb, int colsNb) const {
    wxASSERT(refData.rows() == evalData.rows());
    wxASSERT(refData.cols() == evalData.cols());
    wxASSERT(refData.rows() == rowsNb);
    wxASSERT(refData.cols() == colsNb);
    wxASSERT(evalData.rows() == rowsNb);
    wxASSERT(evalData.cols() == colsNb);

    if (!_checkNaNs || (!refData.hasNaN() && !evalData.hasNaN())) {
        float refStdDev = std::sqrt((refData - refData.mean()).square().sum() / (float)(refData.size() - 1));
        float evalStdDev = std::sqrt((evalData - evalData.mean()).square().sum() / (float)(evalData.size() - 1));

        return std::fabs(refStdDev - evalStdDev);

    } else {
        // Two passes: means and count first, then the centered square sums
        // (keeping the numerically stable two-pass variance).
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

        if (size <= 1) {
            wxLogVerbose(_("Not enough data to process the DSD score."));
            return NAN;
        }

        float refMean = refSum / float(size);
        float evalMean = evalSum / float(size);

        float refSqSum = 0, evalSqSum = 0;
        for (Eigen::Index i = 0; i < n; ++i) {
            if (std::isnan(r[i]) || std::isnan(e[i])) continue;
            float refDiff = r[i] - refMean;
            float evalDiff = e[i] - evalMean;
            refSqSum += refDiff * refDiff;
            evalSqSum += evalDiff * evalDiff;
        }

        float refStdDev = std::sqrt(refSqSum / (float)(size - 1));
        float evalStdDev = std::sqrt(evalSqSum / (float)(size - 1));

        return std::fabs(refStdDev - evalStdDev);
    }
}
