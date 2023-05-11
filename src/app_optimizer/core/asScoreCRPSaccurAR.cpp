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

#include "asScoreCRPSaccurAR.h"

#include "asScoreCRPSAR.h"
#include "asScoreCRPSsharpAR.h"

asScoreCRPSaccurAR::asScoreCRPSaccurAR()
    : asScore(asScore::CRPSaccuracyAR, _("CRPS Accuracy Approx Rectangle"),
              _("Continuous Ranked Probability Score Accuracy approximation with the rectangle method"), Asc, 0, NAN) {

}

float asScoreCRPSaccurAR::Assess(float obs, const a1f& values, int nbElements) const {
    wxASSERT(values.size() > 1);
    wxASSERT(nbElements > 0);

    // Check inputs
    if (!CheckObservedValue(obs)) {
        return NAN;
    }
    if (!CheckVectorLength(values, nbElements)) {
        wxLogWarning(_("Problems in a vector length."));
        return NAN;
    }

    asScoreCRPSAR scoreCRPSAR = asScoreCRPSAR();
    float CRPS = scoreCRPSAR.Assess(obs, values, nbElements);
    asScoreCRPSsharpAR scoreCRPSsharpnessAR = asScoreCRPSsharpAR();
    float CRPSsharpness = scoreCRPSsharpnessAR.Assess(obs, values, nbElements);

    return CRPS - CRPSsharpness;
}

bool asScoreCRPSaccurAR::ProcessScoreClimatology(const a1f& refVals, const a1f& climatologyData) {
    return true;
}
