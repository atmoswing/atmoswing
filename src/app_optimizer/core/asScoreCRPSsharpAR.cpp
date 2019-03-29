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

#include "asScoreCRPSsharpAR.h"
#include "asScoreCRPSAR.h"

asScoreCRPSsharpAR::asScoreCRPSsharpAR()
        : asScore(asScore::CRPSsharpnessAR, _("CRPS Accuracy Approx Rectangle"),
                  _("Continuous Ranked Probability Score Accuracy approximation with the rectangle method"), Asc, 0,
                  NaNf)
{

}

asScoreCRPSsharpAR::~asScoreCRPSsharpAR()
{
    //dtor
}

float asScoreCRPSsharpAR::Assess(float obs, const a1f &values, int nbElements) const
{
    wxASSERT(values.size() > 1);
    wxASSERT(nbElements > 0);

    // Check inputs
    if (!CheckObservedValue(obs)) {
        return NaNf;
    }
    if (!CheckVectorLength(values, nbElements)) {
        wxLogWarning(_("Problems in a vector length."));
        return NaNf;
    }

    // The median
    float xmed = 0;

    // Create the container to sort the data
    a1f x(nbElements);

    // Remove the NaNs and copy content
    int nbPredict = CleanNans(values, x, nbElements);

    // Sort the forcast array
    asSortArray(&x[0], &x[nbPredict - 1], Asc);

    // Indices for the left and right part (according to the median) of the distribution
    float mid = ((float) nbPredict - 1) / (float) 2;
    int indLeftEnd = floor(mid);
    int indRightStart = ceil(mid);

    // Get the median value
    if (indLeftEnd != indRightStart) {
        xmed = x(indLeftEnd) + (x(indRightStart) - x(indLeftEnd)) * 0.5;
    } else {
        xmed = x(indLeftEnd);
    }

    asScoreCRPSAR scoreCRPSAR = asScoreCRPSAR();
    float CRPSsharpness = scoreCRPSAR.Assess(xmed, x, nbElements);

    return CRPSsharpness;
}

bool asScoreCRPSsharpAR::ProcessScoreClimatology(const a1f &refVals, const a1f &climatologyData)
{
    return true;
}

