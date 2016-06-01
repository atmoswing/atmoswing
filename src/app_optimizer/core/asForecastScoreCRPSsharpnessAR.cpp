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

#include "asForecastScoreCRPSsharpnessAR.h"
#include "asForecastScoreCRPSAR.h"

asForecastScoreCRPSsharpnessAR::asForecastScoreCRPSsharpnessAR()
        : asForecastScore()
{
    m_score = asForecastScore::CRPSsharpnessAR;
    m_name = _("CRPS Accuracy Approx Rectangle");
    m_fullName = _("Continuous Ranked Probability Score Accuracy approximation with the rectangle method");
    m_order = Asc;
    m_scaleBest = 0;
    m_scaleWorst = NaNFloat;
}

asForecastScoreCRPSsharpnessAR::~asForecastScoreCRPSsharpnessAR()
{
    //dtor
}

float asForecastScoreCRPSsharpnessAR::Assess(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements) const
{
    wxASSERT(ForcastVals.size() > 1);
    wxASSERT(nbElements > 0);

    // Check the element numbers vs vector length and the observed value
    if (!CheckInputs(0, ForcastVals, nbElements)) {
        asLogWarning(_("The inputs are not conform in the CRPS processing function"));
        return NaNFloat;
    }

    // The median
    float xmed = 0;

    // Create the container to sort the data
    Array1DFloat x(nbElements);

    // Remove the NaNs and copy content
    int nbForecasts = CleanNans(ForcastVals, x, nbElements);

    // Sort the forcast array
    asTools::SortArray(&x[0], &x[nbForecasts - 1], Asc);

    // Indices for the left and right part (according to the median) of the distribution
    float mid = ((float) nbForecasts - 1) / (float) 2;
    int indLeftEnd = floor(mid);
    int indRightStart = ceil(mid);

    // Get the median value
    if (indLeftEnd != indRightStart) {
        xmed = x(indLeftEnd) + (x(indRightStart) - x(indLeftEnd)) * 0.5;
    } else {
        xmed = x(indLeftEnd);
    }

    asForecastScoreCRPSAR scoreCRPSAR = asForecastScoreCRPSAR();
    float CRPSsharpness = scoreCRPSAR.Assess(xmed, x, nbElements);

    return CRPSsharpness;
}

bool asForecastScoreCRPSsharpnessAR::ProcessScoreClimatology(const Array1DFloat &refVals,
                                                             const Array1DFloat &climatologyData)
{
    return true;
}

