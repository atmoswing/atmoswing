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
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2014 Pascal Horton, Terr@num.
 * Portions Copyright 2014 Renaud Marty, DREAL.
 */
 
#include "asForecastScoreFinalCRPSpotential.h"

asForecastScoreFinalCRPSpotential::asForecastScoreFinalCRPSpotential(Period period)
:
asForecastScoreFinal(period)
{
    m_has2DArrayArgument = true;
}

asForecastScoreFinalCRPSpotential::asForecastScoreFinalCRPSpotential(const wxString& periodString)
:
asForecastScoreFinal(periodString)
{
    m_has2DArrayArgument = true;
}

asForecastScoreFinalCRPSpotential::~asForecastScoreFinalCRPSpotential()
{
    //dtor
}

float asForecastScoreFinalCRPSpotential::Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray)
{
    asLogError(_("The CRPS score needs a 2D array as input !"));
    return NaNFloat;
}


float asForecastScoreFinalCRPSpotential::Assess(Array1DFloat &targetDates, Array2DFloat &forecastScores, asTimeArray &timeArray)
{
    wxASSERT(targetDates.rows()>1);
    wxASSERT(forecastScores.rows()>1);
    wxASSERT(forecastScores.cols()>1);

    // Process average on every column
    Array1DFloat means = forecastScores.colwise().mean();

    // Extract corresponding arrays
    int binsNbs = means.size()/3;
    Array1DFloat alpha = means.segment(0,binsNbs);
    Array1DFloat beta = means.segment(binsNbs,binsNbs);
    Array1DFloat g = means.segment(2*binsNbs,binsNbs);

    // Compute o (coefficent-wise operations)
    Array1DFloat o = beta / (alpha+beta);
    wxASSERT(o.size()==alpha.size());

    // Create the p array (coefficent-wise operations)
    Array1DFloat p = Array1DFloat::LinSpaced(binsNbs, 0, binsNbs-1);
    p = p/(binsNbs-1);

    // Compute CRPS potential
    float potential = 0;

    for (int i=0; i<binsNbs; i++)
    {
        if (!asTools::IsNaN(g[i]) && !asTools::IsInf(g[i]) && !asTools::IsNaN(o[i]) && !asTools::IsInf(o[i]))
        {
            potential += g[i]*o[i]*(1-o[i]);
        }
    }

    return potential;
}
