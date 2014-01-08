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
 * Portions Copyright 2008-2013 University of Lausanne.
 * Portions Copyright 2013 Pascal Horton, Terr@num.
 */
 
#include "asForecastScoreMAE.h"

asForecastScoreMAE::asForecastScoreMAE()
:
asForecastScore()
{
    m_Score = asForecastScore::MAE;
    m_Name = _("Mean absolute error");
    m_FullName = _("Mean absolute error");
    m_Order = Asc;
    m_ScaleBest = 0;
    m_ScaleWorst = NaNFloat;
}

asForecastScoreMAE::~asForecastScoreMAE()
{
    //dtor
}

float asForecastScoreMAE::Assess(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements)
{
    wxASSERT(ForcastVals.size()>1);
    wxASSERT(nbElements>0);
    wxASSERT(!asTools::IsNaN(m_Percentile));
    wxASSERT(m_Percentile>0);
    wxASSERT(m_Percentile<1);

    // Create the container to sort the data
    Array1DFloat x(nbElements);

    // Remove the NaNs and copy content
    int nbForecasts = CleanNans(ForcastVals, x, nbElements);
    if(nbForecasts==asNOT_FOUND)
    {
        asLogWarning(_("Only NaNs as inputs in the CRPS processing function."));
        return NaNFloat;
    }
    else if(nbForecasts<=2)
    {
        asLogWarning(_("Not enough elements to process the CRPS."));
        return NaNFloat;
    }

    // Sort the forcast array
    asTools::SortArray(&x[0], &x[nbForecasts-1], Asc);

    float xPercentile = NaNFloat;
    float score = NaNFloat;

    // Containers
    Array1DFloat F(nbForecasts);

    // Parameters for the estimated distribution from Gringorten (a=0.44, b=0.12).
    // Choice based on [Cunnane, C., 1978, Unbiased plotting positions—A review: Journal of Hydrology, v. 37, p. 205–222.]
    // Bontron used a=0.375, b=0.25, that are optimal for a normal distribution
    float irep = 0.44f;
    float nrep = 0.12f;

    // Build the cumulative distribution function for the middle of the x
    float divisor = 1.0f/(nbForecasts+nrep);
    for(float i=0; i<nbForecasts; i++)
    {

        F(i)=(i+1.0f-irep)*divisor; // i+1 as i starts from 0
    }

    // Indices for the left and right part (according to xObs)
    int indLeft = asTools::SortedArraySearchFloor(&F[0], &F[nbForecasts-1], m_Percentile);
    int indRight = asTools::SortedArraySearchCeil(&F[0], &F[nbForecasts-1], m_Percentile);
    wxASSERT(indLeft>=0);
    wxASSERT(indRight>=0);
    wxASSERT(indLeft<=indRight);

    if (indLeft==indRight)
    {
        xPercentile = x[indLeft];
    }
    else
    {
        xPercentile = x(indLeft)+(x(indRight)-x(indLeft))*(m_Percentile-F(indLeft))/(F(indRight)-F(indLeft));
    }

    score = abs(ObservedVal-xPercentile);

    return score;
}

bool asForecastScoreMAE::ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData)
{
    return true;
}
