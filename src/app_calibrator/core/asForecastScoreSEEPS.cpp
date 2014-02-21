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
 
#include "asForecastScoreSEEPS.h"

asForecastScoreSEEPS::asForecastScoreSEEPS()
:
asForecastScore()
{
    m_Score = asForecastScore::SEEPS;
    m_Name = _("Stable equitable error in probability space");
    m_FullName = _("Stable equitable error in probability space");
    m_Order = NoOrder;
    m_ScaleBest = NaNFloat;
    m_ScaleWorst = NaNFloat;
    m_p1 = NaNFloat;
    m_p3 = NaNFloat;
    m_ThresNull = 0.2f;
    m_ThresHigh = NaNFloat;
    m_UsesClimatology = true;
}

asForecastScoreSEEPS::~asForecastScoreSEEPS()
{
    //dtor
}

float asForecastScoreSEEPS::Assess(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements)
{
    wxASSERT(ForcastVals.size()>1);
    wxASSERT(nbElements>0);
    wxASSERT(!asTools::IsNaN(m_p1));
    wxASSERT(!asTools::IsNaN(m_p3));
    wxASSERT(!asTools::IsNaN(m_ThresHigh));
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

    float forecastedVal = NaNFloat;
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
        forecastedVal = x[indLeft];
    }
    else
    {
        forecastedVal = x(indLeft)+(x(indRight)-x(indLeft))*(m_Percentile-F(indLeft))/(F(indRight)-F(indLeft));
    }

    float fcstV = forecastedVal;
    float obsV = ObservedVal;

	// Forecasted 1, observed 1
	if (fcstV<=m_ThresNull && obsV<=m_ThresNull)
    {
        score = 0.0f;
	}
	// Forecasted 2, observed 1
	else if (fcstV<=m_ThresNull && obsV>m_ThresNull && obsV<=m_ThresHigh)
    {
        score = 0.5*(1.0/(1.0-m_p1));
	}
	// Forecasted 3, observed 1
	else if (fcstV<=m_ThresNull && obsV>m_ThresHigh)
    {
        score = 0.5*((1.0/m_p3)+(1.0/1.0-m_p1));
	}
	// Forecasted 1, observed 2
	else if (fcstV>m_ThresNull && fcstV<=m_ThresHigh && obsV<=m_ThresNull)
    {
        score = 0.5*(1.0/m_p1);
	}
	// Forecasted 2, observed 2
	else if (fcstV>m_ThresNull && fcstV<=m_ThresHigh && obsV>m_ThresNull && obsV<=m_ThresHigh)
    {
        score = 0.0f;
	}
	// Forecasted 3, observed 2
	else if (fcstV>m_ThresNull && fcstV<=m_ThresHigh && obsV>m_ThresHigh)
    {
        score = 0.5*(1.0/m_p3);
	}
	// Forecasted 1, observed 3
	else if (fcstV>m_ThresHigh && obsV<=m_ThresNull)
    {
        score = 0.5*((1.0/m_p1)+(1.0/(1.0-m_p3)));
	}
	// Forecasted 2, observed 3
	else if (fcstV>m_ThresHigh && obsV>m_ThresNull && obsV<=m_ThresHigh)
    {
        score = 0.5*(1.0/(1.0-m_p3));
	}
	// Forecasted 3, observed 3
	else if (fcstV>m_ThresHigh && obsV>m_ThresHigh)
    {
        score = 0.0f;
	}

    return score;
}

bool asForecastScoreSEEPS::ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData)
{
    wxASSERT(!asTools::HasNaN(&refVals[0], &refVals[refVals.size()-1]));
    wxASSERT(!asTools::HasNaN(&climatologyData[0], &climatologyData[climatologyData.size()-1]));

    Array1DFloat climatologyDataSorted = climatologyData;
    asTools::SortArray(&climatologyDataSorted[0], &climatologyDataSorted[climatologyDataSorted.size()-1], Asc);

    // Find first value above the lower threshold
    int rowAboveThreshold1 = asTools::SortedArraySearchFloor(&climatologyDataSorted[0], &climatologyDataSorted[climatologyDataSorted.size()-1], m_Threshold);
    while (climatologyDataSorted[rowAboveThreshold1]<=m_Threshold)
    {
        rowAboveThreshold1++;
    }

    // Process probability (without processing the empirical frequencies...). Do not substract 1 because it is in 0 basis.
    m_p1 = (float)rowAboveThreshold1/(float)climatologyDataSorted.size();

    m_p3 = (1-m_p1)/3.0f;

    int indexThreshold2 = climatologyDataSorted.size()*(m_p1+2*m_p3);
    m_ThresHigh = climatologyDataSorted[indexThreshold2];

    return true;
}
