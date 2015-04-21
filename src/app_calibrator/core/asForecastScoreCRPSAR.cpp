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
 
#include "asForecastScoreCRPSAR.h"

asForecastScoreCRPSAR::asForecastScoreCRPSAR()
:
asForecastScore()
{
    m_score = asForecastScore::CRPSAR;
    m_name = _("CRPS Approx Rectangle");
    m_fullName = _("Continuous Ranked Probability Score approximation with the rectangle method");
    m_order = Asc;
    m_scaleBest = 0;
    m_scaleWorst = NaNFloat;
}

asForecastScoreCRPSAR::~asForecastScoreCRPSAR()
{
    //dtor
}

float asForecastScoreCRPSAR::Assess(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements)
{
    wxASSERT(ForcastVals.size()>1);
    wxASSERT(nbElements>0);

    // Check the element numbers vs vector length and the observed value
    if(!CheckInputs(ObservedVal, ForcastVals, nbElements))
    {
        asLogWarning(_("The inputs are not conform in the CRPS processing function"));
        return NaNFloat;
    }

    // Create the container to sort the data
    Array1DFloat x(nbElements);
    float xObs = ObservedVal;

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

    float CRPS = 0;

    // Cumulative frequency
    Array1DFloat F = asTools::GetCumulativeFrequency(nbForecasts);

    // Indices for the left and right part (according to xObs) of the distribution
    int indLeftStart = 0;
    int indLeftEnd = 0;
    int indRightStart = nbForecasts-1;
    int indRightEnd = nbForecasts-1;

    // Find FxObs, fix xObs and integrate beyond limits
    float FxObs, xObsCorr = xObs;
    if (xObs<=x[0]) // If xObs before the distribution
    {
        indRightStart = 0;
        FxObs = 0;
        xObsCorr = x[indRightStart];
        CRPS += (xObsCorr-xObs);
    }
    else if (xObs>x[nbForecasts-1]) // If xObs after the distribution
    {
        indLeftEnd = nbForecasts-1;
        FxObs = 1;
        xObsCorr = x[indLeftEnd];
        CRPS += (xObs-xObsCorr);
    }
    else // If xObs inside the distribution
    {
        indLeftEnd = asTools::SortedArraySearchFloor(&x[0], &x[nbForecasts-1], xObs);
        if((indLeftEnd!=nbForecasts-1) & (indLeftEnd!=asNOT_FOUND) & (indLeftEnd!=asOUT_OF_RANGE))
        {
            indRightStart = indLeftEnd+1;
            if(x(indRightStart)==x(indLeftEnd))
            {
                FxObs = (F(indLeftEnd)+F(indRightStart))*0.5;
            }
            else
            {
                FxObs = F(indLeftEnd)+(F(indRightStart)-F(indLeftEnd))*(xObs-x(indLeftEnd))/(x(indRightStart)-x(indLeftEnd));
            }
            // Integrate the CRPS around FxObs
            CRPS += (FxObs*FxObs-F(indLeftEnd)*F(indLeftEnd))*(xObsCorr-0.5*(x[indLeftEnd]+xObsCorr)); // Left
            CRPS += ((1-FxObs)*(1-FxObs)-(1-F(indRightStart))*(1-F(indRightStart)))*(0.5*(xObsCorr+x[indRightStart])-xObsCorr); // Right
        }
    }

    // Integrate on the left part below F(0). First slice from the bottom.
    CRPS += (F(indLeftStart)*F(indLeftStart))*(xObsCorr-x[indLeftStart]);

    // Integrate on the left part
    for (int i=indLeftStart; i<indLeftEnd; i++)
    {
        CRPS += (F(i+1)*F(i+1)-F(i)*F(i))*(xObsCorr-0.5f*(x[i]+x[i+1]));
    }

    // Integrate on the right part
    for (int i=indRightStart; i<indRightEnd; i++)
    {
        CRPS += ((1.0f-F(i))*(1.0f-F(i))-(1.0f-F(i+1))*(1.0f-F(i+1)))*(0.5f*(x[i]+x[i+1])-xObsCorr);
    }

    // Integrate on the right part above F(indRightEnd). First slice from the bottom.
    CRPS += ((1-F(indRightEnd))*(1-F(indRightEnd)))*(x[indRightEnd]-xObsCorr);

    return CRPS;
}

bool asForecastScoreCRPSAR::ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData)
{
    return true;
}
