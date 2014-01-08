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
 
#include "asForecastScoreCRPSARloadF0.h"

asForecastScoreCRPSARloadF0::asForecastScoreCRPSARloadF0()
:
asForecastScore()
{
    m_Score = asForecastScore::CRPSAR;
    m_Name = _("CRPS Approx Rectangle");
    m_FullName = _("Continuous Ranked Probability Score approximation with the rectangle method");
    m_Order = Asc;
    m_ScaleBest = 0;
    m_ScaleWorst = NaNFloat;
}

asForecastScoreCRPSARloadF0::~asForecastScoreCRPSARloadF0()
{
    //dtor
}

float asForecastScoreCRPSARloadF0::Assess(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements)
{
    asLogError(_("The class asForecastScoreCRPSARloadF0 should not be used !"));
    return NaNFloat;
}

float asForecastScoreCRPSARloadF0::Assess(float ObservedVal, float F0, const Array1DFloat &ForcastVals, int nbElements)
{
    wxASSERT(ForcastVals.size()>1);
    wxASSERT(nbElements>0);
    wxASSERT(F0>=0);
    wxASSERT(F0<=1);

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

    // Parameters for the estimated distribution from Gringorten (a=0.44, b=0.12).
    // Choice based on [Cunnane, C., 1978, Unbiased plotting positions—A review: Journal of Hydrology, v. 37, p. 205–222.]
    // Bontron used a=0.375, b=0.25, that are optimal for a normal distribution
    float irep = 0.44f;
    float nrep = 0.12f;

    // Change the values for unit testing to compare to the results from Grenoble
    if (g_UnitTesting)
    {
        irep = 0.375;
        nrep = 0.25;
    }

    // Identify the last 0
    int indexLastZero = -1;
    for(int i=0; i<nbElements; i++)
    {
        if (x[i]==0)
        {
            indexLastZero = i;
        }
    }

    // Original distribution
    Array1DFloat Fana(nbForecasts);
    float divisor = 1.0f/(nbForecasts+nrep);
    for(float i=0; i<nbForecasts; i++)
    {
        Fana(i)=(i+1.0f-irep)*divisor; // i+1 as i starts from 0
    }

    // Display F(0)analog
    bool dispF0analog = false;
    if (dispF0analog)
    {
        if (indexLastZero>=0)
        {
            wxLogWarning("%f", Fana(indexLastZero));
        }
        else
        {
            wxLogWarning("%d", 0);
        }
    }

    // Containers
    Array1DFloat F;

    if (indexLastZero==-1) // No 0
    {
        if (F0>0)
        {
            // Add 2 points from 0 to F(0)
            int newlength = nbForecasts+2;
            Array1DFloat x2(newlength);
            x2[0] = 0;
            x2[1] = 0;
            x2.segment(2,nbForecasts) = x;
            x = x2;

            F.resize(newlength);
            F(0)=(1.0f-irep)*divisor;
            if(F0>F(0))
            {
                F(1) = F0;
            }
            else
            {
                F(1) = F(0);
            }

            for(float i=2; i<newlength; i++)
            {
                int indexPrev = i-2+indexLastZero;
                F(i)=F0+(Fana(indexPrev)-0.0)*((1-F0)/(1-0.0));
            }

            nbForecasts = newlength;
        }
        else
        {
            // Do nothing
            F = Fana;
        }
    }
    else if (indexLastZero==nbElements-1) // Only 0
    {
        // Do nothing
        F = Fana;
    }
    else
    {
        // Add 2 points from 0 to F(0)
        int newlength = nbForecasts-indexLastZero+1;
        Array1DFloat x2(newlength);
        x2[0] = 0;
        x2[1] = 0;
        x2.segment(2,newlength-2) = x.segment(indexLastZero+1,newlength-2);
        x = x2;

        F.resize(newlength);
        F(0)=(1.0f-irep)*divisor;
        if(F0>F(0))
        {
            F(1) = F0;
        }
        else
        {
            F(1) = F(0);
        }

        //float ratio = (1.0f-F0)/(1.0f-Fana(indexLastZero)); // Fana(0) = 0 if no null rainfall
        for(float i=2; i<newlength; i++)
        {
            int indexPrev = i-2+indexLastZero;
            F(i)=F0+(Fana(indexPrev)-Fana(indexLastZero))*((1-F0)/(1-Fana(indexLastZero)));
        }

        nbForecasts = newlength;
    }

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

bool asForecastScoreCRPSARloadF0::ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData)
{
    return true;
}
