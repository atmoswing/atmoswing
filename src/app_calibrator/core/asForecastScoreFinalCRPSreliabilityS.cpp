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
 
#include "asForecastScoreFinalCRPSreliabilityS.h"

asForecastScoreFinalCRPSreliabilityS::asForecastScoreFinalCRPSreliabilityS(Period period)
:
asForecastScoreFinal(period)
{
    m_Has2DArrayArgument = true;
    m_UsesClimatology = true;
}

asForecastScoreFinalCRPSreliabilityS::asForecastScoreFinalCRPSreliabilityS(const wxString& periodString)
:
asForecastScoreFinal(periodString)
{
    m_Has2DArrayArgument = true;
    m_UsesClimatology = true;
}

asForecastScoreFinalCRPSreliabilityS::~asForecastScoreFinalCRPSreliabilityS()
{
    //dtor
}

float asForecastScoreFinalCRPSreliabilityS::Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray)
{
    asLogError(_("The CRPS reliability score needs a 2D array as input !"));
    return NaNFloat;
}


float asForecastScoreFinalCRPSreliabilityS::Assess(Array1DFloat &targetDates, Array2DFloat &forecastScores, asTimeArray &timeArray)
{
    wxASSERT(!asTools::IsNaN(m_ScoreClimatology));

    // Process the final score
    asForecastScoreFinal* forecastScoreFinal = asForecastScoreFinal::GetInstance("CRPSreliability", "Total");

    float reliability = 0;
    reliability = forecastScoreFinal->Assess(targetDates, forecastScores, timeArray);
    
    wxDELETE(forecastScoreFinal);

    float skillScore = (reliability-m_ScoreClimatology) / (0.0f-m_ScoreClimatology);

    return skillScore;
}
