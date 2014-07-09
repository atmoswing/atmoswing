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
 */
 
#include "asForecastScoreCRPSHersbachDecompS.h"
#include "asForecastScoreCRPSHersbachDecomp.h"
#include "asForecastScoreFinal.h"

asForecastScoreCRPSHersbachDecompS::asForecastScoreCRPSHersbachDecompS()
:
asForecastScore()
{
    m_Score = asForecastScore::CRPSHersbachDecompS;
    m_Name = _("Skill Score of the CRPS decomposition by Hersbach");
    m_FullName = _("Skill Score of the Hersbach decomposition of the Continuous Ranked Probability Score (Hersbach, 2000)");
    m_Order = Desc;
    m_ScaleBest = 1;
    m_ScaleWorst = NaNFloat;
    m_SingleValue = false;
    m_UsesClimatology = true;
}

asForecastScoreCRPSHersbachDecompS::~asForecastScoreCRPSHersbachDecompS()
{
    //dtor
}

float asForecastScoreCRPSHersbachDecompS::Assess(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements)
{
    asLogError(_("The Hersbach decomposition of the CRPS cannot provide a single score value !"));
    return NaNFloat;
}

Array1DFloat asForecastScoreCRPSHersbachDecompS::AssessOnArray(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements)
{
    wxASSERT(ForcastVals.size()>1);
    wxASSERT(nbElements>0);

    wxASSERT(m_ScoreClimatology!=0);

    // First process the CRPS and then the skill score
    asForecastScoreCRPSHersbachDecomp forecastScore = asForecastScoreCRPSHersbachDecomp();
    forecastScore.SetThreshold(GetThreshold());
    forecastScore.SetPercentile(GetPercentile());
    Array1DFloat scoreVal = forecastScore.AssessOnArray(ObservedVal, ForcastVals, nbElements);

    return scoreVal;
}

bool asForecastScoreCRPSHersbachDecompS::ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData)
{
    wxASSERT(!asTools::HasNaN(&refVals[0], &refVals[refVals.size()-1]));
    wxASSERT(!asTools::HasNaN(&climatologyData[0], &climatologyData[climatologyData.size()-1]));

    // Containers for final results
    Array2DFloat forecastScores(refVals.size(), 3*(climatologyData.size()+1));

    // Set the original score and process
    asForecastScore* forecastScore = asForecastScore::GetInstance(asForecastScore::CRPSHersbachDecomp);
    forecastScore->SetThreshold(GetThreshold());
    forecastScore->SetPercentile(GetPercentile());

    for (int i_targtime=0; i_targtime<refVals.size(); i_targtime++)
    {
        if (!asTools::IsNaN(refVals(i_targtime)))
        {
            forecastScores.row(i_targtime) = forecastScore->AssessOnArray(refVals(i_targtime), climatologyData, climatologyData.size());
        }
        else
        {
            forecastScores.row(i_targtime) = Array1DFloat::Ones(3*(climatologyData.size()+1))*NaNFloat;
        }
    }

    wxDELETE(forecastScore);

    // Process the final score
    asForecastScoreFinal* forecastScoreFinal = asForecastScoreFinal::GetInstance("CRPSreliability", "Total");
    Array1DFloat targetDates;
    asTimeArray timeArray;

    m_ScoreClimatology = forecastScoreFinal->Assess(targetDates, forecastScores, timeArray);
    
    wxDELETE(forecastScoreFinal);

    asLogMessage(wxString::Format(_("Score of the climatology: %g."), m_ScoreClimatology));

    return true;
}
