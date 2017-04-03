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

#include "asForecastScoreCRPSS.h"
#include "asForecastScoreCRPSAR.h"

asForecastScoreCRPSS::asForecastScoreCRPSS()
        : asForecastScore()
{
    m_score = asForecastScore::CRPSS;
    m_name = _("CRPS Skill Score");
    m_fullName = _(
            "Continuous Ranked Probability Score Skill Score based on the approximation with the rectangle method");
    m_order = Desc;
    m_scaleBest = 1;
    m_scaleWorst = NaNFloat;
    m_usesClimatology = true;
}

asForecastScoreCRPSS::~asForecastScoreCRPSS()
{
    //dtor
}

float asForecastScoreCRPSS::Assess(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements) const
{
    wxASSERT(ForcastVals.size() > 1);
    wxASSERT(nbElements > 0);

    wxASSERT(m_scoreClimatology != 0);

    // First process the CRPS and then the skill score
    asForecastScoreCRPSAR scoreCRPS = asForecastScoreCRPSAR();
    scoreCRPS.SetThreshold(GetThreshold());
    scoreCRPS.SetQuantile(GetQuantile());
    float score = scoreCRPS.Assess(ObservedVal, ForcastVals, nbElements);
    float skillScore = (score - m_scoreClimatology) / ((float) 0 - m_scoreClimatology);

    return skillScore;
}

bool asForecastScoreCRPSS::ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData)
{
    wxASSERT(!asTools::HasNaN(&refVals[0], &refVals[refVals.size() - 1]));
    wxASSERT(!asTools::HasNaN(&climatologyData[0], &climatologyData[climatologyData.size() - 1]));

    // Containers for final results
    Array1DFloat scoresClimatology(refVals.size());

    // Set the original score and process
    asForecastScore *forecastScore = asForecastScore::GetInstance(asForecastScore::CRPSAR);
    forecastScore->SetThreshold(GetThreshold());
    forecastScore->SetQuantile(GetQuantile());

    for (int i_reftime = 0; i_reftime < refVals.size(); i_reftime++) {
        if (!asTools::IsNaN(refVals(i_reftime))) {
            scoresClimatology(i_reftime) = forecastScore->Assess(refVals(i_reftime), climatologyData,
                                                                 climatologyData.size());
        } else {
            scoresClimatology(i_reftime) = NaNFloat;
        }
    }

    wxDELETE(forecastScore);

    m_scoreClimatology = asTools::Mean(&scoresClimatology[0], &scoresClimatology[scoresClimatology.size() - 1]);

    wxLogVerbose(_("Score of the climatology: %g."), m_scoreClimatology);

    return true;
}
