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

#include "asForecastScoreBSS.h"
#include "asForecastScoreBS.h"

asForecastScoreBSS::asForecastScoreBSS()
        : asForecastScore()
{
    m_score = asForecastScore::BSS;
    m_name = _("BS Skill Score");
    m_fullName = _("Brier Skill Score");
    m_order = Desc;
    m_scaleBest = 1;
    m_scaleWorst = NaNf;
    m_usesClimatology = true;
}

asForecastScoreBSS::~asForecastScoreBSS()
{
    //dtor
}

float asForecastScoreBSS::Assess(float ObservedVal, const a1f &ForcastVals, int nbElements) const
{
    wxASSERT(ForcastVals.size() > 1);
    wxASSERT(nbElements > 0);

    wxASSERT(m_scoreClimatology != 0);

    // First process the BS and then the skill score
    asForecastScoreBS scoreBS = asForecastScoreBS();
    scoreBS.SetThreshold(GetThreshold());
    scoreBS.SetQuantile(GetQuantile());
    float score = scoreBS.Assess(ObservedVal, ForcastVals, nbElements);
    float skillScore = (score - m_scoreClimatology) / ((float) 0 - m_scoreClimatology);

    return skillScore;
}

bool asForecastScoreBSS::ProcessScoreClimatology(const a1f &refVals, const a1f &climatologyData)
{
    wxASSERT(!asTools::HasNaN(&refVals[0], &refVals[refVals.size() - 1]));
    wxASSERT(!asTools::HasNaN(&climatologyData[0], &climatologyData[climatologyData.size() - 1]));

    // Containers for final results
    a1f scoresClimatology(refVals.size());

    // Set the original score and process
    asForecastScore *forecastScore = asForecastScore::GetInstance(asForecastScore::BS);
    forecastScore->SetThreshold(GetThreshold());
    forecastScore->SetQuantile(GetQuantile());

    for (int iRefTime = 0; iRefTime < refVals.size(); iRefTime++) {
        if (!asTools::IsNaN(refVals(iRefTime))) {
            scoresClimatology(iRefTime) = forecastScore->Assess(refVals(iRefTime), climatologyData,
                                                                 climatologyData.size());
        } else {
            scoresClimatology(iRefTime) = NaNf;
        }
    }

    wxDELETE(forecastScore);

    m_scoreClimatology = asTools::Mean(&scoresClimatology[0], &scoresClimatology[scoresClimatology.size() - 1]);

    wxLogVerbose(_("Score of the climatology: %g."), m_scoreClimatology);

    return true;
}
