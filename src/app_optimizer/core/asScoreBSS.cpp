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

#include "asScoreBSS.h"
#include "asScoreBS.h"

asScoreBSS::asScoreBSS()
        : asScore(asScore::BSS, _("BS Skill Score"), _("BS Skill Score"), Desc, 1, NaNf, true)
{
}

asScoreBSS::~asScoreBSS()
{
    //dtor
}

float asScoreBSS::Assess(float observedVal, const a1f &forcastVals, int nbElements) const
{
    wxASSERT(forcastVals.size() > 1);
    wxASSERT(nbElements > 0);
    wxASSERT(m_scoreClimatology != 0);

    // Check inputs
    if (!CheckObservedValue(observedVal)) {
        return NaNf;
    }
    if (!CheckVectorLength( forcastVals, nbElements)) {
        wxLogWarning(_("Problems in a vector length."));
        return NaNf;
    }

    // First process the BS and then the skill score
    asScoreBS scoreBS = asScoreBS();
    scoreBS.SetThreshold(GetThreshold());
    scoreBS.SetQuantile(GetQuantile());
    float score = scoreBS.Assess(observedVal, forcastVals, nbElements);
    float skillScore = (score - m_scoreClimatology) / ((float) 0 - m_scoreClimatology);

    return skillScore;
}

bool asScoreBSS::ProcessScoreClimatology(const a1f &refVals, const a1f &climatologyData)
{
    wxASSERT(!asTools::HasNaN(&refVals[0], &refVals[refVals.size() - 1]));
    wxASSERT(!asTools::HasNaN(&climatologyData[0], &climatologyData[climatologyData.size() - 1]));

    // Containers for final results
    a1f scoresClimatology(refVals.size());

    // Set the original score and process
    asScore *score = asScore::GetInstance(asScore::BS);
    score->SetThreshold(GetThreshold());
    score->SetQuantile(GetQuantile());

    for (int iRefTime = 0; iRefTime < refVals.size(); iRefTime++) {
        if (!asTools::IsNaN(refVals(iRefTime))) {
            scoresClimatology(iRefTime) = score->Assess(refVals(iRefTime), climatologyData, climatologyData.size());
        } else {
            scoresClimatology(iRefTime) = NaNf;
        }
    }

    wxDELETE(score);

    m_scoreClimatology = asTools::Mean(&scoresClimatology[0], &scoresClimatology[scoresClimatology.size() - 1]);

    wxLogVerbose(_("Score of the climatology: %g."), m_scoreClimatology);

    return true;
}
