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

#include "asScoreCRPSS.h"
#include "asScoreCRPSAR.h"

asScoreCRPSS::asScoreCRPSS()
        : asScore(asScore::CRPSS, _("CRPS Skill Score"),
                  _("Continuous Ranked Probability Score Skill Score based on the approximation with the rectangle method"),
                  Desc, 1, NaNf, true)
{

}

asScoreCRPSS::~asScoreCRPSS()
{
    //dtor
}

float asScoreCRPSS::Assess(float ObservedVal, const a1f &ForcastVals, int nbElements) const
{
    wxASSERT(ForcastVals.size() > 1);
    wxASSERT(nbElements > 0);
    wxASSERT(m_scoreClimatology != 0);

    // Check inputs
    if (!CheckObservedValue(ObservedVal)) {
        return NaNf;
    }
    if (!CheckVectorLength( ForcastVals, nbElements)) {
        wxLogWarning(_("Problems in a vector length."));
        return NaNf;
    }

    // First process the CRPS and then the skill score
    asScoreCRPSAR scoreCRPS = asScoreCRPSAR();
    scoreCRPS.SetThreshold(GetThreshold());
    scoreCRPS.SetQuantile(GetQuantile());
    float score = scoreCRPS.Assess(ObservedVal, ForcastVals, nbElements);
    float skillScore = (score - m_scoreClimatology) / ((float) 0 - m_scoreClimatology);

    return skillScore;
}

bool asScoreCRPSS::ProcessScoreClimatology(const a1f &refVals, const a1f &climatologyData)
{
    // Containers for final results
    a1f scoresClimatology(refVals.size());

    // Set the original score and process
    asScore *score = asScore::GetInstance(asScore::CRPSAR);
    score->SetThreshold(GetThreshold());
    score->SetQuantile(GetQuantile());

    for (int iReftime = 0; iReftime < refVals.size(); iReftime++) {
        if (!asTools::IsNaN(refVals(iReftime))) {
            scoresClimatology(iReftime) = score->Assess(refVals(iReftime), climatologyData, climatologyData.size());
        } else {
            scoresClimatology(iReftime) = NaNf;
        }
    }

    wxDELETE(score);

    m_scoreClimatology = asTools::Mean(&scoresClimatology[0], &scoresClimatology[scoresClimatology.size() - 1]);

    wxLogVerbose(_("Score of the climatology: %g."), m_scoreClimatology);

    return true;
}
