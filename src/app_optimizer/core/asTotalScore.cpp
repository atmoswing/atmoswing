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

#include "asTotalScore.h"

#include "asTotalScoreB.h"
#include "asTotalScoreCRPSpotential.h"
#include "asTotalScoreCRPSreliability.h"
#include "asTotalScoreF.h"
#include "asTotalScoreFAR.h"
#include "asTotalScoreGSS.h"
#include "asTotalScoreH.h"
#include "asTotalScoreHSS.h"
#include "asTotalScoreMSE.h"
#include "asTotalScoreMean.h"
#include "asTotalScoreMeanWithClim.h"
#include "asTotalScorePC.h"
#include "asTotalScorePSS.h"
#include "asTotalScoreRMSE.h"
#include "asTotalScoreRankHistogram.h"
#include "asTotalScoreRankHistogramReliability.h"
#include "asTotalScoreTS.h"

asTotalScore::asTotalScore(const wxString& periodString)
    : m_singleValue(true),
      m_has2DArrayArgument(false),
      m_ranksNb(0) {
    if (periodString.CmpNoCase("Total") == 0) {
        m_period = asTotalScore::Total;
    } else if (periodString.CmpNoCase("SpecificPeriod") == 0) {
        m_period = asTotalScore::SpecificPeriod;
    } else if (periodString.CmpNoCase("Summer") == 0) {
        m_period = asTotalScore::Summer;
    } else if (periodString.CmpNoCase("Fall") == 0) {
        m_period = asTotalScore::Fall;
    } else if (periodString.CmpNoCase("Winter") == 0) {
        m_period = asTotalScore::Winter;
    } else if (periodString.CmpNoCase("Spring") == 0) {
        m_period = asTotalScore::Spring;
    } else {
        wxLogError(_("The total score period was not correctly set."));
        m_period = asTotalScore::Total;
    }
}

asTotalScore* asTotalScore::GetInstance(const wxString& scoreString, const wxString& periodString) {
    if (scoreString.CmpNoCase("CRPSSkillScore") == 0) {
        asTotalScore* score = new asTotalScoreMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPSS") == 0) {
        asTotalScore* score = new asTotalScoreMeanWithClim(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPS") == 0) {
        asTotalScore* score = new asTotalScoreMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPSAR") == 0) {
        asTotalScore* score = new asTotalScoreMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPSEP") == 0) {
        asTotalScore* score = new asTotalScoreMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPSaccuracy") == 0) {
        asTotalScore* score = new asTotalScoreMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPSaccuracyAR") == 0) {
        asTotalScore* score = new asTotalScoreMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPSaccuracyEP") == 0) {
        asTotalScore* score = new asTotalScoreMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPSsharpness") == 0) {
        asTotalScore* score = new asTotalScoreMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPSsharpnessAR") == 0) {
        asTotalScore* score = new asTotalScoreMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPSsharpnessEP") == 0) {
        asTotalScore* score = new asTotalScoreMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPSreliability") == 0) {
        asTotalScore* score = new asTotalScoreCRPSreliability(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPSpotential") == 0) {
        asTotalScore* score = new asTotalScoreCRPSpotential(periodString);
        return score;
    } else if (scoreString.CmpNoCase("DF0") == 0) {
        asTotalScore* score = new asTotalScoreMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("PC") == 0) {
        asTotalScore* score = new asTotalScorePC(periodString);
        return score;
    } else if (scoreString.CmpNoCase("TS") == 0) {
        asTotalScore* score = new asTotalScoreTS(periodString);
        return score;
    } else if (scoreString.CmpNoCase("BIAS") == 0) {
        asTotalScore* score = new asTotalScoreB(periodString);
        return score;
    } else if (scoreString.CmpNoCase("FARA") == 0) {
        asTotalScore* score = new asTotalScoreFAR(periodString);
        return score;
    } else if (scoreString.CmpNoCase("H") == 0) {
        asTotalScore* score = new asTotalScoreH(periodString);
        return score;
    } else if (scoreString.CmpNoCase("F") == 0) {
        asTotalScore* score = new asTotalScoreF(periodString);
        return score;
    } else if (scoreString.CmpNoCase("HSS") == 0) {
        asTotalScore* score = new asTotalScoreHSS(periodString);
        return score;
    } else if (scoreString.CmpNoCase("PSS") == 0) {
        asTotalScore* score = new asTotalScorePSS(periodString);
        return score;
    } else if (scoreString.CmpNoCase("GSS") == 0) {
        asTotalScore* score = new asTotalScoreGSS(periodString);
        return score;
    } else if (scoreString.CmpNoCase("MAE") == 0) {
        asTotalScore* score = new asTotalScoreMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("MSE") == 0) {
        asTotalScore* score = new asTotalScoreMSE(periodString);
        return score;
    } else if (scoreString.CmpNoCase("RMSE") == 0) {
        asTotalScore* score = new asTotalScoreRMSE(periodString);
        return score;
    } else if (scoreString.CmpNoCase("BS") == 0) {
        asTotalScore* score = new asTotalScoreMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("BSS") == 0) {
        asTotalScore* score = new asTotalScoreMeanWithClim(periodString);
        return score;
    } else if (scoreString.CmpNoCase("SEEPS") == 0) {
        asTotalScore* score = new asTotalScoreMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("RankHistogram") == 0) {
        asTotalScore* score = new asTotalScoreRankHistogram(periodString);
        return score;
    } else if (scoreString.CmpNoCase("RankHistogramReliability") == 0) {
        asTotalScore* score = new asTotalScoreRankHistogramReliability(periodString);
        return score;
    } else {
        wxLogError(_("The total score was not correctly set."));
        asTotalScore* score = new asTotalScoreMean(periodString);
        return score;
    }
}

asTotalScore::~asTotalScore() {}

a1f asTotalScore::AssessOnArray(const a1f& targetDates, const a1f& scores, const asTimeArray& timeArray) const {
    wxLogError(_("This asTotalScore class has no AssessOnArray method implemented !"));
    return a1f();
}

float asTotalScore::Assess(const a1f& targetDates, const a2f& scores, const asTimeArray& timeArray) const {
    wxLogError(_("This asTotalScore class has no Assess method implemented with a 2D array as argument !"));
    return NAN;
}
