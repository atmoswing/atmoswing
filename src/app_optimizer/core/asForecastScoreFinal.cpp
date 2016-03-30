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

#include "asForecastScoreFinal.h"

#include "asForecastScoreFinalMean.h"
#include "asForecastScoreFinalPC.h"
#include "asForecastScoreFinalTS.h"
#include "asForecastScoreFinalB.h"
#include "asForecastScoreFinalFAR.h"
#include "asForecastScoreFinalH.h"
#include "asForecastScoreFinalF.h"
#include "asForecastScoreFinalHSS.h"
#include "asForecastScoreFinalPSS.h"
#include "asForecastScoreFinalGSS.h"
#include "asForecastScoreFinalRMSE.h"
#include "asForecastScoreFinalRankHistogram.h"
#include "asForecastScoreFinalRankHistogramReliability.h"
#include "asForecastScoreFinalCRPSreliability.h"
#include "asForecastScoreFinalCRPSpotential.h"

asForecastScoreFinal::asForecastScoreFinal(Period period)
{
    m_period = period;
    m_singleValue = true;
    m_has2DArrayArgument = false;
    m_ranksNb = 0;
}

asForecastScoreFinal::asForecastScoreFinal(const wxString &periodString)
{
    m_singleValue = true;
    m_has2DArrayArgument = false;
    m_ranksNb = 0;

    if (periodString.CmpNoCase("Total") == 0) {
        m_period = asForecastScoreFinal::Total;
    } else if (periodString.CmpNoCase("SpecificPeriod") == 0) {
        m_period = asForecastScoreFinal::SpecificPeriod;
    } else if (periodString.CmpNoCase("Summer") == 0) {
        m_period = asForecastScoreFinal::Summer;
    } else if (periodString.CmpNoCase("Automn") == 0) {
        m_period = asForecastScoreFinal::Automn;
    } else if (periodString.CmpNoCase("Winter") == 0) {
        m_period = asForecastScoreFinal::Winter;
    } else if (periodString.CmpNoCase("Spring") == 0) {
        m_period = asForecastScoreFinal::Spring;
    } else {
        asLogError(_("The final forecast score period was not correctly set."));
        m_period = asForecastScoreFinal::Total;
    }
}

asForecastScoreFinal *asForecastScoreFinal::GetInstance(const wxString &scoreString, const wxString &periodString)
{
    if (scoreString.CmpNoCase("CRPSSkillScore") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPSS") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPS") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPSAR") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPSEP") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPSaccuracy") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPSaccuracyAR") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPSaccuracyEP") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPSsharpness") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPSsharpnessAR") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPSsharpnessEP") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPSreliability") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalCRPSreliability(periodString);
        return score;
    } else if (scoreString.CmpNoCase("CRPSpotential") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalCRPSpotential(periodString);
        return score;
    } else if (scoreString.CmpNoCase("DF0") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("PC") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalPC(periodString);
        return score;
    } else if (scoreString.CmpNoCase("TS") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalTS(periodString);
        return score;
    } else if (scoreString.CmpNoCase("BIAS") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalB(periodString);
        return score;
    } else if (scoreString.CmpNoCase("FARA") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalFAR(periodString);
        return score;
    } else if (scoreString.CmpNoCase("H") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalH(periodString);
        return score;
    } else if (scoreString.CmpNoCase("F") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalF(periodString);
        return score;
    } else if (scoreString.CmpNoCase("HSS") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalHSS(periodString);
        return score;
    } else if (scoreString.CmpNoCase("PSS") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalPSS(periodString);
        return score;
    } else if (scoreString.CmpNoCase("GSS") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalGSS(periodString);
        return score;
    } else if (scoreString.CmpNoCase("MAE") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("RMSE") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalRMSE(periodString);
        return score;
    } else if (scoreString.CmpNoCase("BS") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("BSS") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("SEEPS") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalMean(periodString);
        return score;
    } else if (scoreString.CmpNoCase("RankHistogram") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalRankHistogram(periodString);
        return score;
    } else if (scoreString.CmpNoCase("RankHistogramReliability") == 0) {
        asForecastScoreFinal *score = new asForecastScoreFinalRankHistogramReliability(periodString);
        return score;
    } else {
        asLogError(_("The final forecast score was not correctly set."));
        asForecastScoreFinal *score = new asForecastScoreFinalMean(periodString);
        return score;
    }
}

asForecastScoreFinal::~asForecastScoreFinal()
{
    //dtor
}

Array1DFloat asForecastScoreFinal::AssessOnArray(Array1DFloat &targetDates, Array1DFloat &forecastScores,
                                                 asTimeArray &timeArray)
{
    asLogError(_("This asForecastScoreFinal class has no AssessOnArray method implemented !"));
    return Array1DFloat();
}

float asForecastScoreFinal::Assess(Array1DFloat &targetDates, Array2DFloat &forecastScores, asTimeArray &timeArray)
{
    asLogError(_("This asForecastScoreFinal class has no Assess method implemented with a 2D array as argument !"));
    return NaNFloat;
}
