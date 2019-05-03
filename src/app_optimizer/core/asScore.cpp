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

#include "asScore.h"
#include "asScoreBS.h"
#include "asScoreBSS.h"
#include "asScoreCRPSAR.h"
#include "asScoreCRPSEP.h"
#include "asScoreCRPSHersbachDecomp.h"
#include "asScoreCRPSS.h"
#include "asScoreCRPSaccurAR.h"
#include "asScoreCRPSaccurEP.h"
#include "asScoreCRPSsharpAR.h"
#include "asScoreCRPSsharpEP.h"
#include "asScoreContingencyTable.h"
#include "asScoreDF0.h"
#include "asScoreMAE.h"
#include "asScoreMSE.h"
#include "asScoreRankHistogram.h"
#include "asScoreSEEPS.h"

asScore::asScore()
        : m_score(Undefined),
          m_order(Asc),
          m_scaleBest(NaNf),
          m_scaleWorst(NaNf),
          m_usesClimatology(false),
          m_singleValue(true),
          m_onMean(false),
          m_scoreClimatology(0),
          m_threshold(NaNf),
          m_quantile(NaNf)
{

}

asScore::asScore(Score score, const wxString &name, const wxString &fullname, Order order, float scaleBest,
                 float scaleWorst, bool usesClimatology, bool singleValue)
        : m_score(score),
          m_name(name),
          m_fullName(fullname),
          m_order(order),
          m_scaleBest(scaleBest),
          m_scaleWorst(scaleWorst),
          m_usesClimatology(usesClimatology),
          m_singleValue(singleValue),
          m_onMean(false),
          m_scoreClimatology(0),
          m_threshold(NaNf),
          m_quantile(NaNf)
{

}

asScore *asScore::GetInstance(Score scoreEnum)
{
    switch (scoreEnum) {
        case (CRPSS): {
            asScore *score = new asScoreCRPSS();
            return score;
        }
        case (CRPSAR): {
            asScore *score = new asScoreCRPSAR();
            return score;
        }
        case (CRPSEP): {
            asScore *score = new asScoreCRPSEP();
            return score;
        }
        case (CRPSaccuracyAR): {
            asScore *score = new asScoreCRPSaccurAR();
            return score;
        }
        case (CRPSaccuracyEP): {
            asScore *score = new asScoreCRPSaccurEP();
            return score;
        }
        case (CRPSsharpnessAR): {
            asScore *score = new asScoreCRPSsharpAR();
            return score;
        }
        case (CRPSsharpnessEP): {
            asScore *score = new asScoreCRPSsharpEP();
            return score;
        }
        case (CRPSreliability): {
            asScore *score = new asScoreCRPSHersbachDecomp();
            return score;
        }
        case (CRPSpotential): {
            asScore *score = new asScoreCRPSHersbachDecomp();
            return score;
        }
        case (CRPSHersbachDecomp): {
            asScore *score = new asScoreCRPSHersbachDecomp();
            return score;
        }
        case (DF0): {
            asScore *score = new asScoreDF0();
            return score;
        }
        case (PC): {
            asScore *score = new asScoreContingencyTable();
            return score;
        }
        case (TS): {
            asScore *score = new asScoreContingencyTable();
            return score;
        }
        case (BIAS): {
            asScore *score = new asScoreContingencyTable();
            return score;
        }
        case (FARA): {
            asScore *score = new asScoreContingencyTable();
            return score;
        }
        case (H): {
            asScore *score = new asScoreContingencyTable();
            return score;
        }
        case (F): {
            asScore *score = new asScoreContingencyTable();
            return score;
        }
        case (HSS): {
            asScore *score = new asScoreContingencyTable();
            return score;
        }
        case (PSS): {
            asScore *score = new asScoreContingencyTable();
            return score;
        }
        case (GSS): {
            asScore *score = new asScoreContingencyTable();
            return score;
        }
        case (ContingencyTable): {
            asScore *score = new asScoreContingencyTable();
            return score;
        }
        case (MAE): {
            asScore *score = new asScoreMAE();
            return score;
        }
        case (MSE): {
            asScore *score = new asScoreMSE();
            return score;
        }
        case (RMSE): {
            asScore *score = new asScoreMSE();
            return score;
        }
        case (BS): {
            asScore *score = new asScoreBS();
            return score;
        }
        case (BSS): {
            asScore *score = new asScoreBSS();
            return score;
        }
        case (SEEPS): {
            asScore *score = new asScoreSEEPS();
            return score;
        }
        case (RankHistogram): {
            asScore *score = new asScoreRankHistogram();
            return score;
        }
        case (RankHistogramReliability): {
            asScore *score = new asScoreRankHistogram();
            return score;
        }
        default: {
            wxLogError(_("The score was not correctly set (undefined)."));
            return nullptr;
        }
    }
}

asScore *asScore::GetInstance(const wxString &scoreString)
{
    if (scoreString.CmpNoCase("CRPSSkillScore") == 0) {
        asScore *score = new asScoreCRPSS();
        return score;
    } else if (scoreString.CmpNoCase("CRPSS") == 0) {
        asScore *score = new asScoreCRPSS();
        return score;
    } else if (scoreString.CmpNoCase("CRPS") == 0) {
        asScore *score = new asScoreCRPSAR();
        return score;
    } else if (scoreString.CmpNoCase("CRPSAR") == 0) {
        asScore *score = new asScoreCRPSAR();
        return score;
    } else if (scoreString.CmpNoCase("CRPSEP") == 0) {
        asScore *score = new asScoreCRPSEP();
        return score;
    } else if (scoreString.CmpNoCase("CRPSaccuracy") == 0) {
        asScore *score = new asScoreCRPSaccurAR();
        return score;
    } else if (scoreString.CmpNoCase("CRPSaccuracyAR") == 0) {
        asScore *score = new asScoreCRPSaccurAR();
        return score;
    } else if (scoreString.CmpNoCase("CRPSaccuracyEP") == 0) {
        asScore *score = new asScoreCRPSaccurEP();
        return score;
    } else if (scoreString.CmpNoCase("CRPSsharpness") == 0) {
        asScore *score = new asScoreCRPSsharpAR();
        return score;
    } else if (scoreString.CmpNoCase("CRPSsharpnessAR") == 0) {
        asScore *score = new asScoreCRPSsharpAR();
        return score;
    } else if (scoreString.CmpNoCase("CRPSsharpnessEP") == 0) {
        asScore *score = new asScoreCRPSsharpEP();
        return score;
    } else if (scoreString.CmpNoCase("CRPSreliability") == 0) {
        asScore *score = new asScoreCRPSHersbachDecomp();
        return score;
    } else if (scoreString.CmpNoCase("CRPSpotential") == 0) {
        asScore *score = new asScoreCRPSHersbachDecomp();
        return score;
    } else if (scoreString.CmpNoCase("DF0") == 0) {
        asScore *score = new asScoreDF0();
        return score;
    } else if (scoreString.CmpNoCase("PC") == 0) {
        asScore *score = new asScoreContingencyTable();
        return score;
    } else if (scoreString.CmpNoCase("TS") == 0) {
        asScore *score = new asScoreContingencyTable();
        return score;
    } else if (scoreString.CmpNoCase("BIAS") == 0) {
        asScore *score = new asScoreContingencyTable();
        return score;
    } else if (scoreString.CmpNoCase("FARA") == 0) {
        asScore *score = new asScoreContingencyTable();
        return score;
    } else if (scoreString.CmpNoCase("H") == 0) {
        asScore *score = new asScoreContingencyTable();
        return score;
    } else if (scoreString.CmpNoCase("F") == 0) {
        asScore *score = new asScoreContingencyTable();
        return score;
    } else if (scoreString.CmpNoCase("HSS") == 0) {
        asScore *score = new asScoreContingencyTable();
        return score;
    } else if (scoreString.CmpNoCase("PSS") == 0) {
        asScore *score = new asScoreContingencyTable();
        return score;
    } else if (scoreString.CmpNoCase("GSS") == 0) {
        asScore *score = new asScoreContingencyTable();
        return score;
    } else if (scoreString.CmpNoCase("MAE") == 0) {
        asScore *score = new asScoreMAE();
        return score;
    } else if (scoreString.CmpNoCase("MSE") == 0) {
        asScore *score = new asScoreMSE();
        return score;
    } else if (scoreString.CmpNoCase("RMSE") == 0) {
        asScore *score = new asScoreMSE();
        return score;
    } else if (scoreString.CmpNoCase("BS") == 0) {
        asScore *score = new asScoreBS();
        return score;
    } else if (scoreString.CmpNoCase("BSS") == 0) {
        asScore *score = new asScoreBSS();
        return score;
    } else if (scoreString.CmpNoCase("SEEPS") == 0) {
        asScore *score = new asScoreSEEPS();
        return score;
    } else if (scoreString.CmpNoCase("RankHistogram") == 0) {
        asScore *score = new asScoreRankHistogram();
        return score;
    } else if (scoreString.CmpNoCase("RankHistogramReliability") == 0) {
        asScore *score = new asScoreRankHistogram();
        return score;
    } else {
        wxLogError(_("The score was not correctly set (cannot use %s)."), scoreString);
        asScore *score = new asScoreCRPSAR();
        return score;
    }
}

a1f asScore::AssessOnArray(float obs, const a1f &values, int nbElements) const
{
    wxLogError(_("This asScore class has no AssessOnArrays method implemented !"));

    return a1f();
}

bool asScore::CheckObservedValue(float obs) const
{
    // Check that the observed value is not a NaN
    if (asIsNaN(obs)) {
        wxLogVerbose(_("The observed value is a NaN for the score calculation."));
        return false;
    }

    return true;
}

bool asScore::CheckVectorLength(const a1f &values, int nbElements) const
{
    // Check the element numbers vs vector length
    wxASSERT_MSG(values.rows() >= nbElements,
                 _("The required elements number is above the vector length in the score calculation."));
    wxASSERT(nbElements > 1);
    if (values.rows() < nbElements) {
        wxLogError(_("The required elements number is above the vector length in the score calculation."));
        return false;
    }

    return true;
}

int asScore::CleanNans(const a1f &valuesIn, a1f &valuesOut, int nbElements) const
{
    // Remove the NaNs and copy content
    int nbPredict = 0, nbNans = 0, iVal = 0;
    while (iVal < nbElements) {
        if (!asIsNaN(valuesIn[iVal])) {
            valuesOut(nbPredict) = valuesIn[iVal];
            nbPredict++;
        } else {
            nbNans++;
        }
        iVal++;
    }

    if (nbPredict < 1) {
        wxLogError(_("Not enough data to perform the score calculation."));
        return asNOT_FOUND;
    }

    return nbPredict;
}
