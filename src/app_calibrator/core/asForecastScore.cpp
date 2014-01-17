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
 
#include "asForecastScore.h"
#include "asForecastScoreCRPSS.h"
#include "asForecastScoreCRPSAR.h"
#include "asForecastScoreCRPSEP.h"
#include "asForecastScoreCRPSaccuracyAR.h"
#include "asForecastScoreCRPSaccuracyEP.h"
#include "asForecastScoreCRPSsharpnessAR.h"
#include "asForecastScoreCRPSsharpnessEP.h"
#include "asForecastScoreDF0.h"
#include "asForecastScoreContingencyTable.h"
#include "asForecastScoreMAE.h"
#include "asForecastScoreRMSE.h"
#include "asForecastScoreBS.h"
#include "asForecastScoreBSS.h"
#include "asForecastScoreSEEPS.h"

asForecastScore::asForecastScore()
{
    m_ScoreClimatology = 0;
    m_Threshold = NaNFloat;
    m_Percentile = NaNFloat;
    m_UsesClimatology = false;
}

asForecastScore* asForecastScore::GetInstance(Score scoreEnum)
{
    switch (scoreEnum)
    {
        case (CRPSS):
        {
            asForecastScore* score = new asForecastScoreCRPSS();
            return score;
        }
        case (CRPSAR):
        {
            asForecastScore* score = new asForecastScoreCRPSAR();
            return score;
        }
        case (CRPSEP):
        {
            asForecastScore* score = new asForecastScoreCRPSEP();
            return score;
        }
        case (CRPSaccuracyAR):
        {
            asForecastScore* score = new asForecastScoreCRPSaccuracyAR();
            return score;
        }
        case (CRPSaccuracyEP):
        {
            asForecastScore* score = new asForecastScoreCRPSaccuracyEP();
            return score;
        }
        case (CRPSsharpnessAR):
        {
            asForecastScore* score = new asForecastScoreCRPSsharpnessAR();
            return score;
        }
        case (CRPSsharpnessEP):
        {
            asForecastScore* score = new asForecastScoreCRPSsharpnessEP();
            return score;
        }
        case (DF0):
        {
            asForecastScore* score = new asForecastScoreDF0();
            return score;
        }
        case (PC):
        {
            asForecastScore* score = new asForecastScoreContingencyTable();
            return score;
        }
        case (TS):
        {
            asForecastScore* score = new asForecastScoreContingencyTable();
            return score;
        }
        case (BIAS):
        {
            asForecastScore* score = new asForecastScoreContingencyTable();
            return score;
        }
        case (FARA):
        {
            asForecastScore* score = new asForecastScoreContingencyTable();
            return score;
        }
        case (H):
        {
            asForecastScore* score = new asForecastScoreContingencyTable();
            return score;
        }
        case (F):
        {
            asForecastScore* score = new asForecastScoreContingencyTable();
            return score;
        }
        case (HSS):
        {
            asForecastScore* score = new asForecastScoreContingencyTable();
            return score;
        }
        case (PSS):
        {
            asForecastScore* score = new asForecastScoreContingencyTable();
            return score;
        }
        case (GSS):
        {
            asForecastScore* score = new asForecastScoreContingencyTable();
            return score;
        }
        case (ContingencyTable):
        {
            asForecastScore* score = new asForecastScoreContingencyTable();
            return score;
        }
        case (MAE):
        {
            asForecastScore* score = new asForecastScoreMAE();
            return score;
        }
        case (RMSE):
        {
            asForecastScore* score = new asForecastScoreRMSE();
            return score;
        }
        case (BS):
        {
            asForecastScore* score = new asForecastScoreBS();
            return score;
        }
        case (BSS):
        {
            asForecastScore* score = new asForecastScoreBSS();
            return score;
        }
        case (SEEPS):
        {
            asForecastScore* score = new asForecastScoreSEEPS();
            return score;
        }
    }

    return NULL;
}

asForecastScore* asForecastScore::GetInstance(const wxString& scoreString)
{
    if (scoreString.CmpNoCase("CRPSSkillScore")==0)
    {
        asForecastScore* score = new asForecastScoreCRPSS();
        return score;
    }
    else if (scoreString.CmpNoCase("CRPSS")==0)
    {
        asForecastScore* score = new asForecastScoreCRPSS();
        return score;
    }
    else if (scoreString.CmpNoCase("CRPS")==0)
    {
        asForecastScore* score = new asForecastScoreCRPSAR();
        return score;
    }
    else if (scoreString.CmpNoCase("CRPSAR")==0)
    {
        asForecastScore* score = new asForecastScoreCRPSAR();
        return score;
    }
    else if (scoreString.CmpNoCase("CRPSEP")==0)
    {
        asForecastScore* score = new asForecastScoreCRPSEP();
        return score;
    }
    else if (scoreString.CmpNoCase("CRPSaccuracy")==0)
    {
        asForecastScore* score = new asForecastScoreCRPSaccuracyAR();
        return score;
    }
    else if (scoreString.CmpNoCase("CRPSaccuracyAR")==0)
    {
        asForecastScore* score = new asForecastScoreCRPSaccuracyAR();
        return score;
    }
    else if (scoreString.CmpNoCase("CRPSaccuracyEP")==0)
    {
        asForecastScore* score = new asForecastScoreCRPSaccuracyEP();
        return score;
    }
    else if (scoreString.CmpNoCase("CRPSsharpness")==0)
    {
        asForecastScore* score = new asForecastScoreCRPSsharpnessAR();
        return score;
    }
    else if (scoreString.CmpNoCase("CRPSsharpnessAR")==0)
    {
        asForecastScore* score = new asForecastScoreCRPSsharpnessAR();
        return score;
    }
    else if (scoreString.CmpNoCase("CRPSsharpnessEP")==0)
    {
        asForecastScore* score = new asForecastScoreCRPSsharpnessEP();
        return score;
    }
    else if (scoreString.CmpNoCase("DF0")==0)
    {
        asForecastScore* score = new asForecastScoreDF0();
        return score;
    }
    else if (scoreString.CmpNoCase("PC")==0)
    {
        asForecastScore* score = new asForecastScoreContingencyTable();
        return score;
    }
    else if (scoreString.CmpNoCase("TS")==0)
    {
        asForecastScore* score = new asForecastScoreContingencyTable();
        return score;
    }
    else if (scoreString.CmpNoCase("BIAS")==0)
    {
        asForecastScore* score = new asForecastScoreContingencyTable();
        return score;
    }
    else if (scoreString.CmpNoCase("FARA")==0)
    {
        asForecastScore* score = new asForecastScoreContingencyTable();
        return score;
    }
    else if (scoreString.CmpNoCase("H")==0)
    {
        asForecastScore* score = new asForecastScoreContingencyTable();
        return score;
    }
    else if (scoreString.CmpNoCase("F")==0)
    {
        asForecastScore* score = new asForecastScoreContingencyTable();
        return score;
    }
    else if (scoreString.CmpNoCase("HSS")==0)
    {
        asForecastScore* score = new asForecastScoreContingencyTable();
        return score;
    }
    else if (scoreString.CmpNoCase("PSS")==0)
    {
        asForecastScore* score = new asForecastScoreContingencyTable();
        return score;
    }
    else if (scoreString.CmpNoCase("GSS")==0)
    {
        asForecastScore* score = new asForecastScoreContingencyTable();
        return score;
    }
    else if (scoreString.CmpNoCase("MAE")==0)
    {
        asForecastScore* score = new asForecastScoreMAE();
        return score;
    }
    else if (scoreString.CmpNoCase("RMSE")==0)
    {
        asForecastScore* score = new asForecastScoreRMSE();
        return score;
    }
    else if (scoreString.CmpNoCase("BS")==0)
    {
        asForecastScore* score = new asForecastScoreBS();
        return score;
    }
    else if (scoreString.CmpNoCase("BSS")==0)
    {
        asForecastScore* score = new asForecastScoreBSS();
        return score;
    }
    else if (scoreString.CmpNoCase("SEEPS")==0)
    {
        asForecastScore* score = new asForecastScoreSEEPS();
        return score;
    }
    else
    {
		asLogError(wxString::Format(_("The forecast score was not correctly set (cannot use %s)."), scoreString.c_str()));
        asForecastScore* score = new asForecastScoreCRPSAR();
        return score;
    }
}

asForecastScore::~asForecastScore()
{
    //dtor
}

bool asForecastScore::CheckInputs(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements)
{
    // Check the element numbers vs vector length
    wxASSERT_MSG(ForcastVals.rows()>=nbElements, _("The required elements number is above the vector length in the score calculation."));
    wxASSERT(nbElements>1);
    if (ForcastVals.rows()<nbElements)
    {
        asLogError(_("The required elements number is above the vector length in the score calculation."));
        return false;
    }

    // Check that the observed value is not a NaN
    if (asTools::IsNaN(ObservedVal))
    {
        asLogWarning(_("The observed value is a NaN for the CRPS score calculation."));
        return false;
    }

    return true;
}

int asForecastScore::CleanNans(const Array1DFloat &ForcastVals, Array1DFloat &ForcastValsSorted, int nbElements)
{
    // Remove the NaNs and copy content
    int nbForecasts = 0, nbNans = 0;
    int i_val = 0;
    while (nbForecasts<nbElements)
    {
        // Add a check to not overflow the array
        if(i_val>=nbElements)
        {
            if(i_val==ForcastVals.rows())
            {
                asLogWarning(_("Tried to access an element outside of the vector in the score calculation."));
                asLogWarning(wxString::Format(_("Desired analogs nb (%d), Usable elements nb (%d), NaNs (%d) ."), nbElements, nbForecasts, nbNans));
                break;
            }
        }

        if (!asTools::IsNaN(ForcastVals[i_val]))
        {
            ForcastValsSorted(nbForecasts) = ForcastVals[i_val];
            nbForecasts++;
        }
        else
        {
            nbNans++;
        }
        i_val++;
    }

    if (nbForecasts<1)
    {
        asLogError(_("Not enough data to perform the score calculation."));
        return asNOT_FOUND;
    }

    return nbForecasts;
}
