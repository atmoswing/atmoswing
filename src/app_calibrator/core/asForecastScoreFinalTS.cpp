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
 
#include "asForecastScoreFinalTS.h"

asForecastScoreFinalTS::asForecastScoreFinalTS(Period period)
:
asForecastScoreFinal(period)
{

}

asForecastScoreFinalTS::asForecastScoreFinalTS(const wxString& periodString)
:
asForecastScoreFinal(periodString)
{

}

asForecastScoreFinalTS::~asForecastScoreFinalTS()
{
    //dtor
}

float asForecastScoreFinalTS::Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray)
{
    wxASSERT(targetDates.rows()>1);
    wxASSERT(forecastScores.rows()>1);

    int countA=0, countB=0, countC=0, countTot=0;

    switch (m_Period)
    {
        case (asForecastScoreFinal::Total):
        {
            for (int i=0; i<forecastScores.size(); i++)
            {
                countTot++;
                if (forecastScores[i]==1)
                {
                    countA++;
                }
                else if (forecastScores[i]==2)
                {
                    countB++;
                }
                else if (forecastScores[i]==3)
                {
                    countC++;
                }
                else if (forecastScores[i]==4)
                {
                    //
                }
                else
                {
                    asLogError(wxString::Format(_("The TS score (%f) is not an authorized value."), forecastScores[i]));
                    return NaNFloat;
                }
            }
            break;
        }

        default:
        {
            asThrowException(_("Period not yet implemented in asForecastScoreFinalTS."));
        }
    }

    float score;

    if (countTot>0)
    {
        if ((countA+countB+countC)>0)
        {
            score = static_cast<float>(countA)/static_cast<float>(countA+countB+countC);
        }
        else
        {
            score = 0;
        }
    }
    else
    {
        score = NaNFloat;
    }

    return score;
}
