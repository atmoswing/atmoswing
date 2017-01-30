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

#include "asForecastScoreFinalGSS.h"

asForecastScoreFinalGSS::asForecastScoreFinalGSS(Period period)
        : asForecastScoreFinal(period)
{

}

asForecastScoreFinalGSS::asForecastScoreFinalGSS(const wxString &periodString)
        : asForecastScoreFinal(periodString)
{

}

asForecastScoreFinalGSS::~asForecastScoreFinalGSS()
{
    //dtor
}

float asForecastScoreFinalGSS::Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray) const
{
    wxASSERT(targetDates.rows() > 1);
    wxASSERT(forecastScores.rows() > 1);

    int countA = 0, countB = 0, countC = 0, countD = 0, countTot = 0;

    switch (m_period) {
        case (asForecastScoreFinal::Total): {
            for (int i = 0; i < forecastScores.size(); i++) {
                countTot++;
                if (forecastScores[i] == 1) {
                    countA++;
                } else if (forecastScores[i] == 2) {
                    countB++;
                } else if (forecastScores[i] == 3) {
                    countC++;
                } else if (forecastScores[i] == 4) {
                    countD++;
                } else {
                    wxLogError(_("The GSS score (%f) is not an authorized value."), forecastScores[i]);
                    return NaNFloat;
                }
            }
            break;
        }

        default: {
            asThrowException(_("Period not yet implemented in asForecastScoreFinalGSS."));
        }
    }

    float score;

    if (countTot > 0) {
        float a = (float) countA;
        float b = (float) countB;
        float c = (float) countC;
        float d = (float) countD;
        float aref;
        if ((a + b + c + d) > 0) {
            aref = (a + b) * (a + c) / (a + b + c + d);
        } else {
            return 0;
        }
        if ((a - aref + b + c) > 0) {
            score = (a - aref) / (a - aref + b + c);
        } else {
            return 0;
        }
    } else {
        score = NaNFloat;
    }

    return score;
}
