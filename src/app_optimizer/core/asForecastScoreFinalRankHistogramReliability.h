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
 * Portions Copyright 2014-2015 Pascal Horton, Terranum.
 */

#ifndef ASFORECASTSCOREFINALRANKHISTOGRAMRELIABILITY_H
#define ASFORECASTSCOREFINALRANKHISTOGRAMRELIABILITY_H

#include <asIncludes.h>
#include <asForecastScoreFinal.h>

class asForecastScoreFinalRankHistogramReliability
        : public asForecastScoreFinal
{
public:
    asForecastScoreFinalRankHistogramReliability(Period period);

    asForecastScoreFinalRankHistogramReliability(const wxString &periodString);

    virtual ~asForecastScoreFinalRankHistogramReliability();

    float Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray) const;

    float AssessOnBootstrap(Array1DFloat &histogramPercent, int forecastScoresSize) const;

protected:

private:

};

#endif // ASFORECASTSCOREFINALRANKHISTOGRAMRELIABILITY_H
