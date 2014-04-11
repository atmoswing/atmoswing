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
 
#ifndef ASPROCESSORFORECASTSCORE_H
#define ASPROCESSORFORECASTSCORE_H

#include <asIncludes.h>

class asTimeArray;
class asResultsAnalogsValues;
class asForecastScore;
class asParametersScoring;
class asResultsAnalogsForecastScores;
class asResultsAnalogsForecastScoreFinal;


class asProcessorForecastScore: public wxObject
{
public:

    /** Analogs forecast score calculation for every day
    * \param AnalogsValues The ResAnalogsValues structure
    * \param ForecastScore The score object to assess the forecast
    * \param AnalogsNb The number of analogs to consider
    * \return The ResAnalogsForecastScore structure
    */
    static bool GetAnalogsForecastScores(asResultsAnalogsValues &anaValues, asForecastScore *forecastScore, asParametersScoring &params, asResultsAnalogsForecastScores &results);

    static bool GetAnalogsForecastScoresLoadF0(asResultsAnalogsValues &anaValues, asForecastScore *forecastScore, asParametersScoring &params, asResultsAnalogsForecastScores &results);

    /** Analogs final score
    * \param AnalogsScores The ResAnalogsForecastScores structure
    * \param TimeArray The dates that should be considered to sum the score
    * \return The ResAnalogsFinalScore structure
    */
    static bool GetAnalogsForecastScoreFinal(asResultsAnalogsForecastScores &anaScores, asTimeArray &timeArray, asParametersScoring &params, asResultsAnalogsForecastScoreFinal &results);

    /** Analogs final score. Returns the final score value
    * \param AnalogsScores The ResAnalogsForecastScores structure
    * \param TimeArray The dates that should be considered to sum the score
    * \return The final score value
    */
    //static float GetAnalogsForecastScoreFinal(asResultsAnalogsForecastScores &anaScores, asTimeArray &timeArray, asParametersScoring &params);
protected:
private:
};

#endif
