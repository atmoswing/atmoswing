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
