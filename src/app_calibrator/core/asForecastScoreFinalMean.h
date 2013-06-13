#ifndef ASFORECASTSCOREFINALMEAN_H
#define ASFORECASTSCOREFINALMEAN_H

#include <asIncludes.h>
#include <asForecastScoreFinal.h>

class asForecastScoreFinalMean: public asForecastScoreFinal
{
public:
    asForecastScoreFinalMean(Period period);

    asForecastScoreFinalMean(const wxString& periodString);

    virtual ~asForecastScoreFinalMean();

    float Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray);

protected:
private:

};

#endif // ASFORECASTSCOREFINALMEAN_H
