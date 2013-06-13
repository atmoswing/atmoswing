#ifndef ASFORECASTSCOREFINALPSS_H
#define ASFORECASTSCOREFINALPSS_H

#include <asIncludes.h>
#include <asForecastScoreFinal.h>

class asForecastScoreFinalPSS: public asForecastScoreFinal
{
public:
    asForecastScoreFinalPSS(Period period);

    asForecastScoreFinalPSS(const wxString& periodString);

    virtual ~asForecastScoreFinalPSS();

    float Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray);

protected:
private:

};

#endif // ASFORECASTSCOREFINALPSS_H
