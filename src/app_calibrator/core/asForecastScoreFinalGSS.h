#ifndef ASFORECASTSCOREFINALGSS_H
#define ASFORECASTSCOREFINALGSS_H

#include <asIncludes.h>
#include <asForecastScoreFinal.h>

class asForecastScoreFinalGSS: public asForecastScoreFinal
{
public:
    asForecastScoreFinalGSS(Period period);

    asForecastScoreFinalGSS(const wxString& periodString);

    virtual ~asForecastScoreFinalGSS();

    float Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray);

protected:
private:

};

#endif // ASFORECASTSCOREFINALGSS_H
