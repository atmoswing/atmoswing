#ifndef ASFORECASTSCOREFINALHSS_H
#define ASFORECASTSCOREFINALHSS_H

#include <asIncludes.h>
#include <asForecastScoreFinal.h>

class asForecastScoreFinalHSS: public asForecastScoreFinal
{
public:
    asForecastScoreFinalHSS(Period period);

    asForecastScoreFinalHSS(const wxString& periodString);

    virtual ~asForecastScoreFinalHSS();

    float Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray);

protected:
private:

};

#endif // ASFORECASTSCOREFINALHSS_H
