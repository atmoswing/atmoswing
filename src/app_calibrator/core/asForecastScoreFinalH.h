#ifndef ASFORECASTSCOREFINALH_H
#define ASFORECASTSCOREFINALH_H

#include <asIncludes.h>
#include <asForecastScoreFinal.h>

class asForecastScoreFinalH: public asForecastScoreFinal
{
public:
    asForecastScoreFinalH(Period period);

    asForecastScoreFinalH(const wxString& periodString);

    virtual ~asForecastScoreFinalH();

    float Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray);

protected:
private:

};

#endif // ASFORECASTSCOREFINALH_H
