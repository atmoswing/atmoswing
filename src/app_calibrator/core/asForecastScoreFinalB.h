#ifndef ASFORECASTSCOREFINALB_H
#define ASFORECASTSCOREFINALB_H

#include <asIncludes.h>
#include <asForecastScoreFinal.h>

class asForecastScoreFinalB: public asForecastScoreFinal
{
public:
    asForecastScoreFinalB(Period period);

    asForecastScoreFinalB(const wxString& periodString);

    virtual ~asForecastScoreFinalB();

    float Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray);

protected:
private:

};

#endif // ASFORECASTSCOREFINALB_H
