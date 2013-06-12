#ifndef ASFORECASTSCOREFINALPC_H
#define ASFORECASTSCOREFINALPC_H

#include <asIncludes.h>
#include <asForecastScoreFinal.h>

class asForecastScoreFinalPC: public asForecastScoreFinal
{
public:
    asForecastScoreFinalPC(Period period);

    asForecastScoreFinalPC(const wxString& periodString);

    virtual ~asForecastScoreFinalPC();

    float Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray);

protected:
private:

};

#endif // ASFORECASTSCOREFINALPC_H
