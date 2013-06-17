#ifndef ASFORECASTSCOREFINALRMSE_H
#define ASFORECASTSCOREFINALRMSE_H

#include <asIncludes.h>
#include <asForecastScoreFinal.h>

class asForecastScoreFinalRMSE: public asForecastScoreFinal
{
public:
    asForecastScoreFinalRMSE(Period period);

    asForecastScoreFinalRMSE(const wxString& periodString);

    virtual ~asForecastScoreFinalRMSE();

    float Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray);

protected:
private:

};

#endif // ASFORECASTSCOREFINALRMSE_H
