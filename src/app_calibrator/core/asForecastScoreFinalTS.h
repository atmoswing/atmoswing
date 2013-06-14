#ifndef ASFORECASTSCOREFINALTS_H
#define ASFORECASTSCOREFINALTS_H

#include <asIncludes.h>
#include <asForecastScoreFinal.h>

class asForecastScoreFinalTS: public asForecastScoreFinal
{
public:
    asForecastScoreFinalTS(Period period);

    asForecastScoreFinalTS(const wxString& periodString);

    virtual ~asForecastScoreFinalTS();

    float Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray);

protected:
private:

};

#endif // ASFORECASTSCOREFINALTS_H
