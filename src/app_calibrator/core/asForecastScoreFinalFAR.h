#ifndef ASFORECASTSCOREFINALFAR_H
#define ASFORECASTSCOREFINALFAR_H

#include <asIncludes.h>
#include <asForecastScoreFinal.h>

class asForecastScoreFinalFAR: public asForecastScoreFinal
{
public:
    asForecastScoreFinalFAR(Period period);

    asForecastScoreFinalFAR(const wxString& periodString);

    virtual ~asForecastScoreFinalFAR();

    float Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray);

protected:
private:

};

#endif // ASFORECASTSCOREFINALFAR_H
