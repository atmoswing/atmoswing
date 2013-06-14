#ifndef ASFORECASTSCOREFINALF_H
#define ASFORECASTSCOREFINALF_H

#include <asIncludes.h>
#include <asForecastScoreFinal.h>

class asForecastScoreFinalF: public asForecastScoreFinal
{
public:
    asForecastScoreFinalF(Period period);

    asForecastScoreFinalF(const wxString& periodString);

    virtual ~asForecastScoreFinalF();

    float Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray);

protected:
private:

};

#endif // ASFORECASTSCOREFINALF_H
