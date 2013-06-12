#ifndef ASFORECASTSCOREFINAL_H
#define ASFORECASTSCOREFINAL_H

#include <asIncludes.h>

#include <asTimeArray.h>

class asForecastScoreFinal: public wxObject
{
public:

    enum Period //!< Enumaration of forcast score combinations
    {
        Total, // total mean
        SpecificPeriod, // partial mean
        Summer, // partial mean on summer only
        Automn, // partial mean on fall only
        Winter, // partial mean on winter only
        Spring, // partial mean on spring only
    };

    asForecastScoreFinal(Period period);

    asForecastScoreFinal(const wxString& periodString);

    virtual ~asForecastScoreFinal();

    static asForecastScoreFinal* GetInstance(const wxString& scoreString, const wxString& periodString);

    virtual float Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray) = 0;

    Period GetPeriod()
    {
        return m_Period;
    }

protected:
    Period m_Period;

private:


};

#endif // ASFORECASTSCOREFINAL_H
