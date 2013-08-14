#include "asForecastScoreFinalHSS.h"

asForecastScoreFinalHSS::asForecastScoreFinalHSS(Period period)
:
asForecastScoreFinal(period)
{

}

asForecastScoreFinalHSS::asForecastScoreFinalHSS(const wxString& periodString)
:
asForecastScoreFinal(periodString)
{

}

asForecastScoreFinalHSS::~asForecastScoreFinalHSS()
{
    //dtor
}

float asForecastScoreFinalHSS::Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray)
{
    wxASSERT(targetDates.rows()>1);
    wxASSERT(forecastScores.rows()>1);

    int countA=0, countB=0, countC=0, countD=0, countTot=0;

    switch (m_Period)
    {
        case (asForecastScoreFinal::Total):
        {
            for (int i=0; i<forecastScores.size(); i++)
            {
                countTot++;
                if (forecastScores[i]==1)
                {
                    countA++;
                }
                else if (forecastScores[i]==2)
                {
                    countB++;
                }
                else if (forecastScores[i]==3)
                {
                    countC++;
                }
                else if (forecastScores[i]==4)
                {
                    countD++;
                }
                else
                {
                    asLogError(wxString::Format(_("The HSS score (%f) is not an authorized value."), forecastScores[i]));
                    return NaNFloat;
                }
            }
            break;
        }

        default:
        {
            asThrowException(_("Period not yet implemented in asForecastScoreFinalHSS."));
        }
    }

    float score;

    if (countTot>0)
    {
        float a = (float)countA;
        float b = (float)countB;
        float c = (float)countC;
        float d = (float)countD;
        float divisor = ((a+c)*(c+d)+(a+b)*(b+d));
        if (divisor>0)
        {
            score = 2*(a*d-b*c)/divisor;
        }
        else
        {
            score = 0;
        }
    }
    else
    {
        score = NaNFloat;
    }

    return score;
}
