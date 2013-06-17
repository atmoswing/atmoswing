#include "asForecastScoreFinalGSS.h"

asForecastScoreFinalGSS::asForecastScoreFinalGSS(Period period)
:
asForecastScoreFinal(period)
{

}

asForecastScoreFinalGSS::asForecastScoreFinalGSS(const wxString& periodString)
:
asForecastScoreFinal(periodString)
{

}

asForecastScoreFinalGSS::~asForecastScoreFinalGSS()
{
    //dtor
}

float asForecastScoreFinalGSS::Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray)
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
                    asLogError(wxString::Format(_("The GSS score (%f) is not an authorized value."), forecastScores[i]));
                    return NaNFloat;
                }
            }
            break;
        }

        default:
        {
            asThrowException(_("Period not yet implemented in asForecastScoreFinalGSS."));
        }
    }

    float score;

    if (countTot>0)
    {
        float a = (float)countA;
        float b = (float)countB;
        float c = (float)countC;
        float d = (float)countD;
        float aref = 1;
        if ((a+b+c+d)>0)
        {
            aref = (a+b)*(a+c)/(a+b+c+d);
        }
        else
        {
            return 0;
        }
        if ((a-aref+b+c)>0)
        {
            score = (a-aref)/(a-aref+b+c);
        }
        else
        {
            return 0;
        }
    }
    else
    {
        score = NaNFloat;
    }

    return score;
}
