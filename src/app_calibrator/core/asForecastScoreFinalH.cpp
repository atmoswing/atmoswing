#include "asForecastScoreFinalH.h"

asForecastScoreFinalH::asForecastScoreFinalH(Period period)
:
asForecastScoreFinal(period)
{

}

asForecastScoreFinalH::asForecastScoreFinalH(const wxString& periodString)
:
asForecastScoreFinal(periodString)
{

}

asForecastScoreFinalH::~asForecastScoreFinalH()
{
    //dtor
}

float asForecastScoreFinalH::Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray)
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
                    asLogError(wxString::Format(_("The H score (%f) is not an authorized value."), forecastScores[i]));
                    return NaNFloat;
                }
            }
            break;
        }

        default:
        {
            asThrowException(_("Period not yet implemented in asForecastScoreFinalH."));
        }
    }

    float score;

    if (countTot>0)
    {
        if(((float)countA+(float)countC)>0)
        {
            score = (float)countA/((float)countA+(float)countC);
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
