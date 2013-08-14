#include "asForecastScoreFinalPC.h"

asForecastScoreFinalPC::asForecastScoreFinalPC(Period period)
:
asForecastScoreFinal(period)
{

}

asForecastScoreFinalPC::asForecastScoreFinalPC(const wxString& periodString)
:
asForecastScoreFinal(periodString)
{

}

asForecastScoreFinalPC::~asForecastScoreFinalPC()
{
    //dtor
}

float asForecastScoreFinalPC::Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray)
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
                    asLogError(wxString::Format(_("The PC score (%f) is not an authorized value."), forecastScores[i]));
                    return NaNFloat;
                }
            }
            break;
        }

        default:
        {
            asThrowException(_("Period not yet implemented in asForecastScoreFinalPC."));
        }
    }

    float score;
/*
    wxLogMessage("A=%d",countA);
    wxLogMessage("B=%d",countB);
    wxLogMessage("C=%d",countC);
    wxLogMessage("D=%d",countD);
    wxLogMessage("A+C=%d",countA+countC);
    wxLogMessage("Tot=%d",countTot);
*/
    if (countTot>0)
    {
        score = ((float)countA+(float)countD)/(float)countTot;
    }
    else
    {
        score = NaNFloat;
    }

    return score;
}
