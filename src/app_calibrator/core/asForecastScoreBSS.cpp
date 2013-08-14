#include "asForecastScoreBSS.h"
#include "asForecastScoreBS.h"

asForecastScoreBSS::asForecastScoreBSS()
:
asForecastScore()
{
    m_Score = asForecastScore::BSS;
    m_Name = _("BS Skill Score");
    m_FullName = _("Brier Skill Score");
    m_Order = Desc;
    m_ScaleBest = 1;
    m_ScaleWorst = NaNFloat;
    m_UsesClimatology = true;
}

asForecastScoreBSS::~asForecastScoreBSS()
{
    //dtor
}

float asForecastScoreBSS::Assess(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements)
{
    wxASSERT(ForcastVals.size()>1);
    wxASSERT(nbElements>0);

    wxASSERT(m_ScoreClimatology!=0);

    // First process the BS and then the skill score
    asForecastScoreBS scoreBS = asForecastScoreBS();
    scoreBS.SetThreshold(GetThreshold());
    scoreBS.SetPercentile(GetPercentile());
    float score = scoreBS.Assess(ObservedVal, ForcastVals, nbElements);
    float skillScore = (score-m_ScoreClimatology) / ((float)0-m_ScoreClimatology);

    return skillScore;
}

bool asForecastScoreBSS::ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData)
{
    wxASSERT(!asTools::HasNaN(&refVals[0], &refVals[refVals.size()-1]));
    wxASSERT(!asTools::HasNaN(&climatologyData[0], &climatologyData[climatologyData.size()-1]));

    // Containers for final results
    m_ArrayScoresClimatology.resize(refVals.size());

    // Set the original score and process
    asForecastScore* forecastScore = asForecastScore::GetInstance(asForecastScore::BS);
    forecastScore->SetThreshold(GetThreshold());
    forecastScore->SetPercentile(GetPercentile());

    for (int i_reftime=0; i_reftime<refVals.size(); i_reftime++)
    {
        if (!asTools::IsNaN(refVals(i_reftime)))
        {
            m_ArrayScoresClimatology(i_reftime) = forecastScore->Assess(refVals(i_reftime), climatologyData, climatologyData.size());
        }
        else
        {
            m_ArrayScoresClimatology(i_reftime) = NaNFloat;
        }
    }

    wxDELETE(forecastScore);

    m_ScoreClimatology = asTools::Mean(&m_ArrayScoresClimatology[0],&m_ArrayScoresClimatology[m_ArrayScoresClimatology.size()-1]);

    asLogMessage(wxString::Format(_("Score of the climatology: %g."), m_ScoreClimatology));

    return true;
}
