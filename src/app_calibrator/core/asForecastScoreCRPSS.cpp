#include "asForecastScoreCRPSS.h"
#include "asForecastScoreCRPSAR.h"

asForecastScoreCRPSS::asForecastScoreCRPSS()
:
asForecastScore()
{
    m_Score = asForecastScore::CRPSS;
    m_Name = _("CRPS Skill Score");
    m_FullName = _("Continuous Ranked Probability Score Skill Score based on the approximation with the rectangle method");
    m_Order = Desc;
    m_ScaleBest = 1;
    m_ScaleWorst = NaNFloat;
    m_UsesClimatology = true;
}

asForecastScoreCRPSS::~asForecastScoreCRPSS()
{
    //dtor
}

float asForecastScoreCRPSS::Assess(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements)
{
    wxASSERT(ForcastVals.size()>1);
    wxASSERT(nbElements>0);

    wxASSERT(m_ScoreClimatology!=0);

    // First process the CRPS and then the skill score
    asForecastScoreCRPSAR scoreCRPS = asForecastScoreCRPSAR();
    scoreCRPS.SetThreshold(GetThreshold());
    scoreCRPS.SetPercentile(GetPercentile());
    float score = scoreCRPS.Assess(ObservedVal, ForcastVals, nbElements);
    float skillScore = (score-m_ScoreClimatology) / ((float)0-m_ScoreClimatology);

    return skillScore;
}

bool asForecastScoreCRPSS::ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData)
{
    wxASSERT(!asTools::HasNaN(&refVals[0], &refVals[refVals.size()-1]));
    wxASSERT(!asTools::HasNaN(&climatologyData[0], &climatologyData[climatologyData.size()-1]));

    // Containers for final results
    m_ArrayScoresClimatology.resize(refVals.size());

    // Set the original score and process
    asForecastScore* forecastScore = asForecastScore::GetInstance(asForecastScore::CRPSAR);
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
