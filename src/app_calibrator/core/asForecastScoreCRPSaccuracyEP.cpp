#include "asForecastScoreCRPSaccuracyEP.h"
#include "asForecastScoreCRPSEP.h"
#include "asForecastScoreCRPSsharpnessEP.h"

asForecastScoreCRPSaccuracyEP::asForecastScoreCRPSaccuracyEP()
:
asForecastScore()
{
    m_Score = asForecastScore::CRPSaccuracyEP;
    m_Name = _("CRPS Accuracy Exact Primitive");
    m_FullName = _("Continuous Ranked Probability Score Accuracy exact solution");
    m_Order = Asc;
    m_ScaleBest = 0;
    m_ScaleWorst = NaNFloat;
}

asForecastScoreCRPSaccuracyEP::~asForecastScoreCRPSaccuracyEP()
{
    //dtor
}

float asForecastScoreCRPSaccuracyEP::Assess(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements)
{
    wxASSERT(ForcastVals.size()>1);
    wxASSERT(nbElements>0);

    asForecastScoreCRPSEP scoreCRPSEP = asForecastScoreCRPSEP();
    float CRPS = scoreCRPSEP.Assess(ObservedVal, ForcastVals, nbElements);
    asForecastScoreCRPSsharpnessEP scoreCRPSsharpnessEP = asForecastScoreCRPSsharpnessEP();
    float CRPSsharpness = scoreCRPSsharpnessEP.Assess(ObservedVal, ForcastVals, nbElements);

    return CRPS-CRPSsharpness;
}

bool asForecastScoreCRPSaccuracyEP::ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData)
{
    return true;
}
