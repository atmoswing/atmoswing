#include "asForecastScoreCRPSaccuracyAR.h"
#include "asForecastScoreCRPSAR.h"
#include "asForecastScoreCRPSsharpnessAR.h"

asForecastScoreCRPSaccuracyAR::asForecastScoreCRPSaccuracyAR()
:
asForecastScore()
{
    m_Score = asForecastScore::CRPSaccuracyAR;
    m_Name = _("CRPS Accuracy Approx Rectangle");
    m_FullName = _("Continuous Ranked Probability Score Accuracy approximation with the rectangle method");
    m_Order = Asc;
    m_ScaleBest = 0;
    m_ScaleWorst = NaNFloat;
}

asForecastScoreCRPSaccuracyAR::~asForecastScoreCRPSaccuracyAR()
{
    //dtor
}

float asForecastScoreCRPSaccuracyAR::Assess(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements)
{
    wxASSERT(ForcastVals.size()>1);
    wxASSERT(nbElements>0);

    asForecastScoreCRPSAR scoreCRPSAR = asForecastScoreCRPSAR();
    float CRPS = scoreCRPSAR.Assess(ObservedVal, ForcastVals, nbElements);
    asForecastScoreCRPSsharpnessAR scoreCRPSsharpnessAR = asForecastScoreCRPSsharpnessAR();
    float CRPSsharpness = scoreCRPSsharpnessAR.Assess(ObservedVal, ForcastVals, nbElements);

    return CRPS-CRPSsharpness;
}

bool asForecastScoreCRPSaccuracyAR::ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData)
{
    return true;
}
