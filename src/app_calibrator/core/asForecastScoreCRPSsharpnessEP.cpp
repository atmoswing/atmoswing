#include "asForecastScoreCRPSsharpnessEP.h"
#include "asForecastScoreCRPSEP.h"

asForecastScoreCRPSsharpnessEP::asForecastScoreCRPSsharpnessEP()
:
asForecastScore()
{
    m_Score = asForecastScore::CRPSsharpnessEP;
    m_Name = _("CRPS Sharpness Exact Primitive");
    m_FullName = _("Continuous Ranked Probability Score Sharpness exact solution");
    m_Order = Asc;
    m_ScaleBest = 0;
    m_ScaleWorst = NaNFloat;
}

asForecastScoreCRPSsharpnessEP::~asForecastScoreCRPSsharpnessEP()
{
    //dtor
}

float asForecastScoreCRPSsharpnessEP::Assess(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements)
{
    wxASSERT(ForcastVals.size()>1);
    wxASSERT(nbElements>0);

    // Check the element numbers vs vector length and the observed value
    if(!CheckInputs(0, ForcastVals, nbElements))
    {
        asLogWarning(_("The inputs are not conform in the CRPS processing function"));
        return NaNFloat;
    }

    // The median
    float xmed = 0;

    // Create the container to sort the data
    Array1DFloat x(nbElements);

    // Remove the NaNs and copy content
    int nbForecasts = CleanNans(ForcastVals, x, nbElements);

    // Sort the forcast array
    asTools::SortArray(&x[0], &x[nbForecasts-1], Asc);

    // Indices for the left and right part (according to the median) of the distribution
    float mid = ((float)nbForecasts-1)/(float)2;
    int indLeftEnd = floor(mid);
    int indRightStart = ceil(mid);

    // Get the median value
    if(indLeftEnd!=indRightStart)
    {
        xmed = x(indLeftEnd)+(x(indRightStart)-x(indLeftEnd))*0.5;
    }
    else
    {
        xmed = x(indLeftEnd);
    }

    asForecastScoreCRPSEP scoreCRPSEP = asForecastScoreCRPSEP();
    float CRPSsharpness = scoreCRPSEP.Assess(xmed, x, nbElements);

    return CRPSsharpness;
}

bool asForecastScoreCRPSsharpnessEP::ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData)
{
    return true;
}
