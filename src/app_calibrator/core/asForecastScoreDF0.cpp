#include "asForecastScoreDF0.h"

asForecastScoreDF0::asForecastScoreDF0()
:
asForecastScore()
{
    m_Score = asForecastScore::DF0;
    m_Name = _("Difference of F(0)");
    m_FullName = _("Absolute difference of the frequency of null precipitations.");
    m_Order = Asc;
    m_ScaleBest = 0;
    m_ScaleWorst = NaNFloat;
}

asForecastScoreDF0::~asForecastScoreDF0()
{
    //dtor
}

float asForecastScoreDF0::Assess(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements)
{
    wxASSERT(ForcastVals.size()>1);
    wxASSERT(nbElements>0);

    // Check the element numbers vs vector length and the observed value
    if(!CheckInputs(ObservedVal, ForcastVals, nbElements))
    {
        asLogWarning(_("The inputs are not conform in the CRPS processing function"));
        return NaNFloat;
    }

    // Create the container to sort the data
    Array1DFloat x(nbElements);
    float xObs = ObservedVal;

    // Remove the NaNs and copy content
    int nbForecasts = CleanNans(ForcastVals, x, nbElements);
    if(nbForecasts==asNOT_FOUND)
    {
        asLogWarning(_("Only NaNs as inputs in the DF0 processing function."));
        return NaNFloat;
    }
    else if(nbForecasts<=2)
    {
        asLogWarning(_("Not enough elements to process the DF0."));
        return NaNFloat;
    }

    // Sort the forcast array
    asTools::SortArray(&x[0], &x[nbForecasts-1], Asc);

    float score = 0;

    // Containers
    Array1DFloat F(nbForecasts);

    // Parameters for the estimated distribution from Gringorten (a=0.44, b=0.12).
    // Choice based on [Cunnane, C., 1978, Unbiased plotting positions—A review: Journal of Hydrology, v. 37, p. 205–222.]
    // Bontron used a=0.375, b=0.25, that are optimal for a normal distribution
    float irep = 0.44f;
    float nrep = 0.12f;

    // Change the values for unit testing to compare to the results from Grenoble
    if (g_UnitTesting)
    {
        irep = 0.375;
        nrep = 0.25;
    }

	// Identify the last 0
    int indexLastZero = -1;
    for(int i=0; i<nbElements; i++)
    {
        if (x[i]==0)
        {
            indexLastZero = i;
        }
    }

    // Build the cumulative distribution function for the middle of the x
    float divisor = 1.0f/(nbForecasts+nrep);
    for(float i=0; i<nbForecasts; i++)
    {

        F(i)=(i+1.0f-irep)*divisor; // i+1 as i starts from 0
    }

	// Display F(0)analog
    bool dispF0 = false;
    if (dispF0)
    {
        if (indexLastZero>=0)
        {
            wxLogWarning("%f", F(indexLastZero));
        }
        else
        {
            wxLogWarning("%d", 0);
        }
    }

    // Find FxObs, fix xObs and integrate beyond limits
    float FxObs;
    if (xObs>0.0) // If precipitation
    {
        FxObs = 1;
    }
    else
    {
        FxObs = 0;
    }

	score = abs((1.0f-F(indexLastZero))-FxObs);

    return score;
}

bool asForecastScoreDF0::ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData)
{
    return true;
}
