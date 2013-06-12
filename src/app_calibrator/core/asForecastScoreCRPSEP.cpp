#include "asForecastScoreCRPSEP.h"

asForecastScoreCRPSEP::asForecastScoreCRPSEP()
:
asForecastScore()
{
    m_Score = asForecastScore::CRPSEP;
    m_Name = _("CRPS Exact Primitive");
    m_FullName = _("Continuous Ranked Probability Score exact solution");
    m_Order = Asc;
    m_ScaleBest = 0;
    m_ScaleWorst = NaNFloat;
}

asForecastScoreCRPSEP::~asForecastScoreCRPSEP()
{
    //dtor
}

float asForecastScoreCRPSEP::Assess(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements)
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
    if(nbForecasts==asNOT_FOUND){
        asLogWarning(_("Only NaNs as inputs in the CRPS processing function"));
        return NaNFloat;
    }

    // Sort the forcast array
    asTools::SortArray(&x[0], &x[nbForecasts-1], Asc);

    float CRPS = 0;

    // Containers
    Array1DFloat F(nbForecasts);
    float FxObs = 0;
    float DF, DVal;

    // Parameters for the estimated distribution from Gringorten (a=0.44, b=0.12).
    // Choice based on [Cunnane, C., 1978, Unbiased plotting positions—A review: Journal of Hydrology, v. 37, p. 205–222.]
    // Bontron used a=0.375, b=25, that are optimal for a normal distribution
    float irep = 0.44f;
    float nrep = 0.12f;

    // Change the values for unit testing to compare to the results from Grenoble
    if (g_UnitTesting)
    {
        irep = 0.375;
        nrep = 0.25;
    }

    // Build the cumulative distribution function
    float divisor = 1.0f/(nbForecasts+nrep);
    for(float i=0; i<nbForecasts; i++)
    {

        F(i)=(i+1.0f-irep)*divisor; // i+1 as i starts from 0
    }

    // Indices for the left and right part (according to xObs) of the distribution
    int indLeftStart = 0;
    int indLeftEnd = 0;
    int indRightStart = nbForecasts-1;
    int indRightEnd = nbForecasts-1;

    // Find FxObs, fix xObs and integrate beyond limits
    if (xObs<=x[0]) // If xObs before the distribution
    {
        indRightStart = 0;
        FxObs = 0;
        CRPS += x[indRightStart]-xObs;
    }
    else if (xObs>x[nbForecasts-1]) // If xObs after the distribution
    {
        indLeftEnd = nbForecasts-1;
        FxObs = 1;
        CRPS += xObs-x[indLeftEnd];
    }
    else // If xObs inside the distribution
    {
        indLeftEnd = asTools::SortedArraySearchFloor(&x[0], &x[nbForecasts-1], xObs);
        if((indLeftEnd!=nbForecasts-1) & (indLeftEnd!=asNOT_FOUND) & (indLeftEnd!=asOUT_OF_RANGE))
        {
            indRightStart = indLeftEnd+1;
            if(x(indRightStart)==x(indLeftEnd))
            {
                FxObs = (F(indLeftEnd)+F(indRightStart))*0.5;
            }
            else
            {
                FxObs = F(indLeftEnd)+(F(indRightStart)-F(indLeftEnd))*(xObs-x(indLeftEnd))/(x(indRightStart)-x(indLeftEnd));
            }

            // Integrate the CRPS around FxObs
            // First part - from x(indLeftEnd) to xobs
            DF = FxObs-F(indLeftEnd);
            DVal = xObs-x(indLeftEnd);
            if (DVal!=0)
            {
                float a = DF/DVal;
                float b = -x(indLeftEnd)*a+F(indLeftEnd);
                CRPS += (a*a/3)*(xObs*xObs*xObs-x(indLeftEnd)*x(indLeftEnd)*x(indLeftEnd))
                       + (a*b)*(xObs*xObs-x(indLeftEnd)*x(indLeftEnd))
                       + (b*b)*(xObs-x(indLeftEnd));
            }

            // Second part - from xobs to x(indRightStart)
            DF = F(indRightStart)-FxObs;
            DVal = x(indRightStart)-xObs;
            if (DVal!=0)
            {
                float a = -DF/DVal;
                float b = -xObs*(-a)+FxObs;
                b = 1-b;
                CRPS += (a*a/3)*(x(indRightStart)*x(indRightStart)*x(indRightStart)-xObs*xObs*xObs)
                       + (a*b)*(x(indRightStart)*x(indRightStart)-xObs*xObs)
                       + (b*b)*(x(indRightStart)-xObs);
            }
        }
    }

    // Integrate on the left part
    for (int i=indLeftStart; i<indLeftEnd; i++)
    {
        DF = F(i+1)-F(i);
        DVal = x(i+1)-x(i);
        if (DVal!=0)
        {
            // Build a line y=ax+b
            float a = DF/DVal;
            float b = -x(i)*a+F(i);

            // CRPS after integration with H=0
            CRPS += (a*a/3)*(x(i+1)*x(i+1)*x(i+1)-x(i)*x(i)*x(i))
                    + (a*b)*(x(i+1)*x(i+1)-x(i)*x(i))
                    + (b*b)*(x(i+1)-x(i));
        }
    }

    // Integrate on the right part
    for (int i=indRightStart; i<indRightEnd; i++)
    {
        DF = F(i+1)-F(i);
        DVal = x(i+1)-x(i);
        if (DVal!=0)
        {
            // Build a line y=ax+b and switch it (a -> -a & b -> 1-b) to easily integrate
            float a = -DF/DVal;
            float b = -x(i)*(-a)+F(i);
            b = 1-b;

            // CRPS after integration with H=0 as we switched the axis
            CRPS += (a*a/3)*(x(i+1)*x(i+1)*x(i+1)-x(i)*x(i)*x(i))
                    + (a*b)*(x(i+1)*x(i+1)-x(i)*x(i))
                    + (b*b)*(x(i+1)-x(i));
        }
    }

    return CRPS;
}

bool asForecastScoreCRPSEP::ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData)
{
    return true;
}

