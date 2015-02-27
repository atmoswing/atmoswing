/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * When distributing Covered Code, include this CDDL Header Notice in
 * each file and include the License file (licence.txt). If applicable,
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 *
 * The Original Software is AtmoSwing. The Initial Developer of the
 * Original Software is Pascal Horton of the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 */

#include "asTools.h"

void asTools::InitRandom()
{
    srand (time(NULL));
}

int asTools::Random(int min, int max, int step)
{
    // Initialize random seed
    double norm = ((double)rand() / (double)(RAND_MAX) );

    double dmin = (double)min;
    double dmax = (double)max;
    double dstep = (double)step;

    double range = 0;
    if (step<1)
    {
        dstep = 1;
    }

    range = norm*(dmax+dstep-dmin); // Add the step to account for cases at the limits, so they have the same probability.
    range = floor(range/dstep)*dstep;

    double val = range+dmin;
    int intval = asTools::Round(val);

    if (intval>max) intval -= step;
    if (intval<min) intval += step;

    return intval;
}

float asTools::Random(float min, float max, float step)
{
    return (float)asTools::Random((double)min, (double)max, (double)step);
}

double asTools::Random(double min, double max, double step)
{
    // Initialize random seed
    double norm = ((double)rand() / (double)(RAND_MAX) );

    double range = 0;
    if (step==0)
    {
        range = norm*(max-min);
    }
    else
    {
        range = norm*(max+step-min); // Add the step to account for cases at the limits, so they have the same probability.
        range = floor(range/step)*step; // -step/2.0 for cases at the limits.
    }

    double val = range+min;

    if (val>max) val -= step;
    if (val<min) val += step;

    return val;
}

int asTools::RandomNormalDistribution(int mean, int stDev, int step)
{
    return (int)asTools::RandomNormalDistribution((double)mean, (double)stDev, (double)step);
}

float asTools::RandomNormalDistribution(float mean, float stDev, float step)
{
    return (float)asTools::RandomNormalDistribution((double)mean, (double)stDev, (double)step);
}

double asTools::RandomNormalDistribution(double mean, double stDev, double step)
{
    // Initialize random seed
    double u1 = ((double)rand() / (double)(RAND_MAX) );
    double u2 = ((double)rand() / (double)(RAND_MAX) );

    // Exclude 0
    while (u1==0)
    {
        u1 = ((double)rand() / (double)(RAND_MAX) );
    }
    while (u2==0)
    {
        u2 = ((double)rand() / (double)(RAND_MAX) );
    }

    // Box–Muller transform
    double z0 = sqrt(-2*log(u1))*cos(2*M_PI*u2);
    //double z1 = sqrt(-2*log(u1))*sin(2*M_PI*u2);

    z0 *= stDev;

    if (step!=0)
    {
        z0 = step*asTools::Round(z0/step);
    }

    z0 += mean;

    return z0;
}

bool asTools::IsRound(float value)
{
    float valueround = Round(value);
    if (abs(value-valueround)<0.000001)
        return true;
    return false;
}

bool asTools::IsRound(double value)
{
    double valueround = Round(value);
    if (abs(value-valueround)<0.000000000001)
        return true;
    return false;
}

float asTools::Round(float value)
{
    if(value>0)
    {
        return floor( value + 0.5 );
    } else {
        return ceil( value - 0.5 );
    }
}

double asTools::Round(double value)
{
    if(value>0)
    {
        return floor( value + 0.5 );
    } else {
        return ceil( value - 0.5 );
    }
}

float asTools::Mean(int* pArrStart, int* pArrEnd)
{
    float sum = 0, nb = 0;
    for (int i=0; i<=pArrEnd-pArrStart; i++)
    {
        // Dones't check for NaNs, as there are no NaN for intergers
        sum += (float) *(pArrStart+i);
        nb++;
    }
    return sum/nb;
}

float asTools::Mean(float* pArrStart, float* pArrEnd)
{
    float sum = 0, nb = 0;
    for (int i=0; i<=pArrEnd-pArrStart; i++)
    {
        if(!asTools::IsNaN(*(pArrStart+i)))
        {
            sum += *(pArrStart+i);
            nb++;
        }
    }
    return sum/nb;
}

double asTools::Mean(double* pArrStart, double* pArrEnd)
{
    double sum = 0, nb = 0;
    for (int i=0; i<=pArrEnd-pArrStart; i++)
    {
        if(!asTools::IsNaN(*(pArrStart+i)))
        {
            sum += *(pArrStart+i);
            nb++;
        }
    }
    return sum/nb;
}

float asTools::StDev(int* pArrStart, int* pArrEnd, int sample)
{
    float sum = 0, sumsquares = 0, nb = 0;
    for (int i=0; i<=pArrEnd-pArrStart; i++)
    {
        // Dones't check for NaNs, as there are no NaN for intergers
        sum += *(pArrStart+i);
        sumsquares += (*(pArrStart+i)) * (*(pArrStart+i));
        nb++;
    }

    if (sample == asSAMPLE)
    {
        return sqrt((sumsquares - (sum*sum/nb))/(nb-1));
    }
    else if (sample == asENTIRE_POPULATION)
    {
        return sqrt((sumsquares - (sum*sum/nb))/(nb));
    }
    else
    {
        return NaNFloat;
    }
}

float asTools::StDev(float* pArrStart, float* pArrEnd, int sample)
{
    float sum = 0, sumsquares = 0, nb = 0;
    for (int i=0; i<=pArrEnd-pArrStart; i++)
    {
        if(!asTools::IsNaN(*(pArrStart+i)))
        {
            sum += *(pArrStart+i);
            sumsquares += (*(pArrStart+i)) * (*(pArrStart+i));
            nb++;
        }
    }

    if (sample == asSAMPLE)
    {
        return sqrt((sumsquares - (sum*sum/nb))/(nb-1));
    }
    else if (sample == asENTIRE_POPULATION)
    {
        return sqrt((sumsquares - (sum*sum/nb))/(nb));
    }
    else
    {
        return NaNFloat;
    }
}

double asTools::StDev(double* pArrStart, double* pArrEnd, int sample)
{
    double sum = 0, sumsquares = 0, nb = 0;
    for (int i=0; i<=pArrEnd-pArrStart; i++)
    {
        if(!asTools::IsNaN(*(pArrStart+i)))
        {
            sum += *(pArrStart+i);
            sumsquares += (*(pArrStart+i)) * (*(pArrStart+i));
            nb++;
        }
    }

    if (sample == asSAMPLE)
    {
        return sqrt((sumsquares - (sum*sum/nb))/(nb-1));
    }
    else if (sample == asENTIRE_POPULATION)
    {
        return sqrt((sumsquares - (sum*sum/nb))/(nb));
    }
    else
    {
        return NaNDouble;
    }
}

Array1DFloat asTools::GetCumulativeFrequency(int size)
{
    Array1DFloat F(size);

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

    float divisor = 1.0f/(size+nrep);
    for(float i=0; i<size; i++)
    {
        F(i)=(i+1.0f-irep)*divisor; // i+1 as i starts from 0
    }

    return F;
}

float asTools::GetValueForQuantile(Array1DFloat &values, float quantile)
{
    float value = NaNFloat;
    int size = values.size();

    // Sort the forcast array
    asTools::SortArray(&values[0], &values[size-1], Asc);

    // Cumulative frequency
    Array1DFloat F = asTools::GetCumulativeFrequency(size);

    // Check limits
    if (quantile<=F[0]) return values[0];
    if (quantile>=F[size-1]) return values[size-1];

    // Indices for the left and right part (according to xObs)
    int indLeft = asTools::SortedArraySearchFloor(&F[0], &F[size-1], quantile);
    int indRight = asTools::SortedArraySearchCeil(&F[0], &F[size-1], quantile);
    wxASSERT(indLeft>=0);
    wxASSERT(indRight>=0);
    wxASSERT(indLeft<=indRight);

    if (indLeft==indRight)
    {
        value = values[indLeft];
    }
    else
    {
        value = values(indLeft)+(values(indRight)-values(indLeft))*(quantile-F(indLeft))/(F(indRight)-F(indLeft));
    }

    return value;
}

bool asTools::IsNaN(int value)
{
    return value == NaNInt;
}

bool asTools::IsNaN(float value)
{
    return value != value;
}

bool asTools::IsNaN(double value)
{
    return value != value;
}

bool asTools::IsInf(float value)
{
    return value == InfFloat;
}

bool asTools::IsInf(double value)
{
    return value == InfDouble;
}

bool asTools::IsInf(long double value)
{
    return value == InfLongDouble;
}

int asTools::CountNotNaN(const float* pArrStart, const float* pArrEnd)
{
    int counter = 0;

    for (int i=0; i<=pArrEnd-pArrStart; i++)
    {
        if(!asTools::IsNaN(*(pArrStart+i)))
        {
            counter++;
        }
    }

    return counter;
}

int asTools::CountNotNaN(const double* pArrStart, const double* pArrEnd)
{
    int counter = 0;

    for (int i=0; i<=pArrEnd-pArrStart; i++)
    {
        if(!asTools::IsNaN(*(pArrStart+i)))
        {
            counter++;
        }
    }

    return counter;
}

bool asTools::HasNaN(const Array2DFloat &data)
{
    for (int i=0; i<data.rows(); i++)
    {
        for (int j=0; j<data.cols(); j++)
        {
            if(asTools::IsNaN(data(i,j)))
            {
                return true;
            }
        }
    }

    return false;
}

bool asTools::HasNaN(const float* pArrStart, const float* pArrEnd)
{
    for (int i=0; i<=pArrEnd-pArrStart; i++)
    {
        if(asTools::IsNaN(*(pArrStart+i)))
        {
            return true;
        }
    }

    return false;
}

bool asTools::HasNaN(const double* pArrStart, const double* pArrEnd)
{
    for (int i=0; i<=pArrEnd-pArrStart; i++)
    {
        if(asTools::IsNaN(*(pArrStart+i)))
        {
            return true;
        }
    }

    return false;
}

int asTools::MinArray(int* pArrStart, int* pArrEnd)
{
    int min;
    int i = 0;

    min = *(pArrStart+i);

    for (i=i; i<=pArrEnd-pArrStart; i++)
    {
        if(*(pArrStart+i)<min)
        {
            min = *(pArrStart+i);
        }
    }

    return min;
}

float asTools::MinArray(float* pArrStart, float* pArrEnd)
{
    float min;
    int i = 0;

    // Manage the case where the first elements are NaNs
    while (asTools::IsNaN(*(pArrStart+i)))
    {
        i++;
    }
    min = *(pArrStart+i);

    for (i=i; i<=pArrEnd-pArrStart; i++)
    {
        if(!asTools::IsNaN(*(pArrStart+i)))
        {
            if(*(pArrStart+i)<min)
            {
                min = *(pArrStart+i);
            }
        }
    }

    return min;
}

double asTools::MinArray(double* pArrStart, double* pArrEnd)
{
    double min;
    int i = 0;

    // Manage the case where the first elements are NaNs
    while (asTools::IsNaN(*(pArrStart+i)))
    {
        i++;
    }
    min = *(pArrStart+i);

    for (i=i; i<=pArrEnd-pArrStart; i++)
    {
        if(!asTools::IsNaN(*(pArrStart+i)))
        {
            if(*(pArrStart+i)<min)
            {
                min = *(pArrStart+i);
            }
        }
    }

    return min;
}

int asTools::MinArrayIndex(int* pArrStart, int* pArrEnd)
{
    int min;
    int index;
    int i = 0;

    min = *(pArrStart+i);
    index = 0;

    for (i=i; i<=pArrEnd-pArrStart; i++)
    {
        if(*(pArrStart+i)<min)
        {
            min = *(pArrStart+i);
            index = i;
        }
    }

    return index;
}

int asTools::MinArrayIndex(float* pArrStart, float* pArrEnd)
{
    float min;
    int index;
    int i = 0;

    min = *(pArrStart+i);
    index = 0;

    for (i=i; i<=pArrEnd-pArrStart; i++)
    {
        if(*(pArrStart+i)<min)
        {
            min = *(pArrStart+i);
            index = i;
        }
    }

    return index;
}

int asTools::MinArrayIndex(double* pArrStart, double* pArrEnd)
{
    double min;
    int index;
    int i = 0;

    min = *(pArrStart+i);
    index = 0;

    for (i=i; i<=pArrEnd-pArrStart; i++)
    {
        if(*(pArrStart+i)<min)
        {
            min = *(pArrStart+i);
            index = i;
        }
    }

    return index;
}

int asTools::MaxArray(int* pArrStart, int* pArrEnd)
{
    int max;
    int i = 0;

    max = *(pArrStart+i);

    for (i=i; i<=pArrEnd-pArrStart; i++)
    {
        if(*(pArrStart+i)>max)
        {
            max = *(pArrStart+i);
        }
    }

    return max;
}

float asTools::MaxArray(float* pArrStart, float* pArrEnd)
{
    float max;
    int i = 0;

    // Manage the case where the first elements are NaNs
    while (asTools::IsNaN(*(pArrStart+i)))
    {
        i++;
    }
    max = *(pArrStart+i);

    for (i=i; i<=pArrEnd-pArrStart; i++)
    {
        if(!asTools::IsNaN(*(pArrStart+i)))
        {
            if(*(pArrStart+i)>max)
            {
                max = *(pArrStart+i);
            }
        }
    }

    return max;
}

double asTools::MaxArray(double* pArrStart, double* pArrEnd)
{
    double max;
    int i = 0;

    // Manage the case where the first elements are NaNs
    while (asTools::IsNaN(*(pArrStart+i)))
    {
        i++;
    }
    max = *(pArrStart+i);

    for (i=i; i<=pArrEnd-pArrStart; i++)
    {
        if(!asTools::IsNaN(*(pArrStart+i)))
        {
            if(*(pArrStart+i)>max)
            {
                max = *(pArrStart+i);
            }
        }
    }

    return max;
}

int asTools::MaxArrayIndex(int* pArrStart, int* pArrEnd)
{
    int max;
    int index;
    int i = 0;

    max = *(pArrStart+i);
    index = 0;

    for (i=i; i<=pArrEnd-pArrStart; i++)
    {
        if(*(pArrStart+i)>max)
        {
            max = *(pArrStart+i);
            index = i;
        }
    }

    return index;
}

int asTools::MaxArrayIndex(float* pArrStart, float* pArrEnd)
{
    float max;
    int index;
    int i = 0;

    max = *(pArrStart+i);
    index = 0;

    for (i=i; i<=pArrEnd-pArrStart; i++)
    {
        if(*(pArrStart+i)>max)
        {
            max = *(pArrStart+i);
            index = i;
        }
    }

    return index;
}

int asTools::MaxArrayIndex(double* pArrStart, double* pArrEnd)
{
    double max;
    int index;
    int i = 0;

    max = *(pArrStart+i);
    index = 0;

    for (i=i; i<=pArrEnd-pArrStart; i++)
    {
        if(*(pArrStart+i)>max)
        {
            max = *(pArrStart+i);
            index = i;
        }
    }

    return index;
}

int asTools::MinArrayStep(int* pArrStart, int* pArrEnd, int tolerance)
{
    // Copy data to not alter original array
    Array1DInt copyData(pArrEnd-pArrStart+1);

    for (int i=0; i<=pArrEnd-pArrStart; i++)
    {
        copyData[i] = *(pArrStart+i);
    }

    // Sort the array
    asTools::SortArray(&copyData[0], &copyData[copyData.size()-1], Asc);

    // Find min step
    int i = 1;

    while (copyData[i]-copyData[i-1]<=tolerance)
    {
        i++;
        if(i==copyData.size())
        {
            return asNOT_FOUND;
        }
    }

    int minstep = copyData[i]-copyData[i-1];

    for (i=i; i<copyData.size(); i++)
    {
        int currentval = abs(copyData[i]-copyData[i-1]);
        if((currentval<minstep) & (currentval>tolerance))
        {
            minstep = currentval;
        }
    }

    return minstep;
}

float asTools::MinArrayStep(float* pArrStart, float* pArrEnd, float tolerance)
{
    int nbNotNans = asTools::CountNotNaN(pArrStart, pArrEnd);
    int j = 0;

    // Copy data to not alter original array
    Array1DFloat copyData(nbNotNans);

    // Remove Nans
    for (int i=0; i<=pArrEnd-pArrStart; i++)
    {
        if(!asTools::IsNaN(*(pArrStart+i)))
        {
            copyData[j] = *(pArrStart+i);
            j++;
        }
    }

    // Sort the array
    asTools::SortArray(&copyData[0], &copyData[copyData.size()-1], Asc);

    // Find min step
    int i = 1;

    while (copyData[i]-copyData[i-1]<=tolerance)
    {
        i++;
        if(i==copyData.size())
        {
            return asNOT_FOUND;
        }
    }

    float minstep = copyData[i]-copyData[i-1];

    for (i=i; i<copyData.size(); i++)
    {
        float currentval = abs(copyData[i]-copyData[i-1]);
        if((currentval<minstep) & (currentval>tolerance))
        {
            minstep = currentval;
        }
    }

    return minstep;
}

double asTools::MinArrayStep(double* pArrStart, double* pArrEnd, double tolerance)
{
    int nbNotNans = asTools::CountNotNaN(pArrStart, pArrEnd);
    int j = 0;

    // Copy data to not alter original array
    Array1DDouble copyData = Array1DDouble(nbNotNans);

    // Remove Nans
    for (int i=0; i<=pArrEnd-pArrStart; i++)
    {
        if(!asTools::IsNaN(*(pArrStart+i)))
        {
            copyData[j] = *(pArrStart+i);
            j++;
        }
    }

    // Sort the array
    asTools::SortArray(&copyData[0], &copyData[copyData.size()-1], Asc);

    // Find min step
    int i = 1;

    while (copyData[i]-copyData[i-1]<=tolerance)
    {
        i++;
        if(i==copyData.size())
        {
            return asNOT_FOUND;
        }
    }

    double minstep = copyData[i]-copyData[i-1];

    for (i=i; i<copyData.size(); i++)
    {
        double currentval = abs(copyData[i]-copyData[i-1]);
        if((currentval<minstep) & (currentval>tolerance))
        {
            minstep = currentval;
        }
    }

    return minstep;
}

Array1DInt asTools::ExtractUniqueValues(int* pArrStart, int* pArrEnd, int tolerance)
{
    int j=0;

    // Copy data to not alter original array
    VectorInt copyData(pArrEnd-pArrStart+1);

    // Remove Nans
    for (int i=0; i<=pArrEnd-pArrStart; i++)
    {
        copyData[j] = *(pArrStart+i);
        j++;
    }

    // Sort the array
    asTools::SortArray(&copyData[0], &copyData[copyData.size()-1], Asc);

    // Extract unique values
    VectorInt copyDataUniques;
    copyDataUniques.reserve(pArrEnd-pArrStart+1);
    copyDataUniques.push_back(copyData[0]); // Add first value

    for (unsigned int i=1; i<copyData.size(); i++)
    {
        if((abs(copyData[i]-copyData[i-1])>tolerance))
        {
            copyDataUniques.push_back(copyData[i]);
        }
    }

    // Copy data to the final container
    Array1DInt resultArray(copyDataUniques.size());

    for (unsigned int i=0; i<copyDataUniques.size(); i++)
    {
        resultArray[i] = copyDataUniques[i];
    }

    return resultArray;
}

Array1DFloat asTools::ExtractUniqueValues(float* pArrStart, float* pArrEnd, float tolerance)
{
    int nbNotNans = asTools::CountNotNaN(pArrStart, pArrEnd);
    int j=0;

    // Copy data to not alter original array
    VectorFloat copyData(nbNotNans);

    // Remove Nans
    for (int i=0; i<=pArrEnd-pArrStart; i++)
    {
        if(!asTools::IsNaN(*(pArrStart+i)))
        {
            copyData[j] = *(pArrStart+i);
            j++;
        }
    }

    // Sort the array
    asTools::SortArray(&copyData[0], &copyData[copyData.size()-1], Asc);

    // Extract unique values
    VectorFloat copyDataUniques;
    copyDataUniques.reserve(nbNotNans);
    copyDataUniques.push_back(copyData[0]); // Add first value

    for (unsigned int i=1; i<copyData.size(); i++)
    {
        if((abs(copyData[i]-copyData[i-1])>tolerance))
        {
            copyDataUniques.push_back(copyData[i]);
        }
    }

    // Copy data to the final container
    Array1DFloat resultArray(copyDataUniques.size());

    for (unsigned int i=0; i<copyDataUniques.size(); i++)
    {
        resultArray[i] = copyDataUniques[i];
    }

    return resultArray;
}

Array1DDouble asTools::ExtractUniqueValues(double* pArrStart, double* pArrEnd, double tolerance)
{
    int nbNotNans = asTools::CountNotNaN(pArrStart, pArrEnd);
    int j=0;

    // Copy data to not alter original array
    VectorDouble copyData = VectorDouble(nbNotNans);

    // Remove Nans
    for (int i=0; i<=pArrEnd-pArrStart; i++)
    {
        if(!asTools::IsNaN(*(pArrStart+i)))
        {
            copyData[j] = *(pArrStart+i);
            j++;
        }
    }

    // Sort the array
    asTools::SortArray(&copyData[0], &copyData[copyData.size()-1], Asc);

    // Extract unique values
    VectorDouble copyDataUniques;
    copyDataUniques.reserve(nbNotNans);
    copyDataUniques.push_back(copyData[0]); // Add first value

    for (unsigned int i=1; i<copyData.size(); i++)
    {
        if((abs(copyData[i]-copyData[i-1])>tolerance))
        {
            copyDataUniques.push_back(copyData[i]);
        }
    }

    // Copy data to the final container
    Array1DDouble resultArray = Array1DDouble(copyDataUniques.size());

    for (unsigned int i=0; i<copyDataUniques.size(); i++)
    {
        resultArray[i] = copyDataUniques[i];
    }

    return resultArray;
}

int asTools::SortedArraySearch(int* pArrStart, int* pArrEnd, int targetvalue, int tolerance, int showWarning)
{
    return SortedArraySearchT<int>(pArrStart, pArrEnd, targetvalue, tolerance, showWarning);
}

int asTools::SortedArraySearch(float* pArrStart, float* pArrEnd, float targetvalue, float tolerance, int showWarning)
{
    return SortedArraySearchT<float>(pArrStart, pArrEnd, targetvalue, tolerance, showWarning);
}

int asTools::SortedArraySearch(double* pArrStart, double* pArrEnd, double targetvalue, double tolerance, int showWarning)
{
    return SortedArraySearchT<double>(pArrStart, pArrEnd, targetvalue, tolerance, showWarning);
}

template< class T >
int asTools::SortedArraySearchT(T* pArrStart, T* pArrEnd, T targetvalue, T tolerance, int showWarning)
{
    wxASSERT(pArrStart);
    wxASSERT(pArrEnd);

    T *pFirst = NULL, *pMid = NULL, *pLast = NULL;
    int vlength = pArrEnd-pArrStart;

    // Initialize first and last variables.
    pFirst = pArrStart;
    pLast = pArrEnd;

    // Check array order
    if (*pLast>*pFirst)
    {
        // Binary search
        while (pFirst <= pLast)
        {
            vlength = pLast - pFirst;
            pMid = pFirst + vlength/2;
            if (targetvalue-tolerance > *pMid)
            {
                pFirst = pMid + 1;
            }
            else if (targetvalue+tolerance < *pMid)
            {
                pLast = pMid - 1;
            }
            else
            {
                // Return found index
                return pMid - pArrStart;
            }
        }

        // Check the pointers
        if (pLast-pArrStart<0)
        {
            pLast = pArrStart;
        } else if (pLast-pArrEnd>0) {
            pLast = pArrEnd;
        }

        // If the value was not found, return closest value inside tolerance
        if (abs(targetvalue-*pLast)<=abs(targetvalue-*(pLast+1)))
        {
            if(abs(targetvalue-*pLast)<=tolerance)
            {
                return pLast - pArrStart;
            } else {
                // Check that the value is whithin the array. Do it here to allow a margin for the tolerance
                if (targetvalue>*pArrEnd || targetvalue<*pArrStart)
                {
                    if (showWarning == asSHOW_WARNINGS){
                        asLogWarning(_("The value is out of the array range."));
                    }
                    return asOUT_OF_RANGE;
                }
                if (showWarning == asSHOW_WARNINGS){
                    asLogWarning(_("The value was not found in the array."));
                }
                return asNOT_FOUND;
            }
        } else {
            if(abs(targetvalue-*(pLast+1))<=tolerance)
            {
                return pLast - pArrStart + 1;
            } else {
                // Check that the value is whithin the array. Do it here to allow a margin for the tolerance
                if (targetvalue>*pArrEnd || targetvalue<*pArrStart)
                {
                    if (showWarning == asSHOW_WARNINGS){
                        asLogWarning(_("The value is out of the array range."));
                    }
                    return asOUT_OF_RANGE;
                }
                if (showWarning == asSHOW_WARNINGS){
                    asLogWarning(_("The value was not found in the array."));
                }
                return asNOT_FOUND;
            }
        }
    }
    else if (*pLast<*pFirst)
    {
        // Binary search
        while (pFirst <= pLast)
        {
            vlength = pLast - pFirst;
            pMid = pFirst + vlength/2;
            if (targetvalue-tolerance > *pMid)
            {
                pLast = pMid - 1;
            }
            else if (targetvalue+tolerance < *pMid)
            {
                pFirst = pMid + 1;
            }
            else
            {
                // Return found index
                return pMid - pArrStart;
            }
        }

        // Check the pointers
        if (pFirst-pArrStart<0)
        {
            pFirst = pArrStart;
        } else if (pFirst-pArrEnd>0) {
            pFirst = pArrEnd;
        }

        // If the value was not found, return closest value inside tolerance
        if (abs(targetvalue-*pFirst)<=abs(targetvalue-*(pFirst-1)))
        {
            if(abs(targetvalue-*pFirst)<=tolerance)
            {
                return pFirst - pArrStart;
            } else {
                // Check that the value is whithin the array. Do it here to allow a margin for the tolerance.
                if (targetvalue<*pArrEnd || targetvalue>*pArrStart)
                {
                    if (showWarning == asSHOW_WARNINGS){
                        asLogWarning(_("The value is out of the array range."));
                    }
                    return asOUT_OF_RANGE;
                }
                if (showWarning == asSHOW_WARNINGS){
                    asLogWarning(_("The value was not found in the array."));
                }
                return asNOT_FOUND;
            }
        } else {
            if(abs(targetvalue-*(pLast+1))<=tolerance)
            {
                return pFirst - pArrStart - 1;
            } else {
                // Check that the value is whithin the array. Do it here to allow a margin for the tolerance.
                if (targetvalue<*pArrEnd || targetvalue>*pArrStart)
                {
                    if (showWarning == asSHOW_WARNINGS){
                        asLogWarning(_("The value is out of the array range."));
                    }
                    return asOUT_OF_RANGE;
                }
                if (showWarning == asSHOW_WARNINGS){
                    asLogWarning(_("The value was not found in the array."));
                }
                return asNOT_FOUND;
            }
        }
    }
    else
    {
        if (pLast-pFirst==0)
        {
            if( *pFirst>=targetvalue-tolerance && *pFirst<=targetvalue+tolerance )
            {
                return 0; // Value corresponds
            } else {
                return asOUT_OF_RANGE;
            }
        }

        if( *pFirst>=targetvalue-tolerance && *pFirst<=targetvalue+tolerance )
        {
            return 0; // Value corresponds
        } else {
            return asOUT_OF_RANGE;
        }
    }
}

int asTools::SortedArraySearchClosest(int* pArrStart, int* pArrEnd, int targetvalue, int showWarning)
{
    return SortedArraySearchClosestT<int>(pArrStart, pArrEnd, targetvalue, showWarning);
}

int asTools::SortedArraySearchClosest(float* pArrStart, float* pArrEnd, float targetvalue, int showWarning)
{
    return SortedArraySearchClosestT<float>(pArrStart, pArrEnd, targetvalue, showWarning);
}

int asTools::SortedArraySearchClosest(double* pArrStart, double* pArrEnd, double targetvalue, int showWarning)
{
    return SortedArraySearchClosestT<double>(pArrStart, pArrEnd, targetvalue, showWarning);
}

template< class T >
int asTools::SortedArraySearchClosestT(T* pArrStart, T* pArrEnd, T targetvalue, int showWarning)
{
    wxASSERT(pArrStart);
    wxASSERT(pArrEnd);

    T *pFirst = NULL, *pMid = NULL, *pLast = NULL;
    int vlength = pArrEnd-pArrStart;
    int IndexMid = asNOT_FOUND;

    // Initialize first and last variables.
    pFirst = pArrStart;
    pLast = pArrEnd;

    // Check array order
    if (*pLast>*pFirst)
    {
        // Check that the value is whithin the array
        if (targetvalue>*pLast || targetvalue<*pFirst)
        {
            if (showWarning == asSHOW_WARNINGS){
                asLogWarning(_("The value is out of the array range."));
            }
            return asOUT_OF_RANGE;
        }

        // Binary search
        while (pFirst <= pLast)
        {
            vlength = pLast - pFirst;
            pMid = pFirst + vlength/2;
            if (targetvalue > *pMid)
            {
                pFirst = pMid + 1;
            }
            else if (targetvalue < *pMid)
            {
                pLast = pMid - 1;
            }
            else
            {
                // Return found index
                IndexMid = pMid - pArrStart;
                return IndexMid;
            }
        }

        // Check the pointers
        if (pLast-pArrStart<0)
        {
            pLast = pArrStart;
        } else if (pLast-pArrEnd>0) {
            pLast = pArrEnd;
        }

        // If the value was not found, return closest value
        if (abs(targetvalue-*pLast)<=abs(targetvalue-*(pLast+1)))
        {
            return pLast - pArrStart;
        } else {
            return pLast - pArrStart + 1;
        }
    }
    else if (*pLast<*pFirst)
    {
        // Check that the value is whithin the array
        if (targetvalue<*pLast || targetvalue>*pFirst)
        {
            if (showWarning == asSHOW_WARNINGS){
                asLogWarning(_("The value is out of the array range."));
            }
            return asOUT_OF_RANGE;
        }

        // Binary search
        while (pFirst <= pLast)
        {
            vlength = pLast - pFirst;
            pMid = pFirst + vlength/2;
            if (targetvalue > *pMid)
            {
                pLast = pMid - 1;
            }
            else if (targetvalue < *pMid)
            {
                pFirst = pMid + 1;
            }
            else
            {
                // Return found index
                IndexMid = pMid - pArrStart;
                return IndexMid;
            }
        }

        // Check the pointers
        if (pFirst-pArrStart<0)
        {
            pFirst = pArrStart;
        } else if (pFirst-pArrEnd>0) {
            pFirst = pArrEnd;
        }

        // If the value was not found, return closest value
        if (abs(targetvalue-*pFirst)<=abs(targetvalue-*(pFirst-1)))
        {
            return pFirst - pArrStart;
        } else {
            return pFirst - pArrStart - 1;
        }
    }
    else
    {
        if (pLast-pFirst==0)
        {
            if( *pFirst==targetvalue )
            {
                return 0; // Value corresponds
            } else {
                return asOUT_OF_RANGE;
            }
        }

        if( *pFirst==targetvalue )
        {
            return 0; // Value corresponds
        } else {
            return asOUT_OF_RANGE;
        }
    }
}

int asTools::SortedArraySearchFloor(int* pArrStart, int* pArrEnd, int targetvalue, int showWarning)
{
    return SortedArraySearchFloorT<int>(pArrStart, pArrEnd, targetvalue, showWarning);
}

int asTools::SortedArraySearchFloor(float* pArrStart, float* pArrEnd, float targetvalue, int showWarning)
{
    return SortedArraySearchFloorT<float>(pArrStart, pArrEnd, targetvalue, showWarning);
}

int asTools::SortedArraySearchFloor(double* pArrStart, double* pArrEnd, double targetvalue, int showWarning)
{
    return SortedArraySearchFloorT<double>(pArrStart, pArrEnd, targetvalue, showWarning);
}

template< class T >
int asTools::SortedArraySearchFloorT(T* pArrStart, T* pArrEnd, T targetvalue, int showWarning)
{
    wxASSERT(pArrStart);
    wxASSERT(pArrEnd);

    T *pFirst = NULL, *pMid = NULL, *pLast = NULL;
    int vlength = pArrEnd-pArrStart;

    // Initialize first and last variables.
    pFirst = pArrStart;
    pLast = pArrEnd;

    // Check array order
    if ( *pLast > *pFirst )
    {
        // Check that the value is whithin the array
        if (targetvalue>*pLast || targetvalue<*pFirst)
        {
            if (showWarning == asSHOW_WARNINGS)
            {
                asLogWarning(_("The value is out of the array range."));
            }
            return asOUT_OF_RANGE;
        }

        // Binary search
        while ( pFirst <= pLast )
        {
            vlength = pLast - pFirst;
            pMid = pFirst + vlength/2;
            if (targetvalue > *pMid)
            {
                pFirst = pMid + 1;
            }
            else if (targetvalue < *pMid)
            {
                pLast = pMid - 1;
            }
            else
            {
                // Return found index
                return pMid - pArrStart;
            }
        }

        // Check the pointers
        if (pLast-pArrStart<0)
        {
            pLast = pArrStart;
        } else if (pLast-pArrEnd>0) {
            pLast = pArrEnd;
        }

        // If the value was not found, return floor value
        return pLast - pArrStart;
    }
    else if (*pLast<*pFirst)
    {
        // Check that the value is whithin the array
        if (targetvalue<*pLast || targetvalue>*pFirst)
        {
            if (showWarning == asSHOW_WARNINGS){
                asLogWarning(_("The value is out of the array range."));
            }
            return asOUT_OF_RANGE;
        }

        // Binary search
        while (pFirst <= pLast)
        {
            vlength = pLast - pFirst;
            pMid = pFirst + vlength/2;
            if (targetvalue > *pMid)
            {
                pLast = pMid - 1;
            }
            else if (targetvalue < *pMid)
            {
                pFirst = pMid + 1;
            }
            else
            {
                // Return found index
                return pMid - pArrStart;
            }
        }

        // Check the pointers
        if (pFirst-pArrStart<0)
        {
            pFirst = pArrStart;
        } else if (pFirst-pArrEnd>0) {
            pFirst = pArrEnd;
        }

        // If the value was not found, return floor value
        return pFirst - pArrStart;
    }
    else
    {
        if (pLast-pFirst==0)
        {
            if( *pFirst==targetvalue )
            {
                return 0; // Value corresponds
            } else {
                return asOUT_OF_RANGE;
            }
        }

        if( *pFirst==targetvalue )
        {
            return 0; // Value corresponds
        } else {
            return asOUT_OF_RANGE;
        }
    }
}


int asTools::SortedArraySearchCeil(int* pArrStart, int* pArrEnd, int targetvalue, int showWarning)
{
    return SortedArraySearchCeilT<int>(pArrStart, pArrEnd, targetvalue, showWarning);
}

int asTools::SortedArraySearchCeil(float* pArrStart, float* pArrEnd, float targetvalue, int showWarning)
{
    return SortedArraySearchCeilT<float>(pArrStart, pArrEnd, targetvalue, showWarning);
}

int asTools::SortedArraySearchCeil(double* pArrStart, double* pArrEnd, double targetvalue, int showWarning)
{
    return SortedArraySearchCeilT<double>(pArrStart, pArrEnd, targetvalue, showWarning);
}

template< class T >
int asTools::SortedArraySearchCeilT(T* pArrStart, T* pArrEnd, T targetvalue, int showWarning)
{
    wxASSERT(pArrStart);
    wxASSERT(pArrEnd);

    T *pFirst = NULL, *pMid = NULL, *pLast = NULL;
    int vlength = pArrEnd-pArrStart;

    // Initialize first and last variables.
    pFirst = pArrStart;
    pLast = pArrEnd;

    // Check array order
    if (*pLast>*pFirst)
    {
        // Check that the value is whithin the array
        if (targetvalue>*pLast || targetvalue<*pFirst)
        {
            if (showWarning == asSHOW_WARNINGS){
                asLogWarning(_("The value is out of the array range."));
            }
            return asOUT_OF_RANGE;
        }

        // Binary search
        while (pFirst <= pLast)
        {
            vlength = pLast - pFirst;
            pMid = pFirst + vlength/2;
            if (targetvalue > *pMid)
            {
                pFirst = pMid + 1;
            }
            else if (targetvalue < *pMid)
            {
                pLast = pMid - 1;
            }
            else
            {
                // Return found index
                return pMid - pArrStart;
            }
        }

        // Check the pointers
        if (pLast-pArrStart<0)
        {
            pLast = pArrStart;
        } else if (pLast-pArrEnd>0) {
            pLast = pArrEnd;
        }

        // If the value was not found, return ceil value
        return pLast - pArrStart + 1;
    }
    else if (*pLast<*pFirst)
    {
        // Check that the value is whithin the array
        if (targetvalue<*pLast || targetvalue>*pFirst)
        {
            if (showWarning == asSHOW_WARNINGS){
                asLogWarning(_("The value is out of the array range."));
            }
            return asOUT_OF_RANGE;
        }

        // Binary search
        while (pFirst <= pLast)
        {
            vlength = pLast - pFirst;
            pMid = pFirst + vlength/2;
            if (targetvalue > *pMid)
            {
                pLast = pMid - 1;
            }
            else if (targetvalue < *pMid)
            {
                pFirst = pMid + 1;
            }
            else
            {
                // Return found index
                return pMid - pArrStart;
            }
        }

        // Check the pointers
        if (pFirst-pArrStart<0)
        {
            pFirst = pArrStart;
        } else if (pFirst-pArrEnd>0) {
            pFirst = pArrEnd;
        }

        // If the value was not found, return ceil value
        return pFirst - pArrStart - 1;
    }
    else
    {
        if (pLast-pFirst==0)
        {
            if( *pFirst==targetvalue )
            {
                return 0; // Value corresponds
            } else {
                return asOUT_OF_RANGE;
            }
        }

        if( *pFirst==targetvalue )
        {
            return 0; // Value corresponds
        } else {
            return asOUT_OF_RANGE;
        }
    }
}

bool asTools::SortedArrayInsert(int* pArrStart, int* pArrEnd, Order order, int val){
     return SortedArrayInsert<int>(pArrStart, pArrEnd, order, val);
}

bool asTools::SortedArrayInsert(float* pArrStart, float* pArrEnd, Order order, float val){
     return SortedArrayInsert<float>(pArrStart, pArrEnd, order, val);
}

bool asTools::SortedArrayInsert(double* pArrStart, double* pArrEnd, Order order, double val){
     return SortedArrayInsert<double>(pArrStart, pArrEnd, order, val);
}

template <class T>
bool asTools::SortedArrayInsert(T* pArrStart, T* pArrEnd, Order order, T val)
{
    wxASSERT(pArrStart);
    wxASSERT(pArrEnd);

    // Check the array length
    int vlength = pArrEnd-pArrStart;

    // Index where to insert the new element
    int i_next = 0;

    // Check order
    switch(order){
        case (Asc):
        {
            i_next = asTools::SortedArraySearchCeil(pArrStart, pArrEnd, val, asHIDE_WARNINGS);
            if (i_next==asOUT_OF_RANGE)
            {
                i_next = 0;
            }
            break;
        }
        case (Desc):
        {
            i_next = asTools::SortedArraySearchFloor(pArrStart, pArrEnd, val, asHIDE_WARNINGS);
            if (i_next==asOUT_OF_RANGE)
            {
                i_next = 0;
            }
            break;
        }
        case (NoOrder):
        {
            asLogError(_("Incorrect value of the order enumeration."));
            return false;
        }
    }

    // Swap next elements
    for (int i=vlength-1;i>=i_next;i--) // Minus 1 becuase we overwrite the last element by the previous one
    {
        pArrStart[i+1] = pArrStart[i];
    }

    // Insert new element
    pArrStart[i_next] = val;

    return true;
}

bool asTools::SortedArraysInsert(int* pArrRefStart, int* pArrRefEnd, int* pArrOtherStart, int* pArrOtherEnd, Order order, int valRef, int valOther){
     return SortedArraysInsert<int>(pArrRefStart, pArrRefEnd, pArrOtherStart, pArrOtherEnd, order, valRef, valOther);
}

bool asTools::SortedArraysInsert(float* pArrRefStart, float* pArrRefEnd, float* pArrOtherStart, float* pArrOtherEnd, Order order, float valRef, float valOther){
     return SortedArraysInsert<float>(pArrRefStart, pArrRefEnd, pArrOtherStart, pArrOtherEnd, order, valRef, valOther);
}

bool asTools::SortedArraysInsert(double* pArrRefStart, double* pArrRefEnd, double* pArrOtherStart, double* pArrOtherEnd, Order order, double valRef, double valOther){
     return SortedArraysInsert<double>(pArrRefStart, pArrRefEnd, pArrOtherStart, pArrOtherEnd, order, valRef, valOther);
}

template <class T>
bool asTools::SortedArraysInsert(T* pArrRefStart, T* pArrRefEnd, T* pArrOtherStart, T* pArrOtherEnd, Order order, T valRef, T valOther)
{
    wxASSERT(pArrRefStart);
    wxASSERT(pArrRefEnd);
    wxASSERT(pArrOtherStart);
    wxASSERT(pArrOtherEnd);

    // Check the other array length
    int vlength = pArrRefEnd-pArrRefStart;
    int ovlength = pArrOtherEnd-pArrOtherStart;

    if (vlength!=ovlength){
        asLogError(_("The dimension of the two arrays are not equal."));
        return false;
    } else if (vlength==0){
        asLogMessage(_("The array has an unique value."));
        return true;
    } else if (vlength<0) {
        asLogError(_("The array has a negative size..."));
        return false;
    }

    // Index where to insert the new element
    int i_next = 0;

    // Check order
    switch(order){
        case (Asc):
        {
            i_next = asTools::SortedArraySearchCeil(pArrRefStart, pArrRefEnd, valRef, asHIDE_WARNINGS);
            if (i_next==asOUT_OF_RANGE)
            {
                i_next = 0;
            }
            break;
        }
        case (Desc):
        {
            i_next = asTools::SortedArraySearchFloor(pArrRefStart, pArrRefEnd, valRef, asHIDE_WARNINGS);
            if (i_next==asOUT_OF_RANGE)
            {
                i_next = 0;
            }
            break;
        }
        case (NoOrder):
        {
            asLogError(_("Incorrect value of the order enumeration."));
            return false;
        }
    }

    // Swap next elements
    for (int i=vlength-1;i>=i_next;i--) // Minus 1 becuase we overwrite the last element by the previous one
    {
        pArrRefStart[i+1] = pArrRefStart[i];
        pArrOtherStart[i+1] = pArrOtherStart[i];
    }

    // Insert new element
    pArrRefStart[i_next] = valRef;
    pArrOtherStart[i_next] = valOther;

    return true;
}

bool asTools::SortArray(int* pArrRefStart, int* pArrRefEnd, Order order){
     return SortArrayT<int>(pArrRefStart, pArrRefEnd, order);
}

bool asTools::SortArray(float* pArrRefStart, float* pArrRefEnd, Order order){
     return SortArrayT<float>(pArrRefStart, pArrRefEnd, order);
}

bool asTools::SortArray(double* pArrRefStart, double* pArrRefEnd, Order order){
     return SortArrayT<double>(pArrRefStart, pArrRefEnd, order);
}

template <class T>
bool asTools::SortArrayT(T* pArrRefStart, T* pArrRefEnd, Order order)
{
    wxASSERT(pArrRefStart);
    wxASSERT(pArrRefEnd);

    // Check the array length
    int vlength = pArrRefEnd-pArrRefStart;

    if (vlength>0)
    {
        int    low = 0, high = vlength;
        asTools::QuickSort<T>(pArrRefStart,low,high,order);
    } else if (vlength==0){
        asLogMessage(_("The array has an unique value."));
        return true;
    } else {
        asLogError(_("The array has a negative size..."));
        return false;
    }
    return true;
}

bool asTools::SortArrays(int* pArrRefStart, int* pArrRefEnd, int* pArrOtherStart, int* pArrOtherEnd, Order order){
     return SortArraysT<int>(pArrRefStart, pArrRefEnd, pArrOtherStart, pArrOtherEnd, order);
}

bool asTools::SortArrays(float* pArrRefStart, float* pArrRefEnd, float* pArrOtherStart, float* pArrOtherEnd, Order order){
     return SortArraysT<float>(pArrRefStart, pArrRefEnd, pArrOtherStart, pArrOtherEnd, order);
}

bool asTools::SortArrays(double* pArrRefStart, double* pArrRefEnd, double* pArrOtherStart, double* pArrOtherEnd, Order order){
     return SortArraysT<double>(pArrRefStart, pArrRefEnd, pArrOtherStart, pArrOtherEnd, order);
}

template <class T>
bool asTools::SortArraysT(T* pArrRefStart, T* pArrRefEnd, T* pArrOtherStart, T* pArrOtherEnd, Order order)
{
    wxASSERT(pArrRefStart);
    wxASSERT(pArrRefEnd);
    wxASSERT(pArrOtherStart);
    wxASSERT(pArrOtherEnd);

    // Check the other array length
    int vlength = pArrRefEnd-pArrRefStart;
    int ovlength = pArrOtherEnd-pArrOtherStart;

    if (vlength>0 && vlength==ovlength)
    {
        int    low = 0, high = vlength;
        asTools::QuickSortMulti<T>(pArrRefStart,pArrOtherStart,low,high,order);
    } else if (vlength!=ovlength){
        asLogError(_("The dimension of the two arrays are not equal."));
        return false;
    } else if (vlength==0){
        asLogMessage(_("The array has an unique value."));
        return true;
    } else {
        asLogError(_("The array has a negative size..."));
        return false;
    }
    return true;
}

template <class T>
void asTools::QuickSort(T *pArr, int low, int high, Order order )
{
    int L, R;
    T pivot, tmp;

    R = high;
    L = low;

    pivot = pArr[((int) ((low+high) / 2))];

    do {

        switch(order){
            case (Asc):
            {
                while (pArr[L]<pivot) L++;
                while (pArr[R]>pivot) R--;
                break;
            }
            case (Desc):
            {
                while (pArr[L]>pivot) L++;
                while (pArr[R]<pivot) R--;
                break;
            }
            case (NoOrder):
            {
                asLogError(_("Incorrect value of the order enumeration."));
                break;
            }
        }

        if (R>=L) {
            if (R!=L) {
                tmp = pArr[R];
                pArr[R] = pArr[L];
                pArr[L] = tmp;
            }

            R--;
            L++;
        }
    } while (L <= R);

    if (low < R) asTools::QuickSort<T>(pArr,low,R,order);
    if (L < high) asTools::QuickSort<T>(pArr,L,high,order);

}

template <class T>
void asTools::QuickSortMulti(T *pArrRef, T *pArrOther, int low, int high, Order order )
{
    int L, R;
    T pivot, tmp;

    R = high;
    L = low;

    pivot = pArrRef[((int) ((low+high) / 2))];

    do {

        switch(order){
            case (Asc):
            {
                while (pArrRef[L]<pivot) L++;
                while (pArrRef[R]>pivot) R--;
                break;
            }
            case (Desc):
            {
                while (pArrRef[L]>pivot) L++;
                while (pArrRef[R]<pivot) R--;
                break;
            }
            case (NoOrder):
            {
                asLogError(_("Incorrect value of the order enumeration."));
                break;
            }
        }

        if (R>=L) {
            if (R!=L) {
                // Reference array
                tmp = pArrRef[R];
                pArrRef[R] = pArrRef[L];
                pArrRef[L] = tmp;
                // Other array
                tmp = pArrOther[R];
                pArrOther[R] = pArrOther[L];
                pArrOther[L] = tmp;
            }

            R--;
            L++;
        }
    } while (L <= R);

    if (low < R) asTools::QuickSortMulti<T>(pArrRef,pArrOther,low,R,order);
    if (L < high) asTools::QuickSortMulti<T>(pArrRef,pArrOther,L,high,order);

}

