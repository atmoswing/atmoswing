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
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#include "asUtilities.h"

bool asRemoveDir(const wxString &path)
{
    wxString f = wxFindFirstFile(path + DS + "*.*");
    while (!f.empty()) {
        wxRemoveFile(f);
        f = wxFindNextFile();
    }

    return wxRmdir(path);
}

void asInitRandom()
{
    srand(time(NULL));
}

int asRandom(const int min, const int max, const int step)
{
    // Initialize random seed
    double norm = ((double) rand() / (double) (RAND_MAX));

    auto dmin = (double) min;
    auto dmax = (double) max;
    auto dstep = (double) step;

    double range = 0;
    if (step < 1) {
        dstep = 1;
    }

    range = norm * (dmax + dstep -
                    dmin); // Add the step to account for cases at the limits, so they have the same probability.
    range = floor(range / dstep) * dstep;

    double val = range + dmin;
    int intval = asRound(val);

    if (intval > max)
        intval -= step;
    if (intval < min)
        intval += step;

    return intval;
}

float asRandom(const float min, const float max, const float step)
{
    return (float) asRandom((double) min, (double) max, (double) step);
}

double asRandom(const double min, const double max, const double step)
{
    // Initialize random seed
    double norm = ((double) rand() / (double) (RAND_MAX));

    double range = 0;
    if (step == 0) {
        range = norm * (max - min);
    } else {
        range = norm * (max + step -
                        min); // Add the step to account for cases at the limits, so they have the same probability.
        range = floor(range / step) * step; // -step/2.0 for cases at the limits.
    }

    double val = range + min;

    if (val > max)
        val -= step;
    if (val < min)
        val += step;

    return val;
}

int asRandomNormal(const int mean, const int stDev, const int step)
{
    return (int) asRandomNormal((double) mean, (double) stDev, (double) step);
}

float asRandomNormal(const float mean, const float stDev, const float step)
{
    return (float) asRandomNormal((double) mean, (double) stDev, (double) step);
}

double asRandomNormal(const double mean, const double stDev, const double step)
{
    // Initialize random seed
    double u1 = ((double) rand() / (double) (RAND_MAX));
    double u2 = ((double) rand() / (double) (RAND_MAX));

    // Exclude 0
    while (u1 == 0) {
        u1 = ((double) rand() / (double) (RAND_MAX));
    }
    while (u2 == 0) {
        u2 = ((double) rand() / (double) (RAND_MAX));
    }

    // Box-Muller transform
    double z0 = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
    //double z1 = sqrt(-2*log(u1))*sin(2*M_PI*u2);

    z0 *= stDev;

    if (step != 0) {
        z0 = step * asRound(z0 / step);
    }

    z0 += mean;

    return z0;
}

bool asIsRound(const float value)
{
    float valueround = asRound(value);

    return std::abs(value - valueround) < 0.000001;
}

bool asIsRound(const double value)
{
    double valueround = asRound(value);

    return std::abs(value - valueround) < 0.000000000001;
}

float asRound(const float value)
{
    if (value > 0) {
        return (float) floor(value + 0.5);
    } else {
        return (float) ceil(value - 0.5);
    }
}

double asRound(const double value)
{
    if (value > 0) {
        return floor(value + 0.5);
    } else {
        return ceil(value - 0.5);
    }
}

float asMean(const int *pArrStart, const int *pArrEnd)
{
    float sum = 0, nb = 0;
    for (int i = 0; i <= pArrEnd - pArrStart; i++) {
        // Does not check for NaNs, as there are no NaN for integers
        sum += (float) *(pArrStart + i);
        nb++;
    }
    return sum / nb;
}

float asMean(const float *pArrStart, const float *pArrEnd)
{
    float sum = 0, nb = 0;
    for (int i = 0; i <= pArrEnd - pArrStart; i++) {
        if (!asIsNaN(*(pArrStart + i))) {
            sum += *(pArrStart + i);
            nb++;
        }
    }
    return sum / nb;
}

double asMean(const double *pArrStart, const double *pArrEnd)
{
    double sum = 0, nb = 0;
    for (int i = 0; i <= pArrEnd - pArrStart; i++) {
        if (!asIsNaN(*(pArrStart + i))) {
            sum += *(pArrStart + i);
            nb++;
        }
    }
    return sum / nb;
}

float asStDev(const int *pArrStart, const int *pArrEnd, const int sample)
{
    float sum = 0, sumsquares = 0, nb = 0;
    for (int i = 0; i <= pArrEnd - pArrStart; i++) {
        // Dones't check for NaNs, as there are no NaN for intergers
        sum += *(pArrStart + i);
        sumsquares += (*(pArrStart + i)) * (*(pArrStart + i));
        nb++;
    }

    if (sample == asSAMPLE) {
        return sqrt((sumsquares - (sum * sum / nb)) / (nb - 1));
    } else if (sample == asENTIRE_POPULATION) {
        return sqrt((sumsquares - (sum * sum / nb)) / (nb));
    } else {
        return NaNf;
    }
}

float asStDev(const float *pArrStart, const float *pArrEnd, const int sample)
{
    float sum = 0, sumsquares = 0, nb = 0;
    for (int i = 0; i <= pArrEnd - pArrStart; i++) {
        if (!asIsNaN(*(pArrStart + i))) {
            sum += *(pArrStart + i);
            sumsquares += (*(pArrStart + i)) * (*(pArrStart + i));
            nb++;
        }
    }

    if (sample == asSAMPLE) {
        return sqrt((sumsquares - (sum * sum / nb)) / (nb - 1));
    } else if (sample == asENTIRE_POPULATION) {
        return sqrt((sumsquares - (sum * sum / nb)) / (nb));
    } else {
        return NaNf;
    }
}

double asStDev(const double *pArrStart, const double *pArrEnd, const int sample)
{
    double sum = 0, sumsquares = 0, nb = 0;
    for (int i = 0; i <= pArrEnd - pArrStart; i++) {
        if (!asIsNaN(*(pArrStart + i))) {
            sum += *(pArrStart + i);
            sumsquares += (*(pArrStart + i)) * (*(pArrStart + i));
            nb++;
        }
    }

    if (sample == asSAMPLE) {
        return sqrt((sumsquares - (sum * sum / nb)) / (nb - 1));
    } else if (sample == asENTIRE_POPULATION) {
        return sqrt((sumsquares - (sum * sum / nb)) / (nb));
    } else {
        return NaNd;
    }
}

a1f asGetCumulativeFrequency(const int size)
{
    a1f F(size);

    // Parameters for the estimated distribution from Gringorten (a=0.44, b=0.12).
    // Choice based on [Cunnane, C., 1978, Unbiased plotting positions—A review: Journal of Hydrology, v. 37, p. 205–222.]
    // Bontron used a=0.375, b=0.25, that are optimal for a normal distribution
    float irep = 0.44f;
    float nrep = 0.12f;

    // Change the values for unit testing to compare to the results from Grenoble
    if (g_unitTesting) {
        irep = 0.375;
        nrep = 0.25;
    }

    float divisor = 1.0f / (size + nrep);
    for (int i = 0; i < size; i++) {
        F(i) = ((float)i + 1.0f - irep) * divisor; // i+1 as i starts from 0
    }

    return F;
}

float asGetValueForQuantile(const a1f &values, const float quantile)
{
    float value = NaNf;
    int size = values.size();

    a1f valuesCopy = values;

    // Sort the forcast array
    asSortArray(&valuesCopy[0], &valuesCopy[size - 1], Asc);

    // Cumulative frequency
    a1f F = asGetCumulativeFrequency(size);

    // Check limits
    if (quantile <= F[0])
        return valuesCopy[0];
    if (quantile >= F[size - 1])
        return valuesCopy[size - 1];

    // Indices for the left and right part (according to xObs)
    int indLeft = asFindFloor(&F[0], &F[size - 1], quantile);
    int indRight = asFindCeil(&F[0], &F[size - 1], quantile);
    wxASSERT(indLeft >= 0);
    wxASSERT(indRight >= 0);
    wxASSERT(indLeft <= indRight);

    if (indLeft < 0 || indRight < 0) {
        wxLogError(_("An unexpected error occurred."));
        return NaNf;
    }

    if (indLeft == indRight) {
        value = valuesCopy[indLeft];
    } else {
        value = valuesCopy(indLeft) +
                (valuesCopy(indRight) - valuesCopy(indLeft)) * (quantile - F(indLeft)) / (F(indRight) - F(indLeft));
    }

    return value;
}

bool asIsNaN(const int value)
{
    return value == NaNi;
}

bool asIsNaN(const float value)
{
    return value != value;
}

bool asIsNaN(const double value)
{
    return value != value;
}

bool asIsInf(const float value)
{
    return value == Inff;
}

bool asIsInf(const double value)
{
    return value == Infd;
}

bool asIsInf(const long double value)
{
    return value == Infld;
}

int asCountNotNaN(const float *pArrStart, const float *pArrEnd)
{
    int counter = 0;

    for (int i = 0; i <= pArrEnd - pArrStart; i++) {
        if (!asIsNaN(*(pArrStart + i))) {
            counter++;
        }
    }

    return counter;
}

int asCountNotNaN(const double *pArrStart, const double *pArrEnd)
{
    int counter = 0;

    for (int i = 0; i <= pArrEnd - pArrStart; i++) {
        if (!asIsNaN(*(pArrStart + i))) {
            counter++;
        }
    }

    return counter;
}

bool asHasNaN(const a2f &data)
{
    return !((data == data)).all();
}

bool asHasNaN(const float *pArrStart, const float *pArrEnd)
{
    for (int i = 0; i <= pArrEnd - pArrStart; i++) {
        if (asIsNaN(*(pArrStart + i))) {
            return true;
        }
    }

    return false;
}

bool asHasNaN(const double *pArrStart, const double *pArrEnd)
{
    for (int i = 0; i <= pArrEnd - pArrStart; i++) {
        if (asIsNaN(*(pArrStart + i))) {
            return true;
        }
    }

    return false;
}

int asMinArray(const int *pArrStart, const int *pArrEnd)
{
    int min;

    min = *(pArrStart);

    for (int i = 0; i <= pArrEnd - pArrStart; i++) {
        if (*(pArrStart + i) < min) {
            min = *(pArrStart + i);
        }
    }

    return min;
}

float asMinArray(const float *pArrStart, const float *pArrEnd)
{
    float min;
    int i = 0;

    // Manage the case where the first elements are NaNs
    while (asIsNaN(*(pArrStart + i))) {
        i++;
    }
    min = *(pArrStart + i);

    for (; i <= pArrEnd - pArrStart; i++) {
        if (!asIsNaN(*(pArrStart + i))) {
            if (*(pArrStart + i) < min) {
                min = *(pArrStart + i);
            }
        }
    }

    return min;
}

double asMinArray(const double *pArrStart, const double *pArrEnd)
{
    double min;
    int i = 0;

    // Manage the case where the first elements are NaNs
    while (asIsNaN(*(pArrStart + i))) {
        i++;
    }
    min = *(pArrStart + i);

    for (; i <= pArrEnd - pArrStart; i++) {
        if (!asIsNaN(*(pArrStart + i))) {
            if (*(pArrStart + i) < min) {
                min = *(pArrStart + i);
            }
        }
    }

    return min;
}

int asMinArrayIndex(const int *pArrStart, const int *pArrEnd)
{
    int min;
    int index;
    int i = 0;

    min = *(pArrStart + i);
    index = 0;

    for (; i <= pArrEnd - pArrStart; i++) {
        if (*(pArrStart + i) < min) {
            min = *(pArrStart + i);
            index = i;
        }
    }

    return index;
}

int asMinArrayIndex(const float *pArrStart, const float *pArrEnd)
{
    float min;
    int index;
    int i = 0;

    min = *(pArrStart + i);
    index = 0;

    for (; i <= pArrEnd - pArrStart; i++) {
        if (*(pArrStart + i) < min) {
            min = *(pArrStart + i);
            index = i;
        }
    }

    return index;
}

int asMinArrayIndex(const double *pArrStart, const double *pArrEnd)
{
    double min;
    int index;
    int i = 0;

    min = *(pArrStart + i);
    index = 0;

    for (; i <= pArrEnd - pArrStart; i++) {
        if (*(pArrStart + i) < min) {
            min = *(pArrStart + i);
            index = i;
        }
    }

    return index;
}

int asMaxArray(const int *pArrStart, const int *pArrEnd)
{
    int max;
    int i = 0;

    max = *(pArrStart + i);

    for (; i <= pArrEnd - pArrStart; i++) {
        if (*(pArrStart + i) > max) {
            max = *(pArrStart + i);
        }
    }

    return max;
}

float asMaxArray(const float *pArrStart, const float *pArrEnd)
{
    float max;
    int i = 0;

    // Manage the case where the first elements are NaNs
    while (asIsNaN(*(pArrStart + i))) {
        i++;
    }
    max = *(pArrStart + i);

    for (; i <= pArrEnd - pArrStart; i++) {
        if (!asIsNaN(*(pArrStart + i))) {
            if (*(pArrStart + i) > max) {
                max = *(pArrStart + i);
            }
        }
    }

    return max;
}

double asMaxArray(const double *pArrStart, const double *pArrEnd)
{
    double max;
    int i = 0;

    // Manage the case where the first elements are NaNs
    while (asIsNaN(*(pArrStart + i))) {
        i++;
    }
    max = *(pArrStart + i);

    for (; i <= pArrEnd - pArrStart; i++) {
        if (!asIsNaN(*(pArrStart + i))) {
            if (*(pArrStart + i) > max) {
                max = *(pArrStart + i);
            }
        }
    }

    return max;
}

int asMaxArrayIndex(const int *pArrStart, const int *pArrEnd)
{
    int max;
    int index;
    int i = 0;

    max = *(pArrStart + i);
    index = 0;

    for (; i <= pArrEnd - pArrStart; i++) {
        if (*(pArrStart + i) > max) {
            max = *(pArrStart + i);
            index = i;
        }
    }

    return index;
}

int asMaxArrayIndex(const float *pArrStart, const float *pArrEnd)
{
    float max;
    int index;
    int i = 0;

    max = *(pArrStart + i);
    index = 0;

    for (; i <= pArrEnd - pArrStart; i++) {
        if (*(pArrStart + i) > max) {
            max = *(pArrStart + i);
            index = i;
        }
    }

    return index;
}

int asMaxArrayIndex(const double *pArrStart, const double *pArrEnd)
{
    double max;
    int index;
    int i = 0;

    max = *(pArrStart + i);
    index = 0;

    for (; i <= pArrEnd - pArrStart; i++) {
        if (*(pArrStart + i) > max) {
            max = *(pArrStart + i);
            index = i;
        }
    }

    return index;
}

int asMinArrayStep(const int *pArrStart, const int *pArrEnd, const int tolerance)
{
    // Copy data to not alter original array
    a1i copyData(pArrEnd - pArrStart + 1);

    for (int i = 0; i <= pArrEnd - pArrStart; i++) {
        copyData[i] = *(pArrStart + i);
    }

    // Sort the array
    asSortArray(&copyData[0], &copyData[copyData.size() - 1], Asc);

    // Find min step
    int i = 1;

    while (copyData[i] - copyData[i - 1] <= tolerance) {
        i++;
        if (i == copyData.size()) {
            return asNOT_FOUND;
        }
    }

    int minstep = copyData[i] - copyData[i - 1];

    for (; i < copyData.size(); i++) {
        int currentval = std::abs(copyData[i] - copyData[i - 1]);
        if ((currentval < minstep) & (currentval > tolerance)) {
            minstep = currentval;
        }
    }

    return minstep;
}

float asMinArrayStep(const float *pArrStart, const float *pArrEnd, const float tolerance)
{
    int nbNotNans = asCountNotNaN(pArrStart, pArrEnd);
    int j = 0;

    // Copy data to not alter original array
    a1f copyData(nbNotNans);

    // Remove Nans
    for (int i = 0; i <= pArrEnd - pArrStart; i++) {
        if (!asIsNaN(*(pArrStart + i))) {
            copyData[j] = *(pArrStart + i);
            j++;
        }
    }

    // Sort the array
    asSortArray(&copyData[0], &copyData[copyData.size() - 1], Asc);

    // Find min step
    int i = 1;

    while (copyData[i] - copyData[i - 1] <= tolerance) {
        i++;
        if (i == copyData.size()) {
            return asNOT_FOUND;
        }
    }

    float minstep = copyData[i] - copyData[i - 1];

    for (; i < copyData.size(); i++) {
        float currentval = std::abs(copyData[i] - copyData[i - 1]);
        if ((currentval < minstep) & (currentval > tolerance)) {
            minstep = currentval;
        }
    }

    return minstep;
}

double asMinArrayStep(const double *pArrStart, const double *pArrEnd, const double tolerance)
{
    int nbNotNans = asCountNotNaN(pArrStart, pArrEnd);
    int j = 0;

    // Copy data to not alter original array
    a1d copyData(nbNotNans);

    // Remove Nans
    for (int i = 0; i <= pArrEnd - pArrStart; i++) {
        if (!asIsNaN(*(pArrStart + i))) {
            copyData[j] = *(pArrStart + i);
            j++;
        }
    }

    // Sort the array
    asSortArray(&copyData[0], &copyData[copyData.size() - 1], Asc);

    // Find min step
    int i = 1;

    while (copyData[i] - copyData[i - 1] <= tolerance) {
        i++;
        if (i == copyData.size()) {
            return asNOT_FOUND;
        }
    }

    double minstep = copyData[i] - copyData[i - 1];

    for (; i < copyData.size(); i++) {
        double currentval = std::abs(copyData[i] - copyData[i - 1]);
        if ((currentval < minstep) & (currentval > tolerance)) {
            minstep = currentval;
        }
    }

    return minstep;
}

a1i asExtractUniqueValues(const int *pArrStart, const int *pArrEnd, const int tolerance)
{
    int j = 0;

    // Copy data to not alter original array
    vi copyData(pArrEnd - pArrStart + 1);

    // Remove Nans
    for (int i = 0; i <= pArrEnd - pArrStart; i++) {
        copyData[j] = *(pArrStart + i);
        j++;
    }

    // Sort the array
    asSortArray(&copyData[0], &copyData[copyData.size() - 1], Asc);

    // Extract unique values
    vi copyDataUniques;
    copyDataUniques.reserve(pArrEnd - pArrStart + 1);
    copyDataUniques.push_back(copyData[0]); // Add first value

    for (unsigned int i = 1; i < copyData.size(); i++) {
        if ((std::abs(copyData[i] - copyData[i - 1]) > tolerance)) {
            copyDataUniques.push_back(copyData[i]);
        }
    }

    // Copy data to the final container
    a1i resultArray(copyDataUniques.size());

    for (unsigned int i = 0; i < copyDataUniques.size(); i++) {
        resultArray[i] = copyDataUniques[i];
    }

    return resultArray;
}

a1f asExtractUniqueValues(const float *pArrStart, const float *pArrEnd, const float tolerance)
{
    auto nbNotNans = (unsigned int) asCountNotNaN(pArrStart, pArrEnd);
    int j = 0;

    // Copy data to not alter original array
    vf copyData(nbNotNans);

    // Remove Nans
    for (int i = 0; i <= pArrEnd - pArrStart; i++) {
        if (!asIsNaN(*(pArrStart + i))) {
            copyData[j] = *(pArrStart + i);
            j++;
        }
    }

    // Sort the array
    asSortArray(&copyData[0], &copyData[copyData.size() - 1], Asc);

    // Extract unique values
    vf copyDataUniques;
    copyDataUniques.reserve(nbNotNans);
    copyDataUniques.push_back(copyData[0]); // Add first value

    for (unsigned int i = 1; i < copyData.size(); i++) {
        if ((std::abs(copyData[i] - copyData[i - 1]) > tolerance)) {
            copyDataUniques.push_back(copyData[i]);
        }
    }

    // Copy data to the final container
    a1f resultArray(copyDataUniques.size());

    for (unsigned int i = 0; i < copyDataUniques.size(); i++) {
        resultArray[i] = copyDataUniques[i];
    }

    return resultArray;
}

a1d asExtractUniqueValues(const double *pArrStart, const double *pArrEnd, const double tolerance)
{
    auto nbNotNans = (unsigned int) asCountNotNaN(pArrStart, pArrEnd);
    int j = 0;

    // Copy data to not alter original array
    vd copyData = vd(nbNotNans);

    // Remove Nans
    for (int i = 0; i <= pArrEnd - pArrStart; i++) {
        if (!asIsNaN(*(pArrStart + i))) {
            copyData[j] = *(pArrStart + i);
            j++;
        }
    }

    // Sort the array
    asSortArray(&copyData[0], &copyData[copyData.size() - 1], Asc);

    // Extract unique values
    vd copyDataUniques;
    copyDataUniques.reserve(nbNotNans);
    copyDataUniques.push_back(copyData[0]); // Add first value

    for (unsigned int i = 1; i < copyData.size(); i++) {
        if ((std::abs(copyData[i] - copyData[i - 1]) > tolerance)) {
            copyDataUniques.push_back(copyData[i]);
        }
    }

    // Copy data to the final container
    a1d resultArray(copyDataUniques.size());

    for (unsigned int i = 0; i < copyDataUniques.size(); i++) {
        resultArray[i] = copyDataUniques[i];
    }

    return resultArray;
}

int asFind(const int *pArrStart, const int *pArrEnd, const int targetValue, const int tolerance, const int showWarning)
{
    return asFindT<int>(pArrStart, pArrEnd, targetValue, tolerance, showWarning);
}

int asFind(const float *pArrStart, const float *pArrEnd, const float targetValue, const float tolerance,
           const int showWarning)
{
    return asFindT<float>(pArrStart, pArrEnd, targetValue, tolerance, showWarning);
}

int asFind(const double *pArrStart, const double *pArrEnd, const double targetValue, const double tolerance,
           int showWarning)
{
    return asFindT<double>(pArrStart, pArrEnd, targetValue, tolerance, showWarning);
}

template<class T>
int asFindT(const T *pArrStart, const T *pArrEnd, const T targetValue, const T tolerance, const int showWarning)
{
    wxASSERT(pArrStart);
    wxASSERT(pArrEnd);

    T *pFirst = nullptr, *pMid = nullptr, *pLast = nullptr;
    int vlength;

    // Initialize first and last variables.
    pFirst = (T *) pArrStart;
    pLast = (T *) pArrEnd;

    // Check array order
    if (*pLast > *pFirst) {
        // Binary search
        while (pFirst <= pLast) {
            vlength = (int) (pLast - pFirst);
            pMid = pFirst + vlength / 2;
            if (targetValue - tolerance > *pMid) {
                pFirst = pMid + 1;
            } else if (targetValue + tolerance < *pMid) {
                pLast = pMid - 1;
            } else {
                // Return found index
                return static_cast<int>(pMid - pArrStart);
            }
        }

        // Check the pointers
        if (pLast - pArrStart < 0) {
            pLast = (T *) pArrStart;
        } else if (pLast - pArrEnd > 0) {
            pLast = (T *) pArrEnd - 1;
        } else if (pLast - pArrEnd == 0) {
            pLast -= 1;
        }

        // If the value was not found, return closest value inside tolerance
        if (std::abs(targetValue - *pLast) <= std::abs(targetValue - *(pLast + 1))) {
            if (std::abs(targetValue - *pLast) <= tolerance) {
                return static_cast<int>(pLast - pArrStart);
            } else {
                // Check that the value is within the array. Do it here to allow a margin for the tolerance
                if (targetValue > *pArrEnd || targetValue < *pArrStart) {
                    if (showWarning == asSHOW_WARNINGS) {
                        wxLogWarning(_("The value (%f) is out of the array range."), static_cast<float>(targetValue));
                    }
                    return asOUT_OF_RANGE;
                }
                if (showWarning == asSHOW_WARNINGS) {
                    wxLogWarning(_("The value was not found in the array."));
                }
                return asNOT_FOUND;
            }
        } else {
            if (std::abs(targetValue - *(pLast + 1)) <= tolerance) {
                return static_cast<int>(pLast - pArrStart + 1);
            } else {
                // Check that the value is whithin the array. Do it here to allow a margin for the tolerance
                if (targetValue > *pArrEnd || targetValue < *pArrStart) {
                    if (showWarning == asSHOW_WARNINGS) {
                        wxLogWarning(_("The value (%f) is out of the array range."), static_cast<float>(targetValue));
                    }
                    return asOUT_OF_RANGE;
                }
                if (showWarning == asSHOW_WARNINGS) {
                    wxLogWarning(_("The value was not found in the array."));
                }
                return asNOT_FOUND;
            }
        }
    } else if (*pLast < *pFirst) {
        // Binary search
        while (pFirst <= pLast) {
            vlength = static_cast<int>(pLast - pFirst);
            pMid = pFirst + vlength / 2;
            if (targetValue - tolerance > *pMid) {
                pLast = pMid - 1;
            } else if (targetValue + tolerance < *pMid) {
                pFirst = pMid + 1;
            } else {
                // Return found index
                return static_cast<int>(pMid - pArrStart);
            }
        }

        // Check the pointers
        if (pFirst - pArrStart < 0) {
            pFirst = (T *) pArrStart + 1;
        } else if (pFirst - pArrEnd > 0) {
            pFirst = (T *) pArrEnd;
        } else if (pFirst - pArrStart == 0) {
            pFirst += 1;
        }

        // If the value was not found, return closest value inside tolerance
        if (std::abs(targetValue - *pFirst) <= std::abs(targetValue - *(pFirst - 1))) {
            if (std::abs(targetValue - *pFirst) <= tolerance) {
                return static_cast<int>(pFirst - pArrStart);
            } else {
                // Check that the value is whithin the array. Do it here to allow a margin for the tolerance.
                if (targetValue < *pArrEnd || targetValue > *pArrStart) {
                    if (showWarning == asSHOW_WARNINGS) {
                        wxLogWarning(_("The value (%f) is out of the array range."), static_cast<float>(targetValue));
                    }
                    return asOUT_OF_RANGE;
                }
                if (showWarning == asSHOW_WARNINGS) {
                    wxLogWarning(_("The value was not found in the array."));
                }
                return asNOT_FOUND;
            }
        } else {
            if (std::abs(targetValue - *(pFirst - 1)) <= tolerance) {
                return static_cast<int>(pFirst - pArrStart - 1);
            } else {
                // Check that the value is whithin the array. Do it here to allow a margin for the tolerance.
                if (targetValue < *pArrEnd || targetValue > *pArrStart) {
                    if (showWarning == asSHOW_WARNINGS) {
                        wxLogWarning(_("The value (%f) is out of the array range."), static_cast<float>(targetValue));
                    }
                    return asOUT_OF_RANGE;
                }
                if (showWarning == asSHOW_WARNINGS) {
                    wxLogWarning(_("The value was not found in the array."));
                }
                return asNOT_FOUND;
            }
        }
    } else {
        if (pLast - pFirst == 0) {
            if (*pFirst >= targetValue - tolerance && *pFirst <= targetValue + tolerance) {
                return 0; // Value corresponds
            } else {
                return asOUT_OF_RANGE;
            }
        }

        if (*pFirst >= targetValue - tolerance && *pFirst <= targetValue + tolerance) {
            return 0; // Value corresponds
        } else {
            return asOUT_OF_RANGE;
        }
    }
}

int asFindClosest(const int *pArrStart, const int *pArrEnd, const int targetValue, const int showWarning)
{
    return asFindClosestT<int>(pArrStart, pArrEnd, targetValue, showWarning);
}

int asFindClosest(const float *pArrStart, const float *pArrEnd, const float targetValue, const int showWarning)
{
    return asFindClosestT<float>(pArrStart, pArrEnd, targetValue, showWarning);
}

int asFindClosest(const double *pArrStart, const double *pArrEnd, const double targetValue, const int showWarning)
{
    return asFindClosestT<double>(pArrStart, pArrEnd, targetValue, showWarning);
}

template<class T>
int asFindClosestT(const T *pArrStart, const T *pArrEnd, const T targetValue, const int showWarning)
{
    wxASSERT(pArrStart);
    wxASSERT(pArrEnd);

    T *pFirst = nullptr, *pMid = nullptr, *pLast = nullptr;
    int vlength;

    // Initialize first and last variables.
    pFirst = (T *) pArrStart;
    pLast = (T *) pArrEnd;

    // Check array order
    if (*pLast > *pFirst) {
        // Check that the value is whithin the array
        if (targetValue > *pLast || targetValue < *pFirst) {
            if (showWarning == asSHOW_WARNINGS) {
                wxLogWarning(_("The value (%f) is out of the array range."), static_cast<float>(targetValue));
            }
            return asOUT_OF_RANGE;
        }

        // Binary search
        while (pFirst <= pLast) {
            vlength = (int) (pLast - pFirst);
            pMid = pFirst + vlength / 2;
            if (targetValue > *pMid) {
                pFirst = pMid + 1;
            } else if (targetValue < *pMid) {
                pLast = pMid - 1;
            } else {
                // Return found index
                return (int) (pMid - pArrStart);
            }
        }

        // Check the pointers
        if (pLast - pArrStart < 0) {
            pLast = (T *) pArrStart;
        } else if (pLast - pArrEnd > 0) {
            pLast = (T *) pArrEnd - 1;
        } else if (pLast - pArrEnd == 0) {
            pLast -= 1;
        }

        // If the value was not found, return closest value
        if (std::abs(targetValue - *pLast) <= std::abs(targetValue - *(pLast + 1))) {
            return static_cast<int>(pLast - pArrStart);
        } else {
            return static_cast<int>(pLast - pArrStart + 1);
        }
    } else if (*pLast < *pFirst) {
        // Check that the value is whithin the array
        if (targetValue < *pLast || targetValue > *pFirst) {
            if (showWarning == asSHOW_WARNINGS) {
                wxLogWarning(_("The value (%f) is out of the array range."), static_cast<float>(targetValue));
            }
            return asOUT_OF_RANGE;
        }

        // Binary search
        while (pFirst <= pLast) {
            vlength = static_cast<int>(pLast - pFirst);
            pMid = pFirst + vlength / 2;
            if (targetValue > *pMid) {
                pLast = pMid - 1;
            } else if (targetValue < *pMid) {
                pFirst = pMid + 1;
            } else {
                // Return found index
                return static_cast<int>(pMid - pArrStart);
            }
        }

        // Check the pointers
        if (pFirst - pArrStart < 0) {
            pFirst = (T *) pArrStart + 1;
        } else if (pFirst - pArrEnd > 0) {
            pFirst = (T *) pArrEnd;
        } else if (pFirst - pArrStart == 0) {
            pFirst += 1;
        }

        // If the value was not found, return closest value
        if (std::abs(targetValue - *pFirst) <= std::abs(targetValue - *(pFirst - 1))) {
            return static_cast<int>(pFirst - pArrStart);
        } else {
            return static_cast<int>(pFirst - pArrStart - 1);
        }
    } else {
        if (pLast - pFirst == 0) {
            if (*pFirst == targetValue) {
                return 0; // Value corresponds
            } else {
                return asOUT_OF_RANGE;
            }
        }

        if (*pFirst == targetValue) {
            return 0; // Value corresponds
        } else {
            return asOUT_OF_RANGE;
        }
    }
}

int asFindFloor(const int *pArrStart, const int *pArrEnd, const int targetValue, const int showWarning)
{
    return asFindFloorT<int>(pArrStart, pArrEnd, targetValue, showWarning);
}

int asFindFloor(const float *pArrStart, const float *pArrEnd, const float targetValue, const int showWarning)
{
    return asFindFloorT<float>(pArrStart, pArrEnd, targetValue, showWarning);
}

int asFindFloor(const double *pArrStart, const double *pArrEnd, const double targetValue, const int showWarning)
{
    return asFindFloorT<double>(pArrStart, pArrEnd, targetValue, showWarning);
}

template<class T>
int asFindFloorT(const T *pArrStart, const T *pArrEnd, const T targetValue, const int showWarning)
{
    wxASSERT(pArrStart);
    wxASSERT(pArrEnd);

    T *pFirst = nullptr, *pMid = nullptr, *pLast = nullptr;
    int vlength;

    // Initialize first and last variables.
    pFirst = (T *) pArrStart;
    pLast = (T *) pArrEnd;

    double tolerance = 0;
    /*
    if (*pFirst != *pLast) {
        tolerance = (double) std::abs(*pFirst - *(pFirst + 1)) / 100.0;
    }*/

    // Check array order
    if (*pLast > *pFirst) {
        // Check that the value is within the array
        if (targetValue - tolerance > *pLast || targetValue + tolerance < *pFirst) {
            if (showWarning == asSHOW_WARNINGS) {
                wxLogWarning(_("The value (%f) is out of the array range."), static_cast<float>(targetValue));
            }
            return asOUT_OF_RANGE;
        }

        // Binary search
        while (pFirst <= pLast) {
            vlength = (int) (pLast - pFirst);
            pMid = pFirst + vlength / 2;
            if (targetValue - tolerance > *pMid) {
                pFirst = pMid + 1;
            } else if (targetValue + tolerance < *pMid) {
                pLast = pMid - 1;
            } else {
                // Return found index
                return static_cast<int>(pMid - pArrStart);
            }
        }

        // Check the pointers
        if (pLast - pArrStart < 0) {
            pLast = (T *) pArrStart;
        } else if (pLast - pArrEnd > 0) {
            pLast = (T *) pArrEnd;
        }

        // If the value was not found, return floor value
        return static_cast<int>(pLast - pArrStart);
    } else if (*pLast < *pFirst) {
        // Check that the value is within the array
        if (targetValue + tolerance < *pLast || targetValue - tolerance > *pFirst) {
            if (showWarning == asSHOW_WARNINGS) {
                wxLogWarning(_("The value (%f) is out of the array range."), static_cast<float>(targetValue));
            }
            return asOUT_OF_RANGE;
        }

        // Binary search
        while (pFirst <= pLast) {
            vlength = static_cast<int>(pLast - pFirst);
            pMid = pFirst + vlength / 2;
            if (targetValue - tolerance > *pMid) {
                pLast = pMid - 1;
            } else if (targetValue + tolerance < *pMid) {
                pFirst = pMid + 1;
            } else {
                // Return found index
                return static_cast<int>(pMid - pArrStart);
            }
        }

        // Check the pointers
        if (pFirst - pArrStart < 0) {
            pFirst = (T *) pArrStart;
        } else if (pFirst - pArrEnd > 0) {
            pFirst = (T *) pArrEnd;
        }

        // If the value was not found, return floor value
        return static_cast<int>(pFirst - pArrStart);
    } else {
        if (pLast - pFirst == 0) {
            if (std::abs(*pFirst - targetValue) <= tolerance) {
                return 0; // Value corresponds
            } else {
                return asOUT_OF_RANGE;
            }
        }

        if (std::abs(*pFirst - targetValue) <= tolerance) {
            return 0; // Value corresponds
        } else {
            return asOUT_OF_RANGE;
        }
    }
}


int asFindCeil(const int *pArrStart, const int *pArrEnd, const int targetValue, const int showWarning)
{
    return asFindCeilT<int>(pArrStart, pArrEnd, targetValue, showWarning);
}

int asFindCeil(const float *pArrStart, const float *pArrEnd, const float targetValue, const int showWarning)
{
    return asFindCeilT<float>(pArrStart, pArrEnd, targetValue, showWarning);
}

int asFindCeil(const double *pArrStart, const double *pArrEnd, const double targetValue, const int showWarning)
{
    return asFindCeilT<double>(pArrStart, pArrEnd, targetValue, showWarning);
}

template<class T>
int asFindCeilT(const T *pArrStart, const T *pArrEnd, const T targetValue, const int showWarning)
{
    wxASSERT(pArrStart);
    wxASSERT(pArrEnd);

    T *pFirst = nullptr, *pMid = nullptr, *pLast = nullptr;
    int vlength;

    // Initialize first and last variables.
    pFirst = (T *) pArrStart;
    pLast = (T *) pArrEnd;

    double tolerance = 0;
    /*
    if (*pFirst != *pLast) {
        tolerance = (double) std::abs(*pFirst - *(pFirst + 1)) / 100.0;
    }*/

    // Check array order
    if (*pLast > *pFirst) {
        // Check that the value is within the array
        if (targetValue - tolerance > *pLast || targetValue + tolerance < *pFirst) {
            if (showWarning == asSHOW_WARNINGS) {
                wxLogWarning(_("The value (%f) is out of the array range."), static_cast<float>(targetValue));
            }
            return asOUT_OF_RANGE;
        }

        // Binary search
        while (pFirst <= pLast) {
            vlength = (int) (pLast - pFirst);
            pMid = pFirst + vlength / 2;
            if (targetValue - tolerance > *pMid) {
                pFirst = pMid + 1;
            } else if (targetValue + tolerance < *pMid) {
                pLast = pMid - 1;
            } else {
                // Return found index
                return static_cast<int>(pMid - pArrStart);
            }
        }

        // Check the pointers
        if (pLast - pArrStart < 0) {
            pLast = (T *) pArrStart;
        } else if (pLast - pArrEnd > 0) {
            pLast = (T *) pArrEnd;
        }

        // If the value was not found, return ceil value
        return static_cast<int>(pLast - pArrStart + 1);
    } else if (*pLast < *pFirst) {
        // Check that the value is within the array
        if (targetValue + tolerance < *pLast || targetValue - tolerance > *pFirst) {
            if (showWarning == asSHOW_WARNINGS) {
                wxLogWarning(_("The value (%f) is out of the array range."), static_cast<float>(targetValue));
            }
            return asOUT_OF_RANGE;
        }

        // Binary search
        while (pFirst <= pLast) {
            vlength = (int) (pLast - pFirst);
            pMid = pFirst + vlength / 2;
            if (targetValue - tolerance > *pMid) {
                pLast = pMid - 1;
            } else if (targetValue + tolerance < *pMid) {
                pFirst = pMid + 1;
            } else {
                // Return found index
                return static_cast<int>(pMid - pArrStart);
            }
        }

        // Check the pointers
        if (pFirst - pArrStart < 0) {
            pFirst = (T *) pArrStart;
        } else if (pFirst - pArrEnd > 0) {
            pFirst = (T *) pArrEnd;
        }

        // If the value was not found, return ceil value
        return static_cast<int>(pFirst - pArrStart - 1);
    } else {
        if (pLast - pFirst == 0) {
            if (std::abs(*pFirst - targetValue) <= tolerance) {
                return 0; // Value corresponds
            } else {
                return asOUT_OF_RANGE;
            }
        }

        if (std::abs(*pFirst - targetValue) <= tolerance) {
            return 0; // Value corresponds
        } else {
            return asOUT_OF_RANGE;
        }
    }
}

bool asArrayInsert(int *pArrStart, int *pArrEnd, const Order order, const int val)
{
    return asArrayInsertT<int>(pArrStart, pArrEnd, order, val);
}

bool asArrayInsert(float *pArrStart, float *pArrEnd, const Order order, const float val)
{
    return asArrayInsertT<float>(pArrStart, pArrEnd, order, val);
}

bool asArrayInsert(double *pArrStart, double *pArrEnd, const Order order, const double val)
{
    return asArrayInsertT<double>(pArrStart, pArrEnd, order, val);
}

template<class T>
bool asArrayInsertT(T *pArrStart, T *pArrEnd, const Order order, const T val)
{
    wxASSERT(pArrStart);
    wxASSERT(pArrEnd);

    // Check the array length
    int vlength = pArrEnd - pArrStart;

    // Index where to insert the new element
    int iNext = 0;

    // Check order
    switch (order) {
        case (Asc): {
            iNext = asFindCeil(pArrStart, pArrEnd, val, asHIDE_WARNINGS);
            if (iNext == asOUT_OF_RANGE) {
                iNext = 0;
            }
            break;
        }
        case (Desc): {
            iNext = asFindFloor(pArrStart, pArrEnd, val, asHIDE_WARNINGS);
            if (iNext == asOUT_OF_RANGE) {
                iNext = 0;
            }
            break;
        }
        default: {
            wxLogError(_("Incorrect value of the order enumeration."));
            return false;
        }
    }

    // Swap next elements
    for (int i = vlength - 1; i >= iNext; i--) // Minus 1 becuase we overwrite the last element by the previous one
    {
        pArrStart[i + 1] = pArrStart[i];
    }

    // Insert new element
    pArrStart[iNext] = val;

    return true;
}

bool asArraysInsert(int *pArrRefStart, int *pArrRefEnd, int *pArrOtherStart, int *pArrOtherEnd, const Order order,
                    const int valRef, const int valOther)
{
    return asArraysInsertT<int>(pArrRefStart, pArrRefEnd, pArrOtherStart, pArrOtherEnd, order, valRef, valOther);
}

bool asArraysInsert(float *pArrRefStart, float *pArrRefEnd, float *pArrOtherStart, float *pArrOtherEnd,
                    const Order order, const float valRef, const float valOther)
{
    return asArraysInsertT<float>(pArrRefStart, pArrRefEnd, pArrOtherStart, pArrOtherEnd, order, valRef, valOther);
}

bool asArraysInsert(double *pArrRefStart, double *pArrRefEnd, double *pArrOtherStart, double *pArrOtherEnd,
                    const Order order, const double valRef, const double valOther)
{
    return asArraysInsertT<double>(pArrRefStart, pArrRefEnd, pArrOtherStart, pArrOtherEnd, order, valRef, valOther);
}

template<class T>
bool asArraysInsertT(T *pArrRefStart, T *pArrRefEnd, T *pArrOtherStart, T *pArrOtherEnd, const Order order,
                     const T valRef, const T valOther)
{
    wxASSERT(pArrRefStart);
    wxASSERT(pArrRefEnd);
    wxASSERT(pArrOtherStart);
    wxASSERT(pArrOtherEnd);

    // Check the other array length
    auto vlength = (int) (pArrRefEnd - pArrRefStart);
    auto ovlength = (int) (pArrOtherEnd - pArrOtherStart);

    if (vlength != ovlength) {
        wxLogError(_("The dimension of the two arrays are not equal."));
        return false;
    } else if (vlength == 0) {
        wxLogVerbose(_("The array has an unique value."));
        return true;
    } else if (vlength < 0) {
        wxLogError(_("The array has a negative size..."));
        return false;
    }

    // Index where to insert the new element
    int iNext = 0;

    // Check order
    switch (order) {
        case (Asc): {
            iNext = asFindCeil(pArrRefStart, pArrRefEnd, valRef, asHIDE_WARNINGS);
            if (iNext == asOUT_OF_RANGE) {
                iNext = 0;
            }
            break;
        }
        case (Desc): {
            iNext = asFindFloor(pArrRefStart, pArrRefEnd, valRef, asHIDE_WARNINGS);
            if (iNext == asOUT_OF_RANGE) {
                iNext = 0;
            }
            break;
        }
        default: {
            wxLogError(_("Incorrect value of the order enumeration."));
            return false;
        }
    }

    // Swap next elements
    for (int i = vlength - 1; i >= iNext; i--) // Minus 1 because we overwrite the last element by the previous one
    {
        pArrRefStart[i + 1] = pArrRefStart[i];
        pArrOtherStart[i + 1] = pArrOtherStart[i];
    }

    // Insert new element
    pArrRefStart[iNext] = valRef;
    pArrOtherStart[iNext] = valOther;

    return true;
}

bool asSortArray(int *pArrRefStart, int *pArrRefEnd, const Order order)
{
    return asSortArrayT<int>(pArrRefStart, pArrRefEnd, order);
}

bool asSortArray(float *pArrRefStart, float *pArrRefEnd, const Order order)
{
    return asSortArrayT<float>(pArrRefStart, pArrRefEnd, order);
}

bool asSortArray(double *pArrRefStart, double *pArrRefEnd, const Order order)
{
    return asSortArrayT<double>(pArrRefStart, pArrRefEnd, order);
}

template<class T>
bool asSortArrayT(T *pArrRefStart, T *pArrRefEnd, const Order order)
{
    wxASSERT(pArrRefStart);
    wxASSERT(pArrRefEnd);

    // Check the array length
    auto vlength = (int) (pArrRefEnd - pArrRefStart);

    if (vlength > 0) {
        int low = 0, high = vlength;
        asQuickSort<T>(pArrRefStart, low, high, order);
    } else if (vlength == 0) {
        wxLogVerbose(_("The array has an unique value."));
        return true;
    } else {
        wxLogError(_("The array has a negative size..."));
        return false;
    }
    return true;
}

bool asSortArrays(int *pArrRefStart, int *pArrRefEnd, int *pArrOtherStart, int *pArrOtherEnd, const Order order)
{
    return asSortArraysT<int>(pArrRefStart, pArrRefEnd, pArrOtherStart, pArrOtherEnd, order);
}

bool asSortArrays(float *pArrRefStart, float *pArrRefEnd, float *pArrOtherStart, float *pArrOtherEnd,
                         Order order)
{
    return asSortArraysT<float>(pArrRefStart, pArrRefEnd, pArrOtherStart, pArrOtherEnd, order);
}

bool asSortArrays(double *pArrRefStart, double *pArrRefEnd, double *pArrOtherStart, double *pArrOtherEnd,
                         Order order)
{
    return asSortArraysT<double>(pArrRefStart, pArrRefEnd, pArrOtherStart, pArrOtherEnd, order);
}

template<class T>
bool asSortArraysT(T *pArrRefStart, T *pArrRefEnd, T *pArrOtherStart, T *pArrOtherEnd, const Order order)
{
    wxASSERT(pArrRefStart);
    wxASSERT(pArrRefEnd);
    wxASSERT(pArrOtherStart);
    wxASSERT(pArrOtherEnd);

    // Check the other array length
    auto vlength = (int) (pArrRefEnd - pArrRefStart);
    auto ovlength = (int) (pArrOtherEnd - pArrOtherStart);

    if (vlength > 0 && vlength == ovlength) {
        int low = 0, high = vlength;
        asQuickSortMulti<T>(pArrRefStart, pArrOtherStart, low, high, order);
    } else if (vlength != ovlength) {
        wxLogError(_("The dimension of the two arrays are not equal."));
        return false;
    } else if (vlength == 0) {
        wxLogVerbose(_("The array has an unique value."));
        return true;
    } else {
        wxLogError(_("The array has a negative size..."));
        return false;
    }
    return true;
}

template<class T>
void asQuickSort(T *pArr, const int low, const int high, const Order order)
{
    int L, R;
    T pivot, tmp;

    R = high;
    L = low;

    pivot = pArr[(low + high) / 2];

    do {

        switch (order) {
            case (Asc): {
                while (pArr[L] < pivot)
                    L++;
                while (pArr[R] > pivot)
                    R--;
                break;
            }
            case (Desc): {
                while (pArr[L] > pivot)
                    L++;
                while (pArr[R] < pivot)
                    R--;
                break;
            }
            default: {
                wxLogError(_("Incorrect value of the order enumeration."));
                break;
            }
        }

        if (R >= L) {
            if (R != L) {
                tmp = pArr[R];
                pArr[R] = pArr[L];
                pArr[L] = tmp;
            }

            R--;
            L++;
        }
    } while (L <= R);

    if (low < R)
        asQuickSort<T>(pArr, low, R, order);
    if (L < high)
        asQuickSort<T>(pArr, L, high, order);

}

template<class T>
void asQuickSortMulti(T *pArrRef, T *pArrOther, const int low, const int high, const Order order)
{
    int L, R;
    T pivot, tmp;

    R = high;
    L = low;

    pivot = pArrRef[(low + high) / 2];

    do {

        switch (order) {
            case (Asc): {
                while (pArrRef[L] < pivot)
                    L++;
                while (pArrRef[R] > pivot)
                    R--;
                break;
            }
            case (Desc): {
                while (pArrRef[L] > pivot)
                    L++;
                while (pArrRef[R] < pivot)
                    R--;
                break;
            }
            default: {
                wxLogError(_("Incorrect value of the order enumeration."));
                break;
            }
        }

        if (R >= L) {
            if (R != L) {
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

    if (low < R)
        asQuickSortMulti<T>(pArrRef, pArrOther, low, R, order);
    if (L < high)
        asQuickSortMulti<T>(pArrRef, pArrOther, L, high, order);

}
