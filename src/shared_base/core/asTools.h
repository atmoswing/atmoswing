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

#ifndef ASTOOLS_H
#define ASTOOLS_H

#include <ctime>    // For time()
#include <cstdlib>  // For srand() and rand()

#include <asIncludes.h>

class asTools
        : public wxObject
{
public:
    static void InitRandom();

    /** Generate a random number
     * \link http://members.cox.net/srice1/random/crandom.html
     */
    static int Random(const int min, const int max, const int step = 1);

    /** Generate a random number
     * \link http://members.cox.net/srice1/random/crandom.html
     */
    static float Random(const float min, const float max, const float step = 0);

    /** Generate a random number
     * \link http://members.cox.net/srice1/random/crandom.html
     */
    static double Random(const double min, const double max, const double step = 0);

    static int RandomNormalDistribution(const int mean, const int stDev, const int step = 1);

    static float RandomNormalDistribution(const float mean, const float stDev, const float step = 0);

    static double RandomNormalDistribution(const double mean, const double stDev, const double step = 0);

    static bool IsRound(const float value);

    static bool IsRound(const double value);

    static float Round(const float value);

    static double Round(const double value);

    static float Mean(const int *pArrStart, const int *pArrEnd);

    static float Mean(const float *pArrStart, const float *pArrEnd);

    static double Mean(const double *pArrStart, const double *pArrEnd);

    /** Standard Deviation of an array
     * \link http://easycalculation.com/statistics/learn-standard-deviation.php
     */
    static float StDev(const int *pArrStart, const int *pArrEnd, const int sample = asSAMPLE);

    /** Standard Deviation of an array
     * \link http://easycalculation.com/statistics/learn-standard-deviation.php
     */
    static float StDev(const float *pArrStart, const float *pArrEnd, const int sample = asSAMPLE);

    /** Standard Deviation of an array
     * \link http://easycalculation.com/statistics/learn-standard-deviation.php
     */
    static double StDev(const double *pArrStart, const double *pArrEnd, const int sample = asSAMPLE);

    static a1f GetCumulativeFrequency(const int size);

    static float GetValueForQuantile(const a1f &values, const float quantile);

    static bool IsNaN(const int value);

    /** Check if the value is a NaN
     * \link http://www.parashift.com/c++-faq-lite/newbie.html
     */
    static bool IsNaN(const float value);

    /** Check if the value is a NaN
     * \link http://www.parashift.com/c++-faq-lite/newbie.html
     */
    static bool IsNaN(const double value);

    static bool IsInf(const float value);

    static bool IsInf(const double value);

    static bool IsInf(const long double value);

    static int CountNotNaN(const float *pArrStart, const float *pArrEnd);

    static int CountNotNaN(const double *pArrStart, const double *pArrEnd);

    static bool HasNaN(const a2f &data);

    static bool HasNaN(const float *pArrStart, const float *pArrEnd);

    static bool HasNaN(const double *pArrStart, const double *pArrEnd);

    static int MinArray(const int *pArrStart, const int *pArrEnd);

    static float MinArray(const float *pArrStart, const float *pArrEnd);

    static double MinArray(const double *pArrStart, const double *pArrEnd);

    static int MinArrayIndex(const int *pArrStart, const int *pArrEnd);

    static int MinArrayIndex(const float *pArrStart, const float *pArrEnd);

    static int MinArrayIndex(const double *pArrStart, const double *pArrEnd);

    static int MaxArray(const int *pArrStart, const int *pArrEnd);

    static float MaxArray(const float *pArrStart, const float *pArrEnd);

    static double MaxArray(const double *pArrStart, const double *pArrEnd);

    static int MaxArrayIndex(const int *pArrStart, const int *pArrEnd);

    static int MaxArrayIndex(const float *pArrStart, const float *pArrEnd);

    static int MaxArrayIndex(const double *pArrStart, const double *pArrEnd);

    static int MinArrayStep(const int *pArrStart, const int *pArrEnd, const int tolerance = 0);

    static float MinArrayStep(const float *pArrStart, const float *pArrEnd, const float tolerance = 0.000001);

    static double MinArrayStep(const double *pArrStart, const double *pArrEnd, const double tolerance = 0.000000001);

    static a1i ExtractUniqueValues(const int *pArrStart, const int *pArrEnd, const int tolerance = 0);

    static a1f ExtractUniqueValues(const float *pArrStart, const float *pArrEnd,
                                            const float tolerance = 0.000001);

    static a1d ExtractUniqueValues(const double *pArrStart, const double *pArrEnd,
                                             const double tolerance = 0.000000001);

    static int SortedArraySearch(const int *pArrStart, const int *pArrEnd, const int targetValue,
                                 const int tolerance = 0, const int showWarning = asSHOW_WARNINGS);

    static int SortedArraySearch(const float *pArrStart, const float *pArrEnd, const float targetValue,
                                 const float tolerance = 0.0, const int showWarning = asSHOW_WARNINGS);

    static int SortedArraySearch(const double *pArrStart, const double *pArrEnd, const double targetValue,
                                 const double tolerance = 0.0, const int showWarning = asSHOW_WARNINGS);

    template<class T>
    static int SortedArraySearchT(const T *pArrStart, const T *pArrEnd, const T targetValue, const T tolerance = 0,
                                  const int showWarning = asSHOW_WARNINGS);

    static int SortedArraySearchClosest(const int *pArrStart, const int *pArrEnd, const int targetValue,
                                        const int showWarning = asSHOW_WARNINGS);

    static int SortedArraySearchClosest(const float *pArrStart, const float *pArrEnd, const float targetValue,
                                        const int showWarning = asSHOW_WARNINGS);

    static int SortedArraySearchClosest(const double *pArrStart, const double *pArrEnd, const double targetValue,
                                        const int showWarning = asSHOW_WARNINGS);

    template<class T>
    static int SortedArraySearchClosestT(const T *pArrStart, const T *pArrEnd, const T targetValue,
                                         const int showWarning = asSHOW_WARNINGS);

    static int SortedArraySearchFloor(const int *pArrStart, const int *pArrEnd, const int targetValue,
                                      const int tolerance = 0, const int showWarning = asSHOW_WARNINGS);

    static int SortedArraySearchFloor(const float *pArrStart, const float *pArrEnd, const float targetValue,
                                      const float tolerance = 0, const int showWarning = asSHOW_WARNINGS);

    static int SortedArraySearchFloor(const double *pArrStart, const double *pArrEnd, const double targetValue,
                                      const double tolerance = 0, const int showWarning = asSHOW_WARNINGS);

    template<class T>
    static int SortedArraySearchFloorT(const T *pArrStart, const T *pArrEnd, const T targetValue,
                                       const T tolerance = 0, const int showWarning = asSHOW_WARNINGS);

    static int SortedArraySearchCeil(const int *pArrStart, const int *pArrEnd, const int targetValue,
                                     const int tolerance = 0, const int showWarning = asSHOW_WARNINGS);

    static int SortedArraySearchCeil(const float *pArrStart, const float *pArrEnd, const float targetValue,
                                     const float tolerance = 0, const int showWarning = asSHOW_WARNINGS);

    static int SortedArraySearchCeil(const double *pArrStart, const double *pArrEnd, const double targetValue,
                                     const double tolerance = 0, const int showWarning = asSHOW_WARNINGS);

    template<class T>
    static int SortedArraySearchCeilT(const T *pArrStart, const T *pArrEnd, const T targetValue,
                                      const T tolerance = 0, const int showWarning = asSHOW_WARNINGS);

    static bool SortedArrayInsert(int *pArrStart, int *pArrEnd, const Order order, const int val);

    static bool SortedArrayInsert(float *pArrStart, float *pArrEnd, const Order order, const float val);

    static bool SortedArrayInsert(double *pArrStart, double *pArrEnd, const Order order, const double val);

    template<class T>
    static bool SortedArrayInsert(T *pArrStart, T *pArrEnd, const Order order, const T val);

    static bool SortedArraysInsert(int *pArrRefStart, int *pArrRefEnd, int *pArrOtherStart, int *pArrOtherEnd,
                                   const Order order, const int valRef, const int valOther);

    static bool SortedArraysInsert(float *pArrRefStart, float *pArrRefEnd, float *pArrOtherStart, float *pArrOtherEnd,
                                   const Order order, const float valRef, const float valOther);

    static bool SortedArraysInsert(double *pArrRefStart, double *pArrRefEnd, double *pArrOtherStart,
                                   double *pArrOtherEnd, const Order order, const double valRef, const double valOther);

    template<class T>
    static bool SortedArraysInsert(T *pArrRefStart, T *pArrRefEnd, T *pArrOtherStart, T *pArrOtherEnd,
                                   const Order order, T valRef, const T valOther);

    static bool SortArray(int *pArrRefStart, int *pArrRefEnd, const Order order);

    static bool SortArray(float *pArrRefStart, float *pArrRefEnd, const Order order);

    static bool SortArray(double *pArrRefStart, double *pArrRefEnd, const Order order);

    template<class T>
    static bool SortArrayT(T *pArrRefStart, T *pArrRefEnd, const Order order);

    static bool SortArrays(int *pArrRefStart, int *pArrRefEnd, int *pArrOtherStart, int *pArrOtherEnd,
                           const Order order);

    static bool SortArrays(float *pArrRefStart, float *pArrRefEnd, float *pArrOtherStart, float *pArrOtherEnd,
                           const Order order);

    static bool SortArrays(double *pArrRefStart, double *pArrRefEnd, double *pArrOtherStart, double *pArrOtherEnd,
                           const Order order);

    template<class T>
    static bool SortArraysT(T *pArrRefStart, T *pArrRefEnd, T *pArrOtherStart, T *pArrOtherEnd, const Order order);

protected:

private:
    template<class T>
    static void QuickSort(T *pArr, const int low, const int high, const Order order);

    template<class T>
    static void QuickSortMulti(T *pArr, T *pArrOther, const int low, const int high, const Order order);

};

#endif
