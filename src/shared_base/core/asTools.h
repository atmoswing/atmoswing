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
    static int Random(int min, int max, int step = 1);

    /** Generate a random number
     * \link http://members.cox.net/srice1/random/crandom.html
     */
    static float Random(float min, float max, float step = 0);

    /** Generate a random number
     * \link http://members.cox.net/srice1/random/crandom.html
     */
    static double Random(double min, double max, double step = 0);

    static int RandomNormalDistribution(int mean, int stDev, int step = 1);

    static float RandomNormalDistribution(float mean, float stDev, float step = 0);

    static double RandomNormalDistribution(double mean, double stDev, double step = 0);

    static bool IsRound(float value);

    static bool IsRound(double value);

    static float Round(float value);

    static double Round(double value);

    static float Mean(int *pArrStart, int *pArrEnd);

    static float Mean(float *pArrStart, float *pArrEnd);

    static double Mean(double *pArrStart, double *pArrEnd);

    /** Standard Deviation of an array
     * \link http://easycalculation.com/statistics/learn-standard-deviation.php
     */
    static float StDev(int *pArrStart, int *pArrEnd, int sample = asSAMPLE);

    /** Standard Deviation of an array
     * \link http://easycalculation.com/statistics/learn-standard-deviation.php
     */
    static float StDev(float *pArrStart, float *pArrEnd, int sample = asSAMPLE);

    /** Standard Deviation of an array
     * \link http://easycalculation.com/statistics/learn-standard-deviation.php
     */
    static double StDev(double *pArrStart, double *pArrEnd, int sample = asSAMPLE);

    static Array1DFloat GetCumulativeFrequency(int size);

    static float GetValueForQuantile(Array1DFloat &values, float quantile);

    static bool IsNaN(int value);

    /** Check if the value is a NaN
     * \link http://www.parashift.com/c++-faq-lite/newbie.html
     */
    static bool IsNaN(float value);

    /** Check if the value is a NaN
     * \link http://www.parashift.com/c++-faq-lite/newbie.html
     */
    static bool IsNaN(double value);

    static bool IsInf(float value);

    static bool IsInf(double value);

    static bool IsInf(long double value);

    static int CountNotNaN(const float *pArrStart, const float *pArrEnd);

    static int CountNotNaN(const double *pArrStart, const double *pArrEnd);

    static bool HasNaN(const Array2DFloat &data);

    static bool HasNaN(const float *pArrStart, const float *pArrEnd);

    static bool HasNaN(const double *pArrStart, const double *pArrEnd);

    static int MinArray(int *pArrStart, int *pArrEnd);

    static float MinArray(float *pArrStart, float *pArrEnd);

    static double MinArray(double *pArrStart, double *pArrEnd);

    static int MinArrayIndex(int *pArrStart, int *pArrEnd);

    static int MinArrayIndex(float *pArrStart, float *pArrEnd);

    static int MinArrayIndex(double *pArrStart, double *pArrEnd);

    static int MaxArray(int *pArrStart, int *pArrEnd);

    static float MaxArray(float *pArrStart, float *pArrEnd);

    static double MaxArray(double *pArrStart, double *pArrEnd);

    static int MaxArrayIndex(int *pArrStart, int *pArrEnd);

    static int MaxArrayIndex(float *pArrStart, float *pArrEnd);

    static int MaxArrayIndex(double *pArrStart, double *pArrEnd);

    static int MinArrayStep(int *pArrStart, int *pArrEnd, int tolerance = 0);

    static float MinArrayStep(float *pArrStart, float *pArrEnd, float tolerance = 0.000001);

    static double MinArrayStep(double *pArrStart, double *pArrEnd, double tolerance = 0.000000001);

    static Array1DInt ExtractUniqueValues(int *pArrStart, int *pArrEnd, int tolerance = 0);

    static Array1DFloat ExtractUniqueValues(float *pArrStart, float *pArrEnd, float tolerance = 0.000001);

    static Array1DDouble ExtractUniqueValues(double *pArrStart, double *pArrEnd, double tolerance = 0.000000001);

    static int SortedArraySearch(int *pArrStart, int *pArrEnd, int targetvalue, int tolerance = 0,
                                 int showWarning = asSHOW_WARNINGS);

    static int SortedArraySearch(float *pArrStart, float *pArrEnd, float targetvalue, float tolerance = 0.0,
                                 int showWarning = asSHOW_WARNINGS);

    static int SortedArraySearch(double *pArrStart, double *pArrEnd, double targetvalue, double tolerance = 0.0,
                                 int showWarning = asSHOW_WARNINGS);

    template<class T>
    static int SortedArraySearchT(T *pArrStart, T *pArrEnd, T targetvalue, T tolerance = 0,
                                  int showWarning = asSHOW_WARNINGS);

    static int SortedArraySearchClosest(int *pArrStart, int *pArrEnd, int targetvalue,
                                        int showWarning = asSHOW_WARNINGS);

    static int SortedArraySearchClosest(float *pArrStart, float *pArrEnd, float targetvalue,
                                        int showWarning = asSHOW_WARNINGS);

    static int SortedArraySearchClosest(double *pArrStart, double *pArrEnd, double targetvalue,
                                        int showWarning = asSHOW_WARNINGS);

    template<class T>
    static int SortedArraySearchClosestT(T *pArrStart, T *pArrEnd, T targetvalue, int showWarning = asSHOW_WARNINGS);

    static int SortedArraySearchFloor(int *pArrStart, int *pArrEnd, int targetvalue, int showWarning = asSHOW_WARNINGS);

    static int SortedArraySearchFloor(float *pArrStart, float *pArrEnd, float targetvalue,
                                      int showWarning = asSHOW_WARNINGS);

    static int SortedArraySearchFloor(double *pArrStart, double *pArrEnd, double targetvalue,
                                      int showWarning = asSHOW_WARNINGS);

    template<class T>
    static int SortedArraySearchFloorT(T *pArrStart, T *pArrEnd, T targetvalue, int showWarning = asSHOW_WARNINGS);

    static int SortedArraySearchCeil(int *pArrStart, int *pArrEnd, int targetvalue, int showWarning = asSHOW_WARNINGS);

    static int SortedArraySearchCeil(float *pArrStart, float *pArrEnd, float targetvalue,
                                     int showWarning = asSHOW_WARNINGS);

    static int SortedArraySearchCeil(double *pArrStart, double *pArrEnd, double targetvalue,
                                     int showWarning = asSHOW_WARNINGS);

    template<class T>
    static int SortedArraySearchCeilT(T *pArrStart, T *pArrEnd, T targetvalue, int showWarning = asSHOW_WARNINGS);

    static bool SortedArrayInsert(int *pArrStart, int *pArrEnd, Order order, int val);

    static bool SortedArrayInsert(float *pArrStart, float *pArrEnd, Order order, float val);

    static bool SortedArrayInsert(double *pArrStart, double *pArrEnd, Order order, double val);

    template<class T>
    static bool SortedArrayInsert(T *pArrStart, T *pArrEnd, Order order, T val);

    static bool SortedArraysInsert(int *pArrRefStart, int *pArrRefEnd, int *pArrOtherStart, int *pArrOtherEnd,
                                   Order order, int valRef, int valOther);

    static bool SortedArraysInsert(float *pArrRefStart, float *pArrRefEnd, float *pArrOtherStart, float *pArrOtherEnd,
                                   Order order, float valRef, float valOther);

    static bool SortedArraysInsert(double *pArrRefStart, double *pArrRefEnd, double *pArrOtherStart,
                                   double *pArrOtherEnd, Order order, double valRef, double valOther);

    template<class T>
    static bool SortedArraysInsert(T *pArrRefStart, T *pArrRefEnd, T *pArrOtherStart, T *pArrOtherEnd, Order order,
                                   T valRef, T valOther);

    static bool SortArray(int *pArrRefStart, int *pArrRefEnd, Order order);

    static bool SortArray(float *pArrRefStart, float *pArrRefEnd, Order order);

    static bool SortArray(double *pArrRefStart, double *pArrRefEnd, Order order);

    template<class T>
    static bool SortArrayT(T *pArrRefStart, T *pArrRefEnd, Order order);

    static bool SortArrays(int *pArrRefStart, int *pArrRefEnd, int *pArrOtherStart, int *pArrOtherEnd, Order order);

    static bool SortArrays(float *pArrRefStart, float *pArrRefEnd, float *pArrOtherStart, float *pArrOtherEnd,
                           Order order);

    static bool SortArrays(double *pArrRefStart, double *pArrRefEnd, double *pArrOtherStart, double *pArrOtherEnd,
                           Order order);

    template<class T>
    static bool SortArraysT(T *pArrRefStart, T *pArrRefEnd, T *pArrOtherStart, T *pArrOtherEnd, Order order);

protected:

private:
    template<class T>
    static void QuickSort(T *pArr, int low, int high, Order order);

    template<class T>
    static void QuickSortMulti(T *pArr, T *pArrOther, int low, int high, Order order);

};

#endif
