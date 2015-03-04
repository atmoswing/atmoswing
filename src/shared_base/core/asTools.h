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

#ifndef ASTOOLS_H
#define ASTOOLS_H

#include <ctime>    // For time()
#include <cstdlib>  // For srand() and rand()

#include <asIncludes.h>

class asTools: public wxObject
{
public:

    /** Initialize the random generator (seed)
     * \note Caution, must never be called in a loop! Only once before.
     */
    static void InitRandom();

    /** Generate a random number
     * \param min The minimum value
     * \param max The maximum value
     * \param max The step
     * \return True if round.
     * \link http://members.cox.net/srice1/random/crandom.html
     */
    static int Random(int min, int max, int step=1);

    /** Generate a random number
     * \param min The minimum value
     * \param max The maximum value
     * \param max The step
     * \return True if round.
     * \link http://members.cox.net/srice1/random/crandom.html
     */
    static float Random(float min, float max, float step=0);

    /** Generate a random number
     * \param min The minimum value
     * \param max The maximum value
     * \param max The step
     * \return True if round.
     * \link http://members.cox.net/srice1/random/crandom.html
     */
    static double Random(double min, double max, double step=0);

    static int RandomNormalDistribution(int mean, int stDev, int step=1);

    static float RandomNormalDistribution(float mean, float stDev, float step=0);

    static double RandomNormalDistribution(double mean, double stDev, double step=0);

    /** Check if a value is round
     * \param value The value to check
     * \return True if round.
     */
    static bool IsRound(float value);

    /** Check if a value is round
     * \param value The value to check
     * \return True if round.
     */
    static bool IsRound(double value);

    /** Simple rounding of a value
     * \param value The value to round
     * \return The rounded value.
     */
    static float Round(float value);

    /** Simple rounding of a value
     * \param value The value to round
     * \return The rounded value.
     */
    static double Round(double value);

    /** Mean of an array
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \return The mean value.
     */
    static float Mean(int* pArrStart, int* pArrEnd);

    /** Mean of an array
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \return The mean value.
     */
    static float Mean(float* pArrStart, float* pArrEnd);

    /** Mean of an array
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \return The mean value.
     */
    static double Mean(double* pArrStart, double* pArrEnd);

    /** Standard Deviation of an array
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \param sample asSAMPLE for stdev of a sample (1/(n-1)), asENTIRE_POPULATION for stdev of the entire population (1/n)
     * \return The standard deviation.
     * \link http://easycalculation.com/statistics/learn-standard-deviation.php
     */
    static float StDev(int* pArrStart, int* pArrEnd, int sample = asSAMPLE);

    /** Standard Deviation of an array
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \param sample asSAMPLE for stdev of a sample (1/(n-1)), asENTIRE_POPULATION for stdev of the entire population (1/n)
     * \return The standard deviation.
     * \link http://easycalculation.com/statistics/learn-standard-deviation.php
     */
    static float StDev(float* pArrStart, float* pArrEnd, int sample = asSAMPLE);

    /** Standard Deviation of an array
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \param sample asSAMPLE for stdev of a sample (1/(n-1)), asENTIRE_POPULATION for stdev of the entire population (1/n)
     * \return The standard deviation.
     * \link http://easycalculation.com/statistics/learn-standard-deviation.php
     */
    static double StDev(double* pArrStart, double* pArrEnd, int sample = asSAMPLE);

    /** Get the cumulative frequency
     * \return The cumulative frequency corresponding to a certain sample size
     */
    static Array1DFloat GetCumulativeFrequency(int size);

    static float GetValueForQuantile(Array1DFloat &values, float quantile);

    /** Check if the value is a NaN
     * \param value The value to check
     * \return True if NaN.
     */
    static bool IsNaN(int value);

    /** Check if the value is a NaN
     * \param value The value to check
     * \return True if NaN.
     * \link http://www.parashift.com/c++-faq-lite/newbie.html
     */
    static bool IsNaN(float value);

    /** Check if the value is a NaN
     * \param value The value to check
     * \return True if NaN.
     * \link http://www.parashift.com/c++-faq-lite/newbie.html
     */
    static bool IsNaN(double value);

    /** Check if the value is an Inf
     * \param value The value to check
     * \return True if Inf.
     */
    static bool IsInf(float value);

    /** Check if the value is an Inf
     * \param value The value to check
     * \return True if Inf.
     */
    static bool IsInf(double value);

    /** Check if the value is an Inf
     * \param value The value to check
     * \return True if Inf.
     */
    static bool IsInf(long double value);

    /** Counts the number of not NaNs in the array
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \return The number of not NaNs.
     */
    static int CountNotNaN(const float* pArrStart, const float* pArrEnd);

    /** Counts the number of not NaNs in the array
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \return The number of not NaNs.
     */
    static int CountNotNaN(const double* pArrStart, const double* pArrEnd);

    static bool HasNaN(const Array2DFloat &data);

    /** Check if there is any NaN in the array
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \return True if NaNs were found.
     */
    static bool HasNaN(const float* pArrStart, const float* pArrEnd);

    /** Check if there is any NaN in the array
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \return True if NaNs were found.
     */
    static bool HasNaN(const double* pArrStart, const double* pArrEnd);

    /** Search the minimum value in an array of ints.
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \return The minimum value.
     */
    static int MinArray(int* pArrStart, int* pArrEnd);

    /** Search the minimum value in an array of floats.
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \return The minimum value.
     */
    static float MinArray(float* pArrStart, float* pArrEnd);

    /** Search the minimum value in an array of doubles.
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \return The minimum value.
     */
    static double MinArray(double* pArrStart, double* pArrEnd);

    /** Search the minimum value in an array of ints.
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \return The minimum value index.
     */
    static int MinArrayIndex(int* pArrStart, int* pArrEnd);

    /** Search the minimum value in an array of floats.
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \return The minimum value index.
     */
    static int MinArrayIndex(float* pArrStart, float* pArrEnd);

    /** Search the minimum value in an array of doubles.
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \return The minimum value index.
     */
    static int MinArrayIndex(double* pArrStart, double* pArrEnd);

    /** Search the maximum value in an array of ints.
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \return The maximum value.
     */
    static int MaxArray(int* pArrStart, int* pArrEnd);

    /** Search the maximum value in an array of floats.
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \return The maximum value.
     */
    static float MaxArray(float* pArrStart, float* pArrEnd);

    /** Search the maximum value in an array of doubles.
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \return The maximum value.
     */
    static double MaxArray(double* pArrStart, double* pArrEnd);

    /** Search the maximum value in an array of ints.
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \return The maximum value index.
     */
    static int MaxArrayIndex(int* pArrStart, int* pArrEnd);

    /** Search the maximum value in an array of floats.
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \return The maximum value index.
     */
    static int MaxArrayIndex(float* pArrStart, float* pArrEnd);

    /** Search the maximum value in an array of doubles.
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \return The maximum value index.
     */
    static int MaxArrayIndex(double* pArrStart, double* pArrEnd);

    /** Search the minimum step between values in an array.
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \param tolerance A tolerance to ignore too small differences
     * \return The minimum step.
     */
    static int MinArrayStep(int* pArrStart, int* pArrEnd, int tolerance = 0);

    /** Search the minimum step between values in an array.
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \param tolerance A tolerance to ignore too small differences
     * \return The minimum step.
     */
    static float MinArrayStep(float* pArrStart, float* pArrEnd, float tolerance = 0.000001);

    /** Search the minimum step between values in an array.
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \param tolerance A tolerance to ignore too small differences
     * \return The minimum step.
     */
    static double MinArrayStep(double* pArrStart, double* pArrEnd, double tolerance = 0.000000001);

    /** Extract unique values from an array.
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \param tolerance A tolerance to ignore too small differences
     * \return An array of unique values.
     */
    static Array1DInt ExtractUniqueValues(int* pArrStart, int* pArrEnd, int tolerance = 0);

    /** Extract unique values from an array.
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \param tolerance A tolerance to ignore too small differences
     * \return An array of unique values.
     */
    static Array1DFloat ExtractUniqueValues(float* pArrStart, float* pArrEnd, float tolerance = 0.000001);

    /** Extract unique values from an array.
     * \param pArrStart The beginning (pointer) of the vector/array
     * \param pArrEnd The end (pointer) of the vector/array
     * \param tolerance A tolerance to ignore too small differences
     * \return An array of unique values.
     */
    static Array1DDouble ExtractUniqueValues(double* pArrStart, double* pArrEnd, double tolerance = 0.000000001);

    /** Binaray searching of a value in an array of ints. Interface to the template
     * \param pArrStart The beginning (pointer) of the vector/array to sort
     * \param pArrEnd The end (pointer) of the vector/array to sort
     * \param targetvalue The searched value
     * \param tolerance a tolerance to search within
     * \return The array index where the value was found. -1 if not found.
     */
    static int SortedArraySearch(int* pArrStart, int* pArrEnd, int targetvalue, int tolerance = 0, int showWarning = asSHOW_WARNINGS);

    /** Binaray searching of a value in an array of floats. Interface to the template
     * \param pArrStart The beginning (pointer) of the vector/array to sort
     * \param pArrEnd The end (pointer) of the vector/array to sort
     * \param targetvalue The searched value
     * \param tolerance a tolerance to search within
     * \return The array index where the value was found. -1 if not found.
     */
    static int SortedArraySearch(float* pArrStart, float* pArrEnd, float targetvalue, float tolerance = 0.0, int showWarning = asSHOW_WARNINGS);

    /** Binaray searching of a value in an array of doubles. Interface to the template
     * \param pArrStart The beginning (pointer) of the vector/array to sort
     * \param pArrEnd The end (pointer) of the vector/array to sort
     * \param targetvalue The searched value
     * \param tolerance a tolerance to search within
     * \return The array index where the value was found. -1 if not found.
     */
    static int SortedArraySearch(double* pArrStart, double* pArrEnd, double targetvalue, double tolerance = 0.0, int showWarning = asSHOW_WARNINGS);

    /** Binaray searching of a value in an array
     * \param pArrStart The beginning (pointer) of the vector/array to sort
     * \param pArrEnd The end (pointer) of the vector/array to sort
     * \param targetvalue The searched value
     * \param tolerance a tolerance to search within
     * \return The array index where the value was found. -1 if not found.
     */
    template< class T >
    static int SortedArraySearchT(T* pArrStart, T* pArrEnd, T targetvalue, T tolerance = 0, int showWarning = asSHOW_WARNINGS);

    /** Binaray searching of a value in an array of ints. Return the index of the closest value. Interface to the template
     * \param pArrStart The beginning (pointer) of the vector/array to sort
     * \param pArrEnd The end (pointer) of the vector/array to sort
     * \param targetvalue The searched value
     * \return The closest array index where the value was found. -1 if not found.
     */
    static int SortedArraySearchClosest(int* pArrStart, int* pArrEnd, int targetvalue, int showWarning = asSHOW_WARNINGS);

    /** Binaray searching of a value in an array of floats. Return the index of the closest value. Interface to the template
     * \param pArrStart The beginning (pointer) of the vector/array to sort
     * \param pArrEnd The end (pointer) of the vector/array to sort
     * \param targetvalue The searched value
     * \return The closest array index where the value was found. -1 if not found.
     */
    static int SortedArraySearchClosest(float* pArrStart, float* pArrEnd, float targetvalue, int showWarning = asSHOW_WARNINGS);

    /** Binaray searching of a value in an array of doubles. Return the index of the closest value. Interface to the template
     * \param pArrStart The beginning (pointer) of the vector/array to sort
     * \param pArrEnd The end (pointer) of the vector/array to sort
     * \param targetvalue The searched value
     * \return The closest array index where the value was found. -1 if not found.
     */
    static int SortedArraySearchClosest(double* pArrStart, double* pArrEnd, double targetvalue, int showWarning = asSHOW_WARNINGS);

    /** Binaray searching of a value in an array. Return the index of the closest value.
     * \param pArrStart The beginning (pointer) of the vector/array to sort
     * \param pArrEnd The end (pointer) of the vector/array to sort
     * \param targetvalue The searched value
     * \return The closest array index where the value was found. -1 if not found.
     */
    template< class T >
    static int SortedArraySearchClosestT(T* pArrStart, T* pArrEnd, T targetvalue, int showWarning = asSHOW_WARNINGS);

    /** Binaray searching of a value in an array of ints. Return the index of the floor value. Interface to the template
     * \param pArrStart The beginning (pointer) of the vector/array to sort
     * \param pArrEnd The end (pointer) of the vector/array to sort
     * \param targetvalue The searched value
     * \return The floor array index where the value was found. -1 if not found.
     */
    static int SortedArraySearchFloor(int* pArrStart, int* pArrEnd, int targetvalue, int showWarning = asSHOW_WARNINGS);

    /** Binaray searching of a value in an array of floats. Return the index of the floor value. Interface to the template
     * \param pArrStart The beginning (pointer) of the vector/array to sort
     * \param pArrEnd The end (pointer) of the vector/array to sort
     * \param targetvalue The searched value
     * \return The floor array index where the value was found. -1 if not found.
     */
    static int SortedArraySearchFloor(float* pArrStart, float* pArrEnd, float targetvalue, int showWarning = asSHOW_WARNINGS);

    /** Binaray searching of a value in an array of doubles. Return the index of the floor value. Interface to the template
     * \param pArrStart The beginning (pointer) of the vector/array to sort
     * \param pArrEnd The end (pointer) of the vector/array to sort
     * \param targetvalue The searched value
     * \return The floor array index where the value was found. -1 if not found.
     */
    static int SortedArraySearchFloor(double* pArrStart, double* pArrEnd, double targetvalue, int showWarning = asSHOW_WARNINGS);

    /** Binaray searching of a value in an array. Return the index of the floor value.
     * \param pArrStart The beginning (pointer) of the vector/array to sort
     * \param pArrEnd The end (pointer) of the vector/array to sort
     * \param targetvalue The searched value
     * \return The floor array index where the value was found. -1 if not found.
     */
    template< class T >
    static int SortedArraySearchFloorT(T* pArrStart, T* pArrEnd, T targetvalue, int showWarning = asSHOW_WARNINGS);

    /** Binaray searching of a value in an array of ints. Return the index of the ceil value. Interface to the template
     * \param pArrStart The beginning (pointer) of the vector/array to sort
     * \param pArrEnd The end (pointer) of the vector/array to sort
     * \param targetvalue The searched value
     * \return The floor array index where the value was found. -1 if not found.
     */
    static int SortedArraySearchCeil(int* pArrStart, int* pArrEnd, int targetvalue, int showWarning = asSHOW_WARNINGS);

    /** Binaray searching of a value in an array of floats. Return the index of the ceil value. Interface to the template
     * \param pArrStart The beginning (pointer) of the vector/array to sort
     * \param pArrEnd The end (pointer) of the vector/array to sort
     * \param targetvalue The searched value
     * \return The floor array index where the value was found. -1 if not found.
     */
    static int SortedArraySearchCeil(float* pArrStart, float* pArrEnd, float targetvalue, int showWarning = asSHOW_WARNINGS);

    /** Binaray searching of a value in an array of doubles. Return the index of the ceil value. Interface to the template
     * \param pArrStart The beginning (pointer) of the vector/array to sort
     * \param pArrEnd The end (pointer) of the vector/array to sort
     * \param targetvalue The searched value
     * \return The floor array index where the value was found. -1 if not found.
     */
    static int SortedArraySearchCeil(double* pArrStart, double* pArrEnd, double targetvalue, int showWarning = asSHOW_WARNINGS);

    /** Binaray searching of a value in an array. Return the index of the ceil value.
     * \param pArrStart The beginning (pointer) of the vector/array to sort
     * \param pArrEnd The end (pointer) of the vector/array to sort
     * \param targetvalue The searched value
     * \return The floor array index where the value was found. -1 if not found.
     */
    template< class T >
    static int SortedArraySearchCeilT(T* pArrStart, T* pArrEnd, T targetvalue, int showWarning = asSHOW_WARNINGS);

    /** Insert a new value in a sorted array of ints. Interface to the template
     * \param pArrStart The beginning (pointer) of the vector/array to sort
     * \param pArrEnd The end (pointer) of the vector/array to sort
     * \param order The order of the array (Asc or Desc)
     * \param val The value to insert
     * \return True on success.
     */
    static bool SortedArrayInsert(int* pArrStart, int* pArrEnd, Order order, int val);

    /** Insert a new value in a sorted array of floats. Interface to the template
     * \param pArrStart The beginning (pointer) of the vector/array to sort
     * \param pArrEnd The end (pointer) of the vector/array to sort
     * \param order The order of the array (Asc or Desc)
     * \param val The value to insert
     * \return True on success.
     */
    static bool SortedArrayInsert(float* pArrStart, float* pArrEnd, Order order, float val);

    /** Insert a new value in a sorted array of doubles. Interface to the template
     * \param pArrStart The beginning (pointer) of the vector/array to sort
     * \param pArrEnd The end (pointer) of the vector/array to sort
     * \param order The order of the array (Asc or Desc)
     * \param val The value to insert
     * \return True on success.
     */
    static bool SortedArrayInsert(double* pArrStart, double* pArrEnd, Order order, double val);

    /** Insert a new value in a sorted array. Template
     * \param pArrStart The beginning (pointer) of the vector/array to sort
     * \param pArrEnd The end (pointer) of the vector/array to sort
     * \param order The order of the array (Asc or Desc)
     * \param val The value to insert
     * \return True on success.
     */
    template <class T>
    static bool SortedArrayInsert(T* pArrStart, T* pArrEnd, Order order, T val);

    /** Insert a new value in a sorted array and the corresponding value in another array. Interface to the template
    * \param pArrRefStart The beginning (pointer) of the sorted vector/array
    * \param pArrRefEnd The end (pointer) of the sorted vector/array
    * \param pArrOtherStart The beginning (pointer) of the other vector/array
    * \param pArrOtherEnd The end (pointer) of the other vector/array
    * \param order The order of the array (Asc or Desc)
    * \param valRef The value to insert in the sorted array
    * \param valOther The value to insert in the second array
    * \return True in case of success, false otherwise
    */
    static bool SortedArraysInsert(int* pArrRefStart, int* pArrRefEnd, int* pArrOtherStart, int* pArrOtherEnd, Order order, int valRef, int valOther);

    /** Insert a new value in a sorted array and the corresponding value in another array. Interface to the template
    * \param pArrRefStart The beginning (pointer) of the sorted vector/array
    * \param pArrRefEnd The end (pointer) of the sorted vector/array
    * \param pArrOtherStart The beginning (pointer) of the other vector/array
    * \param pArrOtherEnd The end (pointer) of the other vector/array
    * \param order The order of the array (Asc or Desc)
    * \param valRef The value to insert in the sorted array
    * \param valOther The value to insert in the second array
    * \return True in case of success, false otherwise
    */
    static bool SortedArraysInsert(float* pArrRefStart, float* pArrRefEnd, float* pArrOtherStart, float* pArrOtherEnd, Order order, float valRef, float valOther);

    /** Insert a new value in a sorted array and the corresponding value in another array. Interface to the template
    * \param pArrRefStart The beginning (pointer) of the sorted vector/array
    * \param pArrRefEnd The end (pointer) of the sorted vector/array
    * \param pArrOtherStart The beginning (pointer) of the other vector/array
    * \param pArrOtherEnd The end (pointer) of the other vector/array
    * \param order The order of the array (Asc or Desc)
    * \param valRef The value to insert in the sorted array
    * \param valOther The value to insert in the second array
    * \return True in case of success, false otherwise
    */
    static bool SortedArraysInsert(double* pArrRefStart, double* pArrRefEnd, double* pArrOtherStart, double* pArrOtherEnd, Order order, double valRef, double valOther);

    /** Insert a new value in a sorted array and the corresponding value in another array. Template
    * \param pArrRefStart The beginning (pointer) of the sorted vector/array
    * \param pArrRefEnd The end (pointer) of the sorted vector/array
    * \param pArrOtherStart The beginning (pointer) of the other vector/array
    * \param pArrOtherEnd The end (pointer) of the other vector/array
    * \param order The order of the array (Asc or Desc)
    * \param valRef The value to insert in the sorted array
    * \param valOther The value to insert in the second array
    * \return True in case of success, false otherwise
    */
    template <class T>
    static bool SortedArraysInsert(T* pArrRefStart, T* pArrRefEnd, T* pArrOtherStart, T* pArrOtherEnd, Order order, T valRef, T valOther);

    /** A sorting algorithm for one or two vecor/array or other containers. Interface to the template
    * \param pArrRefStart The beginning (pointer) of the vector/array to sort
    * \param pArrRefEnd The end (pointer) of the vector/array to sort
    * \param order The sorting order (ASC or DESC)
    * \return True in case of success, false otherwise
    * \note Based on the Quick Sort approach (pivot)
    */
    static bool SortArray(int* pArrRefStart, int* pArrRefEnd, Order order);

    /** A sorting algorithm for one or two vecor/array or other containers. Interface to the template
    * \param pArrRefStart The beginning (pointer) of the vector/array to sort
    * \param pArrRefEnd The end (pointer) of the vector/array to sort
    * \param order The sorting order (ASC or DESC)
    * \param pArrOtherStart The beginning (pointer) of the other vector/array to sort as the first one
    * \param pArrOtherEnd The end (pointer) of the other vector/array to sort as the first one
    * \return True in case of success, false otherwise
    * \note Based on the Quick Sort approach (pivot)
    */
    static bool SortArray(float* pArrRefStart, float* pArrRefEnd, Order order);

    /** A sorting algorithm for one or two vecor/array or other containers. Interface to the template
    * \param pArrRefStart The beginning (pointer) of the vector/array to sort
    * \param pArrRefEnd The end (pointer) of the vector/array to sort
    * \param order The sorting order (ASC or DESC)
    * \param pArrOtherStart The beginning (pointer) of the other vector/array to sort as the first one
    * \param pArrOtherEnd The end (pointer) of the other vector/array to sort as the first one
    * \return True in case of success, false otherwise
    * \note Based on the Quick Sort approach (pivot)
    */
    static bool SortArray(double* pArrRefStart, double* pArrRefEnd, Order order);

    /** A sorting algorithm for one or two vecor/array or other containers. Based on Quick Sort
    * \param pArrRefStart The beginning (pointer) of the vector/array to sort
    * \param pArrRefEnd The end (pointer) of the vector/array to sort
    * \param order The sorting order (ASC or DESC)
    * \param pArrOtherStart The beginning (pointer) of the other vector/array to sort as the first one
    * \param pArrOtherEnd The end (pointer) of the other vector/array to sort as the first one
    * \return True in case of success, false otherwise
    * \note Based on the Quick Sort approach (pivot)
    */
    template <class T>
    static bool SortArrayT(T* pArrRefStart, T* pArrRefEnd, Order order);

    /** A sorting algorithm for one or two vecor/array or other containers. Interface to the template
    * \param pArrRefStart The beginning (pointer) of the vector/array to sort
    * \param pArrRefEnd The end (pointer) of the vector/array to sort
    * \param order The sorting order (ASC or DESC)
    * \param pArrOtherStart The beginning (pointer) of the other vector/array to sort as the first one
    * \param pArrOtherEnd The end (pointer) of the other vector/array to sort as the first one
    * \return True in case of success, false otherwise
    * \note Based on the Quick Sort approach (pivot)
    */
    static bool SortArrays(int* pArrRefStart, int* pArrRefEnd, int* pArrOtherStart, int* pArrOtherEnd, Order order);

    /** A sorting algorithm for one or two vecor/array or other containers. Interface to the template
    * \param pArrRefStart The beginning (pointer) of the vector/array to sort
    * \param pArrRefEnd The end (pointer) of the vector/array to sort
    * \param order The sorting order (ASC or DESC)
    * \param pArrOtherStart The beginning (pointer) of the other vector/array to sort as the first one
    * \param pArrOtherEnd The end (pointer) of the other vector/array to sort as the first one
    * \return True in case of success, false otherwise
    * \note Based on the Quick Sort approach (pivot)
    */
    static bool SortArrays(float* pArrRefStart, float* pArrRefEnd, float* pArrOtherStart, float* pArrOtherEnd, Order order);

    /** A sorting algorithm for one or two vecor/array or other containers. Interface to the template
    * \param pArrRefStart The beginning (pointer) of the vector/array to sort
    * \param pArrRefEnd The end (pointer) of the vector/array to sort
    * \param order The sorting order (ASC or DESC)
    * \param pArrOtherStart The beginning (pointer) of the other vector/array to sort as the first one
    * \param pArrOtherEnd The end (pointer) of the other vector/array to sort as the first one
    * \return True in case of success, false otherwise
    * \note Based on the Quick Sort approach (pivot)
    */
    static bool SortArrays(double* pArrRefStart, double* pArrRefEnd, double* pArrOtherStart, double* pArrOtherEnd, Order order);

    /** A sorting algorithm for one or two vecor/array or other containers. Based on Quick Sort
    * \param pArrRefStart The beginning (pointer) of the vector/array to sort
    * \param pArrRefEnd The end (pointer) of the vector/array to sort
    * \param order The sorting order (ASC or DESC)
    * \param pArrOtherStart The beginning (pointer) of the other vector/array to sort as the first one
    * \param pArrOtherEnd The end (pointer) of the other vector/array to sort as the first one
    * \return True in case of success, false otherwise
    * \note Based on the Quick Sort approach (pivot)
    */
    template <class T>
    static bool SortArraysT(T* pArrRefStart, T* pArrRefEnd, T* pArrOtherStart, T* pArrOtherEnd, Order order);

protected:
private:

    /** QuickSort - core of algorithm for 1 vector/array. Do not call it, call SortVectors
    * \param pArr The beginning (pointer) of the vector/array to sort
    * \param low The index of the first element of the block to sort
    * \param high The index of the last element of the block to sort
    * \param order The sorting order (ASC or DESC)
    * \note Based on the Quick Sort approach (pivot). Inspired of the work of Martin Ziacek, Martin.Ziacek@swh.sk, http://www.swh.sk
    */
    template <class T>
    static void QuickSort(T *pArr, int low, int high, Order order );

    /** QuickSort - core of algorithm for 2 vectors/arrays. Do not call it, call SortVectors
    * \param pArr The beginning (pointer) of the vector/array to sort
    * \param pArrOther The beginning (pointer) of the other vector/array
    * \param low The index of the first element of the block to sort
    * \param high The index of the last element of the block to sort
    * \param order The sorting order (ASC or DESC)
    * \note Based on the Quick Sort approach (pivot). Inspired of the work of Martin Ziacek, Martin.Ziacek@swh.sk, http://www.swh.sk
    */
    template <class T>
    static void QuickSortMulti(T *pArr, T *pArrOther, int low, int high, Order order );


};

#endif
