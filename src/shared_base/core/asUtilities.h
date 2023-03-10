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

#ifndef AS_UTILITIES_H
#define AS_UTILITIES_H

#include <ctime>
#include <random>

#include "asIncludes.h"

template <typename... Args>
wxString asStrF(const wxString& format, Args... args) {
    return wxString::Format(format, args...);
}

void asThrow(const wxString& msg);

bool asRemoveDir(const wxString& path);

void asInitRandom();

/** Generate a random number
 * \link http://members.cox.net/srice1/random/crandom.html
 */
int asRandom(int min, int max, int step = 1);

/** Generate a random number
 * \link http://members.cox.net/srice1/random/crandom.html
 */
float asRandom(float min, float max, float step = 0);

/** Generate a random number
 * \link http://members.cox.net/srice1/random/crandom.html
 */
double asRandom(double min, double max, double step = 0);

int asRandomNormal(int mean, int stDev, int step = 1);

float asRandomNormal(float mean, float stDev, float step = 0);

double asRandomNormal(double mean, double stDev, double step = 0);

bool asIsRound(float value);

bool asIsRound(double value);

float asRound(float value);

double asRound(double value);

float asMean(const int* pArrStart, const int* pArrEnd);

float asMean(const float* pArrStart, const float* pArrEnd);

double asMean(const double* pArrStart, const double* pArrEnd);

/** Standard Deviation of an array
 * \link http://easycalculation.com/statistics/learn-standard-deviation.php
 */
float asStDev(const int* pArrStart, const int* pArrEnd, int sample = asSAMPLE);

/** Standard Deviation of an array
 * \link http://easycalculation.com/statistics/learn-standard-deviation.php
 */
float asStDev(const float* pArrStart, const float* pArrEnd, int sample = asSAMPLE);

/** Standard Deviation of an array
 * \link http://easycalculation.com/statistics/learn-standard-deviation.php
 */
double asStDev(const double* pArrStart, const double* pArrEnd, int sample = asSAMPLE);

a1f asGetCumulativeFrequency(int size);

float asGetValueForQuantile(const a1f& values, float quantile);

bool asIsNaN(int value);

/** Check if the value is a NaN
 * \link http://www.parashift.com/c++-faq-lite/newbie.html
 */
bool asIsNaN(float value);

/** Check if the value is a NaN
 * \link http://www.parashift.com/c++-faq-lite/newbie.html
 */
bool asIsNaN(double value);

bool asIsInf(float value);

bool asIsInf(double value);

bool asIsInf(long double value);

int asCountNotNaN(const float* pArrStart, const float* pArrEnd);

int asCountNotNaN(const double* pArrStart, const double* pArrEnd);

bool asHasNaN(const a2f& data);

bool asHasNaN(const float* pArrStart, const float* pArrEnd);

bool asHasNaN(const double* pArrStart, const double* pArrEnd);

int asMinArray(const int* pArrStart, const int* pArrEnd);

float asMinArray(const float* pArrStart, const float* pArrEnd);

double asMinArray(const double* pArrStart, const double* pArrEnd);

int asMinArrayIndex(const int* pArrStart, const int* pArrEnd);

int asMinArrayIndex(const float* pArrStart, const float* pArrEnd);

int asMinArrayIndex(const double* pArrStart, const double* pArrEnd);

int asMaxArray(const int* pArrStart, const int* pArrEnd);

float asMaxArray(const float* pArrStart, const float* pArrEnd);

double asMaxArray(const double* pArrStart, const double* pArrEnd);

int asMaxArrayIndex(const int* pArrStart, const int* pArrEnd);

int asMaxArrayIndex(const float* pArrStart, const float* pArrEnd);

int asMaxArrayIndex(const double* pArrStart, const double* pArrEnd);

int asMinArrayStep(const int* pArrStart, const int* pArrEnd, int tolerance = 0);

float asMinArrayStep(const float* pArrStart, const float* pArrEnd, float tolerance = 0.000001);

double asMinArrayStep(const double* pArrStart, const double* pArrEnd, double tolerance = 0.000000001);

a1i asExtractUniqueValues(const int* pArrStart, const int* pArrEnd, int tolerance = 0);

a1f asExtractUniqueValues(const float* pArrStart, const float* pArrEnd, float tolerance = 0.000001);

a1d asExtractUniqueValues(const double* pArrStart, const double* pArrEnd, double tolerance = 0.000000001);

int asFind(const int* pArrStart, const int* pArrEnd, int targetValue, int tolerance = 0,
           int showWarning = asSHOW_WARNINGS);

int asFind(const float* pArrStart, const float* pArrEnd, float targetValue, float tolerance = 0.0,
           int showWarning = asSHOW_WARNINGS);

int asFind(const double* pArrStart, const double* pArrEnd, double targetValue, double tolerance = 0.0,
           int showWarning = asSHOW_WARNINGS);

template <class T>
int asFindT(const T* pArrStart, const T* pArrEnd, T targetValue, T tolerance = 0, int showWarning = asSHOW_WARNINGS);

int asFindClosest(const int* pArrStart, const int* pArrEnd, int targetValue, int showWarning = asSHOW_WARNINGS);

int asFindClosest(const float* pArrStart, const float* pArrEnd, float targetValue, int showWarning = asSHOW_WARNINGS);

int asFindClosest(const double* pArrStart, const double* pArrEnd, double targetValue,
                  int showWarning = asSHOW_WARNINGS);

template <class T>
int asFindClosestT(const T* pArrStart, const T* pArrEnd, T targetValue, int showWarning = asSHOW_WARNINGS);

int asFindFloor(const int* pArrStart, const int* pArrEnd, int targetValue, int showWarning = asSHOW_WARNINGS);

int asFindFloor(const float* pArrStart, const float* pArrEnd, float targetValue, int showWarning = asSHOW_WARNINGS);

int asFindFloor(const double* pArrStart, const double* pArrEnd, double targetValue, int showWarning = asSHOW_WARNINGS);

template <class T>
int asFindFloorT(const T* pArrStart, const T* pArrEnd, T targetValue, int showWarning = asSHOW_WARNINGS);

int asFindCeil(const int* pArrStart, const int* pArrEnd, int targetValue, int showWarning = asSHOW_WARNINGS);

int asFindCeil(const float* pArrStart, const float* pArrEnd, float targetValue, int showWarning = asSHOW_WARNINGS);

int asFindCeil(const double* pArrStart, const double* pArrEnd, double targetValue, int showWarning = asSHOW_WARNINGS);

template <class T>
int asFindCeilT(const T* pArrStart, const T* pArrEnd, T targetValue, int showWarning = asSHOW_WARNINGS);

bool asArrayInsert(int* pArrStart, int* pArrEnd, Order order, int val);

bool asArrayInsert(float* pArrStart, float* pArrEnd, Order order, float val);

bool asArrayInsert(double* pArrStart, double* pArrEnd, Order order, double val);

template <class T>
bool asArrayInsertT(T* pArrStart, T* pArrEnd, Order order, T val);

bool asArraysInsert(int* pArrRefStart, int* pArrRefEnd, int* pArrOtherStart, int* pArrOtherEnd, Order order, int valRef,
                    int valOther);

bool asArraysInsert(float* pArrRefStart, float* pArrRefEnd, float* pArrOtherStart, float* pArrOtherEnd, Order order,
                    float valRef, float valOther);

bool asArraysInsert(double* pArrRefStart, double* pArrRefEnd, double* pArrOtherStart, double* pArrOtherEnd, Order order,
                    double valRef, double valOther);

template <class T>
bool asArraysInsertT(T* pArrRefStart, T* pArrRefEnd, T* pArrOtherStart, T* pArrOtherEnd, Order order, T valRef,
                     T valOther);

bool asSortArray(int* pArrRefStart, int* pArrRefEnd, Order order);

bool asSortArray(float* pArrRefStart, float* pArrRefEnd, Order order);

bool asSortArray(double* pArrRefStart, double* pArrRefEnd, Order order);

template <class T>
bool asSortArrayT(T* pArrRefStart, T* pArrRefEnd, Order order);

bool asSortArrays(int* pArrRefStart, int* pArrRefEnd, int* pArrOtherStart, int* pArrOtherEnd, Order order);

bool asSortArrays(float* pArrRefStart, float* pArrRefEnd, float* pArrOtherStart, float* pArrOtherEnd, Order order);

bool asSortArrays(double* pArrRefStart, double* pArrRefEnd, double* pArrOtherStart, double* pArrOtherEnd, Order order);

template <class T>
bool asSortArraysT(T* pArrRefStart, T* pArrRefEnd, T* pArrOtherStart, T* pArrOtherEnd, Order order);

template <class T>
void asQuickSort(T* pArr, int low, int high, Order order);

template <class T>
void asQuickSortMulti(T* pArr, T* pArrOther, int low, int high, Order order);

vf asExtractVectorFrom(const wxString& data);

wxString asVectorToString(const vf& data);

wxString asExtractParamValueAndCut(wxString& str, const wxString& tag);

#endif
