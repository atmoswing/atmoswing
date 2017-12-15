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

#include <asTools.h>
#include <asFileAscii.h>
#include "gtest/gtest.h"


TEST(Tools, IsRoundFloatTrue)
{
    float value = 2;
    EXPECT_TRUE(asTools::IsRound(value));
}

TEST(Tools, IsRoundFloatFalse)
{
    float value = 2.0001f;
    EXPECT_FALSE(asTools::IsRound(value));
}

TEST(Tools, IsRoundDoubleFalse)
{
    double value = 2.00000001;
    EXPECT_FALSE(asTools::IsRound(value));
}

TEST(Tools, RoundFloatUp)
{
    float value = 2.5;
    float result = asTools::Round(value);
    EXPECT_EQ(3, result);
}

TEST(Tools, RoundDoubleUp)
{
    double value = 2.5;
    double result = asTools::Round(value);
    EXPECT_EQ(3, result);
}

TEST(Tools, RoundFloatDown)
{
    float value = 2.49999f;
    float result = asTools::Round(value);
    EXPECT_EQ(2, result);
}

TEST(Tools, RoundDoubleDown)
{
    double value = 2.499999999999;
    double result = asTools::Round(value);
    EXPECT_EQ(2, result);
}

TEST(Tools, RoundNegativeFloatDown)
{
    float value = -2.5f;
    float result = asTools::Round(value);
    EXPECT_EQ(-3, result);
}

TEST(Tools, RoundNegativeDoubleDown)
{
    double value = -2.5;
    double result = asTools::Round(value);
    EXPECT_EQ(-3, result);
}

TEST(Tools, RoundNegativeFloatUp)
{
    float value = -2.49999f;
    float result = asTools::Round(value);
    EXPECT_EQ(-2, result);
}

TEST(Tools, RoundNegativeDoubleUp)
{
    double value = -2.499999999999;
    double result = asTools::Round(value);
    EXPECT_EQ(-2, result);
}

TEST(Tools, RoundSmallFloatUp)
{
    float value = 0.5;
    float result = asTools::Round(value);
    EXPECT_EQ(1, result);
}

TEST(Tools, RoundSmallDoubleUp)
{
    double value = 0.5;
    double result = asTools::Round(value);
    EXPECT_EQ(1, result);
}

TEST(Tools, RoundSmallFloatDown)
{
    float value = 0.49999f;
    float result = asTools::Round(value);
    EXPECT_EQ(0, result);
}

TEST(Tools, RoundSmallDoubleDown)
{
    double value = 0.499999999999;
    double result = asTools::Round(value);
    EXPECT_EQ(0, result);
}

TEST(Tools, RoundSmallNegativeFloatDown)
{
    float value = -0.5f;
    float result = asTools::Round(value);
    EXPECT_EQ(-1, result);
}

TEST(Tools, RoundSmallNegativeDoubleDown)
{
    double value = -0.5;
    double result = asTools::Round(value);
    EXPECT_EQ(-1, result);
}

TEST(Tools, RoundSmallNegativeFloatUp)
{
    float value = -0.49999f;
    float result = asTools::Round(value);
    EXPECT_EQ(0, result);
}

TEST(Tools, RoundSmallNegativeDoubleUp)
{
    double value = -0.499999999999;
    double result = asTools::Round(value);
    EXPECT_EQ(0, result);
}

TEST(Tools, IsNaNOne)
{
    int value = 1;
    EXPECT_FALSE(asTools::IsNaN((float) value));
}

TEST(Tools, IsNaNZero)
{
    int value = 0;
    EXPECT_FALSE(asTools::IsNaN((float) value));
}

TEST(Tools, IsNaNFloatTrue)
{
    float value = NaNf;
    EXPECT_TRUE(asTools::IsNaN(value));
}

TEST(Tools, IsNaNDoubleTrue)
{
    double value = NaNd;
    EXPECT_TRUE(asTools::IsNaN(value));
}

TEST(Tools, IsInfFloatFalse)
{
    float value = -2151;
    EXPECT_FALSE(asTools::IsInf(value));
}

TEST(Tools, IsInfDoubleFalse)
{
    double value = -2151;
    EXPECT_FALSE(asTools::IsInf(value));
}

TEST(Tools, IsInfLongDoubleFalse)
{
    long double value = -2151;
    EXPECT_FALSE(asTools::IsInf(value));
}

TEST(Tools, IsInfFloatTrue)
{
    float value = Inff;
    EXPECT_TRUE(asTools::IsInf(value));
}

TEST(Tools, IsInfDoubleTrue)
{
    double value = Infd;
    EXPECT_TRUE(asTools::IsInf(value));
}

TEST(Tools, IsInfLongDoubleTrue)
{
    long double value = Infld;
    EXPECT_TRUE(asTools::IsInf(value));
}

TEST(Tools, CountNotNaNFloat)
{
    float array[] = {0.3465f, 1.345f, 2.76f, 3.69f, 5.58f, NaNf, 8.34f, 9.75f, 10.0f, NaNf};
    float *pVectStart = &array[0];
    float *pVectEnd = &array[9];
    int result = asTools::CountNotNaN(pVectStart, pVectEnd);
    EXPECT_EQ(8, result);
}

TEST(Tools, CountNotNaNDouble)
{
    double array[] = {0.3465, 1.345, 2.76, 3.69, 5.58, NaNd, 8.34, 9.75, 10, NaNd};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    int result = asTools::CountNotNaN(pVectStart, pVectEnd);
    EXPECT_EQ(8, result);
}

TEST(Tools, FindMinInt)
{
    int array[] = {0, 0, -1, 1, 2, 3, 5, 0, 8, 9, 10, 0};
    int result = asTools::MinArray(&array[0], &array[11]);
    EXPECT_EQ(-1, result);
}

TEST(Tools, FindMinFloat)
{
    float array[] = {NaNf, NaNf, 0.3465f, 1.345f, 2.76f, 3.69f, 5.58f, NaNf, 8.34f, 9.75f, 10.0f, NaNf};
    float result = asTools::MinArray(&array[0], &array[11]);
    EXPECT_FLOAT_EQ(0.3465, result);
}

TEST(Tools, FindMinDouble)
{
    double array[] = {NaNd, NaNd, 0.3465, 1.345, 2.76, 3.69, 5.58, NaNd, 8.34, 9.75, 10, NaNd};
    double result = asTools::MinArray(&array[0], &array[11]);
    EXPECT_DOUBLE_EQ(0.3465, result);
}

TEST(Tools, FindMaxInt)
{
    int array[] = {0, 0, -1, 1, 2, 3, 5, 0, 8, 9, 10, 0};
    int result = asTools::MaxArray(&array[0], &array[11]);
    EXPECT_EQ(10, result);
}

TEST(Tools, FindMaxFloat)
{
    float array[] = {NaNf, NaNf, 0.3465f, 1.345f, 2.76f, 3.69f, 5.58f, NaNf, 8.34f, 9.75f, 10.12f,
                     NaNf};
    float result = asTools::MaxArray(&array[0], &array[11]);
    EXPECT_FLOAT_EQ(10.12, result);
}

TEST(Tools, FindMaxDouble)
{
    double array[] = {NaNd, NaNd, 0.3465, 1.345, 2.76, 3.69, 5.58, NaNd, 8.34, 9.75, 10.12, NaNd};
    double result = asTools::MaxArray(&array[0], &array[11]);
    EXPECT_DOUBLE_EQ(10.12, result);
}

TEST(Tools, FindMinStepInt)
{
    int array[] = {0, 10, 0, 1, 3, 5, 0, 8, 2, 9, 0, 0};
    int result = asTools::MinArrayStep(&array[0], &array[11], 0);
    EXPECT_EQ(1, result);
}

TEST(Tools, FindMinStepFloat)
{
    float array[] = {NaNf, 10.12f, NaNf, 1.345f, 1.345f, 3.69f, 5.58f, NaNf, 8.34f, 2.76f, 9.75f, 0.3465f,
                     NaNf};
    float result = asTools::MinArrayStep(&array[0], &array[11], 0.0001f);
    EXPECT_FLOAT_EQ(0.37, result);
}

TEST(Tools, FindMinStepDouble)
{
    double array[] = {NaNd, 10.12, NaNd, 1.345, 1.345, 3.69, 5.58, NaNd, 8.34, 2.76, 9.75, 0.3465,
                      NaNd};
    double result = asTools::MinArrayStep(&array[0], &array[11]);
    EXPECT_FLOAT_EQ(0.37, result);
}

TEST(Tools, ExtractUniqueValuesInt)
{
    int array[] = {0, 10, 0, 1, 3, 5, 1, 8, 2, 9, 0, 9};;
    a1i result(asTools::ExtractUniqueValues(&array[0], &array[11]));
    EXPECT_EQ(0, result[0]);
    EXPECT_EQ(1, result[1]);
    EXPECT_EQ(2, result[2]);
    EXPECT_EQ(3, result[3]);
    EXPECT_EQ(5, result[4]);
    EXPECT_EQ(8, result[5]);
    EXPECT_EQ(9, result[6]);
    EXPECT_EQ(10, result[7]);
}

TEST(Tools, ExtractUniqueValuesFloat)
{
    float array[] = {NaNf, 10.12f, NaNf, 1.345f, 1.345f, 3.69f, 5.58f, NaNf, 8.34f, 2.76f, 9.75f, 0.3465f,
                     NaNf};
    a1f result(asTools::ExtractUniqueValues(&array[0], &array[11], 0.0001f));
    EXPECT_FLOAT_EQ(0.3465, result[0]);
    EXPECT_FLOAT_EQ(1.345, result[1]);
    EXPECT_FLOAT_EQ(2.76, result[2]);
    EXPECT_FLOAT_EQ(3.69, result[3]);
    EXPECT_FLOAT_EQ(5.58, result[4]);
    EXPECT_FLOAT_EQ(8.34, result[5]);
    EXPECT_FLOAT_EQ(9.75, result[6]);
    EXPECT_FLOAT_EQ(10.12, result[7]);
}

TEST(Tools, ExtractUniqueValuesDouble)
{
    double array[] = {NaNd, 10.12, NaNd, 1.345, 1.345, 3.69, 5.58, NaNd, 8.34, 2.76, 9.75, 0.3465,
                      NaNd};
    a1d result(asTools::ExtractUniqueValues(&array[0], &array[11]));
    EXPECT_DOUBLE_EQ(0.3465, result[0]);
    EXPECT_DOUBLE_EQ(1.345, result[1]);
    EXPECT_DOUBLE_EQ(2.76, result[2]);
    EXPECT_DOUBLE_EQ(3.69, result[3]);
    EXPECT_DOUBLE_EQ(5.58, result[4]);
    EXPECT_DOUBLE_EQ(8.34, result[5]);
    EXPECT_DOUBLE_EQ(9.75, result[6]);
    EXPECT_DOUBLE_EQ(10.12, result[7]);
}

TEST(Tools, SortedArraySearchIntAscFirst)
{
    int array[] = {0, 1, 2, 3, 5, 7, 8, 9, 10, 100};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int targetvalue = 0;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchIntAscMid)
{
    int array[] = {0, 1, 2, 3, 5, 7, 8, 9, 10, 100};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int targetvalue = 8;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(6, result);
}

TEST(Tools, SortedArraySearchIntAscLast)
{
    int array[] = {0, 1, 2, 3, 5, 7, 8, 9, 10, 100};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int targetvalue = 100;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, result);
}

TEST(Tools, SortedArraySearchIntAscOutofRange)
{
    int array[] = {0, 1, 2, 3, 5, 7, 8, 9, 10, 100};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int targetvalue = 1000;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 0, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchIntAscNotFound)
{
    int array[] = {0, 1, 2, 3, 5, 7, 8, 9, 10, 100};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int targetvalue = 6;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 0, asHIDE_WARNINGS);
    EXPECT_EQ(asNOT_FOUND, result);
}

TEST(Tools, SortedArraySearchIntAscTolerFirst)
{
    int array[] = {0, 3, 4, 5, 6, 7, 8, 9, 10, 100};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int targetvalue = -1;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchIntAscTolerFirstLimit)
{
    int array[] = {0, 1, 2, 3, 5, 7, 8, 9, 10, 100};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int targetvalue = -1;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchIntAscTolerMid)
{
    int array[] = {0, 1, 2, 3, 5, 7, 8, 9, 10, 100};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int targetvalue = 11;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(8, result);
}

TEST(Tools, SortedArraySearchIntAscTolerMidLimit)
{
    int array[] = {0, 1, 2, 3, 5, 7, 8, 9, 10, 100};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int targetvalue = 11;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(8, result);
}

TEST(Tools, SortedArraySearchIntAscTolerLast)
{
    int array[] = {0, 1, 2, 3, 5, 7, 8, 9, 10, 100};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int targetvalue = 102;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 3);
    EXPECT_EQ(9, result);
}

TEST(Tools, SortedArraySearchIntAscTolerLastLimit)
{
    int array[] = {0, 1, 2, 3, 5, 7, 8, 9, 10, 100};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int targetvalue = 102;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 2);
    EXPECT_EQ(9, result);
}

TEST(Tools, SortedArraySearchIntDescFirst)
{
    int array[] = {100, 10, 9, 8, 7, 5, 3, 2, 1, 0};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int targetvalue = 100;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchIntDescMid)
{
    int array[] = {100, 10, 9, 8, 7, 5, 3, 2, 1, 0};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int targetvalue = 8;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(3, result);
}

TEST(Tools, SortedArraySearchIntDescLast)
{
    int array[] = {100, 10, 9, 8, 7, 5, 3, 2, 1, 0};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int targetvalue = 0;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, result);
}

TEST(Tools, SortedArraySearchIntDescOutofRange)
{
    int array[] = {100, 10, 9, 8, 7, 5, 3, 2, 1, 0};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int targetvalue = -1;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 0, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchIntDescNotFound)
{
    int array[] = {100, 10, 9, 8, 7, 5, 3, 2, 1, 0};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int targetvalue = 6;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 0, asHIDE_WARNINGS);
    EXPECT_EQ(asNOT_FOUND, result);
}

TEST(Tools, SortedArraySearchIntDescTolerFirst)
{
    int array[] = {100, 10, 9, 8, 7, 5, 3, 2, 1, 0};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int targetvalue = -1;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(9, result);
}

TEST(Tools, SortedArraySearchIntDescTolerFirstLimit)
{
    int array[] = {100, 10, 9, 8, 7, 5, 3, 2, 1, 0};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int targetvalue = -1;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(9, result);
}

TEST(Tools, SortedArraySearchIntDescTolerMid)
{
    int array[] = {100, 10, 9, 8, 7, 5, 3, 2, 1, 0};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int targetvalue = 11;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 3);
    EXPECT_EQ(1, result);
}

TEST(Tools, SortedArraySearchIntDescTolerMidLimit)
{
    int array[] = {100, 10, 9, 8, 7, 5, 3, 2, 1, 0};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int targetvalue = 11;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(1, result);
}

TEST(Tools, SortedArraySearchIntDescTolerLast)
{
    int array[] = {100, 10, 9, 8, 7, 5, 3, 2, 1, 0};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int targetvalue = 102;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 3);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchIntDescTolerLastLimit)
{
    int array[] = {100, 10, 9, 8, 7, 5, 3, 2, 1, 0};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int targetvalue = 102;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 2);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchIntUniqueVal)
{
    int array[] = {9};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[0];
    int targetvalue = 9;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchIntUniqueValToler)
{
    int array[] = {9};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[0];
    int targetvalue = 8;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchIntUniqueValOutofRange)
{
    int array[] = {9};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[0];
    int targetvalue = 11;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchIntArraySameVal)
{
    int array[] = {9, 9, 9, 9};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[3];
    int targetvalue = 9;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchIntArraySameValTolerDown)
{
    int array[] = {9, 9, 9, 9};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[3];
    int targetvalue = 8;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchIntArraySameValTolerUp)
{
    int array[] = {9, 9, 9, 9};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[3];
    int targetvalue = 10;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchIntArraySameValOutofRange)
{
    int array[] = {9, 9, 9, 9};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[3];
    int targetvalue = 11;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchDoubleAscFirst)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 0.354;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchDoubleAscMid)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 5.75;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(5, result);
}

TEST(Tools, SortedArraySearchDoubleAscLast)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 100;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, result);
}

TEST(Tools, SortedArraySearchDoubleAscOutofRange)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 1000;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 0.0, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchDoubleAscNotFound)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 6;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 0.0, asHIDE_WARNINGS);
    EXPECT_EQ(asNOT_FOUND, result);
}

TEST(Tools, SortedArraySearchDoubleAscTolerFirst)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = -1.12;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 3);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchDoubleAscTolerFirstLimit)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = -1;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1.354);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchDoubleAscTolerFirstOutLimit)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = -1;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1.353, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchDoubleAscTolerMid)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 11;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(8, result);
}

TEST(Tools, SortedArraySearchDoubleAscTolerMidLimit)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 11.45;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(8, result);
}

TEST(Tools, SortedArraySearchDoubleAscTolerMidLimitOut)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 11.45;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 0.99, asHIDE_WARNINGS);
    EXPECT_EQ(asNOT_FOUND, result);
}

TEST(Tools, SortedArraySearchDoubleAscTolerLast)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 102.21;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 3);
    EXPECT_EQ(9, result);
}

TEST(Tools, SortedArraySearchDoubleAscTolerLastLimit)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 101.5;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1.5);
    EXPECT_EQ(9, result);
}

TEST(Tools, SortedArraySearchDoubleAscTolerLastOutLimit)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 101.5;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1.499, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchDoubleDescFirst)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 100;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchDoubleDescMid)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 5.75;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(4, result);
}

TEST(Tools, SortedArraySearchDoubleDescLast)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 0.354;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, result);
}

TEST(Tools, SortedArraySearchDoubleDescOutofRange)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = -1.23;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 0.0, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchDoubleDescNotFound)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 6.2;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 0.0, asHIDE_WARNINGS);
    EXPECT_EQ(asNOT_FOUND, result);
}

TEST(Tools, SortedArraySearchDoubleDescTolerFirst)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = -1;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 2);
    EXPECT_EQ(9, result);
}

TEST(Tools, SortedArraySearchDoubleDescTolerFirstLimit)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = -1;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1.354);
    EXPECT_EQ(9, result);
}

TEST(Tools, SortedArraySearchDoubleDescTolerFirstOutLimit)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = -1;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1.353, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchDoubleDescTolerMid)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 11.23;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 3);
    EXPECT_EQ(1, result);
}

TEST(Tools, SortedArraySearchDoubleDescTolerMidLimit)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 11.45;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(1, result);
}

TEST(Tools, SortedArraySearchDoubleDescTolerMidOutLimit)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 11.45;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 0.999, asHIDE_WARNINGS);
    EXPECT_EQ(asNOT_FOUND, result);
}

TEST(Tools, SortedArraySearchDoubleDescTolerLast)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 102.42;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 3);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchDoubleDescTolerLastLimit)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 102.21;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 2.21);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchDoubleDescTolerLastOutLimit)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 102.21;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 2.2, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchDoubleUniqueVal)
{
    double array[] = {9.3401};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[0];
    double targetvalue = 9.3401;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchDoubleUniqueValToler)
{
    double array[] = {9.3401};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[0];
    double targetvalue = 8;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1.3401);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchDoubleUniqueValOutofRange)
{
    double array[] = {9.3401};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[0];
    double targetvalue = 11;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchDoubleArraySameVal)
{
    double array[] = {9.34, 9.34, 9.34, 9.34};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[3];
    double targetvalue = 9.34;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchDoubleArraySameValTolerDown)
{
    double array[] = {9.34, 9.34, 9.34, 9.34};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[3];
    double targetvalue = 8;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1.5);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchDoubleArraySameValTolerUp)
{
    double array[] = {9.34, 9.34, 9.34, 9.34};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[3];
    double targetvalue = 10;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchDoubleArraySameValOutofRange)
{
    double array[] = {9.34, 9.34, 9.34, 9.34};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[3];
    double targetvalue = 11;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchRoleOfToleranceInSearch)
{
    a1d values;
    values.resize(94);
    values <<
    -88.542, -86.653, -84.753, -82.851, -80.947, -79.043, -77.139, -75.235, -73.331, -71.426, -69.522, -67.617, -65.713, -63.808, -61.903, -59.999, -58.094, -56.189, -54.285, -52.380, -50.475, -48.571, -46.666, -44.761, -42.856, -40.952, -39.047, -37.142, -35.238, -33.333, -31.428, -29.523, -27.619, -25.714, -23.809, -21.904, -20.000, -18.095, -16.190, -14.286, -12.381, -10.476, -08.571, -06.667, -04.762, -02.857, -00.952, 00.952, 02.857, 04.762, 06.667, 08.571, 10.476, 12.381, 14.286, 16.190, 18.095, 20.000, 21.904, 23.809, 25.714, 27.619, 29.523, 31.428, 33.333, 35.238, 37.142, 39.047, 40.952, 42.856, 44.761, 46.666, 48.571, 50.475, 52.380, 54.285, 56.189, 58.094, 59.999, 61.903, 63.808, 65.713, 67.617, 69.522, 71.426, 73.331, 75.235, 77.139, 79.043, 80.947, 82.851, 84.753, 86.653, 88.542;
    double *pVectStart = &values[0];
    double *pVectEnd = &values[93];
    double targetvalue = 29.523;
    int result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 0.01);
    EXPECT_EQ(62, result);
}

TEST(Tools, SortedArraySearchClosestDoubleAscFirst)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 0.394;
    int result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchClosestDoubleAscMid)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 5.55;
    int result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(5, result);
}

TEST(Tools, SortedArraySearchClosestDoubleAscLast)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 99.9;
    int result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, result);
}

TEST(Tools, SortedArraySearchClosestDoubleAscOutofRange)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 1000;
    int result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchClosestDoubleDescFirst)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 100;
    int result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchClosestDoubleDescMid)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 5.55;
    int result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(4, result);
}

TEST(Tools, SortedArraySearchClosestDoubleDescLast)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 0.354;
    int result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, result);
}

TEST(Tools, SortedArraySearchClosestDoubleDescOutofRange)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = -1.23;
    int result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchClosestDoubleUniqueVal)
{
    double array[] = {9.3401};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[0];
    double targetvalue = 9.3401;
    int result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchClosestDoubleUniqueValOutofRange)
{
    double array[] = {9.3401};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[0];
    double targetvalue = 11;
    int result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchClosestDoubleArraySameVal)
{
    double array[] = {9.34, 9.34, 9.34, 9.34};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[3];
    double targetvalue = 9.34;
    int result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchClosestDoubleArraySameValOutofRange)
{
    double array[] = {9.34, 9.34, 9.34, 9.34};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[3];
    double targetvalue = 11;
    int result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchFloorDoubleAscFirst)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 1.394;
    int result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchFloorDoubleAscMid)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 5.55;
    int result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(4, result);
}

TEST(Tools, SortedArraySearchFloorDoubleAscLast)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 99.9;
    int result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(8, result);
}

TEST(Tools, SortedArraySearchFloorDoubleAscLastExact)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 100;
    int result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, result);
}

TEST(Tools, SortedArraySearchFloorDoubleAscOutofRange)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 1000;
    int result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue, 0, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchFloorDoubleDescFirst)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 40.12;
    int result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(1, result);
}

TEST(Tools, SortedArraySearchFloorDoubleDescMid)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 5.55;
    int result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(5, result);
}

TEST(Tools, SortedArraySearchFloorDoubleDescLast)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 0.360;
    int result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, result);
}

TEST(Tools, SortedArraySearchFloorDoubleDescLastExact)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 0.354;
    int result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, result);
}

TEST(Tools, SortedArraySearchFloorDoubleDescOutofRange)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = -1.23;
    int result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue, 0, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchFloorDoubleUniqueVal)
{
    double array[] = {9.3401};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[0];
    double targetvalue = 9.3401;
    int result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchFloorDoubleUniqueValOutofRange)
{
    double array[] = {9.3401};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[0];
    double targetvalue = 11;
    int result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue, 0, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchFloorDoubleArraySameVal)
{
    double array[] = {9.34, 9.34, 9.34, 9.34};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[3];
    double targetvalue = 9.34;
    int result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchFloorDoubleArraySameValOutofRange)
{
    double array[] = {9.34, 9.34, 9.34, 9.34};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[3];
    double targetvalue = 11;
    int result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue, 0, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchCeilDoubleAscFirst)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 0.354;
    int result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchCeilDoubleAscMid)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 5.55;
    int result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(5, result);
}

TEST(Tools, SortedArraySearchCeilDoubleAscLast)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 10.46;
    int result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, result);
}

TEST(Tools, SortedArraySearchCeilDoubleAscLastExact)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 100;
    int result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, result);
}

TEST(Tools, SortedArraySearchCeilDoubleAscOutofRange)
{
    double array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 1000;
    int result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue, 0, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchCeilDoubleDescFirst)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 40.12;
    int result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchCeilDoubleDescMid)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 5.55;
    int result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(4, result);
}

TEST(Tools, SortedArraySearchCeilDoubleDescLast)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 0.360;
    int result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(8, result);
}

TEST(Tools, SortedArraySearchCeilDoubleDescLastExact)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = 0.354;
    int result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, result);
}

TEST(Tools, SortedArraySearchCeilDoubleDescOutofRange)
{
    double array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double targetvalue = -1.23;
    int result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue, 0, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchCeilDoubleUniqueVal)
{
    double array[] = {9.3401};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[0];
    double targetvalue = 9.3401;
    int result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchCeilDoubleUniqueValOutofRange)
{
    double array[] = {9.3401};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[0];
    double targetvalue = 11;
    int result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue, 0, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArraySearchCeilDoubleArraySameVal)
{
    double array[] = {9.34, 9.34, 9.34, 9.34};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[3];
    double targetvalue = 9.34;
    int result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, result);
}

TEST(Tools, SortedArraySearchCeilDoubleArraySameValOutofRange)
{
    double array[] = {9.34, 9.34, 9.34, 9.34};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[3];
    double targetvalue = 11;
    int result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue, 0, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, result);
}

TEST(Tools, SortedArrayInsertIntAscFirst)
{
    int array[] = {2, 3, 4, 6, 9, 17, 18, 20, 40, 100};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int newvalue = 1;
    EXPECT_TRUE(asTools::SortedArrayInsert(pVectStart, pVectEnd, Asc, newvalue));
    int arrayResults[] = {1, 2, 3, 4, 6, 9, 17, 18, 20, 40};
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(arrayResults[i], array[i]);
    }
}

TEST(Tools, SortedArrayInsertIntAscFirstNeg)
{
    int array[] = {0, 1, 4, 6, 9, 17, 18, 20, 40, 100};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int newvalue = -2;
    EXPECT_TRUE(asTools::SortedArrayInsert(pVectStart, pVectEnd, Asc, newvalue));
    int arrayResults[] = {-2, 0, 1, 4, 6, 9, 17, 18, 20, 40};
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(arrayResults[i], array[i]);
    }
}

TEST(Tools, SortedArrayInsertIntAscMid)
{
    int array[] = {0, 1, 4, 6, 9, 17, 18, 20, 40, 100};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int newvalue = 8;
    EXPECT_TRUE(asTools::SortedArrayInsert(pVectStart, pVectEnd, Asc, newvalue));
    int arrayResults[] = {0, 1, 4, 6, 8, 9, 17, 18, 20, 40};
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(arrayResults[i], array[i]);
    }
}

TEST(Tools, SortedArrayInsertIntAscEnd)
{
    int array[] = {0, 1, 4, 6, 9, 17, 18, 20, 40, 100};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int newvalue = 90;
    EXPECT_TRUE(asTools::SortedArrayInsert(pVectStart, pVectEnd, Asc, newvalue));
    int arrayResults[] = {0, 1, 4, 6, 9, 17, 18, 20, 40, 90};
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(arrayResults[i], array[i]);
    }
}

TEST(Tools, SortedArrayInsertFloatAscMid)
{
    float array[] = {0.134631f, 1.13613f, 4.346f, 6.835f, 9.1357f, 17.23456f, 18.2364f, 20.75f, 40.54f, 100.235f};
    float *pVectStart = &array[0];
    float *pVectEnd = &array[9];
    float newvalue = 9.105646f;
    EXPECT_TRUE(asTools::SortedArrayInsert(pVectStart, pVectEnd, Asc, newvalue));
    float arrayResults[] = {0.134631f, 1.13613f, 4.346f, 6.835f, 9.105646f, 9.1357f, 17.23456f, 18.2364f, 20.75f,
                            40.54f};
    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(arrayResults[i], array[i]);
    }
}

TEST(Tools, SortedArrayInsertDoubleAscMid)
{
    double array[] = {0.134631, 1.13613, 4.346, 6.835, 9.1357, 17.23456, 18.2364, 20.75, 40.54, 100.235};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double newvalue = 9.105646;
    EXPECT_TRUE(asTools::SortedArrayInsert(pVectStart, pVectEnd, Asc, newvalue));
    double arrayResults[] = {0.134631, 1.13613, 4.346, 6.835, 9.105646, 9.1357, 17.23456, 18.2364, 20.75, 40.54};
    for (int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(arrayResults[i], array[i]);
    }
}

TEST(Tools, SortedArrayInsertIntDescFirst)
{
    int array[] = {100, 40, 20, 18, 17, 9, 6, 4, 3, 2};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int newvalue = 101;
    EXPECT_TRUE(asTools::SortedArrayInsert(pVectStart, pVectEnd, Desc, newvalue));
    int arrayResults[] = {101, 100, 40, 20, 18, 17, 9, 6, 4, 3};
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(arrayResults[i], array[i]);
    }
}

TEST(Tools, SortedArrayInsertIntDescMid)
{
    int array[] = {100, 40, 20, 18, 17, 9, 6, 4, 3, 2};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int newvalue = 8;
    EXPECT_TRUE(asTools::SortedArrayInsert(pVectStart, pVectEnd, Desc, newvalue));
    int arrayResults[] = {100, 40, 20, 18, 17, 9, 8, 6, 4, 3};
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(arrayResults[i], array[i]);
    }
}

TEST(Tools, SortedArrayInsertIntDescEnd)
{
    int array[] = {100, 40, 20, 18, 17, 9, 6, 4, 3, 2};
    int *pVectStart = &array[0];
    int *pVectEnd = &array[9];
    int newvalue = 3;
    EXPECT_TRUE(asTools::SortedArrayInsert(pVectStart, pVectEnd, Desc, newvalue));
    int arrayResults[] = {100, 40, 20, 18, 17, 9, 6, 4, 3, 3};
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(arrayResults[i], array[i]);
    }
}

TEST(Tools, SortedArrayInsertFloatDescMid)
{
    float array[] = {100.1345f, 40.2345f, 20.2345f, 18.567f, 17.2134f, 9.67f, 6.1346f, 4.7135f, 3.1f, 2.2345f};
    float *pVectStart = &array[0];
    float *pVectEnd = &array[9];
    float newvalue = 9.105646f;
    EXPECT_TRUE(asTools::SortedArrayInsert(pVectStart, pVectEnd, Desc, newvalue));
    float arrayResults[] = {100.1345f, 40.2345f, 20.2345f, 18.567f, 17.2134f, 9.67f, 9.105646f, 6.1346f, 4.7135f, 3.1f};
    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(arrayResults[i], array[i]);
    }
}

TEST(Tools, SortedArraysInsertIntAscFirst)
{
    int arrayRef[] = {2, 3, 4, 6, 9, 17, 18, 20, 40, 100};
    int arrayOther[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int newvalueRef = 1;
    int newvalueOther = 11;
    EXPECT_TRUE(
            asTools::SortedArraysInsert(&arrayRef[0], &arrayRef[9], &arrayOther[0], &arrayOther[9], Asc, newvalueRef,
                                        newvalueOther));
    int arrayResultsRef[] = {1, 2, 3, 4, 6, 9, 17, 18, 20, 40};
    int arrayResultsOther[] = {11, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(arrayResultsRef[i], arrayRef[i]);
        EXPECT_EQ(arrayResultsOther[i], arrayOther[i]);
    }
}

TEST(Tools, SortedArraysInsertIntAscMid)
{
    int arrayRef[] = {2, 3, 4, 6, 9, 17, 18, 20, 40, 100};
    int arrayOther[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int newvalueRef = 11;
    int newvalueOther = 11;
    EXPECT_TRUE(
            asTools::SortedArraysInsert(&arrayRef[0], &arrayRef[9], &arrayOther[0], &arrayOther[9], Asc, newvalueRef,
                                        newvalueOther));
    int arrayResultsRef[] = {2, 3, 4, 6, 9, 11, 17, 18, 20, 40};
    int arrayResultsOther[] = {1, 2, 3, 4, 5, 11, 6, 7, 8, 9};
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(arrayResultsRef[i], arrayRef[i]);
        EXPECT_EQ(arrayResultsOther[i], arrayOther[i]);
    }
}

TEST(Tools, SortedArraysInsertIntAscLast)
{
    int arrayRef[] = {2, 3, 4, 6, 9, 17, 18, 20, 40, 100};
    int arrayOther[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int newvalueRef = 99;
    int newvalueOther = 11;
    EXPECT_TRUE(
            asTools::SortedArraysInsert(&arrayRef[0], &arrayRef[9], &arrayOther[0], &arrayOther[9], Asc, newvalueRef,
                                        newvalueOther));
    int arrayResultsRef[] = {2, 3, 4, 6, 9, 17, 18, 20, 40, 99};
    int arrayResultsOther[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11};
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(arrayResultsRef[i], arrayRef[i]);
        EXPECT_EQ(arrayResultsOther[i], arrayOther[i]);
    }
}

TEST(Tools, SortedArraysInsertFloatAscMid)
{
    float arrayRef[] = {2.254f, 3.345f, 4.625f, 6.47f, 9.7f, 17.245f, 18.0f, 20.67f, 40.25f, 100.25f};
    float arrayOther[] = {1.7f, 2.4f, 3.346f, 4.7f, 5.1346f, 6.715f, 7.1346f, 8.1357f, 9.1346f, 10.715f};
    float newvalueRef = 11.175f;
    float newvalueOther = 11.1346f;
    EXPECT_TRUE(
            asTools::SortedArraysInsert(&arrayRef[0], &arrayRef[9], &arrayOther[0], &arrayOther[9], Asc, newvalueRef,
                                        newvalueOther));
    float arrayResultsRef[] = {2.254f, 3.345f, 4.625f, 6.47f, 9.7f, 11.175f, 17.245f, 18.0f, 20.67f, 40.25f};
    float arrayResultsOther[] = {1.7f, 2.4f, 3.346f, 4.7f, 5.1346f, 11.1346f, 6.715f, 7.1346f, 8.1357f, 9.1346f};
    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(arrayResultsRef[i], arrayRef[i]);
        EXPECT_FLOAT_EQ(arrayResultsOther[i], arrayOther[i]);
    }
}

TEST(Tools, SortedArraysInsertDoubleAscMid)
{
    double arrayRef[] = {2.254, 3.345, 4.625, 6.47, 9.7, 17.245, 18, 20.67, 40.25, 100.25};
    double arrayOther[] = {1.7, 2.4, 3.346, 4.7, 5.1346, 6.715, 7.1346, 8.1357, 9.1346, 10.715};
    double newvalueRef = 11.175;
    double newvalueOther = 11.1346;
    EXPECT_TRUE(
            asTools::SortedArraysInsert(&arrayRef[0], &arrayRef[9], &arrayOther[0], &arrayOther[9], Asc, newvalueRef,
                                        newvalueOther));
    double arrayResultsRef[] = {2.254, 3.345, 4.625, 6.47, 9.7, 11.175, 17.245, 18, 20.67, 40.25};
    double arrayResultsOther[] = {1.7, 2.4, 3.346, 4.7, 5.1346, 11.1346, 6.715, 7.1346, 8.1357, 9.1346};
    for (int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(arrayResultsRef[i], arrayRef[i]);
        EXPECT_DOUBLE_EQ(arrayResultsOther[i], arrayOther[i]);
    }
}

TEST(Tools, SortArrayAsc)
{
    double arrayRef[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double array[] = {9.65, 2.7, 0.354, 100, 5.75, 1.932, 8.2, 10.45, 5.021, 3.56};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    EXPECT_TRUE(asTools::SortArray(pVectStart, pVectEnd, Asc));
    for (int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(arrayRef[i], array[i]);
    }
}

TEST(Tools, SortArrayDesc)
{
    double arrayRef[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double array[] = {9.65, 2.7, 0.354, 100, 5.75, 1.932, 8.2, 10.45, 5.021, 3.56};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    EXPECT_TRUE(asTools::SortArray(pVectStart, pVectEnd, Desc));
    for (int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(arrayRef[i], array[i]);
    }
}

TEST(Tools, SortArraysAsc)
{
    double arrayRef[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double array[] = {9.65, 2.7, 0.354, 100, 5.75, 1.932, 8.2, 10.45, 5.021, 3.56};
    double arrayOtherRef[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double arrayOther[] = {8, 3, 1, 10, 6, 2, 7, 9, 5, 4};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double *pVectStartOther = &arrayOther[0];
    double *pVectEndOther = &arrayOther[9];
    EXPECT_TRUE(asTools::SortArrays(pVectStart, pVectEnd, pVectStartOther, pVectEndOther, Asc));
    for (int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(arrayRef[i], array[i]);
        EXPECT_DOUBLE_EQ(arrayOtherRef[i], arrayOther[i]);
    }
}

TEST(Tools, SortArraysDesc)
{
    double arrayRef[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double array[] = {9.65, 2.7, 0.354, 100, 5.75, 1.932, 8.2, 10.45, 5.021, 3.56};
    double arrayOtherRef[] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    double arrayOther[] = {8, 3, 1, 10, 6, 2, 7, 9, 5, 4};
    double *pVectStart = &array[0];
    double *pVectEnd = &array[9];
    double *pVectStartOther = &arrayOther[0];
    double *pVectEndOther = &arrayOther[9];
    EXPECT_TRUE(asTools::SortArrays(pVectStart, pVectEnd, pVectStartOther, pVectEndOther, Desc));
    for (int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(arrayRef[i], array[i]);
        EXPECT_DOUBLE_EQ(arrayOtherRef[i], arrayOther[i]);
    }
}

TEST(Tools, MeanInt)
{
    int array[] = {36, 8, 4, 6, 9, 56, 234, 45, 2475, 8, 2, 68, 9};
    float result = asTools::Mean(&array[0], &array[12]);
    EXPECT_FLOAT_EQ(227.692308, result);
}

TEST(Tools, MeanFloat)
{
    float array[] = {36.6348600000f, 8.6544964000f, 4.2346846410f, 6.5284684000f, 9.5498463130f, 56.6546544100f,
                     234.6549840000f, 45.6876513510f, 2475.6465413513f, 8.8765431894f, 2.7764850000f, 68.1000000000f,
                     9.6846510000f};
    float result = asTools::Mean(&array[0], &array[12]);
    EXPECT_FLOAT_EQ(228.283374311977, result);
}

TEST(Tools, MeanDouble)
{
    double array[] = {36.6348600000, 8.6544964000, 4.2346846410, 6.5284684000, 9.5498463130, 56.6546544100,
                      234.6549840000, 45.6876513510, 2475.6465413513, 8.8765431894, 2.7764850000, 68.1000000000,
                      9.6846510000};
    double result = asTools::Mean(&array[0], &array[12]);
    EXPECT_DOUBLE_EQ(228.283374311977, result);
}

TEST(Tools, SdtDevSampleIntSample)
{
    int array[] = {6, 113, 78, 35, 23, 56, 23, 2};
    float result = asTools::StDev(&array[0], &array[7], asSAMPLE);
    EXPECT_FLOAT_EQ(38.17254062, result);
}

TEST(Tools, SdtDevSampleIntEntirePop)
{
    int array[] = {6, 113, 78, 35, 23, 56, 23, 2};
    float result = asTools::StDev(&array[0], &array[7], asENTIRE_POPULATION);
    EXPECT_FLOAT_EQ(35.70714214, result);
}

TEST(Tools, SdtDevSampleFloatSample)
{
    float array[] = {6.1465134f, 113.134613f, 78.214334f, 35.23562346f, 23.21342f, 56.4527245f, 23.24657457f, 2.98467f};
    float result = asTools::StDev(&array[0], &array[7], asSAMPLE);
    EXPECT_FLOAT_EQ(38.05574973, result);
}

TEST(Tools, SdtDevSampleFloatEntirePop)
{
    float array[] = {6.1465134f, 113.134613f, 78.214334f, 35.23562346f, 23.21342f, 56.4527245f, 23.24657457f, 2.98467f};
    float result = asTools::StDev(&array[0], &array[7], asENTIRE_POPULATION);
    EXPECT_FLOAT_EQ(35.59789427, result);
}

TEST(Tools, SdtDevSampleDoubleSample)
{
    double array[] = {6.1465134, 113.134613, 78.214334, 35.23562346, 23.21342, 56.4527245, 23.24657457, 2.98467};
    double result = asTools::StDev(&array[0], &array[7], asSAMPLE);
    EXPECT_FLOAT_EQ(38.05574973, result);
}

TEST(Tools, SdtDevSampleDoubleEntirePop)
{
    double array[] = {6.1465134, 113.134613, 78.214334, 35.23562346, 23.21342, 56.4527245, 23.24657457, 2.98467};
    double result = asTools::StDev(&array[0], &array[7], asENTIRE_POPULATION);
    EXPECT_FLOAT_EQ(35.59789427, result);
}

TEST(Tools, RandomInt)
{
    asTools::InitRandom();
    int result1, result2;
    result1 = asTools::Random(0, 10000, 2);
    result2 = asTools::Random(0, 10000, 2);
    EXPECT_FALSE(result1 == result2);
}

TEST(Tools, RandomFloat)
{
    asTools::InitRandom();
    float result1, result2;
    float start, end, step;

    start = 0;
    end = 1000;
    step = 2.5;
    result1 = asTools::Random(start, end, step);
    result2 = asTools::Random(start, end, step);

    EXPECT_FALSE(result1 == result2);

    EXPECT_EQ(0, std::fmod(result1, step));
    EXPECT_EQ(0, std::fmod(result2, step));

    start = 0.5;
    end = 1000;
    step = 2.5;
    result1 = asTools::Random(start, end, step);
    result2 = asTools::Random(start, end, step);

    EXPECT_EQ(0, std::fmod(result1 - start, step));
    EXPECT_EQ(0, std::fmod(result2 - start, step));
}

TEST(Tools, RandomDouble)
{
    asTools::InitRandom();
    double result1, result2;
    double start, end, step;

    start = 0;
    end = 1000;
    step = 2.5;
    result1 = asTools::Random(start, end, step);
    result2 = asTools::Random(start, end, step);

    EXPECT_FALSE(result1 == result2);

    EXPECT_EQ(0, std::fmod(result1, step));
    EXPECT_EQ(0, std::fmod(result2, step));

    start = 0.5;
    end = 1000;
    step = 2.5;
    result1 = asTools::Random(start, end, step);
    result2 = asTools::Random(start, end, step);

    EXPECT_EQ(0, std::fmod(result1 - start, step));
    EXPECT_EQ(0, std::fmod(result2 - start, step));
}

// View resulting file on Matlab:
// 1. drag'n'drop in the Worskspace
// 2. figure; hist(data(:,i), 100);
TEST(Tools, RandomUniformDistributionToFile)
{
    asTools::InitRandom();

    // Create a file
    wxString tmpFile = wxFileName::CreateTempFileName("test_unidist");
    tmpFile.Append(".txt");

    asFileAscii fileRes(tmpFile, asFileAscii::Replace);
    if (!fileRes.Open())
        return;

    wxString header;
    header = _("RandomUniformDistribution processed ") + asTime::GetStringTime(asTime::NowMJD(asLOCAL));
    fileRes.AddLineContent(header);

    wxString content = wxEmptyString;

    for (int i = 0; i < 10000; i++) {
        double result1 = asTools::Random(0.0, 0.2);
        double result2 = asTools::Random(0.0, 1.0);
        double result3 = asTools::Random(0.0, 5.0);
        double result4 = asTools::Random(-2.0, 2.0);
        double result5 = asTools::Random(0.0, 10.0);
        double result6 = asTools::Random((float) 0.0, (float) 10.0, (float) 2.5);
        double result7 = asTools::Random((int) 0, (int) 10);
        double result8 = asTools::Random((int) 0, (int) 10, (int) 2);
        content.Append(wxString::Format("%g\t%g\t%g\t%g", result1, result2, result3, result4));
        content.Append(wxString::Format("\t%g\t%g\t%g\t%g\n", result5, result6, result7, result8));
    }

    fileRes.AddLineContent(content);

    fileRes.Close();
}

// View resulting file on Matlab:
// 1. drag'n'drop in the Worskspace
// 2. figure; hist(data(:,i), 100);
TEST(Tools, RandomNormalDistributionToFile)
{
    asTools::InitRandom();

    // Create a file
    wxString tmpFile = wxFileName::CreateTempFileName("test_normdist");
    tmpFile.Append(".txt");

    asFileAscii fileRes(tmpFile, asFileAscii::Replace);
    if (!fileRes.Open())
        return;

    wxString header;
    header = _("RandomNormalDistribution processed ") + asTime::GetStringTime(asTime::NowMJD(asLOCAL));
    fileRes.AddLineContent(header);

    wxString content = wxEmptyString;

    for (int i = 0; i < 10000; i++) {
        double result1 = asTools::RandomNormalDistribution(0.0, 0.2);
        double result2 = asTools::RandomNormalDistribution(0.0, 1.0);
        double result3 = asTools::RandomNormalDistribution(0.0, 5.0);
        double result4 = asTools::RandomNormalDistribution(-2.0, 0.5);
        double result5 = asTools::RandomNormalDistribution(10.0, 5.0);
        double result6 = asTools::RandomNormalDistribution((float) 10.0, (float) 5.0, (float) 2.5);
        double result7 = asTools::RandomNormalDistribution((int) 10, (int) 5);
        double result8 = asTools::RandomNormalDistribution((int) 10, (int) 5, (int) 2);
        content.Append(wxString::Format("%g\t%g\t%g\t%g", result1, result2, result3, result4));
        content.Append(wxString::Format("\t%g\t%g\t%g\t%g\n", result5, result6, result7, result8));
    }

    fileRes.AddLineContent(content);

    fileRes.Close();
}
