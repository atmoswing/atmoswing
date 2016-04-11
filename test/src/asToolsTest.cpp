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
    float Value = 2;
    EXPECT_TRUE(asTools::IsRound(Value));
}

TEST(Tools, IsRoundFloatFalse)
{
    float Value = 2.0001f;
    EXPECT_FALSE(asTools::IsRound(Value));
}

TEST(Tools, IsRoundDoubleFalse)
{
    double Value = 2.00000001;
    EXPECT_FALSE(asTools::IsRound(Value));
}

TEST(Tools, RoundFloatUp)
{
    float Value = 2.5;
    float Result = asTools::Round(Value);
    EXPECT_EQ(3, Result);
}

TEST(Tools, RoundDoubleUp)
{
    double Value = 2.5;
    double Result = asTools::Round(Value);
    EXPECT_EQ(3, Result);
}

TEST(Tools, RoundFloatDown)
{
    float Value = 2.49999f;
    float Result = asTools::Round(Value);
    EXPECT_EQ(2, Result);
}

TEST(Tools, RoundDoubleDown)
{
    double Value = 2.499999999999;
    double Result = asTools::Round(Value);
    EXPECT_EQ(2, Result);
}

TEST(Tools, RoundNegativeFloatDown)
{
    float Value = -2.5f;
    float Result = asTools::Round(Value);
    EXPECT_EQ(-3, Result);
}

TEST(Tools, RoundNegativeDoubleDown)
{
    double Value = -2.5;
    double Result = asTools::Round(Value);
    EXPECT_EQ(-3, Result);
}

TEST(Tools, RoundNegativeFloatUp)
{
    float Value = -2.49999f;
    float Result = asTools::Round(Value);
    EXPECT_EQ(-2, Result);
}

TEST(Tools, RoundNegativeDoubleUp)
{
    double Value = -2.499999999999;
    double Result = asTools::Round(Value);
    EXPECT_EQ(-2, Result);
}

TEST(Tools, RoundSmallFloatUp)
{
    float Value = 0.5;
    float Result = asTools::Round(Value);
    EXPECT_EQ(1, Result);
}

TEST(Tools, RoundSmallDoubleUp)
{
    double Value = 0.5;
    double Result = asTools::Round(Value);
    EXPECT_EQ(1, Result);
}

TEST(Tools, RoundSmallFloatDown)
{
    float Value = 0.49999f;
    float Result = asTools::Round(Value);
    EXPECT_EQ(0, Result);
}

TEST(Tools, RoundSmallDoubleDown)
{
    double Value = 0.499999999999;
    double Result = asTools::Round(Value);
    EXPECT_EQ(0, Result);
}

TEST(Tools, RoundSmallNegativeFloatDown)
{
    float Value = -0.5f;
    float Result = asTools::Round(Value);
    EXPECT_EQ(-1, Result);
}

TEST(Tools, RoundSmallNegativeDoubleDown)
{
    double Value = -0.5;
    double Result = asTools::Round(Value);
    EXPECT_EQ(-1, Result);
}

TEST(Tools, RoundSmallNegativeFloatUp)
{
    float Value = -0.49999f;
    float Result = asTools::Round(Value);
    EXPECT_EQ(0, Result);
}

TEST(Tools, RoundSmallNegativeDoubleUp)
{
    double Value = -0.499999999999;
    double Result = asTools::Round(Value);
    EXPECT_EQ(0, Result);
}

TEST(Tools, IsNaNOne)
{
    int Value = 1;
    EXPECT_FALSE(asTools::IsNaN((float) Value));
}

TEST(Tools, IsNaNZero)
{
    int Value = 0;
    EXPECT_FALSE(asTools::IsNaN((float) Value));
}

TEST(Tools, IsNaNFloatTrue)
{
    float Value = NaNFloat;
    EXPECT_TRUE(asTools::IsNaN(Value));
}

TEST(Tools, IsNaNDoubleTrue)
{
    double Value = NaNDouble;
    EXPECT_TRUE(asTools::IsNaN(Value));
}

TEST(Tools, IsInfFloatFalse)
{
    float Value = -2151;
    EXPECT_FALSE(asTools::IsInf(Value));
}

TEST(Tools, IsInfDoubleFalse)
{
    double Value = -2151;
    EXPECT_FALSE(asTools::IsInf(Value));
}

TEST(Tools, IsInfLongDoubleFalse)
{
    long double Value = -2151;
    EXPECT_FALSE(asTools::IsInf(Value));
}

TEST(Tools, IsInfFloatTrue)
{
    float Value = InfFloat;
    EXPECT_TRUE(asTools::IsInf(Value));
}

TEST(Tools, IsInfDoubleTrue)
{
    double Value = InfDouble;
    EXPECT_TRUE(asTools::IsInf(Value));
}

TEST(Tools, IsInfLongDoubleTrue)
{
    long double Value = InfLongDouble;
    EXPECT_TRUE(asTools::IsInf(Value));
}

TEST(Tools, CountNotNaNFloat)
{
    float Array[] = {0.3465f, 1.345f, 2.76f, 3.69f, 5.58f, NaNFloat, 8.34f, 9.75f, 10.0f, NaNFloat};
    float *pVectStart = &Array[0];
    float *pVectEnd = &Array[9];
    int Result = asTools::CountNotNaN(pVectStart, pVectEnd);
    EXPECT_EQ(8, Result);
}

TEST(Tools, CountNotNaNDouble)
{
    double Array[] = {0.3465, 1.345, 2.76, 3.69, 5.58, NaNDouble, 8.34, 9.75, 10, NaNDouble};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    int Result = asTools::CountNotNaN(pVectStart, pVectEnd);
    EXPECT_EQ(8, Result);
}

TEST(Tools, FindMinInt)
{
    int Array[] = {0, 0, -1, 1, 2, 3, 5, 0, 8, 9, 10, 0};
    int Result = asTools::MinArray(&Array[0], &Array[11]);
    EXPECT_EQ(-1, Result);
}

TEST(Tools, FindMinFloat)
{
    float Array[] = {NaNFloat, NaNFloat, 0.3465f, 1.345f, 2.76f, 3.69f, 5.58f, NaNFloat, 8.34f, 9.75f, 10.0f, NaNFloat};
    float Result = asTools::MinArray(&Array[0], &Array[11]);
    EXPECT_FLOAT_EQ(0.3465, Result);
}

TEST(Tools, FindMinDouble)
{
    double Array[] = {NaNDouble, NaNDouble, 0.3465, 1.345, 2.76, 3.69, 5.58, NaNDouble, 8.34, 9.75, 10, NaNDouble};
    double Result = asTools::MinArray(&Array[0], &Array[11]);
    EXPECT_DOUBLE_EQ(0.3465, Result);
}

TEST(Tools, FindMaxInt)
{
    int Array[] = {0, 0, -1, 1, 2, 3, 5, 0, 8, 9, 10, 0};
    int Result = asTools::MaxArray(&Array[0], &Array[11]);
    EXPECT_EQ(10, Result);
}

TEST(Tools, FindMaxFloat)
{
    float Array[] = {NaNFloat, NaNFloat, 0.3465f, 1.345f, 2.76f, 3.69f, 5.58f, NaNFloat, 8.34f, 9.75f, 10.12f,
                     NaNFloat};
    float Result = asTools::MaxArray(&Array[0], &Array[11]);
    EXPECT_FLOAT_EQ(10.12, Result);
}

TEST(Tools, FindMaxDouble)
{
    double Array[] = {NaNDouble, NaNDouble, 0.3465, 1.345, 2.76, 3.69, 5.58, NaNDouble, 8.34, 9.75, 10.12, NaNDouble};
    double Result = asTools::MaxArray(&Array[0], &Array[11]);
    EXPECT_DOUBLE_EQ(10.12, Result);
}

TEST(Tools, FindMinStepInt)
{
    int Array[] = {0, 10, 0, 1, 3, 5, 0, 8, 2, 9, 0, 0};
    int Result = asTools::MinArrayStep(&Array[0], &Array[11], 0);
    EXPECT_EQ(1, Result);
}

TEST(Tools, FindMinStepFloat)
{
    float Array[] = {NaNFloat, 10.12f, NaNFloat, 1.345f, 1.345f, 3.69f, 5.58f, NaNFloat, 8.34f, 2.76f, 9.75f, 0.3465f,
                     NaNFloat};
    float Result = asTools::MinArrayStep(&Array[0], &Array[11], 0.0001f);
    EXPECT_FLOAT_EQ(0.37, Result);
}

TEST(Tools, FindMinStepDouble)
{
    double Array[] = {NaNDouble, 10.12, NaNDouble, 1.345, 1.345, 3.69, 5.58, NaNDouble, 8.34, 2.76, 9.75, 0.3465,
                      NaNDouble};
    double Result = asTools::MinArrayStep(&Array[0], &Array[11]);
    EXPECT_FLOAT_EQ(0.37, Result);
}

TEST(Tools, ExtractUniqueValuesInt)
{
    int Array[] = {0, 10, 0, 1, 3, 5, 1, 8, 2, 9, 0, 9};;
    Array1DInt Result(asTools::ExtractUniqueValues(&Array[0], &Array[11]));
    EXPECT_EQ(0, Result[0]);
    EXPECT_EQ(1, Result[1]);
    EXPECT_EQ(2, Result[2]);
    EXPECT_EQ(3, Result[3]);
    EXPECT_EQ(5, Result[4]);
    EXPECT_EQ(8, Result[5]);
    EXPECT_EQ(9, Result[6]);
    EXPECT_EQ(10, Result[7]);
}

TEST(Tools, ExtractUniqueValuesFloat)
{
    float Array[] = {NaNFloat, 10.12f, NaNFloat, 1.345f, 1.345f, 3.69f, 5.58f, NaNFloat, 8.34f, 2.76f, 9.75f, 0.3465f,
                     NaNFloat};
    Array1DFloat Result(asTools::ExtractUniqueValues(&Array[0], &Array[11], 0.0001f));
    EXPECT_FLOAT_EQ(0.3465, Result[0]);
    EXPECT_FLOAT_EQ(1.345, Result[1]);
    EXPECT_FLOAT_EQ(2.76, Result[2]);
    EXPECT_FLOAT_EQ(3.69, Result[3]);
    EXPECT_FLOAT_EQ(5.58, Result[4]);
    EXPECT_FLOAT_EQ(8.34, Result[5]);
    EXPECT_FLOAT_EQ(9.75, Result[6]);
    EXPECT_FLOAT_EQ(10.12, Result[7]);
}

TEST(Tools, ExtractUniqueValuesDouble)
{
    double Array[] = {NaNDouble, 10.12, NaNDouble, 1.345, 1.345, 3.69, 5.58, NaNDouble, 8.34, 2.76, 9.75, 0.3465,
                      NaNDouble};
    Array1DDouble Result(asTools::ExtractUniqueValues(&Array[0], &Array[11]));
    EXPECT_DOUBLE_EQ(0.3465, Result[0]);
    EXPECT_DOUBLE_EQ(1.345, Result[1]);
    EXPECT_DOUBLE_EQ(2.76, Result[2]);
    EXPECT_DOUBLE_EQ(3.69, Result[3]);
    EXPECT_DOUBLE_EQ(5.58, Result[4]);
    EXPECT_DOUBLE_EQ(8.34, Result[5]);
    EXPECT_DOUBLE_EQ(9.75, Result[6]);
    EXPECT_DOUBLE_EQ(10.12, Result[7]);
}

TEST(Tools, SortedArraySearchIntAscFirst)
{
    int Array[] = {0, 1, 2, 3, 5, 7, 8, 9, 10, 100};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int targetvalue = 0;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchIntAscMid)
{
    int Array[] = {0, 1, 2, 3, 5, 7, 8, 9, 10, 100};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int targetvalue = 8;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(6, Result);
}

TEST(Tools, SortedArraySearchIntAscLast)
{
    int Array[] = {0, 1, 2, 3, 5, 7, 8, 9, 10, 100};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int targetvalue = 100;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, Result);
}

TEST(Tools, SortedArraySearchIntAscOutofRange)
{
    int Array[] = {0, 1, 2, 3, 5, 7, 8, 9, 10, 100};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int targetvalue = 1000;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 0, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchIntAscNotFound)
{
    int Array[] = {0, 1, 2, 3, 5, 7, 8, 9, 10, 100};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int targetvalue = 6;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 0, asHIDE_WARNINGS);
    EXPECT_EQ(asNOT_FOUND, Result);
}

TEST(Tools, SortedArraySearchIntAscTolerFirst)
{
    int Array[] = {0, 3, 4, 5, 6, 7, 8, 9, 10, 100};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int targetvalue = -1;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchIntAscTolerFirstLimit)
{
    int Array[] = {0, 1, 2, 3, 5, 7, 8, 9, 10, 100};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int targetvalue = -1;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchIntAscTolerMid)
{
    int Array[] = {0, 1, 2, 3, 5, 7, 8, 9, 10, 100};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int targetvalue = 11;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(8, Result);
}

TEST(Tools, SortedArraySearchIntAscTolerMidLimit)
{
    int Array[] = {0, 1, 2, 3, 5, 7, 8, 9, 10, 100};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int targetvalue = 11;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(8, Result);
}

TEST(Tools, SortedArraySearchIntAscTolerLast)
{
    int Array[] = {0, 1, 2, 3, 5, 7, 8, 9, 10, 100};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int targetvalue = 102;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 3);
    EXPECT_EQ(9, Result);
}

TEST(Tools, SortedArraySearchIntAscTolerLastLimit)
{
    int Array[] = {0, 1, 2, 3, 5, 7, 8, 9, 10, 100};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int targetvalue = 102;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 2);
    EXPECT_EQ(9, Result);
}

TEST(Tools, SortedArraySearchIntDescFirst)
{
    int Array[] = {100, 10, 9, 8, 7, 5, 3, 2, 1, 0};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int targetvalue = 100;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchIntDescMid)
{
    int Array[] = {100, 10, 9, 8, 7, 5, 3, 2, 1, 0};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int targetvalue = 8;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(3, Result);
}

TEST(Tools, SortedArraySearchIntDescLast)
{
    int Array[] = {100, 10, 9, 8, 7, 5, 3, 2, 1, 0};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int targetvalue = 0;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, Result);
}

TEST(Tools, SortedArraySearchIntDescOutofRange)
{
    int Array[] = {100, 10, 9, 8, 7, 5, 3, 2, 1, 0};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int targetvalue = -1;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 0, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchIntDescNotFound)
{
    int Array[] = {100, 10, 9, 8, 7, 5, 3, 2, 1, 0};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int targetvalue = 6;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 0, asHIDE_WARNINGS);
    EXPECT_EQ(asNOT_FOUND, Result);
}

TEST(Tools, SortedArraySearchIntDescTolerFirst)
{
    int Array[] = {100, 10, 9, 8, 7, 5, 3, 2, 1, 0};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int targetvalue = -1;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(9, Result);
}

TEST(Tools, SortedArraySearchIntDescTolerFirstLimit)
{
    int Array[] = {100, 10, 9, 8, 7, 5, 3, 2, 1, 0};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int targetvalue = -1;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(9, Result);
}

TEST(Tools, SortedArraySearchIntDescTolerMid)
{
    int Array[] = {100, 10, 9, 8, 7, 5, 3, 2, 1, 0};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int targetvalue = 11;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 3);
    EXPECT_EQ(1, Result);
}

TEST(Tools, SortedArraySearchIntDescTolerMidLimit)
{
    int Array[] = {100, 10, 9, 8, 7, 5, 3, 2, 1, 0};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int targetvalue = 11;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(1, Result);
}

TEST(Tools, SortedArraySearchIntDescTolerLast)
{
    int Array[] = {100, 10, 9, 8, 7, 5, 3, 2, 1, 0};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int targetvalue = 102;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 3);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchIntDescTolerLastLimit)
{
    int Array[] = {100, 10, 9, 8, 7, 5, 3, 2, 1, 0};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int targetvalue = 102;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 2);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchIntUniqueVal)
{
    int Array[] = {9};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[0];
    int targetvalue = 9;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchIntUniqueValToler)
{
    int Array[] = {9};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[0];
    int targetvalue = 8;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchIntUniqueValOutofRange)
{
    int Array[] = {9};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[0];
    int targetvalue = 11;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchIntArraySameVal)
{
    int Array[] = {9, 9, 9, 9};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[3];
    int targetvalue = 9;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchIntArraySameValTolerDown)
{
    int Array[] = {9, 9, 9, 9};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[3];
    int targetvalue = 8;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchIntArraySameValTolerUp)
{
    int Array[] = {9, 9, 9, 9};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[3];
    int targetvalue = 10;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchIntArraySameValOutofRange)
{
    int Array[] = {9, 9, 9, 9};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[3];
    int targetvalue = 11;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchDoubleAscFirst)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 0.354;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchDoubleAscMid)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 5.75;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(5, Result);
}

TEST(Tools, SortedArraySearchDoubleAscLast)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 100;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, Result);
}

TEST(Tools, SortedArraySearchDoubleAscOutofRange)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 1000;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 0.0, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchDoubleAscNotFound)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 6;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 0.0, asHIDE_WARNINGS);
    EXPECT_EQ(asNOT_FOUND, Result);
}

TEST(Tools, SortedArraySearchDoubleAscTolerFirst)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = -1.12;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 3);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchDoubleAscTolerFirstLimit)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = -1;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1.354);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchDoubleAscTolerFirstOutLimit)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = -1;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1.353, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchDoubleAscTolerMid)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 11;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(8, Result);
}

TEST(Tools, SortedArraySearchDoubleAscTolerMidLimit)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 11.45;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(8, Result);
}

TEST(Tools, SortedArraySearchDoubleAscTolerMidLimitOut)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 11.45;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 0.99, asHIDE_WARNINGS);
    EXPECT_EQ(asNOT_FOUND, Result);
}

TEST(Tools, SortedArraySearchDoubleAscTolerLast)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 102.21;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 3);
    EXPECT_EQ(9, Result);
}

TEST(Tools, SortedArraySearchDoubleAscTolerLastLimit)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 101.5;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1.5);
    EXPECT_EQ(9, Result);
}

TEST(Tools, SortedArraySearchDoubleAscTolerLastOutLimit)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 101.5;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1.499, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchDoubleDescFirst)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 100;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchDoubleDescMid)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 5.75;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(4, Result);
}

TEST(Tools, SortedArraySearchDoubleDescLast)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 0.354;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, Result);
}

TEST(Tools, SortedArraySearchDoubleDescOutofRange)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = -1.23;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 0.0, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchDoubleDescNotFound)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 6.2;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 0.0, asHIDE_WARNINGS);
    EXPECT_EQ(asNOT_FOUND, Result);
}

TEST(Tools, SortedArraySearchDoubleDescTolerFirst)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = -1;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 2);
    EXPECT_EQ(9, Result);
}

TEST(Tools, SortedArraySearchDoubleDescTolerFirstLimit)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = -1;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1.354);
    EXPECT_EQ(9, Result);
}

TEST(Tools, SortedArraySearchDoubleDescTolerFirstOutLimit)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = -1;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1.353, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchDoubleDescTolerMid)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 11.23;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 3);
    EXPECT_EQ(1, Result);
}

TEST(Tools, SortedArraySearchDoubleDescTolerMidLimit)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 11.45;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(1, Result);
}

TEST(Tools, SortedArraySearchDoubleDescTolerMidOutLimit)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 11.45;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 0.999, asHIDE_WARNINGS);
    EXPECT_EQ(asNOT_FOUND, Result);
}

TEST(Tools, SortedArraySearchDoubleDescTolerLast)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 102.42;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 3);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchDoubleDescTolerLastLimit)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 102.21;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 2.21);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchDoubleDescTolerLastOutLimit)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 102.21;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 2.2, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchDoubleUniqueVal)
{
    double Array[] = {9.3401};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[0];
    double targetvalue = 9.3401;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchDoubleUniqueValToler)
{
    double Array[] = {9.3401};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[0];
    double targetvalue = 8;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1.3401);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchDoubleUniqueValOutofRange)
{
    double Array[] = {9.3401};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[0];
    double targetvalue = 11;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchDoubleArraySameVal)
{
    double Array[] = {9.34, 9.34, 9.34, 9.34};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[3];
    double targetvalue = 9.34;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchDoubleArraySameValTolerDown)
{
    double Array[] = {9.34, 9.34, 9.34, 9.34};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[3];
    double targetvalue = 8;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1.5);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchDoubleArraySameValTolerUp)
{
    double Array[] = {9.34, 9.34, 9.34, 9.34};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[3];
    double targetvalue = 10;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchDoubleArraySameValOutofRange)
{
    double Array[] = {9.34, 9.34, 9.34, 9.34};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[3];
    double targetvalue = 11;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 1, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchRoleOfToleranceInSearch)
{
    Array1DDouble values;
    values.resize(94);
    values <<
    -88.542, -86.653, -84.753, -82.851, -80.947, -79.043, -77.139, -75.235, -73.331, -71.426, -69.522, -67.617, -65.713, -63.808, -61.903, -59.999, -58.094, -56.189, -54.285, -52.380, -50.475, -48.571, -46.666, -44.761, -42.856, -40.952, -39.047, -37.142, -35.238, -33.333, -31.428, -29.523, -27.619, -25.714, -23.809, -21.904, -20.000, -18.095, -16.190, -14.286, -12.381, -10.476, -08.571, -06.667, -04.762, -02.857, -00.952, 00.952, 02.857, 04.762, 06.667, 08.571, 10.476, 12.381, 14.286, 16.190, 18.095, 20.000, 21.904, 23.809, 25.714, 27.619, 29.523, 31.428, 33.333, 35.238, 37.142, 39.047, 40.952, 42.856, 44.761, 46.666, 48.571, 50.475, 52.380, 54.285, 56.189, 58.094, 59.999, 61.903, 63.808, 65.713, 67.617, 69.522, 71.426, 73.331, 75.235, 77.139, 79.043, 80.947, 82.851, 84.753, 86.653, 88.542;
    double *pVectStart = &values[0];
    double *pVectEnd = &values[93];
    double targetvalue = 29.523;
    int Result = asTools::SortedArraySearch(pVectStart, pVectEnd, targetvalue, 0.01);
    EXPECT_EQ(62, Result);
}

TEST(Tools, SortedArraySearchClosestDoubleAscFirst)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 0.394;
    int Result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchClosestDoubleAscMid)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 5.55;
    int Result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(5, Result);
}

TEST(Tools, SortedArraySearchClosestDoubleAscLast)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 99.9;
    int Result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, Result);
}

TEST(Tools, SortedArraySearchClosestDoubleAscOutofRange)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 1000;
    int Result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchClosestDoubleDescFirst)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 100;
    int Result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchClosestDoubleDescMid)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 5.55;
    int Result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(4, Result);
}

TEST(Tools, SortedArraySearchClosestDoubleDescLast)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 0.354;
    int Result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, Result);
}

TEST(Tools, SortedArraySearchClosestDoubleDescOutofRange)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = -1.23;
    int Result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchClosestDoubleUniqueVal)
{
    double Array[] = {9.3401};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[0];
    double targetvalue = 9.3401;
    int Result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchClosestDoubleUniqueValOutofRange)
{
    double Array[] = {9.3401};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[0];
    double targetvalue = 11;
    int Result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchClosestDoubleArraySameVal)
{
    double Array[] = {9.34, 9.34, 9.34, 9.34};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[3];
    double targetvalue = 9.34;
    int Result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchClosestDoubleArraySameValOutofRange)
{
    double Array[] = {9.34, 9.34, 9.34, 9.34};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[3];
    double targetvalue = 11;
    int Result = asTools::SortedArraySearchClosest(pVectStart, pVectEnd, targetvalue, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchFloorDoubleAscFirst)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 1.394;
    int Result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchFloorDoubleAscMid)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 5.55;
    int Result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(4, Result);
}

TEST(Tools, SortedArraySearchFloorDoubleAscLast)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 99.9;
    int Result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(8, Result);
}

TEST(Tools, SortedArraySearchFloorDoubleAscLastExact)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 100;
    int Result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, Result);
}

TEST(Tools, SortedArraySearchFloorDoubleAscOutofRange)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 1000;
    int Result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchFloorDoubleDescFirst)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 40.12;
    int Result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(1, Result);
}

TEST(Tools, SortedArraySearchFloorDoubleDescMid)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 5.55;
    int Result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(5, Result);
}

TEST(Tools, SortedArraySearchFloorDoubleDescLast)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 0.360;
    int Result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, Result);
}

TEST(Tools, SortedArraySearchFloorDoubleDescLastExact)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 0.354;
    int Result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, Result);
}

TEST(Tools, SortedArraySearchFloorDoubleDescOutofRange)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = -1.23;
    int Result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchFloorDoubleUniqueVal)
{
    double Array[] = {9.3401};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[0];
    double targetvalue = 9.3401;
    int Result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchFloorDoubleUniqueValOutofRange)
{
    double Array[] = {9.3401};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[0];
    double targetvalue = 11;
    int Result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchFloorDoubleArraySameVal)
{
    double Array[] = {9.34, 9.34, 9.34, 9.34};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[3];
    double targetvalue = 9.34;
    int Result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchFloorDoubleArraySameValOutofRange)
{
    double Array[] = {9.34, 9.34, 9.34, 9.34};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[3];
    double targetvalue = 11;
    int Result = asTools::SortedArraySearchFloor(pVectStart, pVectEnd, targetvalue, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchCeilDoubleAscFirst)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 0.354;
    int Result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchCeilDoubleAscMid)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 5.55;
    int Result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(5, Result);
}

TEST(Tools, SortedArraySearchCeilDoubleAscLast)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 10.46;
    int Result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, Result);
}

TEST(Tools, SortedArraySearchCeilDoubleAscLastExact)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 100;
    int Result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, Result);
}

TEST(Tools, SortedArraySearchCeilDoubleAscOutofRange)
{
    double Array[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 1000;
    int Result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchCeilDoubleDescFirst)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 40.12;
    int Result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchCeilDoubleDescMid)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 5.55;
    int Result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(4, Result);
}

TEST(Tools, SortedArraySearchCeilDoubleDescLast)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 0.360;
    int Result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(8, Result);
}

TEST(Tools, SortedArraySearchCeilDoubleDescLastExact)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = 0.354;
    int Result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(9, Result);
}

TEST(Tools, SortedArraySearchCeilDoubleDescOutofRange)
{
    double Array[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double targetvalue = -1.23;
    int Result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchCeilDoubleUniqueVal)
{
    double Array[] = {9.3401};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[0];
    double targetvalue = 9.3401;
    int Result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchCeilDoubleUniqueValOutofRange)
{
    double Array[] = {9.3401};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[0];
    double targetvalue = 11;
    int Result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArraySearchCeilDoubleArraySameVal)
{
    double Array[] = {9.34, 9.34, 9.34, 9.34};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[3];
    double targetvalue = 9.34;
    int Result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue);
    EXPECT_EQ(0, Result);
}

TEST(Tools, SortedArraySearchCeilDoubleArraySameValOutofRange)
{
    double Array[] = {9.34, 9.34, 9.34, 9.34};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[3];
    double targetvalue = 11;
    int Result = asTools::SortedArraySearchCeil(pVectStart, pVectEnd, targetvalue, asHIDE_WARNINGS);
    EXPECT_EQ(asOUT_OF_RANGE, Result);
}

TEST(Tools, SortedArrayInsertIntAscFirst)
{
    int Array[] = {2, 3, 4, 6, 9, 17, 18, 20, 40, 100};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int newvalue = 1;
    EXPECT_TRUE(asTools::SortedArrayInsert(pVectStart, pVectEnd, Asc, newvalue));
    int ArrayResults[] = {1, 2, 3, 4, 6, 9, 17, 18, 20, 40};
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(ArrayResults[i], Array[i]);
    }
}

TEST(Tools, SortedArrayInsertIntAscFirstNeg)
{
    int Array[] = {0, 1, 4, 6, 9, 17, 18, 20, 40, 100};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int newvalue = -2;
    EXPECT_TRUE(asTools::SortedArrayInsert(pVectStart, pVectEnd, Asc, newvalue));
    int ArrayResults[] = {-2, 0, 1, 4, 6, 9, 17, 18, 20, 40};
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(ArrayResults[i], Array[i]);
    }
}

TEST(Tools, SortedArrayInsertIntAscMid)
{
    int Array[] = {0, 1, 4, 6, 9, 17, 18, 20, 40, 100};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int newvalue = 8;
    EXPECT_TRUE(asTools::SortedArrayInsert(pVectStart, pVectEnd, Asc, newvalue));
    int ArrayResults[] = {0, 1, 4, 6, 8, 9, 17, 18, 20, 40};
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(ArrayResults[i], Array[i]);
    }
}

TEST(Tools, SortedArrayInsertIntAscEnd)
{
    int Array[] = {0, 1, 4, 6, 9, 17, 18, 20, 40, 100};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int newvalue = 90;
    EXPECT_TRUE(asTools::SortedArrayInsert(pVectStart, pVectEnd, Asc, newvalue));
    int ArrayResults[] = {0, 1, 4, 6, 9, 17, 18, 20, 40, 90};
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(ArrayResults[i], Array[i]);
    }
}

TEST(Tools, SortedArrayInsertFloatAscMid)
{
    float Array[] = {0.134631f, 1.13613f, 4.346f, 6.835f, 9.1357f, 17.23456f, 18.2364f, 20.75f, 40.54f, 100.235f};
    float *pVectStart = &Array[0];
    float *pVectEnd = &Array[9];
    float newvalue = 9.105646f;
    EXPECT_TRUE(asTools::SortedArrayInsert(pVectStart, pVectEnd, Asc, newvalue));
    float ArrayResults[] = {0.134631f, 1.13613f, 4.346f, 6.835f, 9.105646f, 9.1357f, 17.23456f, 18.2364f, 20.75f,
                            40.54f};
    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(ArrayResults[i], Array[i]);
    }
}

TEST(Tools, SortedArrayInsertDoubleAscMid)
{
    double Array[] = {0.134631, 1.13613, 4.346, 6.835, 9.1357, 17.23456, 18.2364, 20.75, 40.54, 100.235};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double newvalue = 9.105646;
    EXPECT_TRUE(asTools::SortedArrayInsert(pVectStart, pVectEnd, Asc, newvalue));
    double ArrayResults[] = {0.134631, 1.13613, 4.346, 6.835, 9.105646, 9.1357, 17.23456, 18.2364, 20.75, 40.54};
    for (int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(ArrayResults[i], Array[i]);
    }
}

TEST(Tools, SortedArrayInsertIntDescFirst)
{
    int Array[] = {100, 40, 20, 18, 17, 9, 6, 4, 3, 2};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int newvalue = 101;
    EXPECT_TRUE(asTools::SortedArrayInsert(pVectStart, pVectEnd, Desc, newvalue));
    int ArrayResults[] = {101, 100, 40, 20, 18, 17, 9, 6, 4, 3};
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(ArrayResults[i], Array[i]);
    }
}

TEST(Tools, SortedArrayInsertIntDescMid)
{
    int Array[] = {100, 40, 20, 18, 17, 9, 6, 4, 3, 2};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int newvalue = 8;
    EXPECT_TRUE(asTools::SortedArrayInsert(pVectStart, pVectEnd, Desc, newvalue));
    int ArrayResults[] = {100, 40, 20, 18, 17, 9, 8, 6, 4, 3};
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(ArrayResults[i], Array[i]);
    }
}

TEST(Tools, SortedArrayInsertIntDescEnd)
{
    int Array[] = {100, 40, 20, 18, 17, 9, 6, 4, 3, 2};
    int *pVectStart = &Array[0];
    int *pVectEnd = &Array[9];
    int newvalue = 3;
    EXPECT_TRUE(asTools::SortedArrayInsert(pVectStart, pVectEnd, Desc, newvalue));
    int ArrayResults[] = {100, 40, 20, 18, 17, 9, 6, 4, 3, 3};
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(ArrayResults[i], Array[i]);
    }
}

TEST(Tools, SortedArrayInsertFloatDescMid)
{
    float Array[] = {100.1345f, 40.2345f, 20.2345f, 18.567f, 17.2134f, 9.67f, 6.1346f, 4.7135f, 3.1f, 2.2345f};
    float *pVectStart = &Array[0];
    float *pVectEnd = &Array[9];
    float newvalue = 9.105646f;
    EXPECT_TRUE(asTools::SortedArrayInsert(pVectStart, pVectEnd, Desc, newvalue));
    float ArrayResults[] = {100.1345f, 40.2345f, 20.2345f, 18.567f, 17.2134f, 9.67f, 9.105646f, 6.1346f, 4.7135f, 3.1f};
    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(ArrayResults[i], Array[i]);
    }
}

TEST(Tools, SortedArraysInsertIntAscFirst)
{
    int ArrayRef[] = {2, 3, 4, 6, 9, 17, 18, 20, 40, 100};
    int ArrayOther[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int newvalueRef = 1;
    int newvalueOther = 11;
    EXPECT_TRUE(
            asTools::SortedArraysInsert(&ArrayRef[0], &ArrayRef[9], &ArrayOther[0], &ArrayOther[9], Asc, newvalueRef,
                                        newvalueOther));
    int ArrayResultsRef[] = {1, 2, 3, 4, 6, 9, 17, 18, 20, 40};
    int ArrayResultsOther[] = {11, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(ArrayResultsRef[i], ArrayRef[i]);
        EXPECT_EQ(ArrayResultsOther[i], ArrayOther[i]);
    }
}

TEST(Tools, SortedArraysInsertIntAscMid)
{
    int ArrayRef[] = {2, 3, 4, 6, 9, 17, 18, 20, 40, 100};
    int ArrayOther[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int newvalueRef = 11;
    int newvalueOther = 11;
    EXPECT_TRUE(
            asTools::SortedArraysInsert(&ArrayRef[0], &ArrayRef[9], &ArrayOther[0], &ArrayOther[9], Asc, newvalueRef,
                                        newvalueOther));
    int ArrayResultsRef[] = {2, 3, 4, 6, 9, 11, 17, 18, 20, 40};
    int ArrayResultsOther[] = {1, 2, 3, 4, 5, 11, 6, 7, 8, 9};
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(ArrayResultsRef[i], ArrayRef[i]);
        EXPECT_EQ(ArrayResultsOther[i], ArrayOther[i]);
    }
}

TEST(Tools, SortedArraysInsertIntAscLast)
{
    int ArrayRef[] = {2, 3, 4, 6, 9, 17, 18, 20, 40, 100};
    int ArrayOther[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int newvalueRef = 99;
    int newvalueOther = 11;
    EXPECT_TRUE(
            asTools::SortedArraysInsert(&ArrayRef[0], &ArrayRef[9], &ArrayOther[0], &ArrayOther[9], Asc, newvalueRef,
                                        newvalueOther));
    int ArrayResultsRef[] = {2, 3, 4, 6, 9, 17, 18, 20, 40, 99};
    int ArrayResultsOther[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11};
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(ArrayResultsRef[i], ArrayRef[i]);
        EXPECT_EQ(ArrayResultsOther[i], ArrayOther[i]);
    }
}

TEST(Tools, SortedArraysInsertFloatAscMid)
{
    float ArrayRef[] = {2.254f, 3.345f, 4.625f, 6.47f, 9.7f, 17.245f, 18.0f, 20.67f, 40.25f, 100.25f};
    float ArrayOther[] = {1.7f, 2.4f, 3.346f, 4.7f, 5.1346f, 6.715f, 7.1346f, 8.1357f, 9.1346f, 10.715f};
    float newvalueRef = 11.175f;
    float newvalueOther = 11.1346f;
    EXPECT_TRUE(
            asTools::SortedArraysInsert(&ArrayRef[0], &ArrayRef[9], &ArrayOther[0], &ArrayOther[9], Asc, newvalueRef,
                                        newvalueOther));
    float ArrayResultsRef[] = {2.254f, 3.345f, 4.625f, 6.47f, 9.7f, 11.175f, 17.245f, 18.0f, 20.67f, 40.25f};
    float ArrayResultsOther[] = {1.7f, 2.4f, 3.346f, 4.7f, 5.1346f, 11.1346f, 6.715f, 7.1346f, 8.1357f, 9.1346f};
    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(ArrayResultsRef[i], ArrayRef[i]);
        EXPECT_FLOAT_EQ(ArrayResultsOther[i], ArrayOther[i]);
    }
}

TEST(Tools, SortedArraysInsertDoubleAscMid)
{
    double ArrayRef[] = {2.254, 3.345, 4.625, 6.47, 9.7, 17.245, 18, 20.67, 40.25, 100.25};
    double ArrayOther[] = {1.7, 2.4, 3.346, 4.7, 5.1346, 6.715, 7.1346, 8.1357, 9.1346, 10.715};
    double newvalueRef = 11.175;
    double newvalueOther = 11.1346;
    EXPECT_TRUE(
            asTools::SortedArraysInsert(&ArrayRef[0], &ArrayRef[9], &ArrayOther[0], &ArrayOther[9], Asc, newvalueRef,
                                        newvalueOther));
    double ArrayResultsRef[] = {2.254, 3.345, 4.625, 6.47, 9.7, 11.175, 17.245, 18, 20.67, 40.25};
    double ArrayResultsOther[] = {1.7, 2.4, 3.346, 4.7, 5.1346, 11.1346, 6.715, 7.1346, 8.1357, 9.1346};
    for (int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(ArrayResultsRef[i], ArrayRef[i]);
        EXPECT_DOUBLE_EQ(ArrayResultsOther[i], ArrayOther[i]);
    }
}

TEST(Tools, SortArrayAsc)
{
    double ArrayRef[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double Array[] = {9.65, 2.7, 0.354, 100, 5.75, 1.932, 8.2, 10.45, 5.021, 3.56};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    EXPECT_TRUE(asTools::SortArray(pVectStart, pVectEnd, Asc));
    for (int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(ArrayRef[i], Array[i]);
    }
}

TEST(Tools, SortArrayDesc)
{
    double ArrayRef[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double Array[] = {9.65, 2.7, 0.354, 100, 5.75, 1.932, 8.2, 10.45, 5.021, 3.56};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    EXPECT_TRUE(asTools::SortArray(pVectStart, pVectEnd, Desc));
    for (int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(ArrayRef[i], Array[i]);
    }
}

TEST(Tools, SortArraysAsc)
{
    double ArrayRef[] = {0.354, 1.932, 2.7, 3.56, 5.021, 5.75, 8.2, 9.65, 10.45, 100};
    double Array[] = {9.65, 2.7, 0.354, 100, 5.75, 1.932, 8.2, 10.45, 5.021, 3.56};
    double ArrayOtherRef[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double ArrayOther[] = {8, 3, 1, 10, 6, 2, 7, 9, 5, 4};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double *pVectStartOther = &ArrayOther[0];
    double *pVectEndOther = &ArrayOther[9];
    EXPECT_TRUE(asTools::SortArrays(pVectStart, pVectEnd, pVectStartOther, pVectEndOther, Asc));
    for (int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(ArrayRef[i], Array[i]);
        EXPECT_DOUBLE_EQ(ArrayOtherRef[i], ArrayOther[i]);
    }
}

TEST(Tools, SortArraysDesc)
{
    double ArrayRef[] = {100, 10.45, 9.65, 8.2, 5.75, 5.021, 3.56, 2.7, 1.932, 0.354};
    double Array[] = {9.65, 2.7, 0.354, 100, 5.75, 1.932, 8.2, 10.45, 5.021, 3.56};
    double ArrayOtherRef[] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    double ArrayOther[] = {8, 3, 1, 10, 6, 2, 7, 9, 5, 4};
    double *pVectStart = &Array[0];
    double *pVectEnd = &Array[9];
    double *pVectStartOther = &ArrayOther[0];
    double *pVectEndOther = &ArrayOther[9];
    EXPECT_TRUE(asTools::SortArrays(pVectStart, pVectEnd, pVectStartOther, pVectEndOther, Desc));
    for (int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(ArrayRef[i], Array[i]);
        EXPECT_DOUBLE_EQ(ArrayOtherRef[i], ArrayOther[i]);
    }
}

TEST(Tools, MeanInt)
{
    int Array[] = {36, 8, 4, 6, 9, 56, 234, 45, 2475, 8, 2, 68, 9};
    float result = asTools::Mean(&Array[0], &Array[12]);
    EXPECT_FLOAT_EQ(227.692308, result);
}

TEST(Tools, MeanFloat)
{
    float Array[] = {36.6348600000f, 8.6544964000f, 4.2346846410f, 6.5284684000f, 9.5498463130f, 56.6546544100f,
                     234.6549840000f, 45.6876513510f, 2475.6465413513f, 8.8765431894f, 2.7764850000f, 68.1000000000f,
                     9.6846510000f};
    float result = asTools::Mean(&Array[0], &Array[12]);
    EXPECT_FLOAT_EQ(228.283374311977, result);
}

TEST(Tools, MeanDouble)
{
    double Array[] = {36.6348600000, 8.6544964000, 4.2346846410, 6.5284684000, 9.5498463130, 56.6546544100,
                      234.6549840000, 45.6876513510, 2475.6465413513, 8.8765431894, 2.7764850000, 68.1000000000,
                      9.6846510000};
    double result = asTools::Mean(&Array[0], &Array[12]);
    EXPECT_DOUBLE_EQ(228.283374311977, result);
}

TEST(Tools, SdtDevSampleIntSample)
{
    int Array[] = {6, 113, 78, 35, 23, 56, 23, 2};
    float result = asTools::StDev(&Array[0], &Array[7], asSAMPLE);
    EXPECT_FLOAT_EQ(38.17254062, result);
}

TEST(Tools, SdtDevSampleIntEntirePop)
{
    int Array[] = {6, 113, 78, 35, 23, 56, 23, 2};
    float result = asTools::StDev(&Array[0], &Array[7], asENTIRE_POPULATION);
    EXPECT_FLOAT_EQ(35.70714214, result);
}

TEST(Tools, SdtDevSampleFloatSample)
{
    float Array[] = {6.1465134f, 113.134613f, 78.214334f, 35.23562346f, 23.21342f, 56.4527245f, 23.24657457f, 2.98467f};
    float result = asTools::StDev(&Array[0], &Array[7], asSAMPLE);
    EXPECT_FLOAT_EQ(38.05574973, result);
}

TEST(Tools, SdtDevSampleFloatEntirePop)
{
    float Array[] = {6.1465134f, 113.134613f, 78.214334f, 35.23562346f, 23.21342f, 56.4527245f, 23.24657457f, 2.98467f};
    float result = asTools::StDev(&Array[0], &Array[7], asENTIRE_POPULATION);
    EXPECT_FLOAT_EQ(35.59789427, result);
}

TEST(Tools, SdtDevSampleDoubleSample)
{
    double Array[] = {6.1465134, 113.134613, 78.214334, 35.23562346, 23.21342, 56.4527245, 23.24657457, 2.98467};
    double result = asTools::StDev(&Array[0], &Array[7], asSAMPLE);
    EXPECT_FLOAT_EQ(38.05574973, result);
}

TEST(Tools, SdtDevSampleDoubleEntirePop)
{
    double Array[] = {6.1465134, 113.134613, 78.214334, 35.23562346, 23.21342, 56.4527245, 23.24657457, 2.98467};
    double result = asTools::StDev(&Array[0], &Array[7], asENTIRE_POPULATION);
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
