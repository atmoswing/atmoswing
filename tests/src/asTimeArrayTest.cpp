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
 */

#include <gtest/gtest.h>
#include "asTimeArray.h"

TEST(TimeArray, BuildArraySimple) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2009, 1, 1);
  double timeStep = 6;
  asTimeArray timeArray(start, end, timeStep, asTimeArray::Simple);
  timeArray.Init();

  EXPECT_DOUBLE_EQ(start, timeArray.GetStart());
  EXPECT_DOUBLE_EQ(end, timeArray.GetEnd());
  EXPECT_DOUBLE_EQ(timeStep, timeArray.GetTimeStepHours());
  EXPECT_DOUBLE_EQ(timeStep / 24, timeArray.GetTimeStepDays());

  EXPECT_DOUBLE_EQ(start, timeArray[0]);
  EXPECT_DOUBLE_EQ(start + (double)1 * 6 / 24, timeArray[1]);
  EXPECT_DOUBLE_EQ(start + (double)2 * 6 / 24, timeArray[2]);
  EXPECT_DOUBLE_EQ(start + (double)3 * 6 / 24, timeArray[3]);
  EXPECT_DOUBLE_EQ(start + (double)4 * 6 / 24, timeArray[4]);
  EXPECT_DOUBLE_EQ(start + (double)5 * 6 / 24, timeArray[5]);
  EXPECT_DOUBLE_EQ(start + (double)10 * 6 / 24, timeArray[10]);
  EXPECT_DOUBLE_EQ(start + (double)100 * 6 / 24, timeArray[100]);
  EXPECT_DOUBLE_EQ(start + (double)1000 * 6 / 24, timeArray[1000]);
  EXPECT_DOUBLE_EQ(start + (double)10000 * 6 / 24, timeArray[10000]);

  a1d datetimeArray = timeArray.GetTimeArray();

  EXPECT_DOUBLE_EQ(start, datetimeArray(0));
  EXPECT_DOUBLE_EQ(start + (double)1 * 6 / 24, datetimeArray(1));
  EXPECT_DOUBLE_EQ(start + (double)2 * 6 / 24, datetimeArray(2));
  EXPECT_DOUBLE_EQ(start + (double)3 * 6 / 24, datetimeArray(3));
  EXPECT_DOUBLE_EQ(start + (double)4 * 6 / 24, datetimeArray(4));
  EXPECT_DOUBLE_EQ(start + (double)5 * 6 / 24, datetimeArray(5));
  EXPECT_DOUBLE_EQ(start + (double)10 * 6 / 24, datetimeArray(10));
  EXPECT_DOUBLE_EQ(start + (double)100 * 6 / 24, datetimeArray(100));
  EXPECT_DOUBLE_EQ(start + (double)1000 * 6 / 24, datetimeArray(1000));
  EXPECT_DOUBLE_EQ(start + (double)10000 * 6 / 24, datetimeArray(10000));

  EXPECT_EQ(4 * 21550 + 1, datetimeArray.rows());
}

TEST(TimeArray, BuildArraySimpleGeneric) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2009, 1, 1);
  double timeStep = 6;
  wxString slctModeString = "Simple";
  asTimeArray timeArray(start, end, timeStep, slctModeString);
  timeArray.Init();

  EXPECT_DOUBLE_EQ(start, timeArray.GetStart());
  EXPECT_DOUBLE_EQ(end, timeArray.GetEnd());
  EXPECT_DOUBLE_EQ(timeStep, timeArray.GetTimeStepHours());
  EXPECT_DOUBLE_EQ(timeStep / 24, timeArray.GetTimeStepDays());

  EXPECT_DOUBLE_EQ(start, timeArray[0]);
  EXPECT_DOUBLE_EQ(start + (double)1 * 6 / 24, timeArray[1]);
  EXPECT_DOUBLE_EQ(start + (double)2 * 6 / 24, timeArray[2]);
  EXPECT_DOUBLE_EQ(start + (double)3 * 6 / 24, timeArray[3]);
  EXPECT_DOUBLE_EQ(start + (double)4 * 6 / 24, timeArray[4]);
  EXPECT_DOUBLE_EQ(start + (double)5 * 6 / 24, timeArray[5]);
  EXPECT_DOUBLE_EQ(start + (double)10 * 6 / 24, timeArray[10]);
  EXPECT_DOUBLE_EQ(start + (double)100 * 6 / 24, timeArray[100]);
  EXPECT_DOUBLE_EQ(start + (double)1000 * 6 / 24, timeArray[1000]);
  EXPECT_DOUBLE_EQ(start + (double)10000 * 6 / 24, timeArray[10000]);

  a1d datetimeArray = timeArray.GetTimeArray();

  EXPECT_DOUBLE_EQ(start, datetimeArray(0));
  EXPECT_DOUBLE_EQ(start + (double)1 * 6 / 24, datetimeArray(1));
  EXPECT_DOUBLE_EQ(start + (double)2 * 6 / 24, datetimeArray(2));
  EXPECT_DOUBLE_EQ(start + (double)3 * 6 / 24, datetimeArray(3));
  EXPECT_DOUBLE_EQ(start + (double)4 * 6 / 24, datetimeArray(4));
  EXPECT_DOUBLE_EQ(start + (double)5 * 6 / 24, datetimeArray(5));
  EXPECT_DOUBLE_EQ(start + (double)10 * 6 / 24, datetimeArray(10));
  EXPECT_DOUBLE_EQ(start + (double)100 * 6 / 24, datetimeArray(100));
  EXPECT_DOUBLE_EQ(start + (double)1000 * 6 / 24, datetimeArray(1000));
  EXPECT_DOUBLE_EQ(start + (double)10000 * 6 / 24, datetimeArray(10000));

  EXPECT_EQ(4 * 21550 + 1, datetimeArray.rows());
}

TEST(TimeArray, BuildArrayDaysIntervalNormal) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2008, 12, 31);
  double timeStep = 6;
  double targetDate = asTime::GetMJD(2010, 06, 01);
  double intervaldays = 60;
  double exclusionDays = 100;

  asTimeArray timeArray(start, end, timeStep, asTimeArray::DaysInterval);
  timeArray.Init(targetDate, intervaldays, exclusionDays);

  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 2, 0, 0), timeArray.GetFirst());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(2008, 7, 31, 0, 0), timeArray.GetLast());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 2, 6, 0), timeArray[1]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 2, 12, 0), timeArray[2]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 27, 0, 0), timeArray[100]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 7, 30, 18, 0), timeArray[479]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 7, 31, 0, 0), timeArray[480]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1951, 4, 2, 0, 0), timeArray[481]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1951, 4, 2, 6, 0), timeArray[482]);
}

TEST(TimeArray, BuildArrayDaysIntervalNormalGeneric) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2008, 12, 31);
  double timeStep = 6;
  double targetDate = asTime::GetMJD(2010, 06, 01);
  double intervaldays = 60;
  double exclusionDays = 100;
  wxString slctModeString = "DaysInterval";

  asTimeArray timeArray(start, end, timeStep, slctModeString);
  timeArray.Init(targetDate, intervaldays, exclusionDays);

  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 2, 0, 0), timeArray.GetFirst());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(2008, 7, 31, 0, 0), timeArray.GetLast());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 2, 6, 0), timeArray[1]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 2, 12, 0), timeArray[2]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 27, 0, 0), timeArray[100]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 7, 30, 18, 0), timeArray[479]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 7, 31, 0, 0), timeArray[480]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1951, 4, 2, 0, 0), timeArray[481]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1951, 4, 2, 6, 0), timeArray[482]);
}

TEST(TimeArray, BuildArrayDaysIntervalNormalMidday) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2008, 12, 31);
  double timeStep = 6;
  double targetDate = asTime::GetMJD(2010, 06, 01, 12, 00);
  double intervaldays = 60;
  double exclusionDays = 100;

  asTimeArray timeArray(start, end, timeStep, asTimeArray::DaysInterval);
  timeArray.Init(targetDate, intervaldays, exclusionDays);

  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 2, 12, 0), timeArray.GetFirst());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(2008, 7, 31, 12, 0), timeArray.GetLast());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 2, 18, 0), timeArray[1]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 3, 0, 0), timeArray[2]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 27, 12, 0), timeArray[100]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 7, 31, 6, 0), timeArray[479]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 7, 31, 12, 0), timeArray[480]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1951, 4, 2, 12, 0), timeArray[481]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1951, 4, 2, 18, 0), timeArray[482]);
}

TEST(TimeArray, BuildArrayDaysIntervalNormalMiddayNotRound) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2008, 12, 31);
  double timeStep = 6;
  double targetDate = asTime::GetMJD(2010, 06, 01, 10, 31);
  double intervaldays = 60;
  double exclusionDays = 100;

  asTimeArray timeArray(start, end, timeStep, asTimeArray::DaysInterval);
  timeArray.Init(targetDate, intervaldays, exclusionDays);

  EXPECT_DOUBLE_EQ(0, 24 * 60 * asTime::GetMJD(1950, 4, 2, 10, 31) - 24 * 60 * timeArray.GetFirst());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 2, 10, 31), timeArray.GetFirst());
  EXPECT_DOUBLE_EQ(0, 24 * 60 * asTime::GetMJD(2008, 7, 31, 10, 31) - 24 * 60 * timeArray.GetLast());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(2008, 7, 31, 10, 31), timeArray.GetLast());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 2, 16, 31), timeArray[1]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 2, 22, 31), timeArray[2]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 27, 10, 31), timeArray[100]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 7, 31, 4, 31), timeArray[479]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 7, 31, 10, 31), timeArray[480]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1951, 4, 2, 10, 31), timeArray[481]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1951, 4, 2, 16, 31), timeArray[482]);
}

TEST(TimeArray, BuildArrayDaysIntervalStartSplitted) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2008, 12, 31);
  double timeStep = 6;
  double targetDate = asTime::GetMJD(2010, 02, 01, 12, 00);
  double intervaldays = 60;
  double exclusionDays = 100;

  asTimeArray timeArray(start, end, timeStep, asTimeArray::DaysInterval);
  timeArray.Init(targetDate, intervaldays, exclusionDays);

  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 1, 0, 0), timeArray.GetFirst());
  EXPECT_DOUBLE_EQ(0, asTime::GetMJD(2008, 4, 1, 12, 0) - timeArray.GetLast());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(2008, 4, 1, 12, 0), timeArray.GetLast());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 1, 6, 0), timeArray[1]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 1, 12, 0), timeArray[2]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 26, 0, 0), timeArray[100]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 2, 12, 0), timeArray[366]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 12, 3, 12, 0), timeArray[367]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 12, 3, 18, 0), timeArray[368]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 12, 4, 0, 0), timeArray[369]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 12, 31, 18, 0), timeArray[366 + 114]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1951, 1, 1, 0, 0), timeArray[366 + 115]);
}

TEST(TimeArray, BuildArrayDaysIntervalEndSplitted) {
  double start = asTime::GetMJD(1950, 1, 1, 0);
  double end = asTime::GetMJD(2008, 12, 31, 18);
  double timeStep = 6;
  double targetDate = asTime::GetMJD(2010, 12, 01, 12, 00);
  double intervaldays = 60;
  double exclusionDays = 100;

  asTimeArray timeArray(start, end, timeStep, asTimeArray::DaysInterval);
  timeArray.Init(targetDate, intervaldays, exclusionDays);

  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 10, 2, 12, 0), timeArray.GetFirst());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(2008, 12, 31, 18, 0), timeArray.GetLast());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 10, 2, 18, 0), timeArray[1]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 10, 3, 0, 0), timeArray[2]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 10, 27, 12, 0), timeArray[100]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1951, 1, 30, 6, 0), timeArray[479]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1951, 1, 30, 12, 0), timeArray[480]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1951, 10, 2, 12, 0), timeArray[481]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1951, 10, 2, 18, 0), timeArray[482]);
}

TEST(TimeArray, BuildArrayDaysIntervalExclusionPeriod) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2008, 12, 31);
  double timeStep = 6;
  double targetDate = asTime::GetMJD(2001, 06, 01);
  double intervaldays = 60;
  double exclusionDays = 100;

  asTimeArray timeArray(start, end, timeStep, asTimeArray::DaysInterval);
  timeArray.Init(targetDate, intervaldays, exclusionDays);
  a1d datetimeArray = timeArray.GetTimeArray();

  bool foundyear = false;
  double year2001start = asTime::GetMJD(2001, 1, 1);
  double year2001end = asTime::GetMJD(2001, 12, 31);

  for (int i = 0; i < datetimeArray.rows(); i++) {
    if (datetimeArray(i) > year2001start && datetimeArray(i) < year2001end) {
      foundyear = true;
    }
  }

  EXPECT_FALSE(foundyear);
}

TEST(TimeArray, BuildArraySeasonDJF6h) {
  double start = asTime::GetMJD(1950, 1, 1, 0);
  double end = asTime::GetMJD(2008, 12, 31, 18);
  double timeStep = 6;

  asTimeArray timeArray(start, end, timeStep, asTimeArray::DJF);
  timeArray.Init();

  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 1, 0, 0), timeArray.GetFirst());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(2008, 12, 31, 18, 0), timeArray.GetLast());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 1, 6, 0), timeArray[1]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 1, 12, 0), timeArray[2]);
}

TEST(TimeArray, BuildArraySeasonJJA6h) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2008, 12, 31);
  double timeStep = 6;

  asTimeArray timeArray(start, end, timeStep, asTimeArray::JJA);
  timeArray.Init();

  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 6, 1, 6, 0), timeArray.GetFirst());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(2008, 8, 31, 18, 0), timeArray.GetLast());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 6, 1, 12, 0), timeArray[1]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 6, 1, 18, 0), timeArray[2]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 8, 31, 18, 0), timeArray[366]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1951, 6, 1, 6, 0), timeArray[367]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1951, 6, 1, 12, 0), timeArray[368]);
}

TEST(TimeArray, BuildArraySeasonDJF24h) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2008, 12, 31);
  double timeStep = 24;

  asTimeArray timeArray(start, end, timeStep, asTimeArray::DJF);
  timeArray.Init();

  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 1, 0, 0), timeArray.GetFirst());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(2008, 12, 31, 0, 0), timeArray.GetLast());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 2, 0, 0), timeArray[1]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 3, 0, 0), timeArray[2]);
}

TEST(TimeArray, BuildArraySeasonMAM24h) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2008, 12, 31);
  double timeStep = 24;

  asTimeArray timeArray(start, end, timeStep, asTimeArray::MAM);
  timeArray.Init();

  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 3, 1, 0, 0), timeArray.GetFirst());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(2008, 5, 31, 0, 0), timeArray.GetLast());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 3, 2, 0, 0), timeArray[1]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 3, 3, 0, 0), timeArray[2]);
}

TEST(TimeArray, BuildArraySeasonJJA24h) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2008, 12, 31);
  double timeStep = 24;

  asTimeArray timeArray(start, end, timeStep, asTimeArray::JJA);
  timeArray.Init();

  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 6, 1, 0, 0), timeArray.GetFirst());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(2008, 8, 31, 0, 0), timeArray.GetLast());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 6, 2, 0, 0), timeArray[1]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 6, 3, 0, 0), timeArray[2]);
}

TEST(TimeArray, BuildArraySeasonSON24h) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2008, 12, 31);
  double timeStep = 24;

  asTimeArray timeArray(start, end, timeStep, asTimeArray::SON);
  timeArray.Init();

  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 9, 1, 0, 0), timeArray.GetFirst());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(2008, 11, 30, 0, 0), timeArray.GetLast());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 9, 2, 0, 0), timeArray[1]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 9, 3, 0, 0), timeArray[2]);
}

TEST(TimeArray, BuildArrayAprilToSeptember) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2008, 12, 31);
  double timeStep = 24;

  asTimeArray timeArray(start, end, timeStep, "april_to_september");
  timeArray.Init();

  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 1, 0, 0), timeArray.GetFirst());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(2008, 9, 30, 0, 0), timeArray.GetLast());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 2, 0, 0), timeArray[1]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 3, 0, 0), timeArray[2]);
}

TEST(TimeArray, BuildArrayAprilToSeptember2) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2008, 12, 31);
  double timeStep = 24;

  asTimeArray timeArray(start, end, timeStep, "AprilToSeptember");
  timeArray.Init();

  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 1, 0, 0), timeArray.GetFirst());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(2008, 9, 30, 0, 0), timeArray.GetLast());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 2, 0, 0), timeArray[1]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 4, 3, 0, 0), timeArray[2]);
}

TEST(TimeArray, BuildArrayJanuaryToDecember) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2008, 12, 31);
  double timeStep = 24;

  asTimeArray timeArray(start, end, timeStep, "january_to_december");
  timeArray.Init();

  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 1, 0, 0), timeArray.GetFirst());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(2008, 12, 31, 0, 0), timeArray.GetLast());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 2, 0, 0), timeArray[1]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 3, 0, 0), timeArray[2]);
}

TEST(TimeArray, BuildArrayJuneToJuly) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2008, 12, 31);
  double timeStep = 24;

  asTimeArray timeArray(start, end, timeStep, "june_to_july");
  timeArray.Init();

  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 6, 1, 0, 0), timeArray.GetFirst());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(2008, 7, 31, 0, 0), timeArray.GetLast());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 6, 2, 0, 0), timeArray[1]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 6, 3, 0, 0), timeArray[2]);
}

TEST(TimeArray, BuildArrayJuneToMarch) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2008, 12, 31);
  double timeStep = 24;

  asTimeArray timeArray(start, end, timeStep, "june_to_march");
  timeArray.Init();

  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 1, 0, 0), timeArray.GetFirst());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(2008, 12, 31, 0, 0), timeArray.GetLast());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 2, 0, 0), timeArray[1]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 3, 0, 0), timeArray[2]);
}

TEST(TimeArray, BuildArrayJuneToJanuary) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2008, 12, 31);
  double timeStep = 24;

  asTimeArray timeArray(start, end, timeStep, "june_to_january");
  timeArray.Init();

  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 1, 0, 0), timeArray.GetFirst());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(2008, 12, 31, 0, 0), timeArray.GetLast());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 2, 0, 0), timeArray[1]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 3, 0, 0), timeArray[2]);
}

TEST(TimeArray, BuildArrayDecemberToJanuary) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2008, 12, 31);
  double timeStep = 24;

  asTimeArray timeArray(start, end, timeStep, "december_to_january");
  timeArray.Init();

  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 1, 0, 0), timeArray.GetFirst());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(2008, 12, 31, 0, 0), timeArray.GetLast());
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 2, 0, 0), timeArray[1]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 3, 0, 0), timeArray[2]);
}

TEST(TimeArray, GetFirstDayHour) {
  double start = asTime::GetMJD(1950, 1, 1, 12, 30);
  double end = asTime::GetMJD(2008, 12, 31, 18, 30);
  double timeStep = 6;
  asTimeArray timeArray(start, end, timeStep, asTimeArray::Simple);
  timeArray.Init();

  EXPECT_FLOAT_EQ(12.5, timeArray.GetStartingHour());
}

TEST(TimeArray, GetLastDayHour) {
  double start = asTime::GetMJD(1950, 1, 1, 12, 30);
  double end = asTime::GetMJD(2008, 12, 31, 18, 30);
  double timeStep = 6;
  asTimeArray timeArray(start, end, timeStep, asTimeArray::Simple);
  timeArray.Init();

  EXPECT_FLOAT_EQ(18.5, timeArray.GetEndingHour());
}

TEST(TimeArray, GetFirstDayYear) {
  double start = asTime::GetMJD(1950, 1, 1, 12, 30);
  double end = asTime::GetMJD(2008, 12, 31, 18, 30);
  double timeStep = 6;
  asTimeArray timeArray(start, end, timeStep, asTimeArray::Simple);
  timeArray.Init();

  EXPECT_EQ(1950, timeArray.GetStartingYear());
}

TEST(TimeArray, GetLastDayYear) {
  double start = asTime::GetMJD(1950, 1, 1, 12, 30);
  double end = asTime::GetMJD(2008, 12, 31, 18, 30);
  double timeStep = 6;
  asTimeArray timeArray(start, end, timeStep, asTimeArray::Simple);
  timeArray.Init();

  EXPECT_EQ(2008, timeArray.GetEndingYear());
}

TEST(TimeArray, GetSize) {
  double start = asTime::GetMJD(1950, 1, 1, 12, 30);
  double end = asTime::GetMJD(1950, 1, 2, 18, 30);
  double timeStep = 6;
  asTimeArray timeArray(start, end, timeStep, asTimeArray::Simple);
  timeArray.Init();

  EXPECT_EQ(6, timeArray.GetSize());
}

TEST(TimeArray, OperatorOverloading) {
  double start = asTime::GetMJD(1950, 1, 1, 12, 30);
  double end = asTime::GetMJD(1950, 1, 2, 18, 30);
  double timeStep = 6;
  asTimeArray timeArray(start, end, timeStep, asTimeArray::Simple);
  timeArray.Init();

  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 1, 12, 30), timeArray[0]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 1, 18, 30), timeArray[1]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 2, 00, 30), timeArray[2]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 2, 06, 30), timeArray[3]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 2, 12, 30), timeArray[4]);
  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 2, 18, 30), timeArray[5]);
}

TEST(TimeArray, GetFirst) {
  double start = asTime::GetMJD(1950, 1, 1, 12, 30);
  double end = asTime::GetMJD(1950, 1, 2, 18, 30);
  double timeStep = 6;
  asTimeArray timeArray(start, end, timeStep, asTimeArray::Simple);
  timeArray.Init();

  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 1, 12, 30), timeArray.GetFirst());
}

TEST(TimeArray, GetLast) {
  double start = asTime::GetMJD(1950, 1, 1, 12, 30);
  double end = asTime::GetMJD(1950, 1, 2, 18, 30);
  double timeStep = 6;
  asTimeArray timeArray(start, end, timeStep, asTimeArray::Simple);
  timeArray.Init();

  EXPECT_DOUBLE_EQ(asTime::GetMJD(1950, 1, 2, 18, 30), timeArray.GetLast());
}

TEST(TimeArray, GetIndexFirstAfter) {
  double start = asTime::GetMJD(1950, 1, 1, 12, 30);
  double end = asTime::GetMJD(1950, 1, 2, 18, 30);
  double timeStep = 6;
  asTimeArray timeArray(start, end, timeStep, asTimeArray::Simple);
  timeArray.Init();

  EXPECT_DOUBLE_EQ(2, timeArray.GetIndexFirstAfter(asTime::GetMJD(1950, 1, 1, 19, 30), 6));
}

TEST(TimeArray, GetIndexFirstAfterEqual) {
  double start = asTime::GetMJD(1950, 1, 1, 12, 30);
  double end = asTime::GetMJD(1950, 1, 2, 18, 30);
  double timeStep = 6;
  asTimeArray timeArray(start, end, timeStep, asTimeArray::Simple);
  timeArray.Init();

  EXPECT_DOUBLE_EQ(1, timeArray.GetIndexFirstAfter(asTime::GetMJD(1950, 1, 1, 18, 30), 6));
}

TEST(TimeArray, GetIndexFirstBefore) {
  double start = asTime::GetMJD(1950, 1, 1, 12, 30);
  double end = asTime::GetMJD(1950, 1, 2, 18, 30);
  double timeStep = 6;
  asTimeArray timeArray(start, end, timeStep, asTimeArray::Simple);
  timeArray.Init();

  EXPECT_DOUBLE_EQ(2, timeArray.GetIndexFirstBefore(asTime::GetMJD(1950, 1, 2, 05, 30), 6));
}

TEST(TimeArray, GetIndexFirstBeforeEqual) {
  double start = asTime::GetMJD(1950, 1, 1, 12, 30);
  double end = asTime::GetMJD(1950, 1, 2, 18, 30);
  double timeStep = 6;
  asTimeArray timeArray(start, end, timeStep, asTimeArray::Simple);
  timeArray.Init();

  EXPECT_DOUBLE_EQ(3, timeArray.GetIndexFirstBefore(asTime::GetMJD(1950, 1, 2, 06, 30), 6));
}

TEST(TimeArray, PopFirst) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2009, 1, 1);
  double timeStep = 24;
  asTimeArray timeArray(start, end, timeStep, asTimeArray::Simple);
  timeArray.Init();

  timeArray.Pop(0);

  EXPECT_DOUBLE_EQ(start + 1, timeArray.GetStart());
  EXPECT_DOUBLE_EQ(end, timeArray.GetEnd());

  EXPECT_DOUBLE_EQ(start + 1, timeArray[0]);
  EXPECT_DOUBLE_EQ(start + (double)2, timeArray[1]);
  EXPECT_DOUBLE_EQ(start + (double)3, timeArray[2]);
  EXPECT_DOUBLE_EQ(start + (double)11, timeArray[10]);
  EXPECT_DOUBLE_EQ(start + (double)101, timeArray[100]);

  EXPECT_EQ(21550, timeArray.GetSize());
}

TEST(TimeArray, PopLast) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2009, 1, 1);
  double timeStep = 24;
  asTimeArray timeArray(start, end, timeStep, asTimeArray::Simple);
  timeArray.Init();

  timeArray.Pop(timeArray.GetSize() - 1);

  EXPECT_DOUBLE_EQ(start, timeArray.GetStart());
  EXPECT_DOUBLE_EQ(end - 1, timeArray.GetEnd());

  EXPECT_DOUBLE_EQ(start, timeArray[0]);
  EXPECT_DOUBLE_EQ(start + (double)1, timeArray[1]);
  EXPECT_DOUBLE_EQ(start + (double)2, timeArray[2]);
  EXPECT_DOUBLE_EQ(start + (double)10, timeArray[10]);
  EXPECT_DOUBLE_EQ(start + (double)100, timeArray[100]);

  EXPECT_EQ(21550, timeArray.GetSize());
}

TEST(TimeArray, PopMiddle) {
  double start = asTime::GetMJD(1950, 1, 1);
  double end = asTime::GetMJD(2009, 1, 1);
  double timeStep = 24;
  asTimeArray timeArray(start, end, timeStep, asTimeArray::Simple);
  timeArray.Init();

  timeArray.Pop(10);

  EXPECT_DOUBLE_EQ(start, timeArray.GetStart());
  EXPECT_DOUBLE_EQ(end, timeArray.GetEnd());

  EXPECT_DOUBLE_EQ(start, timeArray[0]);
  EXPECT_DOUBLE_EQ(start + (double)1, timeArray[1]);
  EXPECT_DOUBLE_EQ(start + (double)2, timeArray[2]);
  EXPECT_DOUBLE_EQ(start + (double)11, timeArray[10]);
  EXPECT_DOUBLE_EQ(start + (double)101, timeArray[100]);

  EXPECT_EQ(21550, timeArray.GetSize());
}