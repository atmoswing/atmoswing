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

#include "asTimeArray.h"
#include "gtest/gtest.h"


TEST(TimeArray, BuildArraySimple)
{
    double start = asTime::GetMJD(1950,1,1);
    double end = asTime::GetMJD(2009,1,1);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    EXPECT_DOUBLE_EQ(start, timearray.GetStart());
    EXPECT_DOUBLE_EQ(end, timearray.GetEnd());
    EXPECT_DOUBLE_EQ(timestephours, timearray.GetTimeStepHours());
    EXPECT_DOUBLE_EQ(timestephours/24, timearray.GetTimeStepDays());

    EXPECT_DOUBLE_EQ(start, timearray[0]);
    EXPECT_DOUBLE_EQ(start+(double)1*6/24, timearray[1]);
    EXPECT_DOUBLE_EQ(start+(double)2*6/24, timearray[2]);
    EXPECT_DOUBLE_EQ(start+(double)3*6/24, timearray[3]);
    EXPECT_DOUBLE_EQ(start+(double)4*6/24, timearray[4]);
    EXPECT_DOUBLE_EQ(start+(double)5*6/24, timearray[5]);
    EXPECT_DOUBLE_EQ(start+(double)10*6/24, timearray[10]);
    EXPECT_DOUBLE_EQ(start+(double)100*6/24, timearray[100]);
    EXPECT_DOUBLE_EQ(start+(double)1000*6/24, timearray[1000]);
    EXPECT_DOUBLE_EQ(start+(double)10000*6/24, timearray[10000]);

    Array1DDouble datetimearray = timearray.GetTimeArray();

    EXPECT_DOUBLE_EQ(start, datetimearray(0));
    EXPECT_DOUBLE_EQ(start+(double)1*6/24, datetimearray(1));
    EXPECT_DOUBLE_EQ(start+(double)2*6/24, datetimearray(2));
    EXPECT_DOUBLE_EQ(start+(double)3*6/24, datetimearray(3));
    EXPECT_DOUBLE_EQ(start+(double)4*6/24, datetimearray(4));
    EXPECT_DOUBLE_EQ(start+(double)5*6/24, datetimearray(5));
    EXPECT_DOUBLE_EQ(start+(double)10*6/24, datetimearray(10));
    EXPECT_DOUBLE_EQ(start+(double)100*6/24, datetimearray(100));
    EXPECT_DOUBLE_EQ(start+(double)1000*6/24, datetimearray(1000));
    EXPECT_DOUBLE_EQ(start+(double)10000*6/24, datetimearray(10000));

    EXPECT_EQ(4*21550+1,datetimearray.rows());
}

TEST(TimeArray, BuildArraySimpleGeneric)
{
    double start = asTime::GetMJD(1950,1,1);
    double end = asTime::GetMJD(2009,1,1);
    double timestephours = 6;
//    double forecastdate = asTime::GetMJD(0,1,1);;
//    double intervaldays = 0;
//    double exclusiondays = 0;
    wxString slctModeString = "Simple";
    asTimeArray timearray(start, end, timestephours, slctModeString);
    timearray.Init();

    EXPECT_DOUBLE_EQ(start, timearray.GetStart());
    EXPECT_DOUBLE_EQ(end, timearray.GetEnd());
    EXPECT_DOUBLE_EQ(timestephours, timearray.GetTimeStepHours());
    EXPECT_DOUBLE_EQ(timestephours/24, timearray.GetTimeStepDays());

    EXPECT_DOUBLE_EQ(start, timearray[0]);
    EXPECT_DOUBLE_EQ(start+(double)1*6/24, timearray[1]);
    EXPECT_DOUBLE_EQ(start+(double)2*6/24, timearray[2]);
    EXPECT_DOUBLE_EQ(start+(double)3*6/24, timearray[3]);
    EXPECT_DOUBLE_EQ(start+(double)4*6/24, timearray[4]);
    EXPECT_DOUBLE_EQ(start+(double)5*6/24, timearray[5]);
    EXPECT_DOUBLE_EQ(start+(double)10*6/24, timearray[10]);
    EXPECT_DOUBLE_EQ(start+(double)100*6/24, timearray[100]);
    EXPECT_DOUBLE_EQ(start+(double)1000*6/24, timearray[1000]);
    EXPECT_DOUBLE_EQ(start+(double)10000*6/24, timearray[10000]);

    Array1DDouble datetimearray = timearray.GetTimeArray();

    EXPECT_DOUBLE_EQ(start, datetimearray(0));
    EXPECT_DOUBLE_EQ(start+(double)1*6/24, datetimearray(1));
    EXPECT_DOUBLE_EQ(start+(double)2*6/24, datetimearray(2));
    EXPECT_DOUBLE_EQ(start+(double)3*6/24, datetimearray(3));
    EXPECT_DOUBLE_EQ(start+(double)4*6/24, datetimearray(4));
    EXPECT_DOUBLE_EQ(start+(double)5*6/24, datetimearray(5));
    EXPECT_DOUBLE_EQ(start+(double)10*6/24, datetimearray(10));
    EXPECT_DOUBLE_EQ(start+(double)100*6/24, datetimearray(100));
    EXPECT_DOUBLE_EQ(start+(double)1000*6/24, datetimearray(1000));
    EXPECT_DOUBLE_EQ(start+(double)10000*6/24, datetimearray(10000));

    EXPECT_EQ(4*21550+1,datetimearray.rows());
}

TEST(TimeArray, BuildArrayDaysIntervalNormal)
{
    double start = asTime::GetMJD(1950,1,1);
    double end = asTime::GetMJD(2008,12,31);
    double timestephours = 6;
    double forecastdate = asTime::GetMJD(2010,06,01);
    double intervaldays = 60;
    double exclusiondays = 100;

    asTimeArray timearray(start, end, timestephours, asTimeArray::DaysInterval);
    timearray.Init(forecastdate, intervaldays, exclusiondays);

    EXPECT_DOUBLE_EQ(intervaldays*24, timearray.GetIntervalHours());
    EXPECT_DOUBLE_EQ(intervaldays, timearray.GetIntervalDays());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,4,2,0,0), timearray.GetFirst());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(2008,7,31,0,0), timearray.GetLast());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,4,2,6,0), timearray[1]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,4,2,12,0), timearray[2]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,4,27,0,0), timearray[100]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,7,30,18,0), timearray[479]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,7,31,0,0), timearray[480]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1951,4,2,0,0), timearray[481]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1951,4,2,6,0), timearray[482]);
}

TEST(TimeArray, BuildArrayDaysIntervalNormalGeneric)
{
    double start = asTime::GetMJD(1950,1,1);
    double end = asTime::GetMJD(2008,12,31);
    double timestephours = 6;
    double forecastdate = asTime::GetMJD(2010,06,01);
    double intervaldays = 60;
    double exclusiondays = 100;
    wxString slctModeString = "DaysInterval";

    asTimeArray timearray(start, end, timestephours, slctModeString);
    timearray.Init(forecastdate, intervaldays, exclusiondays);

    EXPECT_DOUBLE_EQ(intervaldays*24, timearray.GetIntervalHours());
    EXPECT_DOUBLE_EQ(intervaldays, timearray.GetIntervalDays());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,4,2,0,0), timearray.GetFirst());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(2008,7,31,0,0), timearray.GetLast());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,4,2,6,0), timearray[1]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,4,2,12,0), timearray[2]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,4,27,0,0), timearray[100]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,7,30,18,0), timearray[479]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,7,31,0,0), timearray[480]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1951,4,2,0,0), timearray[481]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1951,4,2,6,0), timearray[482]);
}

TEST(TimeArray, BuildArrayDaysIntervalNormalMidday)
{
    double start = asTime::GetMJD(1950,1,1);
    double end = asTime::GetMJD(2008,12,31);
    double timestephours = 6;
    double forecastdate = asTime::GetMJD(2010,06,01,12,00);
    double intervaldays = 60;
    double exclusiondays = 100;

    asTimeArray timearray(start, end, timestephours, asTimeArray::DaysInterval);
    timearray.Init(forecastdate, intervaldays, exclusiondays);

    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,4,2,12,0), timearray.GetFirst());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(2008,7,31,12,0), timearray.GetLast());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,4,2,18,0), timearray[1]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,4,3,0,0), timearray[2]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,4,27,12,0), timearray[100]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,7,31,6,0), timearray[479]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,7,31,12,0), timearray[480]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1951,4,2,12,0), timearray[481]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1951,4,2,18,0), timearray[482]);
}

TEST(TimeArray, BuildArrayDaysIntervalNormalMiddayNotRound)
{
    double start = asTime::GetMJD(1950,1,1);
    double end = asTime::GetMJD(2008,12,31);
    double timestephours = 6;
    double forecastdate = asTime::GetMJD(2010,06,01,10,31);
    double intervaldays = 60;
    double exclusiondays = 100;

    asTimeArray timearray(start, end, timestephours, asTimeArray::DaysInterval);
    timearray.Init(forecastdate, intervaldays, exclusiondays);

    EXPECT_DOUBLE_EQ(0, 24*60*asTime::GetMJD(1950,4,2,10,31)-24*60*timearray.GetFirst());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,4,2,10,31), timearray.GetFirst());
    EXPECT_DOUBLE_EQ(0, 24*60*asTime::GetMJD(2008,7,31,10,31)-24*60*timearray.GetLast());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(2008,7,31,10,31), timearray.GetLast());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,4,2,16,31), timearray[1]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,4,2,22,31), timearray[2]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,4,27,10,31), timearray[100]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,7,31,4,31), timearray[479]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,7,31,10,31), timearray[480]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1951,4,2,10,31), timearray[481]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1951,4,2,16,31), timearray[482]);
}

TEST(TimeArray, BuildArrayDaysIntervalStartSplitted)
{
    double start = asTime::GetMJD(1950,1,1);
    double end = asTime::GetMJD(2008,12,31);
    double timestephours = 6;
    double forecastdate = asTime::GetMJD(2010,02,01,12,00);
    double intervaldays = 60;
    double exclusiondays = 100;

    asTimeArray timearray(start, end, timestephours, asTimeArray::DaysInterval);
    timearray.Init(forecastdate, intervaldays, exclusiondays);

    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,1,1,0,0), timearray.GetFirst());
    EXPECT_DOUBLE_EQ(0, asTime::GetMJD(2008,4,1,12,0)-timearray.GetLast());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(2008,4,1,12,0), timearray.GetLast());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,1,1,6,0), timearray[1]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,1,1,12,0), timearray[2]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,1,26,0,0), timearray[100]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,4,2,12,0), timearray[366]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,12,3,12,0), timearray[367]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,12,3,18,0), timearray[368]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,12,4,0,0), timearray[369]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,12,31,18,0), timearray[366+114]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1951,1,1,0,0), timearray[366+115]);
}

TEST(TimeArray, BuildArrayDaysIntervalEndSplitted)
{
    double start = asTime::GetMJD(1950,1,1,0);
    double end = asTime::GetMJD(2008,12,31,18);
    double timestephours = 6;
    double forecastdate = asTime::GetMJD(2010,12,01,12,00);
    double intervaldays = 60;
    double exclusiondays = 100;

    asTimeArray timearray(start, end, timestephours, asTimeArray::DaysInterval);
    timearray.Init(forecastdate, intervaldays, exclusiondays);

    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,10,2,12,0), timearray.GetFirst());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(2008,12,31,18,0), timearray.GetLast());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,10,2,18,0), timearray[1]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,10,3,0,0), timearray[2]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,10,27,12,0), timearray[100]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1951,1,30,6,0), timearray[479]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1951,1,30,12,0), timearray[480]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1951,10,2,12,0), timearray[481]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1951,10,2,18,0), timearray[482]);
}

TEST(TimeArray, BuildArrayDaysIntervalExclusionPeriod)
{
    double start = asTime::GetMJD(1950,1,1);
    double end = asTime::GetMJD(2008,12,31);
    double timestephours = 6;
    double forecastdate = asTime::GetMJD(2001,06,01);
    double intervaldays = 60;
    double exclusiondays = 100;

    asTimeArray timearray(start, end, timestephours, asTimeArray::DaysInterval);
    timearray.Init(forecastdate, intervaldays, exclusiondays);
    Array1DDouble datetimearray = timearray.GetTimeArray();

    bool foundyear = false;
    double year2001start = asTime::GetMJD(2001,1,1);
    double year2001end = asTime::GetMJD(2001,12,31);

    for (int i=0;i<datetimearray.rows();i++)
    {
        if (datetimearray(i)>year2001start && datetimearray(i)<year2001end)
        {
            foundyear = true;
        }
    }

    EXPECT_FALSE(foundyear);
}

TEST(TimeArray, BuildArraySeason)
{
    double start = asTime::GetMJD(1950,1,1);
    double end = asTime::GetMJD(2008,12,31);
    double timestephours = 6;
    double forecastdate = asTime::GetMJD(2010,07,01,12,00);
    double exclusiondays = 100;

    asTimeArray timearray(start, end, timestephours, asTimeArray::SameSeason);
    timearray.Init(forecastdate, exclusiondays);

    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,6,1,0,0), timearray.GetFirst());
    EXPECT_DOUBLE_EQ(0, asTime::GetMJD(2008,8,31,18,0)-timearray.GetLast());
    EXPECT_DOUBLE_EQ(0, 24*asTime::GetMJD(2008,8,31,18,0)-24*timearray.GetLast());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(2008,8,31,18,0), timearray.GetLast());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,6,1,6,0), timearray[1]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,6,1,12,0), timearray[2]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,8,31,18,0), timearray[367]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1951,6,1,0,0), timearray[368]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1951,6,1,6,0), timearray[369]);
}

TEST(TimeArray, BuildArraySeasonGeneric)
{
    double start = asTime::GetMJD(1950,1,1);
    double end = asTime::GetMJD(2008,12,31);
    double timestephours = 6;
    double forecastdate = asTime::GetMJD(2010,07,01,12,00);
    double exclusiondays = 100;
    wxString slctModeString = "SameSeason";

    asTimeArray timearray(start, end, timestephours, slctModeString);
    timearray.Init(forecastdate, exclusiondays);

    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,6,1,0,0), timearray.GetFirst());
    EXPECT_DOUBLE_EQ(0, asTime::GetMJD(2008,8,31,18,0)-timearray.GetLast());
    EXPECT_DOUBLE_EQ(0, 24*asTime::GetMJD(2008,8,31,18,0)-24*timearray.GetLast());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(2008,8,31,18,0), timearray.GetLast());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,6,1,6,0), timearray[1]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,6,1,12,0), timearray[2]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,8,31,18,0), timearray[367]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1951,6,1,0,0), timearray[368]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1951,6,1,6,0), timearray[369]);
}

TEST(TimeArray, BuildArraySeasonNotRound)
{
    double start = asTime::GetMJD(1950,1,1,0);
    double end = asTime::GetMJD(2008,12,31,18);
    double timestephours = 6;
    double forecastdate = asTime::GetMJD(2010,07,01,14,32);
    double exclusiondays = 100;

    asTimeArray timearray(start, end, timestephours, asTimeArray::SameSeason);
    timearray.Init(forecastdate, exclusiondays);

    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,6,1,2,32), timearray.GetFirst());
    EXPECT_DOUBLE_EQ(0, asTime::GetMJD(2008,8,31,20,32)-timearray.GetLast());
    EXPECT_DOUBLE_EQ(0, 24*asTime::GetMJD(2008,8,31,20,32)-24*timearray.GetLast());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(2008,8,31,20,32), timearray.GetLast());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,6,1,8,32), timearray[1]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,6,1,14,32), timearray[2]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,8,31,20,32), timearray[367]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1951,6,1,2,32), timearray[368]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1951,6,1,8,32), timearray[369]);
}

TEST(TimeArray, BuildArraySeasonDec)
{
    double start = asTime::GetMJD(1950,1,1,0);
    double end = asTime::GetMJD(2008,12,31,18);
    double timestephours = 6;
    double forecastdate = asTime::GetMJD(2010,12,01,12,00);
    double exclusiondays = 100;

    asTimeArray timearray(start, end, timestephours, asTimeArray::SameSeason);
    timearray.Init(forecastdate, exclusiondays);

    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,1,1,0,0), timearray.GetFirst());
    EXPECT_DOUBLE_EQ(0, asTime::GetMJD(2008,12,31,18,0)-timearray.GetLast());
    EXPECT_DOUBLE_EQ(0, 24*asTime::GetMJD(2008,12,31,18,0)-24*timearray.GetLast());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(2008,12,31,18,0), timearray.GetLast());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,1,1,6,0), timearray[1]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,1,1,12,0), timearray[2]);
}

TEST(TimeArray, BuildArraySeasonJan)
{
    double start = asTime::GetMJD(1950,1,1,0);
    double end = asTime::GetMJD(2008,12,31,18);
    double timestephours = 6;
    double forecastdate = asTime::GetMJD(2010,01,01,12,00);
    double exclusiondays = 100;

    asTimeArray timearray(start, end, timestephours, asTimeArray::SameSeason);
    timearray.Init(forecastdate, exclusiondays);

    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,1,1,0,0), timearray.GetFirst());
    EXPECT_DOUBLE_EQ(0, asTime::GetMJD(2008,12,31,18,0)-timearray.GetLast());
    EXPECT_DOUBLE_EQ(0, 24*asTime::GetMJD(2008,12,31,18,0)-24*timearray.GetLast());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(2008,12,31,18,0), timearray.GetLast());
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,1,1,6,0), timearray[1]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,1,1,12,0), timearray[2]);
}

TEST(TimeArray, GetFirstDayHour)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(2008,12,31,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    EXPECT_FLOAT_EQ(12.5, timearray.GetFirstDayHour());
}

TEST(TimeArray, GetLastDayHour)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(2008,12,31,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    EXPECT_FLOAT_EQ(18.5, timearray.GetLastDayHour());
}

TEST(TimeArray, GetFirstDayYear)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(2008,12,31,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    EXPECT_EQ(1950, timearray.GetFirstDayYear());
}

TEST(TimeArray, GetLastDayYear)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(2008,12,31,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    EXPECT_EQ(2008, timearray.GetLastDayYear());
}

TEST(TimeArray, GetSize)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(1950,1,2,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    EXPECT_EQ(6, timearray.GetSize());
}

TEST(TimeArray, OperatorOverloading)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(1950,1,2,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,1,1,12,30), timearray[0]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,1,1,18,30), timearray[1]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,1,2,00,30), timearray[2]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,1,2,06,30), timearray[3]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,1,2,12,30), timearray[4]);
    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,1,2,18,30), timearray[5]);
}

TEST(TimeArray, GetFirst)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(1950,1,2,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,1,1,12,30), timearray.GetFirst());
}

TEST(TimeArray, GetLast)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(1950,1,2,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    EXPECT_DOUBLE_EQ(asTime::GetMJD(1950,1,2,18,30), timearray.GetLast());
}

TEST(TimeArray, GetIndexFirstAfter)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(1950,1,2,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    EXPECT_DOUBLE_EQ(2, timearray.GetIndexFirstAfter(asTime::GetMJD(1950,1,1,19,30)));
}

TEST(TimeArray, GetIndexFirstAfterEqual)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(1950,1,2,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    EXPECT_DOUBLE_EQ(1, timearray.GetIndexFirstAfter(asTime::GetMJD(1950,1,1,18,30)));
}

TEST(TimeArray, GetIndexFirstBefore)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(1950,1,2,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    EXPECT_DOUBLE_EQ(2, timearray.GetIndexFirstBefore(asTime::GetMJD(1950,1,2,05,30)));
}

TEST(TimeArray, GetIndexFirstBeforeEqual)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(1950,1,2,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    EXPECT_DOUBLE_EQ(3, timearray.GetIndexFirstBefore(asTime::GetMJD(1950,1,2,06,30)));
}
