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

#include "include_tests.h"
#include "asTimeArray.h"

#include "UnitTest++.h"

namespace
{

TEST(BuildArraySimple)
{
	wxPrintf("Testing time arrays...\n");
	
    double start = asTime::GetMJD(1950,1,1);
    double end = asTime::GetMJD(2009,1,1);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    CHECK_CLOSE(start, timearray.GetStart(), 0.000001);
    CHECK_CLOSE(end, timearray.GetEnd(), 0.000001);
    CHECK_CLOSE(timestephours, timearray.GetTimeStepHours(), 0.000001);
    CHECK_CLOSE(timestephours/24, timearray.GetTimeStepDays(), 0.000001);

    CHECK_CLOSE(start, timearray[0], 0.000001);
    CHECK_CLOSE(start+(double)1*6/24, timearray[1], 0.000001);
    CHECK_CLOSE(start+(double)2*6/24, timearray[2], 0.000001);
    CHECK_CLOSE(start+(double)3*6/24, timearray[3], 0.000001);
    CHECK_CLOSE(start+(double)4*6/24, timearray[4], 0.000001);
    CHECK_CLOSE(start+(double)5*6/24, timearray[5], 0.000001);
    CHECK_CLOSE(start+(double)10*6/24, timearray[10], 0.000001);
    CHECK_CLOSE(start+(double)100*6/24, timearray[100], 0.000001);
    CHECK_CLOSE(start+(double)1000*6/24, timearray[1000], 0.000001);
    CHECK_CLOSE(start+(double)10000*6/24, timearray[10000], 0.000001);

    Array1DDouble datetimearray = timearray.GetTimeArray();

    CHECK_CLOSE(start, datetimearray(0), 0.000001);
    CHECK_CLOSE(start+(double)1*6/24, datetimearray(1), 0.000001);
    CHECK_CLOSE(start+(double)2*6/24, datetimearray(2), 0.000001);
    CHECK_CLOSE(start+(double)3*6/24, datetimearray(3), 0.000001);
    CHECK_CLOSE(start+(double)4*6/24, datetimearray(4), 0.000001);
    CHECK_CLOSE(start+(double)5*6/24, datetimearray(5), 0.000001);
    CHECK_CLOSE(start+(double)10*6/24, datetimearray(10), 0.000001);
    CHECK_CLOSE(start+(double)100*6/24, datetimearray(100), 0.000001);
    CHECK_CLOSE(start+(double)1000*6/24, datetimearray(1000), 0.000001);
    CHECK_CLOSE(start+(double)10000*6/24, datetimearray(10000), 0.000001);

    CHECK_EQUAL(4*21550+1,datetimearray.rows());
}

TEST(BuildArraySimpleGeneric)
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

    CHECK_CLOSE(start, timearray.GetStart(), 0.000001);
    CHECK_CLOSE(end, timearray.GetEnd(), 0.000001);
    CHECK_CLOSE(timestephours, timearray.GetTimeStepHours(), 0.000001);
    CHECK_CLOSE(timestephours/24, timearray.GetTimeStepDays(), 0.000001);

    CHECK_CLOSE(start, timearray[0], 0.000001);
    CHECK_CLOSE(start+(double)1*6/24, timearray[1], 0.000001);
    CHECK_CLOSE(start+(double)2*6/24, timearray[2], 0.000001);
    CHECK_CLOSE(start+(double)3*6/24, timearray[3], 0.000001);
    CHECK_CLOSE(start+(double)4*6/24, timearray[4], 0.000001);
    CHECK_CLOSE(start+(double)5*6/24, timearray[5], 0.000001);
    CHECK_CLOSE(start+(double)10*6/24, timearray[10], 0.000001);
    CHECK_CLOSE(start+(double)100*6/24, timearray[100], 0.000001);
    CHECK_CLOSE(start+(double)1000*6/24, timearray[1000], 0.000001);
    CHECK_CLOSE(start+(double)10000*6/24, timearray[10000], 0.000001);

    Array1DDouble datetimearray = timearray.GetTimeArray();

    CHECK_CLOSE(start, datetimearray(0), 0.000001);
    CHECK_CLOSE(start+(double)1*6/24, datetimearray(1), 0.000001);
    CHECK_CLOSE(start+(double)2*6/24, datetimearray(2), 0.000001);
    CHECK_CLOSE(start+(double)3*6/24, datetimearray(3), 0.000001);
    CHECK_CLOSE(start+(double)4*6/24, datetimearray(4), 0.000001);
    CHECK_CLOSE(start+(double)5*6/24, datetimearray(5), 0.000001);
    CHECK_CLOSE(start+(double)10*6/24, datetimearray(10), 0.000001);
    CHECK_CLOSE(start+(double)100*6/24, datetimearray(100), 0.000001);
    CHECK_CLOSE(start+(double)1000*6/24, datetimearray(1000), 0.000001);
    CHECK_CLOSE(start+(double)10000*6/24, datetimearray(10000), 0.000001);

    CHECK_EQUAL(4*21550+1,datetimearray.rows());
}

TEST(BuildArrayDaysIntervalNormal)
{
    double start = asTime::GetMJD(1950,1,1);
    double end = asTime::GetMJD(2008,12,31);
    double timestephours = 6;
    double forecastdate = asTime::GetMJD(2010,06,01);
    double intervaldays = 60;
    double exclusiondays = 100;

    asTimeArray timearray(start, end, timestephours, asTimeArray::DaysInterval);
    timearray.Init(forecastdate, intervaldays, exclusiondays);

    CHECK_CLOSE(intervaldays*24, timearray.GetIntervalHours(),0.000001);
    CHECK_CLOSE(intervaldays, timearray.GetIntervalDays(),0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,4,2,0,0), timearray.GetFirst(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(2008,7,31,0,0), timearray.GetLast(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,4,2,6,0), timearray[1], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,4,2,12,0), timearray[2], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,4,27,0,0), timearray[100], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,7,30,18,0), timearray[479], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,7,31,0,0), timearray[480], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1951,4,2,0,0), timearray[481], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1951,4,2,6,0), timearray[482], 0.000001);
}

TEST(BuildArrayDaysIntervalNormalGeneric)
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

    CHECK_CLOSE(intervaldays*24, timearray.GetIntervalHours(),0.000001);
    CHECK_CLOSE(intervaldays, timearray.GetIntervalDays(),0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,4,2,0,0), timearray.GetFirst(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(2008,7,31,0,0), timearray.GetLast(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,4,2,6,0), timearray[1], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,4,2,12,0), timearray[2], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,4,27,0,0), timearray[100], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,7,30,18,0), timearray[479], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,7,31,0,0), timearray[480], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1951,4,2,0,0), timearray[481], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1951,4,2,6,0), timearray[482], 0.000001);
}

TEST(BuildArrayDaysIntervalNormalMidday)
{
    double start = asTime::GetMJD(1950,1,1);
    double end = asTime::GetMJD(2008,12,31);
    double timestephours = 6;
    double forecastdate = asTime::GetMJD(2010,06,01,12,00);
    double intervaldays = 60;
    double exclusiondays = 100;

    asTimeArray timearray(start, end, timestephours, asTimeArray::DaysInterval);
    timearray.Init(forecastdate, intervaldays, exclusiondays);

    CHECK_CLOSE(asTime::GetMJD(1950,4,2,12,0), timearray.GetFirst(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(2008,7,31,12,0), timearray.GetLast(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,4,2,18,0), timearray[1], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,4,3,0,0), timearray[2], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,4,27,12,0), timearray[100], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,7,31,6,0), timearray[479], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,7,31,12,0), timearray[480], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1951,4,2,12,0), timearray[481], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1951,4,2,18,0), timearray[482], 0.000001);
}

TEST(BuildArrayDaysIntervalNormalMiddayNotRound)
{
    double start = asTime::GetMJD(1950,1,1);
    double end = asTime::GetMJD(2008,12,31);
    double timestephours = 6;
    double forecastdate = asTime::GetMJD(2010,06,01,10,31);
    double intervaldays = 60;
    double exclusiondays = 100;

    asTimeArray timearray(start, end, timestephours, asTimeArray::DaysInterval);
    timearray.Init(forecastdate, intervaldays, exclusiondays);

    CHECK_CLOSE(0, 24*60*asTime::GetMJD(1950,4,2,10,31)-24*60*timearray.GetFirst(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,4,2,10,31), timearray.GetFirst(), 0.000001);
    CHECK_CLOSE(0, 24*60*asTime::GetMJD(2008,7,31,10,31)-24*60*timearray.GetLast(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(2008,7,31,10,31), timearray.GetLast(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,4,2,16,31), timearray[1], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,4,2,22,31), timearray[2], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,4,27,10,31), timearray[100], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,7,31,4,31), timearray[479], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,7,31,10,31), timearray[480], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1951,4,2,10,31), timearray[481], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1951,4,2,16,31), timearray[482], 0.000001);
}

TEST(BuildArrayDaysIntervalStartSplitted)
{
    double start = asTime::GetMJD(1950,1,1);
    double end = asTime::GetMJD(2008,12,31);
    double timestephours = 6;
    double forecastdate = asTime::GetMJD(2010,02,01,12,00);
    double intervaldays = 60;
    double exclusiondays = 100;

    asTimeArray timearray(start, end, timestephours, asTimeArray::DaysInterval);
    timearray.Init(forecastdate, intervaldays, exclusiondays);

    CHECK_CLOSE(asTime::GetMJD(1950,1,1,0,0), timearray.GetFirst(), 0.000001);
    CHECK_CLOSE(0, asTime::GetMJD(2008,4,1,12,0)-timearray.GetLast(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(2008,4,1,12,0), timearray.GetLast(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,1,1,6,0), timearray[1], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,1,1,12,0), timearray[2], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,1,26,0,0), timearray[100], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,4,2,12,0), timearray[366], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,12,3,12,0), timearray[367], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,12,3,18,0), timearray[368], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,12,4,0,0), timearray[369], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,12,31,18,0), timearray[366+114], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1951,1,1,0,0), timearray[366+115], 0.000001);
}

TEST(BuildArrayDaysIntervalEndSplitted)
{
    double start = asTime::GetMJD(1950,1,1,0);
    double end = asTime::GetMJD(2008,12,31,18);
    double timestephours = 6;
    double forecastdate = asTime::GetMJD(2010,12,01,12,00);
    double intervaldays = 60;
    double exclusiondays = 100;

    asTimeArray timearray(start, end, timestephours, asTimeArray::DaysInterval);
    timearray.Init(forecastdate, intervaldays, exclusiondays);

    CHECK_CLOSE(asTime::GetMJD(1950,10,2,12,0), timearray.GetFirst(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(2008,12,31,18,0), timearray.GetLast(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,10,2,18,0), timearray[1], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,10,3,0,0), timearray[2], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,10,27,12,0), timearray[100], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1951,1,30,6,0), timearray[479], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1951,1,30,12,0), timearray[480], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1951,10,2,12,0), timearray[481], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1951,10,2,18,0), timearray[482], 0.000001);
}

TEST(BuildArrayDaysIntervalExclusionPeriod)
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

    CHECK_EQUAL(false,foundyear);
}

TEST(BuildArraySeason)
{
    double start = asTime::GetMJD(1950,1,1);
    double end = asTime::GetMJD(2008,12,31);
    double timestephours = 6;
    double forecastdate = asTime::GetMJD(2010,07,01,12,00);
    double exclusiondays = 100;

    asTimeArray timearray(start, end, timestephours, asTimeArray::SameSeason);
    timearray.Init(forecastdate, exclusiondays);

    CHECK_CLOSE(asTime::GetMJD(1950,6,1,0,0), timearray.GetFirst(), 0.000001);
    CHECK_CLOSE(0, asTime::GetMJD(2008,8,31,18,0)-timearray.GetLast(), 0.000001);
    CHECK_CLOSE(0, 24*asTime::GetMJD(2008,8,31,18,0)-24*timearray.GetLast(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(2008,8,31,18,0), timearray.GetLast(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,6,1,6,0), timearray[1], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,6,1,12,0), timearray[2], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,8,31,18,0), timearray[367], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1951,6,1,0,0), timearray[368], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1951,6,1,6,0), timearray[369], 0.000001);
}

TEST(BuildArraySeasonGeneric)
{
    double start = asTime::GetMJD(1950,1,1);
    double end = asTime::GetMJD(2008,12,31);
    double timestephours = 6;
    double forecastdate = asTime::GetMJD(2010,07,01,12,00);
    double exclusiondays = 100;
    wxString slctModeString = "SameSeason";

    asTimeArray timearray(start, end, timestephours, slctModeString);
    timearray.Init(forecastdate, exclusiondays);

    CHECK_CLOSE(asTime::GetMJD(1950,6,1,0,0), timearray.GetFirst(), 0.000001);
    CHECK_CLOSE(0, asTime::GetMJD(2008,8,31,18,0)-timearray.GetLast(), 0.000001);
    CHECK_CLOSE(0, 24*asTime::GetMJD(2008,8,31,18,0)-24*timearray.GetLast(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(2008,8,31,18,0), timearray.GetLast(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,6,1,6,0), timearray[1], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,6,1,12,0), timearray[2], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,8,31,18,0), timearray[367], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1951,6,1,0,0), timearray[368], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1951,6,1,6,0), timearray[369], 0.000001);
}

TEST(BuildArraySeasonNotRound)
{
    double start = asTime::GetMJD(1950,1,1,0);
    double end = asTime::GetMJD(2008,12,31,18);
    double timestephours = 6;
    double forecastdate = asTime::GetMJD(2010,07,01,14,32);
    double exclusiondays = 100;

    asTimeArray timearray(start, end, timestephours, asTimeArray::SameSeason);
    timearray.Init(forecastdate, exclusiondays);

    CHECK_CLOSE(asTime::GetMJD(1950,6,1,2,32), timearray.GetFirst(), 0.000001);
    CHECK_CLOSE(0, asTime::GetMJD(2008,8,31,20,32)-timearray.GetLast(), 0.000001);
    CHECK_CLOSE(0, 24*asTime::GetMJD(2008,8,31,20,32)-24*timearray.GetLast(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(2008,8,31,20,32), timearray.GetLast(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,6,1,8,32), timearray[1], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,6,1,14,32), timearray[2], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,8,31,20,32), timearray[367], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1951,6,1,2,32), timearray[368], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1951,6,1,8,32), timearray[369], 0.000001);
}

TEST(BuildArraySeasonDec)
{
    double start = asTime::GetMJD(1950,1,1,0);
    double end = asTime::GetMJD(2008,12,31,18);
    double timestephours = 6;
    double forecastdate = asTime::GetMJD(2010,12,01,12,00);
    double exclusiondays = 100;

    asTimeArray timearray(start, end, timestephours, asTimeArray::SameSeason);
    timearray.Init(forecastdate, exclusiondays);

    CHECK_CLOSE(asTime::GetMJD(1950,1,1,0,0), timearray.GetFirst(), 0.000001);
    CHECK_CLOSE(0, asTime::GetMJD(2008,12,31,18,0)-timearray.GetLast(), 0.000001);
    CHECK_CLOSE(0, 24*asTime::GetMJD(2008,12,31,18,0)-24*timearray.GetLast(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(2008,12,31,18,0), timearray.GetLast(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,1,1,6,0), timearray[1], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,1,1,12,0), timearray[2], 0.000001);
}

TEST(BuildArraySeasonJan)
{
    double start = asTime::GetMJD(1950,1,1,0);
    double end = asTime::GetMJD(2008,12,31,18);
    double timestephours = 6;
    double forecastdate = asTime::GetMJD(2010,01,01,12,00);
    double exclusiondays = 100;

    asTimeArray timearray(start, end, timestephours, asTimeArray::SameSeason);
    timearray.Init(forecastdate, exclusiondays);

    CHECK_CLOSE(asTime::GetMJD(1950,1,1,0,0), timearray.GetFirst(), 0.000001);
    CHECK_CLOSE(0, asTime::GetMJD(2008,12,31,18,0)-timearray.GetLast(), 0.000001);
    CHECK_CLOSE(0, 24*asTime::GetMJD(2008,12,31,18,0)-24*timearray.GetLast(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(2008,12,31,18,0), timearray.GetLast(), 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,1,1,6,0), timearray[1], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,1,1,12,0), timearray[2], 0.000001);
}

TEST(GetFirstDayHour)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(2008,12,31,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    CHECK_CLOSE(12.5, timearray.GetFirstDayHour(), 0.000001);
}

TEST(GetLastDayHour)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(2008,12,31,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    CHECK_CLOSE(18.5, timearray.GetLastDayHour(), 0.000001);
}

TEST(GetFirstDayYear)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(2008,12,31,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    CHECK_EQUAL(1950, timearray.GetFirstDayYear());
}

TEST(GetLastDayYear)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(2008,12,31,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    CHECK_EQUAL(2008, timearray.GetLastDayYear());
}

TEST(GetSize)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(1950,1,2,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    CHECK_EQUAL(6, timearray.GetSize());
}

TEST(OperatorOverloading)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(1950,1,2,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    CHECK_CLOSE(asTime::GetMJD(1950,1,1,12,30), timearray[0], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,1,1,18,30), timearray[1], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,1,2,00,30), timearray[2], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,1,2,06,30), timearray[3], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,1,2,12,30), timearray[4], 0.000001);
    CHECK_CLOSE(asTime::GetMJD(1950,1,2,18,30), timearray[5], 0.000001);
}

TEST(GetFirst)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(1950,1,2,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    CHECK_CLOSE(asTime::GetMJD(1950,1,1,12,30), timearray.GetFirst(), 0.000001);
}

TEST(GetLast)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(1950,1,2,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    CHECK_CLOSE(asTime::GetMJD(1950,1,2,18,30), timearray.GetLast(), 0.000001);
}

TEST(GetIndexFirstAfter)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(1950,1,2,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    CHECK_CLOSE(2, timearray.GetIndexFirstAfter(asTime::GetMJD(1950,1,1,19,30)), 0.000001);
}

TEST(GetIndexFirstAfterEqual)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(1950,1,2,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    CHECK_CLOSE(1, timearray.GetIndexFirstAfter(asTime::GetMJD(1950,1,1,18,30)), 0.000001);
}

TEST(GetIndexFirstBefore)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(1950,1,2,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    CHECK_CLOSE(2, timearray.GetIndexFirstBefore(asTime::GetMJD(1950,1,2,05,30)), 0.000001);
}

TEST(GetIndexFirstBeforeEqual)
{
    double start = asTime::GetMJD(1950,1,1,12,30);
    double end = asTime::GetMJD(1950,1,2,18,30);
    double timestephours = 6;
    asTimeArray timearray(start, end, timestephours, asTimeArray::Simple);
    timearray.Init();

    CHECK_CLOSE(3, timearray.GetIndexFirstBefore(asTime::GetMJD(1950,1,2,06,30)), 0.000001);
}
}
