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
#include "asTime.h"

#include "gtest/gtest.h"


TEST(Time, IsLeapYearDivisableBy4)
{
	wxPrintf("Testing time functionalities...\n");
	
    const bool Result = asTime::IsLeapYear(1972);
    ASSERT_EQ(true, Result);
}

TEST(Time, IsLeapYearDivisableBy100)
{
    const bool Result = asTime::IsLeapYear(1900);
    ASSERT_EQ(false, Result);
}

TEST(Time, IsLeapYearDivisableBy400)
{
    const bool Result = asTime::IsLeapYear(2000);
    ASSERT_EQ(true, Result);
}

TEST(Time, IsLeapYearNo)
{
    const bool Result = asTime::IsLeapYear(1973);
    ASSERT_EQ(false, Result);
}

TEST(Time, GetMJDNormal_20040101)
{
    const double Result = asTime::GetMJD(2004, 1, 1, 0, 0, 0, asUSE_NORMAL_METHOD);
    CHECK_CLOSE(53005, Result, 0.00000001);
}

TEST(Time, GetMJDNormal_20040101_120000)
{
    double Result = asTime::GetMJD(2004, 1, 1, 12, 0, 0, asUSE_NORMAL_METHOD);
    CHECK_CLOSE(53005.5, Result, 0.00000001);
}

TEST(Time, GetMJDNormal_20101104_120000)
{
    double Result = asTime::GetMJD(2010, 11, 4, 12, 0, 0, asUSE_NORMAL_METHOD);
    CHECK_CLOSE(55504.5, Result, 0.00000001);
}

TEST(Time, GetMJDNormal_20101104_100000)
{
    double Result = asTime::GetMJD(2010, 11, 4, 10, 0, 0, asUSE_NORMAL_METHOD);
    CHECK_CLOSE(55504.41666666651, Result, 0.00000001);
}

TEST(Time, GetMJDNormal_20101104_103245)
{
    double Result = asTime::GetMJD(2010, 11, 4, 10, 32, 45, asUSE_NORMAL_METHOD);
    CHECK_CLOSE(55504.43940972211, Result, 0.00000001);
}

TEST(Time, GetMJDAlternate_20040101)
{
    const double Result = asTime::GetMJD(2004, 1, 1, 0, 0, 0, asUSE_ALTERNATE_METHOD);
    CHECK_CLOSE(53005, Result, 0.00000001);
}

TEST(Time, GetMJDAlternate_20040101_120000)
{
    double Result = asTime::GetMJD(2004, 1, 1, 12, 0, 0, asUSE_ALTERNATE_METHOD);
    CHECK_CLOSE(53005.5, Result, 0.00000001);
}

TEST(Time, GetMJDAlternate_20101104_120000)
{
    double Result = asTime::GetMJD(2010, 11, 4, 12, 0, 0, asUSE_ALTERNATE_METHOD);
    CHECK_CLOSE(55504.5, Result, 0.00000001);
}

TEST(Time, GetMJDAlternate_20101104_100000)
{
    double Result = asTime::GetMJD(2010, 11, 4, 10, 0, 0, asUSE_ALTERNATE_METHOD);
    CHECK_CLOSE(55504.41666666651, Result, 0.00000001);
}

TEST(Time, GetMJDAlternate_20101104_103245)
{
    double Result = asTime::GetMJD(2010, 11, 4, 10, 32, 45, asUSE_ALTERNATE_METHOD);
    CHECK_CLOSE(55504.43940972211, Result, 0.00000001);
}
/*
TEST(Time, GetMJDStructNormal_20040101)
{
    TimeStruct date;
    asTime::TimeStructInit(date);

    date.year = 2004;
    date.month = 1;
    date.day = 1;

    const double Result = asTime::GetMJD(date, asUSE_NORMAL_METHOD);
    CHECK_CLOSE(53005, Result, 0.00000001);
}

TEST(Time, GetMJDStructNormal_20040101_120000)
{
    TimeStruct date;
    asTime::TimeStructInit(date);

    date.year = 2004;
    date.month = 1;
    date.day = 1;
    date.hour = 12;

    double Result = asTime::GetMJD(date, asUSE_NORMAL_METHOD);
    CHECK_CLOSE(53005.5, Result, 0.00000001);
}

TEST(Time, GetMJDStructNormal_20101104_120000)
{
    TimeStruct date;
    asTime::TimeStructInit(date);

    date.year = 2010;
    date.month = 11;
    date.day = 4;
    date.hour = 12;

    double Result = asTime::GetMJD(date, asUSE_NORMAL_METHOD);
    CHECK_CLOSE(55504.5, Result, 0.00000001);
}

TEST(Time, GetMJDStructNormal_20101104_100000)
{
    TimeStruct date;
    asTime::TimeStructInit(date);

    date.year = 2010;
    date.month = 11;
    date.day = 4;
    date.hour = 10;

    double Result = asTime::GetMJD(date, asUSE_NORMAL_METHOD);
    CHECK_CLOSE(55504.41666666651, Result, 0.00000001);
}

TEST(Time, GetMJDStructNormal_20101104_103245)
{
    TimeStruct date;
    asTime::TimeStructInit(date);

    date.year = 2010;
    date.month = 11;
    date.day = 4;
    date.hour = 10;
    date.min = 32;
    date.sec = 45;

    double Result = asTime::GetMJD(date, asUSE_NORMAL_METHOD);
    CHECK_CLOSE(55504.43940972211, Result, 0.00000001);
}

TEST(Time, GetMJDStructAlternate_20040101)
{
    TimeStruct date;
    asTime::TimeStructInit(date);

    date.year = 2004;
    date.month = 1;
    date.day = 1;

    const double Result = asTime::GetMJD(date, asUSE_ALTERNATE_METHOD);
    CHECK_CLOSE(53005, Result, 0.00000001);
}

TEST(Time, GetMJDStructAlternate_20040101_120000)
{
    TimeStruct date;
    asTime::TimeStructInit(date);

    date.year = 2004;
    date.month = 1;
    date.day = 1;
    date.hour = 12;

    double Result = asTime::GetMJD(date, asUSE_ALTERNATE_METHOD);
    CHECK_CLOSE(53005.5, Result, 0.00000001);
}

TEST(Time, GetMJDStructAlternate_20101104_120000)
{
    TimeStruct date;
    asTime::TimeStructInit(date);

    date.year = 2010;
    date.month = 11;
    date.day = 4;
    date.hour = 12;

    double Result = asTime::GetMJD(date, asUSE_ALTERNATE_METHOD);
    CHECK_CLOSE(55504.5, Result, 0.00000001);
}

TEST(Time, GetMJDStructAlternate_20101104_100000)
{
    TimeStruct date;
    asTime::TimeStructInit(date);

    date.year = 2010;
    date.month = 11;
    date.day = 4;
    date.hour = 10;

    double Result = asTime::GetMJD(date, asUSE_ALTERNATE_METHOD);
    CHECK_CLOSE(55504.41666666651, Result, 0.00000001);
}

TEST(Time, GetMJDStructAlternate_20101104_103245)
{
    TimeStruct date;
    asTime::TimeStructInit(date);

    date.year = 2010;
    date.month = 11;
    date.day = 4;
    date.hour = 10;
    date.min = 32;
    date.sec = 45;

    double Result = asTime::GetMJD(date, asUSE_ALTERNATE_METHOD);
    CHECK_CLOSE(55504.43940972211, Result, 0.00000001);
}
*/
TEST(Time, GetTimeStructNormal_20040101)
{
    double Mjd = 53005;
    TimeStruct date = asTime::GetTimeStruct(Mjd, asUSE_NORMAL_METHOD);

    ASSERT_EQ(2004, date.year);
    ASSERT_EQ(1, date.month);
    ASSERT_EQ(1, date.day);
}

TEST(Time, GetTimeStructNormal_20040101_120000)
{
    double Mjd = 53005.5;
    TimeStruct date = asTime::GetTimeStruct(Mjd, asUSE_NORMAL_METHOD);

    ASSERT_EQ(2004, date.year);
    ASSERT_EQ(1, date.month);
    ASSERT_EQ(1, date.day);
    ASSERT_EQ(12, date.hour);
}

TEST(Time, GetTimeStructNormal_20101104_120000)
{
    double Mjd = 55504.5;
    TimeStruct date = asTime::GetTimeStruct(Mjd, asUSE_NORMAL_METHOD);

    ASSERT_EQ(2010, date.year);
    ASSERT_EQ(11, date.month);
    ASSERT_EQ(4, date.day);
    ASSERT_EQ(12, date.hour);
}

TEST(Time, GetTimeStructNormal_20101104_100000)
{
    double Mjd = 55504.41666666651;
    TimeStruct date = asTime::GetTimeStruct(Mjd, asUSE_NORMAL_METHOD);

    ASSERT_EQ(2010, date.year);
    ASSERT_EQ(11, date.month);
    ASSERT_EQ(4, date.day);
    ASSERT_EQ(10, date.hour);
}

TEST(Time, GetTimeStructNormal_20101104_103245)
{
    double Mjd = 55504.43940972211;
    TimeStruct date = asTime::GetTimeStruct(Mjd, asUSE_NORMAL_METHOD);

    ASSERT_EQ(2010, date.year);
    ASSERT_EQ(11, date.month);
    ASSERT_EQ(4, date.day);
    ASSERT_EQ(10, date.hour);
    ASSERT_EQ(32, date.min);
    ASSERT_EQ(45, date.sec);
}

TEST(Time, GetTimeStructAlternate_20040101)
{
    double Mjd = 53005;
    TimeStruct date = asTime::GetTimeStruct(Mjd, asUSE_ALTERNATE_METHOD);

    ASSERT_EQ(2004, date.year);
    ASSERT_EQ(1, date.month);
    ASSERT_EQ(1, date.day);
}

TEST(Time, GetTimeStructAlternate_20040101_120000)
{
    double Mjd = 53005.5;
    TimeStruct date = asTime::GetTimeStruct(Mjd, asUSE_ALTERNATE_METHOD);

    ASSERT_EQ(2004, date.year);
    ASSERT_EQ(1, date.month);
    ASSERT_EQ(1, date.day);
    ASSERT_EQ(12, date.hour);
}

TEST(Time, GetTimeStructAlternate_20101104_120000)
{
    double Mjd = 55504.5;
    TimeStruct date = asTime::GetTimeStruct(Mjd, asUSE_ALTERNATE_METHOD);

    ASSERT_EQ(2010, date.year);
    ASSERT_EQ(11, date.month);
    ASSERT_EQ(4, date.day);
    ASSERT_EQ(12, date.hour);
}

TEST(Time, GetTimeStructAlternate_20101104_100000)
{
    double Mjd = 55504.41666666651;
    TimeStruct date = asTime::GetTimeStruct(Mjd, asUSE_ALTERNATE_METHOD);

    ASSERT_EQ(2010, date.year);
    ASSERT_EQ(11, date.month);
    ASSERT_EQ(4, date.day);
    ASSERT_EQ(10, date.hour);
    ASSERT_EQ(0, date.min);
    ASSERT_EQ(0, date.sec);
}

TEST(Time, GetTimeStructAlternate_20101104_103245)
{
    double Mjd = 55504.43940972211;
    TimeStruct date = asTime::GetTimeStruct(Mjd, asUSE_ALTERNATE_METHOD);

    ASSERT_EQ(2010, date.year);
    ASSERT_EQ(11, date.month);
    ASSERT_EQ(4, date.day);
    ASSERT_EQ(10, date.hour);
    ASSERT_EQ(32, date.min);
    ASSERT_EQ(45, date.sec);
}

TEST(Time, GetTimeStructOther)
{
    TimeStruct date = asTime::GetTimeStruct(2010, 11, 4, 10, 32, 45);

    ASSERT_EQ(2010, date.year);
    ASSERT_EQ(11, date.month);
    ASSERT_EQ(4, date.day);
    ASSERT_EQ(10, date.hour);
    ASSERT_EQ(32, date.min);
    ASSERT_EQ(45, date.sec);
}

TEST(Time, GetYearNormal_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetYear( Mjd, asUSE_NORMAL_METHOD );

    ASSERT_EQ(2010, Result);
}

TEST(Time, GetYearAlternate_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetYear( Mjd, asUSE_ALTERNATE_METHOD );

    ASSERT_EQ(2010, Result);
}

TEST(Time, GetMonthNormal_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetMonth( Mjd, asUSE_NORMAL_METHOD );

    ASSERT_EQ(11, Result);
}

TEST(Time, GetMonthAlternate_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetMonth( Mjd, asUSE_ALTERNATE_METHOD );

    ASSERT_EQ(11, Result);
}

TEST(Time, GetDayNormal_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetDay( Mjd, asUSE_NORMAL_METHOD );

    ASSERT_EQ(4, Result);
}

TEST(Time, GetDayAlternate_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetDay( Mjd, asUSE_ALTERNATE_METHOD );

    ASSERT_EQ(4, Result);
}

TEST(Time, GetHourNormal_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetHour( Mjd, asUSE_NORMAL_METHOD );

    ASSERT_EQ(10, Result);
}

TEST(Time, GetHourAlternate_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetHour( Mjd, asUSE_ALTERNATE_METHOD );

    ASSERT_EQ(10, Result);
}

TEST(Time, GetMinuteNormal_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetMinute( Mjd, asUSE_NORMAL_METHOD );

    ASSERT_EQ(32, Result);
}

TEST(Time, GetMinuteAlternate_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetMinute( Mjd, asUSE_ALTERNATE_METHOD );

    ASSERT_EQ(32, Result);
}

TEST(Time, GetSecondNormal_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetSecond( Mjd, asUSE_NORMAL_METHOD );

    ASSERT_EQ(45, Result);
}

TEST(Time, GetSecondAlternate_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetSecond( Mjd, asUSE_ALTERNATE_METHOD );

    ASSERT_EQ(45, Result);
}
/*
TEST(Time, GetSeasonStartDJF)
{
    TimeStruct val = asTime::GetSeasonStart(DJF);

    ASSERT_EQ(1, val.day);
    ASSERT_EQ(12, val.month);
}

TEST(Time, GetSeasonStartMAM)
{
    TimeStruct val = asTime::GetSeasonStart(MAM);

    ASSERT_EQ(1, val.day);
    ASSERT_EQ(3, val.month);
}

TEST(Time, GetSeasonStartJJA)
{
    TimeStruct val = asTime::GetSeasonStart(JJA);

    ASSERT_EQ(1, val.day);
    ASSERT_EQ(6, val.month);
}

TEST(Time, GetSeasonStartSON)
{
    TimeStruct val = asTime::GetSeasonStart(SON);

    ASSERT_EQ(1, val.day);
    ASSERT_EQ(9, val.month);
}

TEST(Time, GetSeasonEndDJF_1973)
{
    TimeStruct val = asTime::GetSeasonEnd(DJF, 1973);

    ASSERT_EQ(28, val.day);
    ASSERT_EQ(2, val.month);
}

TEST(Time, GetSeasonEndDJF_2000)
{
    TimeStruct val = asTime::GetSeasonEnd(DJF, 2000);

    ASSERT_EQ(29, val.day);
    ASSERT_EQ(2, val.month);
}

TEST(Time, GetSeasonEndDJF_1972)
{
    TimeStruct val = asTime::GetSeasonEnd(DJF, 1972);

    ASSERT_EQ(29, val.day);
    ASSERT_EQ(2, val.month);
}

TEST(Time, GetSeasonEndDJF_1900)
{
    TimeStruct val = asTime::GetSeasonEnd(DJF, 1900);

    ASSERT_EQ(28, val.day);
    ASSERT_EQ(2, val.month);
}

TEST(Time, GetSeasonEndMAM)
{
    TimeStruct val = asTime::GetSeasonEnd(MAM, 1900);

    ASSERT_EQ(31, val.day);
    ASSERT_EQ(5, val.month);
}

TEST(Time, GetSeasonEndJJA)
{
    TimeStruct val = asTime::GetSeasonEnd(JJA, 1900);

    ASSERT_EQ(31, val.day);
    ASSERT_EQ(8, val.month);
}

TEST(Time, GetSeasonEndSON)
{
    TimeStruct val = asTime::GetSeasonEnd(SON, 1900);

    ASSERT_EQ(30, val.day);
    ASSERT_EQ(11, val.month);
}

TEST(Time, GetSeasonJan)
{
    Season Result = asTime::GetSeason(1);
    ASSERT_EQ(DJF, Result);
}

TEST(Time, GetSeasonFeb)
{
    Season Result = asTime::GetSeason(2);
    ASSERT_EQ(DJF, Result);
}

TEST(Time, GetSeasonMar)
{
    Season Result = asTime::GetSeason(3);
    ASSERT_EQ(MAM, Result);
}

TEST(Time, GetSeasonApr)
{
    Season Result = asTime::GetSeason(4);
    ASSERT_EQ(MAM, Result);
}

TEST(Time, GetSeasonMay)
{
    Season Result = asTime::GetSeason(5);
    ASSERT_EQ(MAM, Result);
}

TEST(Time, GetSeasonJun)
{
    Season Result = asTime::GetSeason(6);
    ASSERT_EQ(JJA, Result);
}

TEST(Time, GetSeasonJul)
{
    Season Result = asTime::GetSeason(7);
    ASSERT_EQ(JJA, Result);
}

TEST(Time, GetSeasonAug)
{
    Season Result = asTime::GetSeason(8);
    ASSERT_EQ(JJA, Result);
}

TEST(Time, GetSeasonSep)
{
    Season Result = asTime::GetSeason(9);
    ASSERT_EQ(SON, Result);
}

TEST(Time, GetSeasonOct)
{
    Season Result = asTime::GetSeason(10);
    ASSERT_EQ(SON, Result);
}

TEST(Time, GetSeasonNov)
{
    Season Result = asTime::GetSeason(11);
    ASSERT_EQ(SON, Result);
}

TEST(Time, GetSeasonDec)
{
    Season Result = asTime::GetSeason(12);
    ASSERT_EQ(DJF, Result);
}

TEST(Time, GetSeasonException)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetSeason(13), asException);
    }
}

TEST(Time, TimeTmToTimeStruct)
{
    struct tm DateTm;
    DateTm.tm_year = 2010-1900;
    DateTm.tm_mon = 11-1;
    DateTm.tm_mday = 4;
    DateTm.tm_hour = 10;
    DateTm.tm_min = 32;
    DateTm.tm_sec = 45;

    TimeStruct date = asTime::TimeTmToTimeStruct(DateTm);

    ASSERT_EQ(2010, date.year);
    ASSERT_EQ(11, date.month);
    ASSERT_EQ(4, date.day);
    ASSERT_EQ(10, date.hour);
    ASSERT_EQ(32, date.min);
    ASSERT_EQ(45, date.sec);
}

TEST(Time, TimeTmToMJD)
{
    struct tm DateTm;
    DateTm.tm_year = 2010-1900;
    DateTm.tm_mon = 11-1;
    DateTm.tm_mday = 4;
    DateTm.tm_hour = 10;
    DateTm.tm_min = 32;
    DateTm.tm_sec = 45;

    double mjd = asTime::TimeTmToMJD(DateTm);

    CHECK_CLOSE(55504.43940972211, mjd, 0.000001);
}
*/
TEST(Time, NowLocalMJD)
{
    double mjd = asTime::NowMJD(asLOCAL);
    wxString datestr = asTime::GetStringTime(mjd);

	wxPrintf("Local time is %s\n", datestr);
}

TEST(Time, NowLocalTimeStruct)
{
    TimeStruct date = asTime::NowTimeStruct(asLOCAL);
    wxString datestr = asTime::GetStringTime(date);

	wxPrintf("Local time is %s\n", datestr);
}

TEST(Time, NowMJD)
{
    double mjd = asTime::NowMJD(asUTM);
    wxString datestr = asTime::GetStringTime(mjd);

	wxPrintf("UTM time is %s\n", datestr);
}

TEST(Time, NowTimeStruct)
{
    TimeStruct date = asTime::NowTimeStruct(asUTM);
    wxString datestr = asTime::GetStringTime(date);

	wxPrintf("UTM time is %s\n", datestr);
}

TEST(Time, GetStringDateMJD)
{
    double Mjd = 55504.43940972211;

    wxString datestr = asTime::GetStringTime(Mjd, classic);

    int Result = datestr.CompareTo(_T("04.11.2010"));

    ASSERT_EQ(0, Result);
}

TEST(Time, GetStringDateTimeStruct)
{
    TimeStruct date;
    date.year = 2010;
    date.month = 11;
    date.day = 4;
    date.hour = 10;
    date.min = 32;
    date.sec = 45;

    wxString datestr = asTime::GetStringTime(date, classic);

    int Result = datestr.CompareTo(_T("04.11.2010"));

    ASSERT_EQ(0, Result);
}

TEST(Time, GetStringDateReverseMJD)
{
    double Mjd = 55504.43940972211;

    wxString datestr = asTime::GetStringTime(Mjd, YYYYMMDD);

    int Result = datestr.CompareTo(_T("2010/11/04"));

    ASSERT_EQ(0, Result);
}

TEST(Time, GetStringDateReverseTimeStruct)
{
    TimeStruct date;
    date.year = 2010;
    date.month = 11;
    date.day = 4;
    date.hour = 10;
    date.min = 32;
    date.sec = 45;

    wxString datestr = asTime::GetStringTime(date, YYYYMMDD);

    int Result = datestr.CompareTo(_T("2010/11/04"));

    ASSERT_EQ(0, Result);
}

TEST(Time, GetStringTimeTimeStruct)
{
    TimeStruct date;
    date.year = 2010;
    date.month = 11;
    date.day = 4;
    date.hour = 3;
    date.min = 5;
    date.sec = 5;

    wxString datestr = asTime::GetStringTime(date, timeonly);

    int Result = datestr.CompareTo(_T("03:05"));

    ASSERT_EQ(0, Result);
}

TEST(Time, GetTimeFromStringFormatDDMMYYYY)
{
    double conversion = asTime::GetTimeFromString("23.11.2007", DDMMYYYY);
    double mjd = asTime::GetMJD(2007,11,23);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(Time, GetTimeFromStringFormatDDMMYYYYSlashes)
{
    double conversion = asTime::GetTimeFromString("23/11/2007", DDMMYYYY);
    double mjd = asTime::GetMJD(2007,11,23);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(Time, GetTimeFromStringFormatDDMMYYYYException)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("23.11.07", DDMMYYYY), asException);
    }
}

TEST(Time, GetTimeFromStringFormatYYYYMMDD)
{
    double conversion = asTime::GetTimeFromString("2007.11.23", YYYYMMDD);
    double mjd = asTime::GetMJD(2007,11,23);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(Time, GetTimeFromStringFormatYYYYMMDDException)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("23.11.2007", YYYYMMDD), asException);
    }
}

TEST(Time, GetTimeFromStringFormatDDMMYYYYhhmm)
{
    double conversion = asTime::GetTimeFromString("23.11.2007 13:05", DDMMYYYYhhmm);
    double mjd = asTime::GetMJD(2007,11,23,13,5);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(Time, GetTimeFromStringFormatDDMMYYYYhhmmException)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("23.11.07 13:05", DDMMYYYYhhmm), asException);
    }
}

TEST(Time, GetTimeFromStringFormatYYYYMMDDhhmm)
{
    double conversion = asTime::GetTimeFromString("2007.11.23 13:05", YYYYMMDDhhmm);
    double mjd = asTime::GetMJD(2007,11,23,13,5);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(Time, GetTimeFromStringFormatYYYYMMDDhhmmException)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("23.11.2007 13:05", YYYYMMDDhhmm), asException);
    }
}

TEST(Time, GetTimeFromStringFormatDDMMYYYYhhmmss)
{
    double conversion = asTime::GetTimeFromString("23.11.2007 13:05:01", DDMMYYYYhhmmss);
    double mjd = asTime::GetMJD(2007,11,23,13,5,1);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(Time, GetTimeFromStringFormatDDMMYYYYhhmmssException)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("23.11.07 13:05:01", DDMMYYYYhhmmss), asException);
    }
}

TEST(Time, GetTimeFromStringFormatYYYYMMDDhhmmss)
{
    double conversion = asTime::GetTimeFromString("2007.11.23 13:05:01", YYYYMMDDhhmmss);
    double mjd = asTime::GetMJD(2007,11,23,13,5,1);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(Time, GetTimeFromStringFormatYYYYMMDDhhmmssException)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("23.11.2007 13:05:01", YYYYMMDDhhmmss), asException);
    }
}

TEST(Time, GetTimeFromStringFormathhmmException)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("13:05:01", hhmm), asException);
    }
}

TEST(Time, GetTimeFromStringFormatnowplushours)
{
    double conversion = asTime::GetTimeFromString("+2", nowplushours);
    wxString datestr = asTime::GetStringTime(conversion);

	wxPrintf("UTM time +2 hours is %s\n", datestr);
}

TEST(Time, GetTimeFromStringFormatnowplushoursException)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("+2:23", nowplushours), asException);
    }
}

TEST(Time, GetTimeFromStringFormatnowplushoursExceptionDot)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("+2.23", nowplushours), asException);
    }
}

TEST(Time, GetTimeFromStringFormatnowminushours)
{
    double conversion = asTime::GetTimeFromString("-2", nowminushours);
    wxString datestr = asTime::GetStringTime(conversion);

	wxPrintf("UTM time -2 hours is %s\n", datestr);
}

TEST(Time, GetTimeFromStringFormatnowminushoursException)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("-2:23", nowminushours), asException);
    }
}

TEST(Time, GetTimeFromStringFormatnowminushoursExceptionDot)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("-2.23", nowminushours), asException);
    }
}

TEST(Time, GetTimeFromStringFormatnowminushoursExceptionSignPlus)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("+2", nowminushours), asException);
    }
}

TEST(Time, GetTimeFromStringFormatnowminushoursExceptionSignNo)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("2", nowminushours), asException);
    }
}

TEST(Time, GetTimeFromStringFormatautoDDMMYYYY)
{
    double conversion = asTime::GetTimeFromString("23.11.2007", guess);
    double mjd = asTime::GetMJD(2007,11,23);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(Time, GetTimeFromStringFormatautoDDMMYYYYSlashes)
{
    double conversion = asTime::GetTimeFromString("23/11/2007", guess);
    double mjd = asTime::GetMJD(2007,11,23);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(Time, GetTimeFromStringFormatautoDDMMYYYYException)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("23.11.07", guess), asException);
    }
}

TEST(Time, GetTimeFromStringFormatautoYYYYMMDD)
{
    double conversion = asTime::GetTimeFromString("2007.11.23", guess);
    double mjd = asTime::GetMJD(2007,11,23);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(Time, GetTimeFromStringFormatautoYYYYMMDDException)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("11.2007", guess), asException);
    }
}

TEST(Time, GetTimeFromStringFormatautoDDMMYYYYhhmm)
{
    double conversion = asTime::GetTimeFromString("23.11.2007 13:05", guess);
    double mjd = asTime::GetMJD(2007,11,23,13,5);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(Time, GetTimeFromStringFormatautoDDMMYYYYhhmmException)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("23.11.07 13:05", guess), asException);
    }
}

TEST(Time, GetTimeFromStringFormatautoYYYYMMDDhhmm)
{
    double conversion = asTime::GetTimeFromString("2007.11.23 13:05", guess);
    double mjd = asTime::GetMJD(2007,11,23,13,5);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(Time, GetTimeFromStringFormatautoYYYYMMDDhhmmException)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("23.11.07 13:05", guess), asException);
    }
}

TEST(Time, GetTimeFromStringFormatautoDDMMYYYYhhmmss)
{
    double conversion = asTime::GetTimeFromString("23.11.2007 13:05:01", guess);
    double mjd = asTime::GetMJD(2007,11,23,13,5,1);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(Time, GetTimeFromStringFormatautoDDMMYYYYhhmmssException)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("23.11.07 13:05:01", guess), asException);
    }
}

TEST(Time, GetTimeFromStringFormatautoYYYYMMDDhhmmss)
{
    double conversion = asTime::GetTimeFromString("2007.11.23 13:05:01", guess);
    double mjd = asTime::GetMJD(2007,11,23,13,5,1);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(Time, GetTimeFromStringFormatautoYYYYMMDDhhmmssException)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("23.11.07 13:05:01", guess), asException);
    }
}

TEST(Time, GetTimeFromStringFormatautohhmmException)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("13:05:01", guess), asException);
    }
}

TEST(Time, GetTimeFromStringFormatautonowplushours)
{
    double conversion = asTime::GetTimeFromString("+2", guess);
    wxString datestr = asTime::GetStringTime(conversion);

	wxPrintf("UTM time +2 hours is %s\n", datestr);
}

TEST(Time, GetTimeFromStringFormatautonowplushoursException)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("+2:23", guess), asException);
    }
}

TEST(Time, GetTimeFromStringFormatautonowplushoursExceptionDot)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("+2.23", guess), asException);
    }
}

TEST(Time, GetTimeFromStringFormatautonowminushours)
{
    double conversion = asTime::GetTimeFromString("-2", guess);
    wxString datestr = asTime::GetStringTime(conversion);

	wxPrintf("UTM time -2 hours is %s\n", datestr);
}

TEST(Time, GetTimeFromStringFormatautonowminushoursException)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("-2:23", guess), asException);
    }
}

TEST(Time, GetTimeFromStringFormatautonowminushoursExceptionDot)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("-2.23", guess), asException);
    }
}

TEST(Time, GetTimeFromStringFormatautonowminushoursExceptionSignNo)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("2", guess), asException);
    }
}

TEST(Time, GetTimeFromStringFormatautonowminushoursExceptionSignPlusText)
{
    if(g_unitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("+2hours", guess), asException);
    }
}

TEST(Time, AddYear1972)
{
    double mjd = asTime::GetMJD(1972,11,23,13,5);
    mjd = asTime::AddYear(mjd);
    double mjdafter = asTime::GetMJD(1973,11,23,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}

TEST(Time, AddYear1972Leap)
{
    double mjd = asTime::GetMJD(1972,2,23,13,5);
    mjd = asTime::AddYear(mjd);
    double mjdafter = asTime::GetMJD(1973,2,23,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}

TEST(Time, AddYear1900)
{
    double mjd = asTime::GetMJD(1900,2,23,13,5);
    mjd = asTime::AddYear(mjd);
    double mjdafter = asTime::GetMJD(1901,2,23,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}

TEST(Time, AddYear2000)
{
    double mjd = asTime::GetMJD(2000,11,23,13,5);
    mjd = asTime::AddYear(mjd);
    double mjdafter = asTime::GetMJD(2001,11,23,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}

TEST(Time, AddYear2000Leap)
{
    double mjd = asTime::GetMJD(2000,2,23,13,5);
    mjd = asTime::AddYear(mjd);
    double mjdafter = asTime::GetMJD(2001,2,23,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}

TEST(Time, SubtractYear1972Leap)
{
    double mjd = asTime::GetMJD(1972,11,23,13,5);
    mjd = asTime::SubtractYear(mjd);
    double mjdafter = asTime::GetMJD(1971,11,23,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}

TEST(Time, SubtractYear1972)
{
    double mjd = asTime::GetMJD(1972,2,23,13,5);
    mjd = asTime::SubtractYear(mjd);
    double mjdafter = asTime::GetMJD(1971,2,23,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}

TEST(Time, SubtractYear1900)
{
    double mjd = asTime::GetMJD(1900,11,23,13,5);
    mjd = asTime::SubtractYear(mjd);
    double mjdafter = asTime::GetMJD(1899,11,23,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}

TEST(Time, SubtractYear2000Leap)
{
    double mjd = asTime::GetMJD(2000,11,23,13,5);
    mjd = asTime::SubtractYear(mjd);
    double mjdafter = asTime::GetMJD(1999,11,23,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}

TEST(Time, SubtractYear2000)
{
    double mjd = asTime::GetMJD(2000,2,23,13,5);
    mjd = asTime::SubtractYear(mjd);
    double mjdafter = asTime::GetMJD(1999,2,23,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}

TEST(Time, SubtractYear2000Feb28)
{
    double mjd = asTime::GetMJD(2000,2,28,13,5);
    mjd = asTime::SubtractYear(mjd);
    double mjdafter = asTime::GetMJD(1999,2,28,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}

TEST(Time, SubtractYear2000Feb29)
{
    double mjd = asTime::GetMJD(2000,2,29,13,5);
    mjd = asTime::SubtractYear(mjd);
    double mjdafter = asTime::GetMJD(1999,2,28,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}
