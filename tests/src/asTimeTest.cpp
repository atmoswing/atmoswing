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

#include "asTime.h"
#include "gtest/gtest.h"


TEST(Time, IsLeapYearDivisableBy4)
{
    EXPECT_TRUE(asTime::IsLeapYear(1972));
}

TEST(Time, IsLeapYearDivisableBy100)
{
    EXPECT_FALSE(asTime::IsLeapYear(1900));
}

TEST(Time, IsLeapYearDivisableBy400)
{
    EXPECT_TRUE(asTime::IsLeapYear(2000));
}

TEST(Time, IsLeapYearNo)
{
    EXPECT_FALSE(asTime::IsLeapYear(1973));
}

TEST(Time, GetMJDNormal_20040101)
{
    double result = asTime::GetMJD(2004, 1, 1, 0, 0, 0, asUSE_NORMAL_METHOD);
    EXPECT_DOUBLE_EQ(53005, result);
}

TEST(Time, GetMJDNormal_20040101_120000)
{
    double result = asTime::GetMJD(2004, 1, 1, 12, 0, 0, asUSE_NORMAL_METHOD);
    EXPECT_DOUBLE_EQ(53005.5, result);
}

TEST(Time, GetMJDNormal_20101104_120000)
{
    double result = asTime::GetMJD(2010, 11, 4, 12, 0, 0, asUSE_NORMAL_METHOD);
    EXPECT_DOUBLE_EQ(55504.5, result);
}

TEST(Time, GetMJDNormal_20101104_100000)
{
    double result = asTime::GetMJD(2010, 11, 4, 10, 0, 0, asUSE_NORMAL_METHOD);
    EXPECT_DOUBLE_EQ(55504.41666666651, result);
}

TEST(Time, GetMJDNormal_20101104_103245)
{
    double result = asTime::GetMJD(2010, 11, 4, 10, 32, 45, asUSE_NORMAL_METHOD);
    EXPECT_DOUBLE_EQ(55504.43940972211, result);
}

TEST(Time, GetMJDAlternate_20040101)
{
    double result = asTime::GetMJD(2004, 1, 1, 0, 0, 0, asUSE_ALTERNATE_METHOD);
    EXPECT_DOUBLE_EQ(53005, result);
}

TEST(Time, GetMJDAlternate_20040101_120000)
{
    double result = asTime::GetMJD(2004, 1, 1, 12, 0, 0, asUSE_ALTERNATE_METHOD);
    EXPECT_DOUBLE_EQ(53005.5, result);
}

TEST(Time, GetMJDAlternate_20101104_120000)
{
    double result = asTime::GetMJD(2010, 11, 4, 12, 0, 0, asUSE_ALTERNATE_METHOD);
    EXPECT_DOUBLE_EQ(55504.5, result);
}

TEST(Time, GetMJDAlternate_20101104_100000)
{
    double result = asTime::GetMJD(2010, 11, 4, 10, 0, 0, asUSE_ALTERNATE_METHOD);
    EXPECT_DOUBLE_EQ(55504.41666666651, result);
}

TEST(Time, GetMJDAlternate_20101104_103245)
{
    double result = asTime::GetMJD(2010, 11, 4, 10, 32, 45, asUSE_ALTERNATE_METHOD);
    EXPECT_DOUBLE_EQ(55504.43940972211, result);
}

TEST(Time, GetTimeStructNormal_20040101)
{
    double mjd = 53005;
    Time date = asTime::GetTimeStruct(mjd, asUSE_NORMAL_METHOD);

    EXPECT_EQ(2004, date.year);
    EXPECT_EQ(1, date.month);
    EXPECT_EQ(1, date.day);
}

TEST(Time, GetTimeStructNormal_20040101_120000)
{
    double mjd = 53005.5;
    Time date = asTime::GetTimeStruct(mjd, asUSE_NORMAL_METHOD);

    EXPECT_EQ(2004, date.year);
    EXPECT_EQ(1, date.month);
    EXPECT_EQ(1, date.day);
    EXPECT_EQ(12, date.hour);
}

TEST(Time, GetTimeStructNormal_20101104_120000)
{
    double mjd = 55504.5;
    Time date = asTime::GetTimeStruct(mjd, asUSE_NORMAL_METHOD);

    EXPECT_EQ(2010, date.year);
    EXPECT_EQ(11, date.month);
    EXPECT_EQ(4, date.day);
    EXPECT_EQ(12, date.hour);
}

TEST(Time, GetTimeStructNormal_20101104_100000)
{
    double mjd = 55504.41666666651;
    Time date = asTime::GetTimeStruct(mjd, asUSE_NORMAL_METHOD);

    EXPECT_EQ(2010, date.year);
    EXPECT_EQ(11, date.month);
    EXPECT_EQ(4, date.day);
    EXPECT_EQ(10, date.hour);
}

TEST(Time, GetTimeStructNormal_20101104_103245)
{
    double mjd = 55504.43940972211;
    Time date = asTime::GetTimeStruct(mjd, asUSE_NORMAL_METHOD);

    EXPECT_EQ(2010, date.year);
    EXPECT_EQ(11, date.month);
    EXPECT_EQ(4, date.day);
    EXPECT_EQ(10, date.hour);
    EXPECT_EQ(32, date.min);
    EXPECT_EQ(45, date.sec);
}

TEST(Time, GetTimeStructAlternate_20040101)
{
    double mjd = 53005;
    Time date = asTime::GetTimeStruct(mjd, asUSE_ALTERNATE_METHOD);

    EXPECT_EQ(2004, date.year);
    EXPECT_EQ(1, date.month);
    EXPECT_EQ(1, date.day);
}

TEST(Time, GetTimeStructAlternate_20040101_120000)
{
    double mjd = 53005.5;
    Time date = asTime::GetTimeStruct(mjd, asUSE_ALTERNATE_METHOD);

    EXPECT_EQ(2004, date.year);
    EXPECT_EQ(1, date.month);
    EXPECT_EQ(1, date.day);
    EXPECT_EQ(12, date.hour);
}

TEST(Time, GetTimeStructAlternate_20101104_120000)
{
    double mjd = 55504.5;
    Time date = asTime::GetTimeStruct(mjd, asUSE_ALTERNATE_METHOD);

    EXPECT_EQ(2010, date.year);
    EXPECT_EQ(11, date.month);
    EXPECT_EQ(4, date.day);
    EXPECT_EQ(12, date.hour);
}

TEST(Time, GetTimeStructAlternate_20101104_100000)
{
    double mjd = 55504.41666666651;
    Time date = asTime::GetTimeStruct(mjd, asUSE_ALTERNATE_METHOD);

    EXPECT_EQ(2010, date.year);
    EXPECT_EQ(11, date.month);
    EXPECT_EQ(4, date.day);
    EXPECT_EQ(10, date.hour);
    EXPECT_EQ(0, date.min);
    EXPECT_EQ(0, date.sec);
}

TEST(Time, GetTimeStructAlternate_20101104_103245)
{
    double mjd = 55504.43940972211;
    Time date = asTime::GetTimeStruct(mjd, asUSE_ALTERNATE_METHOD);

    EXPECT_EQ(2010, date.year);
    EXPECT_EQ(11, date.month);
    EXPECT_EQ(4, date.day);
    EXPECT_EQ(10, date.hour);
    EXPECT_EQ(32, date.min);
    EXPECT_EQ(45, date.sec);
}

TEST(Time, GetTimeStructOther)
{
    Time date = asTime::GetTimeStruct(2010, 11, 4, 10, 32, 45);

    EXPECT_EQ(2010, date.year);
    EXPECT_EQ(11, date.month);
    EXPECT_EQ(4, date.day);
    EXPECT_EQ(10, date.hour);
    EXPECT_EQ(32, date.min);
    EXPECT_EQ(45, date.sec);
}

TEST(Time, GetYearNormal_20101104_103245)
{
    double mjd = 55504.43940972211;
    int result = asTime::GetYear(mjd, asUSE_NORMAL_METHOD);

    EXPECT_EQ(2010, result);
}

TEST(Time, GetYearAlternate_20101104_103245)
{
    double mjd = 55504.43940972211;
    int result = asTime::GetYear(mjd, asUSE_ALTERNATE_METHOD);

    EXPECT_EQ(2010, result);
}

TEST(Time, GetMonthNormal_20101104_103245)
{
    double mjd = 55504.43940972211;
    int result = asTime::GetMonth(mjd, asUSE_NORMAL_METHOD);

    EXPECT_EQ(11, result);
}

TEST(Time, GetMonthAlternate_20101104_103245)
{
    double mjd = 55504.43940972211;
    int result = asTime::GetMonth(mjd, asUSE_ALTERNATE_METHOD);

    EXPECT_EQ(11, result);
}

TEST(Time, GetDayNormal_20101104_103245)
{
    double mjd = 55504.43940972211;
    int result = asTime::GetDay(mjd, asUSE_NORMAL_METHOD);

    EXPECT_EQ(4, result);
}

TEST(Time, GetDayAlternate_20101104_103245)
{
    double mjd = 55504.43940972211;
    int result = asTime::GetDay(mjd, asUSE_ALTERNATE_METHOD);

    EXPECT_EQ(4, result);
}

TEST(Time, GetHourNormal_20101104_103245)
{
    double mjd = 55504.43940972211;
    int result = asTime::GetHour(mjd, asUSE_NORMAL_METHOD);

    EXPECT_EQ(10, result);
}

TEST(Time, GetHourAlternate_20101104_103245)
{
    double mjd = 55504.43940972211;
    int result = asTime::GetHour(mjd, asUSE_ALTERNATE_METHOD);

    EXPECT_EQ(10, result);
}

TEST(Time, GetMinuteNormal_20101104_103245)
{
    double mjd = 55504.43940972211;
    int result = asTime::GetMinute(mjd, asUSE_NORMAL_METHOD);

    EXPECT_EQ(32, result);
}

TEST(Time, GetMinuteAlternate_20101104_103245)
{
    double mjd = 55504.43940972211;
    int result = asTime::GetMinute(mjd, asUSE_ALTERNATE_METHOD);

    EXPECT_EQ(32, result);
}

TEST(Time, GetSecondNormal_20101104_103245)
{
    double mjd = 55504.43940972211;
    int result = asTime::GetSecond(mjd, asUSE_NORMAL_METHOD);

    EXPECT_EQ(45, result);
}

TEST(Time, GetSecondAlternate_20101104_103245)
{
    double mjd = 55504.43940972211;
    int result = asTime::GetSecond(mjd, asUSE_ALTERNATE_METHOD);

    EXPECT_EQ(45, result);
}

TEST(Time, NowLocalMJD)
{
    double mjd = asTime::NowMJD(asLOCAL);
    wxString datestr = asTime::GetStringTime(mjd);

    wxPrintf("Local time is %s\n", datestr);
}

TEST(Time, NowLocalTimeStruct)
{
    Time date = asTime::NowTimeStruct(asLOCAL);
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
    Time date = asTime::NowTimeStruct(asUTM);
    wxString datestr = asTime::GetStringTime(date);

    wxPrintf("UTM time is %s\n", datestr);
}

TEST(Time, GetStringDateMJD)
{
    double mjd = 55504.43940972211;

    wxString datestr = asTime::GetStringTime(mjd, classic);

    int result = datestr.CompareTo(_T("04.11.2010"));

    EXPECT_EQ(0, result);
}

TEST(Time, GetStringDateTimeStruct)
{
    Time date;
    date.year = 2010;
    date.month = 11;
    date.day = 4;
    date.hour = 10;
    date.min = 32;
    date.sec = 45;

    wxString datestr = asTime::GetStringTime(date, classic);

    int result = datestr.CompareTo(_T("04.11.2010"));

    EXPECT_EQ(0, result);
}

TEST(Time, GetStringDateReverseMJD)
{
    double mjd = 55504.43940972211;

    wxString datestr = asTime::GetStringTime(mjd, YYYYMMDD);

    int result = datestr.CompareTo(_T("2010/11/04"));

    EXPECT_EQ(0, result);
}

TEST(Time, GetStringDateReverseTimeStruct)
{
    Time date;
    date.year = 2010;
    date.month = 11;
    date.day = 4;
    date.hour = 10;
    date.min = 32;
    date.sec = 45;

    wxString datestr = asTime::GetStringTime(date, YYYYMMDD);

    int result = datestr.CompareTo(_T("2010/11/04"));

    EXPECT_EQ(0, result);
}

TEST(Time, GetStringTimeTimeStruct)
{
    Time date;
    date.year = 2010;
    date.month = 11;
    date.day = 4;
    date.hour = 3;
    date.min = 5;
    date.sec = 5;

    wxString datestr = asTime::GetStringTime(date, timeOnly);

    int result = datestr.CompareTo(_T("03:05"));

    EXPECT_EQ(0, result);
}

TEST(Time, GetTimeFromStringFormatISOdate)
{
    double conversion = asTime::GetTimeFromString("2007-11-23", ISOdate);
    double mjd = asTime::GetMJD(2007, 11, 23);

    EXPECT_DOUBLE_EQ(mjd, conversion);
}

TEST(Time, GetTimeFromStringFormatISOdatetime)
{
    double conversion = asTime::GetTimeFromString("2007-11-23 13:05:01", ISOdateTime);
    double mjd = asTime::GetMJD(2007, 11, 23, 13, 5, 1);

    EXPECT_DOUBLE_EQ(mjd, conversion);
}

TEST(Time, GetTimeFromStringFormatDDMMYYYY)
{
    double conversion = asTime::GetTimeFromString("23.11.2007", DDMMYYYY);
    double mjd = asTime::GetMJD(2007, 11, 23);

    EXPECT_DOUBLE_EQ(mjd, conversion);
}

TEST(Time, GetTimeFromStringFormatDDMMYYYYSlashes)
{
    double conversion = asTime::GetTimeFromString("23/11/2007", DDMMYYYY);
    double mjd = asTime::GetMJD(2007, 11, 23);

    EXPECT_DOUBLE_EQ(mjd, conversion);
}

TEST(Time, GetTimeFromStringFormatDDMMYYYYException)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("23.11.07", DDMMYYYY), asException);
}

TEST(Time, GetTimeFromStringFormatYYYYMMDD)
{
    double conversion = asTime::GetTimeFromString("2007.11.23", YYYYMMDD);
    double mjd = asTime::GetMJD(2007, 11, 23);

    EXPECT_DOUBLE_EQ(mjd, conversion);
}

TEST(Time, GetTimeFromStringFormatYYYYMMDDException)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("23.11.2007", YYYYMMDD), asException);
}

TEST(Time, GetTimeFromStringFormatDDMMYYYYhhmm)
{
    double conversion = asTime::GetTimeFromString("23.11.2007 13:05", DDMMYYYYhhmm);
    double mjd = asTime::GetMJD(2007, 11, 23, 13, 5);

    EXPECT_DOUBLE_EQ(mjd, conversion);
}

TEST(Time, GetTimeFromStringFormatDDMMYYYYhhmmException)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("23.11.07 13:05", DDMMYYYYhhmm), asException);
}

TEST(Time, GetTimeFromStringFormatYYYYMMDDhhmm)
{
    double conversion = asTime::GetTimeFromString("2007.11.23 13:05", YYYYMMDDhhmm);
    double mjd = asTime::GetMJD(2007, 11, 23, 13, 5);

    EXPECT_DOUBLE_EQ(mjd, conversion);
}

TEST(Time, GetTimeFromStringFormatYYYYMMDDhhmmException)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("23.11.2007 13:05", YYYYMMDDhhmm), asException);
}

TEST(Time, GetTimeFromStringFormatDDMMYYYYhhmmss)
{
    double conversion = asTime::GetTimeFromString("23.11.2007 13:05:01", DDMMYYYYhhmmss);
    double mjd = asTime::GetMJD(2007, 11, 23, 13, 5, 1);

    EXPECT_DOUBLE_EQ(mjd, conversion);
}

TEST(Time, GetTimeFromStringFormatDDMMYYYYhhmmssException)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("23.11.07 13:05:01", DDMMYYYYhhmmss), asException);
}

TEST(Time, GetTimeFromStringFormatYYYYMMDDhhmmss)
{
    double conversion = asTime::GetTimeFromString("2007.11.23 13:05:01", YYYYMMDDhhmmss);
    double mjd = asTime::GetMJD(2007, 11, 23, 13, 5, 1);

    EXPECT_DOUBLE_EQ(mjd, conversion);
}

TEST(Time, GetTimeFromStringFormatYYYYMMDDhhmmssException)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("23.11.2007 13:05:01", YYYYMMDDhhmmss), asException);
}

TEST(Time, GetTimeFromStringFormathhmmException)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("13:05:01", hhmm), asException);
}

TEST(Time, GetTimeFromStringFormatnowplushours)
{
    double conversion = asTime::GetTimeFromString("+2", nowPlusHours);
    wxString datestr = asTime::GetStringTime(conversion);

    wxPrintf("UTM time +2 hours is %s\n", datestr);
}

TEST(Time, GetTimeFromStringFormatnowplushoursException)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("+2:23", nowPlusHours), asException);
}

TEST(Time, GetTimeFromStringFormatnowplushoursExceptionDot)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("+2.23", nowPlusHours), asException);
}

TEST(Time, GetTimeFromStringFormatnowminushours)
{
    double conversion = asTime::GetTimeFromString("-2", nowMinusHours);
    wxString datestr = asTime::GetStringTime(conversion);

    wxPrintf("UTM time -2 hours is %s\n", datestr);
}

TEST(Time, GetTimeFromStringFormatnowminushoursException)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("-2:23", nowMinusHours), asException);
}

TEST(Time, GetTimeFromStringFormatnowminushoursExceptionDot)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("-2.23", nowMinusHours), asException);
}

TEST(Time, GetTimeFromStringFormatnowminushoursExceptionSignPlus)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("+2", nowMinusHours), asException);
}

TEST(Time, GetTimeFromStringFormatnowminushoursExceptionSignNo)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("2", nowMinusHours), asException);
}

TEST(Time, GetTimeFromStringFormatautoDDMMYYYY)
{
    double conversion = asTime::GetTimeFromString("23.11.2007", guess);
    double mjd = asTime::GetMJD(2007, 11, 23);

    EXPECT_DOUBLE_EQ(mjd, conversion);
}

TEST(Time, GetTimeFromStringFormatautoDDMMYYYYSlashes)
{
    double conversion = asTime::GetTimeFromString("23/11/2007", guess);
    double mjd = asTime::GetMJD(2007, 11, 23);

    EXPECT_DOUBLE_EQ(mjd, conversion);
}

TEST(Time, GetTimeFromStringFormatautoDDMMYYYYException)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("23.11.07", guess), asException);
}

TEST(Time, GetTimeFromStringFormatautoYYYYMMDD)
{
    double conversion = asTime::GetTimeFromString("2007.11.23", guess);
    double mjd = asTime::GetMJD(2007, 11, 23);

    EXPECT_DOUBLE_EQ(mjd, conversion);
}

TEST(Time, GetTimeFromStringFormatautoYYYYMMDDException)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("11.2007", guess), asException);
}

TEST(Time, GetTimeFromStringFormatautoDDMMYYYYhhmm)
{
    double conversion = asTime::GetTimeFromString("23.11.2007 13:05", guess);
    double mjd = asTime::GetMJD(2007, 11, 23, 13, 5);

    EXPECT_DOUBLE_EQ(mjd, conversion);
}

TEST(Time, GetTimeFromStringFormatautoDDMMYYYYhhmmException)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("23.11.07 13:05", guess), asException);
}

TEST(Time, GetTimeFromStringFormatautoYYYYMMDDhhmm)
{
    double conversion = asTime::GetTimeFromString("2007.11.23 13:05", guess);
    double mjd = asTime::GetMJD(2007, 11, 23, 13, 5);

    EXPECT_DOUBLE_EQ(mjd, conversion);
}

TEST(Time, GetTimeFromStringFormatautoYYYYMMDDhhmmException)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("23.11.07 13:05", guess), asException);
}

TEST(Time, GetTimeFromStringFormatautoDDMMYYYYhhmmss)
{
    double conversion = asTime::GetTimeFromString("23.11.2007 13:05:01", guess);
    double mjd = asTime::GetMJD(2007, 11, 23, 13, 5, 1);

    EXPECT_DOUBLE_EQ(mjd, conversion);
}

TEST(Time, GetTimeFromStringFormatautoDDMMYYYYhhmmssException)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("23.11.07 13:05:01", guess), asException);
}

TEST(Time, GetTimeFromStringFormatautoYYYYMMDDhhmmss)
{
    double conversion = asTime::GetTimeFromString("2007.11.23 13:05:01", guess);
    double mjd = asTime::GetMJD(2007, 11, 23, 13, 5, 1);

    EXPECT_DOUBLE_EQ(mjd, conversion);
}

TEST(Time, GetTimeFromStringFormatautoYYYYMMDDhhmmssException)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("23.11.07 13:05:01", guess), asException);
}

TEST(Time, GetTimeFromStringFormatautohhmmException)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("13:05:01", guess), asException);
}

TEST(Time, GetTimeFromStringFormatautonowplushours)
{
    double conversion = asTime::GetTimeFromString("+2", guess);
    wxString datestr = asTime::GetStringTime(conversion);

    wxPrintf("UTM time +2 hours is %s\n", datestr);
}

TEST(Time, GetTimeFromStringFormatautonowplushoursException)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("+2:23", guess), asException);
}

TEST(Time, GetTimeFromStringFormatautonowplushoursExceptionDot)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("+2.23", guess), asException);
}

TEST(Time, GetTimeFromStringFormatautonowminushours)
{
    double conversion = asTime::GetTimeFromString("-2", guess);
    wxString datestr = asTime::GetStringTime(conversion);

    wxPrintf("UTM time -2 hours is %s\n", datestr);
}

TEST(Time, GetTimeFromStringFormatautonowminushoursException)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("-2:23", guess), asException);
}

TEST(Time, GetTimeFromStringFormatautonowminushoursExceptionDot)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("-2.23", guess), asException);
}

TEST(Time, GetTimeFromStringFormatautonowminushoursExceptionSignNo)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("2", guess), asException);
}

TEST(Time, GetTimeFromStringFormatautonowminushoursExceptionSignPlusText)
{
    wxLogNull logNo;

    ASSERT_THROW(asTime::GetTimeFromString("+2hours", guess), asException);
}

TEST(Time, AddYear1972)
{
    double mjd = asTime::GetMJD(1972, 11, 23, 13, 5);
    mjd = asTime::AddYear(mjd);
    double mjdafter = asTime::GetMJD(1973, 11, 23, 13, 5);

    EXPECT_DOUBLE_EQ(mjdafter, mjd);
}

TEST(Time, AddYear1972Leap)
{
    double mjd = asTime::GetMJD(1972, 2, 23, 13, 5);
    mjd = asTime::AddYear(mjd);
    double mjdafter = asTime::GetMJD(1973, 2, 23, 13, 5);

    EXPECT_DOUBLE_EQ(mjdafter, mjd);
}

TEST(Time, AddYear1900)
{
    double mjd = asTime::GetMJD(1900, 2, 23, 13, 5);
    mjd = asTime::AddYear(mjd);
    double mjdafter = asTime::GetMJD(1901, 2, 23, 13, 5);

    EXPECT_DOUBLE_EQ(mjdafter, mjd);
}

TEST(Time, AddYear2000)
{
    double mjd = asTime::GetMJD(2000, 11, 23, 13, 5);
    mjd = asTime::AddYear(mjd);
    double mjdafter = asTime::GetMJD(2001, 11, 23, 13, 5);

    EXPECT_DOUBLE_EQ(mjdafter, mjd);
}

TEST(Time, AddYear2000Leap)
{
    double mjd = asTime::GetMJD(2000, 2, 23, 13, 5);
    mjd = asTime::AddYear(mjd);
    double mjdafter = asTime::GetMJD(2001, 2, 23, 13, 5);

    EXPECT_DOUBLE_EQ(mjdafter, mjd);
}

TEST(Time, SubtractYear1972Leap)
{
    double mjd = asTime::GetMJD(1972, 11, 23, 13, 5);
    mjd = asTime::SubtractYear(mjd);
    double mjdafter = asTime::GetMJD(1971, 11, 23, 13, 5);

    EXPECT_DOUBLE_EQ(mjdafter, mjd);
}

TEST(Time, SubtractYear1972)
{
    double mjd = asTime::GetMJD(1972, 2, 23, 13, 5);
    mjd = asTime::SubtractYear(mjd);
    double mjdafter = asTime::GetMJD(1971, 2, 23, 13, 5);

    EXPECT_DOUBLE_EQ(mjdafter, mjd);
}

TEST(Time, SubtractYear1900)
{
    double mjd = asTime::GetMJD(1900, 11, 23, 13, 5);
    mjd = asTime::SubtractYear(mjd);
    double mjdafter = asTime::GetMJD(1899, 11, 23, 13, 5);

    EXPECT_DOUBLE_EQ(mjdafter, mjd);
}

TEST(Time, SubtractYear2000Leap)
{
    double mjd = asTime::GetMJD(2000, 11, 23, 13, 5);
    mjd = asTime::SubtractYear(mjd);
    double mjdafter = asTime::GetMJD(1999, 11, 23, 13, 5);

    EXPECT_DOUBLE_EQ(mjdafter, mjd);
}

TEST(Time, SubtractYear2000)
{
    double mjd = asTime::GetMJD(2000, 2, 23, 13, 5);
    mjd = asTime::SubtractYear(mjd);
    double mjdafter = asTime::GetMJD(1999, 2, 23, 13, 5);

    EXPECT_DOUBLE_EQ(mjdafter, mjd);
}

TEST(Time, SubtractYear2000Feb28)
{
    double mjd = asTime::GetMJD(2000, 2, 28, 13, 5);
    mjd = asTime::SubtractYear(mjd);
    double mjdafter = asTime::GetMJD(1999, 2, 28, 13, 5);

    EXPECT_DOUBLE_EQ(mjdafter, mjd);
}

TEST(Time, SubtractYear2000Feb29)
{
    double mjd = asTime::GetMJD(2000, 2, 29, 13, 5);
    mjd = asTime::SubtractYear(mjd);
    double mjdafter = asTime::GetMJD(1999, 2, 28, 13, 5);

    EXPECT_DOUBLE_EQ(mjdafter, mjd);
}
