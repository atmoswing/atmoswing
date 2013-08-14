#include "include_tests.h"
#include "asTime.h"

#include "UnitTest++.h"

namespace
{

TEST(IsLeapYearDivisableBy4)
{
    const bool Result = asTime::IsLeapYear(1972);
    CHECK_EQUAL(true, Result);
}

TEST(IsLeapYearDivisableBy100)
{
    const bool Result = asTime::IsLeapYear(1900);
    CHECK_EQUAL(false, Result);
}

TEST(IsLeapYearDivisableBy400)
{
    const bool Result = asTime::IsLeapYear(2000);
    CHECK_EQUAL(true, Result);
}

TEST(IsLeapYearNo)
{
    const bool Result = asTime::IsLeapYear(1973);
    CHECK_EQUAL(false, Result);
}

TEST(GetMJDNormal_20040101)
{
    const double Result = asTime::GetMJD(2004, 1, 1, 0, 0, 0, asUSE_NORMAL_METHOD);
    CHECK_CLOSE(53005, Result, 0.00000001);
}

TEST(GetMJDNormal_20040101_120000)
{
    double Result = asTime::GetMJD(2004, 1, 1, 12, 0, 0, asUSE_NORMAL_METHOD);
    CHECK_CLOSE(53005.5, Result, 0.00000001);
}

TEST(GetMJDNormal_20101104_120000)
{
    double Result = asTime::GetMJD(2010, 11, 4, 12, 0, 0, asUSE_NORMAL_METHOD);
    CHECK_CLOSE(55504.5, Result, 0.00000001);
}

TEST(GetMJDNormal_20101104_100000)
{
    double Result = asTime::GetMJD(2010, 11, 4, 10, 0, 0, asUSE_NORMAL_METHOD);
    CHECK_CLOSE(55504.41666666651, Result, 0.00000001);
}

TEST(GetMJDNormal_20101104_103245)
{
    double Result = asTime::GetMJD(2010, 11, 4, 10, 32, 45, asUSE_NORMAL_METHOD);
    CHECK_CLOSE(55504.43940972211, Result, 0.00000001);
}

TEST(GetMJDAlternate_20040101)
{
    const double Result = asTime::GetMJD(2004, 1, 1, 0, 0, 0, asUSE_ALTERNATE_METHOD);
    CHECK_CLOSE(53005, Result, 0.00000001);
}

TEST(GetMJDAlternate_20040101_120000)
{
    double Result = asTime::GetMJD(2004, 1, 1, 12, 0, 0, asUSE_ALTERNATE_METHOD);
    CHECK_CLOSE(53005.5, Result, 0.00000001);
}

TEST(GetMJDAlternate_20101104_120000)
{
    double Result = asTime::GetMJD(2010, 11, 4, 12, 0, 0, asUSE_ALTERNATE_METHOD);
    CHECK_CLOSE(55504.5, Result, 0.00000001);
}

TEST(GetMJDAlternate_20101104_100000)
{
    double Result = asTime::GetMJD(2010, 11, 4, 10, 0, 0, asUSE_ALTERNATE_METHOD);
    CHECK_CLOSE(55504.41666666651, Result, 0.00000001);
}

TEST(GetMJDAlternate_20101104_103245)
{
    double Result = asTime::GetMJD(2010, 11, 4, 10, 32, 45, asUSE_ALTERNATE_METHOD);
    CHECK_CLOSE(55504.43940972211, Result, 0.00000001);
}
/*
TEST(GetMJDStructNormal_20040101)
{
    TimeStruct date;
    asTime::TimeStructInit(date);

    date.year = 2004;
    date.month = 1;
    date.day = 1;

    const double Result = asTime::GetMJD(date, asUSE_NORMAL_METHOD);
    CHECK_CLOSE(53005, Result, 0.00000001);
}

TEST(GetMJDStructNormal_20040101_120000)
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

TEST(GetMJDStructNormal_20101104_120000)
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

TEST(GetMJDStructNormal_20101104_100000)
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

TEST(GetMJDStructNormal_20101104_103245)
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

TEST(GetMJDStructAlternate_20040101)
{
    TimeStruct date;
    asTime::TimeStructInit(date);

    date.year = 2004;
    date.month = 1;
    date.day = 1;

    const double Result = asTime::GetMJD(date, asUSE_ALTERNATE_METHOD);
    CHECK_CLOSE(53005, Result, 0.00000001);
}

TEST(GetMJDStructAlternate_20040101_120000)
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

TEST(GetMJDStructAlternate_20101104_120000)
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

TEST(GetMJDStructAlternate_20101104_100000)
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

TEST(GetMJDStructAlternate_20101104_103245)
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
TEST(GetTimeStructNormal_20040101)
{
    double Mjd = 53005;
    TimeStruct date = asTime::GetTimeStruct(Mjd, asUSE_NORMAL_METHOD);

    CHECK_EQUAL(2004, date.year);
    CHECK_EQUAL(1, date.month);
    CHECK_EQUAL(1, date.day);
}

TEST(GetTimeStructNormal_20040101_120000)
{
    double Mjd = 53005.5;
    TimeStruct date = asTime::GetTimeStruct(Mjd, asUSE_NORMAL_METHOD);

    CHECK_EQUAL(2004, date.year);
    CHECK_EQUAL(1, date.month);
    CHECK_EQUAL(1, date.day);
    CHECK_EQUAL(12, date.hour);
}

TEST(GetTimeStructNormal_20101104_120000)
{
    double Mjd = 55504.5;
    TimeStruct date = asTime::GetTimeStruct(Mjd, asUSE_NORMAL_METHOD);

    CHECK_EQUAL(2010, date.year);
    CHECK_EQUAL(11, date.month);
    CHECK_EQUAL(4, date.day);
    CHECK_EQUAL(12, date.hour);
}

TEST(GetTimeStructNormal_20101104_100000)
{
    double Mjd = 55504.41666666651;
    TimeStruct date = asTime::GetTimeStruct(Mjd, asUSE_NORMAL_METHOD);

    CHECK_EQUAL(2010, date.year);
    CHECK_EQUAL(11, date.month);
    CHECK_EQUAL(4, date.day);
    CHECK_EQUAL(10, date.hour);
}

TEST(GetTimeStructNormal_20101104_103245)
{
    double Mjd = 55504.43940972211;
    TimeStruct date = asTime::GetTimeStruct(Mjd, asUSE_NORMAL_METHOD);

    CHECK_EQUAL(2010, date.year);
    CHECK_EQUAL(11, date.month);
    CHECK_EQUAL(4, date.day);
    CHECK_EQUAL(10, date.hour);
    CHECK_EQUAL(32, date.min);
    CHECK_EQUAL(45, date.sec);
}

TEST(GetTimeStructAlternate_20040101)
{
    double Mjd = 53005;
    TimeStruct date = asTime::GetTimeStruct(Mjd, asUSE_ALTERNATE_METHOD);

    CHECK_EQUAL(2004, date.year);
    CHECK_EQUAL(1, date.month);
    CHECK_EQUAL(1, date.day);
}

TEST(GetTimeStructAlternate_20040101_120000)
{
    double Mjd = 53005.5;
    TimeStruct date = asTime::GetTimeStruct(Mjd, asUSE_ALTERNATE_METHOD);

    CHECK_EQUAL(2004, date.year);
    CHECK_EQUAL(1, date.month);
    CHECK_EQUAL(1, date.day);
    CHECK_EQUAL(12, date.hour);
}

TEST(GetTimeStructAlternate_20101104_120000)
{
    double Mjd = 55504.5;
    TimeStruct date = asTime::GetTimeStruct(Mjd, asUSE_ALTERNATE_METHOD);

    CHECK_EQUAL(2010, date.year);
    CHECK_EQUAL(11, date.month);
    CHECK_EQUAL(4, date.day);
    CHECK_EQUAL(12, date.hour);
}

TEST(GetTimeStructAlternate_20101104_100000)
{
    double Mjd = 55504.41666666651;
    TimeStruct date = asTime::GetTimeStruct(Mjd, asUSE_ALTERNATE_METHOD);

    CHECK_EQUAL(2010, date.year);
    CHECK_EQUAL(11, date.month);
    CHECK_EQUAL(4, date.day);
    CHECK_EQUAL(10, date.hour);
    CHECK_EQUAL(0, date.min);
    CHECK_EQUAL(0, date.sec);
}

TEST(GetTimeStructAlternate_20101104_103245)
{
    double Mjd = 55504.43940972211;
    TimeStruct date = asTime::GetTimeStruct(Mjd, asUSE_ALTERNATE_METHOD);

    CHECK_EQUAL(2010, date.year);
    CHECK_EQUAL(11, date.month);
    CHECK_EQUAL(4, date.day);
    CHECK_EQUAL(10, date.hour);
    CHECK_EQUAL(32, date.min);
    CHECK_EQUAL(45, date.sec);
}

TEST(GetTimeStructOther)
{
    TimeStruct date = asTime::GetTimeStruct(2010, 11, 4, 10, 32, 45);

    CHECK_EQUAL(2010, date.year);
    CHECK_EQUAL(11, date.month);
    CHECK_EQUAL(4, date.day);
    CHECK_EQUAL(10, date.hour);
    CHECK_EQUAL(32, date.min);
    CHECK_EQUAL(45, date.sec);
}

TEST(GetYearNormal_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetYear( Mjd, asUSE_NORMAL_METHOD );

    CHECK_EQUAL(2010, Result);
}

TEST(GetYearAlternate_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetYear( Mjd, asUSE_ALTERNATE_METHOD );

    CHECK_EQUAL(2010, Result);
}

TEST(GetMonthNormal_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetMonth( Mjd, asUSE_NORMAL_METHOD );

    CHECK_EQUAL(11, Result);
}

TEST(GetMonthAlternate_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetMonth( Mjd, asUSE_ALTERNATE_METHOD );

    CHECK_EQUAL(11, Result);
}

TEST(GetDayNormal_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetDay( Mjd, asUSE_NORMAL_METHOD );

    CHECK_EQUAL(4, Result);
}

TEST(GetDayAlternate_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetDay( Mjd, asUSE_ALTERNATE_METHOD );

    CHECK_EQUAL(4, Result);
}

TEST(GetHourNormal_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetHour( Mjd, asUSE_NORMAL_METHOD );

    CHECK_EQUAL(10, Result);
}

TEST(GetHourAlternate_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetHour( Mjd, asUSE_ALTERNATE_METHOD );

    CHECK_EQUAL(10, Result);
}

TEST(GetMinuteNormal_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetMinute( Mjd, asUSE_NORMAL_METHOD );

    CHECK_EQUAL(32, Result);
}

TEST(GetMinuteAlternate_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetMinute( Mjd, asUSE_ALTERNATE_METHOD );

    CHECK_EQUAL(32, Result);
}

TEST(GetSecondNormal_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetSecond( Mjd, asUSE_NORMAL_METHOD );

    CHECK_EQUAL(45, Result);
}

TEST(GetSecondAlternate_20101104_103245)
{
    double Mjd = 55504.43940972211;
    const int Result = asTime::GetSecond( Mjd, asUSE_ALTERNATE_METHOD );

    CHECK_EQUAL(45, Result);
}
/*
TEST(GetSeasonStartDJF)
{
    TimeStruct val = asTime::GetSeasonStart(DJF);

    CHECK_EQUAL(1, val.day);
    CHECK_EQUAL(12, val.month);
}

TEST(GetSeasonStartMAM)
{
    TimeStruct val = asTime::GetSeasonStart(MAM);

    CHECK_EQUAL(1, val.day);
    CHECK_EQUAL(3, val.month);
}

TEST(GetSeasonStartJJA)
{
    TimeStruct val = asTime::GetSeasonStart(JJA);

    CHECK_EQUAL(1, val.day);
    CHECK_EQUAL(6, val.month);
}

TEST(GetSeasonStartSON)
{
    TimeStruct val = asTime::GetSeasonStart(SON);

    CHECK_EQUAL(1, val.day);
    CHECK_EQUAL(9, val.month);
}

TEST(GetSeasonEndDJF_1973)
{
    TimeStruct val = asTime::GetSeasonEnd(DJF, 1973);

    CHECK_EQUAL(28, val.day);
    CHECK_EQUAL(2, val.month);
}

TEST(GetSeasonEndDJF_2000)
{
    TimeStruct val = asTime::GetSeasonEnd(DJF, 2000);

    CHECK_EQUAL(29, val.day);
    CHECK_EQUAL(2, val.month);
}

TEST(GetSeasonEndDJF_1972)
{
    TimeStruct val = asTime::GetSeasonEnd(DJF, 1972);

    CHECK_EQUAL(29, val.day);
    CHECK_EQUAL(2, val.month);
}

TEST(GetSeasonEndDJF_1900)
{
    TimeStruct val = asTime::GetSeasonEnd(DJF, 1900);

    CHECK_EQUAL(28, val.day);
    CHECK_EQUAL(2, val.month);
}

TEST(GetSeasonEndMAM)
{
    TimeStruct val = asTime::GetSeasonEnd(MAM, 1900);

    CHECK_EQUAL(31, val.day);
    CHECK_EQUAL(5, val.month);
}

TEST(GetSeasonEndJJA)
{
    TimeStruct val = asTime::GetSeasonEnd(JJA, 1900);

    CHECK_EQUAL(31, val.day);
    CHECK_EQUAL(8, val.month);
}

TEST(GetSeasonEndSON)
{
    TimeStruct val = asTime::GetSeasonEnd(SON, 1900);

    CHECK_EQUAL(30, val.day);
    CHECK_EQUAL(11, val.month);
}

TEST(GetSeasonJan)
{
    Season Result = asTime::GetSeason(1);
    CHECK_EQUAL(DJF, Result);
}

TEST(GetSeasonFeb)
{
    Season Result = asTime::GetSeason(2);
    CHECK_EQUAL(DJF, Result);
}

TEST(GetSeasonMar)
{
    Season Result = asTime::GetSeason(3);
    CHECK_EQUAL(MAM, Result);
}

TEST(GetSeasonApr)
{
    Season Result = asTime::GetSeason(4);
    CHECK_EQUAL(MAM, Result);
}

TEST(GetSeasonMay)
{
    Season Result = asTime::GetSeason(5);
    CHECK_EQUAL(MAM, Result);
}

TEST(GetSeasonJun)
{
    Season Result = asTime::GetSeason(6);
    CHECK_EQUAL(JJA, Result);
}

TEST(GetSeasonJul)
{
    Season Result = asTime::GetSeason(7);
    CHECK_EQUAL(JJA, Result);
}

TEST(GetSeasonAug)
{
    Season Result = asTime::GetSeason(8);
    CHECK_EQUAL(JJA, Result);
}

TEST(GetSeasonSep)
{
    Season Result = asTime::GetSeason(9);
    CHECK_EQUAL(SON, Result);
}

TEST(GetSeasonOct)
{
    Season Result = asTime::GetSeason(10);
    CHECK_EQUAL(SON, Result);
}

TEST(GetSeasonNov)
{
    Season Result = asTime::GetSeason(11);
    CHECK_EQUAL(SON, Result);
}

TEST(GetSeasonDec)
{
    Season Result = asTime::GetSeason(12);
    CHECK_EQUAL(DJF, Result);
}

TEST(GetSeasonException)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetSeason(13), asException);
    }
}

TEST(TimeTmToTimeStruct)
{
    struct tm DateTm;
    DateTm.tm_year = 2010-1900;
    DateTm.tm_mon = 11-1;
    DateTm.tm_mday = 4;
    DateTm.tm_hour = 10;
    DateTm.tm_min = 32;
    DateTm.tm_sec = 45;

    TimeStruct date = asTime::TimeTmToTimeStruct(DateTm);

    CHECK_EQUAL(2010, date.year);
    CHECK_EQUAL(11, date.month);
    CHECK_EQUAL(4, date.day);
    CHECK_EQUAL(10, date.hour);
    CHECK_EQUAL(32, date.min);
    CHECK_EQUAL(45, date.sec);
}

TEST(TimeTmToMJD)
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
TEST(NowLocalMJD)
{
    double mjd = asTime::NowMJD(asLOCAL);
    wxString datestr = asTime::GetStringTime(mjd);

    wxString str;
    str.Printf("Local time is %s\n", datestr.c_str());

    printf("%s", str.mb_str(wxConvUTF8).data());
}

TEST(NowLocalTimeStruct)
{
    TimeStruct date = asTime::NowTimeStruct(asLOCAL);
    wxString datestr = asTime::GetStringTime(date);

    wxString str;
    str.Printf("Local time is %s\n", datestr.c_str());

    printf("%s", str.mb_str(wxConvUTF8).data());
}

TEST(NowMJD)
{
    double mjd = asTime::NowMJD(asUTM);
    wxString datestr = asTime::GetStringTime(mjd);

    wxString str;
    str.Printf("UTM time is %s\n", datestr.c_str());

    printf("%s", str.mb_str(wxConvUTF8).data());
}

TEST(NowTimeStruct)
{
    TimeStruct date = asTime::NowTimeStruct(asUTM);
    wxString datestr = asTime::GetStringTime(date);

    wxString str;
    str.Printf("UTM time is %s\n", datestr.c_str());

    printf("%s", str.mb_str(wxConvUTF8).data());
}

TEST(GetStringDateMJD)
{
    double Mjd = 55504.43940972211;

    wxString datestr = asTime::GetStringTime(Mjd, classic);

    int Result = datestr.CompareTo(_T("04.11.2010"));

    CHECK_EQUAL(0, Result);
}

TEST(GetStringDateTimeStruct)
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

    CHECK_EQUAL(0, Result);
}

TEST(GetStringDateReverseMJD)
{
    double Mjd = 55504.43940972211;

    wxString datestr = asTime::GetStringTime(Mjd, YYYYMMDD);

    int Result = datestr.CompareTo(_T("2010/11/04"));

    CHECK_EQUAL(0, Result);
}

TEST(GetStringDateReverseTimeStruct)
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

    CHECK_EQUAL(0, Result);
}

TEST(GetStringTimeTimeStruct)
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

    CHECK_EQUAL(0, Result);
}

TEST(GetTimeFromStringFormatDDMMYYYY)
{
    double conversion = asTime::GetTimeFromString("23.11.2007", DDMMYYYY);
    double mjd = asTime::GetMJD(2007,11,23);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(GetTimeFromStringFormatDDMMYYYYSlashes)
{
    double conversion = asTime::GetTimeFromString("23/11/2007", DDMMYYYY);
    double mjd = asTime::GetMJD(2007,11,23);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(GetTimeFromStringFormatDDMMYYYYException)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("23.11.07", DDMMYYYY), asException);
    }
}

TEST(GetTimeFromStringFormatYYYYMMDD)
{
    double conversion = asTime::GetTimeFromString("2007.11.23", YYYYMMDD);
    double mjd = asTime::GetMJD(2007,11,23);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(GetTimeFromStringFormatYYYYMMDDException)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("23.11.2007", YYYYMMDD), asException);
    }
}

TEST(GetTimeFromStringFormatDDMMYYYYhhmm)
{
    double conversion = asTime::GetTimeFromString("23.11.2007 13:05", DDMMYYYYhhmm);
    double mjd = asTime::GetMJD(2007,11,23,13,5);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(GetTimeFromStringFormatDDMMYYYYhhmmException)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("23.11.07 13:05", DDMMYYYYhhmm), asException);
    }
}

TEST(GetTimeFromStringFormatYYYYMMDDhhmm)
{
    double conversion = asTime::GetTimeFromString("2007.11.23 13:05", YYYYMMDDhhmm);
    double mjd = asTime::GetMJD(2007,11,23,13,5);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(GetTimeFromStringFormatYYYYMMDDhhmmException)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("23.11.2007 13:05", YYYYMMDDhhmm), asException);
    }
}

TEST(GetTimeFromStringFormatDDMMYYYYhhmmss)
{
    double conversion = asTime::GetTimeFromString("23.11.2007 13:05:01", DDMMYYYYhhmmss);
    double mjd = asTime::GetMJD(2007,11,23,13,5,1);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(GetTimeFromStringFormatDDMMYYYYhhmmssException)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("23.11.07 13:05:01", DDMMYYYYhhmmss), asException);
    }
}

TEST(GetTimeFromStringFormatYYYYMMDDhhmmss)
{
    double conversion = asTime::GetTimeFromString("2007.11.23 13:05:01", YYYYMMDDhhmmss);
    double mjd = asTime::GetMJD(2007,11,23,13,5,1);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(GetTimeFromStringFormatYYYYMMDDhhmmssException)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("23.11.2007 13:05:01", YYYYMMDDhhmmss), asException);
    }
}

TEST(GetTimeFromStringFormathhmmException)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("13:05:01", hhmm), asException);
    }
}

TEST(GetTimeFromStringFormatnowplushours)
{
    double conversion = asTime::GetTimeFromString("+2", nowplushours);
    wxString datestr = asTime::GetStringTime(conversion);

    wxString str;
    str.Printf("UTM time +2 hours is %s\n", datestr.c_str());

    printf("%s", str.mb_str(wxConvUTF8).data());
}

TEST(GetTimeFromStringFormatnowplushoursException)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("+2:23", nowplushours), asException);
    }
}

TEST(GetTimeFromStringFormatnowplushoursExceptionDot)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("+2.23", nowplushours), asException);
    }
}

TEST(GetTimeFromStringFormatnowminushours)
{
    double conversion = asTime::GetTimeFromString("-2", nowminushours);
    wxString datestr = asTime::GetStringTime(conversion);

    wxString str;
    str.Printf("UTM time -2 hours is %s\n", datestr.c_str());

    printf("%s", str.mb_str(wxConvUTF8).data());
}

TEST(GetTimeFromStringFormatnowminushoursException)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("-2:23", nowminushours), asException);
    }
}

TEST(GetTimeFromStringFormatnowminushoursExceptionDot)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("-2.23", nowminushours), asException);
    }
}

TEST(GetTimeFromStringFormatnowminushoursExceptionSignPlus)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("+2", nowminushours), asException);
    }
}

TEST(GetTimeFromStringFormatnowminushoursExceptionSignNo)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("2", nowminushours), asException);
    }
}

TEST(GetTimeFromStringFormatautoDDMMYYYY)
{
    double conversion = asTime::GetTimeFromString("23.11.2007", guess);
    double mjd = asTime::GetMJD(2007,11,23);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(GetTimeFromStringFormatautoDDMMYYYYSlashes)
{
    double conversion = asTime::GetTimeFromString("23/11/2007", guess);
    double mjd = asTime::GetMJD(2007,11,23);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(GetTimeFromStringFormatautoDDMMYYYYException)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("23.11.07", guess), asException);
    }
}

TEST(GetTimeFromStringFormatautoYYYYMMDD)
{
    double conversion = asTime::GetTimeFromString("2007.11.23", guess);
    double mjd = asTime::GetMJD(2007,11,23);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(GetTimeFromStringFormatautoYYYYMMDDException)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("11.2007", guess), asException);
    }
}

TEST(GetTimeFromStringFormatautoDDMMYYYYhhmm)
{
    double conversion = asTime::GetTimeFromString("23.11.2007 13:05", guess);
    double mjd = asTime::GetMJD(2007,11,23,13,5);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(GetTimeFromStringFormatautoDDMMYYYYhhmmException)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("23.11.07 13:05", guess), asException);
    }
}

TEST(GetTimeFromStringFormatautoYYYYMMDDhhmm)
{
    double conversion = asTime::GetTimeFromString("2007.11.23 13:05", guess);
    double mjd = asTime::GetMJD(2007,11,23,13,5);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(GetTimeFromStringFormatautoYYYYMMDDhhmmException)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("23.11.07 13:05", guess), asException);
    }
}

TEST(GetTimeFromStringFormatautoDDMMYYYYhhmmss)
{
    double conversion = asTime::GetTimeFromString("23.11.2007 13:05:01", guess);
    double mjd = asTime::GetMJD(2007,11,23,13,5,1);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(GetTimeFromStringFormatautoDDMMYYYYhhmmssException)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("23.11.07 13:05:01", guess), asException);
    }
}

TEST(GetTimeFromStringFormatautoYYYYMMDDhhmmss)
{
    double conversion = asTime::GetTimeFromString("2007.11.23 13:05:01", guess);
    double mjd = asTime::GetMJD(2007,11,23,13,5,1);

    CHECK_CLOSE(mjd, conversion, 0.00001);
}

TEST(GetTimeFromStringFormatautoYYYYMMDDhhmmssException)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("23.11.07 13:05:01", guess), asException);
    }
}

TEST(GetTimeFromStringFormatautohhmmException)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("13:05:01", guess), asException);
    }
}

TEST(GetTimeFromStringFormatautonowplushours)
{
    double conversion = asTime::GetTimeFromString("+2", guess);
    wxString datestr = asTime::GetStringTime(conversion);

    wxString str;
    str.Printf("UTM time +2 hours is %s\n", datestr.c_str());

    printf("%s", str.mb_str(wxConvUTF8).data());
}

TEST(GetTimeFromStringFormatautonowplushoursException)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("+2:23", guess), asException);
    }
}

TEST(GetTimeFromStringFormatautonowplushoursExceptionDot)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("+2.23", guess), asException);
    }
}

TEST(GetTimeFromStringFormatautonowminushours)
{
    double conversion = asTime::GetTimeFromString("-2", guess);
    wxString datestr = asTime::GetStringTime(conversion);

    wxString str;
    str.Printf("UTM time -2 hours is %s\n", datestr.c_str());

    printf("%s", str.mb_str(wxConvUTF8).data());
}

TEST(GetTimeFromStringFormatautonowminushoursException)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("-2:23", guess), asException);
    }
}

TEST(GetTimeFromStringFormatautonowminushoursExceptionDot)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("-2.23", guess), asException);
    }
}

TEST(GetTimeFromStringFormatautonowminushoursExceptionSignNo)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("2", guess), asException);
    }
}

TEST(GetTimeFromStringFormatautonowminushoursExceptionSignPlusText)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asTime::GetTimeFromString("+2hours", guess), asException);
    }
}

TEST(AddYear1972)
{
    double mjd = asTime::GetMJD(1972,11,23,13,5);
    mjd = asTime::AddYear(mjd);
    double mjdafter = asTime::GetMJD(1973,11,23,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}

TEST(AddYear1972Leap)
{
    double mjd = asTime::GetMJD(1972,2,23,13,5);
    mjd = asTime::AddYear(mjd);
    double mjdafter = asTime::GetMJD(1973,2,23,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}

TEST(AddYear1900)
{
    double mjd = asTime::GetMJD(1900,2,23,13,5);
    mjd = asTime::AddYear(mjd);
    double mjdafter = asTime::GetMJD(1901,2,23,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}

TEST(AddYear2000)
{
    double mjd = asTime::GetMJD(2000,11,23,13,5);
    mjd = asTime::AddYear(mjd);
    double mjdafter = asTime::GetMJD(2001,11,23,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}

TEST(AddYear2000Leap)
{
    double mjd = asTime::GetMJD(2000,2,23,13,5);
    mjd = asTime::AddYear(mjd);
    double mjdafter = asTime::GetMJD(2001,2,23,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}

TEST(SubtractYear1972Leap)
{
    double mjd = asTime::GetMJD(1972,11,23,13,5);
    mjd = asTime::SubtractYear(mjd);
    double mjdafter = asTime::GetMJD(1971,11,23,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}

TEST(SubtractYear1972)
{
    double mjd = asTime::GetMJD(1972,2,23,13,5);
    mjd = asTime::SubtractYear(mjd);
    double mjdafter = asTime::GetMJD(1971,2,23,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}

TEST(SubtractYear1900)
{
    double mjd = asTime::GetMJD(1900,11,23,13,5);
    mjd = asTime::SubtractYear(mjd);
    double mjdafter = asTime::GetMJD(1899,11,23,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}

TEST(SubtractYear2000Leap)
{
    double mjd = asTime::GetMJD(2000,11,23,13,5);
    mjd = asTime::SubtractYear(mjd);
    double mjdafter = asTime::GetMJD(1999,11,23,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}

TEST(SubtractYear2000)
{
    double mjd = asTime::GetMJD(2000,2,23,13,5);
    mjd = asTime::SubtractYear(mjd);
    double mjdafter = asTime::GetMJD(1999,2,23,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}

TEST(SubtractYear2000Feb28)
{
    double mjd = asTime::GetMJD(2000,2,28,13,5);
    mjd = asTime::SubtractYear(mjd);
    double mjdafter = asTime::GetMJD(1999,2,28,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}

TEST(SubtractYear2000Feb29)
{
    double mjd = asTime::GetMJD(2000,2,29,13,5);
    mjd = asTime::SubtractYear(mjd);
    double mjdafter = asTime::GetMJD(1999,2,28,13,5);

    CHECK_CLOSE(mjdafter, mjd, 0.00001);
}
}
