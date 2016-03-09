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

#include "asTime.h"

asTime::asTime()
{
    //ctor
}

asTime::~asTime()
{
    //dtor
}

void asTime::TimeStructInit(TimeStruct &date)
{
    date.year = 0;
    date.month = 0;
    date.day = 0;
    date.hour = 0;
    date.min = 0;
    date.sec = 0;
}

TimeStruct asTime::TimeTmToTimeStruct(const struct tm &date)
{
    TimeStruct timeSt;
    TimeStructInit(timeSt);

    timeSt.year = date.tm_year + 1900;
    timeSt.month = date.tm_mon + 1;
    timeSt.day = date.tm_mday;
    timeSt.hour = date.tm_hour;
    timeSt.min = date.tm_min;
    timeSt.sec = date.tm_sec;

    return timeSt;
}

double asTime::TimeTmToMJD(const struct tm &date)
{
    return GetMJD(date.tm_year + 1900, date.tm_mon + 1, date.tm_mday, date.tm_hour, date.tm_min, date.tm_sec);
}

double asTime::NowMJD(int timezone)
{
    struct tm todaytm;
    time_t todayepoch;

    time(&todayepoch);

    switch (timezone)
    {
        case asUTM:
            todaytm = *gmtime(&todayepoch);
            break;
        case asLOCAL:
            todaytm = *localtime(&todayepoch);
            break;
        default:
            asThrowException(_("The timezone is not correctly set"));
    }

    return TimeTmToMJD(todaytm);
}

TimeStruct asTime::NowTimeStruct(int timezone)
{
    struct tm todaytm;
    time_t todayepoch;

    time(&todayepoch);

    switch (timezone)
    {
        case asUTM:
            todaytm = *gmtime(&todayepoch);
            break;
        case asLOCAL:
            todaytm = *localtime(&todayepoch);
            break;
        default:
            asThrowException(_("The timezone is not correctly set"));
    }

    return TimeTmToTimeStruct(todaytm);
}

wxDateTime asTime::NowWxDateTime(int timezone)
{
    TimeStruct now = NowTimeStruct(timezone);

    wxDateTime::Month month;

    switch(now.month)
    {
    case 1:
        month = wxDateTime::Jan;
        break;
    case 2:
        month = wxDateTime::Feb;
        break;
    case 3:
        month = wxDateTime::Mar;
        break;
    case 4:
        month = wxDateTime::Apr;
        break;
    case 5:
        month = wxDateTime::May;
        break;
    case 6:
        month = wxDateTime::Jun;
        break;
    case 7:
        month = wxDateTime::Jul;
        break;
    case 8:
        month = wxDateTime::Aug;
        break;
    case 9:
        month = wxDateTime::Sep;
        break;
    case 10:
        month = wxDateTime::Oct;
        break;
    case 11:
        month = wxDateTime::Nov;
        break;
    case 12:
        month = wxDateTime::Dec;
        break;
    default:
        month = wxDateTime::Inv_Month;
    }

    // Create datetime object.
    wxDateTime nowWx(now.day, month, now.year, now.hour, now.min, now.sec, 0);

    return nowWx;
}

wxString asTime::GetStringTime(double mjd, const wxString &format)
{
    TimeStruct date = GetTimeStruct(mjd);
    wxString datestr = GetStringTime(date, format);

    return datestr;
}

wxString asTime::GetStringTime(const TimeStruct &date, const wxString &format)
{
    wxString datestr = format;

    wxString year = wxString::Format("%d", date.year);
    wxString month = wxString::Format("%d", date.month);
    if (month.Length()<2) month = "0" + month;
    wxString day = wxString::Format("%d", date.day);
    if (day.Length()<2) day = "0" + day;
    wxString hour = wxString::Format("%d", date.hour);
    if (hour.Length()<2) hour = "0" + hour;
    wxString minute = wxString::Format("%d", date.min);
    if (minute.Length()<2) minute = "0" + minute;

    datestr.Replace("YYYY", year);
    datestr.Replace("YY", year.SubString(2,2));
    datestr.Replace("MM", month, false);
    datestr.Replace("DD", day);
    datestr.Replace("hh", hour);
    datestr.Replace("HH", hour);
    datestr.Replace("mm", minute);
    datestr.Replace("HH", minute);

    return datestr;
}

wxString asTime::GetStringTime(double mjd, TimeFormat format)
{
    TimeStruct date = GetTimeStruct(mjd);
    wxString datestr = GetStringTime(date, format);

    return datestr;
}

wxString asTime::GetStringTime(const TimeStruct &date, TimeFormat format)
{
    wxString datestr = wxEmptyString;

    switch (format)
    {
        case (classic):
        case (DDMMYYYY):
            datestr = wxString::Format("%2.2d.%2.2d.%4.4d",date.day, date.month, date.year);
            break;
        case (YYYYMMDD):
            datestr = wxString::Format("%4.4d/%2.2d/%2.2d",date.year, date.month, date.day);
            break;
        case (full):
        case (DDMMYYYYhhmm):
            datestr = wxString::Format("%2.2d.%2.2d.%4.4d %2.2d:%2.2d",date.day, date.month, date.year, date.hour, date.min);
            break;
        case (YYYYMMDDhhmm):
            datestr = wxString::Format("%4.4d/%2.2d/%2.2d %2.2d:%2.2d",date.year, date.month, date.day, date.hour, date.min);
            break;
        case (DDMMYYYYhhmmss):
            datestr = wxString::Format("%2.2d.%2.2d.%4.4d %2.2d:%2.2d:%2.2d",date.day, date.month, date.year, date.hour, date.min, date.sec);
            break;
        case (YYYYMMDDhhmmss):
            datestr = wxString::Format("%4.4d/%2.2d/%2.2d %2.2d:%2.2d:%2.2d",date.year, date.month, date.day, date.hour, date.min, date.sec);
            break;
        case (timeonly):
        case (hhmm):
            datestr = wxString::Format("%2.2d:%2.2d", date.hour, date.min);
            break;
        case (concentrate):
            datestr = wxString::Format("%4.4d%2.2d%2.2d-%2.2d%2.2d",date.year, date.month, date.day, date.hour, date.min);
            break;
        default:
            asThrowException(_("The date format is not correctly set"));
    }

    return datestr;
}

double asTime::GetTimeFromString(const wxString &datestr, TimeFormat format)
{
    double date;
    wxString errormsglength = _("The length of the input date (%2.2d) is not as expected (%2.2d or %2.2d)");
    wxString errormsgconversion = _("The date (%s) conversion failed. Please check the format");
    long day = 0;
    long month = 0;
    long year = 0;
    long hour = 0;
    long min = 0;
    long sec = 0;

    switch (format)
    {
        case (classic):
        case (DDMMYYYY):

            if (datestr.Len() == 10)
            {
                if (!datestr.Mid(0,2).ToLong(&day)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(3,2).ToLong(&month)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(6,4).ToLong(&year)) asThrowException(wxString::Format(errormsgconversion,datestr));
                return GetMJD(year,month,day);
            }
            else if (datestr.Len() == 8)
            {
                if (!datestr.Mid(0,2).ToLong(&day)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(2,2).ToLong(&month)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(4,4).ToLong(&year)) asThrowException(wxString::Format(errormsgconversion,datestr));
                return GetMJD(year,month,day);
            }
            else
            {
                asThrowException(wxString::Format(errormsglength,(int)datestr.Len(),8,10));
            }

            break;

        case (YYYYMMDD):

            if (datestr.Len() == 10)
            {
                if (!datestr.Mid(0,4).ToLong(&year)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(5,2).ToLong(&month)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(8,2).ToLong(&day)) asThrowException(wxString::Format(errormsgconversion,datestr));
                return GetMJD(year,month,day);
            }
            else if (datestr.Len() == 8)
            {
                if (!datestr.Mid(0,4).ToLong(&year)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(4,2).ToLong(&month)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(6,2).ToLong(&day)) asThrowException(wxString::Format(errormsgconversion,datestr));
                return GetMJD(year,month,day);
            }
            else
            {
                asThrowException(wxString::Format(errormsglength,(int)datestr.Len(),8,10));
            }
            break;

        case (YYYYMMDDhh):

            if (datestr.Len() == 13)
            {
                if (!datestr.Mid(0,4).ToLong(&year)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(5,2).ToLong(&month)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(8,2).ToLong(&day)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(11,2).ToLong(&hour)) asThrowException(wxString::Format(errormsgconversion,datestr));
                return GetMJD(year,month,day,hour);
            }
            else if (datestr.Len() == 10)
            {
                if (!datestr.Mid(0,4).ToLong(&year)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(4,2).ToLong(&month)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(6,2).ToLong(&day)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(8,2).ToLong(&hour)) asThrowException(wxString::Format(errormsgconversion,datestr));
                return GetMJD(year,month,day,hour);
            }
            else
            {
                asThrowException(wxString::Format(errormsglength,(int)datestr.Len(),10,13));
            }
            break;

        case (full):
        case (DDMMYYYYhhmm):

            if (datestr.Len() == 16)
            {
                if (!datestr.Mid(0,2).ToLong(&day)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(3,2).ToLong(&month)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(6,4).ToLong(&year)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(11,2).ToLong(&hour)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(14,2).ToLong(&min)) asThrowException(wxString::Format(errormsgconversion,datestr));
                return GetMJD(year,month,day,hour,min);
            }
            else if (datestr.Len() == 12)
            {
                if (!datestr.Mid(0,2).ToLong(&day)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(2,2).ToLong(&month)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(4,4).ToLong(&year)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(8,2).ToLong(&hour)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(10,2).ToLong(&min)) asThrowException(wxString::Format(errormsgconversion,datestr));
                return GetMJD(year,month,day,hour,min);
            }
            else
            {
                asThrowException(wxString::Format(errormsglength,(int)datestr.Len(),12,16));
            }
            break;

        case (YYYYMMDDhhmm):

            if (datestr.Len() == 16)
            {
                if (!datestr.Mid(0,4).ToLong(&year)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(5,2).ToLong(&month)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(8,2).ToLong(&day)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(11,2).ToLong(&hour)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(14,2).ToLong(&min)) asThrowException(wxString::Format(errormsgconversion,datestr));
                return GetMJD(year,month,day,hour,min);
            }
            else if (datestr.Len() == 12)
            {
                if (!datestr.Mid(0,4).ToLong(&year)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(4,2).ToLong(&month)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(6,2).ToLong(&day)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(8,2).ToLong(&hour)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(10,2).ToLong(&min)) asThrowException(wxString::Format(errormsgconversion,datestr));
                return GetMJD(year,month,day,hour,min);
            }
            else
            {
                asThrowException(wxString::Format(errormsglength,(int)datestr.Len(),12,16));
            }
            break;

        case (DDMMYYYYhhmmss):

            if (datestr.Len() == 19)
            {
                if (!datestr.Mid(0,2).ToLong(&day)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(3,2).ToLong(&month)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(6,4).ToLong(&year)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(11,2).ToLong(&hour)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(14,2).ToLong(&min)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(17,2).ToLong(&sec)) asThrowException(wxString::Format(errormsgconversion,datestr));
                return GetMJD(year,month,day,hour,min,sec);
            }
            else if (datestr.Len() == 14)
            {
                if (!datestr.Mid(0,2).ToLong(&day)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(2,2).ToLong(&month)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(4,4).ToLong(&year)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(8,2).ToLong(&hour)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(10,2).ToLong(&min)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(12,2).ToLong(&sec)) asThrowException(wxString::Format(errormsgconversion,datestr));
                return GetMJD(year,month,day,hour,min,sec);
            }
            else
            {
                asThrowException(wxString::Format(errormsglength,(int)datestr.Len(),14,19));
            }
            break;

        case (YYYYMMDDhhmmss):

            if (datestr.Len() == 19)
            {
                if (!datestr.Mid(0,4).ToLong(&year)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(5,2).ToLong(&month)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(8,2).ToLong(&day)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(11,2).ToLong(&hour)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(14,2).ToLong(&min)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(17,2).ToLong(&sec)) asThrowException(wxString::Format(errormsgconversion,datestr));
                return GetMJD(year,month,day,hour,min,sec);
            }
            else if (datestr.Len() == 14)
            {
                if (!datestr.Mid(0,4).ToLong(&year)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(4,2).ToLong(&month)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(6,2).ToLong(&day)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(8,2).ToLong(&hour)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(10,2).ToLong(&min)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(12,2).ToLong(&sec)) asThrowException(wxString::Format(errormsgconversion,datestr));
                return GetMJD(year,month,day,hour,min,sec);
            }
            else
            {
                asThrowException(wxString::Format(errormsglength,(int)datestr.Len(),14,19));
            }
            break;

        case (timeonly):
        case (hhmm):

            if (datestr.Len() == 5)
            {
                if (!datestr.Mid(0,2).ToLong(&hour)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(3,2).ToLong(&min)) asThrowException(wxString::Format(errormsgconversion,datestr));
                return GetMJD(0,0,0,hour,min);
            }
            else if (datestr.Len() == 4)
            {
                if (!datestr.Mid(0,2).ToLong(&hour)) asThrowException(wxString::Format(errormsgconversion,datestr));
                if (!datestr.Mid(2,2).ToLong(&min)) asThrowException(wxString::Format(errormsgconversion,datestr));
                return GetMJD(0,0,0,hour,min);
            }
            else
            {
                asThrowException(wxString::Format(errormsglength,(int)datestr.Len(),4,5));
            }
            break;

        case (nowplushours):

            if (datestr.Mid(0,1) != "+") asThrowException(_("The date format is not correctly set"));
            if (!datestr.Mid(1).ToLong(&hour)) asThrowException(wxString::Format(errormsgconversion,datestr));
            date = NowMJD(asUTM);
            date += (double) hour/24;
            return date;
            break;

        case (nowminushours):

            if (datestr.Mid(0,1) != "-") asThrowException(_("The date format is not correctly set"));
            if (!datestr.Mid(1).ToLong(&hour)) asThrowException(wxString::Format(errormsgconversion,datestr));
            date = NowMJD(asUTM);
            date -= (double) hour/24;
            return date;
            break;

        case (guess):

            if (datestr.Mid(0,1) == "+")
            {
                if (!datestr.Mid(1).ToLong(&hour)) asThrowException(wxString::Format(errormsgconversion,datestr));
                date = NowMJD(asUTM);
                date += (double) hour/24;
                return date;
            }
            else if (datestr.Mid(0,1) == "-")
            {
                if (!datestr.Mid(1).ToLong(&hour)) asThrowException(wxString::Format(errormsgconversion,datestr));
                date = NowMJD(asUTM);
                date -= (double) hour/24;
                return date;
            }
            else
            {
                if (datestr.Len() == 10)
                {
                    if (datestr.Mid(0,4).ToLong(&year))
                    {
                        if (!datestr.Mid(0,4).ToLong(&year)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        if (!datestr.Mid(5,2).ToLong(&month)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        if (!datestr.Mid(8,2).ToLong(&day)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        return GetMJD(year,month,day);
                    }
                    else if (datestr.Mid(0,2).ToLong(&day))
                    {
                        if (!datestr.Mid(0,2).ToLong(&day)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        if (!datestr.Mid(3,2).ToLong(&month)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        if (!datestr.Mid(6,4).ToLong(&year)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        return GetMJD(year,month,day);
                    }
                    else
                    {
                         asThrowException(wxString::Format(errormsgconversion,datestr));
                    }

                }
                else if (datestr.Len() == 16)
                {
                    if (datestr.Mid(0,4).ToLong(&year))
                    {
                        if (!datestr.Mid(0,4).ToLong(&year)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        if (!datestr.Mid(5,2).ToLong(&month)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        if (!datestr.Mid(8,2).ToLong(&day)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        if (!datestr.Mid(11,2).ToLong(&hour)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        if (!datestr.Mid(14,2).ToLong(&min)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        return GetMJD(year,month,day,hour,min);
                    }
                    else if (datestr.Mid(0,2).ToLong(&day))
                    {
                        if (!datestr.Mid(0,2).ToLong(&day)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        if (!datestr.Mid(3,2).ToLong(&month)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        if (!datestr.Mid(6,4).ToLong(&year)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        if (!datestr.Mid(11,2).ToLong(&hour)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        if (!datestr.Mid(14,2).ToLong(&min)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        return GetMJD(year,month,day,hour,min);
                    }
                    else
                    {
                         asThrowException(wxString::Format(errormsgconversion,datestr));
                    }
                }
                else if (datestr.Len() == 19)
                {
                    if (datestr.Mid(0,4).ToLong(&year))
                    {
                        if (!datestr.Mid(0,4).ToLong(&year)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        if (!datestr.Mid(5,2).ToLong(&month)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        if (!datestr.Mid(8,2).ToLong(&day)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        if (!datestr.Mid(11,2).ToLong(&hour)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        if (!datestr.Mid(14,2).ToLong(&min)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        if (!datestr.Mid(17,2).ToLong(&sec)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        return GetMJD(year,month,day,hour,min,sec);
                    }
                    else if (datestr.Mid(0,2).ToLong(&day))
                    {
                        if (!datestr.Mid(0,2).ToLong(&day)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        if (!datestr.Mid(3,2).ToLong(&month)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        if (!datestr.Mid(6,4).ToLong(&year)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        if (!datestr.Mid(11,2).ToLong(&hour)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        if (!datestr.Mid(14,2).ToLong(&min)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        if (!datestr.Mid(17,2).ToLong(&sec)) asThrowException(wxString::Format(errormsgconversion,datestr));
                        return GetMJD(year,month,day,hour,min,sec);
                    }
                    else
                    {
                         asThrowException(wxString::Format(errormsgconversion,datestr));
                    }
                }
                else if (datestr.Len() == 5)
                {
                    if (!datestr.Mid(0,2).ToLong(&hour)) asThrowException(wxString::Format(errormsgconversion,datestr));
                    if (!datestr.Mid(3,2).ToLong(&min)) asThrowException(wxString::Format(errormsgconversion,datestr));
                    return GetMJD(0,0,0,hour,min);
                }
                else
                {
                    asThrowException(wxString::Format(errormsgconversion,datestr));
                }
            }

            break;
        default:
            asThrowException(_("The date format is not correctly set"));
    }

    return NaNFloat;
}

bool asTime::IsLeapYear(int year)
{
    // Leap year if divisable by 4 and not by 100 or divisable by 400
    if((year % 4 == 0 && year % 100 != 0) || year % 400 == 0)
    {
        return true; // leap
    } else {
        return false; // no leap
    }
}

double asTime::GetMJD(int year, int month, int day, int hour, int minute, int second, int method)
{
    wxASSERT(year>0);
    wxASSERT(month>0);
    wxASSERT(day>0);
    wxASSERT(hour>=0);
    wxASSERT(minute>=0);
    wxASSERT(second>=0);

    double mjd = 0;;

    switch(method)
    {
        case (asUSE_NORMAL_METHOD):
        {
            if ( year < 0 )
                year++;
            float year_corr = ( year > 0 ? 0.0 : 0.75 );
            if ( month <= 2 )
            {
                year--;
                month += 12;
            }
            int b = 0;
            if ( year * 10000.0 + month * 100.0 + day >= 15821015.0 )
            {
                int a = year / 100;
                b = 2 - a + a / 4;
            }
            mjd = (long) ( 365.25 * year - year_corr ) + (long) ( 30.6001 * ( month + 1 ) ) + day + 1720995L + b;

            break;
        }

        case (asUSE_ALTERNATE_METHOD):
        {
            // Adjust BC years
            if ( year < 0 )
                year++;

            mjd = day - 32075L +
                1461L * ( year + 4800L + ( month - 14L ) / 12L ) / 4L +
                367L * ( month - 2L - ( month - 14L ) / 12L * 12L ) / 12L -
                3L * ( ( year + 4900L + ( month - 14L ) / 12L ) / 100L ) / 4L;

            break;
        }
    }

    // The hour part
    mjd += (double) hour/24 + (double) minute/1440 + (double) second/86400;

    // Set to Modified Julian Day
    mjd -= 2400001; // And not 2400000.5 (don't know why)

    return mjd;
}

double asTime::GetMJD(const TimeStruct &date, int method)
{
    return GetMJD(date.year, date.month, date.day, date.hour, date.min, date.sec, method);
}

double asTime::GetMJD(wxDateTime &date, int method)
{
    int year = date.GetYear();
    int month = date.GetMonth()+1;
    int day = date.GetDay();
    int hour = date.GetHour();
    int min = date.GetMinute();
    int sec = date.GetSecond();

    return GetMJD(year, month, day, hour, min, sec, method);
}

wxDateTime asTime::GetWxDateTime(double mjd, int method)
{
    TimeStruct datestruct = GetTimeStruct(mjd, method);

    wxDateTime::Month month = (wxDateTime::Month)(datestruct.month-1);
    wxDateTime datewx(datestruct.day, month, datestruct.year, datestruct.hour, datestruct.min, datestruct.sec );

    return datewx;
}

TimeStruct asTime::GetTimeStruct(int year, int month, int day, int hour, int minute, int second)
{
    wxASSERT(year>=0);
    wxASSERT(month>=0);
    wxASSERT(day>=0);
    wxASSERT(hour>=0);
    wxASSERT(minute>=0);
    wxASSERT(second>=0);

    TimeStruct time;

    time.year = year;
    time.month = month;
    time.day = day;
    time.hour = hour;
    time.min = minute;
    time.sec = second;

    return time;
}

TimeStruct asTime::GetTimeStruct(double mjd, int method)
{
    wxASSERT(mjd>0);

    // To Julian day
    mjd += 2400001; // And not 2400000.5 (don't know why)

    TimeStruct date;
    TimeStructInit(date);

    // Remaining seconds
    double rest = mjd-floor(mjd);
    int sec = asTools::Round(rest*86400);
    date.hour = floor((float)(sec/3600));
    sec -= date.hour*3600;
    date.min = floor((float)(sec/60));
    sec -= date.min*60;
    date.sec = sec;

    switch(method)
    {
        case (asUSE_NORMAL_METHOD):

            long a, b, c, d, e, z;

            z = mjd;
            if ( z < 2299161L )
                a = z;
            else
            {
                long alpha = (long) ( ( z - 1867216.25 ) / 36524.25 );
                a = z + 1 + alpha - alpha / 4;
            }
            b = a + 1524;
            c = (long) ( ( b - 122.1 ) / 365.25 );
            d = (long) ( 365.25 * c );
            e = (long) ( ( b - d ) / 30.6001 );
            date.day = (int) b - d - (long) ( 30.6001 * e );
            date.month = (int) ( e < 13.5 ) ? e - 1 : e - 13;
            date.year = (int) ( date.month > 2.5 ) ? ( c - 4716 ) : c - 4715;
            if ( date.year <= 0 )
                date.year -= 1;

            break;

        case (asUSE_ALTERNATE_METHOD):

            long t1, t2, yr, mo;

            t1 = mjd + 68569L;
            t2 = 4L * t1 / 146097L;
            t1 = t1 - ( 146097L * t2 + 3L ) / 4L;
            yr = 4000L * ( t1 + 1L ) / 1461001L;
            t1 = t1 - 1461L * yr / 4L + 31L;
            mo = 80L * t1 / 2447L;
            date.day = (int) ( t1 - 2447L * mo / 80L );
            t1 = mo / 11L;
            date.month = (int) ( mo + 2L - 12L * t1 );
            date.year = (int) ( 100L * ( t2 - 49L ) + yr + t1 );

            // Correct for BC years
            if ( date.year <= 0 )
                date.year -= 1;

            break;
    }

    return date;
}

int asTime::GetYear( double mjd, int method )
{
    TimeStruct date = asTime::GetTimeStruct(mjd, method);
    return date.year;
}

int asTime::GetMonth( double mjd, int method )
{
    TimeStruct date = asTime::GetTimeStruct(mjd, method);
    return date.month;
}

int asTime::GetDay( double mjd, int method )
{
    TimeStruct date = asTime::GetTimeStruct(mjd, method);
    return date.day;
}

int asTime::GetHour( double mjd, int method )
{
    TimeStruct date = asTime::GetTimeStruct(mjd, method);
    return date.hour;
}

int asTime::GetMinute( double mjd, int method )
{
    TimeStruct date = asTime::GetTimeStruct(mjd, method);
    return date.min;
}

int asTime::GetSecond( double mjd, int method )
{
    TimeStruct date = asTime::GetTimeStruct(mjd, method);
    return date.sec;
}

double asTime::AddYear( double mjd )
{
    TimeStruct timest = GetTimeStruct(mjd);
    int year = timest.year;

    // Check which year to assess as leap or not
    if(timest.month>2)
    {
        year = timest.year+1;
    }

    // Check if the year is a leap year
    if (IsLeapYear(year))
    {
        mjd += 366;
    }
    else
    {
        mjd += 365;
    }

    return mjd;
}

double asTime::SubtractYear( double mjd )
{
    TimeStruct timest = GetTimeStruct(mjd);
    int year = timest.year;

    // Check which year to assess as leap or not
    if(timest.month==2)
    {
        if(timest.day<29)
        {
            year = timest.year-1;
        }
    }
    else if(timest.month<2)
    {
        year = timest.year-1;
    }

    // Check if the year is a leap year
    if (IsLeapYear(year))
    {
        mjd -= 366;
    }
    else
    {
        mjd -= 365;
    }

    return mjd;
}

TimeStruct asTime::GetSeasonStart(Season season)
{
    TimeStruct ret;
    TimeStructInit(ret);

    switch (season)
    {
        case DJF:
            ret.month = 12;
        break;
        case MAM:
            ret.month = 3;
        break;
        case JJA:
            ret.month = 6;
        break;
        case SON:
            ret.month = 9;
        break;
        default:
            asThrowException(_("Not a valid season."));
    }

    ret.day = 1;

    return ret;
}

TimeStruct asTime::GetSeasonEnd(Season season, int year)
{
    TimeStruct ret;
    TimeStructInit(ret);

    switch (season)
    {
        case DJF:
            ret.month = 2;
            if (IsLeapYear(year))
            {
                ret.day = 29;
            } else {
                ret.day = 28;
            }
        break;
        case MAM:
            ret.month = 5;
            ret.day = 31;
        break;
        case JJA:
            ret.month = 8;
            ret.day = 31;
        break;
        case SON:
            ret.month = 11;
            ret.day = 30;
        break;
        default:
            asThrowException(_("Not a valid season."));
    }

    return ret;
}

Season asTime::GetSeason(int month)
{
    Season season;

    switch (month)
    {
        case 1 :
            season = DJF;
        break;
        case 2 :
            season = DJF;
        break;
        case 3 :
            season = MAM;
        break;
        case 4 :
            season = MAM;
        break;
        case 5 :
            season = MAM;
        break;
        case 6 :
            season = JJA;
        break;
        case 7 :
            season = JJA;
        break;
        case 8 :
            season = JJA;
        break;
        case 9 :
            season = SON;
        break;
        case 10 :
            season = SON;
        break;
        case 11 :
            season = SON;
        break;
        case 12 :
            season = DJF;
        break;
        default:
            asThrowException(_("Not a valid month."));
    }

    return season;
}
