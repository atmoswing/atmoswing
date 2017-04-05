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

#ifndef ASTIME_H
#define ASTIME_H

#include <time.h>

#include <asIncludes.h>

class asTime
        : public wxObject
{
public:
    asTime();

    virtual ~asTime();

    static double NowMJD(int timezone = 0);

    static Time NowTimeStruct(int timezone);

    static wxDateTime NowWxDateTime(int timezone);

    static wxString GetStringTime(double mjd, const wxString &format);

    static wxString GetStringTime(const Time &date, const wxString &format);

    static wxString GetStringTime(double mjd, TimeFormat format = full);

    static wxString GetStringTime(const Time &date, TimeFormat format = full);

    static double GetTimeFromString(const wxString &datestr, TimeFormat format = guess);

    static bool IsLeapYear(int year);

    /** Transform a date to a MJD date
     * \author David Burki
     * \link http://www.xmission.com/~tknarr/code/Date.html
     */
    static double GetMJD(int year, int month = 1, int day = 1, int hour = 0, int minute = 0, int second = 0,
                         int method = asUSE_NORMAL_METHOD);

    /** Transform a date to a MJD date
     * \author David Burki
     * \link http://www.xmission.com/~tknarr/code/Date.html
     */
    static double GetMJD(const Time &date, int method = asUSE_NORMAL_METHOD);

    /** Transform a wxDateTime to a MJD date
     * \author David Burki
     * \link http://www.xmission.com/~tknarr/code/Date.html
     */
    static double GetMJD(wxDateTime &date, int method = asUSE_NORMAL_METHOD);

    static wxDateTime GetWxDateTime(double mjd, int method = asUSE_NORMAL_METHOD);

    /** Transform a MJD into a time struct
     * \author David Burki
     * \link http://www.xmission.com/~tknarr/code/Date.html
     */
    static Time GetTimeStruct(double mjd, int method = asUSE_NORMAL_METHOD);

    static Time GetTimeStruct(int year, int month, int day, int hour = 0, int minute = 0, int second = 0);

    static int GetYear(double mjd, int method = asUSE_NORMAL_METHOD);

    static int GetMonth(double mjd, int method = asUSE_NORMAL_METHOD);

    static int GetDay(double mjd, int method = asUSE_NORMAL_METHOD);

    static int GetHour(double mjd, int method = asUSE_NORMAL_METHOD);

    static int GetMinute(double mjd, int method = asUSE_NORMAL_METHOD);

    static int GetSecond(double mjd, int method = asUSE_NORMAL_METHOD);

    static double AddYear(double mjd);

    static double SubtractYear(double mjd);

protected:
    static void TimeStructInit(Time &date);

    static Time TimeTmToTimeStruct(const struct tm &date);

    static double TimeTmToMJD(const struct tm &date);

    static Time GetSeasonStart(Season season);

    static Time GetSeasonEnd(Season season, int year);

    static Season GetSeason(int month);
    
};

#endif
