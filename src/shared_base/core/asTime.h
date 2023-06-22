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

#ifndef AS_TIME_H
#define AS_TIME_H

#include <ctime>

#include "asIncludes.h"

class asTime : public wxObject {
  public:
    asTime() = default;

    ~asTime() override = default;

    static double NowMJD(int timezone = 0);

    static Time NowTimeStruct(int timezone);

    static wxDateTime NowWxDateTime(int timezone);

    static wxString GetStringTime(double mjd, const wxString& format);

    static wxString GetStringTime(const Time& date, const wxString& format);

    static wxString GetStringTime(double mjd, TimeFormat format = DD_MM_YYYY_hh_mm);

    static wxString GetStringTime(const Time& date, TimeFormat format = DD_MM_YYYY_hh_mm);

    static double GetTimeFromString(const wxString& datestr, TimeFormat format = guess);

    static bool IsLeapYear(int year);

    /**
     * Get the date as a Modified Julian Date (MJD).
     *
     * @param year The year.
     * @param month The month.
     * @param day The day.
     * @param hour The hour.
     * @param minute The minute.
     * @param second The second.
     * @param method The method to use.
     * @return The date as a MJD.
     * @author David Burki
     * @note From http://www.xmission.com/~tknarr/code/Date.html
     */
    static double GetMJD(int year, int month = 1, int day = 1, int hour = 0, int minute = 0, int second = 0,
                         int method = asUSE_NORMAL_METHOD);

    /**
     * Transform a time struct to a MJD date
     *
     * @param date The time struct.
     * @param method The method to use.
     * @return The date as a MJD.
     */
    static double GetMJD(const Time& date, int method = asUSE_NORMAL_METHOD);

    /**
     * Transform a wxDateTime to a MJD date
     *
     * @param date The wxDateTime.
     * @param method The method to use.
     * @return The date as a MJD.
     */
    static double GetMJD(wxDateTime& date, int method = asUSE_NORMAL_METHOD);

    static wxDateTime GetWxDateTime(double mjd, int method = asUSE_NORMAL_METHOD);

    /**
     * Get the date as a time struct.
     *
     * @param mjd The date as a MJD.
     * @param method The method to use.
     * @return The date as a time struct.
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
    static void TimeStructInit(Time& date);

    static Time TimeTmToTimeStruct(const struct tm& date);

    static double TimeTmToMJD(const struct tm& date);
};

#endif
