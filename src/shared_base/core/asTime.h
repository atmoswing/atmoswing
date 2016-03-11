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

class asTime: public wxObject
{
public:

    /** Default constructor */
    asTime();

    /** Default destructor */
    virtual ~asTime();

    /** Get the current time as a MJD
     * \param timezone The timezone
     * \return The current time as a MJD number
     */
    static double NowMJD(int timezone = 0);

    /** Get the current time as a TimeStruct
     * \param timezone The timezone
     * \return The current time as a TimeStruct
     */
    static TimeStruct NowTimeStruct(int timezone);

    /** Get the current time as a wxDateTime
     * \param timezone The timezone
     * \return The current time as a wxDateTime
     */
    static wxDateTime NowWxDateTime(int timezone);

    /** Get a string from a date
     * \param mjd The date as a MJD
     * \param format The string format
     * \return The date as a string
     */
    static wxString GetStringTime(double mjd, const wxString &format );

    /** Get a string from a date
     * \param MJD The The date as a TimeStruct
     * \param format The string format
     * \return The date as a string
     */
    static wxString GetStringTime(const TimeStruct &date, const wxString &format );

    /** Get a string from a date
     * \param MJD The date as a MJD
     * \param format The format according to the enum Timeformat
     * \return The date as a string
     */
    static wxString GetStringTime(double mjd, TimeFormat format = full);

    /** Get a string from a date
     * \param date The date as a TimeStruct
     * \param format The format according to the enum Timeformat
     * \return The date as a string
     */
    static wxString GetStringTime(const TimeStruct &date, TimeFormat format = full);

    /** Get the time from a string by parsing
     * \param datestr The date as a string
     * \param format The string format
     * \return The date as a MJD
     */
    static double GetTimeFromString(const wxString &datestr, TimeFormat format = guess);

    /** Test if the year is a leap year or not
     * \param year The year
     * \return True if a leap year, false elsewhere
     */
    static bool IsLeapYear(int year);

    /** Transform a date to a MJD date
     * \param year The year
     * \param month the month
     * \param day The day
     * \param hour The hour
     * \param minute The minute
     * \param second The second
     * \param method The prefered method
     * \return The corresponding date in Modified Julian Day number
     * \author David Burki
     * \link http://www.xmission.com/~tknarr/code/Date.html
     */
    static double GetMJD(int year, int month = 1, int day = 1, int hour = 0, int minute = 0, int second = 0, int method = asUSE_NORMAL_METHOD );

    /** Transform a date to a MJD date
     * \param date The date
     * \param method The prefered method
     * \return The corresponding date in Modified Julian Day number
     * \author David Burki
     * \link http://www.xmission.com/~tknarr/code/Date.html
     */
    static double GetMJD(const TimeStruct &date, int method = asUSE_NORMAL_METHOD );

    /** Transform a wxDateTime to a MJD date
     * \param date The date as wxDateTime
     * \param method The prefered method
     * \return The corresponding date in Modified Julian Day number
     * \author David Burki
     * \link http://www.xmission.com/~tknarr/code/Date.html
     */
    static double GetMJD(wxDateTime &date, int method = asUSE_NORMAL_METHOD );

    /** Transform a MJD date to a wxDateTime
     * \param mjd The date as MJD
     * \param method The prefered method
     * \return The corresponding date as wxDateTime
     */
    static wxDateTime GetWxDateTime(double mjd, int method = asUSE_NORMAL_METHOD );

    /** Transform a MJD into a time struct
     * \param mjd The modified julian day number
     * \param method The prefered method
     * \return The corresponding date in a struct
     * \author David Burki
     * \link http://www.xmission.com/~tknarr/code/Date.html
     */
    static TimeStruct GetTimeStruct( double mjd, int method = asUSE_NORMAL_METHOD );

    /** Create a TimeStruct
     * \param year The year
     * \param month the month
     * \param day The day
     * \param hour The hour
     * \param minute The minute
     * \param second The second
     * \return The corresponding date in a TimeStruct
     */
    static TimeStruct GetTimeStruct(int year, int month, int day, int hour = 0, int minute = 0, int second = 0);

    /** Get the year of a MJD number
     * \param mjd The modified julian day number
     * \param method The prefered method
     * \return The year
     */
    static int GetYear( double mjd, int method = asUSE_NORMAL_METHOD  );

    /** Get the month of a MJD number
     * \param mjd The modified julian day number
     * \param method The prefered method
     * \return The month
     */
    static int GetMonth( double mjd, int method = asUSE_NORMAL_METHOD  );

    /** Get the day of a MJD number
     * \param mjd The modified julian day number
     * \param method The prefered method
     * \return The day
     */
    static int GetDay( double mjd, int method = asUSE_NORMAL_METHOD  );

    /** Get the hour of a MJD number
     * \param mjd The modified julian day number
     * \param method The prefered method
     * \return The hour
     */
    static int GetHour( double mjd, int method = asUSE_NORMAL_METHOD  );

    /** Get the minute of a MJD number
     * \param mjd The modified julian day number
     * \param method The prefered method
     * \return The minute
     */
    static int GetMinute( double mjd, int method = asUSE_NORMAL_METHOD  );

    /** Get the second of a MJD number
     * \param mjd The modified julian day number
     * \param method The prefered method
     * \return The second
     */
    static int GetSecond( double mjd, int method = asUSE_NORMAL_METHOD  );

    /** Add a year to the MJD number
     * \param mjd The modified julian day number
     * \return The modified julian day number with an additional year
     */
    static double AddYear( double mjd );

    /** Subtract a year to the MJD number
     * \param mjd The modified julian day number
     * \return The modified julian day number minus a year
     */
    static double SubtractYear( double mjd );

protected:

    /** Initialize the time structure to 0
     * \param date The time structure
     */
    static void TimeStructInit(TimeStruct &date);

    /** Get the time structure from a struct tm
     * \param date The time in a struct tm format
     * \return the value of TimeStruct
     */
    static TimeStruct TimeTmToTimeStruct(const struct tm &date);

    /** Get the time structure from a MJD
     * \param date The time in a struct tm format
     * \return the value of MJD
     */
    static double TimeTmToMJD(const struct tm &date);

    /** Get the beginning of a season
     * \param season The season of interest
     * \return the value of TimeStruct
     */
    static TimeStruct GetSeasonStart(Season season);

    /** Get the end of a season
     * \param season The season of interest
     * \param year The year of interest
     * \return the value of TimeStruct
     */
    static TimeStruct GetSeasonEnd(Season season, int year);

    /** Get the season corresponding to a month
     * \param month The month of interest
     * \return the season
     */
    static Season GetSeason(int month);

private:
};

#endif
