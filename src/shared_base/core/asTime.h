/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
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
     * \param MJD The date as a MJD
     * \param format The string format
     * \return The date as a string
     */
    static wxString GetStringTime(double MJD, const wxString &format );

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
    static wxString GetStringTime(double MJD, TimeFormat format = full);

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
     * \param Year The year
     * \return True if a leap year, false elsewhere
     */
    static bool IsLeapYear(int Year);

    /** Transform a date to a MJD date
     * \param Year The year
     * \param Month the month
     * \param Day The day
     * \param Hour The hour
     * \param Minute The minute
     * \param Second The second
     * \param Method The prefered method
     * \return The corresponding date in Modified Julian Day number
     * \author David Burki
     * \link http://www.xmission.com/~tknarr/code/Date.html
     */
    static double GetMJD(int Year, int Month = 1, int Day = 1, int Hour = 0, int Minute = 0, int Second = 0, int Method = asUSE_NORMAL_METHOD );

    /** Transform a date to a MJD date
     * \param Date The date
     * \param Method The prefered method
     * \return The corresponding date in Modified Julian Day number
     * \author David Burki
     * \link http://www.xmission.com/~tknarr/code/Date.html
     */
    static double GetMJD(const TimeStruct &Date, int Method = asUSE_NORMAL_METHOD );

    /** Transform a wxDateTime to a MJD date
     * \param date The date as wxDateTime
     * \param Method The prefered method
     * \return The corresponding date in Modified Julian Day number
     * \author David Burki
     * \link http://www.xmission.com/~tknarr/code/Date.html
     */
    static double GetMJD(wxDateTime &date, int Method = asUSE_NORMAL_METHOD );

    /** Transform a MJD date to a wxDateTime
     * \param Mjd The date as mjd
     * \param Method The prefered method
     * \return The corresponding date as wxDateTime
     */
    static wxDateTime GetWxDateTime(double Mjd, int Method = asUSE_NORMAL_METHOD );

    /** Transform a MJD into a time struct
     * \param Mjd The modified julian day number
     * \param Method The prefered method
     * \return The corresponding date in a struct
     * \author David Burki
     * \link http://www.xmission.com/~tknarr/code/Date.html
     */
    static TimeStruct GetTimeStruct( double Mjd, int Method = asUSE_NORMAL_METHOD );

    /** Create a TimeStruct
     * \param Year The year
     * \param Month the month
     * \param Day The day
     * \param Hour The hour
     * \param Minute The minute
     * \param Second The second
     * \return The corresponding date in a TimeStruct
     */
    static TimeStruct GetTimeStruct(int Year, int Month, int Day, int Hour = 0, int Minute = 0, int Second = 0);

    /** Get the year of a MJD number
     * \param Mjd The modified julian day number
     * \param Method The prefered method
     * \return The year
     */
    static int GetYear( double Mjd, int Method = asUSE_NORMAL_METHOD  );

    /** Get the month of a MJD number
     * \param Mjd The modified julian day number
     * \param Method The prefered method
     * \return The month
     */
    static int GetMonth( double Mjd, int Method = asUSE_NORMAL_METHOD  );

    /** Get the day of a MJD number
     * \param Mjd The modified julian day number
     * \param Method The prefered method
     * \return The day
     */
    static int GetDay( double Mjd, int Method = asUSE_NORMAL_METHOD  );

    /** Get the hour of a MJD number
     * \param Mjd The modified julian day number
     * \param Method The prefered method
     * \return The hour
     */
    static int GetHour( double Mjd, int Method = asUSE_NORMAL_METHOD  );

    /** Get the minute of a MJD number
     * \param Mjd The modified julian day number
     * \param Method The prefered method
     * \return The minute
     */
    static int GetMinute( double Mjd, int Method = asUSE_NORMAL_METHOD  );

    /** Get the second of a MJD number
     * \param Mjd The modified julian day number
     * \param Method The prefered method
     * \return The second
     */
    static int GetSecond( double Mjd, int Method = asUSE_NORMAL_METHOD  );

    /** Add a year to the MJD number
     * \param Mjd The modified julian day number
     * \return The modified julian day number with an additional year
     */
    static double AddYear( double Mjd );

    /** Subtract a year to the MJD number
     * \param Mjd The modified julian day number
     * \return The modified julian day number minus a year
     */
    static double SubtractYear( double Mjd );

protected:

    /** Initialize the time structure to 0
     * \param Date The time structure
     */
    static void TimeStructInit(TimeStruct &Date);

    /** Get the time structure from a struct tm
     * \param Date The time in a struct tm format
     * \return the value of TimeStruct
     */
    static TimeStruct TimeTmToTimeStruct(const struct tm &Date);

    /** Get the time structure from a MJD
     * \param Date The time in a struct tm format
     * \return the value of MJD
     */
    static double TimeTmToMJD(const struct tm &Date);

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
