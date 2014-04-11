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
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 */
 
#ifndef ASTIMEARRAY_H
#define ASTIMEARRAY_H

#include <asIncludes.h>
#include <asTime.h>

class asDataPredictand;


class asTimeArray: public asTime
{
public:

    enum Mode //!< Enumaration of the period selection mode
    {
        SingleDay,      // A single day
        Simple,         // A simple full time array
        SameSeason,        // Into the same season in reference to a date
        SeasonDJF,      // The DJF season
        SeasonMAM,      // The MAM season
        SeasonJJA,      // The JJA season
        SeasonSON,      // The SON season
        DaysInterval,    // + or - an amount of days in reference to a date
        PredictandThresholds,
        Custom
    };

    /** Constructor.
     * \param start The beginning of the time array
     * \param end The end of the time array
     * \param timestephours The output timestep in hours
     * \param slctmode The selection mode
     */
    asTimeArray(double start, double end, double timestephours, Mode slctmode);

    /** Constructor.
     * \param start The beginning of the time array
     * \param end The end of the time array
     * \param timestephours The output timestep in hours
     * \param slctModeString The selection mode
     */
    asTimeArray(double start, double end, double timestephours, const wxString &slctModeString);

    /** Constructor for a single day.
     * \param date The date
     * \param slctmode The selection mode
     */
    asTimeArray(double date, Mode slctmode);

    /** Constructor.
     * \param timeArray The time array to consider
     */
    asTimeArray(VectorDouble &timeArray);

    /** Constructor.
     * \param timeArray The time array to consider
     */
    asTimeArray(Array1DDouble &timeArray);

    /** Default destructor */
    virtual ~asTimeArray();

    /** Initialize. Simpler version and seasons
     * \param start The begining of the time array
     * \param end The end of the tima array
     * \param timestephours The output timestep in hours
     */
    bool Init();

    /** Initialize. For seasons based on a date
     * \param forecastdate The forecasting date. The date is considered as is, with no rounding or correction.
     * \param exclusiondays Number of days to exclude around the day that is forcasted
     */
    bool Init(double forecastdate, double exclusiondays);

    /** Initialize. For days interval based on a date
     * \param forecastdate The forecasting date. The date is considered as is, with no rounding or correction.
     * \param intervaldays Selection whithin + or - an amount of days
     * \param exclusiondays Number of days to exclude around the day that is forcasted
     */
    bool Init(double forecastdate, double intervaldays, double exclusiondays);

    /** Initialize. Based on a predictand threshold
     * \param predictand The predictand database
     * \param serieName The name of the serie in the DB
     * \param stationId The ID of the predictand station
     * \param minThreshold The min value to select
     * \param maxThreshold The max value to select
     */
    bool Init(asDataPredictand &predictand, const wxString &serieName, int stationId, float minThreshold, float maxThreshold);

    /** Initialize. Generic version
     * \param forecastdate The forecasting date. The date is considered as is, with no rounding or correction.
     * \param intervaldays Selection whithin + or - an amount of days
     * \param exclusiondays Number of days to exclude around the day that is forcasted
     * \param predictand The predictand database
     * \param serieName The name of the serie in the DB
     * \param stationId The ID of the predictand station
     * \param minThreshold The min value to select
     * \param maxThreshold The max value to select
     */
    bool Init(double forecastdate, double intervaldays, double exclusiondays, asDataPredictand &predictand, const wxString &serieName, int stationId, float minThreshold, float maxThreshold);

    /** Access an element in the time array
     * \return The current value of an item in m_TimeArray
     */
    double operator[] (unsigned int i)
    {
        wxASSERT(m_Initialized);
        wxASSERT(i < (unsigned)GetSize());
        return m_TimeArray[i];
    }

    /** Build up the date array */
    bool BuildArraySimple();

    /** Build up the date array
     * \param forecastdate The forecasting date
     */
    bool BuildArrayDaysInterval(double forecastDate);

    /** Build up the date array
     * \param forecastdate The forecasting date
     */
    bool BuildArraySeasons(double forecastDate);

    /** Build up the date array
     * \param predictand The predictand database
     * \param serieName The name of the serie in the DB
     * \param stationId The ID of the predictand station
     * \param minThreshold The min value to select
     * \param maxThreshold The max value to select
     */
    bool BuildArrayPredictandThresholds(asDataPredictand &predictand, const wxString &serieName, int stationId, float minThreshold, float maxThreshold);


    bool HasForbiddenYears();

    bool IsYearForbidden(int year);

    VectorInt GetForbiddenYears()
    {
        return m_ForbiddenYears;
    }

    void SetForbiddenYears(const VectorInt years)
    {
        m_ForbiddenYears = years;
    }

    bool RemoveYears(const VectorInt &years);


    bool KeepOnlyYears(const VectorInt &years);

    /** Access m_Mode
     * \return The current value of m_Mode
     */
    Mode GetMode()
    {
        return m_Mode;
    }

    /** Check if it is in simple mode
     * \return True if Mode is Simple
     */
    bool IsSimpleMode()
    {
        return (m_Mode==Simple) || (m_Mode==SingleDay);
    }

    /** Access m_Start
     * \return The current value of m_Start
     */
    double GetStart()
    {
        return m_Start;
    }

    /** Get the year of the first day
     * \return The year of the first day
     */
    int GetFirstDayYear()
    {
        return GetYear(m_Start);
    }

    /** Get the hour of the first day
     * \return The hour of the first day
     */
    double GetFirstDayHour()
    {
        double fractpart, intpart;
        fractpart = modf (m_Start , &intpart);
        return fractpart*24;
    }

    /** Access m_End
     * \return The current value of m_End
     */
    double GetEnd()
    {
        return m_End;
    }

    /** Get the year of the last day
     * \return The year of the last day
     */
    int GetLastDayYear()
    {
        return GetYear(m_End);
    }

    /** Get the hour of the last day
     * \return The hour of the last day
     */
    double GetLastDayHour()
    {
        double fractpart, intpart;
        fractpart = modf (m_End , &intpart);
        return fractpart*24;
    }

    /** Access m_TimeStepDays in hours
     * \return The current value of m_TimeStepDays in hours
     */
    double GetTimeStepHours()
    {
        return m_TimeStepDays*24;
    }

    /** Access m_TimeStepDays in days
     * \return The current value of m_TimeStepDays in days
     */
    double GetTimeStepDays()
    {
        return m_TimeStepDays;
    }

    /** Access m_IntervalDays in hours
     * \return The current value of m_IntervalDays
     */
    double GetIntervalHours()
    {
        return m_IntervalDays*24;
    }

    /** Return m_IntervalDays in days
     * \return The current value of m_IntervalDays in days
     */
    double GetIntervalDays()
    {
        return m_IntervalDays;
    }

    /** Access m_ExclusionDays in hours
     * \return The current value of m_ExclusionDays
     */
    double GetExclusionHours()
    {
        return m_ExclusionDays*24;
    }

    /** Access m_ExclusionDays in days
     * \return The current value of m_ExclusionDays in days
     */
    double GetExclusionDays()
    {
        return m_ExclusionDays;
    }

    /** Access m_TimeArray
     * \return The current value of m_TimeArray
     */
    Array1DDouble& GetTimeArray()
    {
        return m_TimeArray;
    }

    /** Get the size of the array
     * \return The size of m_TimeArray
     */
    int GetSize()
    {
        return m_TimeArray.size();
    }

    /** Access the first element in the time array
     * \return The current value of the first element
     */
    double GetFirst()
    {
        wxASSERT(m_Initialized);
        return m_TimeArray(0);
    }

    /** Access the last element in the time array
     * \return The current value of the last element
     */
    double GetLast()
    {
        wxASSERT(m_Initialized);
        return m_TimeArray(m_TimeArray.rows()-1);
    }

    /** Get the pointer of the first element in the time array
     * \return The current pointer of the first element
     */
    double* GetPointerStart()
    {
        wxASSERT(m_Initialized);
        return &m_TimeArray(0);
    }

    /** Get the pointer of the first element in the time array
     * \return The current pointer of the first element
     */
    double* GetPointerEnd()
    {
        wxASSERT(m_Initialized);
        return &m_TimeArray(GetSize()-1);
    }

    /** Access the index of first element in the time array after the given date
     * \return The index of the first element after the given date
     */
    int GetIndexFirstAfter(double date);

    /** Access the index of first element in the time array before the given date
     * \return The index of the first element before the given date
     */
    int GetIndexFirstBefore(double date);


protected:

private:
    bool m_Initialized;
    Mode m_Mode; //!< Member variable "m_Mode"
    Array1DDouble m_TimeArray; //!< Member variable "m_TimeArray"
    double m_Start; //!< Member variable "m_Start"
    double m_End; //!< Member variable "m_End"
    double m_TimeStepDays; //!< Member variable "m_TimeStepDays"
    double m_IntervalDays; //!< Member variable "m_IntervalDays"
    double m_ExclusionDays; //!< Member variable "m_ExclusionDays"
    VectorInt m_ForbiddenYears;

};

#endif
