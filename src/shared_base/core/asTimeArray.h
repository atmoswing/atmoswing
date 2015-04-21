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
    
    /** Constructor for an empty object.
     * \note Should not be used for processing, only to get en empty object !
     */
    asTimeArray();

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
     * \return The current value of an item in m_timeArray
     */
    double operator[] (unsigned int i)
    {
        wxASSERT(m_initialized);
        wxASSERT(i < (unsigned)GetSize());
        return m_timeArray[i];
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
        return m_forbiddenYears;
    }

    void SetForbiddenYears(const VectorInt years)
    {
        m_forbiddenYears = years;
    }

    bool RemoveYears(const VectorInt &years);


    bool KeepOnlyYears(const VectorInt &years);

    /** Access m_mode
     * \return The current value of m_mode
     */
    Mode GetMode()
    {
        return m_mode;
    }

    /** Check if it is in simple mode
     * \return True if Mode is Simple
     */
    bool IsSimpleMode()
    {
        return (m_mode==Simple) || (m_mode==SingleDay);
    }

    /** Access m_start
     * \return The current value of m_start
     */
    double GetStart()
    {
        return m_start;
    }

    /** Get the year of the first day
     * \return The year of the first day
     */
    int GetFirstDayYear()
    {
        return GetYear(m_start);
    }

    /** Get the hour of the first day
     * \return The hour of the first day
     */
    double GetFirstDayHour()
    {
        double fractpart, intpart;
        fractpart = modf (m_start , &intpart);
        return fractpart*24;
    }

    /** Access m_end
     * \return The current value of m_end
     */
    double GetEnd()
    {
        return m_end;
    }

    /** Get the year of the last day
     * \return The year of the last day
     */
    int GetLastDayYear()
    {
        return GetYear(m_end);
    }

    /** Get the hour of the last day
     * \return The hour of the last day
     */
    double GetLastDayHour()
    {
        double fractpart, intpart;
        fractpart = modf (m_end , &intpart);
        return fractpart*24;
    }

    /** Access m_timeStepDays in hours
     * \return The current value of m_timeStepDays in hours
     */
    double GetTimeStepHours()
    {
        return m_timeStepDays*24;
    }

    /** Access m_timeStepDays in days
     * \return The current value of m_timeStepDays in days
     */
    double GetTimeStepDays()
    {
        return m_timeStepDays;
    }

    /** Access m_intervalDays in hours
     * \return The current value of m_intervalDays
     */
    double GetIntervalHours()
    {
        return m_intervalDays*24;
    }

    /** Return m_intervalDays in days
     * \return The current value of m_intervalDays in days
     */
    double GetIntervalDays()
    {
        return m_intervalDays;
    }

    /** Access m_exclusionDays in hours
     * \return The current value of m_exclusionDays
     */
    double GetExclusionHours()
    {
        return m_exclusionDays*24;
    }

    /** Access m_exclusionDays in days
     * \return The current value of m_exclusionDays in days
     */
    double GetExclusionDays()
    {
        return m_exclusionDays;
    }

    /** Access m_timeArray
     * \return The current value of m_timeArray
     */
    Array1DDouble& GetTimeArray()
    {
        return m_timeArray;
    }

    /** Get the size of the array
     * \return The size of m_timeArray
     */
    int GetSize()
    {
        return m_timeArray.size();
    }

    /** Access the first element in the time array
     * \return The current value of the first element
     */
    double GetFirst()
    {
        wxASSERT(m_initialized);
        return m_timeArray(0);
    }

    /** Access the last element in the time array
     * \return The current value of the last element
     */
    double GetLast()
    {
        wxASSERT(m_initialized);
        return m_timeArray(m_timeArray.rows()-1);
    }

    /** Get the pointer of the first element in the time array
     * \return The current pointer of the first element
     */
    double* GetPointerStart()
    {
        wxASSERT(m_initialized);
        return &m_timeArray(0);
    }

    /** Get the pointer of the first element in the time array
     * \return The current pointer of the first element
     */
    double* GetPointerEnd()
    {
        wxASSERT(m_initialized);
        return &m_timeArray(GetSize()-1);
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
    bool m_initialized;
    Mode m_mode; //!< Member variable "m_mode"
    Array1DDouble m_timeArray; //!< Member variable "m_timeArray"
    double m_start; //!< Member variable "m_start"
    double m_end; //!< Member variable "m_end"
    double m_timeStepDays; //!< Member variable "m_timeStepDays"
    double m_intervalDays; //!< Member variable "m_intervalDays"
    double m_exclusionDays; //!< Member variable "m_exclusionDays"
    VectorInt m_forbiddenYears;

};

#endif
