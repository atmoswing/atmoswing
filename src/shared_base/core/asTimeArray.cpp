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

#include "asTimeArray.h"

#include <asDataPredictand.h>


asTimeArray::asTimeArray(double start, double end, double timestephours, Mode slctmode)
        : asTime()
{
    wxASSERT(end > start);
    wxASSERT(timestephours > 0);

    // Get values
    m_initialized = false;
    m_mode = slctmode;
    m_timeStepDays = timestephours / 24;
    m_start = start;
    m_end = end;
    m_intervalDays = 0;
    m_exclusionDays = 0;


}

asTimeArray::asTimeArray(double start, double end, double timestephours, const wxString &slctModeString)
        : asTime()
{
    wxASSERT(end > start);
    wxASSERT(timestephours > 0);

    if (slctModeString.CmpNoCase("simple") == 0) {
        m_mode = Simple;
    } else if (slctModeString.CmpNoCase("same_season") == 0 || slctModeString.CmpNoCase("SameSeason") == 0) {
        m_mode = SameSeason;
    } else if (slctModeString.CmpNoCase("season_DJF") == 0 || slctModeString.CmpNoCase("SeasonDJF") == 0) {
        m_mode = SeasonDJF;
    } else if (slctModeString.CmpNoCase("season_MAM") == 0 || slctModeString.CmpNoCase("SeasonMAM") == 0) {
        m_mode = SeasonMAM;
    } else if (slctModeString.CmpNoCase("season_JJA") == 0 || slctModeString.CmpNoCase("SeasonJJA") == 0) {
        m_mode = SeasonJJA;
    } else if (slctModeString.CmpNoCase("season_SON") == 0 || slctModeString.CmpNoCase("SeasonSON") == 0) {
        m_mode = SeasonSON;
    } else if (slctModeString.CmpNoCase("days_interval") == 0 || slctModeString.CmpNoCase("DaysInterval") == 0) {
        m_mode = DaysInterval;
    } else if (slctModeString.CmpNoCase("predictand_thresholds") == 0 ||
               slctModeString.CmpNoCase("PredictandThresholds") == 0) {
        m_mode = PredictandThresholds;
    } else {
        asLogError(wxString::Format(_("Time array mode not correctly defined (%s)!"), slctModeString));
    }

    // Get values
    m_initialized = false;
    m_timeStepDays = timestephours / 24;
    m_start = start;
    m_end = end;
    m_intervalDays = 0;
    m_exclusionDays = 0;

}

asTimeArray::asTimeArray()
        : asTime()
{
    // Should not be used for processing, only to get en empty object !
    m_initialized = false;
    m_mode = Custom;
    m_timeStepDays = 0;
    m_start = 0;
    m_end = 0;
    m_intervalDays = 0;
    m_exclusionDays = 0;
}

asTimeArray::asTimeArray(double date, Mode slctmode)
        : asTime()
{
    wxASSERT(slctmode == SingleDay);

    // Get values
    m_initialized = false;
    m_mode = slctmode;
    m_timeStepDays = 0;
    m_start = date;
    m_end = date;
    m_intervalDays = 0;
    m_exclusionDays = 0;

}

asTimeArray::asTimeArray(VectorDouble &timeArray)
        : asTime()
{
    wxASSERT(timeArray.size() > 1);
    wxASSERT(timeArray[timeArray.size() - 1] > timeArray[0]);

    // Get values
    m_initialized = false;
    m_mode = Custom;
    m_timeStepDays = timeArray[1] - timeArray[0];
    m_start = timeArray[0];
    m_end = timeArray[timeArray.size() - 1];
    m_intervalDays = 0;
    m_exclusionDays = 0;
    m_timeArray.resize(timeArray.size());

    for (unsigned int i = 0; i < timeArray.size(); i++) {
        m_timeArray[i] = timeArray[i];
    }
}

asTimeArray::asTimeArray(Array1DDouble &timeArray)
        : asTime()
{
    wxASSERT(timeArray.size() > 0);

    // Get values
    m_initialized = false;
    m_mode = Custom;
    m_timeStepDays = timeArray[1] - timeArray[0];
    m_start = timeArray[0];
    m_end = timeArray[timeArray.size() - 1];
    m_intervalDays = 0;
    m_exclusionDays = 0;
    m_timeArray = timeArray;
}

asTimeArray::~asTimeArray()
{
    //dtor
}

bool asTimeArray::Init()
{
    switch (m_mode) {
        case SingleDay: {
            int year = GetYear(m_start);
            if (IsYearForbidden(year)) {
                asLogError(_("The given date "));
            }
            m_timeArray.resize(1);
            m_timeArray[0] = m_start;
            break;
        }
        case Simple: {
            m_timeArray.resize(0);
            if (!BuildArraySimple()) {
                asLogError(_("Time array creation failed"));
                return false;
            }
            break;
        }
        case SeasonDJF: {
            m_timeArray.resize(0);
            double forecastdate = 51544; //01.01.2000
            if (!BuildArraySeasons(forecastdate)) {
                asLogError(_("Time array creation failed"));
                return false;
            }
            break;
        }
        case SeasonMAM: {
            m_timeArray.resize(0);
            double forecastdate = 51635; //01.04.2000
            if (!BuildArraySeasons(forecastdate)) {
                asLogError(_("Time array creation failed"));
                return false;
            }
            break;
        }
        case SeasonJJA: {
            m_timeArray.resize(0);
            double forecastdate = 51726; //01.07.2000
            if (!BuildArraySeasons(forecastdate)) {
                asLogError(_("Time array creation failed"));
                return false;
            }
            break;
        }
        case SeasonSON: {
            m_timeArray.resize(0);
            double forecastdate = 51818; //01.10.2000
            if (!BuildArraySeasons(forecastdate)) {
                asLogError(_("Time array creation failed"));
                return false;
            }
            break;
        }
        case Custom: {
            // Do not resize the array to 0 !
            break;
        }
        default: {
            asLogError(_("The time array mode is not correctly set"));
            return false;
        }
    }

    m_initialized = true;

    return true;
}

bool asTimeArray::Init(double forecastdate, double exclusiondays)
{
    m_timeArray.resize(0);

    wxASSERT(exclusiondays > 0);
    wxASSERT(forecastdate > 15022); // After 1900
    wxASSERT(forecastdate < 88069); // Before 2100
    wxASSERT(m_mode == SameSeason);

    // Get values
    m_exclusionDays = exclusiondays;

    if (!BuildArraySeasons(forecastdate)) {
        asLogError(_("Time array creation failed"));
        return false;
    }

    m_initialized = true;

    return true;
}

bool asTimeArray::Init(double forecastdate, double intervaldays, double exclusiondays)
{
    m_timeArray.resize(0);

    wxASSERT(exclusiondays > 0);
    wxASSERT(intervaldays > 0);
    wxASSERT(forecastdate > 15022); // After 1900
    wxASSERT(forecastdate < 88069); // Before 2100
    wxASSERT(m_mode == DaysInterval);

    // Get values
    m_intervalDays = intervaldays;
    m_exclusionDays = exclusiondays;

    if (!BuildArrayDaysInterval(forecastdate)) {
        asLogError(_("Time array creation failed"));
        return false;
    }

    m_initialized = true;

    return true;
}

bool asTimeArray::Init(asDataPredictand &predictand, const wxString &serieName, int stationId, float minThreshold,
                       float maxThreshold)
{
    m_timeArray.resize(0);

    wxASSERT(m_mode == PredictandThresholds);
    if (m_mode != PredictandThresholds) {
        asLogError(_("The time array mode is not correctly set"));
        return false;
    }

    if (!BuildArrayPredictandThresholds(predictand, serieName, stationId, minThreshold, maxThreshold)) {
        asLogError(_("Time array creation failed"));
        return false;
    }

    m_initialized = true;

    return true;
}

bool asTimeArray::Init(double forecastdate, double intervaldays, double exclusiondays, asDataPredictand &predictand,
                       const wxString &serieName, int stationId, float minThreshold, float maxThreshold)
{
    wxASSERT(exclusiondays > 0);

    m_timeArray.resize(0);

    switch (m_mode) {
        case Simple: {
            if (!BuildArraySimple()) {
                asLogError(_("Time array creation failed"));
                return false;
            }
            break;
        }
        case SeasonDJF: {
            double forecastdate = 51544; //01.01.2000
            if (!BuildArraySeasons(forecastdate)) {
                asLogError(_("Time array creation failed"));
                return false;
            }
            break;
        }
        case SeasonMAM: {
            double forecastdate = 51635; //01.04.2000
            if (!BuildArraySeasons(forecastdate)) {
                asLogError(_("Time array creation failed"));
                return false;
            }
            break;
        }
        case SeasonJJA: {
            double forecastdate = 51726; //01.07.2000
            if (!BuildArraySeasons(forecastdate)) {
                asLogError(_("Time array creation failed"));
                return false;
            }
            break;
        }
        case SeasonSON: {
            double forecastdate = 51818; //01.10.2000
            if (!BuildArraySeasons(forecastdate)) {
                asLogError(_("Time array creation failed"));
                return false;
            }
            break;
        }
        case SameSeason: {
            wxASSERT(forecastdate > 15022); // After 1900
            wxASSERT(forecastdate < 88069); // Before 2100
            m_exclusionDays = exclusiondays;
            if (!BuildArraySeasons(forecastdate)) {
                asLogError(_("Time array creation failed"));
                return false;
            }
            break;
        }
        case DaysInterval: {
            wxASSERT(forecastdate > 15022); // After 1900
            wxASSERT(forecastdate < 88069); // Before 2100
            m_intervalDays = intervaldays;
            m_exclusionDays = exclusiondays;
            if (!BuildArrayDaysInterval(forecastdate)) {
                asLogError(_("Time array creation failed"));
                return false;
            }
            break;
        }
        case PredictandThresholds: {
            if (!BuildArrayPredictandThresholds(predictand, serieName, stationId, minThreshold, maxThreshold)) {
                asLogError(_("Time array creation failed"));
                return false;
            }
            break;
        }
        default: {
            asLogError(_("The time array mode is not correctly set"));
            return false;
        }
    }

    m_initialized = true;

    return true;
}

bool asTimeArray::BuildArraySimple()
{
    // Check the timestep integrity
    wxCHECK(m_timeStepDays > 0, false);
    wxASSERT_MSG(fmod((m_end - m_start), m_timeStepDays) == 0,
                 wxString::Format("start=%f, end=%f, timestepdays=%f", m_start, m_end, m_timeStepDays));
    wxCHECK(fmod((m_end - m_start), m_timeStepDays) == 0, false);

    // Get the time serie length for allocation.
    size_t length = 1 + (m_end - m_start) / m_timeStepDays;
    m_timeArray.resize(length);

    // Build array
    int counter = 0;
    double previousVal = m_start - m_timeStepDays;
    for (size_t i = 0; i < length; i++) {
        double date = previousVal + m_timeStepDays;
        previousVal = date;
        if (HasForbiddenYears()) {
            if (!IsYearForbidden(GetYear(date))) {
                m_timeArray[counter] = date;
                counter++;
            }
        } else {
            m_timeArray[counter] = date;
            counter++;
        }
    }

    // Copy to final array
    m_timeArray.conservativeResize(counter);

    return true;
}

bool asTimeArray::BuildArrayDaysInterval(double forecastDate)
{
    // Check the timestep integrity
    wxASSERT(m_timeStepDays > 0);
    wxASSERT(fmod((m_end - m_start), m_timeStepDays) == 0);
    wxCHECK(m_timeStepDays > 0, false);
    wxCHECK(fmod((m_end - m_start), m_timeStepDays) == 0, false);
    wxASSERT(m_end > m_start);
    wxASSERT(m_start > 0);
    wxASSERT(m_end > 0);

    // The period to exclude
    double excludestart = forecastDate - m_exclusionDays;
    double excludeend = forecastDate + m_exclusionDays;

    // Get the structure of the forecast date
    TimeStruct forecastDateStruct = GetTimeStruct(forecastDate);

    // Array resizing (larger than required)
    int firstYear = GetTimeStruct(m_start).year;
    int lastYear = GetTimeStruct(m_end).year;
    int totLength = (lastYear - firstYear + 1) * 2 * (m_intervalDays + 1) * (1.0 / m_timeStepDays) + 50;
    wxASSERT(totLength > 0);
    wxASSERT(totLength < 289600); // 4 times daily during 200 years...
    m_timeArray.resize(totLength);

    // Loop over the years
    int counter = 0;
    for (int year = firstYear; year <= lastYear; year++) {
        // Get the interval boundaries
        TimeStruct adaptedForecastDateStruct = forecastDateStruct;
        adaptedForecastDateStruct.year = year;
        double currentStart = GetMJD(adaptedForecastDateStruct) - m_intervalDays;
        double currentEnd = GetMJD(adaptedForecastDateStruct) + m_intervalDays;

        // Check for forbiden years (validation)
        if (HasForbiddenYears()) {
            int currentFirstYear = GetYear(currentStart);
            int currentSecondYear = GetYear(currentEnd);

            if (IsYearForbidden(currentFirstYear)) {
                double firstYearEnd = GetMJD(currentFirstYear, 12, 31, 23, 59);
                while (currentStart <= firstYearEnd) {
                    currentStart += m_timeStepDays;
                }
            }
            if (IsYearForbidden(currentSecondYear)) {
                double secondYearStart = GetMJD(currentSecondYear, 1, 1, 0, 0);
                while (currentEnd >= secondYearStart) {
                    currentEnd -= m_timeStepDays;
                }

            }
        }

        for (double thisTimeStep = currentStart; thisTimeStep <= currentEnd; thisTimeStep += m_timeStepDays) {
            if (thisTimeStep >= m_start && thisTimeStep <= m_end) {
                if (thisTimeStep < excludestart || thisTimeStep > excludeend) {
                    wxASSERT(counter < totLength);

                    m_timeArray[counter] = thisTimeStep;
                    counter++;

#ifdef _DEBUG
                    if (counter > 1) {
                        wxASSERT_MSG(m_timeArray[counter - 1] > m_timeArray[counter - 2],
                                     wxString::Format(_("m_timeArray[%d]=%s, m_timeArray[%d]=%s"), counter - 1,
                                                      asTime::GetStringTime(m_timeArray[counter - 1]), counter - 2,
                                                      asTime::GetStringTime(m_timeArray[counter - 2])));
                    }
#endif
                }
            }
        }
    }

    // Check the vector length
    if (m_timeArray.size() != counter) {
        m_timeArray.conservativeResize(counter);
    }

    return true;
}

bool asTimeArray::BuildArraySeasons(double forecastdate)
{
    // Check the timestep integrity
    wxCHECK(m_timeStepDays > 0, false);
    wxCHECK(fmod((m_end - m_start), m_timeStepDays) < 0.000001, false);

    // The period to exclude
    double excludestart, excludeend;
    if (m_exclusionDays == 0) {
        excludestart = 0;
        excludeend = 0;
    } else {
        excludestart = forecastdate - m_exclusionDays;
        excludeend = forecastdate + m_exclusionDays;
    }

    // Get the season
    TimeStruct forecastdatestruct = GetTimeStruct(forecastdate);
    Season forecastseason = GetSeason(forecastdatestruct.month);

    // Adapt the hour if the timestep allows for it
    double hourstart = forecastdatestruct.hour;
    double hourend = forecastdatestruct.hour;
    while ((hourstart - m_timeStepDays * 24) >= 0) {
        hourstart -= m_timeStepDays * 24;
    }
    while ((hourend + m_timeStepDays * 24) < 24) {
        hourend += m_timeStepDays * 24;
    }

    // Adapt the minute if the timestep allows for it
    double minutestart = forecastdatestruct.min;
    double minuteend = forecastdatestruct.min;
    while ((minutestart - m_timeStepDays * 24 * 60) >= 0) {
        minutestart -= m_timeStepDays * 24 * 60;
    }
    while ((minuteend + m_timeStepDays * 24 * 60) < 60) {
        minuteend += m_timeStepDays * 24 * 60;
    }

    // Get the beginning of the time array
    TimeStruct startstruct = GetTimeStruct(m_start);
    TimeStruct endstruct = GetTimeStruct(m_end);

    TimeStruct startfirstyearstruct, endfirstyearstruct;
    TimeStructInit(startfirstyearstruct);
    TimeStructInit(endfirstyearstruct);

    startfirstyearstruct.year = startstruct.year;
    if (forecastseason == DJF) {
        startfirstyearstruct.year--; // This will be fixed further
    }
    startfirstyearstruct.month = GetSeasonStart(forecastseason).month;
    startfirstyearstruct.day = GetSeasonStart(forecastseason).day;
    startfirstyearstruct.hour = hourstart;
    startfirstyearstruct.min = minutestart;

    endfirstyearstruct.year = startstruct.year;
    endfirstyearstruct.month = GetSeasonEnd(forecastseason, startstruct.year).month;
    endfirstyearstruct.day = GetSeasonEnd(forecastseason, startstruct.year).day;
    endfirstyearstruct.hour = hourend;
    endfirstyearstruct.min = minuteend;

    double startfirstyear = GetMJD(startfirstyearstruct);
    double endfirstyear = GetMJD(endfirstyearstruct);

    if (endfirstyear < m_start) {
        startfirstyear = AddYear(startfirstyear);
        endfirstyearstruct.month = GetSeasonEnd(forecastseason, startstruct.year + 1).month;
        endfirstyearstruct.day = GetSeasonEnd(forecastseason, startstruct.year + 1).day;
        endfirstyear = GetMJD(endfirstyearstruct);
    }

    if (startfirstyear < m_start) {
        startfirstyear = m_start;
    }

    startfirstyearstruct = GetTimeStruct(startfirstyear);

    // Get the end of the time array
    TimeStruct startlastyearstruct, endlastyearstruct;
    TimeStructInit(startlastyearstruct);
    TimeStructInit(endlastyearstruct);

    startlastyearstruct.year = endstruct.year;
    startlastyearstruct.month = GetSeasonStart(forecastseason).month;
    startlastyearstruct.day = GetSeasonStart(forecastseason).day;
    startlastyearstruct.hour = hourstart;
    startlastyearstruct.min = minutestart;

    endlastyearstruct.year = endstruct.year;
    if (forecastseason == DJF) {
        endlastyearstruct.year++; // This will be fixed further
    }
    endlastyearstruct.month = GetSeasonEnd(forecastseason, endstruct.year).month;
    endlastyearstruct.day = GetSeasonEnd(forecastseason, endstruct.year).day;
    endlastyearstruct.hour = hourend;
    endlastyearstruct.min = minuteend;

    double startlastyear = GetMJD(startlastyearstruct);
    double endlastyear = GetMJD(endlastyearstruct);

    if (startlastyear > m_end) {
        startlastyear = SubtractYear(startlastyear);
        endlastyearstruct.month = GetSeasonEnd(forecastseason, endstruct.year - 1).month;
        endlastyearstruct.day = GetSeasonEnd(forecastseason, endstruct.year - 1).day;
        endlastyear = GetMJD(endlastyearstruct);
    }

    if (endlastyear > m_end) {
        endlastyear = m_end;
    }

    startlastyearstruct = GetTimeStruct(startlastyear);
    endlastyearstruct = GetTimeStruct(endlastyear);

    // Get the time serie length for allocation.
    // First year
    double difffirstyear = endfirstyear - startfirstyear;
    int rowsnbfirstyear = ceil((difffirstyear + 1) / m_timeStepDays);
    // Last year
    double difflastyear = endlastyear - startlastyear;
    int rowsnblastyear = ceil((difflastyear + 1) / m_timeStepDays);
    // Center - this is approximative, as it considers 92 days per month whatever the month is
    int yearsnb = endlastyearstruct.year - startfirstyearstruct.year - 1;
    int rowsnbcenter = ceil(93 / m_timeStepDays);
    int rowsnbcentertot = yearsnb * rowsnbcenter;
    int totlength = rowsnbfirstyear + rowsnbcentertot + rowsnblastyear;
    // Array resizing
    m_timeArray.resize(totlength);

    // Build the array
    double thistimestep = 0;
    int rowid = 0;
    double seasonend = 0;

    for (int i_year = startfirstyearstruct.year; i_year <= endlastyearstruct.year; i_year++) {
        if (i_year == startfirstyearstruct.year) {
            thistimestep = GetMJD(startfirstyearstruct.year, startfirstyearstruct.month, startfirstyearstruct.day,
                                  startfirstyearstruct.hour, startfirstyearstruct.min);
            seasonend = GetMJD(endfirstyearstruct.year, endfirstyearstruct.month, endfirstyearstruct.day, 23, 59, 59);
        } else if (i_year == endlastyearstruct.year) {
            thistimestep = GetMJD(startlastyearstruct.year, startlastyearstruct.month, startlastyearstruct.day,
                                  startlastyearstruct.hour, startlastyearstruct.min);
            seasonend = GetMJD(endlastyearstruct.year, endlastyearstruct.month, endlastyearstruct.day, 23, 59, 59);
        } else {
            thistimestep = GetMJD(i_year, GetSeasonStart(forecastseason).month, GetSeasonStart(forecastseason).day,
                                  startfirstyearstruct.hour, startfirstyearstruct.min);
            int thismonth, thisday;
            TimeStruct thisyearendstruct;
            if (forecastseason != DJF) {
                thismonth = GetSeasonEnd(forecastseason, i_year).month;
                thisday = GetSeasonEnd(forecastseason, i_year).day;
                thisyearendstruct = asTime::GetTimeStruct(i_year, thismonth, thisday, 23, 59, 59);
            } else {
                thismonth = GetSeasonEnd(forecastseason, i_year + 1).month;
                thisday = GetSeasonEnd(forecastseason, i_year + 1).day;
                thisyearendstruct = asTime::GetTimeStruct(i_year + 1, thismonth, thisday, 23, 59, 59);
            }
            seasonend = GetMJD(thisyearendstruct);
        }

        // Loop on every timestep
        while (thistimestep <= seasonend) {
            if (excludestart == 0 || excludeend == 0) {
                wxASSERT_MSG(rowid < totlength, _("The index is higher than the array size."));

                if (HasForbiddenYears()) {
                    if (!IsYearForbidden(GetYear(thistimestep))) {
                        m_timeArray[rowid] = thistimestep;
                        rowid++;
                    }
                } else {
                    m_timeArray[rowid] = thistimestep;
                    rowid++;
                }
            } else {
                if (thistimestep < excludestart || thistimestep > excludeend) {
                    wxASSERT_MSG(rowid < totlength, _("The index is higher than the array size."));

                    if (HasForbiddenYears()) {
                        if (!IsYearForbidden(GetYear(thistimestep))) {
                            m_timeArray[rowid] = thistimestep;
                            rowid++;
                        }
                    } else {
                        m_timeArray[rowid] = thistimestep;
                        rowid++;
                    }
                }
            }
            thistimestep += m_timeStepDays;
        }
    }

    // Check the vector length
    wxCHECK(m_timeArray.rows() >= rowid, false);
    if (m_timeArray.rows() != rowid) {
        m_timeArray.conservativeResize(rowid);
    }

    return true;
}

bool asTimeArray::BuildArrayPredictandThresholds(asDataPredictand &predictand, const wxString &serieName, int stationId,
                                                 float minThreshold, float maxThreshold)
{
    // Build a simple array for reference
    if (!BuildArraySimple()) {
        asLogError(_("Time array creation failed"));
    }

    // Get the time arrays
    Array1DDouble predictandTimeArray = predictand.GetTime();
    Array1DDouble fullTimeArray = m_timeArray;
    Array1DDouble finalTimeArray = Array1DDouble(fullTimeArray.size());

    // Get data
    Array1DFloat predictandData;
    if(serieName.IsSameAs("data_normalized") || serieName.IsSameAs("DataNormalized"))
    {
        predictandData = predictand.GetDataNormalizedStation(stationId);
    }
    else if(serieName.IsSameAs("data_gross") || serieName.IsSameAs("DataGross"))
    {
        predictandData = predictand.GetDataGrossStation(stationId);
    } else {
        asLogError(_("The predictand serie is not correctly defined in the time array construction."));
        return false;
    }

    // Compare
    int counter = 0;
    int countOut = 0;
    for (int i = 0; i < predictandTimeArray.size(); i++) {
        // Search corresponding date in the time array.
        int rowTimeArray = asTools::SortedArraySearchFloor(&fullTimeArray[0], &fullTimeArray[fullTimeArray.size() - 1],
                                                           predictandTimeArray[i]);

        if (rowTimeArray != asOUT_OF_RANGE && rowTimeArray != asNOT_FOUND) {
            // Check that there is not more than a few hours of difference.
            if (std::abs(predictandTimeArray[i] - fullTimeArray[rowTimeArray]) < 1) {
                if (predictandData[i] >= minThreshold && predictandData[i] <= maxThreshold) {
                    if (HasForbiddenYears()) {
                        if (!IsYearForbidden(GetYear(fullTimeArray[rowTimeArray]))) {
                            finalTimeArray[counter] = fullTimeArray[rowTimeArray];
                            counter++;
                        }
                    } else {
                        finalTimeArray[counter] = fullTimeArray[rowTimeArray];
                        counter++;
                    }
                } else {
                    countOut++;
                }
            } else {
                if (HasForbiddenYears()) {
                    if (!IsYearForbidden(GetYear(predictandTimeArray[i]))) {
                        asLogWarning(wxString::Format(
                                _("The day %s was not considered in the timearray due to difference in hours with %s."),
                                asTime::GetStringTime(fullTimeArray[rowTimeArray], "DD.MM.YYYY hh:mm"),
                                asTime::GetStringTime(predictandTimeArray[i], "DD.MM.YYYY hh:mm")));
                    }
                } else {
                    asLogWarning(wxString::Format(
                            _("The day %s was not considered in the timearray due to difference in hours with %s."),
                            asTime::GetStringTime(fullTimeArray[rowTimeArray], "DD.MM.YYYY hh:mm"),
                            asTime::GetStringTime(predictandTimeArray[i], "DD.MM.YYYY hh:mm")));
                }
            }
        }
    }
    asLogMessage(wxString::Format(_("%d days were in the precipitation range and %d were not."), counter, countOut));

    if (counter == 0) {
        asLogError(_("The selection of the dates on the predictand threshold is empty!"));
        return false;
    }

    // Set result
    m_timeArray.resize(counter);
    m_timeArray = finalTimeArray.head(counter);

    return true;
}

int asTimeArray::GetIndexFirstAfter(double date)
{
    wxASSERT(m_initialized);

    if (date > m_end) {
        asLogWarning(_("Trying to get a date outside of the time array."));
        return NaNDouble;
    }

    int index = asTools::SortedArraySearchCeil(&m_timeArray[0], &m_timeArray[GetSize() - 1], date);

    if (index == asOUT_OF_RANGE)
        return 0;

    return index;
}

int asTimeArray::GetIndexFirstBefore(double date)
{
    wxASSERT(m_initialized);

    if (date < m_start) {
        asLogWarning(_("Trying to get a date outside of the time array."));
        return NaNDouble;
    }

    int index = asTools::SortedArraySearchFloor(&m_timeArray[0], &m_timeArray[GetSize() - 1], date, asHIDE_WARNINGS);

    if (index == asOUT_OF_RANGE)
        return GetSize() - 1;

    return index;
}

bool asTimeArray::RemoveYears(const VectorInt &years)
{
    wxASSERT(m_timeArray.size() > 0);
    wxASSERT(years.size() > 0);

    VectorInt yearsRemove = years;

    asTools::SortArray(&yearsRemove[0], &yearsRemove[yearsRemove.size() - 1], Asc);

    int arraySize = m_timeArray.size();
    Array1DInt flags = Array1DInt::Zero(arraySize);

    for (unsigned int i = 0; i < yearsRemove.size(); i++) {
        int year = yearsRemove[i];
        double mjdStart = GetMJD(year, 1, 1);
        double mjdEnd = GetMJD(year, 12, 31);

        int indexStart = asTools::SortedArraySearchCeil(&m_timeArray[0], &m_timeArray[arraySize - 1], mjdStart,
                                                        asHIDE_WARNINGS);
        int indexEnd = asTools::SortedArraySearchFloor(&m_timeArray[0], &m_timeArray[arraySize - 1], mjdEnd,
                                                       asHIDE_WARNINGS);

        if (indexStart != asOUT_OF_RANGE && indexStart != asNOT_FOUND) {
            if (indexEnd != asOUT_OF_RANGE && indexEnd != asNOT_FOUND) {
                flags.segment(indexStart, indexEnd - indexStart + 1).setOnes();
            } else {
                flags.segment(indexStart, arraySize - indexStart).setOnes();
            }
        } else {
            if (indexEnd != asOUT_OF_RANGE && indexEnd != asNOT_FOUND) {
                flags.segment(0, indexEnd + 1).setOnes();
            } else {
                asLogWarning(_("The given year to remove fall outside of the time array."));
            }
        }
    }

    Array1DDouble newTimeArray(arraySize);
    int counter = 0;

    for (int i = 0; i < arraySize; i++) {
        if (flags[i] == 0) {
            newTimeArray[counter] = m_timeArray[i];
            counter++;
        }
    }

    m_timeArray.resize(0);
    m_timeArray = newTimeArray.segment(0, counter);

    return true;
}

bool asTimeArray::KeepOnlyYears(const VectorInt &years)
{
    wxASSERT(m_timeArray.size() > 0);
    wxASSERT(years.size() > 0);

    VectorInt yearsKeep = years;

    asTools::SortArray(&yearsKeep[0], &yearsKeep[yearsKeep.size() - 1], Asc);

    int arraySize = m_timeArray.size();
    Array1DInt flags = Array1DInt::Zero(arraySize);

    for (unsigned int i = 0; i < yearsKeep.size(); i++) {
        int year = yearsKeep[i];
        double mjdStart = GetMJD(year, 1, 1);
        double mjdEnd = GetMJD(year, 12, 31);

        int indexStart = asTools::SortedArraySearchCeil(&m_timeArray[0], &m_timeArray[arraySize - 1], mjdStart,
                                                        asHIDE_WARNINGS);
        int indexEnd = asTools::SortedArraySearchFloor(&m_timeArray[0], &m_timeArray[arraySize - 1], mjdEnd,
                                                       asHIDE_WARNINGS);

        if (indexStart != asOUT_OF_RANGE && indexStart != asNOT_FOUND) {
            if (indexEnd != asOUT_OF_RANGE && indexEnd != asNOT_FOUND) {
                flags.segment(indexStart, indexEnd - indexStart + 1).setOnes();
            } else {
                flags.segment(indexStart, arraySize - indexStart).setOnes();
            }
        } else {
            if (indexEnd != asOUT_OF_RANGE && indexEnd != asNOT_FOUND) {
                flags.segment(0, indexEnd + 1).setOnes();
            } else {
                asLogWarning(_("The given year to remove fall outside of the time array."));
            }
        }
    }

    Array1DDouble newTimeArray(arraySize);
    int counter = 0;

    for (int i = 0; i < arraySize; i++) {
        if (flags[i] == 1) {
            newTimeArray[counter] = m_timeArray[i];
            counter++;
        }
    }

    m_timeArray.resize(0);
    m_timeArray = newTimeArray.segment(0, counter);

    return true;
}

bool asTimeArray::HasForbiddenYears()
{
    if (m_forbiddenYears.size() == 0)
        return false;

    return true;
}

bool asTimeArray::IsYearForbidden(int year)
{
    if (m_forbiddenYears.size() == 0)
        return false;

    int index = asTools::SortedArraySearch(&m_forbiddenYears[0], &m_forbiddenYears[m_forbiddenYears.size() - 1], year,
                                           0, asHIDE_WARNINGS);

    if (index != asOUT_OF_RANGE && index != asNOT_FOUND) {
        return true;
    }

    return false;
}
