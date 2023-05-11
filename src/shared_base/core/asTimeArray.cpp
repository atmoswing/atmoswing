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
 * Portions Copyright 2019 Pascal Horton, University of Bern.
 */

#include "asTimeArray.h"

#include <math.h>

#include "asPredictand.h"

asTimeArray::asTimeArray(double start, double end, double timeStepHours, Mode mode)
    : asTime(),
      m_initialized(false),
      m_mode(mode),
      m_start(start),
      m_end(end),
      m_timeStepDays(timeStepHours / 24.0) {
    wxASSERT(m_end >= m_start);
    wxASSERT(m_timeStepDays > 0);
}

asTimeArray::asTimeArray(double start, double end, double timeStepHours, const wxString& mode)
    : asTime(),
      m_initialized(false),
      m_start(start),
      m_end(end),
      m_timeStepDays(timeStepHours / 24.0) {
    wxASSERT(m_end >= m_start);
    wxASSERT(m_timeStepDays > 0);

    if (mode.IsSameAs("simple", false)) {
        m_mode = Simple;
    } else if (mode.IsSameAs("DJF", false)) {
        m_mode = DJF;
    } else if (mode.IsSameAs("MAM", false)) {
        m_mode = MAM;
    } else if (mode.IsSameAs("JJA", false)) {
        m_mode = JJA;
    } else if (mode.IsSameAs("SON", false)) {
        m_mode = SON;
    } else if (mode.IsSameAs("days_interval", false) || mode.IsSameAs("DaysInterval", false)) {
        m_mode = DaysInterval;
    } else if (mode.IsSameAs("predictand_thresholds", false) || mode.IsSameAs("PredictandThresholds", false)) {
        m_mode = PredictandThresholds;
    } else {
        if (mode.Contains("_to_") || mode.Contains("To")) {
            m_modeStr = mode;
            m_mode = MonthsSelection;
        } else {
            wxLogError(_("Time array mode not correctly defined (%s)!"), mode);
            m_mode = Custom;
        }
    }
}

asTimeArray::asTimeArray(double date)
    : asTime(),
      m_initialized(false),
      m_mode(SingleDay),
      m_start(date),
      m_end(date),
      m_timeStepDays(0) {}

asTimeArray::asTimeArray(vd& timeArray)
    : asTime(),
      m_initialized(false),
      m_mode(Custom) {
    if (timeArray.size() == 1) {
        m_initialized = false;
        m_mode = SingleDay;
        m_start = timeArray[0];
        m_end = timeArray[0];
        m_timeStepDays = 0;
    } else {
        wxASSERT(timeArray.size() > 1);
        wxASSERT(timeArray[timeArray.size() - 1] > timeArray[0]);

        m_timeStepDays = timeArray[1] - timeArray[0];
        m_start = timeArray[0];
        m_end = timeArray[timeArray.size() - 1];
        m_timeArray.resize(timeArray.size());

        for (int i = 0; i < timeArray.size(); i++) {
            m_timeArray[i] = timeArray[i];
        }
    }
}

asTimeArray::asTimeArray(a1d& timeArray)
    : asTime(),
      m_initialized(false),
      m_mode(Custom) {
    wxASSERT(timeArray.size() > 0);

    // Get values
    m_timeStepDays = timeArray[1] - timeArray[0];
    m_start = timeArray[0];
    m_end = timeArray[timeArray.size() - 1];
    m_timeArray = timeArray;
}

bool asTimeArray::Init() {
    switch (m_mode) {
        case SingleDay: {
            int year = GetYear(m_start);
            if (IsYearForbidden(year)) {
                wxLogError(_("The given date is in an excluded year."));
            }
            m_timeArray.resize(1);
            m_timeArray[0] = m_start;
            break;
        }
        case Simple:
        case DaysInterval: {
            m_timeArray.resize(0);
            if (!BuildArraySimple()) {
                wxLogError(_("Time array creation failed."));
                return false;
            }
            break;
        }
        case DJF:
        case MAM:
        case JJA:
        case SON:
        case MonthsSelection: {
            m_timeArray.resize(0);
            if (!BuildArraySeason()) {
                wxLogError(_("Time array creation failed"));
                return false;
            }
            break;
        }
        case Custom: {
            // Do not resize the array to 0 !
            break;
        }
        default: {
            wxLogError(_("The time array mode is not correctly set"));
            return false;
        }
    }

    m_initialized = true;

    return true;
}

bool asTimeArray::Init(double targetDate, double intervalDays, double exclusionDays) {
    m_timeArray.resize(0);

    switch (m_mode) {
        case DaysInterval: {
            wxASSERT(intervalDays > 0);
            if (!BuildArrayDaysInterval(targetDate, intervalDays)) {
                wxLogError(_("Time array creation failed"));
                return false;
            }
            break;
        }
        case Simple: {
            m_timeArray.resize(0);
            if (!BuildArraySimple()) {
                wxLogError(_("Time array creation failed."));
                return false;
            }
            break;
        }
        case DJF:
        case MAM:
        case JJA:
        case SON:
        case MonthsSelection: {
            m_timeArray.resize(0);
            if (!BuildArraySeason()) {
                wxLogError(_("Time array creation failed"));
                return false;
            }
            break;
        }
        default: {
            wxLogError(_("The time array mode is not allowed for the analogs."));
            return false;
        }
    }

    RemoveExcludedDates(targetDate, exclusionDays);

    m_initialized = true;

    return true;
}

void asTimeArray::RemoveExcludedDates(double targetDate, double exclusionDays) {
    a1d newTimeArray;
    newTimeArray.resize(m_timeArray.size());

    if (exclusionDays == 0) {
        exclusionDays = 30;
        wxLogWarning(_("The 'exclude_days' parameter cannot be 0 or ignored. Defaulted to 30 days."));
    }

    // The period to exclude
    double excludeStart = targetDate - exclusionDays;
    double excludeEnd = targetDate + exclusionDays;

    int counter = 0;
    for (int i = 0; i < m_timeArray.size(); ++i) {
        if (m_timeArray[i] < excludeStart || m_timeArray[i] > excludeEnd) {
            newTimeArray[counter] = m_timeArray[i];
            counter++;
        }
    }
    m_timeArray = newTimeArray;

    // Resize final array
    if (m_timeArray.size() != counter) {
        m_timeArray.conservativeResize(counter);
    }
}

bool asTimeArray::Init(asPredictand& predictand, const wxString& seriesName, int stationId, float minThreshold,
                       float maxThreshold) {
    m_timeArray.resize(0);

    wxASSERT(m_mode == PredictandThresholds);
    if (m_mode != PredictandThresholds) {
        wxLogError(_("The time array mode is not correctly set"));
        return false;
    }

    if (!BuildArrayPredictandThresholds(predictand, seriesName, stationId, minThreshold, maxThreshold)) {
        wxLogError(_("Time array creation failed"));
        return false;
    }

    m_initialized = true;

    return true;
}

void asTimeArray::Pop(int index) {
    if (index < 0 || index >= m_timeArray.size()) {
        return;
    }

    a1d timeArray = m_timeArray;
    m_timeArray.resize(timeArray.size() - 1);

    if (index == 0) {
        m_timeArray = timeArray.bottomRows(timeArray.size() - 1);
        m_start = m_timeArray[0];
    } else if (index == timeArray.size() - 1) {
        m_timeArray = timeArray.topRows(index);
        m_end = m_timeArray[m_timeArray.size() - 1];
    } else {
        m_timeArray.topRows(index) = timeArray.topRows(index);
        m_timeArray.bottomRows(timeArray.size() - 1 - index) = timeArray.bottomRows(timeArray.size() - 1 - index);
    }
}

bool asTimeArray::BuildArraySimple() {
    // Check the time step integrity
    wxCHECK(m_timeStepDays > 0, false);
    wxCHECK(fmod((m_end - m_start), m_timeStepDays) == 0, false);

    // Get the time series length for allocation.
    auto length = int(1 + (m_end - m_start) / m_timeStepDays);
    m_timeArray.resize(length);

    // Build array
    int counter = 0;
    double previousVal = m_start - m_timeStepDays;
    for (int i = 0; i < length; i++) {
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

    // Resize final array
    if (m_timeArray.size() != counter) {
        m_timeArray.conservativeResize(counter);
    }

    return true;
}

bool asTimeArray::BuildArrayDaysInterval(double targetDate, double intervalDays) {
    // Check the timestep integrity
    wxCHECK(m_timeStepDays > 0, false);
    wxCHECK(fmod((m_end - m_start), m_timeStepDays) == 0, false);
    wxASSERT(m_end > m_start);
    wxASSERT(m_start > 0);
    wxASSERT(m_end > 0);

    // Array resizing (larger than required)
    int firstYear = GetTimeStruct(m_start).year;
    int lastYear = GetTimeStruct(m_end).year;
    int totLength = int((lastYear - firstYear + 1) * 2 * (intervalDays + 1) * (1.0 / m_timeStepDays));
    wxASSERT(totLength > 0);
    wxASSERT(totLength < 289600);  // 4 times daily during 200 years...
    m_timeArray.resize(totLength);

    // Loop over the years
    int counter = 0;
    for (int year = firstYear; year <= lastYear; year++) {
        // Get the interval boundaries
        Time targetDateStruct = GetTimeStruct(targetDate);
        targetDateStruct.year = year;
        double currentStart = GetMJD(targetDateStruct) - intervalDays;
        double currentEnd = GetMJD(targetDateStruct) + intervalDays;

        // Check for forbidden years (validation)
        if (HasForbiddenYears()) {
            fixStartIfForbidden(currentStart);
            fixEndIfForbidden(currentEnd);
        }

        double thisTimeStep = currentStart;
        while (thisTimeStep <= currentEnd) {
            if (thisTimeStep >= m_start && thisTimeStep <= m_end) {
                wxASSERT(counter < totLength);
                m_timeArray[counter] = thisTimeStep;
                counter++;
            }
            thisTimeStep += m_timeStepDays;
        }
    }

    // Check the vector length
    if (m_timeArray.size() != counter) {
        m_timeArray.conservativeResize(counter);
    }

    return true;
}

bool asTimeArray::BuildArraySeason() {
    // Check the timestep integrity
    wxCHECK(m_timeStepDays > 0, false);
    wxCHECK(fmod((m_end - m_start), m_timeStepDays) < 0.000001, false);

    // Get the beginning of the time array
    Time start = GetTimeStruct(m_start);
    Time end = GetTimeStruct(m_end);
    int firstHour = 0;
    if (m_timeStepDays < 1.0) {
        firstHour = 24 * m_timeStepDays;
    }
    int lastHour = 24 - 24 * m_timeStepDays;

    // Array resizing
    int maxLength = int((end.year - start.year + 1) * (366 / m_timeStepDays));
    m_timeArray.resize(maxLength);

    // Build the time array
    int counter = 0;
    for (int year = start.year; year <= end.year + 1; year++) {
        double seasonStart = 0;
        double seasonEnd = 0;

        switch (m_mode) {
            case DJF:
                seasonStart = GetMJD(year - 1, 12, 1, firstHour);
                if (IsLeapYear(year)) {
                    seasonEnd = GetMJD(year, 2, 29, lastHour);
                } else {
                    seasonEnd = GetMJD(year, 2, 28, lastHour);
                }
                break;
            case MAM:
                seasonStart = GetMJD(year, 3, 1, firstHour);
                seasonEnd = GetMJD(year, 5, 31, lastHour);
                break;
            case JJA:
                seasonStart = GetMJD(year, 6, 1, firstHour);
                seasonEnd = GetMJD(year, 8, 31, lastHour);
                break;
            case SON:
                seasonStart = GetMJD(year, 9, 1, firstHour);
                seasonEnd = GetMJD(year, 11, 30, lastHour);
                break;
            case MonthsSelection: {
                wxString separator;
                if (m_modeStr.Contains("_to_")) {
                    separator = "_to_";
                } else if (m_modeStr.Contains("To")) {
                    separator = "To";
                }

                int sep = m_modeStr.Find(separator);
                wxString monthStart = m_modeStr.Left(sep);
                wxString monthEnd = m_modeStr.Mid(sep + separator.Length());

                if (monthStart.IsSameAs("January", false)) {
                    seasonStart = GetMJD(year, 1, 1, firstHour);
                } else if (monthStart.IsSameAs("February", false)) {
                    seasonStart = GetMJD(year, 2, 1, firstHour);
                } else if (monthStart.IsSameAs("March", false)) {
                    seasonStart = GetMJD(year, 3, 1, firstHour);
                } else if (monthStart.IsSameAs("April", false)) {
                    seasonStart = GetMJD(year, 4, 1, firstHour);
                } else if (monthStart.IsSameAs("May", false)) {
                    seasonStart = GetMJD(year, 5, 1, firstHour);
                } else if (monthStart.IsSameAs("June", false)) {
                    seasonStart = GetMJD(year, 6, 1, firstHour);
                } else if (monthStart.IsSameAs("July", false)) {
                    seasonStart = GetMJD(year, 7, 1, firstHour);
                } else if (monthStart.IsSameAs("August", false)) {
                    seasonStart = GetMJD(year, 8, 1, firstHour);
                } else if (monthStart.IsSameAs("September", false)) {
                    seasonStart = GetMJD(year, 9, 1, firstHour);
                } else if (monthStart.IsSameAs("October", false)) {
                    seasonStart = GetMJD(year, 10, 1, firstHour);
                } else if (monthStart.IsSameAs("November", false)) {
                    seasonStart = GetMJD(year, 11, 1, firstHour);
                } else if (monthStart.IsSameAs("December", false)) {
                    seasonStart = GetMJD(year, 12, 1, firstHour);
                } else {
                    wxLogError(_("Month '%s' not recognized."), monthStart);
                    return false;
                }

                if (monthEnd.IsSameAs("January", false)) {
                    seasonEnd = GetMJD(year, 1, 31, lastHour);
                } else if (monthEnd.IsSameAs("February", false)) {
                    if (IsLeapYear(year)) {
                        seasonEnd = GetMJD(year, 2, 29, lastHour);
                    } else {
                        seasonEnd = GetMJD(year, 2, 28, lastHour);
                    }
                } else if (monthEnd.IsSameAs("March", false)) {
                    seasonEnd = GetMJD(year, 3, 31, lastHour);
                } else if (monthEnd.IsSameAs("April", false)) {
                    seasonEnd = GetMJD(year, 4, 30, lastHour);
                } else if (monthEnd.IsSameAs("May", false)) {
                    seasonEnd = GetMJD(year, 5, 31, lastHour);
                } else if (monthEnd.IsSameAs("June", false)) {
                    seasonEnd = GetMJD(year, 6, 30, lastHour);
                } else if (monthEnd.IsSameAs("July", false)) {
                    seasonEnd = GetMJD(year, 7, 31, lastHour);
                } else if (monthEnd.IsSameAs("August", false)) {
                    seasonEnd = GetMJD(year, 8, 31, lastHour);
                } else if (monthEnd.IsSameAs("September", false)) {
                    seasonEnd = GetMJD(year, 9, 30, lastHour);
                } else if (monthEnd.IsSameAs("October", false)) {
                    seasonEnd = GetMJD(year, 10, 31, lastHour);
                } else if (monthEnd.IsSameAs("November", false)) {
                    seasonEnd = GetMJD(year, 11, 30, lastHour);
                } else if (monthEnd.IsSameAs("December", false)) {
                    seasonEnd = GetMJD(year, 12, 31, lastHour);
                } else {
                    wxLogError(_("Month '%s' not recognized."), monthEnd);
                    return false;
                }

                if (seasonEnd < seasonStart) {
                    Time timeStr = GetTimeStruct(seasonStart);
                    seasonStart = GetMJD(year - 1, timeStr.month, 1);
                }

                break;
            }
            default:
                wxLogError(_("Season not recognized."));
                return false;
        }

        if (year <= start.year + 1) {
            while (seasonStart < m_start) {
                seasonStart += m_timeStepDays;
            }
        }
        if (year >= end.year) {
            while (seasonEnd > m_end) {
                seasonEnd -= m_timeStepDays;
            }
        }

        double currentDate = seasonStart;
        while (currentDate <= seasonEnd) {
            wxASSERT(counter < maxLength);

            if (HasForbiddenYears()) {
                if (!IsYearForbidden(GetYear(currentDate))) {
                    m_timeArray[counter] = currentDate;
                    counter++;
                }
            } else {
                m_timeArray[counter] = currentDate;
                counter++;
            }
            currentDate += m_timeStepDays;
        }
    }

    // Check the vector length
    wxCHECK(m_timeArray.rows() >= counter, false);
    if (m_timeArray.rows() != counter) {
        m_timeArray.conservativeResize(counter);
    }

    return true;
}

bool asTimeArray::BuildArrayPredictandThresholds(asPredictand& predictand, const wxString& seriesName, int stationId,
                                                 float minThreshold, float maxThreshold) {
    // Build a simple array for reference
    if (!BuildArraySimple()) {
        wxLogError(_("Time array creation failed"));
    }

    // Get the time arrays
    a1d predictandTimeArray = predictand.GetTime();
    a1d finalTimeArray(m_timeArray.size());

    // Get data
    a1f predictandData;
    if (seriesName.IsSameAs("DataNormalized", false) || seriesName.IsSameAs("data_normalized", false)) {
        predictandData = predictand.GetDataNormalizedStation(stationId);
    } else if (seriesName.IsSameAs("DataRaw", false) || seriesName.IsSameAs("data_raw", false)) {
        predictandData = predictand.GetDataRawStation(stationId);
    } else {
        wxLogError(_("The predictand series is not correctly defined in the time array construction."));
        return false;
    }

    // Compare
    int counter = 0;
    int countOut = 0;
    for (int i = 0; i < predictandTimeArray.size(); i++) {
        // Search corresponding date in the time array.
        int rowTimeArray = asFindFloor(&m_timeArray[0], &m_timeArray[m_timeArray.size() - 1], predictandTimeArray[i]);

        if (rowTimeArray == asOUT_OF_RANGE || rowTimeArray == asNOT_FOUND) {
            continue;
        }

        // Check that there is not more than a few hours of difference.
        if (std::abs(predictandTimeArray[i] - m_timeArray[rowTimeArray]) < 1) {
            if (predictandData[i] >= minThreshold && predictandData[i] <= maxThreshold) {
                if (HasForbiddenYears()) {
                    if (!IsYearForbidden(GetYear(m_timeArray[rowTimeArray]))) {
                        finalTimeArray[counter] = m_timeArray[rowTimeArray];
                        counter++;
                    }
                } else {
                    finalTimeArray[counter] = m_timeArray[rowTimeArray];
                    counter++;
                }
            } else {
                countOut++;
            }
        } else {
            if (HasForbiddenYears()) {
                if (!IsYearForbidden(GetYear(predictandTimeArray[i]))) {
                    wxLogWarning(
                        _("The day %s was not considered in the timearray due to difference in hours with %s."),
                        asTime::GetStringTime(m_timeArray[rowTimeArray], "DD.MM.YYYY hh:mm"),
                        asTime::GetStringTime(predictandTimeArray[i], "DD.MM.YYYY hh:mm"));
                }
            } else {
                wxLogWarning(_("The day %s was not considered in the timearray due to difference in hours with %s."),
                             asTime::GetStringTime(m_timeArray[rowTimeArray], "DD.MM.YYYY hh:mm"),
                             asTime::GetStringTime(predictandTimeArray[i], "DD.MM.YYYY hh:mm"));
            }
        }
    }
    wxLogVerbose(_("%d days were in the precipitation range and %d were not."), counter, countOut);

    if (counter == 0) {
        wxLogError(_("The selection of the dates on the predictand threshold is empty!"));
        return false;
    }

    // Set result
    m_timeArray.resize(counter);
    m_timeArray = finalTimeArray.head(counter);

    return true;
}

int asTimeArray::GetClosestIndex(double date) const {
    wxASSERT(m_initialized);

    if (date - 0.00001 > m_end || date + 0.00001 < m_start) {  // Add a second for precision issues
        wxLogWarning(_("Trying to get a date outside of the time array."));
        return 0;
    }

    int index = asFindClosest(&m_timeArray[0], &m_timeArray[GetSize() - 1], date, asHIDE_WARNINGS);

    if (index == asOUT_OF_RANGE) return 0;

    return index;
}

int asTimeArray::GetIndexFirstAfter(double date, double dataTimeStep) const {
    wxASSERT(m_initialized);

    if (dataTimeStep >= 24.0) {
        // At a daily time step, might be defined at 00h or 12h
        double intPart;
        std::modf(date, &intPart);
        date = intPart;
    }

    if (date - 0.00001 > m_end) {  // Add a second for precision issues
        wxLogWarning(_("Trying to get a date outside of the time array."));
        return asOUT_OF_RANGE;
    }

    int index = asFindCeil(&m_timeArray[0], &m_timeArray[GetSize() - 1], date, asHIDE_WARNINGS);

    if (index == asOUT_OF_RANGE && date < m_timeArray[0]) {
        return 0;
    }

    return index;
}

int asTimeArray::GetIndexFirstBefore(double date, double dataTimeStep) const {
    wxASSERT(m_initialized);

    if (date + 0.00001 < m_start) {  // Add a second for precision issues
        wxLogWarning(_("Trying to get a date outside of the time array."));
        return asOUT_OF_RANGE;
    }

    int index = asFindFloor(&m_timeArray[0], &m_timeArray[GetSize() - 1], date, asHIDE_WARNINGS);

    if (index == asOUT_OF_RANGE && date > m_timeArray[GetSize() - 1]) {
        if (dataTimeStep >= 24.0) {
            // At a daily time step, might be defined at 00h or 12h
            double intPart;
            std::modf(date, &intPart);
            date = intPart;

            index = asFindFloor(&m_timeArray[0], &m_timeArray[GetSize() - 1], date, asHIDE_WARNINGS);
            if (index == asOUT_OF_RANGE && date > m_timeArray[GetSize() - 1]) {
                return GetSize() - 1;
            }
        } else {
            return GetSize() - 1;
        }
    }

    return index;
}

bool asTimeArray::RemoveYears(vi years) {
    wxASSERT(m_timeArray.size() > 0);
    wxASSERT(!years.empty());

    asSortArray(&years[0], &years[years.size() - 1], Asc);

    int arraySize = m_timeArray.size();
    a1i flags = a1i::Zero(arraySize);

    for (int year : years) {
        double mjdStart = GetMJD(year, 1, 1);
        double mjdEnd = GetMJD(year, 12, 31);

        int indexStart = asFindCeil(&m_timeArray[0], &m_timeArray[arraySize - 1], mjdStart, asHIDE_WARNINGS);
        int indexEnd = asFindFloor(&m_timeArray[0], &m_timeArray[arraySize - 1], mjdEnd, asHIDE_WARNINGS);

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
                wxLogWarning(_("The given year to remove fall outside of the time array."));
            }
        }
    }

    a1d newTimeArray(arraySize);
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

bool asTimeArray::KeepOnlyYears(vi years) {
    wxASSERT(m_timeArray.size() > 0);
    wxASSERT(!years.empty());

    asSortArray(&years[0], &years[years.size() - 1], Asc);

    int arraySize = m_timeArray.size();
    a1i flags = a1i::Zero(arraySize);

    for (int year : years) {
        double mjdStart = GetMJD(year, 1, 1);
        double mjdEnd = GetMJD(year, 12, 31);

        int indexStart = asFindCeil(&m_timeArray[0], &m_timeArray[arraySize - 1], mjdStart, asHIDE_WARNINGS);
        int indexEnd = asFindFloor(&m_timeArray[0], &m_timeArray[arraySize - 1], mjdEnd, asHIDE_WARNINGS);

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
                wxLogWarning(_("The given year to remove fall outside of the time array."));
            }
        }
    }

    a1d newTimeArray(arraySize);
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

bool asTimeArray::HasForbiddenYears() const {
    return !m_forbiddenYears.empty();
}

bool asTimeArray::IsYearForbidden(int year) const {
    if (m_forbiddenYears.empty()) return false;

    int index = asFind(&m_forbiddenYears[0], &m_forbiddenYears[m_forbiddenYears.size() - 1], year, 0, asHIDE_WARNINGS);

    return index != asOUT_OF_RANGE && index != asNOT_FOUND;
}

void asTimeArray::fixStartIfForbidden(double& currentStart) const {
    int currentStartYear = GetYear(currentStart);
    if (IsYearForbidden(currentStartYear)) {
        double yearEnd = GetMJD(currentStartYear, 12, 31, 23, 59);
        while (currentStart <= yearEnd) {
            currentStart += m_timeStepDays;
        }
    }
}

void asTimeArray::fixEndIfForbidden(double& currentEnd) const {
    int currentEndYear = GetYear(currentEnd);
    if (IsYearForbidden(currentEndYear)) {
        double yearStart = GetMJD(currentEndYear, 1, 1, 0, 0);
        while (currentEnd >= yearStart) {
            currentEnd -= m_timeStepDays;
        }
    }
}

void asTimeArray::KeepOnlyRange(int start, int end) {
    a1d timeArray = m_timeArray;
    wxASSERT(m_timeArray.size() > start);
    wxASSERT(m_timeArray.size() > end);
    m_timeArray.resize(end - start + 1);

    for (int i = 0; i < m_timeArray.size(); i++) {
        m_timeArray[i] = timeArray[start + i];
    }
}