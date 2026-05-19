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
      _initialized(false),
      _mode(mode),
      _start(start),
      _end(end),
      _timeStepDays(timeStepHours / 24.0) {
    wxASSERT(_end >= _start);
    wxASSERT(_timeStepDays > 0);
}

asTimeArray::asTimeArray(double start, double end, double timeStepHours, const wxString& mode)
    : asTime(),
      _initialized(false),
      _start(start),
      _end(end),
      _timeStepDays(timeStepHours / 24.0) {
    wxASSERT(_end >= _start);
    wxASSERT(_timeStepDays > 0);

    if (mode.IsSameAs("simple", false)) {
        _mode = Simple;
    } else if (mode.IsSameAs("DJF", false)) {
        _mode = DJF;
    } else if (mode.IsSameAs("MAM", false)) {
        _mode = MAM;
    } else if (mode.IsSameAs("JJA", false)) {
        _mode = JJA;
    } else if (mode.IsSameAs("SON", false)) {
        _mode = SON;
    } else if (mode.IsSameAs("days_interval", false) || mode.IsSameAs("DaysInterval", false)) {
        _mode = DaysInterval;
    } else if (mode.IsSameAs("predictand_thresholds", false) || mode.IsSameAs("PredictandThresholds", false)) {
        _mode = PredictandThresholds;
    } else {
        if (mode.Contains("_to_") || mode.Contains("To")) {
            _modeStr = mode;
            _mode = MonthsSelection;
        } else {
            wxLogError(_("Time array mode not correctly defined (%s)!"), mode);
            _mode = Custom;
        }
    }
}

asTimeArray::asTimeArray(double date)
    : asTime(),
      _initialized(false),
      _mode(SingleDay),
      _start(date),
      _end(date),
      _timeStepDays(0) {}

asTimeArray::asTimeArray(vd& timeArray)
    : asTime(),
      _initialized(false),
      _mode(Custom) {
    if (timeArray.size() == 1) {
        _initialized = false;
        _mode = SingleDay;
        _start = timeArray[0];
        _end = timeArray[0];
        _timeStepDays = 0;
    } else {
        wxASSERT(timeArray.size() > 1);
        wxASSERT(timeArray[timeArray.size() - 1] > timeArray[0]);

        _timeStepDays = timeArray[1] - timeArray[0];
        _start = timeArray[0];
        _end = timeArray[timeArray.size() - 1];
        _timeArray.resize(timeArray.size());

        for (int i = 0; i < timeArray.size(); i++) {
            _timeArray[i] = timeArray[i];
        }
    }
}

asTimeArray::asTimeArray(a1d& timeArray)
    : asTime(),
      _initialized(false),
      _mode(Custom) {
    wxASSERT(timeArray.size() > 0);

    // Get values
    _timeStepDays = timeArray[1] - timeArray[0];
    _start = timeArray[0];
    _end = timeArray[timeArray.size() - 1];
    _timeArray = timeArray;
}

bool asTimeArray::Init() {
    switch (_mode) {
        case SingleDay: {
            int year = GetYear(_start);
            if (IsYearForbidden(year)) {
                wxLogError(_("The given date is in an excluded year."));
            }
            _timeArray.resize(1);
            _timeArray[0] = _start;
            break;
        }
        case Simple:
        case DaysInterval: {
            _timeArray.resize(0);
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
            _timeArray.resize(0);
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

    _initialized = true;

    return true;
}

bool asTimeArray::Init(double targetDate, double intervalDays, double exclusionDays) {
    _timeArray.resize(0);

    switch (_mode) {
        case DaysInterval: {
            wxASSERT(intervalDays > 0);
            if (!BuildArrayDaysInterval(targetDate, intervalDays)) {
                wxLogError(_("Time array creation failed"));
                return false;
            }
            break;
        }
        case Simple: {
            _timeArray.resize(0);
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
            _timeArray.resize(0);
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

    _initialized = true;

    return true;
}

void asTimeArray::RemoveExcludedDates(double targetDate, double exclusionDays) {
    if (exclusionDays == 0) {
        return;
    }

    a1d newTimeArray;
    newTimeArray.resize(_timeArray.size());

    // The period to exclude
    double excludeStart = targetDate - exclusionDays;
    double excludeEnd = targetDate + exclusionDays;

    int counter = 0;
    for (double time : _timeArray) {
        if (time < excludeStart || time > excludeEnd) {
            newTimeArray[counter] = time;
            counter++;
        }
    }
    _timeArray = newTimeArray;

    // Resize final array
    if (_timeArray.size() != counter) {
        _timeArray.conservativeResize(counter);
    }
}

bool asTimeArray::Init(asPredictand& predictand, const wxString& seriesName, int stationId, float minThreshold,
                       float maxThreshold) {
    _timeArray.resize(0);

    wxASSERT(_mode == PredictandThresholds);
    if (_mode != PredictandThresholds) {
        wxLogError(_("The time array mode is not correctly set"));
        return false;
    }

    if (!BuildArrayPredictandThresholds(predictand, seriesName, stationId, minThreshold, maxThreshold)) {
        wxLogError(_("Time array creation failed"));
        return false;
    }

    _initialized = true;

    return true;
}

void asTimeArray::Pop(int index) {
    if (index < 0 || index >= _timeArray.size()) {
        return;
    }

    a1d timeArray = _timeArray;
    _timeArray.resize(timeArray.size() - 1);

    if (index == 0) {
        _timeArray = timeArray.bottomRows(timeArray.size() - 1);
        _start = _timeArray[0];
    } else if (index == timeArray.size() - 1) {
        _timeArray = timeArray.topRows(index);
        _end = _timeArray[_timeArray.size() - 1];
    } else {
        _timeArray.topRows(index) = timeArray.topRows(index);
        _timeArray.bottomRows(timeArray.size() - 1 - index) = timeArray.bottomRows(timeArray.size() - 1 - index);
    }
}

bool asTimeArray::BuildArraySimple() {
    // Check the time step integrity
    wxCHECK(_timeStepDays > 0, false);
    wxCHECK(fmod((_end - _start), _timeStepDays) == 0, false);

    // Get the time series length for allocation.
    auto length = int(1 + (_end - _start) / _timeStepDays);
    _timeArray.resize(length);

    // Build array
    int counter = 0;
    double previousVal = _start - _timeStepDays;
    for (int i = 0; i < length; i++) {
        double date = previousVal + _timeStepDays;
        previousVal = date;
        if (HasForbiddenYears()) {
            if (!IsYearForbidden(GetYear(date))) {
                _timeArray[counter] = date;
                counter++;
            }
        } else {
            _timeArray[counter] = date;
            counter++;
        }
    }

    // Resize final array
    if (_timeArray.size() != counter) {
        _timeArray.conservativeResize(counter);
    }

    return true;
}

bool asTimeArray::BuildArrayDaysInterval(double targetDate, double intervalDays) {
    // Check the timestep integrity
    wxCHECK(_timeStepDays > 0, false);
    wxCHECK(fmod((_end - _start), _timeStepDays) == 0, false);
    wxASSERT(_end > _start);
    wxASSERT(_start > 0);
    wxASSERT(_end > 0);

    // Array resizing (larger than required)
    int firstYear = GetTimeStruct(_start).year;
    int lastYear = GetTimeStruct(_end).year;
    int totLength = int((lastYear - firstYear + 1) * 2 * (intervalDays + 1) * (1.0 / _timeStepDays));
    wxASSERT(totLength > 0);
    wxASSERT(totLength < 289600);  // 4 times daily during 200 years...
    _timeArray.resize(totLength);

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
            if (thisTimeStep >= _start && thisTimeStep <= _end) {
                wxASSERT(counter < totLength);
                _timeArray[counter] = thisTimeStep;
                counter++;
            }
            thisTimeStep += _timeStepDays;
        }
    }

    // Check the vector length
    if (_timeArray.size() != counter) {
        _timeArray.conservativeResize(counter);
    }

    return true;
}

bool asTimeArray::BuildArraySeason() {
    // Check the timestep integrity
    wxCHECK(_timeStepDays > 0, false);
    wxCHECK(fmod((_end - _start), _timeStepDays) < 0.000001, false);

    // Get the beginning of the time array
    Time start = GetTimeStruct(_start);
    Time end = GetTimeStruct(_end);
    int firstHour = 0;
    if (_timeStepDays < 1.0) {
        firstHour = 24 * _timeStepDays;
    }
    int lastHour = 24 - 24 * _timeStepDays;

    // Array resizing
    int maxLength = int((end.year - start.year + 1) * (366 / _timeStepDays));
    _timeArray.resize(maxLength);

    // Build the time array
    int counter = 0;
    for (int year = start.year; year <= end.year + 1; year++) {
        double seasonStart = 0;
        double seasonEnd = 0;

        switch (_mode) {
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
                if (_modeStr.Contains("_to_")) {
                    separator = "_to_";
                } else if (_modeStr.Contains("To")) {
                    separator = "To";
                }

                int sep = _modeStr.Find(separator);
                wxString monthStart = _modeStr.Left(sep);
                wxString monthEnd = _modeStr.Mid(sep + separator.Length());

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
            while (seasonStart < _start) {
                seasonStart += _timeStepDays;
            }
        }
        if (year >= end.year) {
            while (seasonEnd > _end) {
                seasonEnd -= _timeStepDays;
            }
        }

        double currentDate = seasonStart;
        while (currentDate <= seasonEnd) {
            wxASSERT(counter < maxLength);

            if (HasForbiddenYears()) {
                if (!IsYearForbidden(GetYear(currentDate))) {
                    _timeArray[counter] = currentDate;
                    counter++;
                }
            } else {
                _timeArray[counter] = currentDate;
                counter++;
            }
            currentDate += _timeStepDays;
        }
    }

    // Check the vector length
    wxCHECK(_timeArray.rows() >= counter, false);
    if (_timeArray.rows() != counter) {
        _timeArray.conservativeResize(counter);
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
    a1d finalTimeArray(_timeArray.size());

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
        int rowTimeArray = asFindFloor(&_timeArray[0], &_timeArray[_timeArray.size() - 1], predictandTimeArray[i]);

        if (rowTimeArray == asOUT_OF_RANGE || rowTimeArray == asNOT_FOUND) {
            continue;
        }

        // Check that there is not more than a few hours of difference.
        if (std::abs(predictandTimeArray[i] - _timeArray[rowTimeArray]) < 1) {
            if (predictandData[i] >= minThreshold && predictandData[i] <= maxThreshold) {
                if (HasForbiddenYears()) {
                    if (!IsYearForbidden(GetYear(_timeArray[rowTimeArray]))) {
                        finalTimeArray[counter] = _timeArray[rowTimeArray];
                        counter++;
                    }
                } else {
                    finalTimeArray[counter] = _timeArray[rowTimeArray];
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
                        asTime::GetStringTime(_timeArray[rowTimeArray], "DD.MM.YYYY hh:mm"),
                        asTime::GetStringTime(predictandTimeArray[i], "DD.MM.YYYY hh:mm"));
                }
            } else {
                wxLogWarning(_("The day %s was not considered in the timearray due to difference in hours with %s."),
                             asTime::GetStringTime(_timeArray[rowTimeArray], "DD.MM.YYYY hh:mm"),
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
    _timeArray.resize(counter);
    _timeArray = finalTimeArray.head(counter);

    return true;
}

int asTimeArray::GetClosestIndex(double date) const {
    wxASSERT(_initialized);

    if (date - 0.00001 > _end || date + 0.00001 < _start) {  // Add a second for precision issues
        wxLogWarning(_("Trying to get a date outside of the time array."));
        return 0;
    }

    int index = asFindClosest(&_timeArray[0], &_timeArray[GetSize() - 1], date, asHIDE_WARNINGS);

    if (index == asOUT_OF_RANGE) return 0;

    return index;
}

int asTimeArray::GetIndexFirstAfter(double date, double dataTimeStep) const {
    wxASSERT(_initialized);

    if (dataTimeStep >= 24.0) {
        // At a daily time step, might be defined at 00h or 12h
        double intPart;
        std::modf(date, &intPart);
        date = intPart;
    }

    if (date - 0.00001 > _end) {  // Add a second for precision issues
        wxLogWarning(_("Trying to get a date outside of the time array."));
        return asOUT_OF_RANGE;
    }

    int index = asFindCeil(&_timeArray[0], &_timeArray[GetSize() - 1], date, asHIDE_WARNINGS);

    if (index == asOUT_OF_RANGE && date < _timeArray[0]) {
        return 0;
    }

    return index;
}

int asTimeArray::GetIndexFirstBefore(double date, double dataTimeStep) const {
    wxASSERT(_initialized);

    if (date + 0.00001 < _start) {  // Add a second for precision issues
        wxLogWarning(_("Trying to get a date outside of the time array."));
        return asOUT_OF_RANGE;
    }

    int index = asFindFloor(&_timeArray[0], &_timeArray[GetSize() - 1], date, asHIDE_WARNINGS);

    if (index == asOUT_OF_RANGE && date > _timeArray[GetSize() - 1]) {
        if (dataTimeStep >= 24.0) {
            // At a daily time step, might be defined at 00h or 12h
            double intPart;
            std::modf(date, &intPart);
            date = intPart;

            index = asFindFloor(&_timeArray[0], &_timeArray[GetSize() - 1], date, asHIDE_WARNINGS);
            if (index == asOUT_OF_RANGE && date > _timeArray[GetSize() - 1]) {
                return GetSize() - 1;
            }
        } else {
            return GetSize() - 1;
        }
    }

    return index;
}

bool asTimeArray::RemoveYears(vi years) {
    wxASSERT(_timeArray.size() > 0);
    wxASSERT(!years.empty());

    asSortArray(&years[0], &years[years.size() - 1], Asc);

    int arraySize = _timeArray.size();
    a1i flags = a1i::Zero(arraySize);

    for (int year : years) {
        double mjdStart = GetMJD(year, 1, 1);
        double mjdEnd = GetMJD(year, 12, 31);

        int indexStart = asFindCeil(&_timeArray[0], &_timeArray[arraySize - 1], mjdStart, asHIDE_WARNINGS);
        int indexEnd = asFindFloor(&_timeArray[0], &_timeArray[arraySize - 1], mjdEnd, asHIDE_WARNINGS);

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
            newTimeArray[counter] = _timeArray[i];
            counter++;
        }
    }

    _timeArray.resize(0);
    _timeArray = newTimeArray.segment(0, counter);

    return true;
}

bool asTimeArray::KeepOnlyYears(vi years) {
    wxASSERT(_timeArray.size() > 0);
    wxASSERT(!years.empty());

    asSortArray(&years[0], &years[years.size() - 1], Asc);

    int arraySize = _timeArray.size();
    a1i flags = a1i::Zero(arraySize);

    for (int year : years) {
        double mjdStart = GetMJD(year, 1, 1);
        double mjdEnd = GetMJD(year, 12, 31);

        int indexStart = asFindCeil(&_timeArray[0], &_timeArray[arraySize - 1], mjdStart, asHIDE_WARNINGS);
        int indexEnd = asFindFloor(&_timeArray[0], &_timeArray[arraySize - 1], mjdEnd, asHIDE_WARNINGS);

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
            newTimeArray[counter] = _timeArray[i];
            counter++;
        }
    }

    _timeArray.resize(0);
    _timeArray = newTimeArray.segment(0, counter);

    return true;
}

bool asTimeArray::HasForbiddenYears() const {
    return !_forbiddenYears.empty();
}

bool asTimeArray::IsYearForbidden(int year) const {
    if (_forbiddenYears.empty()) return false;

    int index = asFind(&_forbiddenYears[0], &_forbiddenYears[_forbiddenYears.size() - 1], year, 0, asHIDE_WARNINGS);

    return index != asOUT_OF_RANGE && index != asNOT_FOUND;
}

void asTimeArray::fixStartIfForbidden(double& currentStart) const {
    int currentStartYear = GetYear(currentStart);
    if (IsYearForbidden(currentStartYear)) {
        double yearEnd = GetMJD(currentStartYear, 12, 31, 23, 59);
        while (currentStart <= yearEnd) {
            currentStart += _timeStepDays;
        }
    }
}

void asTimeArray::fixEndIfForbidden(double& currentEnd) const {
    int currentEndYear = GetYear(currentEnd);
    if (IsYearForbidden(currentEndYear)) {
        double yearStart = GetMJD(currentEndYear, 1, 1, 0, 0);
        while (currentEnd >= yearStart) {
            currentEnd -= _timeStepDays;
        }
    }
}

void asTimeArray::KeepOnlyRange(int start, int end) {
    a1d timeArray = _timeArray;
    wxASSERT(_timeArray.size() > start);
    wxASSERT(_timeArray.size() > end);
    _timeArray.resize(end - start + 1);

    for (int i = 0; i < _timeArray.size(); i++) {
        _timeArray[i] = timeArray[start + i];
    }
}