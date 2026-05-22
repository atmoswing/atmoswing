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

#ifndef AS_TIME_ARRAY_H
#define AS_TIME_ARRAY_H

#include "asHeadersBase.h"
#include "asTime.h"

class asPredictand;

class asTimeArray : public asTime {
  public:
    enum Mode {
        SingleDay,        // A single day
        Simple,           // A simple full time array
        DJF,              // The DJF season
        MAM,              // The MAM season
        JJA,              // The JJA season
        SON,              // The SON season
        MonthsSelection,  // Selection of months
        DaysInterval,     // + or - an amount of days in reference to a date
        PredictandThresholds,
        Custom
    };

    asTimeArray(double start, double end, double timeStepHours, Mode mode);

    asTimeArray(double start, double end, double timeStepHours, const wxString& mode);

    explicit asTimeArray(double date);

    explicit asTimeArray(vd& timeArray);

    explicit asTimeArray(a1d& timeArray);

    ~asTimeArray() override = default;

    bool Init();

    bool Init(double targetDate, double intervalDays, double exclusionDays);

    bool Init(asPredictand& predictand, const wxString& seriesName, int stationId, float minThreshold,
              float maxThreshold);

    double operator[](unsigned int i) {
        wxASSERT(_initialized);
        wxASSERT(i < (unsigned)GetSize());
        return _timeArray[i];
    }

    void Pop(int index);

    bool BuildArraySimple();

    bool BuildArrayDaysInterval(double targetDate, double intervalDays);

    bool BuildArraySeason();

    bool BuildArrayPredictandThresholds(asPredictand& predictand, const wxString& seriesName, int stationId,
                                        float minThreshold, float maxThreshold);

    bool HasForbiddenYears() const;

    bool IsYearForbidden(int year) const;

    vi GetForbiddenYears() const {
        return _forbiddenYears;
    }

    void SetForbiddenYears(const vi& years) {
        _forbiddenYears = years;
    }

    bool RemoveYears(vi years);

    bool KeepOnlyYears(vi years);

    double GetStart() const {
        return _start;
    }

    int GetStartingYear() const {
        return GetYear(_start);
    }

    int GetStartingMonth() const {
        return GetMonth(_start);
    }

    int GetStartingDay() const {
        return GetDay(_start);
    }

    double GetStartingHour() const {
        double fractpart, intpart;
        fractpart = modf(_start, &intpart);
        return fractpart * 24;
    }

    double GetEnd() const {
        return _end;
    }

    int GetEndingYear() const {
        return GetYear(_end);
    }

    int GetEndingMonth() const {
        return GetMonth(_end);
    }

    double GetEndingHour() const {
        double fractpart, intpart;
        fractpart = modf(_end, &intpart);
        return fractpart * 24;
    }

    double GetTimeStepHours() const {
        return _timeStepDays * 24;
    }

    double GetTimeStepDays() const {
        return _timeStepDays;
    }

    a1d GetTimeArray() const {
        return _timeArray;
    }

    int GetSize() const {
        return (int)_timeArray.size();
    }

    double GetFirst() const {
        wxASSERT(_initialized);
        return _timeArray(0);
    }

    double GetLast() const {
        wxASSERT(_initialized);
        return _timeArray(_timeArray.rows() - 1);
    }

    int GetClosestIndex(double date) const;

    int GetIndexFirstAfter(double date, double dataTimeStep) const;

    int GetIndexFirstBefore(double date, double dataTimeStep) const;

    void KeepOnlyRange(int start, int end);

  protected:
  private:
    bool _initialized;
    Mode _mode;
    a1d _timeArray;
    double _start;
    double _end;
    double _timeStepDays;
    vi _forbiddenYears;
    wxString _modeStr;

    void fixStartIfForbidden(double& currentStart) const;

    void fixEndIfForbidden(double& currentEnd) const;

    void RemoveExcludedDates(double targetDate, double exclusionDays);
};

#endif
