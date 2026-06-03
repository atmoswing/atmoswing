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
 * Portions Copyright 2014-2015 Pascal Horton, Terranum.
 */

#ifndef AS_FORECAST_MANAGER_H
#define AS_FORECAST_MANAGER_H

#include <wx/arrstr.h>  // wxArrayString

#include "asHeadersBase.h"
#include "asResultsForecastAggregator.h"
#include "asWorkspace.h"

class wxWindow;

class asForecastManager {
  public:
    asForecastManager(wxWindow* parent, asWorkspace* workspace);

    virtual ~asForecastManager();

    void Init();

    bool HasForecasts() const;

    bool HasSubDailyForecasts() const;

    void ClearArrays();

    void ClearForecasts();

    bool Open(const wxString& filePath, bool doRefresh = true);

    bool OpenPastForecast(int methodRow, int forecastRow, const wxString& filePath);

    void LoadPastForecast(int methodRow, int forecastRow);

    void LoadPastForecast(int methodRow);

    void AddDirectoryPastForecasts(const wxString& dir);

    int GetLinearIndex(int methodRow, int forecastRow) const;

    int GetMethodRowFromLinearIndex(int linearIndex) const;

    int GetForecastRowFromLinearIndex(int linearIndex) const;

    asResultsForecastAggregator* GetAggregator() const {
        return _aggregator;
    }

    int GetMethodsNb() const {
        return _aggregator->GetMethodsNb();
    }

    int GetForecastsNb(int methodRow) const {
        return _aggregator->GetForecastsNb(methodRow);
    }

    int GetPastMethodsNb() const {
        return _aggregator->GetPastMethodsNb();
    }

    int GetPastForecastsNb(int methodRow) const {
        return _aggregator->GetPastForecastsNb(methodRow);
    }

    int GetPastForecastsNb(int methodRow, int forecastRow) const {
        return _aggregator->GetPastForecastsNb(methodRow, forecastRow);
    }

    asResultsForecast* GetForecast(int methodRow, int forecastRow) const {
        return _aggregator->GetForecast(methodRow, forecastRow);
    }

    asResultsForecast* GetPastForecast(int methodRow, int forecastRow, int leadtimeRow) const {
        return _aggregator->GetPastForecast(methodRow, forecastRow, leadtimeRow);
    }

    double GetLeadTimeOrigin() const {
        return _leadTimeOrigin;
    }

    void SetLeadTimeOrigin(double val) {
        _leadTimeOrigin = val;
    }

    wxString GetMethodName(int methodRow) const {
        return _aggregator->GetMethodName(methodRow);
    }

    vwxs GetMethodNames() const {
        return _aggregator->GetMethodNames();
    }

    wxArrayString GetMethodNamesWxArray() const {
        return _aggregator->GetMethodNamesWxArray();
    }

    wxString GetForecastName(int methodRow, int forecastRow) const {
        return _aggregator->GetForecastName(methodRow, forecastRow);
    }

    wxArrayString GetForecastNamesWxArray(int methodRow) const {
        return _aggregator->GetForecastNamesWxArray(methodRow);
    }

    wxArrayString GetCombinedForecastNamesWxArray() const {
        return _aggregator->GetCombinedForecastNamesWxArray();
    }

    wxString GetFilePath(int methodRow, int forecastRow) const {
        return _aggregator->GetFilePath(methodRow, forecastRow);
    }

    a1f GetTargetDates(int methodRow) const {
        return _aggregator->GetTargetDates(methodRow);
    }

    a1f GetTargetDates(int methodRow, int forecastRow) const {
        return _aggregator->GetTargetDates(methodRow, forecastRow);
    }

    a1f GetFullTargetDates() const {
        return _aggregator->GetFullTargetDates();
    }

    int GetForecastRowSpecificForStationId(int methodRow, int stationId) const {
        return _aggregator->GetForecastRowSpecificForStationId(methodRow, stationId);
    }

    int GetForecastRowSpecificForStationRow(int methodRow, int stationRow) const {
        return _aggregator->GetForecastRowSpecificForStationRow(methodRow, stationRow);
    }

    wxArrayString GetStationNames(int methodRow, int forecastRow) const {
        return _aggregator->GetStationNames(methodRow, forecastRow);
    }

    wxString GetStationName(int methodRow, int forecastRow, int stationRow) const {
        return _aggregator->GetStationName(methodRow, forecastRow, stationRow);
    }

    wxArrayString GetStationNamesWithHeights(int methodRow, int forecastRow) const {
        return _aggregator->GetStationNamesWithHeights(methodRow, forecastRow);
    }

    wxString GetStationNameWithHeight(int methodRow, int forecastRow, int stationRow) const {
        return _aggregator->GetStationNameWithHeight(methodRow, forecastRow, stationRow);
    }

    wxArrayString GetTargetDatesWxArray(int methodRow, int forecastRow) const {
        return _aggregator->GetTargetDatesWxArray(methodRow, forecastRow);
    }

    vf GetMaxExtent() const {
        return _aggregator->GetMaxExtent();
    }

  protected:
  private:
    wxWindow* _parent;
    asWorkspace* _workspace;
    asResultsForecastAggregator* _aggregator;
    double _leadTimeOrigin;
    wxArrayString _directoriesPastForecasts;
};

#endif
