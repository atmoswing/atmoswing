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

#include "asForecastManager.h"

#include "asIncludes.h"

wxDEFINE_EVENT(asEVT_ACTION_FORECAST_CLEAR, wxCommandEvent);
wxDEFINE_EVENT(asEVT_ACTION_FORECAST_NEW_ADDED, wxCommandEvent);

asForecastManager::asForecastManager(wxWindow* parent, asWorkspace* workspace)
    : _parent(parent),
      _workspace(workspace),
      _aggregator(nullptr),
      _leadTimeOrigin(0) {}

asForecastManager::~asForecastManager() {
    wxDELETE(_aggregator);
}

void asForecastManager::Init() {
    _aggregator = new asResultsForecastAggregator();
}

bool asForecastManager::HasForecasts() const {
    wxASSERT(_aggregator);
    return (_aggregator->GetMethodsNb() > 0);
}

bool asForecastManager::HasSubDailyForecasts() const {
    wxASSERT(_aggregator);

    for (int i = 0; i < _aggregator->GetMethodsNb(); i++) {
        for (int j = 0; j < _aggregator->GetForecastsNb(i); j++) {
            if (_aggregator->GetForecast(i, j)->IsSubDaily()) {
                return true;
            }
        }
    }

    return false;
}

void asForecastManager::AddDirectoryPastForecasts(const wxString& dir) {
    _directoriesPastForecasts.Add(dir);
}

int asForecastManager::GetLinearIndex(int methodRow, int forecastRow) const {
    wxASSERT(_aggregator);
    int counter = 0;
    for (int i = 0; i < _aggregator->GetMethodsNb(); i++) {
        for (int j = 0; j < _aggregator->GetForecastsNb(i); j++) {
            if (i == methodRow && j == forecastRow) {
                return counter;
            }
            counter++;
        }
    }

    wxFAIL;
    return 0;
}

int asForecastManager::GetMethodRowFromLinearIndex(int linearIndex) const {
    wxASSERT(_aggregator);
    int counter = 0;
    for (int i = 0; i < _aggregator->GetMethodsNb(); i++) {
        for (int j = 0; j < _aggregator->GetForecastsNb(i); j++) {
            if (counter == linearIndex) {
                return i;
            }
            counter++;
        }
    }

    wxFAIL;
    return 0;
}

int asForecastManager::GetForecastRowFromLinearIndex(int linearIndex) const {
    wxASSERT(_aggregator);
    int counter = 0;
    for (int i = 0; i < _aggregator->GetMethodsNb(); i++) {
        for (int j = 0; j < _aggregator->GetForecastsNb(i); j++) {
            if (counter == linearIndex) {
                return j;
            }
            counter++;
        }
    }

    wxFAIL;
    return 0;
}

void asForecastManager::ClearArrays() {
    wxASSERT(_aggregator);
    _aggregator->ClearArrays();
}

void asForecastManager::ClearForecasts() {
    ClearArrays();
    _directoriesPastForecasts.Clear();

#if USE_GUI
    wxCommandEvent eventClear(asEVT_ACTION_FORECAST_CLEAR);
    if (_parent != nullptr) {
        _parent->ProcessWindowEvent(eventClear);
    }
#endif
}

bool asForecastManager::Open(const wxString& filePath, bool doRefresh) {
    wxASSERT(_aggregator);

    // Check existance
    if (!wxFileName::FileExists(filePath)) {
        wxLogError(_("The file %s could not be found."), filePath);
        return false;
    }

    // Check extension
    wxFileName fname(filePath);
    wxString extension = fname.GetExt();
    if (!extension.IsSameAs("asff") && !extension.IsSameAs("fcst") && !extension.IsSameAs("nc")) {
        wxLogError(_("The file extension (%s) is not correct."), extension);
        return false;
    }

    // Create and load the forecast
    auto forecast = new asResultsForecast;

    forecast->SetFilePath(filePath);
    if (!forecast->Load()) {
        wxDELETE(forecast);
        return false;
    }

    // Check the lead time origin
    if ((_leadTimeOrigin != 0) && (forecast->GetLeadTimeOrigin() != _leadTimeOrigin)) {
        wxLogVerbose("The forecast file has another lead time origin. Previous files were removed.");
        ClearForecasts();
    }
    _leadTimeOrigin = forecast->GetLeadTimeOrigin();

    // Limit length
    if (forecast->IsSubDaily() && _workspace->GetTimeSeriesMaxLengthSubDaily() > 0) {
        forecast->LimitDataToHours(_workspace->GetTimeSeriesMaxLengthSubDaily());
    } else if (!forecast->IsSubDaily() && _workspace->GetTimeSeriesMaxLengthDaily() > 0) {
        forecast->LimitDataToDays(_workspace->GetTimeSeriesMaxLengthDaily());
    }

    if (_aggregator->Add(forecast)) {
#if USE_GUI
        // Send event
        wxCommandEvent eventNew(asEVT_ACTION_FORECAST_NEW_ADDED);
        if (_parent != nullptr) {
            if (doRefresh) {
                eventNew.SetString("last");
            }
            _parent->ProcessWindowEvent(eventNew);
        }
#else
        if (doRefresh) {
            wxLogVerbose("The GUI should be refreshed.");
        }
#endif
    } else {
        wxDELETE(forecast);
    }

    return true;
}

bool asForecastManager::OpenPastForecast(int methodRow, int forecastRow, const wxString& filePath) {
    wxASSERT(_aggregator);

    // Check existance
    if (!wxFileName::FileExists(filePath)) {
        wxLogError(_("The file %s could not be found."), filePath);
        return false;
    }

    // Check extension
    wxFileName fname(filePath);
    wxString extension = fname.GetExt();
    if (!extension.IsSameAs("asff") && !extension.IsSameAs("fcst") && !extension.IsSameAs("nc")) {
        wxLogError(_("The file extension (%s) is not correct."), extension);
        return false;
    }

    // Create and load the forecast
    auto forecast = new asResultsForecast;

    forecast->SetFilePath(filePath);
    if (!forecast->Load()) {
        wxDELETE(forecast);
        return false;
    }

    // Check the lead time origin
    if (forecast->GetLeadTimeOrigin() >= _leadTimeOrigin) {
        wxDELETE(forecast);
        return false;
    }

    // Limit length
    if (forecast->IsSubDaily() && _workspace->GetTimeSeriesMaxLengthSubDaily() > 0) {
        forecast->LimitDataToHours(_workspace->GetTimeSeriesMaxLengthSubDaily());
    } else if (!forecast->IsSubDaily() && _workspace->GetTimeSeriesMaxLengthDaily() > 0) {
        forecast->LimitDataToDays(_workspace->GetTimeSeriesMaxLengthDaily());
    }

    _aggregator->AddPastForecast(methodRow, forecastRow, forecast);

    wxLogVerbose("Past forecast of %s - %s of the %s loaded", forecast->GetMethodId(), forecast->GetSpecificTag(),
                 forecast->GetLeadTimeOriginString());

    return true;
}

void asForecastManager::LoadPastForecast(int methodRow, int forecastRow) {
    wxASSERT(_aggregator);

    // Check if already loaded
    wxASSERT(_aggregator->GetMethodsNb() > methodRow);
    wxASSERT(_aggregator->GetPastMethodsNb() > methodRow);
    if (_aggregator->GetPastForecastsNb(methodRow, forecastRow) > 0) return;

    // Get the number of days to load
    int nbPastDays = _workspace->GetTimeSeriesPlotPastDaysNb();

    // Get path
    wxString defPath = _workspace->GetForecastsDirectory();
    defPath.Append(DS);

    // Directory
    wxString dirstructure = "YYYY";
    dirstructure.Append(DS);
    dirstructure.Append("MM");
    dirstructure.Append(DS);
    dirstructure.Append("DD");

    for (int bkwd = 0; bkwd <= nbPastDays; bkwd++) {
        double currentTime = _leadTimeOrigin - bkwd;
        wxString directory = asTime::GetStringTime(currentTime, dirstructure);

        // Test for every hour
        for (int hr = 23; hr >= 0; hr--) {
            // Load from default directory
            wxString currentDirPath = defPath;
            currentDirPath.Append(directory);
            currentDirPath.Append(DS);

            double currentTimeHour = floor(currentTime) + (double)hr / 24.0;
            wxString nowstrV1 = asTime::GetStringTime(currentTimeHour, "YYYYMMDDhh");
            wxString nowstrV3 = asTime::GetStringTime(currentTimeHour, "YYYY-MM-DD_hh");
            wxString forecastname = _aggregator->GetForecast(methodRow, forecastRow)->GetMethodId() + '.' +
                                    _aggregator->GetForecast(methodRow, forecastRow)->GetSpecificTag();
            wxString filenameV4 = asStrF("%s.%s.nc", nowstrV3, forecastname);
            wxString fullPathV4 = currentDirPath + filenameV4;
            wxString filenameV3 = asStrF("%s.%s.asff", nowstrV3, forecastname);
            wxString fullPathV3 = currentDirPath + filenameV3;
            wxString filenameV2 = asStrF("%s.%s.asff", nowstrV1, forecastname);
            wxString fullPathV2 = currentDirPath + filenameV2;
            wxString filenameV1 = asStrF("%s.%s.fcst", nowstrV1, forecastname);
            wxString fullPathV1 = currentDirPath + filenameV1;

            if (wxFileName::FileExists(fullPathV4)) {
                OpenPastForecast(methodRow, forecastRow, fullPathV4);
            } else if (wxFileName::FileExists(fullPathV3)) {
                OpenPastForecast(methodRow, forecastRow, fullPathV3);
            } else if (wxFileName::FileExists(fullPathV2)) {
                OpenPastForecast(methodRow, forecastRow, fullPathV2);
            } else if (wxFileName::FileExists(fullPathV1)) {
                OpenPastForecast(methodRow, forecastRow, fullPathV1);
            } else {
                // Load from temporary stored directories
                for (int iDir = 0; iDir < _directoriesPastForecasts.Count(); iDir++) {
                    currentDirPath = _directoriesPastForecasts.Item(iDir);
                    currentDirPath.Append(directory);
                    currentDirPath.Append(DS);
                    fullPathV3 = currentDirPath + filenameV3;
                    fullPathV2 = currentDirPath + filenameV2;
                    fullPathV1 = currentDirPath + filenameV1;

                    if (wxFileName::FileExists(fullPathV3)) {
                        OpenPastForecast(methodRow, forecastRow, fullPathV3);
                        goto quitloop;
                    } else if (wxFileName::FileExists(fullPathV2)) {
                        OpenPastForecast(methodRow, forecastRow, fullPathV2);
                        goto quitloop;
                    } else if (wxFileName::FileExists(fullPathV1)) {
                        OpenPastForecast(methodRow, forecastRow, fullPathV1);
                        goto quitloop;
                    }
                }
            }
        quitloop:;
        }
    }
}

void asForecastManager::LoadPastForecast(int methodRow) {
    for (int i = 0; i < GetForecastsNb(methodRow); i++) {
        LoadPastForecast(methodRow, i);
    }
}
