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

wxDEFINE_EVENT(asEVT_ACTION_FORECAST_CLEAR, wxCommandEvent);
wxDEFINE_EVENT(asEVT_ACTION_FORECAST_NEW_ADDED, wxCommandEvent);

asForecastManager::asForecastManager(wxWindow *parent, asWorkspace *workspace)
        : m_parent(parent),
          m_workspace(workspace),
          m_aggregator(new asResultsForecastAggregator()),
          m_leadTimeOrigin(0)
{

}

asForecastManager::~asForecastManager()
{
    wxDELETE(m_aggregator);
}

bool asForecastManager::HasForecasts() const
{
    return (m_aggregator->GetMethodsNb() > 0);
}

void asForecastManager::AddDirectoryPastForecasts(const wxString &dir)
{
    m_directoriesPastForecasts.Add(dir);
}

int asForecastManager::GetLinearIndex(int methodRow, int forecastRow) const
{
    int counter = 0;
    for (int i = 0; i < m_aggregator->GetMethodsNb(); i++) {
        for (int j = 0; j < m_aggregator->GetForecastsNb(i); j++) {
            if (i == methodRow && j == forecastRow) {
                return counter;
            }
            counter++;
        }
    }

    wxFAIL;
    return 0;
}

int asForecastManager::GetMethodRowFromLinearIndex(int linearIndex) const
{
    int counter = 0;
    for (int i = 0; i < m_aggregator->GetMethodsNb(); i++) {
        for (int j = 0; j < m_aggregator->GetForecastsNb(i); j++) {
            if (counter == linearIndex) {
                return i;
            }
            counter++;
        }
    }

    wxFAIL;
    return 0;
}

int asForecastManager::GetForecastRowFromLinearIndex(int linearIndex) const
{
    int counter = 0;
    for (int i = 0; i < m_aggregator->GetMethodsNb(); i++) {
        for (int j = 0; j < m_aggregator->GetForecastsNb(i); j++) {
            if (counter == linearIndex) {
                return j;
            }
            counter++;
        }
    }

    wxFAIL;
    return 0;
}

void asForecastManager::ClearArrays()
{
    m_aggregator->ClearArrays();
}

void asForecastManager::ClearForecasts()
{
    ClearArrays();
    m_directoriesPastForecasts.Clear();

#if wxUSE_GUI
    wxCommandEvent eventClear(asEVT_ACTION_FORECAST_CLEAR);
    if (m_parent != nullptr) {
        m_parent->ProcessWindowEvent(eventClear);
    }
#endif
}

bool asForecastManager::Open(const wxString &filePath, bool doRefresh)
{
    // Check existance
    if (!wxFileName::FileExists(filePath)) {
        wxLogError(_("The file %s could not be found."), filePath);
        return false;
    }

    // Check extension
    wxFileName fname(filePath);
    wxString extension = fname.GetExt();
    if (!extension.IsSameAs("asff") && !extension.IsSameAs("fcst")) {
        wxLogError(_("The file extension (%s) is not correct (must be .asff)."), extension);
        return false;
    }

    // Create and load the forecast
    auto *forecast = new asResultsForecast;

    forecast->SetFilePath(filePath);
    if (!forecast->Load()) {
        wxDELETE(forecast);
        return false;
    }

    // Check the lead time origin
    if ((m_leadTimeOrigin != 0) && (forecast->GetLeadTimeOrigin() != m_leadTimeOrigin)) {
        wxLogVerbose("The forecast file has another lead time origin. Previous files were removed.");
        ClearForecasts();
    }
    m_leadTimeOrigin = forecast->GetLeadTimeOrigin();

    if (m_aggregator->Add(forecast)) {
#if wxUSE_GUI
        // Send event
        wxCommandEvent eventNew(asEVT_ACTION_FORECAST_NEW_ADDED);
        if (m_parent != nullptr) {
            if (doRefresh) {
                eventNew.SetString("last");
            }
            m_parent->ProcessWindowEvent(eventNew);
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

bool asForecastManager::OpenPastForecast(int methodRow, int forecastRow, const wxString &filePath)
{
    // Check existance
    if (!wxFileName::FileExists(filePath)) {
        wxLogError(_("The file %s could not be found."), filePath);
        return false;
    }

    // Check extension
    wxFileName fname(filePath);
    wxString extension = fname.GetExt();
    if (!extension.IsSameAs("asff") && !extension.IsSameAs("fcst")) {
        wxLogError(_("The file extension (%s) is not correct (must be .asff)."), extension);
        return false;
    }

    // Create and load the forecast
    auto *forecast = new asResultsForecast;

    forecast->SetFilePath(filePath);
    if (!forecast->Load()) {
        wxDELETE(forecast);
        return false;
    }

    // Check the lead time origin
    if (forecast->GetLeadTimeOrigin() >= m_leadTimeOrigin) {
        wxDELETE(forecast);
        return false;
    }
    m_aggregator->AddPastForecast(methodRow, forecastRow, forecast);

    wxLogVerbose("Past forecast of %s - %s of the %s loaded", forecast->GetMethodId(), forecast->GetSpecificTag(),
                 forecast->GetLeadTimeOriginString());

    return true;
}

void asForecastManager::LoadPastForecast(int methodRow, int forecastRow)
{
    // Check if already loaded
    wxASSERT(m_aggregator->GetMethodsNb() > methodRow);
    wxASSERT(m_aggregator->GetPastMethodsNb() > methodRow);
    if (m_aggregator->GetPastForecastsNb(methodRow, forecastRow) > 0)
        return;

    // Get the number of days to load
    int nbPastDays = m_workspace->GetTimeSeriesPlotPastDaysNb();

    // Get path
    wxString defPath = m_workspace->GetForecastsDirectory();
    defPath.Append(wxFileName::GetPathSeparator());

    // Directory
    wxString dirstructure = "YYYY";
    dirstructure.Append(wxFileName::GetPathSeparator());
    dirstructure.Append("MM");
    dirstructure.Append(wxFileName::GetPathSeparator());
    dirstructure.Append("DD");

    for (int bkwd = 0; bkwd <= nbPastDays; bkwd++) {
        double currentTime = m_leadTimeOrigin - bkwd;
        wxString directory = asTime::GetStringTime(currentTime, dirstructure);

        // Test for every hour
        for (int hr = 23; hr >= 0; hr--) {
            // Load from default directory
            wxString currentDirPath = defPath;
            currentDirPath.Append(directory);
            currentDirPath.Append(wxFileName::GetPathSeparator());

            double currentTimeHour = floor(currentTime) + (double)hr / 24.0;
            wxString nowstr = asTime::GetStringTime(currentTimeHour, "YYYYMMDDhh");
            wxString forecastname = m_aggregator->GetForecast(methodRow, forecastRow)->GetMethodId() + '.' +
                                    m_aggregator->GetForecast(methodRow, forecastRow)->GetSpecificTag();
            wxString filename = wxString::Format("%s.%s.asff", nowstr, forecastname);
            wxString fullPath = currentDirPath + filename;
            wxString filenameOld = wxString::Format("%s.%s.fcst", nowstr, forecastname);
            wxString fullPathOld = currentDirPath + filenameOld;

            if (wxFileName::FileExists(fullPath)) {
                OpenPastForecast(methodRow, forecastRow, fullPath);
            } else if (wxFileName::FileExists(fullPathOld)) {
                OpenPastForecast(methodRow, forecastRow, fullPathOld);
            } else {
                // Load from temporarly stored directories
                for (unsigned int iDir = 0; iDir < m_directoriesPastForecasts.Count(); iDir++) {
                    currentDirPath = m_directoriesPastForecasts.Item(iDir);
                    currentDirPath.Append(directory);
                    currentDirPath.Append(wxFileName::GetPathSeparator());
                    fullPath = currentDirPath + filename;
                    fullPathOld = currentDirPath + filenameOld;

                    if (wxFileName::FileExists(fullPath)) {
                        OpenPastForecast(methodRow, forecastRow, fullPath);
                        goto quitloop;
                    } else if (wxFileName::FileExists(fullPathOld)) {
                        OpenPastForecast(methodRow, forecastRow, fullPathOld);
                        goto quitloop;
                    }
                }
            }
            quitloop:;
        }
    }
}

void asForecastManager::LoadPastForecast(int methodRow)
{
    for (int i = 0; i < GetForecastsNb(methodRow); i++) {
        LoadPastForecast(methodRow, i);
    }
}
