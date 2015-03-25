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
 
#include "asForecastManager.h"

wxDEFINE_EVENT(asEVT_ACTION_FORECAST_CLEAR, wxCommandEvent);
wxDEFINE_EVENT(asEVT_ACTION_FORECAST_NEW_ADDED, wxCommandEvent);

asForecastManager::asForecastManager(wxWindow* parent, asWorkspace* workspace)
{
    m_leadTimeOrigin = 0;
    m_parent = parent;
    m_workspace = workspace;
    m_aggregator = new asResultsAnalogsForecastAggregator();
}

asForecastManager::~asForecastManager()
{
    wxDELETE(m_aggregator);
}

bool asForecastManager::HasForecasts()
{
	return (m_aggregator->GetMethodsNb() > 0);
}

void asForecastManager::AddDirectoryPastForecasts(const wxString &dir)
{
    m_directoriesPastForecasts.Add(dir);
}

int asForecastManager::GetLinearIndex(int methodRow, int forecastRow)
{
    int counter = 0;
    for (int i=0; i<m_aggregator->GetMethodsNb(); i++)
    {
        for (int j=0; j<m_aggregator->GetForecastsNb(i); j++)
        {
            if (i==methodRow && j==forecastRow)
            {
                return counter;
            }
            counter++;
        }
    }

    wxFAIL;
    return 0;
}

int asForecastManager::GetMethodRowFromLinearIndex(int linearIndex)
{
    int counter = 0;
    for (int i=0; i<m_aggregator->GetMethodsNb(); i++)
    {
        for (int j=0; j<m_aggregator->GetForecastsNb(i); j++)
        {
            if (counter==linearIndex)
            {
                return i;
            }
            counter++;
        }
    }

    wxFAIL;
    return 0;
}

int asForecastManager::GetForecastRowFromLinearIndex(int linearIndex)
{
    int counter = 0;
    for (int i=0; i<m_aggregator->GetMethodsNb(); i++)
    {
        for (int j=0; j<m_aggregator->GetForecastsNb(i); j++)
        {
            if (counter==linearIndex)
            {
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
        wxCommandEvent eventClear (asEVT_ACTION_FORECAST_CLEAR);
        if (m_parent != NULL) {
            m_parent->ProcessWindowEvent(eventClear);
        }
    #endif
}

bool asForecastManager::Open(const wxString &filePath, bool doRefresh)
{
    // Check existance
    if (!wxFileName::FileExists(filePath))
    {
        asLogError(wxString::Format(_("The file %s could not be found."), filePath));
        return false;
    }

    // Check extension
    wxFileName fname(filePath);
    wxString extension = fname.GetExt();
    if (!extension.IsSameAs("asff") && !extension.IsSameAs("fcst"))
    {
        asLogError(wxString::Format(_("The file extension (%s) is not correct (must be .asff)."), extension));
        return false;
    }

    // Create and load the forecast
    asResultsAnalogsForecast* forecast = new asResultsAnalogsForecast;

    if(!forecast->Load(filePath)) 
    {
        wxDELETE(forecast);
        return false;
    }

    // Check the lead time origin
    if( (m_leadTimeOrigin!=0) && (forecast->GetLeadTimeOrigin()!=m_leadTimeOrigin) )
    {
        asLogMessage("The forecast file has another lead time origin. Previous files were removed.");
        ClearForecasts();
    }
    m_leadTimeOrigin = forecast->GetLeadTimeOrigin();

    m_aggregator->Add(forecast);

    #if wxUSE_GUI
        // Send event
        wxCommandEvent eventNew (asEVT_ACTION_FORECAST_NEW_ADDED);
        if (m_parent != NULL) {
            if (doRefresh)
            {
                eventNew.SetString("last");
            }
            m_parent->ProcessWindowEvent(eventNew);
        }
    #else
        if (doRefresh)
        {
            asLogMessage("The GUI should be refreshed.");
        }
    #endif

    return true;
}

bool asForecastManager::OpenPastForecast(int methodRow, int forecastRow, const wxString &filePath)
{
    // Check existance
    if (!wxFileName::FileExists(filePath))
    {
        asLogError(wxString::Format(_("The file %s could not be found."), filePath));
        return false;
    }

    // Check extension
    wxFileName fname(filePath);
    wxString extension = fname.GetExt();
    if (!extension.IsSameAs("asff") && !extension.IsSameAs("fcst"))
    {
        asLogError(wxString::Format(_("The file extension (%s) is not correct (must be .asff)."), extension));
        return false;
    }

    // Create and load the forecast
    asResultsAnalogsForecast* forecast = new asResultsAnalogsForecast;

    if(!forecast->Load(filePath))
    {
        wxDELETE(forecast);
        return false;
    }

    // Check the lead time origin
    if(forecast->GetLeadTimeOrigin()>=m_leadTimeOrigin)
    {
        wxDELETE(forecast);
        return false;
    }
    m_aggregator->AddPastForecast(methodRow, forecastRow, forecast);

    asLogMessage(wxString::Format("Past forecast of %s - %s of the %s loaded", forecast->GetMethodId(), forecast->GetSpecificTag(), forecast->GetLeadTimeOriginString()));

    return true;
}

void asForecastManager::LoadPastForecast(int methodRow, int forecastRow)
{
    // Check if already loaded
    wxASSERT(m_aggregator->GetMethodsNb()>methodRow);
    wxASSERT(m_aggregator->GetPastMethodsNb()>methodRow);
	if (m_aggregator->GetPastForecastsNb(methodRow, forecastRow)>0) return;

    // Get the number of days to load
    int nbPastDays = m_workspace->GetTimeSeriesPlotPastDaysNb();

    // Get path
    wxString defPath = m_workspace->GetForecastsDirectory();
    defPath.Append(DS);

    // Directory
    wxString dirstructure = "YYYY";
    dirstructure.Append(DS);
    dirstructure.Append("MM");
    dirstructure.Append(DS);
    dirstructure.Append("DD");

    for (int bkwd=0; bkwd<=nbPastDays; bkwd++)
    {
        double currentTime = m_leadTimeOrigin-bkwd;
        wxString directory = asTime::GetStringTime(currentTime, dirstructure);

        // Test for every hour
        for (double hr=23; hr>=0; hr--)
        {
            // Load from default directory
            wxString currentDirPath = defPath;
            currentDirPath.Append(directory);
            currentDirPath.Append(DS);

            double currentTimeHour = floor(currentTime)+ hr/24.0;
            wxString nowstr = asTime::GetStringTime(currentTimeHour, "YYYYMMDDhh");
            wxString forecastname = m_aggregator->GetForecast(methodRow, forecastRow)->GetMethodId() + '.' + m_aggregator->GetForecast(methodRow, forecastRow)->GetSpecificTag();
            wxString filename = wxString::Format("%s.%s.asff",nowstr,forecastname);
            wxString fullPath = currentDirPath + filename;
            wxString filenameOld = wxString::Format("%s.%s.fcst",nowstr,forecastname);
            wxString fullPathOld = currentDirPath + filenameOld;

            if (wxFileName::FileExists(fullPath))
            {
                OpenPastForecast(methodRow, forecastRow, fullPath);
            }
            else if (wxFileName::FileExists(fullPathOld))
            {
                OpenPastForecast(methodRow, forecastRow, fullPathOld);
            }
            else
            {
                // Load from temporarly stored directories
                for (unsigned int i_dir=0; i_dir<m_directoriesPastForecasts.Count(); i_dir++)
                {
                    currentDirPath = m_directoriesPastForecasts.Item(i_dir);
                    currentDirPath.Append(directory);
                    currentDirPath.Append(DS);
                    fullPath = currentDirPath + filename;
                    fullPathOld = currentDirPath + filenameOld;

                    if (wxFileName::FileExists(fullPath))
                    {
                        OpenPastForecast(methodRow, forecastRow, fullPath);
                        goto quitloop;
                    }
                    else if (wxFileName::FileExists(fullPathOld))
                    {
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
    for (int i=0; i<GetForecastsNb(methodRow); i++)
    {
        LoadPastForecast(methodRow, i);
    }
}
