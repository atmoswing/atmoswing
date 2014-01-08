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

#include <asResultsAnalogsForecast.h>

wxDEFINE_EVENT(asEVT_ACTION_FORECAST_CLEAR, wxCommandEvent);
wxDEFINE_EVENT(asEVT_ACTION_FORECAST_NEW_ADDED, wxCommandEvent);

asForecastManager::asForecastManager(wxWindow* parent)
{
    m_LeadTimeOrigin = 0;
    m_Parent = parent;
}

asForecastManager::~asForecastManager()
{
    //dtor
}

void asForecastManager::AddDirectoryPastForecasts(const wxString &dir)
{
    wxString defPath = wxFileConfig::Get()->Read("/StandardPaths/ForecastResultsDir", asConfig::GetDefaultUserWorkingDir() + "ForecastResults" + DS);

    if (!dir.IsSameAs(defPath,false) && !dir.IsSameAs(defPath+DS,false))
    {
        m_DirectoriesPastForecasts.Add(dir);
    }
}

void asForecastManager::ClearArrays()
{
    for (int i=0; (unsigned)i<m_CurrentForecasts.size(); i++)
    {
        wxDELETE(m_CurrentForecasts[i]);
    }
    m_CurrentForecasts.clear();

    for (int i=0; (unsigned)i<m_PastForecasts.size(); i++)
    {
        for (int j=0; (unsigned)j<m_PastForecasts[i].size(); j++)
        {
            wxDELETE(m_PastForecasts[i][j]);
        }
    }
    m_PastForecasts.clear();
}

bool asForecastManager::Open(const wxString &filePath, bool doRefresh)
{
    // Check existance
    if (!wxFileName::FileExists(filePath))
    {
        asLogError(wxString::Format(_("The file %s could not be found."), filePath.c_str()));
        return false;
    }

    // Check extension
    wxFileName fname(filePath);
    wxString extension = fname.GetExt();
    if (!extension.IsSameAs("fcst"))
    {
        asLogError(wxString::Format(_("The file extension (%s) is not correct (must be .fcst)."), extension.c_str()));
        return false;
    }

    // Create and load the forecast
    asResultsAnalogsForecast* forecast = new asResultsAnalogsForecast(wxEmptyString);

    if(!forecast->Load(filePath)) return false;

    // Check the lead time origin
    if( (m_LeadTimeOrigin!=0) && (forecast->GetLeadTimeOrigin()!=m_LeadTimeOrigin) )
    {
        asLogMessage("The forecast file has another lead time origin. Previous files were removed.");
        ClearArrays();
        m_DirectoriesPastForecasts.Clear();

        #if wxUSE_GUI
            wxCommandEvent eventClear (asEVT_ACTION_FORECAST_CLEAR);
            if (m_Parent != NULL) {
                m_Parent->ProcessWindowEvent(eventClear);
            }
        #endif

    }
    m_LeadTimeOrigin = forecast->GetLeadTimeOrigin();

    m_CurrentForecasts.push_back(forecast);
    std::vector <asResultsAnalogsForecast*> emptyVector;
    m_PastForecasts.push_back(emptyVector);

    #if wxUSE_GUI
        // Send event
        wxCommandEvent eventNew (asEVT_ACTION_FORECAST_NEW_ADDED);
        if (m_Parent != NULL) {
            eventNew.SetInt(m_CurrentForecasts.size()-1);
            if (doRefresh)
            {
                eventNew.SetString("last");
            }
            m_Parent->ProcessWindowEvent(eventNew);
        }
    #endif

    return true;
}

bool asForecastManager::OpenPastForecast(const wxString &filePath, int forecastSelection)
{
    // Check existance
    if (!wxFileName::FileExists(filePath))
    {
        asLogError(wxString::Format(_("The file %s could not be found."), filePath.c_str()));
        return false;
    }

    // Check extension
    wxFileName fname(filePath);
    wxString extension = fname.GetExt();
    if (!extension.IsSameAs("fcst"))
    {
        asLogError(wxString::Format(_("The file extension (%s) is not correct (must be .fcst)."), extension.c_str()));
        return false;
    }

    // Create and load the forecast
    asResultsAnalogsForecast* forecast = new asResultsAnalogsForecast(wxEmptyString);

    if(!forecast->Load(filePath)) return false;

    // Check the lead time origin
    if(forecast->GetLeadTimeOrigin()>=m_LeadTimeOrigin) return false;
    m_PastForecasts[forecastSelection].push_back(forecast);

    asLogMessage(wxString::Format("Past forecast of %s of the %s loaded", forecast->GetModelName().c_str(), forecast->GetLeadTimeOriginString().c_str()));

    return true;
}

void asForecastManager::LoadPastForecast(int forecastSelection)
{
    // Check if already loaded
    wxASSERT(m_PastForecasts.size()>(unsigned)forecastSelection);
    if (m_PastForecasts[forecastSelection].size()>0) return;

    // Get the number of days to load
    wxConfigBase *pConfig = wxFileConfig::Get();
    int nbPastDays = 3;
    pConfig->Read("/Plot/PastDaysNb", &nbPastDays, 3);

    // Get path
    wxString defPath = wxFileConfig::Get()->Read("/StandardPaths/ForecastResultsDir", asConfig::GetDefaultUserWorkingDir() + "ForecastResults" + DS);
    defPath.Append(DS);

    // Directory
    wxString dirstructure = "YYYY";
    dirstructure.Append(DS);
    dirstructure.Append("MM");
    dirstructure.Append(DS);
    dirstructure.Append("DD");

    for (int bkwd=0; bkwd<=nbPastDays; bkwd++)
    {
        double currentTime = m_LeadTimeOrigin-bkwd;
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
            wxString modelname = GetCurrentForecast(forecastSelection)->GetModelName();
            wxString ext = "fcst";
            wxString filename = wxString::Format("%s.%s.%s",nowstr.c_str(),modelname.c_str(),ext.c_str());
            wxString fullPath = currentDirPath + filename;

            if (wxFileName::FileExists(fullPath))
            {
                OpenPastForecast(fullPath, forecastSelection);
            }
            else
            {
                // Load from temporarly stored directories
                for (unsigned int i_dir=0; i_dir<m_DirectoriesPastForecasts.Count(); i_dir++)
                {
                    currentDirPath = m_DirectoriesPastForecasts.Item(i_dir);
                    currentDirPath.Append(directory);
                    currentDirPath.Append(DS);
                    fullPath = currentDirPath + filename;

                    if (wxFileName::FileExists(fullPath))
                    {
                        OpenPastForecast(fullPath, forecastSelection);
                        goto quitloop;
                    }
                }
            }
            quitloop:;
        }
    }
}

wxString asForecastManager::GetModelName(int i_fcst)
{
    wxString modelName = wxEmptyString;

    if (m_CurrentForecasts.size()==0) return wxEmptyString;

    wxASSERT(m_CurrentForecasts.size()>(unsigned)i_fcst);

    if(m_CurrentForecasts.size()>(unsigned)i_fcst)
    {
        modelName = m_CurrentForecasts[i_fcst]->GetModelName();
    }

    wxASSERT(!modelName.IsEmpty());

    return modelName;
}

VectorString asForecastManager::GetModelsNames()
{
    VectorString models;

    for (unsigned int i_model=0; i_model<m_CurrentForecasts.size(); i_model++)
    {
        models.push_back(m_CurrentForecasts[i_model]->GetModelName());
    }

    return models;
}

wxArrayString asForecastManager::GetModelsNamesWxArray()
{
    wxArrayString models;

    for (unsigned int i_model=0; i_model<m_CurrentForecasts.size(); i_model++)
    {
        models.Add(m_CurrentForecasts[i_model]->GetModelName());
    }

    return models;
}

VectorString asForecastManager::GetFilePaths()
{
    VectorString files;

    for (unsigned int i_file=0; i_file<m_CurrentForecasts.size(); i_file++)
    {
        files.push_back(m_CurrentForecasts[i_file]->GetFilePath());
    }

    return files;
}

wxArrayString asForecastManager::GetFilePathsWxArray()
{
    wxArrayString files;

    for (unsigned int i_file=0; i_file<m_CurrentForecasts.size(); i_file++)
    {
        files.Add(m_CurrentForecasts[i_file]->GetFilePath());
    }

    return files;
}

Array1DFloat asForecastManager::GetFullTargetDatesVector()
{
    double firstDate = 9999999999, lastDate = 0;

    for (unsigned int i_fcats=0; i_fcats<m_CurrentForecasts.size(); i_fcats++)
    {
        Array1DFloat fcastDates = m_CurrentForecasts[i_fcats]->GetTargetDates();
        if (fcastDates[0]<firstDate)
        {
            firstDate = fcastDates[0];
        }
        if (fcastDates[fcastDates.size()-1]>lastDate)
        {
            lastDate = fcastDates[fcastDates.size()-1];
        }
    }

    int size = asTools::Round(lastDate-firstDate+1);
    Array1DFloat dates = Array1DFloat::LinSpaced(size,firstDate,lastDate);

    return dates;
}

wxArrayString asForecastManager::GetStationNames(int i_fcst)
{
    wxArrayString stationNames;

    if (m_CurrentForecasts.size()==0) return stationNames;

    wxASSERT(m_CurrentForecasts.size()>(unsigned)i_fcst);

    if(m_CurrentForecasts.size()>(unsigned)i_fcst)
    {
        stationNames = m_CurrentForecasts[i_fcst]->GetStationNamesWxArrayString();
    }

    return stationNames;
}

wxString asForecastManager::GetStationName(int i_fcst, int i_stat)
{
    wxString stationName;

    if (m_CurrentForecasts.size()==0) return wxEmptyString;

    wxASSERT(m_CurrentForecasts.size()>(unsigned)i_fcst);

    if(m_CurrentForecasts.size()>(unsigned)i_fcst)
    {
        stationName = m_CurrentForecasts[i_fcst]->GetStationName(i_stat);
    }

    return stationName;
}

wxArrayString asForecastManager::GetStationNamesWithHeights(int i_fcst)
{
    wxArrayString stationNames;

    if (m_CurrentForecasts.size()==0) return stationNames;

    wxASSERT(m_CurrentForecasts.size()>(unsigned)i_fcst);

    if(m_CurrentForecasts.size()>(unsigned)i_fcst)
    {
        stationNames = m_CurrentForecasts[i_fcst]->GetStationNamesAndHeightsWxArrayString();
    }

    return stationNames;
}

wxString asForecastManager::GetStationNameWithHeight(int i_fcst, int i_stat)
{
    wxString stationName;

    if (m_CurrentForecasts.size()==0) return wxEmptyString;

    wxASSERT(m_CurrentForecasts.size()>(unsigned)i_fcst);

    if(m_CurrentForecasts.size()>(unsigned)i_fcst)
    {
        stationName = m_CurrentForecasts[i_fcst]->GetStationNameAndHeight(i_stat);
    }

    return stationName;
}

int asForecastManager::GetLeadTimeLength(int i_fcst)
{
    if (m_CurrentForecasts.size()==0) return 0;

    wxASSERT(m_CurrentForecasts.size()>(unsigned)i_fcst);

    int length = 0;

    if(m_CurrentForecasts.size()>(unsigned)i_fcst)
    {
        length = m_CurrentForecasts[i_fcst]->GetTargetDatesLength();
    }

    wxASSERT(length>0);

    return length;
}

wxArrayString asForecastManager::GetLeadTimes(int i_fcst)
{
    wxArrayString leadTimes;

    if (m_CurrentForecasts.size()==0) return leadTimes;

    wxASSERT(m_CurrentForecasts.size()>(unsigned)i_fcst);

    if(m_CurrentForecasts.size()>(unsigned)i_fcst)
    {
        Array1DFloat dates = m_CurrentForecasts[i_fcst]->GetTargetDates();

        for (int i=0; i<dates.size(); i++)
        {
            leadTimes.Add(asTime::GetStringTime(dates[i], "DD.MM.YYYY"));
        }
    }

    return leadTimes;
}
