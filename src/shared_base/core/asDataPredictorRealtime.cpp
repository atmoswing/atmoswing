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
 
#include "asDataPredictorRealtime.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileNetcdf.h>
#include <asDataPredictorRealtimeGfsForecast.h>
#include <asInternet.h>


asDataPredictorRealtime::asDataPredictorRealtime(const wxString &dataId)
:
asDataPredictor(dataId)
{
    m_RunDateInUse = 0.0;
    m_RestrictDownloads = false;
    m_RestrictTimeHours = 0.0;
    m_RestrictTimeStepHours = 0.0;
    m_ForecastLeadTimeStart = 0.0;
    m_ForecastLeadTimeEnd = 0.0;
    m_ForecastLeadTimeStep = 0.0;
    m_RunHourStart = 0.0;
    m_RunUpdate = 0.0;
}

asDataPredictorRealtime::~asDataPredictorRealtime()
{

}

asDataPredictorRealtime* asDataPredictorRealtime::GetInstance(const wxString &datasetId, const wxString &dataId)
{
    asDataPredictorRealtime* predictor = NULL;

    if (datasetId.IsSameAs("NWS_GFS_Forecast", false))
    {
        predictor = new asDataPredictorRealtimeGfsForecast(dataId);
    }
    else
    {
        asLogError(_("The requested dataset does not exist. Please correct the dataset Id."));
        return NULL;
    }

    if(!predictor->Init())
    {
        asLogError(_("The predictor did not initialize correctly."));
    }

    return predictor;
}

bool asDataPredictorRealtime::Init()
{
    return false;
}

int asDataPredictorRealtime::Download()
{
    // Directory
    wxConfigBase *pConfig = wxFileConfig::Get();
    wxString realtimePredictorSavingDir = pConfig->Read("/Paths/RealtimePredictorSavingDir", wxEmptyString);

    // Internet (cURL)
    asInternet internet;

    return internet.Download(GetUrls(), GetFileNames(), realtimePredictorSavingDir);
}

bool asDataPredictorRealtime::LoadFullArea(double date, float level)
{
    asTimeArray timeArray(date, asTimeArray::SingleDay);
    timeArray.Init();
    m_Level = level;

    return Load(NULL, timeArray);
}

bool asDataPredictorRealtime::Load(asGeoAreaCompositeGrid &desiredArea, asTimeArray &timeArray)
{
    return Load(&desiredArea, timeArray);
}

bool asDataPredictorRealtime::Load(asGeoAreaCompositeGrid *desiredArea, asTimeArray &timeArray)
{
    return false;
}

double asDataPredictorRealtime::UpdateRunDateInUse()
{
    m_FileNames.clear();
    m_Urls.clear();

// TODO (Pascal#1#): Fix the use of m_TimeZoneHours

    // Round time to the last available data
    double runHourStart = (double)m_RunHourStart;
    double runUpdate = (double)m_RunUpdate;
    double hourNow = (m_RunDateInUse-floor(m_RunDateInUse))*24;
    if (runUpdate>0)
    {
        double factorUpdate = floor((hourNow-runHourStart)/runUpdate);
        m_RunDateInUse = floor(m_RunDateInUse)+(factorUpdate*runUpdate)/(double)24;
    }
    else
    {
        m_RunDateInUse = floor(m_RunDateInUse)+runHourStart/(double)24;
    }

    return m_RunDateInUse;
}

double asDataPredictorRealtime::SetRunDateInUse(double val)
{
    // Get date and time
    if(val==0)
    {
        val = asTime::NowMJD(asUTM);
    }

    m_RunDateInUse = val;
    UpdateRunDateInUse();

    return m_RunDateInUse;
}

double asDataPredictorRealtime::DecrementRunDateInUse()
{
    m_FileNames.clear();
    m_Urls.clear();
    m_RunDateInUse -= (double)m_RunUpdate/(double)24;

    return m_RunDateInUse;
}

void asDataPredictorRealtime::RestrictTimeArray(double restrictTimeHours, double restrictTimeStepHours)
{
    m_RestrictDownloads = true;
    m_RestrictTimeHours = restrictTimeHours;
    m_RestrictTimeStepHours = restrictTimeStepHours;
    wxASSERT(m_RestrictTimeStepHours>0);
    wxASSERT(m_RestrictTimeHours>-100);
    wxASSERT(m_RestrictTimeHours<100);
}

bool asDataPredictorRealtime::BuildFilenamesUrls()
{
    m_DataDates.clear();
    m_FileNames.clear();
    m_Urls.clear();

    wxString thisCommand = m_CommandDownload;

    // Replace time in the command
    while (thisCommand.Find("CURRENTDATE")!=wxNOT_FOUND )
    {
        int posStart = thisCommand.Find("CURRENTDATE");
        posStart--;
        thisCommand.Remove(posStart,13); // Removes '[CURRENTDATE-'
        // Find end
        int posEnd = thisCommand.find("]", posStart);

        if(posEnd!=wxNOT_FOUND && posEnd>posStart)
        {
            thisCommand.Remove(posEnd,1); // Removes ']'
            wxString dateFormat = thisCommand.SubString(posStart, posEnd);
            wxString date = asTime::GetStringTime(m_RunDateInUse, dateFormat);
            thisCommand.replace(posStart,date.Length(),date);
        }
    }

    // Restrict the downloads to used data
    if (m_RestrictDownloads)
    {
        // Get the real lead time
        double dayRun = floor(m_RunDateInUse);
        double desiredTime = dayRun+m_RestrictTimeHours/24.0;
        double diff = desiredTime-m_RunDateInUse;
        m_ForecastLeadTimeStart = (int)(diff*24.0);
        m_ForecastLeadTimeStep = m_RestrictTimeStepHours;
        m_ForecastLeadTimeEnd = floor(((double)m_ForecastLeadTimeEnd-(double)m_ForecastLeadTimeStart)/(double)m_ForecastLeadTimeStep)*(double)m_ForecastLeadTimeStep+m_ForecastLeadTimeStart;
    }

    wxASSERT(m_ForecastLeadTimeStep>0);
    wxASSERT(m_ForecastLeadTimeEnd>=m_ForecastLeadTimeStart);

    // Change the leadtimes
    for (int leadtime=m_ForecastLeadTimeStart; leadtime<=m_ForecastLeadTimeEnd; leadtime+=m_ForecastLeadTimeStep)
    {
        int currentLeadtime = leadtime;
        double runDateInUse = m_RunDateInUse;

        // Manage if ledtime if negative -> get previous download
        while(currentLeadtime<0)
        {
            currentLeadtime += m_RunUpdate;
            runDateInUse -= (double)m_RunUpdate/24.0;
        }

        wxString thisCommandLeadTime = thisCommand;

        wxString timeStr = wxString::Format("%d", currentLeadtime);
        wxString timeStrFileName = wxEmptyString;

        thisCommandLeadTime.Replace("[LEADTIME-H]", timeStr);
        thisCommandLeadTime.Replace("[LEADTIME-h]", timeStr);
        if (timeStr.Length()<2) timeStr = "0"+timeStr;
        thisCommandLeadTime.Replace("[LEADTIME-HH]", timeStr);
        thisCommandLeadTime.Replace("[LEADTIME-hh]", timeStr);
        if (timeStr.Length()<3) timeStr = "0"+timeStr;
        timeStrFileName = timeStr;
        thisCommandLeadTime.Replace("[LEADTIME-HHH]", timeStr);
        thisCommandLeadTime.Replace("[LEADTIME-hhh]", timeStr);
        if (timeStr.Length()<4) timeStr = "0"+timeStr;
        thisCommandLeadTime.Replace("[LEADTIME-HHHH]", timeStr);
        thisCommandLeadTime.Replace("[LEADTIME-hhhh]", timeStr);

        // Filename
        wxString dirstructure = "YYYY";
        dirstructure.Append(DS);
        dirstructure.Append("MM");
        dirstructure.Append(DS);
        dirstructure.Append("DD");
        wxString directory = asTime::GetStringTime(runDateInUse, dirstructure);
        wxString nowstr = asTime::GetStringTime(runDateInUse, "YYYYMMDDhh");
        wxString leadtimestr = timeStrFileName;
        wxString ext = asGlobEnums::FileFormatEnumToExtension(m_FileFormat);

        wxString filename = wxString::Format("%s.%s.%s.%s.%s",nowstr.c_str(),m_DatasetId.c_str(),m_DataId.c_str(),leadtimestr.c_str(),ext.c_str());
        wxString filenameres = directory + DS + filename;

        double dataDate = runDateInUse+currentLeadtime/24.0;

        // Save resulting strings
        m_Urls.push_back(thisCommandLeadTime);
        m_FileNames.push_back(filenameres);
        m_DataDates.push_back(dataDate);
    }

    wxASSERT(m_DataDates.size()==m_Urls.size());
    wxASSERT(m_DataDates.size()==m_FileNames.size());

    return true;
}