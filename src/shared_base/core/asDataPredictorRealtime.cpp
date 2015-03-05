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
    m_runDateInUse = 0.0;
    m_restrictDownloads = false;
    m_restrictTimeHours = 0.0;
    m_restrictTimeStepHours = 0.0;
    m_forecastLeadTimeStart = 0.0;
    m_forecastLeadTimeEnd = 0.0;
    m_forecastLeadTimeStep = 0.0;
    m_runHourStart = 0.0;
    m_runUpdate = 0.0;
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
    wxASSERT(!m_predictorsRealtimeDirectory.IsEmpty());

    // Internet (cURL)
    asInternet internet;

    return internet.Download(GetUrls(), GetFileNames(), m_predictorsRealtimeDirectory);
}

bool asDataPredictorRealtime::CheckTimeArray(asTimeArray &timeArray)
{
    return true;
}

double asDataPredictorRealtime::UpdateRunDateInUse()
{
    m_fileNames.clear();
    m_urls.clear();

// TODO (Pascal#1#): Fix the use of m_timeZoneHours

    // Round time to the last available data
    double runHourStart = (double)m_runHourStart;
    double runUpdate = (double)m_runUpdate;
    double hourNow = (m_runDateInUse-floor(m_runDateInUse))*24;
    if (runUpdate>0)
    {
        double factorUpdate = floor((hourNow-runHourStart)/runUpdate);
        m_runDateInUse = floor(m_runDateInUse)+(factorUpdate*runUpdate)/(double)24;
    }
    else
    {
        m_runDateInUse = floor(m_runDateInUse)+runHourStart/(double)24;
    }

    return m_runDateInUse;
}

double asDataPredictorRealtime::SetRunDateInUse(double val)
{
    // Get date and time
    if(val==0)
    {
        val = asTime::NowMJD(asUTM);
    }

    m_runDateInUse = val;
    UpdateRunDateInUse();

    return m_runDateInUse;
}

double asDataPredictorRealtime::DecrementRunDateInUse()
{
    m_fileNames.clear();
    m_urls.clear();
    m_runDateInUse -= (double)m_runUpdate/(double)24;

    return m_runDateInUse;
}

void asDataPredictorRealtime::RestrictTimeArray(double restrictTimeHours, double restrictTimeStepHours)
{
    m_restrictDownloads = true;
    m_restrictTimeHours = restrictTimeHours;
    m_restrictTimeStepHours = restrictTimeStepHours;
    wxASSERT(m_restrictTimeStepHours>0);
    wxASSERT(m_restrictTimeHours>-100);
    wxASSERT(m_restrictTimeHours<100);
}

bool asDataPredictorRealtime::BuildFilenamesUrls()
{
    m_dataDates.clear();
    m_fileNames.clear();
    m_urls.clear();

    wxString thisCommand = m_commandDownload;

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
            wxString date = asTime::GetStringTime(m_runDateInUse, dateFormat);
            thisCommand.replace(posStart,date.Length(),date);
        }
    }

    // Restrict the downloads to used data
    if (m_restrictDownloads)
    {
        // Get the real lead time
        double dayRun = floor(m_runDateInUse);
        double desiredTime = dayRun+m_restrictTimeHours/24.0;
        double diff = desiredTime-m_runDateInUse;
        m_forecastLeadTimeStart = (int)(diff*24.0);
        m_forecastLeadTimeStep = m_restrictTimeStepHours;
        m_forecastLeadTimeEnd = floor(((double)m_forecastLeadTimeEnd-(double)m_forecastLeadTimeStart)/(double)m_forecastLeadTimeStep)*(double)m_forecastLeadTimeStep+m_forecastLeadTimeStart;
    }

    wxASSERT(m_forecastLeadTimeStep>0);
    wxASSERT(m_forecastLeadTimeEnd>=m_forecastLeadTimeStart);

    // Change the leadtimes
    for (int leadtime=m_forecastLeadTimeStart; leadtime<=m_forecastLeadTimeEnd; leadtime+=m_forecastLeadTimeStep)
    {
        int currentLeadtime = leadtime;
        double runDateInUse = m_runDateInUse;

        // Manage if ledtime if negative -> get previous download
        while(currentLeadtime<0)
        {
            currentLeadtime += m_runUpdate;
            runDateInUse -= (double)m_runUpdate/24.0;
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
        wxString ext = m_fileExtension;

        wxString filename = wxString::Format("%s.%s.%s.%s.%s",nowstr.c_str(),m_datasetId.c_str(),m_dataId.c_str(),leadtimestr.c_str(),ext.c_str());
        wxString filenameres = directory + DS + filename;

        double dataDate = runDateInUse+currentLeadtime/24.0;

        // Save resulting strings
        m_urls.push_back(thisCommandLeadTime);
        m_fileNames.push_back(filenameres);
        m_dataDates.push_back(dataDate);
    }

    wxASSERT(m_dataDates.size()==m_urls.size());
    wxASSERT(m_dataDates.size()==m_fileNames.size());

    return true;
}