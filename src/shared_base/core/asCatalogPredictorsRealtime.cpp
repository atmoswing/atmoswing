/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
#include "asCatalogPredictorsRealtime.h"

#include "wx/fileconf.h"

#include <asFileXml.h>
#include <asThreadsManager.h>


asCatalogPredictorsRealtime::asCatalogPredictorsRealtime(const wxString &alternateFilePath)
:
asCatalogPredictors(alternateFilePath)
{
    // Get the xml file path
    if (m_CatalogFilePath.IsEmpty())
    {
        ThreadsManager().CritSectionConfig().Enter();
        m_CatalogFilePath = wxFileConfig::Get()->Read("/StandardPaths/CatalogPredictorsRealtimeFilePath", asConfig::GetDefaultUserConfigDir() + "CatalogPredictorsRealtime.xml");
        ThreadsManager().CritSectionConfig().Leave();
    }

    m_ForecastLeadTimeStart = 0;
    m_ForecastLeadTimeEnd = 0;
    m_ForecastLeadTimeStep = 0;
    m_RunHourStart = 0;
    m_RunUpdate = 0;
    m_RunDateInUse = 0;
    m_RestrictDownloads = false;
    m_RestrictDTimeHours = 0;
    m_RestrictTimeStepHours = 24;
}

asCatalogPredictorsRealtime::~asCatalogPredictorsRealtime()
{
    //dtor
}

double asCatalogPredictorsRealtime::UpdateRunDateInUse()
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

double asCatalogPredictorsRealtime::SetRunDateInUse(double val)
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

double asCatalogPredictorsRealtime::DecrementRunDateInUse()
{
    m_FileNames.clear();
    m_Urls.clear();
    m_RunDateInUse -= (double)m_RunUpdate/(double)24;

    return m_RunDateInUse;
}

bool asCatalogPredictorsRealtime::Load(const wxString &DataSetId, const wxString &DataId)
{
    m_RestrictDownloads = false;

    ThreadsManager().CritSectionTiCPP().Enter();

    // Get data from file
    if (!DataId.IsEmpty())
    {
        if(!LoadDataProp(DataSetId, DataId))
        {
            ThreadsManager().CritSectionTiCPP().Leave();
            return false;
        }
    } else {
        if(!LoadDatasetProp(DataSetId))
        {
            ThreadsManager().CritSectionTiCPP().Leave();
            return false;
        }
    }

    if (m_RunDateInUse==0)
    {
        SetRunDateInUse();
    }
    UpdateRunDateInUse();
    BuildFilenamesUrls();

    ThreadsManager().CritSectionTiCPP().Leave();
    return true;
}

bool asCatalogPredictorsRealtime::LoadDatasetProp(const wxString &DataSetId)
{
    wxString DatasetAccess;

    // Load xml file
    try
    {
        asFileXml xmlFile( m_CatalogFilePath, asFile::ReadOnly );
        if(!xmlFile.Open()) return false;

        // XML struct for the dataset information
        DatasetAccess = wxString::Format("AtmoswingFile.DataSet[%s]", DataSetId.c_str());
        if(!xmlFile.GoToFirstNodeWithPath(DatasetAccess)) return false;

        // Get the data set informations
        m_SetId = DataSetId;
        m_Name = xmlFile.GetFirstElementAttributeValueText(wxEmptyString, "name");
        m_Description = xmlFile.GetFirstElementAttributeValueText(wxEmptyString, "description");

        // Get dataset information
        m_ForecastLeadTimeStart = xmlFile.GetFirstElementValueInt("ForecastLeadTimeStart", 0);
        m_ForecastLeadTimeEnd = xmlFile.GetFirstElementValueInt("ForecastLeadTimeEnd", 0);
        m_ForecastLeadTimeStep = xmlFile.GetFirstElementValueInt("ForecastLeadTimeStep", 0);
        m_RunHourStart = xmlFile.GetFirstElementValueInt("RunHourStart", 0);
        m_RunUpdate = xmlFile.GetFirstElementValueInt("RunUpdate", 0);
        m_TimeZoneHours = xmlFile.GetFirstElementValueFloat("TimeZoneHours", NaNFloat);
        m_TimeStepHours = xmlFile.GetFirstElementValueFloat("TimeStepHours", NaNFloat);
        m_FirstTimeStepHour = xmlFile.GetFirstElementValueFloat("FirstTimeStepHour", NaNFloat);
        m_Start = ConvertStringToDatasetDate(xmlFile.GetFirstElementValueText("Start", wxEmptyString), asSERIE_BEGINNING, m_TimeZoneHours, m_TimeStepHours, m_FirstTimeStepHour);
        m_End = ConvertStringToDatasetDate(xmlFile.GetFirstElementValueText("End", wxEmptyString), asSERIE_END, m_TimeZoneHours, m_TimeStepHours, m_FirstTimeStepHour);
        m_FormatRaw = asGlobEnums::StringToFileFormatEnum(xmlFile.GetFirstElementValueText("FormatRaw", wxEmptyString));
        m_FormatStorage = asGlobEnums::StringToFileFormatEnum(xmlFile.GetFirstElementValueText("FormatStorage", wxEmptyString));
        m_DataPath = wxEmptyString;
        m_Website = xmlFile.GetFirstElementValueText("Website", wxEmptyString);
        m_Ftp = xmlFile.GetFirstElementValueText("Ftp", wxEmptyString);
        m_CoordSys = asGlobEnums::StringToCoordSysEnum(xmlFile.GetFirstElementValueText("CoordinateSys", wxEmptyString));
        m_Nan.push_back(xmlFile.GetFirstElementValueDouble("NaN", NaNDouble));
        while(xmlFile.GetNextElement("NaN"))
        {
            m_Nan.push_back(xmlFile.GetThisElementValueDouble(NaNDouble));
        }

        // Reset the base path
        xmlFile.ClearCurrenNodePath();
    }
    catch(asException& e)
    {
        asLogError(e.GetFullMessage());
        asLogError(_("Failed to parse the catalog file."));
        return false;
    }

    return true;
}

bool asCatalogPredictorsRealtime::LoadDataProp(const wxString &DataSetId, const wxString &DataId)
{
    wxString DataAccess;

    // Get datset information
    try
    {
        if(!asCatalogPredictorsRealtime::LoadDatasetProp(DataSetId)) return false;
    }
    catch(asException& e)
    {
        asLogError(e.GetFullMessage());
        asLogError(_("Failed to load the catalog properties."));
        return false;
    }

    // Load xml file
    asFileXml xmlFile( m_CatalogFilePath, asFile::ReadOnly );
    if(!xmlFile.Open()) return false;

    // XML struct for the dataset information
    DataAccess = wxString::Format("AtmoswingFile.DataSet[%s].DataList.Data[%s]",DataSetId.c_str(),DataId.c_str());
    if(!xmlFile.GoToFirstNodeWithPath(DataAccess)) return false;

    // Get the data information
    m_Data.Id = DataId;
    m_Data.Name = xmlFile.GetFirstElementAttributeValueText(wxEmptyString, "name");
    m_Data.FileLength = asGlobEnums::StringToFileLengthEnum(xmlFile.GetFirstElementValueText("FileLength", wxEmptyString));
    m_Data.FileVarName = xmlFile.GetFirstElementValueText("FileVarName", wxEmptyString);
    m_CommandDownload = xmlFile.GetFirstElementValueText("CommandDownload", wxEmptyString);
    m_Data.Parameter = asGlobEnums::StringToDataParameterEnum(xmlFile.GetFirstElementValueText("Parameter", wxEmptyString));
    m_Data.Unit = asGlobEnums::StringToDataUnitEnum(xmlFile.GetFirstElementValueText("Unit", wxEmptyString));
    m_Data.UaxisStep = xmlFile.GetFirstElementValueDouble("UaxisStep", NaNDouble);
    m_Data.VaxisStep = xmlFile.GetFirstElementValueDouble("VaxisStep", NaNDouble);

    // Reset the base path
    xmlFile.ClearCurrenNodePath();

    return true;
}

void asCatalogPredictorsRealtime::RestrictTimeArray(double restrictDTimeHours, double restrictTimeStepHours)
{
    m_RestrictDownloads = true;
    m_RestrictDTimeHours = restrictDTimeHours;
    m_RestrictTimeStepHours = restrictTimeStepHours;
    wxASSERT(m_RestrictTimeStepHours>0);
    wxASSERT(m_RestrictDTimeHours>-100);
    wxASSERT(m_RestrictDTimeHours<100);
}

bool asCatalogPredictorsRealtime::BuildFilenamesUrls()
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
        double desiredTime = dayRun+m_RestrictDTimeHours/24.0;
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
        wxString datasetid = m_SetId;
        wxString dataid = m_Data.Id;
        wxString nowstr = asTime::GetStringTime(runDateInUse, "YYYYMMDDhh");
        wxString leadtimestr = timeStrFileName;
        wxString ext = asGlobEnums::FileFormatEnumToExtension(m_FormatRaw);

        wxString filename = wxString::Format("%s.%s.%s.%s.%s",nowstr.c_str(),datasetid.c_str(),dataid.c_str(),leadtimestr.c_str(),ext.c_str());
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
