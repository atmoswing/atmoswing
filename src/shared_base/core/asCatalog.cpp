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
 
#include "asCatalog.h"

#include "wx/fileconf.h"

#include <asTime.h>
#include <asFileXml.h>


asCatalog::asCatalog(const wxString &alternateFilePath)
{
    m_CatalogFilePath = alternateFilePath;

    // Initiate some data
    m_SetId = wxEmptyString;
    m_Name = wxEmptyString;
    m_Description = wxEmptyString;
    m_TimeZoneHours = 0;
    m_TimeStepHours = 0;
    m_FirstTimeStepHour = 0;
    m_DataPath = wxEmptyString;

}

asCatalog::~asCatalog()
{
    //dtor
}

double asCatalog::ConvertStringToDatasetDate(const wxString &date_s, int InSerie, float TimeZone, double TimeStepHours, double FirstHour)
{
    if (date_s.IsEmpty()) return NaNDouble;

    // Convert the string into a date
    double date = asTime::GetTimeFromString(date_s, guess);

    // Change units to work in MJD
    double TimeStepDays = TimeStepHours/24;
    double FirstHourDays = FirstHour/24;

    // Add the timezone
    date -= TimeZone/24;

    // Get the day
    int day = floor(date);
    double hour = date-day;

    // Make the date match the dataset definition
    int hourratio = 0;
    if(InSerie == asSERIE_BEGINNING)
    {
        hourratio = ceil((hour-FirstHourDays)/TimeStepDays);
    } else {
        hourratio = floor((hour-FirstHourDays)/TimeStepDays);
    }

    // Build up the final date
    double resdate = day;
    resdate += FirstHourDays + hourratio*TimeStepDays;

    return resdate;
}

asCatalog::DatasetIdList asCatalog::GetDatasetIdList(DataPurpose Purpose, const wxString &AlternateFilePath)
{
    DatasetIdList SetList;
    wxString Filename;

    // Get the xml file path
    if (!AlternateFilePath.IsEmpty())
    {
        Filename = AlternateFilePath;
    }
    else
    {
        ThreadsManager().CritSectionConfig().Enter();
        if (Purpose==Predictand)
        {
            asLogError(_("asCatalog::GetDatasetIdList should not be called any more for predictand data, as there is supposed to be only 1 catalog per dataset."));
            return SetList;
        }
        else if (Purpose==PredictorArchive)
        {
            Filename = wxFileConfig::Get()->Read("/StandardPaths/CatalogPredictorsArchiveFilePath", asConfig::GetDefaultUserConfigDir() + "CatalogPredictorsArchive.xml");
        }
        else if (Purpose==PredictorRealtime)
        {
            Filename = wxFileConfig::Get()->Read("/StandardPaths/CatalogPredictorsRealtimeFilePath", asConfig::GetDefaultUserConfigDir() + "CatalogPredictorsRealtime.xml");
        }
        ThreadsManager().CritSectionConfig().Leave();
    }

    // Load xml file
    asFileXml xmlFile( Filename, asFile::ReadOnly );
    if(!xmlFile.Open()) return SetList;

    // XML struct for the dataset information
    wxString DatasetAccess = "AtmoSwingFile.DataSet";

    // Get the first dataset
    if(xmlFile.GetFirstElementAttributeValueBool(DatasetAccess, "enable", true))
    {
        SetList.Id.push_back(xmlFile.GetFirstElementAttributeValueText(DatasetAccess, "id", wxEmptyString));
        SetList.Name.push_back(xmlFile.GetFirstElementAttributeValueText(DatasetAccess, "name", wxEmptyString));
        SetList.Description.push_back(xmlFile.GetFirstElementAttributeValueText(DatasetAccess, "description", wxEmptyString));
    }

    // Get the other datasets
    while(xmlFile.GetNextElement(DatasetAccess))
    {
        if(xmlFile.GetThisElementAttributeValueBool("enable", true))
        {
            SetList.Id.push_back(xmlFile.GetThisElementAttributeValueText("id", wxEmptyString));
            SetList.Name.push_back(xmlFile.GetThisElementAttributeValueText("name", wxEmptyString));
            SetList.Description.push_back(xmlFile.GetThisElementAttributeValueText("description", wxEmptyString));
        }
    }

    return SetList;

}

asCatalog::DataIdListStr asCatalog::GetDataIdListStr(DataPurpose Purpose, const wxString &DataSetId, const wxString &AlternateFilePath)
{
    DataIdListStr DataList;
    wxString Filename;

    // Get the xml file path
    if (!AlternateFilePath.IsEmpty())
    {
        Filename = AlternateFilePath;
    }
    else
    {
        ThreadsManager().CritSectionConfig().Enter();
        if (Purpose==Predictand)
        {
            asLogError(_("You must provide the path to the predictand catalog."));
        }
        else if (Purpose==PredictorArchive)
        {
            Filename = wxFileConfig::Get()->Read("/StandardPaths/CatalogPredictorsArchiveFilePath", asConfig::GetDefaultUserConfigDir() + "CatalogPredictorsArchive.xml");
        }
        else if (Purpose==PredictorRealtime)
        {
            Filename = wxFileConfig::Get()->Read("/StandardPaths/CatalogPredictorsRealtimeFilePath", asConfig::GetDefaultUserConfigDir() + "CatalogPredictorsRealtime.xml");
        }
        ThreadsManager().CritSectionConfig().Leave();
    }

    // Load xml file
    asFileXml xmlFile( Filename, asFile::ReadOnly );
    if(!xmlFile.Open()) return DataList;

    // XML struct for the dataset information
    wxString DataAccess = wxString::Format("AtmoSwingFile.DataSet[%s].DataList.Data", DataSetId.c_str());

    // Get the first data
    if(xmlFile.GetFirstElementAttributeValueBool(DataAccess, "enable", true))
    {
        DataList.Id.push_back(xmlFile.GetFirstElementAttributeValueText(DataAccess, "id", wxEmptyString));
        DataList.Name.push_back(xmlFile.GetFirstElementAttributeValueText(DataAccess, "name", wxEmptyString));
    }

    // Get the other datasets
    while(xmlFile.GetNextElement(DataAccess))
    {
        if(xmlFile.GetThisElementAttributeValueBool("enable", true))
        {
            DataList.Id.push_back(xmlFile.GetThisElementAttributeValueText("id", wxEmptyString));
            DataList.Name.push_back(xmlFile.GetThisElementAttributeValueText("name", wxEmptyString));
        }
    }

    return DataList;

}

asCatalog::DataIdListInt asCatalog::GetDataIdListInt(DataPurpose Purpose, const wxString &DataSetId, const wxString &AlternateFilePath)
{
    DataIdListInt DataList;
    wxString Filename;

    // Get the xml file path
    if (!AlternateFilePath.IsEmpty())
    {
        Filename = AlternateFilePath;
    }
    else
    {
        ThreadsManager().CritSectionConfig().Enter();
        if (Purpose==Predictand)
        {
            asLogError(_("You must provide the path to the predictand catalog."));
        }
        else if (Purpose==PredictorArchive)
        {
            Filename = wxFileConfig::Get()->Read("/StandardPaths/CatalogPredictorsArchiveFilePath", asConfig::GetDefaultUserConfigDir() + "CatalogPredictorsArchive.xml");
        }
        else if (Purpose==PredictorRealtime)
        {
            Filename = wxFileConfig::Get()->Read("/StandardPaths/CatalogPredictorsRealtimeFilePath", asConfig::GetDefaultUserConfigDir() + "CatalogPredictorsRealtime.xml");
        }
        ThreadsManager().CritSectionConfig().Leave();
    }

    // Load xml file
    asFileXml xmlFile( Filename, asFile::ReadOnly );
    if(!xmlFile.Open()) return DataList;

    // XML struct for the dataset information
    if (Purpose==Predictand)
    {
        wxString DataAccess = "AtmoSwingFile.DataSet.DataList.Data";
        
        // Get the first data
        DataList.Id.push_back(xmlFile.GetFirstElementAttributeValueInt(DataAccess, "id", 0));
        DataList.Name.push_back(xmlFile.GetFirstElementAttributeValueText(DataAccess, "name", wxEmptyString));

        // Get the other datasets
        while(xmlFile.GetNextElement(DataAccess))
        {
            DataList.Id.push_back(xmlFile.GetThisElementAttributeValueInt("id", 0));
            DataList.Name.push_back(xmlFile.GetThisElementAttributeValueText("name", wxEmptyString));
        }
    }
    else
    {
        wxString DataAccess = wxString::Format("AtmoSwingFile.DataSet[%s].DataList.Data", DataSetId.c_str());
        
        // Get the first data
        if(xmlFile.GetFirstElementAttributeValueBool(DataAccess, "enable", true))
        {
            DataList.Id.push_back(xmlFile.GetFirstElementAttributeValueInt(DataAccess, "id", 0));
            DataList.Name.push_back(xmlFile.GetFirstElementAttributeValueText(DataAccess, "name", wxEmptyString));
        }

        // Get the other datasets
        while(xmlFile.GetNextElement(DataAccess))
        {
            if(xmlFile.GetThisElementAttributeValueBool("enable", true))
            {
                DataList.Id.push_back(xmlFile.GetThisElementAttributeValueInt("id", 0));
                DataList.Name.push_back(xmlFile.GetThisElementAttributeValueText("name", wxEmptyString));
            }
        }
    }


    return DataList;

}

wxArrayString asCatalog::GetDataIdListwxArrayString(DataPurpose Purpose, const wxString &DataSetId, const wxString &AlternateFilePath)
{
    asCatalog::DataIdListInt list = asCatalog::GetDataIdListInt(Purpose, DataSetId, AlternateFilePath);

    // Extract data
    wxArrayString array;
    for (unsigned int i=0; i<list.Name.size(); i++)
    {
        if (!list.Name[i].IsEmpty())
        {
            array.Add(list.Name[i]);
        }

    }

    return array;
}
