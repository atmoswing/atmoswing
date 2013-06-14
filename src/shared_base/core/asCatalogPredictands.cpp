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
 
#include "asCatalogPredictands.h"

#include "wx/fileconf.h"

#include <asFileXml.h>


asCatalogPredictands::asCatalogPredictands(const wxString &alternateFilePath)
:
asCatalog(alternateFilePath)
{
    // Get the xml file path
    if (m_CatalogFilePath.IsEmpty())
    {
        ThreadsManager().CritSectionConfig().Enter();
        m_CatalogFilePath = wxFileConfig::Get()->Read("/StandardPaths/CatalogPredictandsFilePath", asConfig::GetDefaultUserConfigDir() + "CatalogPredictands.xml");
        ThreadsManager().CritSectionConfig().Leave();
    }

    // Initiate some data
    m_Station.Id = 0;
    m_Station.Name = wxEmptyString;
    m_Station.Filename = wxEmptyString;
    m_Station.Filepattern = wxEmptyString;
}

asCatalogPredictands::~asCatalogPredictands()
{
    //dtor
}

bool asCatalogPredictands::Load(const wxString &DataSetId, int StationId)
{
    ThreadsManager().CritSectionTiCPP().Enter();

    // Get data from file
    if (StationId)
    {
        if(!LoadDataProp(DataSetId, StationId))
        {
            ThreadsManager().CritSectionTiCPP().Leave();
            return false;
        }
    }
    else
    {
        if(!LoadDatasetProp(DataSetId))
        {
            ThreadsManager().CritSectionTiCPP().Leave();
            return false;
        }
    }

    ThreadsManager().CritSectionTiCPP().Leave();
    return true;
}

bool asCatalogPredictands::LoadDatasetProp(const wxString &DataSetId)
{
    wxString DatasetAccess;

    // Load xml file
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
    m_Parameter = asGlobEnums::StringToDataParameterEnum(xmlFile.GetFirstElementValueText("Parameter"));
    m_Unit = asGlobEnums::StringToDataUnitEnum(xmlFile.GetFirstElementValueText("Unit"));
    m_TimeZoneHours = xmlFile.GetFirstElementValueFloat("TimeZoneHours");
    m_TimeStepHours = xmlFile.GetFirstElementValueFloat("TimeStepHours");
    m_FirstTimeStepHour = xmlFile.GetFirstElementValueFloat("FirstTimeStepHour");
    m_Start = ConvertStringToDatasetDate(xmlFile.GetFirstElementValueText("Start"), asSERIE_BEGINNING, m_TimeZoneHours, m_TimeStepHours, m_FirstTimeStepHour);
    m_End = ConvertStringToDatasetDate(xmlFile.GetFirstElementValueText("End"), asSERIE_END, m_TimeZoneHours, m_TimeStepHours, m_FirstTimeStepHour);
    m_FormatRaw = asGlobEnums::StringToFileFormatEnum(xmlFile.GetFirstElementValueText("FormatRaw"));
    m_FormatStorage = asGlobEnums::StringToFileFormatEnum(xmlFile.GetFirstElementValueText("FormatStorage"));
    m_DataPath = xmlFile.GetFirstElementValueText("Path");
    m_CoordSys = asGlobEnums::StringToCoordSysEnum(xmlFile.GetFirstElementValueText("CoordinateSys"));
    m_Nan.push_back(xmlFile.GetFirstElementValueDouble("NaN", -9999));
    while(xmlFile.GetNextElement("NaN"))
    {
        m_Nan.push_back(xmlFile.GetThisElementValueDouble(-9999));
    }
    wxString tmpDBInclude = xmlFile.GetFirstElementValueText("IncludeinDB");
    m_DBInclude.push_back(PredictandDB(asGlobEnums::StringToPredictandDBEnum(tmpDBInclude)));
    while(xmlFile.GetNextElement("IncludeinDB"))
    {
        tmpDBInclude = xmlFile.GetFirstElementValueText(wxEmptyString);
        m_DBInclude.push_back(PredictandDB(asGlobEnums::StringToPredictandDBEnum(tmpDBInclude)));
    }

    // Reset the base path
    xmlFile.ClearCurrenNodePath();

    return true;
}

bool asCatalogPredictands::LoadDataProp(const wxString &DataSetId, int StationId)
{
    // Initiate the check of the timezone
    wxString DataAccess;

    // Get datset information
    try
    {
        if(!asCatalogPredictands::LoadDatasetProp(DataSetId)) return false;
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
    DataAccess = wxString::Format("AtmoswingFile.DataSet[%s].DataList.Data[%d]",DataSetId.c_str(),StationId);
    if(!xmlFile.GoToFirstNodeWithPath(DataAccess)) return false;

    // Get the data information
    m_Station.Id = StationId;
    m_Station.Name = xmlFile.GetFirstElementAttributeValueText(wxEmptyString, "name");
    m_Station.LocalId = xmlFile.GetFirstElementValueText("LocalID");
    m_Station.Coord.u = xmlFile.GetFirstElementValueDouble("XCoordinate", -1);
    m_Station.Coord.v = xmlFile.GetFirstElementValueDouble("YCoordinate", -1);
    m_Station.Height = xmlFile.GetFirstElementValueFloat("Height", -1);
    m_Station.Filename = xmlFile.GetFirstElementValueText("FileName");
    m_Station.Filepattern = xmlFile.GetFirstElementValueText("FilePattern");
    m_Station.Start = ConvertStringToDatasetDate(xmlFile.GetFirstElementValueText("Start"), true, m_TimeZoneHours, m_TimeStepHours, m_FirstTimeStepHour);
    m_Station.End = ConvertStringToDatasetDate(xmlFile.GetFirstElementValueText("End"), false, m_TimeZoneHours, m_TimeStepHours, m_FirstTimeStepHour);

    // Reset the base path
    xmlFile.ClearCurrenNodePath();

    return true;
}
