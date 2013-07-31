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
 
#include "asCatalogPredictorsArchive.h"

#include "wx/fileconf.h"

#include <asFileXml.h>


asCatalogPredictorsArchive::asCatalogPredictorsArchive(const wxString &alternateFilePath)
:
asCatalogPredictors(alternateFilePath)
{
    // Get the xml file path
    if (m_CatalogFilePath.IsEmpty())
    {
        ThreadsManager().CritSectionConfig().Enter();
        m_CatalogFilePath = wxFileConfig::Get()->Read("/StandardPaths/CatalogPredictorsArchiveFilePath", asConfig::GetDefaultUserConfigDir() + "CatalogPredictorsArchive.xml");
        ThreadsManager().CritSectionConfig().Leave();
    }
}

asCatalogPredictorsArchive::~asCatalogPredictorsArchive()
{
    //dtor
}

bool asCatalogPredictorsArchive::Load(const wxString &DataSetId, const wxString &DataId)
{
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

    ThreadsManager().CritSectionTiCPP().Leave();
    return true;
}

bool asCatalogPredictorsArchive::LoadDatasetProp(const wxString &DataSetId)
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
        m_TimeZoneHours = xmlFile.GetFirstElementValueFloat("TimeZoneHours", NaNFloat);
        m_TimeStepHours = xmlFile.GetFirstElementValueFloat("TimeStepHours", NaNFloat);
        m_FirstTimeStepHour = xmlFile.GetFirstElementValueFloat("FirstTimeStepHour", NaNFloat);
        m_Start = ConvertStringToDatasetDate(xmlFile.GetFirstElementValueText("Start", wxEmptyString), asSERIE_BEGINNING, m_TimeZoneHours, m_TimeStepHours, m_FirstTimeStepHour);
        m_End = ConvertStringToDatasetDate(xmlFile.GetFirstElementValueText("End", wxEmptyString), asSERIE_END, m_TimeZoneHours, m_TimeStepHours, m_FirstTimeStepHour);
        m_FormatRaw = asGlobEnums::StringToFileFormatEnum(xmlFile.GetFirstElementValueText("FormatRaw", wxEmptyString));
        m_FormatStorage = asGlobEnums::StringToFileFormatEnum(xmlFile.GetFirstElementValueText("FormatStorage", wxEmptyString));
        m_DataPath = xmlFile.GetFirstElementValueText("Path", wxEmptyString);
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

bool asCatalogPredictorsArchive::LoadDataProp(const wxString &DataSetId, const wxString &DataId)
{
    wxString DataAccess;

    // Get datset information
    try
    {
        if(!asCatalogPredictorsArchive::LoadDatasetProp(DataSetId)) return false;
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
    m_Data.FileName = xmlFile.GetFirstElementValueText("FileName", wxEmptyString);
    m_Data.FileVarName = xmlFile.GetFirstElementValueText("FileVarName", wxEmptyString);
    m_Data.Parameter = asGlobEnums::StringToDataParameterEnum(xmlFile.GetFirstElementValueText("Parameter", wxEmptyString));
    m_Data.Unit = asGlobEnums::StringToDataUnitEnum(xmlFile.GetFirstElementValueText("Unit", wxEmptyString));
    m_Data.UaxisStep = xmlFile.GetFirstElementValueDouble("UaxisStep", NaNDouble);
    m_Data.VaxisStep = xmlFile.GetFirstElementValueDouble("VaxisStep", NaNDouble);
    m_Data.UaxisShift = xmlFile.GetFirstElementValueDouble("UaxisShift", 0);
    m_Data.VaxisShift = xmlFile.GetFirstElementValueDouble("VaxisShift", 0);

    // Reset the base path
    xmlFile.ClearCurrenNodePath();

    return true;
}
