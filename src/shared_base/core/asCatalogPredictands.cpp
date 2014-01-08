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
 
#include "asCatalogPredictands.h"

#include "wx/fileconf.h"

#include <asFileXml.h>


asCatalogPredictands::asCatalogPredictands(const wxString &filePath)
:
asCatalog(filePath)
{
    m_FormatStorage = netcdf;

    // Get the xml file path
    if (m_CatalogFilePath.IsEmpty())
    {
        asLogError(_("No path was given for the predictand catalog."));
    }

    // Initiate some data
    m_Station.Id = 0;
    m_Station.Name = wxEmptyString;
    m_Station.Filename = wxEmptyString;
    m_Station.Filepattern = wxEmptyString;
    m_DataPath = wxEmptyString;
}

asCatalogPredictands::~asCatalogPredictands()
{
    //dtor
}

bool asCatalogPredictands::Load(int StationId)
{
    ThreadsManager().CritSectionTiCPP().Enter();

    // Get data from file
    if (StationId>0)
    {
        if(!LoadDataProp(StationId))
        {
            ThreadsManager().CritSectionTiCPP().Leave();
            return false;
        }
    }
    else
    {
        if(!LoadDatasetProp())
        {
            ThreadsManager().CritSectionTiCPP().Leave();
            return false;
        }
    }

    ThreadsManager().CritSectionTiCPP().Leave();
    return true;
}

bool asCatalogPredictands::LoadDatasetProp()
{
    wxString DatasetAccess;

    // Load xml file
    asFileXml xmlFile( m_CatalogFilePath, asFile::ReadOnly );
    if(!xmlFile.Open()) return false;

    // XML struct for the dataset information
    DatasetAccess = wxString::Format("AtmoSwingFile.DataSet");
    if(!xmlFile.GoToFirstNodeWithPath(DatasetAccess)) return false;

    // Get the data set informations
    m_SetId = xmlFile.GetFirstElementAttributeValueText(wxEmptyString, "id");;
    m_Name = xmlFile.GetFirstElementAttributeValueText(wxEmptyString, "name");
    m_Description = xmlFile.GetFirstElementAttributeValueText(wxEmptyString, "description");

    // Get dataset information
    m_Parameter = asGlobEnums::StringToDataParameterEnum(xmlFile.GetFirstElementValueText("Parameter"));
    m_Unit = asGlobEnums::StringToDataUnitEnum(xmlFile.GetFirstElementValueText("Unit"));
    m_TemporalResolution = asGlobEnums::StringToDataTemporalResolutionEnum(xmlFile.GetFirstElementValueText("TemporalResolution"));
    m_SpatialAggregation = asGlobEnums::StringToDataSpatialAggregationEnum(xmlFile.GetFirstElementValueText("SpatialAggregation"));

    // Get the timestep
    switch (m_TemporalResolution)
    {
        case (Daily):
            m_TimeStepHours = 24.0;
            break;
        case (SixHourly):
            m_TimeStepHours = 6.0;
            break;
        case (Hourly):
            m_TimeStepHours = 1.0;
            break;
        case (SixHourlyMovingDailyTemporalWindow):
            m_TimeStepHours = 6.0;
            break;
        case (TwoDays):
            m_TimeStepHours = 48.0;
            break;
        case (ThreeDays):
            m_TimeStepHours = 72.0;
            break;
        case (Weekly):
            m_TimeStepHours = 168.0;
            break;
        default:
            wxFAIL;
            m_TimeStepHours = 0;
    }
    
    // Get other dataset information
    m_TimeZoneHours = xmlFile.GetFirstElementValueFloat("TimeZoneHours");
    m_FirstTimeStepHour = xmlFile.GetFirstElementValueFloat("FirstTimeStepHour");
    m_Start = ConvertStringToDatasetDate(xmlFile.GetFirstElementValueText("Start"), asSERIE_BEGINNING, m_TimeZoneHours, m_TimeStepHours, m_FirstTimeStepHour);
    m_End = ConvertStringToDatasetDate(xmlFile.GetFirstElementValueText("End"), asSERIE_END, m_TimeZoneHours, m_TimeStepHours, m_FirstTimeStepHour);
    m_FormatRaw = asGlobEnums::StringToFileFormatEnum(xmlFile.GetFirstElementValueText("FormatRaw"));
    m_CoordSys = asGlobEnums::StringToCoordSysEnum(xmlFile.GetFirstElementValueText("CoordinateSys"));
    m_Nan.push_back(xmlFile.GetFirstElementValueDouble("NaN", -9999));
    while(xmlFile.GetNextElement("NaN"))
    {
        m_Nan.push_back(xmlFile.GetThisElementValueDouble(-9999));
    }

    // Check that there is only one dataset per file
    if(xmlFile.GoToNextSameNode())
    {
        asLogError(_("More than 1 dataset was found in the predictand catalog. This is not anymore allowed."));
        return false;
    }

    // Reset the base path
    xmlFile.ClearCurrenNodePath();

    return true;
}

bool asCatalogPredictands::LoadDataProp(int StationId)
{
    // Initiate the check of the timezone
    wxString DataAccess;

    // Get datset information
    try
    {
        if(!asCatalogPredictands::LoadDatasetProp()) return false;
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
    DataAccess = wxString::Format("AtmoSwingFile.DataSet.DataList.Data[%d]",StationId);
    if(!xmlFile.GoToFirstNodeWithPath(DataAccess))
    {
        asLogError(_("The requested station id was not found in the predictand catalog."));
        return false;
    }

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
