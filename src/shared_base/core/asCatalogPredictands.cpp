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
#include <asTime.h>


asCatalogPredictands::asCatalogPredictands(const wxString &filePath)
:
wxObject()
{
    m_CatalogFilePath = filePath;

    // Initialize some data
    m_SetId = wxEmptyString;
    m_Name = wxEmptyString;
    m_Description = wxEmptyString;
    m_TimeZoneHours = 0;
    m_TimeStepHours = 0;
    m_FirstTimeStepHour = 0;
    m_DataPath = wxEmptyString;
    m_Start = 0;
    m_End = 0;

    // Get the xml file path
    if (m_CatalogFilePath.IsEmpty())
    {
        asLogError(_("No path was given for the predictand catalog."));
    }

    // Initiate some data
    m_DataPath = wxEmptyString;
}

asCatalogPredictands::~asCatalogPredictands()
{
    //dtor
}

bool asCatalogPredictands::Load()
{
    // Load xml file
    asFileXml xmlFile( m_CatalogFilePath, asFile::ReadOnly );
    if(!xmlFile.Open()) return false;

    if(!xmlFile.CheckRootElement()) return false;

    // Get data
    wxXmlNode *nodeDataset = xmlFile.GetRoot()->GetChildren();
    while (nodeDataset) {
        if (nodeDataset->GetName() == "dataset") {

            wxXmlNode *nodeProp = nodeDataset->GetChildren();
            while (nodeProp) {
                if (nodeProp->GetName() == "id") {
                    m_SetId = xmlFile.GetString(nodeProp);
                } else if (nodeProp->GetName() == "name") {
                    m_Name = xmlFile.GetString(nodeProp);
                } else if (nodeProp->GetName() == "description") {
                    m_Description = xmlFile.GetString(nodeProp);
                } else if (nodeProp->GetName() == "parameter") {
                    m_Parameter = asGlobEnums::StringToDataParameterEnum(xmlFile.GetString(nodeProp));
                } else if (nodeProp->GetName() == "unit") {
                    m_Unit = asGlobEnums::StringToDataUnitEnum(xmlFile.GetString(nodeProp));
                } else if (nodeProp->GetName() == "temporal_resolution") {
                    m_TemporalResolution = asGlobEnums::StringToDataTemporalResolutionEnum(xmlFile.GetString(nodeProp));
                } else if (nodeProp->GetName() == "spatial_aggregation") {
                    m_SpatialAggregation = asGlobEnums::StringToDataSpatialAggregationEnum(xmlFile.GetString(nodeProp));
                } else if (nodeProp->GetName() == "time_zone") {
                    m_TimeZoneHours = xmlFile.GetFloat(nodeProp);
                } else if (nodeProp->GetName() == "start") {
                    m_Start = asTime::GetTimeFromString(xmlFile.GetString(nodeProp), guess);
                } else if (nodeProp->GetName() == "end") {
                    m_End = asTime::GetTimeFromString(xmlFile.GetString(nodeProp), guess);
                } else if (nodeProp->GetName() == "first_time_step") {
                    m_FirstTimeStepHour = xmlFile.GetFloat(nodeProp);
                } else if (nodeProp->GetName() == "path") {
                    m_DataPath = xmlFile.GetString(nodeProp);
                } else if (nodeProp->GetName() == "nan") {
                    m_Nan.push_back(xmlFile.GetDouble(nodeProp));
                } else if (nodeProp->GetName() == "coordinate_system") {
                    m_CoordSys = xmlFile.GetString(nodeProp);
                } else if (nodeProp->GetName() == "stations") {
                    wxXmlNode *nodeData = nodeProp->GetChildren();
                    while (nodeData) {
                        if (nodeData->GetName() == "station") {
                            DataStruct station;

                            wxXmlNode *nodeDetail = nodeData->GetChildren();
                            while (nodeDetail) {
                                if (nodeDetail->GetName() == "official_id") {
                                    station.OfficialId = xmlFile.GetString(nodeDetail);
                                } else if (nodeDetail->GetName() == "id") {
                                    wxString idStr = xmlFile.GetString(nodeDetail);
                                    long id;
                                    idStr.ToLong(&id);
                                    station.Id = id;
                                } else if (nodeDetail->GetName() == "name") {
                                    station.Name = xmlFile.GetString(nodeDetail);
                                } else if (nodeDetail->GetName() == "x_coordinate") {
                                    wxString coordXStr = xmlFile.GetString(nodeDetail);
                                    double x;
                                    coordXStr.ToDouble(&x);
                                    station.Coord.x = x;
                                } else if (nodeDetail->GetName() == "y_coordinate") {
                                    wxString coordYStr = xmlFile.GetString(nodeDetail);
                                    double y;
                                    coordYStr.ToDouble(&y);
                                    station.Coord.y = y;
                                } else if (nodeDetail->GetName() == "height") {
                                    wxString heightStr = xmlFile.GetString(nodeDetail);
                                    double height;
                                    heightStr.ToDouble(&height);
                                    station.Height = (float)height;
                                } else if (nodeDetail->GetName() == "file_name") {
                                    station.Filename = xmlFile.GetString(nodeDetail);
                                } else if (nodeDetail->GetName() == "file_pattern") {
                                    station.Filepattern = xmlFile.GetString(nodeDetail);
                                } else if (nodeDetail->GetName() == "start") {
                                    station.Start = asTime::GetTimeFromString(xmlFile.GetString(nodeDetail), guess);
                                } else if (nodeDetail->GetName() == "end") {
                                    station.End = asTime::GetTimeFromString(xmlFile.GetString(nodeDetail), guess);
                                } else {
                                    xmlFile.UnknownNode(nodeDetail);
                                }

                                nodeDetail = nodeDetail->GetNext();
                            }
                            m_Stations.push_back(station);

                        } else {
                            xmlFile.UnknownNode(nodeData);
                        }

                        nodeData = nodeData->GetNext();
                    }

                } else {
                    xmlFile.UnknownNode(nodeProp);
                }

                nodeProp = nodeProp->GetNext();
            }

        } else {
            xmlFile.UnknownNode(nodeDataset);
        }

        nodeDataset = nodeDataset->GetNext();
    }

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
    
    xmlFile.Close();

    return true;
}
