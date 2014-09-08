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
 * Portions Copyright 2014 Pascal Horton, Terr@num.
 */

#include "asWorkspace.h"

asWorkspace::asWorkspace()
:
wxObject()
{
    m_FilePath = asConfig::GetUserDataDir("AtmoSwing") + DS + "Workspace.xml";
    m_CoordinateSys = "EPSG:3857";
    m_ForecastsDirectory = asConfig::GetDocumentsDir()+"AtmoSwing"+DS+"Forecasts";
    m_ColorbarMaxValue = 50.0;
    m_TimeSeriesPlotPastDaysNb = 3;
    m_AlarmsPanelReturnPeriod = 10;
    m_AlarmsPanelPercentile = 0.9f;
}

asWorkspace::~asWorkspace()
{
    //dtor
}

bool asWorkspace::Load(const wxString &filePath)
{
    // Open the file
    m_FilePath = filePath;
    asFileWorkspace fileWorkspace(filePath, asFile::ReadOnly);
    if(!fileWorkspace.Open()) return false;

    if(!fileWorkspace.GoToRootElement()) return false;

    // Get general data
    m_CoordinateSys = fileWorkspace.GetFirstElementAttributeValueText("CoordinateSys", "value");
    m_ForecastsDirectory = fileWorkspace.GetFirstElementAttributeValueText("ForecastsDirectory", "value");

    // GIS layers
    if(!fileWorkspace.GoToFirstNodeWithPath("Layer")) return false;

    // Open new layers
    while(true)
    {
        // Get attributes
        wxString path = fileWorkspace.GetFirstElementAttributeValueText("Path", "value");
        m_LayerPaths.push_back(path);
        wxString type = fileWorkspace.GetFirstElementAttributeValueText("Type", "value");
        m_LayerTypes.push_back(type);
        int transparency = fileWorkspace.GetFirstElementAttributeValueInt("Transparency", "value", 0);
        m_LayerTransparencies.push_back(transparency);
        bool visibility = fileWorkspace.GetFirstElementAttributeValueBool("Visibility", "value", true);
        m_LayerVisibilities.push_back(visibility);
        int width = fileWorkspace.GetFirstElementAttributeValueInt("Width", "value", 1);
        m_LayerLineWidths.push_back(width);
        long lineColorLong = long(fileWorkspace.GetFirstElementAttributeValueInt("LineColor", "value", 0));
        wxColour lineColor;
        lineColor.SetRGB((wxUint32)lineColorLong);
        m_LayerLineColors.push_back(lineColor);
        long fillColorLong = long(fileWorkspace.GetFirstElementAttributeValueInt("FillColor", "value", 0));
        wxColour fillColor;
        fillColor.SetRGB((wxUint32)fillColorLong);
        m_LayerFillColors.push_back(lineColor);
        
        // Find the next layer
        if (!fileWorkspace.GoToNextSameNode()) break;
    }

    return true;
}

bool asWorkspace::Save()
{



    return true;
}

int asWorkspace::GetLayersNb()
{
    int layersNb = (int)m_LayerPaths.size();
    return layersNb;
}