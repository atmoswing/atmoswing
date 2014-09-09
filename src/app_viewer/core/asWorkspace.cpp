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
    ClearLayers();

    // Open the file
    m_FilePath = filePath;
    asFileWorkspace fileWorkspace(filePath, asFile::ReadOnly);
    if(!fileWorkspace.Open()) return false;

    if(!fileWorkspace.GoToRootElement()) return false;

    // Get general data
    m_CoordinateSys = fileWorkspace.GetFirstElementAttributeValueText("CoordinateSys", "value");
    m_ForecastsDirectory = fileWorkspace.GetFirstElementAttributeValueText("ForecastsDirectory", "value");
    
    // Display options
    m_ColorbarMaxValue = fileWorkspace.GetFirstElementAttributeValueDouble("ColorbarMaxValue", "value");
    m_TimeSeriesPlotPastDaysNb = fileWorkspace.GetFirstElementAttributeValueInt("TimeSeriesPlotPastDaysNb", "value");
    m_AlarmsPanelReturnPeriod = fileWorkspace.GetFirstElementAttributeValueInt("AlarmsPanelReturnPeriod", "value");
    m_AlarmsPanelPercentile = fileWorkspace.GetFirstElementAttributeValueFloat("AlarmsPanelPercentile", "value");

    // GIS layers
    if(!fileWorkspace.GoToFirstNodeWithPath("GISLayers")) return false;
    
    if(fileWorkspace.GoToFirstNodeWithPath("Layer"))
    {
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
            int width = fileWorkspace.GetFirstElementAttributeValueInt("LineWidth", "value", 1);
            m_LayerLineWidths.push_back(width);
            wxString lineColorStr = fileWorkspace.GetFirstElementAttributeValueText("LineColor", "value", "black");
            wxColour lineColor;
            wxFromString(lineColorStr, &lineColor);
            m_LayerLineColors.push_back(lineColor);
            wxString fillColorStr = fileWorkspace.GetFirstElementAttributeValueText("FillColor", "value", "black");
            wxColour fillColor;
            wxFromString(fillColorStr, &fillColor);
            m_LayerFillColors.push_back(lineColor);
            wxBrushStyle brushStyle = (wxBrushStyle)fileWorkspace.GetFirstElementAttributeValueInt("BrushStyle", "value", wxBRUSHSTYLE_TRANSPARENT);
            m_LayerBrushStyles.push_back(brushStyle);
        
            // Find the next layer
            if (!fileWorkspace.GoToNextSameNode()) break;
        }
    }

    return true;
}

bool asWorkspace::Save()
{
    // Open the file
    asFileWorkspace fileWorkspace(m_FilePath, asFile::Replace);
    if(!fileWorkspace.Open()) return false;

    if(!fileWorkspace.InsertRootElement()) return false;

    // Get general data
    if(!fileWorkspace.InsertElementAndAttribute("", "CoordinateSys", "value", m_CoordinateSys)) return false;
    if(!fileWorkspace.InsertElementAndAttribute("", "ForecastsDirectory", "value", m_ForecastsDirectory)) return false;
    
    // Display options
    wxString strColorbarMaxValue;
    strColorbarMaxValue << m_ColorbarMaxValue;
    if(!fileWorkspace.InsertElementAndAttribute("", "ColorbarMaxValue", "value", strColorbarMaxValue)) return false;
    wxString strTimeSeriesPlotPastDaysNb;
    strTimeSeriesPlotPastDaysNb << m_TimeSeriesPlotPastDaysNb;
    if(!fileWorkspace.InsertElementAndAttribute("", "TimeSeriesPlotPastDaysNb", "value", strTimeSeriesPlotPastDaysNb)) return false;
    wxString strAlarmsPanelReturnPeriod;
    strAlarmsPanelReturnPeriod << m_AlarmsPanelReturnPeriod;
    if(!fileWorkspace.InsertElementAndAttribute("", "AlarmsPanelReturnPeriod", "value", strAlarmsPanelReturnPeriod)) return false;
    wxString strAlarmsPanelPercentile;
    strAlarmsPanelPercentile << m_AlarmsPanelPercentile;
    if(!fileWorkspace.InsertElementAndAttribute("", "AlarmsPanelPercentile", "value", strAlarmsPanelPercentile)) return false;

    // GIS layers
    if(!fileWorkspace.InsertElement("", "GISLayers")) return false;
    if(!fileWorkspace.GoToFirstNodeWithPath("GISLayers")) return false;
    
    for (int i_layer=0; i_layer<GetLayersNb(); i_layer++)
    {
        if(!fileWorkspace.InsertElement("", "Layer")) return false;
        if(!fileWorkspace.GoToLastNodeWithPath("Layer")) return false;

        if(!fileWorkspace.InsertElementAndAttribute("", "Path", "value", m_LayerPaths[i_layer])) return false;
        if(!fileWorkspace.InsertElementAndAttribute("", "Type", "value", m_LayerTypes[i_layer])) return false;
        wxString strTransparency;
        strTransparency << m_LayerTransparencies[i_layer];
        if(!fileWorkspace.InsertElementAndAttribute("", "Transparency", "value", strTransparency)) return false;
        wxString strVisibility;
        strVisibility << m_LayerVisibilities[i_layer];
        if(!fileWorkspace.InsertElementAndAttribute("", "Visibility", "value", strVisibility)) return false;
        wxString strWidth;
        strWidth << m_LayerLineWidths[i_layer];
        if(!fileWorkspace.InsertElementAndAttribute("", "LineWidth", "value", strWidth)) return false;
        if(!fileWorkspace.InsertElementAndAttribute("", "LineColor", "value", wxToString(m_LayerLineColors[i_layer]))) return false;
        if(!fileWorkspace.InsertElementAndAttribute("", "FillColor", "value", wxToString(m_LayerFillColors[i_layer]))) return false;
        wxString strBrush;
        strBrush << m_LayerBrushStyles[i_layer];
        if(!fileWorkspace.InsertElementAndAttribute("", "BrushStyle", "value", strBrush)) return false;
        
        if(!fileWorkspace.GoANodeBack()) return false;
    }

    fileWorkspace.Save();

    return true;
}

int asWorkspace::GetLayersNb()
{
    int layersNb = (int)m_LayerPaths.size();
    return layersNb;
}

void asWorkspace::ClearLayers()
{
    m_LayerPaths.clear();
    m_LayerTypes.clear();
    m_LayerTransparencies.clear();
    m_LayerVisibilities.clear();
    m_LayerLineWidths.clear();
    m_LayerLineColors.clear();
    m_LayerFillColors.clear();
    m_LayerBrushStyles.clear();
}

void asWorkspace::AddLayer()
{
    int nb = m_LayerPaths.size()+1;
    m_LayerPaths.resize(nb);
    m_LayerTypes.resize(nb);
    m_LayerTransparencies.resize(nb);
    m_LayerVisibilities.resize(nb);
    m_LayerLineWidths.resize(nb);
    m_LayerLineColors.resize(nb);
    m_LayerFillColors.resize(nb);
    m_LayerBrushStyles.resize(nb);
}
