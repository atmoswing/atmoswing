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
    m_HasChanged = false;
    m_FilePath = asConfig::GetDocumentsDir() + "AtmoSwing" + DS + "Workspace.xml";
    m_CoordinateSys = "EPSG:3857";
    m_ForecastsDirectory = asConfig::GetDocumentsDir() + "AtmoSwing" + DS + "Forecasts";
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
    if(!fileWorkspace.Open())
    {
        asLogError(_("Cannot open the workspace file."));
        return false;
    }
    if(!fileWorkspace.CheckRootElement())
    {
        asLogError(_("Errors were found in the workspace file."));
        return false;
    }

    // Get data
    wxXmlNode *node = fileWorkspace.GetRoot()->GetChildren();
    while (node) {
        if (node->GetName() == "coordinate_system") {
            m_CoordinateSys = fileWorkspace.GetString(node);
        } else if (node->GetName() == "forecast_directory") {
            m_ForecastsDirectory = fileWorkspace.GetString(node);
        } else if (node->GetName() == "colorbar_max_value") {
            m_ColorbarMaxValue = fileWorkspace.GetDouble(node);
        } else if (node->GetName() == "plot_time_series_past_days_nb") {
            m_TimeSeriesPlotPastDaysNb = fileWorkspace.GetInt(node);
        } else if (node->GetName() == "panel_alarms_return_period") {
            m_AlarmsPanelReturnPeriod = fileWorkspace.GetInt(node);
        } else if (node->GetName() == "panel_alarms_percentile") {
            m_AlarmsPanelPercentile = fileWorkspace.GetFloat(node);
        } else if (node->GetName() == "layer") {
            wxXmlNode *nodeLayer = node->GetChildren();
            while (nodeLayer) {
                if (nodeLayer->GetName() == "path") {
                    wxString path = fileWorkspace.GetString(nodeLayer);
                    wxFileName absolutePath(path);
                    absolutePath.Normalize();
                    m_LayerPaths.push_back(absolutePath.GetFullPath());
                } else if (nodeLayer->GetName() == "type") {
                    m_LayerTypes.push_back(fileWorkspace.GetString(nodeLayer));
                } else if (nodeLayer->GetName() == "transparency") {
                    m_LayerTransparencies.push_back(fileWorkspace.GetInt(nodeLayer));
                } else if (nodeLayer->GetName() == "visibility") {
                    m_LayerVisibilities.push_back(fileWorkspace.GetBool(nodeLayer));
                } else if (nodeLayer->GetName() == "line_width") {
                    m_LayerLineWidths.push_back(fileWorkspace.GetInt(nodeLayer));
                #if wxUSE_GUI
                } else if (nodeLayer->GetName() == "line_color") {
                    wxString lineColorStr = fileWorkspace.GetString(nodeLayer);
                    wxColour lineColor;
                    wxFromString(lineColorStr, &lineColor);
                    m_LayerLineColors.push_back(lineColor);
                } else if (nodeLayer->GetName() == "fill_color") {
                    wxString fillColorStr = fileWorkspace.GetString(nodeLayer);
                    wxColour fillColor;
                    wxFromString(fillColorStr, &fillColor);
                    m_LayerFillColors.push_back(fillColor);
                } else if (nodeLayer->GetName() == "brush_style") {
                    wxBrushStyle brushStyle = (wxBrushStyle)fileWorkspace.GetInt(nodeLayer);
                    m_LayerBrushStyles.push_back(brushStyle);
                #endif
                } else {
                    fileWorkspace.UnknownNode(nodeLayer);
                }

                nodeLayer = nodeLayer->GetNext();
            }

            if (m_LayerPaths.size()!=m_LayerTypes.size() ||
                m_LayerPaths.size()!=m_LayerTransparencies.size() ||
                m_LayerPaths.size()!=m_LayerVisibilities.size() ||
                m_LayerPaths.size()!=m_LayerLineWidths.size() 
                #if wxUSE_GUI 
                || m_LayerPaths.size()!=m_LayerLineColors.size()
                || m_LayerPaths.size()!=m_LayerFillColors.size()
                || m_LayerPaths.size()!=m_LayerBrushStyles.size()
                #endif
                )
            {
                asLogError(_("The number of elements in the layers is not consistent in the workspace file."));
                return false;
            }

        } else {
            fileWorkspace.UnknownNode(node);
        }

        node = node->GetNext();
    }

    return true;
}

bool asWorkspace::Save()
{
    // Open the file
    asFileWorkspace fileWorkspace(m_FilePath, asFile::Replace);
    if(!fileWorkspace.Open()) return false;

    if(!fileWorkspace.EditRootElement()) return false;
    fileWorkspace.GetRoot()->AddAttribute("target", "viewer");

    // General data
    fileWorkspace.AddChild(fileWorkspace.CreateNodeWithValue("coordinate_system", m_CoordinateSys));
    fileWorkspace.AddChild(fileWorkspace.CreateNodeWithValue("forecast_directory", m_ForecastsDirectory));
    fileWorkspace.AddChild(fileWorkspace.CreateNodeWithValue("colorbar_max_value", m_ColorbarMaxValue));
    fileWorkspace.AddChild(fileWorkspace.CreateNodeWithValue("plot_time_series_past_days_nb", m_TimeSeriesPlotPastDaysNb));
    fileWorkspace.AddChild(fileWorkspace.CreateNodeWithValue("panel_alarms_return_period", m_AlarmsPanelReturnPeriod));
    fileWorkspace.AddChild(fileWorkspace.CreateNodeWithValue("panel_alarms_percentile", m_AlarmsPanelPercentile));

    // GIS layers
    for (int i_layer=0; i_layer<GetLayersNb(); i_layer++)
    {
        wxXmlNode * nodeLayer = new wxXmlNode(wxXML_ELEMENT_NODE ,"layer" );

        nodeLayer->AddChild(fileWorkspace.CreateNodeWithValue("path", m_LayerPaths[i_layer]));
        nodeLayer->AddChild(fileWorkspace.CreateNodeWithValue("type", m_LayerTypes[i_layer]));
        nodeLayer->AddChild(fileWorkspace.CreateNodeWithValue("transparency", m_LayerTransparencies[i_layer]));
        nodeLayer->AddChild(fileWorkspace.CreateNodeWithValue("visibility", m_LayerVisibilities[i_layer]));
        nodeLayer->AddChild(fileWorkspace.CreateNodeWithValue("line_width", m_LayerLineWidths[i_layer]));
        #if wxUSE_GUI
            nodeLayer->AddChild(fileWorkspace.CreateNodeWithValue("line_color", wxToString(m_LayerLineColors[i_layer])));
            nodeLayer->AddChild(fileWorkspace.CreateNodeWithValue("fill_color", wxToString(m_LayerFillColors[i_layer])));
            wxString strBrush;
            strBrush << m_LayerBrushStyles[i_layer];
            nodeLayer->AddChild(fileWorkspace.CreateNodeWithValue("brush_style", strBrush));
        #endif

        fileWorkspace.AddChild(nodeLayer);
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
    #if wxUSE_GUI
        m_LayerLineColors.clear();
        m_LayerFillColors.clear();
        m_LayerBrushStyles.clear();
    #endif
}

void asWorkspace::AddLayer()
{
    int nb = m_LayerPaths.size()+1;
    m_LayerPaths.resize(nb);
    m_LayerTypes.resize(nb);
    m_LayerTransparencies.resize(nb);
    m_LayerVisibilities.resize(nb);
    m_LayerLineWidths.resize(nb);
    #if wxUSE_GUI
        m_LayerLineColors.resize(nb);
        m_LayerFillColors.resize(nb);
        m_LayerBrushStyles.resize(nb);
    #endif
}
