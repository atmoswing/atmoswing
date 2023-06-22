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
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2014-2015 Pascal Horton, Terranum.
 */

#include "asWorkspace.h"

asWorkspace::asWorkspace()
    : wxObject(),
      m_hasChanged(false),
      m_filePath(asConfig::GetDocumentsDir() + "AtmoSwing" + DS + "Workspace.asvw"),
      m_coordinateSys("EPSG:3857"),
      m_forecastsDirectory(asConfig::GetDocumentsDir() + "AtmoSwing" + DS + "Forecasts"),
      m_colorbarMaxValue(50.0),
      m_timeSeriesPlotPastDaysNb(3),
      m_timeSeriesMaxLengthDaily(-1),
      m_timeSeriesMaxLengthSubDaily(-1),
      m_alarmsPanelReturnPeriod(10),
      m_alarmsPanelQuantile(0.9f) {}

bool asWorkspace::Load(const wxString& filePath) {
    ClearLayers();

    // Open the file
    m_filePath = filePath;
    asFileWorkspace file(filePath, asFile::ReadOnly);
    if (!file.Open()) {
        wxLogError(_("Cannot open the workspace file."));
        return false;
    }
    if (!file.CheckRootElement()) {
        wxLogError(_("Errors were found in the workspace file."));
        return false;
    }

    // Get data
    wxXmlNode* node = file.GetRoot()->GetChildren();
    while (node) {
        if (node->GetName() == "coordinate_system") {
            m_coordinateSys = asFileWorkspace::GetString(node);
        } else if (node->GetName() == "forecast_directory") {
            m_forecastsDirectory = asFileWorkspace::GetString(node);
        } else if (node->GetName() == "colorbar_max_value") {
            m_colorbarMaxValue = asFileWorkspace::GetDouble(node);
        } else if (node->GetName() == "plot_time_series_past_days_nb") {
            m_timeSeriesPlotPastDaysNb = asFileWorkspace::GetInt(node);
        } else if (node->GetName() == "time_series_max_length_daily") {
            m_timeSeriesMaxLengthDaily = asFileWorkspace::GetInt(node);
        } else if (node->GetName() == "time_series_max_length_sub_daily") {
            m_timeSeriesMaxLengthSubDaily = asFileWorkspace::GetInt(node);
        } else if (node->GetName() == "panel_alarms_return_period") {
            m_alarmsPanelReturnPeriod = asFileWorkspace::GetInt(node);
        } else if (node->GetName() == "panel_alarms_quantile") {
            m_alarmsPanelQuantile = asFileWorkspace::GetFloat(node);
        } else if (node->GetName() == "predictors") {
            wxXmlNode* nodePredictor = node->GetChildren();
            while (nodePredictor) {
                if (nodePredictor->GetName() == "predictor") {
                    wxXmlNode* nodePredictorData = nodePredictor->GetChildren();
                    while (nodePredictorData) {
                        if (nodePredictorData->GetName() == "dir") {
                            m_predictorDirs.push_back(asFileWorkspace::GetString(nodePredictorData));
                        } else if (nodePredictorData->GetName() == "id") {
                            m_predictorIds.push_back(asFileWorkspace::GetString(nodePredictorData));
                        } else {
                            file.UnknownNode(nodePredictorData);
                        }

                        nodePredictorData = nodePredictorData->GetNext();
                    }
                } else {
                    file.UnknownNode(nodePredictor);
                }

                nodePredictor = nodePredictor->GetNext();
            }
        } else if (node->GetName() == "layers") {
            wxXmlNode* nodeLayer = node->GetChildren();
            while (nodeLayer) {
                if (nodeLayer->GetName() == "layer") {
                    wxXmlNode* nodeLayerData = nodeLayer->GetChildren();
                    while (nodeLayerData) {
                        if (nodeLayerData->GetName() == "path") {
                            wxString path = asFileWorkspace::GetString(nodeLayerData);
                            wxFileName absolutePath(path);
                            if (absolutePath.IsRelative()) {
                                absolutePath.MakeAbsolute(wxFileName(m_filePath).GetPath());
                            }
                            m_layerPaths.push_back(absolutePath.GetFullPath());
                        } else if (nodeLayerData->GetName() == "type") {
                            m_layerTypes.push_back(asFileWorkspace::GetString(nodeLayerData));
                        } else if (nodeLayerData->GetName() == "transparency") {
                            m_layerTransparencies.push_back(asFileWorkspace::GetInt(nodeLayerData));
                        } else if (nodeLayerData->GetName() == "visibility") {
                            m_layerVisibilities.push_back(asFileWorkspace::GetBool(nodeLayerData));
                        } else if (nodeLayerData->GetName() == "line_width") {
                            m_layerLineWidths.push_back(asFileWorkspace::GetInt(nodeLayerData));
#if USE_GUI
                        } else if (nodeLayerData->GetName() == "line_color") {
                            wxString lineColorStr = asFileWorkspace::GetString(nodeLayerData);
                            wxColour lineColor;
                            wxFromString(lineColorStr, &lineColor);
                            m_layerLineColors.push_back(lineColor);
                        } else if (nodeLayerData->GetName() == "fill_color") {
                            wxString fillColorStr = asFileWorkspace::GetString(nodeLayerData);
                            wxColour fillColor;
                            wxFromString(fillColorStr, &fillColor);
                            m_layerFillColors.push_back(fillColor);
                        } else if (nodeLayerData->GetName() == "brush_style") {
                            auto brushStyle = (wxBrushStyle)asFileWorkspace::GetInt(nodeLayerData);
                            m_layerBrushStyles.push_back(brushStyle);
#endif
                        } else {
                            file.UnknownNode(nodeLayerData);
                        }

                        nodeLayerData = nodeLayerData->GetNext();
                    }
                } else {
                    file.UnknownNode(nodeLayer);
                }

                nodeLayer = nodeLayer->GetNext();
            }

            if (m_layerPaths.size() != m_layerTypes.size() || m_layerPaths.size() != m_layerTransparencies.size() ||
                m_layerPaths.size() != m_layerVisibilities.size() || m_layerPaths.size() != m_layerLineWidths.size()
#if USE_GUI
                || m_layerPaths.size() != m_layerLineColors.size() || m_layerPaths.size() != m_layerFillColors.size() ||
                m_layerPaths.size() != m_layerBrushStyles.size()
#endif
            ) {
                wxLogError(_("The number of elements in the layers is not consistent in the workspace file."));
                return false;
            }

        } else {
            file.UnknownNode(node);
        }

        node = node->GetNext();
    }

    return true;
}

bool asWorkspace::Save() const {
    // Open the file
    asFileWorkspace file(m_filePath, asFile::Replace);
    if (!file.Open()) return false;

    if (!file.EditRootElement()) return false;

    // General data
    file.AddChild(file.CreateNode("coordinate_system", m_coordinateSys));
    file.AddChild(file.CreateNode("forecast_directory", m_forecastsDirectory));
    file.AddChild(file.CreateNode("colorbar_max_value", m_colorbarMaxValue));
    file.AddChild(file.CreateNode("plot_time_series_past_days_nb", m_timeSeriesPlotPastDaysNb));
    file.AddChild(file.CreateNode("time_series_max_length_daily", m_timeSeriesMaxLengthDaily));
    file.AddChild(file.CreateNode("time_series_max_length_sub_daily", m_timeSeriesMaxLengthSubDaily));
    file.AddChild(file.CreateNode("panel_alarms_return_period", m_alarmsPanelReturnPeriod));
    file.AddChild(file.CreateNode("panel_alarms_quantile", m_alarmsPanelQuantile));

    // Predictors
    wxXmlNode* nodePredictors = new wxXmlNode(wxXML_ELEMENT_NODE, "predictors");
    for (int iPtor = 0; iPtor < m_predictorIds.size(); iPtor++) {
        wxXmlNode* nodePredictor = new wxXmlNode(wxXML_ELEMENT_NODE, "predictor");
        nodePredictor->AddChild(file.CreateNode("dir", m_predictorDirs[iPtor]));
        nodePredictor->AddChild(file.CreateNode("id", m_predictorIds[iPtor]));
        nodePredictors->AddChild(nodePredictor);
    }
    file.AddChild(nodePredictors);

    // GIS layers
    wxXmlNode* nodeLayers = new wxXmlNode(wxXML_ELEMENT_NODE, "layers");
    for (int iLayer = 0; iLayer < GetLayersNb(); iLayer++) {
        wxXmlNode* nodeLayer = new wxXmlNode(wxXML_ELEMENT_NODE, "layer");

        wxString path = m_layerPaths[iLayer];
        if (path.StartsWith(wxFileName(m_filePath).GetPath())) {
            wxFileName relativePath = wxFileName(path);
            relativePath.MakeRelativeTo(wxFileName(m_filePath).GetPath());
            path = relativePath.GetFullPath();
        }

        nodeLayer->AddChild(file.CreateNode("path", path));
        nodeLayer->AddChild(file.CreateNode("type", m_layerTypes[iLayer]));
        nodeLayer->AddChild(file.CreateNode("transparency", m_layerTransparencies[iLayer]));
        nodeLayer->AddChild(file.CreateNode("visibility", m_layerVisibilities[iLayer]));
        nodeLayer->AddChild(file.CreateNode("line_width", m_layerLineWidths[iLayer]));
#if USE_GUI
        nodeLayer->AddChild(file.CreateNode("line_color", wxToString(m_layerLineColors[iLayer])));
        nodeLayer->AddChild(file.CreateNode("fill_color", wxToString(m_layerFillColors[iLayer])));
        wxString strBrush;
        strBrush << m_layerBrushStyles[iLayer];
        nodeLayer->AddChild(file.CreateNode("brush_style", strBrush));
#endif

        nodeLayers->AddChild(nodeLayer);
    }
    file.AddChild(nodeLayers);

    file.Save();

    return true;
}

int asWorkspace::GetLayersNb() const {
    auto layersNb = (int)m_layerPaths.size();
    return layersNb;
}

void asWorkspace::ClearLayers() {
    m_layerPaths.clear();
    m_layerTypes.clear();
    m_layerTransparencies.clear();
    m_layerVisibilities.clear();
    m_layerLineWidths.clear();
#if USE_GUI
    m_layerLineColors.clear();
    m_layerFillColors.clear();
    m_layerBrushStyles.clear();
#endif
}

void asWorkspace::AddLayer() {
    int nb = m_layerPaths.size() + 1;
    m_layerPaths.resize(nb);
    m_layerTypes.resize(nb);
    m_layerTransparencies.resize(nb);
    m_layerVisibilities.resize(nb);
    m_layerLineWidths.resize(nb);
#if USE_GUI
    m_layerLineColors.resize(nb);
    m_layerFillColors.resize(nb);
    m_layerBrushStyles.resize(nb);
#endif
}

void asWorkspace::ClearPredictorDirs() {
    m_predictorIds.clear();
    m_predictorDirs.clear();
}

void asWorkspace::AddPredictorDir(const wxString& id, const wxString& dir) {
    m_predictorIds.push_back(id);
    m_predictorDirs.push_back(dir);
}

wxString asWorkspace::GetPredictorId(int i, const wxString& defVal) {
    if (m_predictorIds.size() < i) {
        return defVal;
    }
    return m_predictorIds[i - 1];
}

wxString asWorkspace::GetPredictorDir(int i) {
    if (m_predictorDirs.size() < i) {
        return wxEmptyString;
    }
    return m_predictorDirs[i - 1];
}

wxString asWorkspace::GetPredictorDir(wxString& datasetId) {
    for (int i = 0; i < m_predictorIds.size(); i++) {
        wxString id = m_predictorIds[i];
        if (datasetId.IsSameAs(id, false)) {
            return m_predictorDirs[i];
        }
    }

    return wxEmptyString;
}
