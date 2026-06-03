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

#include "asIncludes.h"

asWorkspace::asWorkspace()
    : wxObject(),
      _hasChanged(false),
      _filePath(asConfig::GetDocumentsDir() + "AtmoSwing" + DS + "Workspace.asvw"),
      _coordinateSys("EPSG:3857"),
      _forecastsDirectory(asConfig::GetDocumentsDir() + "AtmoSwing" + DS + "Forecasts"),
      _colorbarMaxValue(50.0),
      _timeSeriesPlotPastDaysNb(3),
      _timeSeriesMaxLengthDaily(-1),
      _timeSeriesMaxLengthSubDaily(-1),
      _alarmsPanelReturnPeriod(10),
      _alarmsPanelQuantile(0.9f) {}

bool asWorkspace::Load(const wxString& filePath) {
    ClearLayers();

    // Open the file
    _filePath = filePath;
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
            _coordinateSys = asFileWorkspace::GetString(node);
        } else if (node->GetName() == "forecast_directory") {
            _forecastsDirectory = asFileWorkspace::GetString(node);
        } else if (node->GetName() == "colorbar_max_value") {
            _colorbarMaxValue = asFileWorkspace::GetDouble(node);
        } else if (node->GetName() == "plot_time_series_past_days_nb") {
            _timeSeriesPlotPastDaysNb = asFileWorkspace::GetInt(node);
        } else if (node->GetName() == "time_series_max_length_daily") {
            _timeSeriesMaxLengthDaily = asFileWorkspace::GetInt(node);
        } else if (node->GetName() == "time_series_max_length_sub_daily") {
            _timeSeriesMaxLengthSubDaily = asFileWorkspace::GetInt(node);
        } else if (node->GetName() == "panel_alarms_return_period") {
            _alarmsPanelReturnPeriod = asFileWorkspace::GetInt(node);
        } else if (node->GetName() == "panel_alarms_quantile") {
            _alarmsPanelQuantile = asFileWorkspace::GetFloat(node);
        } else if (node->GetName() == "predictors") {
            wxXmlNode* nodePredictor = node->GetChildren();
            while (nodePredictor) {
                if (nodePredictor->GetName() == "predictor") {
                    wxXmlNode* nodePredictorData = nodePredictor->GetChildren();
                    while (nodePredictorData) {
                        if (nodePredictorData->GetName() == "dir") {
                            _predictorDirs.push_back(asFileWorkspace::GetString(nodePredictorData));
                        } else if (nodePredictorData->GetName() == "id") {
                            _predictorIds.push_back(asFileWorkspace::GetString(nodePredictorData));
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
                                absolutePath.MakeAbsolute(wxFileName(_filePath).GetPath());
                            }
                            _layerPaths.push_back(absolutePath.GetFullPath());
                        } else if (nodeLayerData->GetName() == "type") {
                            _layerTypes.push_back(asFileWorkspace::GetString(nodeLayerData));
                        } else if (nodeLayerData->GetName() == "transparency") {
                            _layerTransparencies.push_back(asFileWorkspace::GetInt(nodeLayerData));
                        } else if (nodeLayerData->GetName() == "visibility") {
                            _layerVisibilities.push_back(asFileWorkspace::GetBool(nodeLayerData));
                        } else if (nodeLayerData->GetName() == "line_width") {
                            _layerLineWidths.push_back(asFileWorkspace::GetInt(nodeLayerData));
#if USE_GUI
                        } else if (nodeLayerData->GetName() == "line_color") {
                            wxString lineColorStr = asFileWorkspace::GetString(nodeLayerData);
                            wxColour lineColor;
                            wxFromString(lineColorStr, &lineColor);
                            _layerLineColors.push_back(lineColor);
                        } else if (nodeLayerData->GetName() == "fill_color") {
                            wxString fillColorStr = asFileWorkspace::GetString(nodeLayerData);
                            wxColour fillColor;
                            wxFromString(fillColorStr, &fillColor);
                            _layerFillColors.push_back(fillColor);
                        } else if (nodeLayerData->GetName() == "brush_style") {
                            auto brushStyle = (wxBrushStyle)asFileWorkspace::GetInt(nodeLayerData);
                            _layerBrushStyles.push_back(brushStyle);
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

            if (_layerPaths.size() != _layerTypes.size() || _layerPaths.size() != _layerTransparencies.size() ||
                _layerPaths.size() != _layerVisibilities.size() || _layerPaths.size() != _layerLineWidths.size()
#if USE_GUI
                || _layerPaths.size() != _layerLineColors.size() || _layerPaths.size() != _layerFillColors.size() ||
                _layerPaths.size() != _layerBrushStyles.size()
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
    asFileWorkspace file(_filePath, asFile::Replace);
    if (!file.Open()) return false;

    if (!file.EditRootElement()) return false;

    // General data
    file.AddChild(file.CreateNode("coordinate_system", _coordinateSys));
    file.AddChild(file.CreateNode("forecast_directory", _forecastsDirectory));
    file.AddChild(file.CreateNode("colorbar_max_value", _colorbarMaxValue));
    file.AddChild(file.CreateNode("plot_time_series_past_days_nb", _timeSeriesPlotPastDaysNb));
    file.AddChild(file.CreateNode("time_series_max_length_daily", _timeSeriesMaxLengthDaily));
    file.AddChild(file.CreateNode("time_series_max_length_sub_daily", _timeSeriesMaxLengthSubDaily));
    file.AddChild(file.CreateNode("panel_alarms_return_period", _alarmsPanelReturnPeriod));
    file.AddChild(file.CreateNode("panel_alarms_quantile", _alarmsPanelQuantile));

    // Predictors
    wxXmlNode* nodePredictors = new wxXmlNode(wxXML_ELEMENT_NODE, "predictors");
    for (int iPtor = 0; iPtor < _predictorIds.size(); iPtor++) {
        wxXmlNode* nodePredictor = new wxXmlNode(wxXML_ELEMENT_NODE, "predictor");
        nodePredictor->AddChild(file.CreateNode("dir", _predictorDirs[iPtor]));
        nodePredictor->AddChild(file.CreateNode("id", _predictorIds[iPtor]));
        nodePredictors->AddChild(nodePredictor);
    }
    file.AddChild(nodePredictors);

    // GIS layers
    wxXmlNode* nodeLayers = new wxXmlNode(wxXML_ELEMENT_NODE, "layers");
    for (int iLayer = 0; iLayer < GetLayersNb(); iLayer++) {
        wxXmlNode* nodeLayer = new wxXmlNode(wxXML_ELEMENT_NODE, "layer");

        wxString path = _layerPaths[iLayer];
        if (path.StartsWith(wxFileName(_filePath).GetPath())) {
            wxFileName relativePath = wxFileName(path);
            relativePath.MakeRelativeTo(wxFileName(_filePath).GetPath());
            path = relativePath.GetFullPath();
        }

        nodeLayer->AddChild(file.CreateNode("path", path));
        nodeLayer->AddChild(file.CreateNode("type", _layerTypes[iLayer]));
        nodeLayer->AddChild(file.CreateNode("transparency", _layerTransparencies[iLayer]));
        nodeLayer->AddChild(file.CreateNode("visibility", _layerVisibilities[iLayer]));
        nodeLayer->AddChild(file.CreateNode("line_width", _layerLineWidths[iLayer]));
#if USE_GUI
        nodeLayer->AddChild(file.CreateNode("line_color", wxToString(_layerLineColors[iLayer])));
        nodeLayer->AddChild(file.CreateNode("fill_color", wxToString(_layerFillColors[iLayer])));
        wxString strBrush;
        strBrush << _layerBrushStyles[iLayer];
        nodeLayer->AddChild(file.CreateNode("brush_style", strBrush));
#endif

        nodeLayers->AddChild(nodeLayer);
    }
    file.AddChild(nodeLayers);

    if (!file.Save()) {
        wxLogError(_("Could not save workspace file %s."), _filePath);
        return false;
    }

    return true;
}

int asWorkspace::GetLayersNb() const {
    auto layersNb = (int)_layerPaths.size();
    return layersNb;
}

void asWorkspace::ClearLayers() {
    _layerPaths.clear();
    _layerTypes.clear();
    _layerTransparencies.clear();
    _layerVisibilities.clear();
    _layerLineWidths.clear();
#if USE_GUI
    _layerLineColors.clear();
    _layerFillColors.clear();
    _layerBrushStyles.clear();
#endif
}

void asWorkspace::AddLayer() {
    int nb = _layerPaths.size() + 1;
    _layerPaths.resize(nb);
    _layerTypes.resize(nb);
    _layerTransparencies.resize(nb);
    _layerVisibilities.resize(nb);
    _layerLineWidths.resize(nb);
#if USE_GUI
    _layerLineColors.resize(nb);
    _layerFillColors.resize(nb);
    _layerBrushStyles.resize(nb);
#endif
}

void asWorkspace::ClearPredictorDirs() {
    _predictorIds.clear();
    _predictorDirs.clear();
}

void asWorkspace::AddPredictorDir(const wxString& id, const wxString& dir) {
    _predictorIds.push_back(id);
    _predictorDirs.push_back(dir);
}

wxString asWorkspace::GetPredictorId(int i, const wxString& defVal) {
    if (_predictorIds.size() < i) {
        return defVal;
    }
    return _predictorIds[i - 1];
}

wxString asWorkspace::GetPredictorDir(int i) {
    if (_predictorDirs.size() < i) {
        return wxEmptyString;
    }
    return _predictorDirs[i - 1];
}

wxString asWorkspace::GetPredictorDir(wxString& datasetId) {
    for (int i = 0; i < _predictorIds.size(); i++) {
        wxString id = _predictorIds[i];
        if (datasetId.IsSameAs(id, false)) {
            return _predictorDirs[i];
        }
    }

    return wxEmptyString;
}
