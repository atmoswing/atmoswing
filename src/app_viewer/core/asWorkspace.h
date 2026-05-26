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

#ifndef AS_WORKSPACE_H
#define AS_WORKSPACE_H

#include "asFileWorkspace.h"
#include "asHeadersBase.h"

#if USE_GUI

#include <wx/brush.h>
#include <wx/colour.h>

#endif

class asWorkspace : public wxObject {
  public:
    asWorkspace();

    ~asWorkspace() override = default;

    bool Load(const wxString& filePath);

    bool Save() const;

    int GetLayersNb() const;

    void ClearLayers();

    void AddLayer();

    void ClearPredictorDirs();

    void AddPredictorDir(const wxString& id, const wxString& dir);

    wxString GetPredictorId(int i, const wxString& defVal = wxEmptyString);

    wxString GetPredictorDir(int i);

    wxString GetPredictorDir(wxString& datasetId);

    wxString GetFilePath() const {
        return _filePath;
    }

    void SetFilePath(wxString& path) {
        _filePath = path;
    }

    wxString GetForecastsDirectory() const {
        return _forecastsDirectory;
    }

    void SetForecastsDirectory(const wxString& val) {
        _forecastsDirectory = val;
    }

    wxString GetLayerPath(int i) const {
        wxASSERT((int)_layerPaths.size() > i);
        return _layerPaths[i];
    }

    void SetLayerPath(int i, const wxString& val) {
        wxASSERT((int)_layerPaths.size() > i);
        _layerPaths[i] = val;
    }

    wxString GetLayerType(int i) const {
        wxASSERT((int)_layerTypes.size() > i);
        return _layerTypes[i];
    }

    void SetLayerType(int i, const wxString& val) {
        wxASSERT((int)_layerTypes.size() > i);
        _layerTypes[i] = val;
    }

    int GetLayerTransparency(int i) const {
        wxASSERT((int)_layerTransparencies.size() > i);
        return _layerTransparencies[i];
    }

    void SetLayerTransparency(int i, int val) {
        wxASSERT((int)_layerTransparencies.size() > i);
        _layerTransparencies[i] = val;
    }

    bool GetLayerVisibility(int i) const {
        wxASSERT((int)_layerVisibilities.size() > i);
        return _layerVisibilities[i];
    }

    void SetLayerVisibility(int i, bool val) {
        wxASSERT((int)_layerVisibilities.size() > i);
        _layerVisibilities[i] = val;
    }

    int GetLayerLineWidth(int i) const {
        wxASSERT((int)_layerLineWidths.size() > i);
        return _layerLineWidths[i];
    }

    void SetLayerLineWidth(int i, int val) {
        wxASSERT((int)_layerLineWidths.size() > i);
        _layerLineWidths[i] = val;
    }

#if USE_GUI

    wxColour GetLayerLineColor(int i) const {
        wxASSERT((int)_layerLineColors.size() > i);
        return _layerLineColors[i];
    }

    void SetLayerLineColor(int i, wxColour& val) {
        wxASSERT((int)_layerLineColors.size() > i);
        _layerLineColors[i] = val;
    }

    wxColour GetLayerFillColor(int i) const {
        wxASSERT((int)_layerFillColors.size() > i);
        return _layerFillColors[i];
    }

    void SetLayerFillColor(int i, wxColour& val) {
        wxASSERT((int)_layerFillColors.size() > i);
        _layerFillColors[i] = val;
    }

    wxBrushStyle GetLayerBrushStyle(int i) const {
        wxASSERT((int)_layerBrushStyles.size() > i);
        return _layerBrushStyles[i];
    }

    void SetLayerBrushStyle(int i, wxBrushStyle& val) {
        wxASSERT((int)_layerBrushStyles.size() > i);
        _layerBrushStyles[i] = val;
    }

#endif

    double GetColorbarMaxValue() const {
        return _colorbarMaxValue;
    }

    void SetColorbarMaxValue(double val) {
        _colorbarMaxValue = val;
    }

    int GetTimeSeriesPlotPastDaysNb() const {
        return _timeSeriesPlotPastDaysNb;
    }

    void SetTimeSeriesPlotPastDaysNb(int val) {
        _timeSeriesPlotPastDaysNb = val;
    }

    int GetTimeSeriesMaxLengthDaily() const {
        return _timeSeriesMaxLengthDaily;
    }

    void SetTimeSeriesMaxLengthDaily(int val) {
        _timeSeriesMaxLengthDaily = val;
    }

    int GetTimeSeriesMaxLengthSubDaily() const {
        return _timeSeriesMaxLengthSubDaily;
    }

    void SetTimeSeriesMaxLengthSubDaily(int val) {
        _timeSeriesMaxLengthSubDaily = val;
    }

    int GetAlarmsPanelReturnPeriod() const {
        return _alarmsPanelReturnPeriod;
    }

    void SetAlarmsPanelReturnPeriod(int val) {
        _alarmsPanelReturnPeriod = val;
    }

    float GetAlarmsPanelQuantile() const {
        return _alarmsPanelQuantile;
    }

    void SetAlarmsPanelQuantile(float val) {
        _alarmsPanelQuantile = val;
    }

    bool HasChanged() const {
        return _hasChanged;
    }

    void SetHasChanged(bool val) {
        _hasChanged = val;
    }

  protected:
  private:
    bool _hasChanged;
    wxString _filePath;
    wxString _coordinateSys;
    wxString _forecastsDirectory;
    vwxs _layerPaths;
    vwxs _layerTypes;
    vi _layerTransparencies;
    vb _layerVisibilities;
    vi _layerLineWidths;
#if USE_GUI
    vector<wxColour> _layerLineColors;
    vector<wxColour> _layerFillColors;
    vector<wxBrushStyle> _layerBrushStyles;
#endif
    double _colorbarMaxValue;
    int _timeSeriesPlotPastDaysNb;
    int _timeSeriesMaxLengthDaily;
    int _timeSeriesMaxLengthSubDaily;
    int _alarmsPanelReturnPeriod;
    float _alarmsPanelQuantile;
    vwxs _predictorIds;
    vwxs _predictorDirs;
};

#endif
