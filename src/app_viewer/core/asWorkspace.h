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
#include "asIncludes.h"

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

    wxString GetFilePath() const {
        return m_filePath;
    }

    void SetFilePath(wxString& path) {
        m_filePath = path;
    }

    wxString GetForecastsDirectory() const {
        return m_forecastsDirectory;
    }

    void SetForecastsDirectory(const wxString& val) {
        m_forecastsDirectory = val;
    }

    wxString GetLayerPath(int i) const {
        wxASSERT((int)m_layerPaths.size() > i);
        return m_layerPaths[i];
    }

    void SetLayerPath(int i, const wxString& val) {
        wxASSERT((int)m_layerPaths.size() > i);
        m_layerPaths[i] = val;
    }

    wxString GetLayerType(int i) const {
        wxASSERT((int)m_layerTypes.size() > i);
        return m_layerTypes[i];
    }

    void SetLayerType(int i, const wxString& val) {
        wxASSERT((int)m_layerTypes.size() > i);
        m_layerTypes[i] = val;
    }

    int GetLayerTransparency(int i) const {
        wxASSERT((int)m_layerTransparencies.size() > i);
        return m_layerTransparencies[i];
    }

    void SetLayerTransparency(int i, int val) {
        wxASSERT((int)m_layerTransparencies.size() > i);
        m_layerTransparencies[i] = val;
    }

    bool GetLayerVisibility(int i) const {
        wxASSERT((int)m_layerVisibilities.size() > i);
        return m_layerVisibilities[i];
    }

    void SetLayerVisibility(int i, bool val) {
        wxASSERT((int)m_layerVisibilities.size() > i);
        m_layerVisibilities[i] = val;
    }

    int GetLayerLineWidth(int i) const {
        wxASSERT((int)m_layerLineWidths.size() > i);
        return m_layerLineWidths[i];
    }

    void SetLayerLineWidth(int i, int val) {
        wxASSERT((int)m_layerLineWidths.size() > i);
        m_layerLineWidths[i] = val;
    }

#if USE_GUI

    wxColour GetLayerLineColor(int i) const {
        wxASSERT((int)m_layerLineColors.size() > i);
        return m_layerLineColors[i];
    }

    void SetLayerLineColor(int i, wxColour& val) {
        wxASSERT((int)m_layerLineColors.size() > i);
        m_layerLineColors[i] = val;
    }

    wxColour GetLayerFillColor(int i) const {
        wxASSERT((int)m_layerFillColors.size() > i);
        return m_layerFillColors[i];
    }

    void SetLayerFillColor(int i, wxColour& val) {
        wxASSERT((int)m_layerFillColors.size() > i);
        m_layerFillColors[i] = val;
    }

    wxBrushStyle GetLayerBrushStyle(int i) const {
        wxASSERT((int)m_layerBrushStyles.size() > i);
        return m_layerBrushStyles[i];
    }

    void SetLayerBrushStyle(int i, wxBrushStyle& val) {
        wxASSERT((int)m_layerBrushStyles.size() > i);
        m_layerBrushStyles[i] = val;
    }

#endif

    double GetColorbarMaxValue() const {
        return m_colorbarMaxValue;
    }

    void SetColorbarMaxValue(double val) {
        m_colorbarMaxValue = val;
    }

    int GetTimeSeriesPlotPastDaysNb() const {
        return m_timeSeriesPlotPastDaysNb;
    }

    void SetTimeSeriesPlotPastDaysNb(int val) {
        m_timeSeriesPlotPastDaysNb = val;
    }

    int GetAlarmsPanelReturnPeriod() const {
        return m_alarmsPanelReturnPeriod;
    }

    void SetAlarmsPanelReturnPeriod(int val) {
        m_alarmsPanelReturnPeriod = val;
    }

    float GetAlarmsPanelQuantile() const {
        return m_alarmsPanelQuantile;
    }

    void SetAlarmsPanelQuantile(float val) {
        m_alarmsPanelQuantile = val;
    }

    bool HasChanged() const {
        return m_hasChanged;
    }

    void SetHasChanged(bool val) {
        m_hasChanged = val;
    }

  protected:
  private:
    bool m_hasChanged;
    wxString m_filePath;
    wxString m_coordinateSys;
    wxString m_forecastsDirectory;
    vwxs m_layerPaths;
    vwxs m_layerTypes;
    vi m_layerTransparencies;
    vb m_layerVisibilities;
    vi m_layerLineWidths;
#if USE_GUI
    std::vector<wxColour> m_layerLineColors;
    std::vector<wxColour> m_layerFillColors;
    std::vector<wxBrushStyle> m_layerBrushStyles;
#endif
    double m_colorbarMaxValue;
    int m_timeSeriesPlotPastDaysNb;
    int m_alarmsPanelReturnPeriod;
    float m_alarmsPanelQuantile;
};

#endif
