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
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#ifndef AS_BATCH_FORECASTS_H
#define AS_BATCH_FORECASTS_H

#include "asFileBatchForecasts.h"
#include "asIncludes.h"

class asBatchForecasts : public wxObject {
  public:
    asBatchForecasts();

    ~asBatchForecasts() override = default;

    bool Load(const wxString &filePath);

    bool Save() const;

    int GetForecastsNb() const;

    void ClearForecasts();

    void AddForecast();

    wxString GetFilePath() const {
        return m_filePath;
    }

    void SetFilePath(const wxString &path) {
        m_filePath = path;
    }

    bool HasChanged() const {
        return m_hasChanged;
    }

    bool HasExports() const;

    bool ExportSyntheticXml() const {
        return m_exportSyntheticXml;
    }

    void SetExportSyntheticXml(bool val) {
        m_exportSyntheticXml = val;
    }

    bool ExportSyntheticTxt() const {
        return m_exportSyntheticTxt;
    }

    void SetExportSyntheticTxt(bool val) {
        m_exportSyntheticTxt = val;
    }

    void SetHasChanged(bool val) {
        m_hasChanged = val;
    }

    wxString GetForecastsOutputDirectory() const {
        return m_forecastsOutputDirectory;
    }

    void SetForecastsOutputDirectory(const wxString &val) {
        m_forecastsOutputDirectory = val;
    }

    wxString GetExportsOutputDirectory() const {
        return m_exportsOutputDirectory;
    }

    void SetExportsOutputDirectory(const wxString &val) {
        m_exportsOutputDirectory = val;
    }

    wxString GetParametersFileDirectory() const {
        return m_parametersFileDirectory;
    }

    void SetParametersFileDirectory(const wxString &val) {
        m_parametersFileDirectory = val;
    }

    wxString GetPredictorsArchiveDirectory() const {
        return m_predictorsArchiveDirectory;
    }

    void SetPredictorsArchiveDirectory(const wxString &val) {
        m_predictorsArchiveDirectory = val;
    }

    wxString GetPredictorsRealtimeDirectory() const {
        return m_predictorsRealtimeDirectory;
    }

    void SetPredictorsRealtimeDirectory(const wxString &val) {
        m_predictorsRealtimeDirectory = val;
    }

    wxString GetPredictandDBDirectory() const {
        return m_predictandDBDirectory;
    }

    void SetPredictandDBDirectory(const wxString &val) {
        m_predictandDBDirectory = val;
    }

    wxString GetForecastFileName(int i) const {
        wxASSERT((int)m_forecastFileNames.size() > i);
        return m_forecastFileNames[i];
    }

    void SetForecastFileName(int i, const wxString &val) {
        wxASSERT((int)m_forecastFileNames.size() > i);
        m_forecastFileNames[i] = val;
    }

  protected:
  private:
    bool m_hasChanged;
    bool m_exportSyntheticXml;
    bool m_exportSyntheticTxt;
    wxString m_filePath;
    wxString m_forecastsOutputDirectory;
    wxString m_exportsOutputDirectory;
    wxString m_parametersFileDirectory;
    wxString m_predictorsArchiveDirectory;
    wxString m_predictorsRealtimeDirectory;
    wxString m_predictandDBDirectory;
    vwxs m_forecastFileNames;
};

#endif
