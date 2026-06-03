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
#include "asHeadersBase.h"

class asBatchForecasts : public wxObject {
  public:
    enum Export {
        None = 0,
        FullXml = 10,
        SmallCsv = 20,
        CustomCsvFVG = 30,
    };

    asBatchForecasts();

    ~asBatchForecasts() override = default;

    bool Load(const wxString& filePath);

    bool Save() const;

    int GetForecastsNb() const;

    void ClearForecasts();

    void AddForecast();

    wxString GetFilePath() const {
        return _filePath;
    }

    void SetFilePath(const wxString& path) {
        _filePath = path;
    }

    bool HasChanged() const {
        return _hasChanged;
    }

    bool HasExports() const;

    Export GetExport() const {
        return _export;
    }

    void SetExport(Export val) {
        _export = val;
    }

    void SetHasChanged(bool val) {
        _hasChanged = val;
    }

    wxString GetForecastsOutputDirectory() const {
        return _forecastsOutputDirectory;
    }

    void SetForecastsOutputDirectory(const wxString& val) {
        _forecastsOutputDirectory = val;
    }

    wxString GetExportsOutputDirectory() const {
        return _exportsOutputDirectory;
    }

    void SetExportsOutputDirectory(const wxString& val) {
        _exportsOutputDirectory = val;
    }

    wxString GetParametersFileDirectory() const {
        return _parametersFileDirectory;
    }

    void SetParametersFileDirectory(const wxString& val) {
        _parametersFileDirectory = val;
    }

    wxString GetPredictorsArchiveDirectory() const {
        return _predictorsArchiveDirectory;
    }

    void SetPredictorsArchiveDirectory(const wxString& val) {
        _predictorsArchiveDirectory = val;
    }

    wxString GetPredictorsRealtimeDirectory() const {
        return _predictorsRealtimeDirectory;
    }

    void SetPredictorsRealtimeDirectory(const wxString& val) {
        _predictorsRealtimeDirectory = val;
    }

    wxString GetPredictandDBDirectory() const {
        return _predictandDBDirectory;
    }

    void SetPredictandDBDirectory(const wxString& val) {
        _predictandDBDirectory = val;
    }

    wxString GetForecastFileName(int i) const {
        wxASSERT((int)_forecastFileNames.size() > i);
        return _forecastFileNames[i];
    }

    void SetForecastFileName(int i, const wxString& val) {
        wxASSERT((int)_forecastFileNames.size() > i);
        _forecastFileNames[i] = val;
    }

  protected:
  private:
    bool _hasChanged;
    Export _export;
    wxString _filePath;
    wxString _forecastsOutputDirectory;
    wxString _exportsOutputDirectory;
    wxString _parametersFileDirectory;
    wxString _predictorsArchiveDirectory;
    wxString _predictorsRealtimeDirectory;
    wxString _predictandDBDirectory;
    vwxs _forecastFileNames;
};

#endif
