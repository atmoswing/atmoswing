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

#include "asBatchForecasts.h"

#include "asIncludes.h"

asBatchForecasts::asBatchForecasts()
    : wxObject(),
      _hasChanged(false),
      _export(None) {
    wxString baseDir = asConfig::GetDocumentsDir() + "AtmoSwing" + DS;
    _filePath = baseDir + "Parameters" + DS + "BatchForecasts.asfb";
    _forecastsOutputDirectory = baseDir + "Forecasts";
    _exportsOutputDirectory = baseDir + "Exports";
    _parametersFileDirectory = baseDir + "Parameters";
    _predictorsArchiveDirectory = baseDir + "Data" + DS + "Archive predictors";
    _predictorsRealtimeDirectory = baseDir + "Data" + DS + "Forecasted predictors";
    _predictandDBDirectory = baseDir + "Data" + DS + "Predictands";
}

bool asBatchForecasts::Load(const wxString& filePath) {
    ClearForecasts();

    // Open the file
    _filePath = filePath;
    asFileBatchForecasts fileBatch(filePath, asFile::ReadOnly);
    if (!fileBatch.Open()) {
        wxLogError(_("Cannot open the batch file."));
        return false;
    }
    if (!fileBatch.CheckRootElement()) {
        wxLogError(_("Errors were found in the batch file."));
        return false;
    }

    // Get data
    wxXmlNode* node = fileBatch.GetRoot()->GetChildren();
    while (node) {
        if (node->GetName() == "forecasts_output_directory") {
            _forecastsOutputDirectory = asFileBatchForecasts::GetString(node);
        } else if (node->GetName() == "exports_output_directory") {
            _exportsOutputDirectory = asFileBatchForecasts::GetString(node);
        } else if (node->GetName() == "parameters_files_directory") {
            _parametersFileDirectory = asFileBatchForecasts::GetString(node);
        } else if (node->GetName() == "predictors_archive_directory") {
            _predictorsArchiveDirectory = asFileBatchForecasts::GetString(node);
        } else if (node->GetName() == "predictors_realtime_directory") {
            _predictorsRealtimeDirectory = asFileBatchForecasts::GetString(node);
        } else if (node->GetName() == "predictand_db_directory") {
            _predictandDBDirectory = asFileBatchForecasts::GetString(node);
        } else if (node->GetName() == "export_synthesis") {
            _export = (asBatchForecasts::Export)asFileBatchForecasts::GetInt(node);
        } else if (node->GetName() == "forecasts") {
            wxXmlNode* nodeForecast = node->GetChildren();
            while (nodeForecast) {
                if (nodeForecast->GetName() == "filename") {
                    _forecastFileNames.push_back(asFileBatchForecasts::GetString(nodeForecast));
                } else {
                    fileBatch.UnknownNode(nodeForecast);
                }

                nodeForecast = nodeForecast->GetNext();
            }

        } else {
            fileBatch.UnknownNode(node);
        }

        node = node->GetNext();
    }

    return true;
}

bool asBatchForecasts::Save() const {
    // Open the file
    asFileBatchForecasts fileBatch(_filePath, asFile::Replace);
    if (!fileBatch.Open()) return false;

    if (!fileBatch.EditRootElement()) return false;

    // Get general data
    fileBatch.AddChild(fileBatch.CreateNode("forecasts_output_directory", _forecastsOutputDirectory));
    fileBatch.AddChild(fileBatch.CreateNode("exports_output_directory", _exportsOutputDirectory));
    fileBatch.AddChild(fileBatch.CreateNode("parameters_files_directory", _parametersFileDirectory));
    fileBatch.AddChild(fileBatch.CreateNode("predictors_archive_directory", _predictorsArchiveDirectory));
    fileBatch.AddChild(fileBatch.CreateNode("predictors_realtime_directory", _predictorsRealtimeDirectory));
    fileBatch.AddChild(fileBatch.CreateNode("predictand_db_directory", _predictandDBDirectory));
    fileBatch.AddChild(fileBatch.CreateNode("export_synthesis", _export));

    // Forecasts
    wxXmlNode* nodeForecasts = new wxXmlNode(wxXML_ELEMENT_NODE, "forecasts");
    for (int iFcst = 0; iFcst < GetForecastsNb(); iFcst++) {
        nodeForecasts->AddChild(fileBatch.CreateNode("filename", _forecastFileNames[iFcst]));
    }
    fileBatch.AddChild(nodeForecasts);

    fileBatch.Save();

    return true;
}

int asBatchForecasts::GetForecastsNb() const {
    auto forecastsNb = (int)_forecastFileNames.size();
    return forecastsNb;
}

void asBatchForecasts::ClearForecasts() {
    _forecastFileNames.clear();
}

void asBatchForecasts::AddForecast() {
    long nb = _forecastFileNames.size() + 1;
    _forecastFileNames.resize(nb);
}

bool asBatchForecasts::HasExports() const {
    return _export != None;
}