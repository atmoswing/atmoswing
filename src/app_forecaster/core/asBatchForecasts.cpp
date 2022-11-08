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

asBatchForecasts::asBatchForecasts()
    : wxObject(),
      m_hasChanged(false),
      m_export(None) {
    wxString baseDir = asConfig::GetDocumentsDir() + "AtmoSwing" + DS;
    m_filePath = baseDir + "Parameters" + DS + "BatchForecasts.asfb";
    m_forecastsOutputDirectory = baseDir + "Forecasts";
    m_exportsOutputDirectory = baseDir + "Exports";
    m_parametersFileDirectory = baseDir + "Parameters";
    m_predictorsArchiveDirectory = baseDir + "Data" + DS + "Archive predictors";
    m_predictorsRealtimeDirectory = baseDir + "Data" + DS + "Forecasted predictors";
    m_predictandDBDirectory = baseDir + "Data" + DS + "Predictands";
}

bool asBatchForecasts::Load(const wxString& filePath) {
    ClearForecasts();

    // Open the file
    m_filePath = filePath;
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
            m_forecastsOutputDirectory = asFileBatchForecasts::GetString(node);
        } else if (node->GetName() == "exports_output_directory") {
            m_exportsOutputDirectory = asFileBatchForecasts::GetString(node);
        } else if (node->GetName() == "parameters_files_directory") {
            m_parametersFileDirectory = asFileBatchForecasts::GetString(node);
        } else if (node->GetName() == "predictors_archive_directory") {
            m_predictorsArchiveDirectory = asFileBatchForecasts::GetString(node);
        } else if (node->GetName() == "predictors_realtime_directory") {
            m_predictorsRealtimeDirectory = asFileBatchForecasts::GetString(node);
        } else if (node->GetName() == "predictand_db_directory") {
            m_predictandDBDirectory = asFileBatchForecasts::GetString(node);
        } else if (node->GetName() == "export_synthesis") {
            m_export = (asBatchForecasts::Export)asFileBatchForecasts::GetInt(node);
        } else if (node->GetName() == "forecasts") {
            wxXmlNode* nodeForecast = node->GetChildren();
            while (nodeForecast) {
                if (nodeForecast->GetName() == "filename") {
                    m_forecastFileNames.push_back(asFileBatchForecasts::GetString(nodeForecast));
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
    asFileBatchForecasts fileBatch(m_filePath, asFile::Replace);
    if (!fileBatch.Open()) return false;

    if (!fileBatch.EditRootElement()) return false;

    // Get general data
    fileBatch.AddChild(fileBatch.CreateNodeWithValue("forecasts_output_directory", m_forecastsOutputDirectory));
    fileBatch.AddChild(fileBatch.CreateNodeWithValue("exports_output_directory", m_exportsOutputDirectory));
    fileBatch.AddChild(fileBatch.CreateNodeWithValue("parameters_files_directory", m_parametersFileDirectory));
    fileBatch.AddChild(fileBatch.CreateNodeWithValue("predictors_archive_directory", m_predictorsArchiveDirectory));
    fileBatch.AddChild(fileBatch.CreateNodeWithValue("predictors_realtime_directory", m_predictorsRealtimeDirectory));
    fileBatch.AddChild(fileBatch.CreateNodeWithValue("predictand_db_directory", m_predictandDBDirectory));
    fileBatch.AddChild(fileBatch.CreateNodeWithValue("export_synthesis", m_export));

    // Forecasts
    wxXmlNode* nodeForecasts = new wxXmlNode(wxXML_ELEMENT_NODE, "forecasts");
    for (int iFcst = 0; iFcst < GetForecastsNb(); iFcst++) {
        nodeForecasts->AddChild(fileBatch.CreateNodeWithValue("filename", m_forecastFileNames[iFcst]));
    }
    fileBatch.AddChild(nodeForecasts);

    fileBatch.Save();

    return true;
}

int asBatchForecasts::GetForecastsNb() const {
    auto forecastsNb = (int)m_forecastFileNames.size();
    return forecastsNb;
}

void asBatchForecasts::ClearForecasts() {
    m_forecastFileNames.clear();
}

void asBatchForecasts::AddForecast() {
    long nb = m_forecastFileNames.size() + 1;
    m_forecastFileNames.resize(nb);
}

bool asBatchForecasts::HasExports() const {
    return m_export != None;
}