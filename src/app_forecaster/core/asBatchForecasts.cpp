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

#include "asBatchForecasts.h"

asBatchForecasts::asBatchForecasts()
:
wxObject()
{
    m_HasChanged = false;
    m_FilePath = asConfig::GetDocumentsDir() + "AtmoSwing" + DS + "Parameters" + DS + "BatchForecasts.xml";
    m_ForecastsOutputDirectory = asConfig::GetDocumentsDir() + "AtmoSwing" + DS + "Forecasts";
    m_ParametersFileDirectory = asConfig::GetDocumentsDir() + "AtmoSwing" + DS + "Parameters";
    m_PredictorsArchiveDirectory = asConfig::GetDocumentsDir() + "AtmoSwing" + DS + "Data" + DS + "Archive predictors";
    m_PredictorsRealtimeDirectory = asConfig::GetDocumentsDir() + "AtmoSwing" + DS + "Data" + DS + "Forecasted predictors";
    m_PredictandDBDirectory = asConfig::GetDocumentsDir() + "AtmoSwing" + DS + "Data" + DS + "Predictands";
}

asBatchForecasts::~asBatchForecasts()
{
    //dtor
}

bool asBatchForecasts::Load(const wxString &filePath)
{
    ClearForecasts();

    // Open the file
    m_FilePath = filePath;
    asFileBatchForecasts fileBatch(filePath, asFile::ReadOnly);
    if(!fileBatch.Open())
    {
        asLogError(_("Cannot open the batch file."));
        return false;
    }
    if(!fileBatch.CheckRootElement())
    {
        asLogError(_("Errors were found in the batch file."));
        return false;
    }

    // Get data
    wxXmlNode *node = fileBatch.GetRoot()->GetChildren();
    while (node) {
        if (node->GetName() == "forecasts_output_directory") {
            m_ForecastsOutputDirectory = fileBatch.GetString(node);
        } else if (node->GetName() == "parameters_files_directory") {
            m_ParametersFileDirectory = fileBatch.GetString(node);
        } else if (node->GetName() == "predictors_archive_directory") {
            m_PredictorsArchiveDirectory = fileBatch.GetString(node);
        } else if (node->GetName() == "predictors_realtime_directory") {
            m_PredictorsRealtimeDirectory = fileBatch.GetString(node);
        } else if (node->GetName() == "predictand_db_directory") {
            m_PredictandDBDirectory = fileBatch.GetString(node);
        } else if (node->GetName() == "forecasts") {
            wxXmlNode *nodeForecast = node->GetChildren();
            while (nodeForecast) {
                if (nodeForecast->GetName() == "filename") {
                    m_ForecastFileNames.push_back(fileBatch.GetString(nodeForecast));
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

bool asBatchForecasts::Save()
{
    // Open the file
    asFileBatchForecasts fileBatch(m_FilePath, asFile::Replace);
    if(!fileBatch.Open()) return false;

    if(!fileBatch.EditRootElement()) return false;

    // Get general data
    fileBatch.AddChild(fileBatch.CreateNodeWithValue("forecasts_output_directory", m_ForecastsOutputDirectory));
    fileBatch.AddChild(fileBatch.CreateNodeWithValue("parameters_files_directory", m_ParametersFileDirectory));
    fileBatch.AddChild(fileBatch.CreateNodeWithValue("predictors_archive_directory", m_PredictorsArchiveDirectory));
    fileBatch.AddChild(fileBatch.CreateNodeWithValue("predictors_realtime_directory", m_PredictorsRealtimeDirectory));
    fileBatch.AddChild(fileBatch.CreateNodeWithValue("predictand_db_directory", m_PredictandDBDirectory));

    // Forecasts
    wxXmlNode * nodeForecasts = new wxXmlNode(wxXML_ELEMENT_NODE ,"forecasts" );
    for (int i_forecast=0; i_forecast<GetForecastsNb(); i_forecast++)
    {
        nodeForecasts->AddChild(fileBatch.CreateNodeWithValue("filename", m_ForecastFileNames[i_forecast]));
    }
    fileBatch.AddChild(nodeForecasts);

    fileBatch.Save();

    return true;
}

int asBatchForecasts::GetForecastsNb()
{
    int forecastsNb = (int)m_ForecastFileNames.size();
    return forecastsNb;
}

void asBatchForecasts::ClearForecasts()
{
    m_ForecastFileNames.clear();
}

void asBatchForecasts::AddForecast()
{
    int nb = m_ForecastFileNames.size()+1;
    m_ForecastFileNames.resize(nb);
}
