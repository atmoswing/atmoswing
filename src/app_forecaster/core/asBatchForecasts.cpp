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
    ClearModels();

    // Open the file
    m_FilePath = filePath;
    asFileBatchForecasts fileBatch(filePath, asFile::ReadOnly);
    if(!fileBatch.Open())
    {
        asLogError(_("Cannot open the batch file."));
        return false;
    }

    if(!fileBatch.GoToRootElement())
    {
        asLogError(_("Errors were found in the batch file."));
        return false;
    }

    // Get general data
    m_ForecastsOutputDirectory = fileBatch.GetFirstElementAttributeValueText("ForecastsOutputDirectory", "value");
    m_ParametersFileDirectory = fileBatch.GetFirstElementAttributeValueText("ParametersFileDirectory", "value");
    m_PredictorsArchiveDirectory = fileBatch.GetFirstElementAttributeValueText("PredictorsArchiveDirectory", "value");
    m_PredictorsRealtimeDirectory = fileBatch.GetFirstElementAttributeValueText("PredictorsRealtimeDirectory", "value");
    m_PredictandDBDirectory = fileBatch.GetFirstElementAttributeValueText("PredictandDBDirectory", "value");

    // Models
    if(!fileBatch.GoToFirstNodeWithPath("Models"))
    {
        asLogError(_("Errors were found in the batch file."));
        return false;
    }
    
    if(fileBatch.GoToFirstNodeWithPath("Model"))
    {
        // Open new models
        while(true)
        {
            wxString modelName = fileBatch.GetThisElementAttributeValueText("name");
            wxString modelDescr = fileBatch.GetThisElementAttributeValueText("description");
            wxString modelFileName = fileBatch.GetFirstElementAttributeValueText("ModelFileName", "value");
            wxString modelPredictandDB = fileBatch.GetFirstElementAttributeValueText("PredictandDB", "value");

            m_ModelNames.push_back(modelName);
            m_ModelDescriptions.push_back(modelDescr);
            m_ModelFileNames.push_back(modelFileName);
            m_ModelPredictandDBs.push_back(modelPredictandDB);

            // Find the next model
            if (!fileBatch.GoToNextSameNode()) break;
        }
    }
    else
    {
        asLogError(_("Errors were found in the batch file."));
        return false;
    }

    return true;
}

bool asBatchForecasts::Save()
{
    // Open the file
    asFileBatchForecasts fileBatch(m_FilePath, asFile::Replace);
    if(!fileBatch.Open()) return false;

    if(!fileBatch.InsertRootElement()) return false;

    // Get general data
    if(!fileBatch.InsertElementAndAttribute("", "ForecastsOutputDirectory", "value", m_ForecastsOutputDirectory)) return false;
    if(!fileBatch.InsertElementAndAttribute("", "ParametersFileDirectory", "value", m_ParametersFileDirectory)) return false;
    if(!fileBatch.InsertElementAndAttribute("", "PredictorsArchiveDirectory", "value", m_PredictorsArchiveDirectory)) return false;
    if(!fileBatch.InsertElementAndAttribute("", "PredictorsRealtimeDirectory", "value", m_PredictorsRealtimeDirectory)) return false;
    if(!fileBatch.InsertElementAndAttribute("", "PredictandDBDirectory", "value", m_PredictandDBDirectory)) return false;

    // Models
    if(!fileBatch.InsertElement("", "Models")) return false;
    if(!fileBatch.GoToFirstNodeWithPath("Models")) return false;
    
    for (int i_model=0; i_model<GetModelsNb(); i_model++)
    {
        if(!fileBatch.InsertElement("", "Model")) return false;
        if(!fileBatch.GoToLastNodeWithPath("Model")) return false;

        if(!fileBatch.SetElementAttribute("", "name", m_ModelNames[i_model])) return false;
        if(!fileBatch.SetElementAttribute("", "description", m_ModelDescriptions[i_model])) return false;
        if(!fileBatch.InsertElementAndAttribute("", "ModelFileName", "value", m_ModelFileNames[i_model])) return false;
        if(!fileBatch.InsertElementAndAttribute("", "PredictandDB", "value", m_ModelPredictandDBs[i_model])) return false;
        
        if(!fileBatch.GoANodeBack()) return false;
    }

    fileBatch.Save();

    return true;
}

int asBatchForecasts::GetModelsNb()
{
    int modelsNb = (int)m_ModelNames.size();
    return modelsNb;
}

void asBatchForecasts::ClearModels()
{
    m_ModelNames.clear();
    m_ModelDescriptions.clear();
    m_ModelFileNames.clear();
    m_ModelPredictandDBs.clear();
}

void asBatchForecasts::AddModel()
{
    int nb = m_ModelNames.size()+1;
    m_ModelNames.resize(nb);
    m_ModelDescriptions.resize(nb);
    m_ModelFileNames.resize(nb);
    m_ModelPredictandDBs.resize(nb);
}
