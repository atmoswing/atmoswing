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
 * Portions Copyright 2008-2013 University of Lausanne.
 * Portions Copyright 2013 Pascal Horton, Terr@num.
 */

#include "asResultsAnalogsForecastScoreFinal.h"

#include "asFileNetcdf.h"
#include "asParametersScoring.h"


asResultsAnalogsForecastScoreFinal::asResultsAnalogsForecastScoreFinal()
:
asResults()
{
    m_hasSingleValue = true;
    m_forecastScore = NaNFloat;

    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Read("/Optimizer/IntermediateResults/SaveFinalForecastScore", &m_saveIntermediateResults, false);
    wxFileConfig::Get()->Read("/Optimizer/IntermediateResults/LoadFinalForecastScore", &m_loadIntermediateResults, false);
    ThreadsManager().CritSectionConfig().Leave();
}

asResultsAnalogsForecastScoreFinal::~asResultsAnalogsForecastScoreFinal()
{
    //dtor
}

void asResultsAnalogsForecastScoreFinal::Init(asParametersScoring &params)
{
    if(m_saveIntermediateResults || m_loadIntermediateResults) BuildFileName(params);

    // Set to nan to avoid keeping old results
    m_forecastScore = NaNFloat;
    m_forecastScoreArray.resize(0);
}

void asResultsAnalogsForecastScoreFinal::BuildFileName(asParametersScoring &params)
{
    ThreadsManager().CritSectionConfig().Enter();
    m_filePath = wxFileConfig::Get()->Read("/Paths/IntermediateResultsDir", asConfig::GetDefaultUserWorkingDir() + "IntermediateResults" + DS);
    ThreadsManager().CritSectionConfig().Leave();
    m_filePath.Append(DS);
    m_filePath.Append(wxString::Format("AnalogsForecastScoreFinal_id_%s_step_%d", GetPredictandStationIdsList(), m_currentStep));
    m_filePath.Append(".nc");
}

bool asResultsAnalogsForecastScoreFinal::Save(const wxString &AlternateFilePath)
{
    // If we don't want to save, skip
    if(!m_saveIntermediateResults) return false;
    wxString message = _("Saving intermediate file: ") + m_filePath;
    asLogMessage(message);

    // Get the file path
    wxString ResultsFile;
    if (AlternateFilePath.IsEmpty())
    {
        ResultsFile = m_filePath;
    }
    else
    {
        ResultsFile = AlternateFilePath;
    }

    ThreadsManager().CritSectionNetCDF().Enter();

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(ResultsFile, asFileNetcdf::Replace);
    if(!ncFile.Open())
    {
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Define dimensions.
    ncFile.DefDim("forecast_score");

    // The dimensions name array is used to pass the dimensions to the variable.
    VectorStdString DimNames1;
    DimNames1.push_back("forecast_score");

    // Define variables
    ncFile.DefVar("forecast_score", NC_FLOAT, 1, DimNames1);

    // Put attributes
    DefForecastScoreFinalAttributes(ncFile);

    // End definitions: leave define mode
    ncFile.EndDef();

    // Provide sizes for variables
    size_t start1D[] = {0};
    size_t count1D[] = {1};

    // Write data
    ncFile.PutVarArray("forecast_score", start1D, count1D, &m_forecastScore);

    // Close:save new netCDF dataset
    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();

    return true;
}

bool asResultsAnalogsForecastScoreFinal::Load(const wxString &AlternateFilePath)
{
    // Makes no sense to load at this stage.
    return false;
}
