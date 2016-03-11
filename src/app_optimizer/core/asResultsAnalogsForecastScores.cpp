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
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#include "asResultsAnalogsForecastScores.h"

#include "asFileNetcdf.h"
#include "asParametersScoring.h"


asResultsAnalogsForecastScores::asResultsAnalogsForecastScores()
:
asResults()
{
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Read("/Optimizer/IntermediateResults/SaveForecastScores", &m_saveIntermediateResults, false);
    wxFileConfig::Get()->Read("/Optimizer/IntermediateResults/LoadForecastScores", &m_loadIntermediateResults, false);
    ThreadsManager().CritSectionConfig().Leave();
}

asResultsAnalogsForecastScores::~asResultsAnalogsForecastScores()
{
    //dtor
}

void asResultsAnalogsForecastScores::Init(asParametersScoring &params)
{
    m_predictandStationIds = params.GetPredictandStationIds();
    if(m_saveIntermediateResults || m_loadIntermediateResults) BuildFileName(params);

    // Resize to 0 to avoid keeping old results
    m_targetDates.resize(0);
    m_forecastScores.resize(0);
    m_forecastScores2DArray.resize(0,0);
}

void asResultsAnalogsForecastScores::BuildFileName(asParametersScoring &params)
{
    ThreadsManager().CritSectionConfig().Enter();
    m_filePath = wxFileConfig::Get()->Read("/Paths/IntermediateResultsDir", asConfig::GetDefaultUserWorkingDir() + "IntermediateResults" + DS);
    ThreadsManager().CritSectionConfig().Leave();
    m_filePath.Append(DS);
    m_filePath.Append(wxString::Format("AnalogsForecastScores_id_%s_step_%d", GetPredictandStationIdsList(), m_currentStep));
    m_filePath.Append(".nc");
}

bool asResultsAnalogsForecastScores::Save(const wxString &AlternateFilePath)
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

    // Get the elements size
    size_t Ntime = m_forecastScores.rows();

    ThreadsManager().CritSectionNetCDF().Enter();

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(ResultsFile, asFileNetcdf::Replace);
    if(!ncFile.Open())
    {
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Define dimensions. Time is the unlimited dimension.
    ncFile.DefDim("time");

    // The dimensions name array is used to pass the dimensions to the variable.
    VectorStdString DimNames1;
    DimNames1.push_back("time");

    // Define variables: the scores and the corresponding dates
    ncFile.DefVar("target_dates", NC_FLOAT, 1, DimNames1);
    ncFile.DefVar("forecast_scores", NC_FLOAT, 1, DimNames1);

    // Put attributes
    DefTargetDatesAttributes(ncFile);
    DefForecastScoresAttributes(ncFile);

    // End definitions: leave define mode
    ncFile.EndDef();

    // Provide sizes for variables
    size_t start1D[] = {0};
    size_t count1D[] = {Ntime};

    // Write data
    ncFile.PutVarArray("target_dates", start1D, count1D, &m_targetDates(0));
    ncFile.PutVarArray("forecast_scores", start1D, count1D, &m_forecastScores(0));

    // Close:save new netCDF dataset
    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();

    return true;
}

bool asResultsAnalogsForecastScores::Load(const wxString &AlternateFilePath)
{
    // If we don't want to save or the file doesn't exist
    if(!m_loadIntermediateResults) return false;
    if(!Exists()) return false;
    if(m_currentStep!=0) return false;

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

    // Open the NetCDF file
    asFileNetcdf ncFile(ResultsFile, asFileNetcdf::ReadOnly);
    if(!ncFile.Open())
    {
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Get the elements size
    int TimeLength = ncFile.GetDimLength("time");

    // Get time and data
    m_targetDates.resize( TimeLength );
    ncFile.GetVar("target_dates", &m_targetDates[0]);
    m_forecastScores.resize( TimeLength );
    ncFile.GetVar("forecast_scores", &m_forecastScores[0]);

    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();

    return true;
}
