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
        : asResults()
{
}

asResultsAnalogsForecastScores::~asResultsAnalogsForecastScores()
{
}

void asResultsAnalogsForecastScores::Init(asParametersScoring &params)
{
    m_predictandStationIds = params.GetPredictandStationIds();

    // Resize to 0 to avoid keeping old results
    m_targetDates.resize(0);
    m_forecastScores.resize(0);
    m_forecastScores2DArray.resize(0, 0);
}

void asResultsAnalogsForecastScores::BuildFileName()
{
    ThreadsManager().CritSectionConfig().Enter();
    m_filePath = wxFileConfig::Get()->Read("/Paths/OptimizerResultsDir", asConfig::GetDefaultUserWorkingDir());
    ThreadsManager().CritSectionConfig().Leave();
    if (!m_subFolder.IsEmpty()) {
        m_filePath.Append(DS);
        m_filePath.Append(m_subFolder);
    }
    m_filePath.Append(DS);
    m_filePath.Append(wxString::Format("AnalogsForecastScores_id_%s_step_%d", GetPredictandStationIdsList(),
                                       m_currentStep));
    m_filePath.Append(".nc");
}

bool asResultsAnalogsForecastScores::Save()
{
    BuildFileName();

    wxString message = _("Saving intermediate file: ") + m_filePath;
    asLogMessage(message);

    // Get the elements size
    size_t Ntime = (size_t)m_forecastScores.rows();

    ThreadsManager().CritSectionNetCDF().Enter();

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(m_filePath, asFileNetcdf::Replace);
    if (!ncFile.Open()) {
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

bool asResultsAnalogsForecastScores::Load()
{
    BuildFileName();

    if (!Exists())
        return false;

    ThreadsManager().CritSectionNetCDF().Enter();

    // Open the NetCDF file
    asFileNetcdf ncFile(m_filePath, asFileNetcdf::ReadOnly);
    if (!ncFile.Open()) {
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Get the elements size
    size_t TimeLength = ncFile.GetDimLength("time");

    // Resize
    m_targetDates.resize(TimeLength);
    m_forecastScores.resize(TimeLength);

    // Get time and data
    ncFile.GetVar("target_dates", &m_targetDates[0]);
    ncFile.GetVar("forecast_scores", &m_forecastScores[0]);

    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();

    return true;
}
