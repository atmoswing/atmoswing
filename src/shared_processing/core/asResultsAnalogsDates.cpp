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

#include "asResultsAnalogsDates.h"

#include "asFileNetcdf.h"


asResultsAnalogsDates::asResultsAnalogsDates()
        : asResults()
{
}

asResultsAnalogsDates::~asResultsAnalogsDates()
{
}

void asResultsAnalogsDates::Init(asParameters &params)
{
    m_predictandStationIds = params.GetPredictandStationIds();

    // Resize to 0 to avoid keeping old results
    m_targetDates.resize(0);
    m_analogsCriteria.resize(0, 0);
    m_analogsDates.resize(0, 0);
}

void asResultsAnalogsDates::BuildFileName()
{
    ThreadsManager().CritSectionConfig().Enter();
    m_filePath = wxFileConfig::Get()->Read("/Paths/OptimizerResultsDir", asConfig::GetDefaultUserWorkingDir());
    ThreadsManager().CritSectionConfig().Leave();
    if (!m_subFolder.IsEmpty()) {
        m_filePath.Append(DS);
        m_filePath.Append(m_subFolder);
    }
    m_filePath.Append(DS);
    m_filePath.Append(wxString::Format("AnalogsDates_id_%s_step_%d", GetPredictandStationIdsList(), m_currentStep));
    m_filePath.Append(".nc");
}

bool asResultsAnalogsDates::Save()
{
    BuildFileName();

    // Get the elements size
    size_t Ntime = (size_t) m_analogsCriteria.rows();
    size_t Nanalogs = (size_t) m_analogsCriteria.cols();

    ThreadsManager().CritSectionNetCDF().Enter();

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(m_filePath, asFileNetcdf::Replace);
    if (!ncFile.Open()) {
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Define dimensions. Time is the unlimited dimension.
    ncFile.DefDim("time");
    ncFile.DefDim("analogs", Nanalogs);

    // The dimensions name array is used to pass the dimensions to the variable.
    VectorStdString DimNames1;
    DimNames1.push_back("time");
    VectorStdString DimNames2;
    DimNames2.push_back("time");
    DimNames2.push_back("analogs");

    // Define variables: the analogcriteria and the corresponding dates
    ncFile.DefVar("target_dates", NC_FLOAT, 1, DimNames1);
    ncFile.DefVar("analog_criteria", NC_FLOAT, 2, DimNames2);
    ncFile.DefVar("analog_dates", NC_FLOAT, 2, DimNames2);
    ncFile.DefVarDeflate("analog_dates");

    // Put attributes
    DefTargetDatesAttributes(ncFile);
    DefAnalogsCriteriaAttributes(ncFile);
    DefAnalogsDatesAttributes(ncFile);

    // End definitions: leave define mode
    ncFile.EndDef();

    // Provide sizes for variables
    size_t start1D[] = {0};
    size_t count1D[] = {Ntime};
    size_t start2D[] = {0, 0};
    size_t count2D[] = {Ntime, Nanalogs};

    // Write data
    ncFile.PutVarArray("target_dates", start1D, count1D, &m_targetDates(0));
    ncFile.PutVarArray("analog_criteria", start2D, count2D, &m_analogsCriteria(0));
    ncFile.PutVarArray("analog_dates", start2D, count2D, &m_analogsDates(0));

    // Close:save new netCDF dataset
    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();
    return true;
}

bool asResultsAnalogsDates::Load()
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
    size_t Ntime = ncFile.GetDimLength("time");
    size_t Nanalogs = ncFile.GetDimLength("analogs");

    // Get time
    m_targetDates.resize(Ntime);
    ncFile.GetVar("target_dates", &m_targetDates[0]);

    // Check last value
    if (m_targetDates[m_targetDates.size() - 1] < m_targetDates[0]) {
        asLogError(_("The target date array is not consistent in the temp file (last value makes no sense)."));
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Sizes
    size_t startTA[] = {0, 0};
    size_t countTA[] = {Ntime, Nanalogs};

    // Resize containers
    m_analogsCriteria.resize(Ntime, Nanalogs);
    m_analogsDates.resize(Ntime, Nanalogs);

    // Get data
    ncFile.GetVarArray("analog_criteria", startTA, countTA, &m_analogsCriteria(0));
    ncFile.GetVarArray("analog_dates", startTA, countTA, &m_analogsDates(0));

    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();
    return true;
}
