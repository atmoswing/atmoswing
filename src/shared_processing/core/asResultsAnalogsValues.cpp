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

#include "asResultsAnalogsValues.h"

#include "asFileNetcdf.h"


asResultsAnalogsValues::asResultsAnalogsValues()
        : asResults()
{
}

asResultsAnalogsValues::~asResultsAnalogsValues()
{
}

void asResultsAnalogsValues::Init(asParameters &params)
{
    m_predictandStationIds = params.GetPredictandStationIds();

    // Resize to 0 to avoid keeping old results
    m_targetDates.resize(0);
    m_targetValuesNorm.resize(0);
    m_targetValuesGross.resize(0);
    m_analogsCriteria.resize(0, 0);
    m_analogsValuesNorm.resize(0);
    m_analogsValuesGross.resize(0);
}

void asResultsAnalogsValues::BuildFileName()
{
    ThreadsManager().CritSectionConfig().Enter();
    m_filePath = wxFileConfig::Get()->Read("/Paths/OptimizerResultsDir", asConfig::GetDefaultUserWorkingDir());
    ThreadsManager().CritSectionConfig().Leave();
    m_filePath.Append(DS);
    m_filePath.Append(wxString::Format("AnalogsValues_id_%s_step_%d", GetPredictandStationIdsList(), m_currentStep));
    m_filePath.Append(".nc");
}

bool asResultsAnalogsValues::Save()
{
    BuildFileName();

    // Get the elements size
    size_t Ntime = (size_t)m_analogsCriteria.rows();
    size_t Nanalogs = (size_t)m_analogsCriteria.cols();
    size_t Nstations = m_predictandStationIds.size();

    ThreadsManager().CritSectionNetCDF().Enter();

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(m_filePath, asFileNetcdf::Replace);
    if (!ncFile.Open()) {
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Define dimensions.
    ncFile.DefDim("stations", Nstations);
    ncFile.DefDim("time", Ntime);
    ncFile.DefDim("analogs", Nanalogs);

    // The dimensions name array is used to pass the dimensions to the variable.
    VectorStdString dimS;
    dimS.push_back("stations");
    VectorStdString dimT;
    dimT.push_back("time");
    VectorStdString dimTA;
    dimTA.push_back("time");
    dimTA.push_back("analogs");
    VectorStdString dimST;
    dimST.push_back("stations");
    dimST.push_back("time");
    VectorStdString dimSTA;
    dimSTA.push_back("stations");
    dimSTA.push_back("time");
    dimSTA.push_back("analogs");

    // Define variables: the analogcriteria and the corresponding dates
    ncFile.DefVar("stations", NC_INT, 1, dimS);
    ncFile.DefVar("target_dates", NC_FLOAT, 1, dimT);
    ncFile.DefVar("target_values_norm", NC_FLOAT, 2, dimST);
    ncFile.DefVar("target_values_gross", NC_FLOAT, 2, dimST);
    ncFile.DefVar("analog_criteria", NC_FLOAT, 2, dimTA);
    ncFile.DefVar("analog_values_norm", NC_FLOAT, 3, dimSTA);
    ncFile.DefVar("analog_values_gross", NC_FLOAT, 3, dimSTA);

    // Put attributes
    DefTargetDatesAttributes(ncFile);
    DefTargetValuesNormAttributes(ncFile);
    DefTargetValuesGrossAttributes(ncFile);
    DefAnalogsCriteriaAttributes(ncFile);
    DefAnalogsValuesNormAttributes(ncFile);
    DefAnalogsValuesGrossAttributes(ncFile);

    // End definitions: leave define mode
    ncFile.EndDef();

    // Provide sizes for variables
    size_t startS[] = {0};
    size_t countS[] = {Nstations};
    size_t startT[] = {0};
    size_t countT[] = {Ntime};
    size_t startTA[] = {0, 0};
    size_t countTA[] = {Ntime, Nanalogs};
    size_t startST[] = {0, 0};
    size_t countST[] = {Nstations, Ntime};
    size_t startSTA[] = {0, 0, 0};
    size_t countSTA[] = {Nstations, Ntime, Nanalogs};

    // Write data
    ncFile.PutVarArray("stations", startS, countS, &m_targetDates(0));
    ncFile.PutVarArray("target_dates", startT, countT, &m_targetDates(0));
    ncFile.PutVarArray("target_values_norm", startST, countST, &m_targetValuesNorm[0](0));
    ncFile.PutVarArray("target_values_gross", startST, countST, &m_targetValuesGross[0](0));
    ncFile.PutVarArray("analog_criteria", startTA, countTA, &m_analogsCriteria(0));
    ncFile.PutVarArray("analog_values_norm", startSTA, countSTA, &m_analogsValuesNorm[0](0));
    ncFile.PutVarArray("analog_values_gross", startSTA, countSTA, &m_analogsValuesGross[0](0));

    // Close:save new netCDF dataset
    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();

    return true;
}

bool asResultsAnalogsValues::Load()
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
    size_t Nstations = ncFile.GetDimLength("stations");
    size_t Ntime = ncFile.GetDimLength("time");
    size_t Nanalogs = ncFile.GetDimLength("analogs");

    // Get time
    m_targetDates.resize(Ntime);
    ncFile.GetVar("target_dates", &m_targetDates[0]);

    // Sizes
    size_t startTA[] = {0, 0};
    size_t countTA[] = {Ntime, Nanalogs};
    size_t startST[] = {0, 0};
    size_t countST[] = {Nstations, Ntime};
    size_t startSTA[] = {0, 0, 0};
    size_t countSTA[] = {Nstations, Ntime, Nanalogs};

    // Resize containers
    m_predictandStationIds.resize(Nstations);
    m_targetValuesNorm.resize(Nstations, Array1DFloat(Ntime));
    m_targetValuesGross.resize(Nstations, Array1DFloat(Ntime));
    m_analogsCriteria.resize(Ntime, Nanalogs);
    m_analogsValuesNorm.resize(Nstations, Array2DFloat(Ntime, Nanalogs));
    m_analogsValuesGross.resize(Nstations, Array2DFloat(Ntime, Nanalogs));

    // Get data
    ncFile.GetVar("stations", &m_predictandStationIds[0]);
    ncFile.GetVarArray("target_values_norm", startST, countST, &m_targetValuesNorm[0](0));
    ncFile.GetVarArray("target_values_gross", startST, countST, &m_targetValuesGross[0](0));
    ncFile.GetVarArray("analog_criteria", startTA, countTA, &m_analogsCriteria(0));
    ncFile.GetVarArray("analog_values_norm", startSTA, countSTA, &m_analogsValuesNorm[0](0));
    ncFile.GetVarArray("analog_values_gross", startSTA, countSTA, &m_analogsValuesGross[0](0));

    ThreadsManager().CritSectionNetCDF().Leave();

    ncFile.Close();

    return true;
}
