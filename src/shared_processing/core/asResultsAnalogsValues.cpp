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
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Read("/Optimizer/IntermediateResults/SaveAnalogValues", &m_saveIntermediateResults, false);
    wxFileConfig::Get()->Read("/Optimizer/IntermediateResults/LoadAnalogValues", &m_loadIntermediateResults, false);
    ThreadsManager().CritSectionConfig().Leave();
}

asResultsAnalogsValues::~asResultsAnalogsValues()
{
    //dtor
}

void asResultsAnalogsValues::Init(asParameters &params)
{
    m_predictandStationIds = params.GetPredictandStationIds();
    if (m_saveIntermediateResults || m_loadIntermediateResults)
        BuildFileName();

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
    m_filePath = wxFileConfig::Get()->Read("/Paths/IntermediateResultsDir",
                                           asConfig::GetDefaultUserWorkingDir() + "IntermediateResults" + DS);
    ThreadsManager().CritSectionConfig().Leave();
    m_filePath.Append(DS);
    m_filePath.Append(wxString::Format("AnalogsValues_id_%s_step_%d", GetPredictandStationIdsList(), m_currentStep));
    m_filePath.Append(".nc");
}

bool asResultsAnalogsValues::Save(const wxString &AlternateFilePath)
{
    // If we don't want to save, skip
    if (!m_saveIntermediateResults)
        return false;
    wxString message = _("Saving intermediate file: ") + m_filePath;
    asLogMessage(message);

    // Get the file path
    wxString ResultsFile;
    if (AlternateFilePath.IsEmpty()) {
        ResultsFile = m_filePath;
    } else {
        ResultsFile = AlternateFilePath;
    }

    // Get the elements size
    size_t Ntime = m_analogsCriteria.rows();
    size_t Nanalogs = m_analogsCriteria.cols();
    size_t Nstations = m_predictandStationIds.size();

    ThreadsManager().CritSectionNetCDF().Enter();

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(ResultsFile, asFileNetcdf::Replace);
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

    // Set the matrices in vectors
    int totLength = Ntime * Nanalogs;
    VectorFloat analogsCriteria(totLength);
    VectorFloat analogsValuesNorm(Nstations * totLength);
    VectorFloat analogsValuesGross(Nstations * totLength);
    int ind = 0;

    // Fill the criteria data
    for (unsigned int i_time = 0; i_time < Ntime; i_time++) {
        for (unsigned int i_analog = 0; i_analog < Nanalogs; i_analog++) {
            ind = i_analog;
            ind += i_time * Nanalogs;
            analogsCriteria[ind] = m_analogsCriteria(i_time, i_analog);
        }
    }

    // Fill the values data
    for (unsigned int i_st = 0; i_st < Nstations; i_st++) {
        for (unsigned int i_time = 0; i_time < Ntime; i_time++) {
            for (unsigned int i_analog = 0; i_analog < Nanalogs; i_analog++) {
                ind = i_analog;
                ind += i_time * Nanalogs;
                analogsValuesNorm[ind] = m_analogsValuesNorm[i_st](i_time, i_analog);
                analogsValuesGross[ind] = m_analogsValuesGross[i_st](i_time, i_analog);
            }
        }
    }

    // Write data
    ncFile.PutVarArray("stations", startS, countS, &m_targetDates(0));
    ncFile.PutVarArray("target_dates", startT, countT, &m_targetDates(0));
    ncFile.PutVarArray("target_values_norm", startST, countST, &m_targetValuesNorm[0]);
    ncFile.PutVarArray("target_values_gross", startST, countST, &m_targetValuesGross[0]);
    ncFile.PutVarArray("analog_criteria", startTA, countTA, &analogsCriteria[0]);
    ncFile.PutVarArray("analog_values_norm", startSTA, countSTA, &analogsValuesNorm[0]);
    ncFile.PutVarArray("analog_values_gross", startSTA, countSTA, &analogsValuesGross[0]);

    // Close:save new netCDF dataset
    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();

    return true;
}

bool asResultsAnalogsValues::Load(const wxString &AlternateFilePath)
{
    // If we don't want to save or the file doesn't exist
    if (!m_loadIntermediateResults)
        return false;
    if (!Exists())
        return false;
    if (m_currentStep != 0)
        return false;

    // Get the file path
    wxString ResultsFile;
    if (AlternateFilePath.IsEmpty()) {
        ResultsFile = m_filePath;
    } else {
        ResultsFile = AlternateFilePath;
    }

    ThreadsManager().CritSectionNetCDF().Enter();

    // Open the NetCDF file
    asFileNetcdf ncFile(ResultsFile, asFileNetcdf::ReadOnly);
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

    // Create vectors for matrices data
    VectorFloat targetValuesNorm(Nstations * Ntime);
    VectorFloat targetValuesGross(Nstations * Ntime);
    VectorFloat analogsCriteria(Ntime * Nanalogs);
    VectorFloat analogsValuesNorm(Nstations * Ntime * Nanalogs);
    VectorFloat analogsValuesGross(Nstations * Ntime * Nanalogs);

    // Sizes
    size_t startTA[] = {0, 0};
    size_t countTA[] = {Ntime, Nanalogs};
    size_t startST[] = {0, 0};
    size_t countST[] = {Nstations, Ntime};
    size_t startSTA[] = {0, 0, 0};
    size_t countSTA[] = {Nstations, Ntime, Nanalogs};

    // Get data
    m_predictandStationIds.resize(Nstations);
    ncFile.GetVar("stations", &m_predictandStationIds[0]);
    ncFile.GetVarArray("target_values_norm", startST, countST, &targetValuesNorm[0]);
    ncFile.GetVarArray("target_values_gross", startST, countST, &targetValuesGross[0]);
    ncFile.GetVarArray("analog_criteria", startTA, countTA, &analogsCriteria[0]);
    ncFile.GetVarArray("analog_values_norm", startSTA, countSTA, &analogsValuesNorm[0]);
    ncFile.GetVarArray("analog_values_gross", startSTA, countSTA, &analogsValuesGross[0]);

    // Set data into the matrices
    m_analogsCriteria.resize(Ntime, Nanalogs);
    int ind = 0;
    for (size_t i_time = 0; i_time < Ntime; i_time++) {
        for (int i_analog = 0; i_analog < (int) Nanalogs; i_analog++) {
            ind = i_analog;
            ind += i_time * Nanalogs;
            m_analogsCriteria(i_time, i_analog) = analogsCriteria[ind];
        }
    }

    for (size_t i_st = 0; i_st < Nstations; i_st++) {
        Array2DFloat analogsValuesNormStation(Ntime, Nanalogs);
        Array2DFloat analogsValuesGrossStation(Ntime, Nanalogs);
        int ind = 0;
        for (int i_time = 0; i_time < (int) Ntime; i_time++) {
            for (int i_analog = 0; i_analog < (int) Nanalogs; i_analog++) {
                ind = i_analog;
                ind += i_time * Nanalogs;
                m_analogsCriteria(i_time, i_analog) = analogsCriteria[ind];
                analogsValuesNormStation(i_time, i_analog) = analogsValuesNorm[ind];
                analogsValuesGrossStation(i_time, i_analog) = analogsValuesGross[ind];
            }
        }
        m_analogsValuesNorm.push_back(analogsValuesNormStation);
        m_analogsValuesGross.push_back(analogsValuesGrossStation);
    }

    ThreadsManager().CritSectionNetCDF().Leave();

    ncFile.Close();

    return true;
}
