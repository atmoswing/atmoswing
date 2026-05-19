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

#include "asResultsValues.h"

#include "asFileNetcdf.h"

asResultsValues::asResultsValues()
    : asResults() {}

asResultsValues::~asResultsValues() {}

void asResultsValues::Init(asParameters* params) {
    _predictandStationIds = params->GetPredictandStationIds();

    // Resize to 0 to avoid keeping old results
    _targetDates.resize(0);
    _targetValuesNorm.resize(0);
    _targetValuesRaw.resize(0);
    _analogsCriteria.resize(0, 0);
    _analogsValuesNorm.resize(0);
    _analogsValuesRaw.resize(0);
}

void asResultsValues::BuildFileName() {
    ThreadsManager().CritSectionConfig().Enter();
    _filePath = wxFileConfig::Get()->Read("/Paths/ResultsDir", asConfig::GetDefaultUserWorkingDir());
    ThreadsManager().CritSectionConfig().Leave();
    if (!_subFolder.IsEmpty()) {
        _filePath.Append(DS);
        _filePath.Append(_subFolder);
    }
    _filePath.Append(DS);
    _filePath.Append(asStrF("AnalogValues_id_%s_step_%d", GetPredictandStationIdsList(), _currentStep));
    _filePath.Append(".nc");
}

bool asResultsValues::Save() {
    BuildFileName();

    // Get the elements size
    size_t nTime = (size_t)_analogsCriteria.rows();
    size_t nAnalogs = (size_t)_analogsCriteria.cols();
    size_t nStations = _predictandStationIds.size();

    ThreadsManager().CritSectionNetCDF().Enter();

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(_filePath, asFileNetcdf::Replace);
    if (!ncFile.Open()) {
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Define dimensions.
    ncFile.DefDim("stations", nStations);
    ncFile.DefDim("time", nTime);
    ncFile.DefDim("analogs", nAnalogs);

    // The dimensions name array is used to pass the dimensions to the variable.
    vstds dimS;
    dimS.push_back("stations");
    vstds dimT;
    dimT.push_back("time");
    vstds dimTA;
    dimTA.push_back("time");
    dimTA.push_back("analogs");
    vstds dimST;
    dimST.push_back("stations");
    dimST.push_back("time");
    vstds dimSTA;
    dimSTA.push_back("stations");
    dimSTA.push_back("time");
    dimSTA.push_back("analogs");

    // Define variables: the analogcriteria and the corresponding dates
    ncFile.DefVar("stations", NC_INT, 1, dimS);
    ncFile.DefVar("target_dates", NC_FLOAT, 1, dimT);
    ncFile.DefVar("target_values_norm", NC_FLOAT, 2, dimST);
    ncFile.DefVar("target_values_raw", NC_FLOAT, 2, dimST);
    ncFile.DefVar("analog_criteria", NC_FLOAT, 2, dimTA);
    ncFile.DefVarDeflate("analog_criteria");
    ncFile.DefVar("analog_values_norm", NC_FLOAT, 3, dimSTA);
    ncFile.DefVarDeflate("analog_values_norm");
    ncFile.DefVar("analog_values_raw", NC_FLOAT, 3, dimSTA);
    ncFile.DefVarDeflate("analog_values_raw");

    // Put attributes
    DefTargetDatesAttributes(ncFile);
    DefTargetValuesNormAttributes(ncFile);
    DefTargetValuesRawAttributes(ncFile);
    DefAnalogsCriteriaAttributes(ncFile);
    DefAnalogsValuesNormAttributes(ncFile);
    DefAnalogsValuesRawAttributes(ncFile);

    // End definitions: leave define mode
    ncFile.EndDef();

    // Provide sizes for variables
    size_t startS[] = {0};
    size_t countS[] = {nStations};
    size_t startT[] = {0};
    size_t countT[] = {nTime};
    size_t startTA[] = {0, 0};
    size_t countTA[] = {nTime, nAnalogs};
    size_t startST[] = {0, 0};
    size_t countST[] = {nStations, nTime};
    size_t startSTA[] = {0, 0, 0};
    size_t countSTA[] = {nStations, nTime, nAnalogs};

    // Write data
    ncFile.PutVarArray("stations", startS, countS, &_predictandStationIds[0]);
    ncFile.PutVarArray("target_dates", startT, countT, &_targetDates(0));
    ncFile.PutVarArray("target_values_norm", startST, countST, &_targetValuesNorm[0](0));
    ncFile.PutVarArray("target_values_raw", startST, countST, &_targetValuesRaw[0](0));
    ncFile.PutVarArray("analog_criteria", startTA, countTA, &_analogsCriteria(0));
    ncFile.PutVarArray("analog_values_norm", startSTA, countSTA, &_analogsValuesNorm[0](0));
    ncFile.PutVarArray("analog_values_raw", startSTA, countSTA, &_analogsValuesRaw[0](0));

    // Close:save new netCDF dataset
    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();

    return true;
}

bool asResultsValues::Load() {
    BuildFileName();

    if (!Exists()) return false;

    ThreadsManager().CritSectionNetCDF().Enter();

    // Open the NetCDF file
    asFileNetcdf ncFile(_filePath, asFileNetcdf::ReadOnly);
    if (!ncFile.Open()) {
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Get the elements size
    size_t nStations = ncFile.GetDimLength("stations");
    size_t nTime = ncFile.GetDimLength("time");
    size_t nAnalogs = ncFile.GetDimLength("analogs");

    // Get time
    _targetDates.resize(nTime);
    ncFile.GetVar("target_dates", &_targetDates[0]);

    // Sizes
    size_t startTA[] = {0, 0};
    size_t countTA[] = {nTime, nAnalogs};
    size_t startST[] = {0, 0};
    size_t countST[] = {nStations, nTime};
    size_t startSTA[] = {0, 0, 0};
    size_t countSTA[] = {nStations, nTime, nAnalogs};

    // Resize containers
    _predictandStationIds.resize(nStations);
    _targetValuesNorm.resize(nStations, a1f(nTime));
    _targetValuesRaw.resize(nStations, a1f(nTime));
    _analogsCriteria.resize(nTime, nAnalogs);
    _analogsValuesNorm.resize(nStations, a2f(nTime, nAnalogs));
    _analogsValuesRaw.resize(nStations, a2f(nTime, nAnalogs));

    // Get data
    ncFile.GetVar("stations", &_predictandStationIds[0]);
    ncFile.GetVarArray("target_values_norm", startST, countST, &_targetValuesNorm[0](0));
    ncFile.GetVarArray("target_values_raw", startST, countST, &_targetValuesRaw[0](0));
    ncFile.GetVarArray("analog_criteria", startTA, countTA, &_analogsCriteria(0));
    ncFile.GetVarArray("analog_values_norm", startSTA, countSTA, &_analogsValuesNorm[0](0));
    ncFile.GetVarArray("analog_values_raw", startSTA, countSTA, &_analogsValuesRaw[0](0));

    ThreadsManager().CritSectionNetCDF().Leave();

    ncFile.Close();

    return true;
}
