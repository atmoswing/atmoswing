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

#include "asResultsDates.h"

#include "asFileNetcdf.h"

asResultsDates::asResultsDates()
    : asResults() {}

asResultsDates::~asResultsDates() {}

void asResultsDates::Init(asParameters* params) {
    _predictandStationIds = params->GetPredictandStationIds();

    // Resize to 0 to avoid keeping old results
    _targetDates.resize(0);
    _analogsCriteria.resize(0, 0);
    _analogsDates.resize(0, 0);
}

void asResultsDates::BuildFileName() {
    ThreadsManager().CritSectionConfig().Enter();
    _filePath = wxFileConfig::Get()->Read("/Paths/ResultsDir", asConfig::GetDefaultUserWorkingDir());
    ThreadsManager().CritSectionConfig().Leave();
    if (!_subFolder.IsEmpty()) {
        _filePath.Append(DS);
        _filePath.Append(_subFolder);
    }
    _filePath.Append(DS);
    _filePath.Append(asStrF("AnalogDates_id_%s_step_%d", GetPredictandStationIdsList(), _currentStep));
    _filePath.Append(".nc");
}

bool asResultsDates::Save() {
    BuildFileName();

    // Get the elements size
    size_t nTime = (size_t)_analogsCriteria.rows();
    size_t nAnalogs = (size_t)_analogsCriteria.cols();

    ThreadsManager().CritSectionNetCDF().Enter();

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(_filePath, asFileNetcdf::Replace);
    if (!ncFile.Open()) {
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Define dimensions. Time is the unlimited dimension.
    ncFile.DefDim("time");
    ncFile.DefDim("analogs", nAnalogs);

    // The dimensions name array is used to pass the dimensions to the variable.
    vstds dimNames1;
    dimNames1.push_back("time");
    vstds dimNames2;
    dimNames2.push_back("time");
    dimNames2.push_back("analogs");

    // Define variables: the analogcriteria and the corresponding dates
    ncFile.DefVar("target_dates", NC_FLOAT, 1, dimNames1);
    ncFile.DefVar("analog_criteria", NC_FLOAT, 2, dimNames2);
    ncFile.DefVar("analog_dates", NC_FLOAT, 2, dimNames2);
    ncFile.DefVarDeflate("analog_dates");

    // Put attributes
    DefTargetDatesAttributes(ncFile);
    DefAnalogsCriteriaAttributes(ncFile);
    DefAnalogsDatesAttributes(ncFile);

    // End definitions: leave define mode
    ncFile.EndDef();

    // Provide sizes for variables
    size_t start1D[] = {0};
    size_t count1D[] = {nTime};
    size_t start2D[] = {0, 0};
    size_t count2D[] = {nTime, nAnalogs};

    // Write data
    ncFile.PutVarArray("target_dates", start1D, count1D, &_targetDates(0));
    ncFile.PutVarArray("analog_criteria", start2D, count2D, &_analogsCriteria(0));
    ncFile.PutVarArray("analog_dates", start2D, count2D, &_analogsDates(0));

    // Close:save new netCDF dataset
    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();
    return true;
}

bool asResultsDates::Load() {
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
    size_t nTime = ncFile.GetDimLength("time");
    size_t nAnalogs = ncFile.GetDimLength("analogs");

    // Get time
    _targetDates.resize(nTime);
    ncFile.GetVar("target_dates", &_targetDates[0]);

    // Check last value
    if (_targetDates[_targetDates.size() - 1] < _targetDates[0]) {
        wxLogError(_("The target date array is not consistent in the temp file (last value makes no sense)."));
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Sizes
    size_t startTA[] = {0, 0};
    size_t countTA[] = {nTime, nAnalogs};

    // Resize containers
    _analogsCriteria.resize(nTime, nAnalogs);
    _analogsDates.resize(nTime, nAnalogs);

    // Get data
    ncFile.GetVarArray("analog_criteria", startTA, countTA, &_analogsCriteria(0));
    ncFile.GetVarArray("analog_dates", startTA, countTA, &_analogsDates(0));

    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();
    return true;
}
