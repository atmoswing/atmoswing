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
 * Portions Copyright 2017-2019 Pascal Horton, University of Bern.
 */

#include "asPredictorEcmwfEra5.h"
#include "asIncludes.h"

#include <wx/dir.h>
#include <wx/regex.h>

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorEcmwfEra5::asPredictorEcmwfEra5(const wxString& dataId)
    : asPredictor(dataId) {
    // Set the basic properties.
    _datasetId = "ECMWF_ERA5";
    _provider = "ECMWF";
    _datasetName = "ERA5";
    _fileType = asFile::Netcdf;
    _strideAllowed = true;
    _nanValues.push_back(-32767);
    _fStr.dimLatName = "latitude";
    _fStr.dimLonName = "longitude";
    _fStr.dimTimeName = "time";
    _fStr.dimLevelName = "level";
}

bool asPredictorEcmwfEra5::Init() {
    CheckLevelTypeIsDefined();

    // Identify data ID and set the corresponding properties.
    if (IsPressureLevel()) {
        _fStr.hasLevelDim = true;
        if (_dataId.IsSameAs("d", false)) {
            _parameter = Divergence;
            _parameterName = "Divergence";
            _fileVarName = "d";
            _unit = per_s;
        } else if (IsPotentialVorticity()) {
            _parameter = PotentialVorticity;
            _parameterName = "Potential vorticity";
            _fileVarName = "pv";
            _unit = degKm2_kg_s;
        } else if (IsSpecificHumidity()) {
            _parameter = SpecificHumidity;
            _parameterName = "Specific humidity";
            _fileVarName = "q";
            _unit = kg_kg;
        } else if (IsRelativeHumidity()) {
            _parameter = RelativeHumidity;
            _parameterName = "Relative humidity";
            _fileVarName = "r";
            _unit = percent;
        } else if (IsAirTemperature()) {
            _parameter = AirTemperature;
            _parameterName = "Temperature";
            _fileVarName = "t";
            _unit = degK;
        } else if (IsUwindComponent()) {
            _parameter = Uwind;
            _parameterName = "U component of wind";
            _fileVarName = "u";
            _unit = _s;
        } else if (IsVwindComponent()) {
            _parameter = Vwind;
            _parameterName = "V component of wind";
            _fileVarName = "v";
            _unit = _s;
        } else if (_dataId.IsSameAs("vo", false)) {
            _parameter = Vorticity;
            _parameterName = "Vorticity (relative)";
            _fileVarName = "vo";
            _unit = per_s;
        } else if (IsVerticalVelocity()) {
            _parameter = VerticalVelocity;
            _parameterName = "Vertical velocity";
            _fileVarName = "w";
            _unit = Pa_s;
        } else if (IsGeopotential()) {
            _parameter = Geopotential;
            _parameterName = "Geopotential";
            _fileVarName = "z";
            _unit = m2_s2;
        } else {
            _parameter = ParameterUndefined;
            _parameterName = "Undefined";
            _fileVarName = _dataId;
            _unit = UnitUndefined;
        }

    } else if (IsSurfaceLevel() || _product.IsSameAs("single", false)) {
        _fStr.hasLevelDim = false;
        // Surface analysis
        if (_dataId.IsSameAs("d2m", false)) {
            _parameter = DewpointTemperature;
            _parameterName = "2 metre dewpoint temperature";
            _fileVarName = "d2m";
            _unit = degK;
        } else if (IsSeaLevelPressure()) {
            _parameter = Pressure;
            _parameterName = "Sea level pressure";
            _fileVarName = "msl";
            _unit = Pa;
        } else if (_dataId.IsSameAs("sd", false)) {
            _parameter = SnowWaterEquivalent;
            _parameterName = "Snow depth";
            _fileVarName = "sd";
            _unit = m;
        } else if (_dataId.IsSameAs("sst", false)) {
            _parameter = SeaSurfaceTemperature;
            _parameterName = "Sea surface temperature";
            _fileVarName = "sst";
            _unit = degK;
        } else if (_dataId.IsSameAs("t2m", false)) {
            _parameter = AirTemperature;
            _parameterName = "2 metre temperature";
            _fileVarName = "t2m";
            _unit = degK;
        } else if (_dataId.IsSameAs("tcw", false)) {
            _parameter = TotalColumnWater;
            _parameterName = "Total column water";
            _fileVarName = "tcw";
            _unit = kg_m2;
        } else if (_dataId.IsSameAs("tcwv", false)) {
            _parameter = PrecipitableWater;
            _parameterName = "Total column water vapour";
            _fileVarName = "tcwv";
            _unit = kg_m2;
        } else if (_dataId.IsSameAs("u10", false)) {
            _parameter = Uwind;
            _parameterName = "10 metre U wind component";
            _fileVarName = "u10";
            _unit = _s;
        } else if (_dataId.IsSameAs("v10", false)) {
            _parameter = Vwind;
            _parameterName = "10 metre V wind component";
            _fileVarName = "v10";
            _unit = _s;
        } else if (IsTotalPrecipitation()) {
            _parameter = Precipitation;
            _parameterName = "Total precipitation";
            _fileVarName = "tp";
            _unit = m;
        } else if (_dataId.IsSameAs("cape", false)) {
            _parameter = CAPE;
            _parameterName = "Convective available potential energy";
            _fileVarName = "cape";
            _unit = J_kg;
        } else if (_dataId.IsSameAs("ie", false)) {
            _parameter = MoistureFlux;
            _parameterName = "Instantaneous moisture flux";
            _fileVarName = "ie";
            _unit = kg_m2_s;
        } else if (_dataId.IsSameAs("ssr", false)) {
            _parameter = Radiation;
            _parameterName = "Surface net solar radiation";
            _fileVarName = "ssr";
            _unit = J_m2;
        } else if (_dataId.IsSameAs("ssrd", false)) {
            _parameter = Radiation;
            _parameterName = "Surface solar radiation downwards";
            _fileVarName = "ssrd";
            _unit = J_m2;
        } else if (_dataId.IsSameAs("str", false)) {
            _parameter = Radiation;
            _parameterName = "Surface net thermal radiation";
            _fileVarName = "str";
            _unit = J_m2;
        } else if (_dataId.IsSameAs("strd", false)) {
            _parameter = Radiation;
            _parameterName = "Surface thermal radiation downwards";
            _fileVarName = "strd";
            _unit = J_m2;
        } else {
            _parameter = ParameterUndefined;
            _parameterName = "Undefined";
            _fileVarName = _dataId;
            _unit = UnitUndefined;
        }

    } else {
        wxLogError(_("level type not implemented for this reanalysis dataset."));
        return false;
    }

    // Check data ID
    if (_fileVarName.IsEmpty()) {
        wxLogError(_("The provided data ID (%s) does not match any possible option in the dataset %s."), _dataId,
                   _datasetName);
        return false;
    }

    // Check directory is set
    if (GetDirectoryPath().IsEmpty()) {
        wxLogError(_("The path to the directory has not been set for the data %s from the dataset %s."), _dataId,
                   _datasetName);
        return false;
    }

    // Set to initialized
    _initialized = true;

    return true;
}

void asPredictorEcmwfEra5::ListFiles(asTimeArray& timeArray) {
    // Case 1: single file with the variable name
    wxString filePath = GetFullDirectoryPath() + _fileVarName + ".nc";

    if (wxFileExists(filePath)) {
        _files.push_back(filePath);
        return;
    }

    // Case 2: yearly files
    wxArrayString listFiles;
    size_t nbFiles = wxDir::GetAllFiles(GetFullDirectoryPath(), &listFiles, "*.nc");

    if (nbFiles == 0) {
        throw runtime_error(_("No ERA5 file found."));
    }

    listFiles.Sort();

    double firstYear = timeArray.GetStartingYear();
    double lastYear = timeArray.GetEndingYear();

    for (size_t i = 0; i < listFiles.Count(); ++i) {
        if (!listFiles.Item(i).StartsWith(GetFullDirectoryPath() + _fileVarName + ".")) {
            continue;
        }

        wxRegEx reDates("\\d{4,}", wxRE_ADVANCED);
        if (!reDates.Matches(listFiles.Item(i))) {
            continue;
        }

        wxString datesSrt = reDates.GetMatch(listFiles.Item(i));
        double fileYear = 0;
        datesSrt.ToDouble(&fileYear);

        if (fileYear < firstYear || fileYear > lastYear) {
            continue;
        }

        _files.push_back(listFiles.Item(i));
    }

    if (!_files.empty()) {
        return;
    }

    // Case 3: list all files from the directory
    for (size_t i = 0; i < listFiles.Count(); ++i) {
        _files.push_back(listFiles.Item(i));
    }
}

void asPredictorEcmwfEra5::ConvertToMjd(a1d& time, double refValue) const {
    time = (time / 24.0) + asTime::GetMJD(1900, 1, 1);
}
