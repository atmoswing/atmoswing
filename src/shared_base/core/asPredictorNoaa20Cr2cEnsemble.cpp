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
 * Portions Copyright 2016 Pascal Horton, University of Bern.
 */

#include "asPredictorNoaa20Cr2cEnsemble.h"
#include "asIncludes.h"

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorNoaa20Cr2cEnsemble::asPredictorNoaa20Cr2cEnsemble(const wxString& dataId)
    : asPredictor(dataId) {
    // Set the basic properties.
    _datasetId = "NOAA_20CR_v2c_ens";
    _provider = "NOAA";
    _datasetName = "Twentieth Century Reanalysis (v2c) Ensemble";
    _fileType = asFile::Netcdf;
    _isEnsemble = true;
    _strideAllowed = true;
    _nanValues.push_back(-9.96921 * std::pow(10.f, 36.f));
    _fStr.dimLatName = "lat";
    _fStr.dimLonName = "lon";
    _fStr.dimTimeName = "time";
    _fStr.dimMemberName = "ensemble_member";
    _fStr.hasLevelDim = false;
}

bool asPredictorNoaa20Cr2cEnsemble::Init() {
    CheckLevelTypeIsDefined();

    // Identify data ID and set the corresponding properties.
    if (_product.IsSameAs("analysis", false)) {
        if (IsSeaLevelPressure()) {
            _parameter = Pressure;
            _parameterName = "Sea level pressure";
            _fileVarName = "prmsl";
            _unit = Pa;
        } else if (IsPrecipitableWater()) {
            _parameter = PrecipitableWater;
            _parameterName = "Precipitable water";
            _fileVarName = "pwat";
            _unit = kg_m2;
        } else if (_dataId.IsSameAs("omega500", false)) {
            _parameter = VerticalVelocity;
            _parameterName = "Vertical velocity at 500 hPa";
            _fileVarName = "omega500";
            _unit = Pa_s;
        } else if (_dataId.IsSameAs("rh850", false)) {
            _parameter = RelativeHumidity;
            _parameterName = "Relative Humidity at 850 hPa";
            _fileVarName = "rh850";
            _unit = percent;
        } else if (_dataId.IsSameAs("rh9950", false)) {
            _parameter = RelativeHumidity;
            _parameterName = "Relative Humidity at the pressure level 0.995 times the surface pressure";
            _fileVarName = "rh850";
            _unit = percent;
        } else if (_dataId.IsSameAs("t850", false)) {
            _parameter = AirTemperature;
            _parameterName = "Air Temperature at 850 hPa";
            _fileVarName = "t850";
            _unit = degK;
        } else if (_dataId.IsSameAs("t9950", false)) {
            _parameter = AirTemperature;
            _parameterName = "Air Temperature at the pressure level 0.995 times the surface pressure";
            _fileVarName = "t9950";
            _unit = degK;
        } else if (_dataId.IsSameAs("z200", false)) {
            _parameter = GeopotentialHeight;
            _parameterName = "Geopotential height at 200 hPa";
            _fileVarName = "z200";
            _unit = m;
        } else if (_dataId.IsSameAs("z500", false)) {
            _parameter = GeopotentialHeight;
            _parameterName = "Geopotential height at 500 hPa";
            _fileVarName = "z500";
            _unit = m;
        } else if (_dataId.IsSameAs("z1000", false)) {
            _parameter = GeopotentialHeight;
            _parameterName = "Geopotential height at 1000 hPa";
            _fileVarName = "z1000";
            _unit = m;
        } else {
            _parameter = ParameterUndefined;
            _parameterName = "Undefined";
            _fileVarName = _dataId;
            _unit = UnitUndefined;
        }
        _fileNamePattern = _fileVarName + "_%d.nc";

    } else if (_product.IsSameAs("first_guess", false)) {
        if (IsPrecipitationRate()) {
            _parameter = PrecipitationRate;
            _parameterName = "Precipitation rate";
            _fileVarName = "prate";
            _unit = kg_m2_s;
        } else {
            _parameter = ParameterUndefined;
            _parameterName = "Undefined";
            _fileVarName = _dataId;
            _unit = UnitUndefined;
        }
        _fileNamePattern = _fileVarName + "_%d.nc";

    } else {
        wxLogError(_("Product type not implemented for this reanalysis dataset."));
        return false;
    }

    // Check data ID
    if (_fileNamePattern.IsEmpty() || _fileVarName.IsEmpty()) {
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

void asPredictorNoaa20Cr2cEnsemble::ListFiles(asTimeArray& timeArray) {
    for (int iYear = timeArray.GetStartingYear(); iYear <= timeArray.GetEndingYear(); iYear++) {
        _files.push_back(GetFullDirectoryPath() + asStrF(_fileNamePattern, iYear));
    }
}

void asPredictorNoaa20Cr2cEnsemble::ConvertToMjd(a1d& time, double refValue) const {
    time = (time / 24.0) + asTime::GetMJD(1, 1, 1);
}
