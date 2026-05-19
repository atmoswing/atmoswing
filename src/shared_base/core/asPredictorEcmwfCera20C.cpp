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
 * Portions Copyright 2017 Pascal Horton, University of Bern.
 */

#include "asPredictorEcmwfCera20C.h"

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorEcmwfCera20C::asPredictorEcmwfCera20C(const wxString& dataId)
    : asPredictor(dataId) {
    // Set the basic properties.
    _datasetId = "ECMWF_CERA_20C";
    _provider = "ECMWF";
    _datasetName = "Coupled ERA 20th Century";
    _fileType = asFile::Netcdf;
    _isEnsemble = true;
    _strideAllowed = true;
    _fStr.dimLatName = "latitude";
    _fStr.dimLonName = "longitude";
    _fStr.dimTimeName = "time";
    _fStr.dimLevelName = "level";
    _fStr.dimMemberName = "number";
}

bool asPredictorEcmwfCera20C::Init() {
    CheckLevelTypeIsDefined();

    // List of variables: http://rda.ucar.edu/datasets/ds627.0/docs/era_interim_grib_table.html

    // Identify data ID and set the corresponding properties.
    if (IsPressureLevel()) {
        _fStr.hasLevelDim = true;
        if (IsGeopotential()) {
            _parameter = Geopotential;
            _parameterName = "Geopotential";
            _fileVarName = "z";
            _unit = m2_s2;
        } else if (IsAirTemperature()) {
            _parameter = AirTemperature;
            _parameterName = "Temperature";
            _fileVarName = "t";
            _unit = degK;
        } else if (IsRelativeHumidity()) {
            _parameter = RelativeHumidity;
            _parameterName = "Relative humidity";
            _fileVarName = "r";
            _unit = percent;
        } else if (IsVerticalVelocity()) {
            _parameter = VerticalVelocity;
            _parameterName = "Vertical velocity";
            _fileVarName = "w";
            _unit = Pa_s;
        } else {
            _parameter = ParameterUndefined;
            _parameterName = "Undefined";
            _fileVarName = _dataId;
            _unit = UnitUndefined;
        }
        _fileNamePattern = _fileVarName + ".%d.nc";

    } else if (IsSurfaceLevel()) {
        _fStr.hasLevelDim = false;
        if (IsTotalColumnWater()) {
            _parameter = TotalColumnWater;
            _parameterName = "Total column water";
            _fileVarName = "tcw";
            _unit = kg_m2;
        } else if (IsTotalPrecipitation()) {
            _parameter = Precipitation;
            _parameterName = "Total precipitation";
            _fileVarName = "tp";
            _unit = m;
        } else if (IsSeaLevelPressure()) {
            _parameter = Pressure;
            _parameterName = "Sea level pressure";
            _fileVarName = "msl";
            _unit = Pa;
        } else {
            _parameter = ParameterUndefined;
            _parameterName = "Undefined";
            _fileVarName = _dataId;
            _unit = UnitUndefined;
        }
        _fileNamePattern = _fileVarName + ".%d.nc";

    } else {
        wxLogError(_("level type not implemented for this reanalysis dataset."));
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

void asPredictorEcmwfCera20C::ListFiles(asTimeArray& timeArray) {
    for (int iYear = timeArray.GetStartingYear(); iYear <= timeArray.GetEndingYear(); iYear++) {
        _files.push_back(GetFullDirectoryPath() + asStrF(_fileNamePattern, iYear));
    }
}

void asPredictorEcmwfCera20C::ConvertToMjd(a1d& time, double refValue) const {
    time = (time / 24.0) + asTime::GetMJD(1900, 1, 1);
}
