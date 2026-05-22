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

#include "asPredictorNoaa20Cr2c.h"
#include "asIncludes.h"

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorNoaa20Cr2c::asPredictorNoaa20Cr2c(const wxString& dataId)
    : asPredictor(dataId) {
    // Set the basic properties.
    _datasetId = "NOAA_20CR_v2c";
    _provider = "NOAA";
    _datasetName = "Twentieth Century Reanalysis (v2c)";
    _fileType = asFile::Netcdf;
    _strideAllowed = true;
    _nanValues.push_back(-9.96921 * std::pow(10.f, 36.f));
    _fStr.dimLatName = "lat";
    _fStr.dimLonName = "lon";
    _fStr.dimTimeName = "time";
    _fStr.dimLevelName = "level";
}

bool asPredictorNoaa20Cr2c::Init() {
    CheckLevelTypeIsDefined();

    // Identify data ID and set the corresponding properties.
    if (IsPressureLevel()) {
        _fStr.hasLevelDim = true;
        if (IsAirTemperature()) {
            _parameter = AirTemperature;
            _parameterName = "Air Temperature";
            _fileVarName = "air";
            _unit = degK;
        } else if (IsGeopotentialHeight()) {
            _parameter = GeopotentialHeight;
            _parameterName = "Geopotential height";
            _fileVarName = "hgt";
            _unit = m;
        } else if (IsVerticalVelocity()) {
            _parameter = VerticalVelocity;
            _parameterName = "Vertical velocity";
            _fileVarName = "omega";
            _unit = Pa_s;
        } else if (IsRelativeHumidity()) {
            _parameter = RelativeHumidity;
            _parameterName = "Relative Humidity";
            _fileVarName = "rhum";
            _unit = percent;
        } else if (IsSpecificHumidity()) {
            _parameter = SpecificHumidity;
            _parameterName = "Specific Humidity";
            _fileVarName = "shum";
            _unit = kg_kg;
        } else if (IsUwindComponent()) {
            _parameter = Uwind;
            _parameterName = "U-Wind";
            _fileVarName = "uwnd";
            _unit = _s;
        } else if (IsVwindComponent()) {
            _parameter = Vwind;
            _parameterName = "V-Wind";
            _fileVarName = "vwnd";
            _unit = _s;
        } else {
            _parameter = ParameterUndefined;
            _parameterName = "Undefined";
            _fileVarName = _dataId;
            _unit = UnitUndefined;
        }
        _fileNamePattern = _fileVarName + ".%d.nc";

    } else if (IsSurfaceLevel() || _product.IsSameAs("monolevel", false)) {
        _fStr.hasLevelDim = false;
        if (IsPrecipitableWater()) {
            _parameter = PrecipitableWater;
            _parameterName = "Precipitable water";
            _fileNamePattern = "pr_wtr.eatm.%d.nc";
            _fileVarName = "pr_wtr";
            _unit = mm;
        } else if (IsSeaLevelPressure()) {
            _parameter = Pressure;
            _parameterName = "Sea level pressure";
            _fileNamePattern = "prmsl.%d.nc";
            _fileVarName = "prmsl";
            _unit = Pa;
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
            return false;
        }

    } else if (IsSurfaceFluxesLevel() || _product.IsSameAs("surface_gauss", false) ||
               _product.IsSameAs("gauss", false) || _product.IsSameAs("gaussian", false)) {
        _fStr.hasLevelDim = false;
        if (IsPrecipitationRate()) {
            _parameter = PrecipitationRate;
            _parameterName = "Precipitation rate";
            _fileNamePattern = "prate.%d.nc";
            _fileVarName = "prate";
            _unit = kg_m2_s;
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
            return false;
        }

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

void asPredictorNoaa20Cr2c::ListFiles(asTimeArray& timeArray) {
    for (int iYear = timeArray.GetStartingYear(); iYear <= timeArray.GetEndingYear(); iYear++) {
        _files.push_back(GetFullDirectoryPath() + asStrF(_fileNamePattern, iYear));
    }
}

void asPredictorNoaa20Cr2c::ConvertToMjd(a1d& time, double refValue) const {
    time = (time / 24.0) + asTime::GetMJD(1800, 1, 1);
}
