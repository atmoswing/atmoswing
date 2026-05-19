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
 * Portions Copyright 2019 Pascal Horton, University of Bern.
 */

#include "asPredictorEcmwfIfs.h"

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorEcmwfIfs::asPredictorEcmwfIfs(const wxString& dataId)
    : asPredictor(dataId) {
    // Set the basic properties.
    _datasetId = "ECMWF_IFS";
    _provider = "ECMWF";
    _datasetName = "Integrated Forecasting System (IFS) grib files";
    _fileType = asFile::Grib;
    _isEnsemble = false;
    _strideAllowed = false;
    _fStr.hasLevelDim = false;
    _fStr.singleTimeStep = true;
    _parameter = ParameterUndefined;
}

bool asPredictorEcmwfIfs::Init() {
    // Identify data ID and set the corresponding properties.
    if (_dataId.IsSameAs("z", false)) {
        _parameter = Geopotential;
        _gribCode = {0, 128, 129, 100};
        _unit = m2_s2;
        _fStr.hasLevelDim = true;
    } else if (_dataId.IsSameAs("gh", false)) {
        _parameter = GeopotentialHeight;
        _gribCode = {0, 128, 156, 100};
        _unit = m;
        _fStr.hasLevelDim = true;
    } else if (IsAirTemperature()) {
        _parameter = AirTemperature;
        _gribCode = {0, 128, 130, 100};
        _unit = degK;
        _fStr.hasLevelDim = true;
    } else if (IsVerticalVelocity()) {
        _parameter = VerticalVelocity;
        _gribCode = {0, 128, 135, 100};
        _unit = Pa_s;
        _fStr.hasLevelDim = true;
    } else if (IsRelativeHumidity()) {
        _parameter = RelativeHumidity;
        _gribCode = {0, 128, 157, 100};
        _unit = percent;
        _fStr.hasLevelDim = true;
    } else if (IsSpecificHumidity()) {
        _parameter = SpecificHumidity;
        _gribCode = {0, 128, 133, 100};
        _unit = percent;
        _fStr.hasLevelDim = true;
    } else if (IsUwindComponent()) {
        _parameter = Uwind;
        _gribCode = {0, 128, 131, 100};
        _unit = _s;
        _fStr.hasLevelDim = true;
    } else if (IsVwindComponent()) {
        _parameter = Vwind;
        _gribCode = {0, 128, 132, 100};
        _unit = _s;
        _fStr.hasLevelDim = true;
    } else if (_dataId.IsSameAs("thetaE", false)) {
        _parameter = PotentialTemperature;
        _gribCode = {0, 3, 113, 100};
        _unit = W_m2;
        _fStr.hasLevelDim = true;
    } else if (_dataId.IsSameAs("thetaES", false)) {
        _parameter = PotentialTemperature;
        _gribCode = {0, 3, 114, 100};
        _unit = W_m2;
        _fStr.hasLevelDim = true;
    } else if (IsTotalColumnWaterVapour()) {
        _parameter = PrecipitableWater;
        _gribCode = {0, 128, 137, 200};
        _unit = mm;
    } else if (IsPrecipitableWater()) {
        _parameter = PrecipitableWater;
        _gribCode = {0, 128, 136, 200};
        _unit = mm;
    } else {
        wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
        return false;
    }

    // Set to initialized
    _initialized = true;

    return true;
}

void asPredictorEcmwfIfs::ConvertToMjd(a1d& time, double refValue) const {
    time = (time / 24.0) + refValue;
}
