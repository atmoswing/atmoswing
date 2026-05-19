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

#include "asPredictorNcepCfsr.h"

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorNcepCfsr::asPredictorNcepCfsr(const wxString& dataId)
    : asPredictor(dataId) {
    // Set the basic properties.
    _datasetId = "NCEP_CFSR";
    _provider = "NCEP";
    _datasetName = "CFSR";
    _fileType = asFile::Grib;
    _strideAllowed = false;
}

bool asPredictorNcepCfsr::Init() {
    CheckLevelTypeIsDefined();

    // Last element in grib code: level type (http://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_table4-5.shtml)

    // Identify data ID and set the corresponding properties.
    if (IsPressureLevel()) {
        _fStr.hasLevelDim = true;
        _fStr.singleLevel = true;
        if (IsGeopotentialHeight()) {
            _parameter = GeopotentialHeight;
            _gribCode = {0, 3, 5, 100};
            _parameterName = "Geopotential height @ Isobaric surface";
            _unit = gpm;
        } else if (IsPrecipitableWater()) {
            _parameter = PrecipitableWater;
            _gribCode = {0, 1, 3, 200};
            _parameterName = "Precipitable water @ Entire atmosphere layer";
            _unit = kg_m2;
        } else if (IsSeaLevelPressure()) {
            _parameter = Pressure;
            _gribCode = {0, 3, 0, 101};
            _parameterName = "Pressure @ Mean sea level";
            _unit = Pa;
        } else if (IsRelativeHumidity()) {
            _parameter = RelativeHumidity;
            _gribCode = {0, 1, 1, 100};
            _parameterName = "Relative humidity @ Isobaric surface";
            _unit = percent;
        } else if (IsAirTemperature()) {
            _parameter = AirTemperature;
            _gribCode = {0, 0, 0, 100};
            _parameterName = "Temperature @ Isobaric surface";
            _unit = degK;
        } else {
            wxLogError(_("Parameter '%s' not implemented yet."), _dataId);
            return false;
        }
        _fileNamePattern = "%4d/%4d%02d/%4d%02d%02d/pgbhnl.gdas.%4d%02d%02d%02d.grb2";

    } else if (IsIsentropicLevel()) {
        wxLogError(_("Isentropic levels for CFSR are not implemented yet."));
        return false;

    } else if (IsSurfaceFluxesLevel()) {
        wxLogError(_("Surface fluxes grids for CFSR are not implemented yet."));
        return false;

    } else {
        wxLogError(_("level type not implemented for this reanalysis dataset."));
        return false;
    }

    // Check data ID
    if (_fileNamePattern.IsEmpty() || _gribCode[2] == asNOT_FOUND) {
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

    wxASSERT(_gribCode.size() == 4);

    // Set to initialized
    _initialized = true;

    return true;
}

void asPredictorNcepCfsr::ListFiles(asTimeArray& timeArray) {
    a1d tArray = timeArray.GetTimeArray();

    for (int i = 0; i < tArray.size(); i++) {
        Time t = asTime::GetTimeStruct(tArray[i]);
        _files.push_back(GetFullDirectoryPath() + asStrF(_fileNamePattern, t.year, t.year, t.month, t.year, t.month,
                                                         t.day, t.year, t.month, t.day, t.hour));
    }
}
