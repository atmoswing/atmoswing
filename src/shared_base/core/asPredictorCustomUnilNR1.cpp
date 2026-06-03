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

#include "asPredictorCustomUnilNR1.h"
#include "asIncludes.h"

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorCustomUnilNR1::asPredictorCustomUnilNR1(const wxString& dataId)
    : asPredictor(dataId) {
    // Set the basic properties.
    _datasetId = "Custom_Unil_NR1";
    _provider = "NCEP/NCAR";
    _transformedBy = "Pascal Horton";
    _datasetName = "Reanalysis 1 subset";
    _fileType = asFile::Netcdf;
    _strideAllowed = true;
    _nanValues.push_back(32767);
    _nanValues.push_back(936 * std::pow(10.f, 34.f));
    _fStr.dimLatName = "lat";
    _fStr.dimLonName = "lon";
    _fStr.dimTimeName = "time";
    _fStr.dimLevelName = "level";
}

bool asPredictorCustomUnilNR1::Init() {
    // Identify data ID and set the corresponding properties.
    if (IsGeopotentialHeight()) {
        _fStr.hasLevelDim = true;
        _parameter = GeopotentialHeight;
        _parameterName = "Geopotential height";
        _fileNamePattern = "hgt.nc";
        _fileVarName = "hgt";
        _unit = m;
    } else if (IsAirTemperature()) {
        _fStr.hasLevelDim = true;
        _parameter = AirTemperature;
        _parameterName = "Air Temperature";
        _fileNamePattern = "air.nc";
        _fileVarName = "air";
        _unit = degK;
    } else if (IsVerticalVelocity()) {
        _fStr.hasLevelDim = true;
        _parameter = VerticalVelocity;
        _parameterName = "Vertical velocity";
        _fileNamePattern = "omega.nc";
        _fileVarName = "omega";
        _unit = Pa_s;
    } else if (IsRelativeHumidity()) {
        _fStr.hasLevelDim = true;
        _parameter = RelativeHumidity;
        _parameterName = "Relative Humidity";
        _fileNamePattern = "rhum.nc";
        _fileVarName = "rhum";
        _unit = percent;
    } else if (IsSpecificHumidity()) {
        _fStr.hasLevelDim = true;
        _parameter = SpecificHumidity;
        _parameterName = "Specific Humidity";
        _fileNamePattern = "shum.nc";
        _fileVarName = "shum";
        _unit = kg_kg;
    } else if (IsUwindComponent()) {
        _fStr.hasLevelDim = true;
        _parameter = Uwind;
        _parameterName = "U-Wind";
        _fileNamePattern = "uwnd.nc";
        _fileVarName = "uwnd";
        _unit = _s;
    } else if (IsVwindComponent()) {
        _fStr.hasLevelDim = true;
        _parameter = Vwind;
        _parameterName = "V-Wind";
        _fileNamePattern = "vwnd.nc";
        _fileVarName = "vwnd";
        _unit = _s;
    } else if (IsPrecipitableWater()) {
        _fStr.hasLevelDim = false;
        _parameter = PrecipitableWater;
        _parameterName = "Precipitable water";
        _fileNamePattern = "pr_wtr.nc";
        _fileVarName = "pr_wtr";
        _unit = mm;
    } else {
        wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
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

void asPredictorCustomUnilNR1::ListFiles(asTimeArray& timeArray) {
    _files.push_back(GetFullDirectoryPath() + _fileNamePattern);
}

void asPredictorCustomUnilNR1::ConvertToMjd(a1d& time, double refValue) const {
    time /= 24.0;
    if (time[0] < 500 * 365) {               // New format
        time += asTime::GetMJD(1800, 1, 1);  // to MJD: add a negative time span
    } else {                                 // Old format
        time += asTime::GetMJD(1, 1, 1);     // to MJD: add a negative time span
    }
}
