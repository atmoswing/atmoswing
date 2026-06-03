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
 * Portions Copyright 2014-2015 Pascal Horton, Terranum.
 */

#include "asPredictorCustomLtheNR1.h"
#include "asIncludes.h"

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorCustomLtheNR1::asPredictorCustomLtheNR1(const wxString& dataId)
    : asPredictorCustomUnilNR1(dataId) {
    // Set the basic properties.
    _datasetId = "Custom_LTHE_NR1";
    _provider = "NCEP/NCAR";
    _transformedBy = "LTHE";
    _datasetName = "Reanalysis 1 subset from LTHE";
    _fileType = asFile::Netcdf;
    _strideAllowed = true;
    _nanValues.push_back(32767);
    _nanValues.push_back(936 * std::pow(10.f, 34.f));
    _fStr.dimLatName = "lat";
    _fStr.dimLonName = "lon";
    _fStr.dimTimeName = "time";
    _fStr.dimLevelName = "level";
    _fStr.hasLevelDim = true;
}

bool asPredictorCustomLtheNR1::Init() {
    // Identify data ID and set the corresponding properties.
    if (_dataId.IsSameAs("hgt_500hPa", false)) {
        _parameter = GeopotentialHeight;
        _parameterName = "Geopotential height";
        _fileNamePattern = "NCEP_R1_lthe_hgt_500hPa.nc";
        _fileVarName = "hgt";
        _unit = m;
    } else if (_dataId.IsSameAs("hgt_1000hPa", false)) {
        _parameter = GeopotentialHeight;
        _parameterName = "Geopotential height";
        _fileNamePattern = "NCEP_R1_lthe_hgt_1000hPa.nc";
        _fileVarName = "hgt";
        _unit = m;
    } else if (IsPrecipitableWater()) {
        _parameter = PrecipitableWater;
        _parameterName = "Precipitable water";
        _fileNamePattern = "NCEP_R1_lthe_prwtr.nc";
        _fileVarName = "pwa";
        _unit = mm;
    } else if (IsRelativeHumidity()) {
        _parameter = RelativeHumidity;
        _parameterName = "Relative Humidity";
        _fileNamePattern = "NCEP_R1_lthe_rhum.nc";
        _fileVarName = "rhum";
        _unit = percent;
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

void asPredictorCustomLtheNR1::ListFiles(asTimeArray& timeArray) {
    _files.push_back(GetFullDirectoryPath() + _fileNamePattern);
}

void asPredictorCustomLtheNR1::ConvertToMjd(a1d& time, double refValue) const {
    time /= 24.0;
    if (time[0] < 500 * 365) {               // New format
        time += asTime::GetMJD(1800, 1, 1);  // to MJD: add a negative time span
    } else {                                 // Old format
        time += asTime::GetMJD(1, 1, 1);     // to MJD: add a negative time span
    }
}
