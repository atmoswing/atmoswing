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

#include "asPredictorCustomUnilOisst2.h"

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorCustomUnilOisst2::asPredictorCustomUnilOisst2(const wxString& dataId)
    : asPredictor(dataId) {
    // Set the basic properties.
    _datasetId = "Custom_Unil_OISST_v2";
    _provider = "NOAA";
    _transformedBy = "Pascal Horton";
    _datasetName = "Optimum Interpolation Sea Surface Temperature, version 2, subset";
    _fileType = asFile::Netcdf;
    _strideAllowed = true;
    _nanValues.push_back(32767);
    _nanValues.push_back(936 * std::pow(10.f, 34.f));
    _fStr.dimLatName = "lat";
    _fStr.dimLonName = "lon";
    _fStr.dimTimeName = "time";
    _fStr.hasLevelDim = false;
}

bool asPredictorCustomUnilOisst2::Init() {
    // Identify data ID and set the corresponding properties.
    if (_dataId.IsSameAs("sst", false)) {
        _parameter = SeaSurfaceTemperature;
        _parameterName = "Sea Surface Temperature";
        _fileNamePattern = "sst_1deg.nc";
        _fileVarName = "sst";
        _unit = degC;
    } else if (_dataId.IsSameAs("sst_anom", false)) {
        _parameter = SeaSurfaceTemperatureAnomaly;
        _parameterName = "Sea Surface Temperature Anomaly";
        _fileNamePattern = "sst_anom_1deg.nc";
        _fileVarName = "anom";
        _unit = degC;
    } else {
        wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
        return false;
    }

    // Check data ID
    if (_fileNamePattern.IsEmpty() || _fileVarName.IsEmpty()) {
        wxLogError(_("The provided data ID (%s) does not match any possible option in dataset %s."), _dataId,
                   _datasetName);
        return false;
    }

    // Check directory is set
    if (GetDirectoryPath().IsEmpty()) {
        wxLogError(_("The path to the directory has not been set for the data %s from dataset %s."), _dataId,
                   _datasetName);
        return false;
    }

    // Set to initialized
    _initialized = true;

    return true;
}

void asPredictorCustomUnilOisst2::ListFiles(asTimeArray& timeArray) {
    _files.push_back(GetFullDirectoryPath() + _fileNamePattern);
}

void asPredictorCustomUnilOisst2::ConvertToMjd(a1d& time, double refValue) const {
    time /= 24.0;
    if (time[0] < 500 * 365) {               // New format
        time += asTime::GetMJD(1800, 1, 1);  // to MJD: add a negative time span
    } else {                                 // Old format
        time += asTime::GetMJD(1, 1, 1);     // to MJD: add a negative time span
    }
}
