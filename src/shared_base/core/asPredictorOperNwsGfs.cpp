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

#include "asPredictorOperNwsGfs.h"
#include "asIncludes.h"

#include <wx/fileconf.h>

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorOperNwsGfs::asPredictorOperNwsGfs(const wxString& dataId)
    : asPredictorOper(dataId) {
    // Set the basic properties.
    _datasetId = "NWS_GFS";
    _provider = "NWS";
    _transformedBy = wxEmptyString;
    _datasetName = "Global Forecast System";
    _fileType = asFile::Grib;
    _leadTimeStart = 0;
    _leadTimeStep = 6;
    _runHourStart = 0;
    _runUpdate = 6;
    _strideAllowed = false;
    _shouldDownload = true;
    _fileExtension = "grib2";
    _fStr.hasLevelDim = false;
    _fStr.singleTimeStep = true;
    _parameter = ParameterUndefined;
}

asPredictorOperNwsGfs::~asPredictorOperNwsGfs() {}

bool asPredictorOperNwsGfs::Init() {
    wxConfigBase* pConfig = wxFileConfig::Get();

    // Last element in grib code: level type (http://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_table4-5.shtml)

    // Identify data ID and set the corresponding properties.
    if (IsGeopotentialHeight()) {
        _parameter = GeopotentialHeight;
        _gribCode = {0, 3, 5, 100};
        _commandDownload = pConfig->Read("/PredictorsUrl/GFS/hgt", _commandDownload);
        _unit = m;
        _fStr.hasLevelDim = true;
    } else if (IsAirTemperature()) {
        _parameter = AirTemperature;
        _gribCode = {0, 0, 0, 100};
        _commandDownload = pConfig->Read("/PredictorsUrl/GFS/temp", _commandDownload);
        _unit = degK;
        _fStr.hasLevelDim = true;
    } else if (IsVerticalVelocity()) {
        _parameter = VerticalVelocity;
        _gribCode = {0, 2, 8, 100};
        _commandDownload = pConfig->Read("/PredictorsUrl/GFS/vvel", _commandDownload);
        _unit = Pa_s;
        _fStr.hasLevelDim = true;
    } else if (IsRelativeHumidity()) {
        _parameter = RelativeHumidity;
        _gribCode = {0, 1, 1, 100};
        _commandDownload = pConfig->Read("/PredictorsUrl/GFS/rh", _commandDownload);
        _unit = percent;
        _fStr.hasLevelDim = true;
    } else if (IsUwindComponent()) {
        _parameter = Uwind;
        _gribCode = {0, 2, 2, 100};
        _commandDownload = pConfig->Read("/PredictorsUrl/GFS/uwnd", _commandDownload);
        _unit = _s;
        _fStr.hasLevelDim = true;
    } else if (IsVwindComponent()) {
        _parameter = Vwind;
        _gribCode = {0, 2, 3, 100};
        _commandDownload = pConfig->Read("/PredictorsUrl/GFS/vwnd", _commandDownload);
        _unit = _s;
        _fStr.hasLevelDim = true;
    } else if (IsPrecipitableWater()) {
        _parameter = PrecipitableWater;
        _gribCode = {0, 1, 3, 200};
        _commandDownload = pConfig->Read("/PredictorsUrl/GFS/pwat", _commandDownload);
        _unit = mm;
    } else {
        wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
        return false;
    }

    // Set to initialized
    _initialized = true;

    return true;
}
