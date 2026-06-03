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
 * Portions Copyright 2023 Pascal Horton, Terranum.
 */

#include "asPredictorOperMfArpege.h"
#include "asIncludes.h"

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorOperMfArpege::asPredictorOperMfArpege(const wxString& dataId)
    : asPredictorOper(dataId) {
    // Set the basic properties.
    _datasetId = "MF_ARPEGE";
    _provider = "METEOFRANCE";
    _datasetName = "ARPEGE grib files";
    _fileType = asFile::Grib;
    _isEnsemble = false;
    _strideAllowed = false;
    _fStr.dimLatName = "lat";
    _fStr.dimLonName = "lon";
    _fStr.dimTimeName = "time";
    _fStr.dimLevelName = "isobaric";
    _fStr.hasLevelDim = false;
    _fStr.singleTimeStep = true;
    _parameter = ParameterUndefined;
    _fileExtension = "grb";
    _leadTimeStart = 0;
    _leadTimeStep = 6;
    _runHourStart = 0;
    _runUpdate = 12;
}

bool asPredictorOperMfArpege::Init() {
    // Identify data ID and set the corresponding properties.
    if (IsGeopotential()) {
        _parameter = Geopotential;
        _gribCode = {0, 3, 4, 100};
        _unit = m2_s2;
        _fStr.hasLevelDim = true;
        _fileNamePattern = "ARP_GEOPOTENTIAL__ISOBARIC_SURFACE_%d_%s_%s.grb";
    } else if (IsRelativeHumidity()) {
        _parameter = RelativeHumidity;
        _gribCode = {0, 1, 1, 100};
        _unit = percent;
        _fStr.hasLevelDim = true;
        _fileNamePattern = "ARP_RELATIVE_HUMIDITY__ISOBARIC_SURFACE_%d_%s_%s.grb";
    } else if (IsTotalColumnWaterVapour()) {
        _parameter = PrecipitableWater;
        _gribCode = {0, 1, 64, 1};
        _unit = kg_m2;
        _fStr.hasLevelDim = false;
        _fileNamePattern = "ARP_TOTAL_COLUMN_INTEGRATED_WATER_VAPOUR__GROUND_OR_WATER_SURFACE_%d_%s_%s.grb";
    } else if (IsAirTemperature()) {
        _parameter = AirTemperature;
        _gribCode = {0, 0, 0, 100};
        _unit = degK;
        _fStr.hasLevelDim = true;
        _fileNamePattern = "ARP_TEMPERATURE__ISOBARIC_SURFACE_%d_%s_%s.grb";
    } else if (IsVerticalVelocity()) {
        _parameter = VerticalVelocity;
        _gribCode = {0, 2, 8, 100};
        _unit = Pa_s;
        _fStr.hasLevelDim = true;
        _fileNamePattern = "ARP_VERTICAL_VELOCITY_PRESSURE__ISOBARIC_SURFACE_%d_%s_%s.grb";
    } else {
        wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
        return false;
    }

    // Set to initialized
    _initialized = true;

    return true;
}

void asPredictorOperMfArpege::ConvertToMjd(a1d& time, double refValue) const {
    time = (time / 24.0) + refValue;
}

wxString asPredictorOperMfArpege::GetFileName(const double date, const int leadTime) {
    double mjdTarget = date + double(leadTime) / 24.0;
    wxString dateTarget = asTime::GetStringTime(mjdTarget, "YYYYMMDDhhmm");
    wxString dateForecast = asTime::GetStringTime(date, "YYYYMMDDhhmm");

    return asStrF(_fileNamePattern, (int)_level, dateTarget, dateForecast);
}
