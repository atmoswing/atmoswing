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
 * Portions Copyright 2019-2020 Pascal Horton, University of Bern.
 */

#include "asPredictorOperCustomFvg.h"
#include "asIncludes.h"

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorOperCustomFvg::asPredictorOperCustomFvg(const wxString& dataId)
    : asPredictorOperEcmwfIfs(dataId) {
    // Set the basic properties.
    _datasetId = "Custom_MeteoFVG";
    _datasetName = "Integrated Forecasting System (IFS) grib files at Meteo FVG";
    _fStr.hasLevelDim = true;
    _leadTimeStart = 6;
    _runHourStart = 0;
    _runUpdate = 24;
    _percentMissingAllowed = 70;
}

bool asPredictorOperCustomFvg::Init() {
    _parameter = Other;

    if (_dataId.IsSameAs("DP500925", false)) {
        _gribCode = {0, 3, 113, 100};
    } else if (_dataId.IsSameAs("LRT700500", false)) {
        _gribCode = {0, 128, 130, 100};
    } else if (_dataId.IsSameAs("LRT850500", false)) {
        _gribCode = {0, 128, 130, 100};
    } else if (_dataId.IsSameAs("LRTE700500", false)) {
        _gribCode = {0, 3, 113, 100};
    } else if (_dataId.IsSameAs("LRTE850500", false)) {
        _gribCode = {0, 3, 113, 100};
    } else if (_dataId.IsSameAs("MB500850", false)) {
        _parameter = MaximumBuoyancy;
        _gribCode = {0, 3, 114, 100};
    } else if (_dataId.IsSameAs("MB500925", false)) {
        _parameter = MaximumBuoyancy;
        _gribCode = {0, 3, 114, 100};
    } else if (_dataId.IsSameAs("MB700925", false)) {
        _parameter = MaximumBuoyancy;
        _gribCode = {0, 3, 114, 100};
    } else if (_dataId.IsSameAs("MB850500", false)) {
        _parameter = MaximumBuoyancy;
        _gribCode = {0, 3, 114, 100};
    } else if (_dataId.Contains("thetaES")) {
        _parameter = PotentialTemperature;
        _gribCode = {0, 3, 114, 100};
        _unit = W_m2;
    } else if (_dataId.Contains("thetaE")) {
        _parameter = PotentialTemperature;
        _gribCode = {0, 3, 113, 100};
        _unit = W_m2;
    } else if (_dataId.Contains("vflux")) {
        _parameter = MomentumFlux;
        _gribCode = {0, 3, 125, 100};
        _unit = kg_m2_s;
    } else if (_dataId.Contains("uflux")) {
        _parameter = MomentumFlux;
        _gribCode = {0, 3, 124, 100};
        _unit = kg_m2_s;
    } else if (_dataId.Contains("2t_sfc")) {
        _parameter = AirTemperature;
        _gribCode = {0, 128, 167, 1};
        _unit = degK;
    } else if (_dataId.Contains("10u_sfc")) {
        _parameter = Uwind;
        _gribCode = {0, 128, 165, 1};
        _unit = _s;
    } else if (_dataId.Contains("10v_sfc")) {
        _parameter = Vwind;
        _gribCode = {0, 128, 166, 1};
        _unit = _s;
    } else if (_dataId.Contains("cp_sfc")) {
        _parameter = Precipitation;
        _gribCode = {0, 128, 143, 1};
        _unit = m;
    } else if (_dataId.Contains("msl_sfc")) {
        _parameter = Pressure;
        _gribCode = {0, 128, 151, 1};
        _unit = Pa;
    } else if (_dataId.Contains("tp_sfc")) {
        _parameter = AirTemperature;
        _gribCode = {0, 128, 228, 1};
        _unit = degK;
    } else if (_dataId.Contains("q")) {
        _parameter = SpecificHumidity;
        _gribCode = {0, 128, 133, 100};
        _unit = percent;
    } else if (_dataId.Contains("gh")) {
        _parameter = GeopotentialHeight;
        _gribCode = {0, 128, 156, 100};
        _unit = m;
    } else if (_dataId.Contains("t")) {
        _parameter = AirTemperature;
        _gribCode = {0, 128, 130, 100};
        _unit = degK;
    } else if (_dataId.Contains("w")) {
        _parameter = VerticalVelocity;
        _gribCode = {0, 128, 135, 100};
        _unit = Pa_s;
    } else if (_dataId.Contains("r")) {
        _parameter = RelativeHumidity;
        _gribCode = {0, 128, 157, 100};
        _unit = percent;
    } else if (_dataId.Contains("u")) {
        _parameter = Uwind;
        _gribCode = {0, 128, 131, 100};
        _unit = _s;
    } else if (_dataId.Contains("v")) {
        _parameter = Vwind;
        _gribCode = {0, 128, 132, 100};
        _unit = _s;
    } else {
        wxLogError(_("No '%s' parameter identified."), _dataId);
        return false;
    }

    _initialized = true;

    return true;
}

double asPredictorOperCustomFvg::FixTimeValue(double time) const {
    if (_dataId.Contains("cp_sfc")) {
        time -= 3.0 / 24.0;
    } else if (_dataId.Contains("tp_sfc")) {
        time -= 3.0 / 24.0;
    }

    return time;
}

wxString asPredictorOperCustomFvg::GetDirStructure(const double date) {
    wxString dirStructure = "YYYYMMDD";
    dirStructure.Append(DS);
    dirStructure.Append("grib");

    return asTime::GetStringTime(date, dirStructure);
}

wxString asPredictorOperCustomFvg::GetFileName(const double date, const int leadTime) {
    wxString timeStr = asStrF("%d", leadTime);
    if (timeStr.Length() < 2) timeStr = "0" + timeStr;

    wxString dateStr = asTime::GetStringTime(date, "YYYYMMDD");

    return asStrF("%s.%s%s.%s", _dataId, dateStr, timeStr, _fileExtension);
}
