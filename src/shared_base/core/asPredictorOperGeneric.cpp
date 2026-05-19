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

#include "asPredictorOperGeneric.h"

#include <wx/dir.h>
#include <wx/regex.h>

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorOperGeneric::asPredictorOperGeneric(const wxString& dataId)
    : asPredictorOper(dataId) {
    // Set the basic properties.
    _datasetId = "Generic";
    _provider = "";
    _datasetName = "Generic";
    _fileType = asFile::Netcdf;
    _strideAllowed = true;
    _nanValues.push_back(-32767);
    _nanValues.push_back(3.4E38f);
    _nanValues.push_back(100000002004087730000.0);
    _fStr.dimLatName = "lat";
    _fStr.dimLonName = "lon";
    _fStr.dimTimeName = "time";
    _fStr.dimLevelName = "level";
}

bool asPredictorOperGeneric::Init() {
    _parameter = ParameterUndefined;
    _parameterName = "Undefined";
    _fileVarName = _dataId;
    _unit = UnitUndefined;
    _fStr.hasLevelDim = true;
    _fileNamePattern = "%s." + _datasetId + "." + _dataId + ".nc";

    if (_dataId.IsSameAs("d", false)) {
        _parameter = Divergence;
        _parameterName = "Divergence";
        _unit = per_s;
    } else if (IsPotentialVorticity()) {
        _parameter = PotentialVorticity;
        _parameterName = "Potential vorticity";
        _unit = degKm2_kg_s;
    } else if (IsSpecificHumidity()) {
        _parameter = SpecificHumidity;
        _parameterName = "Specific humidity";
        _unit = kg_kg;
    } else if (IsRelativeHumidity()) {
        _parameter = RelativeHumidity;
        _parameterName = "Relative humidity";
        _unit = percent;
    } else if (IsAirTemperature()) {
        _parameter = AirTemperature;
        _parameterName = "Temperature";
        _unit = degK;
    } else if (IsUwindComponent()) {
        _parameter = Uwind;
        _parameterName = "U component of wind";
        _unit = _s;
    } else if (IsVwindComponent()) {
        _parameter = Vwind;
        _parameterName = "V component of wind";
        _unit = _s;
    } else if (_dataId.IsSameAs("vo", false)) {
        _parameter = Vorticity;
        _parameterName = "Vorticity (relative)";
        _unit = per_s;
    } else if (IsVerticalVelocity()) {
        _parameter = VerticalVelocity;
        _parameterName = "Vertical velocity";
        _unit = Pa_s;
    } else if (IsGeopotential()) {
        _parameter = Geopotential;
        _parameterName = "Geopotential";
        _unit = m2_s2;
    } else if (IsGeopotentialHeight()) {
        _parameter = GeopotentialHeight;
        _parameterName = "Geopotential height";
        _unit = m;
    }

    // Surface analysis
    if (_dataId.IsSameAs("d2m", false)) {
        _parameter = DewpointTemperature;
        _parameterName = "2 metre dewpoint temperature";
        _unit = degK;
    } else if (IsSeaLevelPressure()) {
        _parameter = Pressure;
        _parameterName = "Sea level pressure";
        _unit = Pa;
    } else if (_dataId.IsSameAs("sd", false)) {
        _parameter = SnowWaterEquivalent;
        _parameterName = "Snow depth";
        _unit = m;
    } else if (_dataId.IsSameAs("sst", false)) {
        _parameter = SeaSurfaceTemperature;
        _parameterName = "Sea surface temperature";
        _unit = degK;
    } else if (_dataId.IsSameAs("t2m", false)) {
        _parameter = AirTemperature;
        _parameterName = "2 metre temperature";
        _unit = degK;
    } else if (_dataId.IsSameAs("tcw", false)) {
        _parameter = TotalColumnWater;
        _parameterName = "Total column water";
        _unit = kg_m2;
    } else if (_dataId.IsSameAs("tcwv", false)) {
        _parameter = PrecipitableWater;
        _parameterName = "Total column water vapour";
        _unit = kg_m2;
    } else if (IsPrecipitableWater()) {
        _parameter = PrecipitableWater;
        _parameterName = "Precipitable water";
        _unit = kg_m2;
    } else if (_dataId.IsSameAs("u10", false)) {
        _parameter = Uwind;
        _parameterName = "10 metre U wind component";
        _unit = _s;
    } else if (_dataId.IsSameAs("v10", false)) {
        _parameter = Vwind;
        _parameterName = "10 metre V wind component";
        _unit = _s;
    } else if (IsTotalPrecipitation()) {
        _parameter = Precipitation;
        _parameterName = "Total precipitation";
        _unit = m;
    } else if (_dataId.IsSameAs("cape", false)) {
        _parameter = CAPE;
        _parameterName = "Convective available potential energy";
        _unit = J_kg;
    } else if (_dataId.IsSameAs("ie", false)) {
        _parameter = MoistureFlux;
        _parameterName = "Instantaneous moisture flux";
        _unit = kg_m2_s;
    } else if (_dataId.IsSameAs("ssr", false)) {
        _parameter = Radiation;
        _parameterName = "Surface net solar radiation";
        _unit = J_m2;
    } else if (_dataId.IsSameAs("ssrd", false)) {
        _parameter = Radiation;
        _parameterName = "Surface solar radiation downwards";
        _unit = J_m2;
    } else if (_dataId.IsSameAs("str", false)) {
        _parameter = Radiation;
        _parameterName = "Surface net thermal radiation";
        _unit = J_m2;
    } else if (_dataId.IsSameAs("strd", false)) {
        _parameter = Radiation;
        _parameterName = "Surface thermal radiation downwards";
        _unit = J_m2;
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

void asPredictorOperGeneric::ConvertToMjd(a1d& time, double refValue) const {
    // Nothing to do
}

wxString asPredictorOperGeneric::GetFileName(const double date, const int leadTime) {
    wxString dateForecast = asTime::GetStringTime(date, "YYYYMMDDhhmm");

    return asStrF(_fileNamePattern, dateForecast);
}