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
    m_datasetId = "Generic";
    m_provider = "";
    m_datasetName = "Generic";
    m_fileType = asFile::Netcdf;
    m_strideAllowed = true;
    m_nanValues.push_back(-32767);
    m_nanValues.push_back(3.4E38f);
    m_nanValues.push_back(100000002004087730000.0);
    m_fStr.dimLatName = "lat";
    m_fStr.dimLonName = "lon";
    m_fStr.dimTimeName = "time";
    m_fStr.dimLevelName = "level";
}

bool asPredictorOperGeneric::Init() {
    m_parameter = ParameterUndefined;
    m_parameterName = "Undefined";
    m_fileVarName = m_dataId;
    m_unit = UnitUndefined;
    m_fStr.hasLevelDim = true;
    m_fileNamePattern = "%s." + m_datasetId + "." + m_dataId + ".nc";

    if (m_dataId.IsSameAs("d", false)) {
        m_parameter = Divergence;
        m_parameterName = "Divergence";
        m_unit = per_s;
    } else if (IsPotentialVorticity()) {
        m_parameter = PotentialVorticity;
        m_parameterName = "Potential vorticity";
        m_unit = degKm2_kg_s;
    } else if (IsSpecificHumidity()) {
        m_parameter = SpecificHumidity;
        m_parameterName = "Specific humidity";
        m_unit = kg_kg;
    } else if (IsRelativeHumidity()) {
        m_parameter = RelativeHumidity;
        m_parameterName = "Relative humidity";
        m_unit = percent;
    } else if (IsAirTemperature()) {
        m_parameter = AirTemperature;
        m_parameterName = "Temperature";
        m_unit = degK;
    } else if (IsUwindComponent()) {
        m_parameter = Uwind;
        m_parameterName = "U component of wind";
        m_unit = m_s;
    } else if (IsVwindComponent()) {
        m_parameter = Vwind;
        m_parameterName = "V component of wind";
        m_unit = m_s;
    } else if (m_dataId.IsSameAs("vo", false)) {
        m_parameter = Vorticity;
        m_parameterName = "Vorticity (relative)";
        m_unit = per_s;
    } else if (IsVerticalVelocity()) {
        m_parameter = VerticalVelocity;
        m_parameterName = "Vertical velocity";
        m_unit = Pa_s;
    } else if (IsGeopotential()) {
        m_parameter = Geopotential;
        m_parameterName = "Geopotential";
        m_unit = m2_s2;
    } else if (IsGeopotentialHeight()) {
        m_parameter = GeopotentialHeight;
        m_parameterName = "Geopotential height";
        m_unit = m;
    }

    // Surface analysis
    if (m_dataId.IsSameAs("d2m", false)) {
        m_parameter = DewpointTemperature;
        m_parameterName = "2 metre dewpoint temperature";
        m_unit = degK;
    } else if (IsSeaLevelPressure()) {
        m_parameter = Pressure;
        m_parameterName = "Sea level pressure";
        m_unit = Pa;
    } else if (m_dataId.IsSameAs("sd", false)) {
        m_parameter = SnowWaterEquivalent;
        m_parameterName = "Snow depth";
        m_unit = m;
    } else if (m_dataId.IsSameAs("sst", false)) {
        m_parameter = SeaSurfaceTemperature;
        m_parameterName = "Sea surface temperature";
        m_unit = degK;
    } else if (m_dataId.IsSameAs("t2m", false)) {
        m_parameter = AirTemperature;
        m_parameterName = "2 metre temperature";
        m_unit = degK;
    } else if (m_dataId.IsSameAs("tcw", false)) {
        m_parameter = TotalColumnWater;
        m_parameterName = "Total column water";
        m_unit = kg_m2;
    } else if (m_dataId.IsSameAs("tcwv", false)) {
        m_parameter = PrecipitableWater;
        m_parameterName = "Total column water vapour";
        m_unit = kg_m2;
    } else if (IsPrecipitableWater()) {
        m_parameter = PrecipitableWater;
        m_parameterName = "Precipitable water";
        m_unit = kg_m2;
    } else if (m_dataId.IsSameAs("u10", false)) {
        m_parameter = Uwind;
        m_parameterName = "10 metre U wind component";
        m_unit = m_s;
    } else if (m_dataId.IsSameAs("v10", false)) {
        m_parameter = Vwind;
        m_parameterName = "10 metre V wind component";
        m_unit = m_s;
    } else if (IsTotalPrecipitation()) {
        m_parameter = Precipitation;
        m_parameterName = "Total precipitation";
        m_unit = m;
    } else if (m_dataId.IsSameAs("cape", false)) {
        m_parameter = CAPE;
        m_parameterName = "Convective available potential energy";
        m_unit = J_kg;
    } else if (m_dataId.IsSameAs("ie", false)) {
        m_parameter = MoistureFlux;
        m_parameterName = "Instantaneous moisture flux";
        m_unit = kg_m2_s;
    } else if (m_dataId.IsSameAs("ssr", false)) {
        m_parameter = Radiation;
        m_parameterName = "Surface net solar radiation";
        m_unit = J_m2;
    } else if (m_dataId.IsSameAs("ssrd", false)) {
        m_parameter = Radiation;
        m_parameterName = "Surface solar radiation downwards";
        m_unit = J_m2;
    } else if (m_dataId.IsSameAs("str", false)) {
        m_parameter = Radiation;
        m_parameterName = "Surface net thermal radiation";
        m_unit = J_m2;
    } else if (m_dataId.IsSameAs("strd", false)) {
        m_parameter = Radiation;
        m_parameterName = "Surface thermal radiation downwards";
        m_unit = J_m2;
    }

    // Check directory is set
    if (GetDirectoryPath().IsEmpty()) {
        wxLogError(_("The path to the directory has not been set for the data %s from the dataset %s."), m_dataId,
                   m_datasetName);
        return false;
    }

    // Set to initialized
    m_initialized = true;

    return true;
}

void asPredictorOperGeneric::ConvertToMjd(a1d& time, double refValue) const {
    // Nothing to do
}

wxString asPredictorOperGeneric::GetFileName(const double date, const int leadTime) {
    wxString dateForecast = asTime::GetStringTime(date, "YYYYMMDDhhmm");

    return asStrF(m_fileNamePattern, dateForecast);
}