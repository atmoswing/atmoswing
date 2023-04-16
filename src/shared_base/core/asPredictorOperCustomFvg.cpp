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

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorOperCustomFvg::asPredictorOperCustomFvg(const wxString& dataId)
    : asPredictorOperEcmwfIfs(dataId) {
    // Set the basic properties.
    m_datasetId = "Custom_MeteoFVG";
    m_datasetName = "Integrated Forecasting System (IFS) grib files at Meteo FVG";
    m_fStr.hasLevelDim = true;
    m_leadTimeStart = 6;
    m_runHourStart = 0;
    m_runUpdate = 24;
    m_percentMissingAllowed = 70;
}

bool asPredictorOperCustomFvg::Init() {
    m_parameter = Other;

    if (m_dataId.IsSameAs("DP500925", false)) {
        m_gribCode = {0, 3, 113, 100};
    } else if (m_dataId.IsSameAs("LRT700500", false)) {
        m_gribCode = {0, 128, 130, 100};
    } else if (m_dataId.IsSameAs("LRT850500", false)) {
        m_gribCode = {0, 128, 130, 100};
    } else if (m_dataId.IsSameAs("LRTE700500", false)) {
        m_gribCode = {0, 3, 113, 100};
    } else if (m_dataId.IsSameAs("LRTE850500", false)) {
        m_gribCode = {0, 3, 113, 100};
    } else if (m_dataId.IsSameAs("MB500850", false)) {
        m_parameter = MaximumBuoyancy;
        m_gribCode = {0, 3, 114, 100};
    } else if (m_dataId.IsSameAs("MB500925", false)) {
        m_parameter = MaximumBuoyancy;
        m_gribCode = {0, 3, 114, 100};
    } else if (m_dataId.IsSameAs("MB700925", false)) {
        m_parameter = MaximumBuoyancy;
        m_gribCode = {0, 3, 114, 100};
    } else if (m_dataId.IsSameAs("MB850500", false)) {
        m_parameter = MaximumBuoyancy;
        m_gribCode = {0, 3, 114, 100};
    } else if (m_dataId.Contains("thetaES")) {
        m_parameter = PotentialTemperature;
        m_gribCode = {0, 3, 114, 100};
        m_unit = W_m2;
    } else if (m_dataId.Contains("thetaE")) {
        m_parameter = PotentialTemperature;
        m_gribCode = {0, 3, 113, 100};
        m_unit = W_m2;
    } else if (m_dataId.Contains("vflux")) {
        m_parameter = MomentumFlux;
        m_gribCode = {0, 3, 125, 100};
        m_unit = kg_m2_s;
    } else if (m_dataId.Contains("uflux")) {
        m_parameter = MomentumFlux;
        m_gribCode = {0, 3, 124, 100};
        m_unit = kg_m2_s;
    } else if (m_dataId.Contains("2t_sfc")) {
        m_parameter = AirTemperature;
        m_gribCode = {0, 128, 167, 1};
        m_unit = degK;
    } else if (m_dataId.Contains("10u_sfc")) {
        m_parameter = Uwind;
        m_gribCode = {0, 128, 165, 1};
        m_unit = m_s;
    } else if (m_dataId.Contains("10v_sfc")) {
        m_parameter = Vwind;
        m_gribCode = {0, 128, 166, 1};
        m_unit = m_s;
    } else if (m_dataId.Contains("cp_sfc")) {
        m_parameter = Precipitation;
        m_gribCode = {0, 128, 143, 1};
        m_unit = m;
    } else if (m_dataId.Contains("msl_sfc")) {
        m_parameter = Pressure;
        m_gribCode = {0, 128, 151, 1};
        m_unit = Pa;
    } else if (m_dataId.Contains("tp_sfc")) {
        m_parameter = AirTemperature;
        m_gribCode = {0, 128, 228, 1};
        m_unit = degK;
    } else if (m_dataId.Contains("q")) {
        m_parameter = SpecificHumidity;
        m_gribCode = {0, 128, 133, 100};
        m_unit = percent;
    } else if (m_dataId.Contains("gh")) {
        m_parameter = GeopotentialHeight;
        m_gribCode = {0, 128, 156, 100};
        m_unit = m;
    } else if (m_dataId.Contains("t")) {
        m_parameter = AirTemperature;
        m_gribCode = {0, 128, 130, 100};
        m_unit = degK;
    } else if (m_dataId.Contains("w")) {
        m_parameter = VerticalVelocity;
        m_gribCode = {0, 128, 135, 100};
        m_unit = Pa_s;
    } else if (m_dataId.Contains("r")) {
        m_parameter = RelativeHumidity;
        m_gribCode = {0, 128, 157, 100};
        m_unit = percent;
    } else if (m_dataId.Contains("u")) {
        m_parameter = Uwind;
        m_gribCode = {0, 128, 131, 100};
        m_unit = m_s;
    } else if (m_dataId.Contains("v")) {
        m_parameter = Vwind;
        m_gribCode = {0, 128, 132, 100};
        m_unit = m_s;
    } else {
        wxLogError(_("No '%s' parameter identified."), m_dataId);
        return false;
    }

    m_initialized = true;

    return true;
}

double asPredictorOperCustomFvg::FixTimeValue(double time) const {
    if (m_dataId.Contains("cp_sfc")) {
        time -= 3.0 / 24.0;
    } else if (m_dataId.Contains("tp_sfc")) {
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

    return asStrF("%s.%s%s.%s", m_dataId, dateStr, timeStr, m_fileExtension);
}