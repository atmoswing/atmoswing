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

#include "asAreaGrid.h"
#include "asTimeArray.h"

/**
 * The constructor for the operational ARPEGE predictor from Meteo France.
 *
 * @param dataId Identifier of the data variable (meteorological parameter).
 */
asPredictorOperMfArpege::asPredictorOperMfArpege(const wxString& dataId)
    : asPredictorOper(dataId) {
    // Set the basic properties.
    m_datasetId = "MF_ARPEGE";
    m_provider = "METEOFRANCE";
    m_datasetName = "ARPEGE grib files";
    m_fileType = asFile::Grib;
    m_isEnsemble = false;
    m_strideAllowed = false;
    m_fStr.dimLatName = "lat";
    m_fStr.dimLonName = "lon";
    m_fStr.dimTimeName = "time";
    m_fStr.dimLevelName = "isobaric";
    m_fStr.hasLevelDim = false;
    m_fStr.singleTimeStep = true;
    m_parameter = ParameterUndefined;
    m_fileExtension = "grb";
    m_leadTimeStart = 0;
    m_leadTimeStep = 6;
    m_runHourStart = 0;
    m_runUpdate = 12;
}

/**
 * Initialize the parameters of the data source.
 *
 * @return True if the initialisation went well.
 */
bool asPredictorOperMfArpege::Init() {
    // Identify data ID and set the corresponding properties.
    if (IsGeopotential()) {
        m_parameter = Geopotential;
        m_gribCode = {0, 3, 4, 100};
        m_unit = m2_s2;
        m_fStr.hasLevelDim = true;
        m_fileNamePattern = "ARP_GEOPOTENTIAL__ISOBARIC_SURFACE_%d_%s_%s.grb";
    } else if (IsRelativeHumidity()) {
        m_parameter = RelativeHumidity;
        m_gribCode = {0, 1, 1, 100};
        m_unit = percent;
        m_fStr.hasLevelDim = true;
        m_fileNamePattern = "ARP_RELATIVE_HUMIDITY__ISOBARIC_SURFACE_%d_%s_%s.grb";
    } else if (IsTotalColumnWaterVapour()) {
        m_parameter = PrecipitableWater;
        m_gribCode = {0, 1, 64, 1};
        m_unit = kg_m2;
        m_fStr.hasLevelDim = false;
        m_fileNamePattern = "ARP_TOTAL_COLUMN_INTEGRATED_WATER_VAPOUR__GROUND_OR_WATER_SURFACE_%d_%s_%s.grb";
    } else if (IsAirTemperature()) {
        m_parameter = AirTemperature;
        m_gribCode = {0, 0, 0, 100};
        m_unit = degK;
        m_fStr.hasLevelDim = true;
        m_fileNamePattern = "ARP_TEMPERATURE__ISOBARIC_SURFACE_%d_%s_%s.grb";
    } else if (IsVerticalVelocity()) {
        m_parameter = VerticalVelocity;
        m_gribCode = {0, 2, 8, 100};
        m_unit = Pa_s;
        m_fStr.hasLevelDim = true;
        m_fileNamePattern = "ARP_VERTICAL_VELOCITY_PRESSURE__ISOBARIC_SURFACE_%d_%s_%s.grb";
    } else {
        wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), m_dataId, m_product);
        return false;
    }

    // Set to initialized
    m_initialized = true;

    return true;
}

/**
 * Convert the tima array from hours to MJD.
 *
 * @param time The time array in hours (as in the files).
 * @param refValue The reference value to add to the time array (as in the files).
 */
void asPredictorOperMfArpege::ConvertToMjd(a1d& time, double refValue) const {
    time = (time / 24.0) + refValue;
}

/**
 * Get the file name from the forecast date and the lead time.
 *
 * @param date The forecast date.
 * @param leadTime The lead time.
 * @return The file name.
 */
wxString asPredictorOperMfArpege::GetFileName(const double date, const int leadTime) {
    double mjdTarget = date + double(leadTime) / 24.0;
    wxString dateTarget = asTime::GetStringTime(mjdTarget, "YYYYMMDDhhmm");
    wxString dateForecast = asTime::GetStringTime(date, "YYYYMMDDhhmm");

    return asStrF(m_fileNamePattern, (int)m_level, dateTarget, dateForecast);
}