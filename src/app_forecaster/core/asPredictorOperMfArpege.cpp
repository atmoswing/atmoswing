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

#include "asAreaGrid.h"
#include "asPredictorOperMfArpege.h"
#include "asTimeArray.h"

asPredictorOperMfArpege::asPredictorOperMfArpege(const wxString& dataId)
    : asPredictorOper(dataId) {
    // Set the basic properties.
    m_datasetId = "MF_ARPEGE_Forecast";
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
    m_nanValues.push_back(NaNd);
    m_nanValues.push_back(NaNf);
    m_parameter = ParameterUndefined;
    m_fileExtension = "grb";
    m_leadTimeStart = 0;
    m_leadTimeEnd = 240;
    m_leadTimeStep = 6;
    m_runHourStart = 0;
    m_runUpdate = 6;
    m_restrictHours = 0;
    m_restrictTimeStepHours = 24;
}

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
    } else {
        wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), m_dataId, m_product);
        return false;
    }

    // Set to initialized
    m_initialized = true;

    return true;
}

void asPredictorOperMfArpege::ConvertToMjd(a1d& time, double refValue) const {
    time = (time / 24.0) + refValue;
}

wxString asPredictorOperMfArpege::GetFileName(const double date, const int leadTime) {
    double mjdTarget = date + double(leadTime) / 24.0;
    wxString dateTarget = asTime::GetStringTime(mjdTarget, "YYYYMMDDhhmm");
    wxString dateForecast = asTime::GetStringTime(date, "YYYYMMDDhhmm");

    return asStrF(m_fileNamePattern, (int)m_level, dateTarget, dateForecast);
}