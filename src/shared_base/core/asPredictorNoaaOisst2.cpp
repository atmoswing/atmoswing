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

#include "asPredictorNoaaOisst2.h"

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorNoaaOisst2::asPredictorNoaaOisst2(const wxString& dataId)
    : asPredictor(dataId) {
    // Set the basic properties.
    m_datasetId = "NOAA_OISST_v2";
    m_provider = "NOAA";
    m_datasetName = "Optimum Interpolation Sea Surface Temperature, version 2";
    m_fileType = asFile::Netcdf;
    m_strideAllowed = true;
    m_nanValues.push_back(32767);
    m_nanValues.push_back(936 * std::pow(10.f, 34.f));
    m_fileNamePattern = "%d/AVHRR/sst4-path-eot.%4d%02d%02d.nc";
    m_fStr.dimLatName = "lat";
    m_fStr.dimLonName = "lon";
    m_fStr.hasLevelDim = false;
}

bool asPredictorNoaaOisst2::Init() {
    // Identify data ID and set the corresponding properties.
    if (m_dataId.IsSameAs("sst", false)) {
        m_parameter = SeaSurfaceTemperature;
        m_parameterName = "Sea Surface Temperature";
        m_fileVarName = "sst";
        m_unit = degC;
    } else if (m_dataId.IsSameAs("sst_anom", false)) {
        m_parameter = SeaSurfaceTemperatureAnomaly;
        m_parameterName = "Sea Surface Temperature Anomaly";
        m_fileVarName = "anom";
        m_unit = degC;
    } else {
        m_parameter = ParameterUndefined;
        m_parameterName = "Undefined";
        m_fileVarName = m_dataId;
        m_unit = UnitUndefined;
    }

    // Check data ID
    if (m_fileNamePattern.IsEmpty() || m_fileVarName.IsEmpty()) {
        wxLogError(_("The provided data ID (%s) does not match any possible option in the dataset %s."), m_dataId,
                   m_datasetName);
        return false;
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

void asPredictorNoaaOisst2::ListFiles(asTimeArray& timeArray) {
    for (double date = timeArray.GetFirst(); date <= timeArray.GetLast(); date++) {
        // Build the file path (ex: %d/AVHRR/sst4-path-eot.%4d%02d%02d.nc)
        m_files.push_back(GetFullDirectoryPath() + wxString::Format(m_fileNamePattern, asTime::GetYear(date),
                                                                    asTime::GetYear(date), asTime::GetMonth(date),
                                                                    asTime::GetDay(date)));
    }
}

void asPredictorNoaaOisst2::ConvertToMjd(a1d& time, double refValue) const {
    time += asTime::GetMJD(1978, 1, 1);
}
