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

#include "asAreaCompGrid.h"
#include "asTimeArray.h"

asPredictorCustomUnilOisst2::asPredictorCustomUnilOisst2(const wxString &dataId) : asPredictor(dataId) {
  // Set the basic properties.
  m_datasetId = "Custom_Unil_OISST_v2";
  m_provider = "NOAA";
  m_transformedBy = "Pascal Horton";
  m_datasetName = "Optimum Interpolation Sea Surface Temperature, version 2, subset";
  m_fileType = asFile::Netcdf;
  m_strideAllowed = true;
  m_nanValues.push_back(32767);
  m_nanValues.push_back(936 * std::pow(10.f, 34.f));
  m_fStr.dimLatName = "lat";
  m_fStr.dimLonName = "lon";
  m_fStr.dimTimeName = "time";
  m_fStr.hasLevelDim = false;
}

bool asPredictorCustomUnilOisst2::Init() {
  // Identify data ID and set the corresponding properties.
  if (m_dataId.IsSameAs("sst", false)) {
    m_parameter = SeaSurfaceTemperature;
    m_parameterName = "Sea Surface Temperature";
    m_fileNamePattern = "sst_1deg.nc";
    m_fileVarName = "sst";
    m_unit = degC;
  } else if (m_dataId.IsSameAs("sst_anom", false)) {
    m_parameter = SeaSurfaceTemperatureAnomaly;
    m_parameterName = "Sea Surface Temperature Anomaly";
    m_fileNamePattern = "sst_anom_1deg.nc";
    m_fileVarName = "anom";
    m_unit = degC;
  } else {
    wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), m_dataId, m_product);
    return false;
  }

  // Check data ID
  if (m_fileNamePattern.IsEmpty() || m_fileVarName.IsEmpty()) {
    wxLogError(_("The provided data ID (%s) does not match any possible option in dataset %s."), m_dataId,
               m_datasetName);
    return false;
  }

  // Check directory is set
  if (GetDirectoryPath().IsEmpty()) {
    wxLogError(_("The path to the directory has not been set for the data %s from dataset %s."), m_dataId,
               m_datasetName);
    return false;
  }

  // Set to initialized
  m_initialized = true;

  return true;
}

void asPredictorCustomUnilOisst2::ListFiles(asTimeArray &timeArray) {
  m_files.push_back(GetFullDirectoryPath() + m_fileNamePattern);
}

double asPredictorCustomUnilOisst2::ConvertToMjd(double timeValue, double refValue) const {
  timeValue = (timeValue / 24.0);             // hours to days
  if (timeValue < 500 * 365) {                // New format
    timeValue += asTime::GetMJD(1800, 1, 1);  // to MJD: add a negative time span
  } else {                                    // Old format
    timeValue += asTime::GetMJD(1, 1, 1);     // to MJD: add a negative time span
  }

  return timeValue;
}
