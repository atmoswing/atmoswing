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
 * Portions Copyright 2016 Pascal Horton, University of Bern.
 */

#include "asPredictorNoaa20Cr2c.h"

#include "asAreaCompGrid.h"
#include "asTimeArray.h"

asPredictorNoaa20Cr2c::asPredictorNoaa20Cr2c(const wxString &dataId) : asPredictor(dataId) {
  // Set the basic properties.
  m_datasetId = "NOAA_20CR_v2c";
  m_provider = "NOAA";
  m_datasetName = "Twentieth Century Reanalysis (v2c)";
  m_fileType = asFile::Netcdf;
  m_strideAllowed = true;
  m_nanValues.push_back(-9.96921 * std::pow(10.f, 36.f));
  m_fStr.dimLatName = "lat";
  m_fStr.dimLonName = "lon";
  m_fStr.dimTimeName = "time";
  m_fStr.dimLevelName = "level";
}

bool asPredictorNoaa20Cr2c::Init() {
  CheckLevelTypeIsDefined();

  // Identify data ID and set the corresponding properties.
  if (IsPressureLevel()) {
    m_fStr.hasLevelDim = true;
    if (IsAirTemperature()) {
      m_parameter = AirTemperature;
      m_parameterName = "Air Temperature";
      m_fileVarName = "air";
      m_unit = degK;
    } else if (IsGeopotentialHeight()) {
      m_parameter = GeopotentialHeight;
      m_parameterName = "Geopotential height";
      m_fileVarName = "hgt";
      m_unit = m;
    } else if (IsVerticalVelocity()) {
      m_parameter = VerticalVelocity;
      m_parameterName = "Vertical velocity";
      m_fileVarName = "omega";
      m_unit = Pa_s;
    } else if (IsRelativeHumidity()) {
      m_parameter = RelativeHumidity;
      m_parameterName = "Relative Humidity";
      m_fileVarName = "rhum";
      m_unit = percent;
    } else if (IsSpecificHumidity()) {
      m_parameter = SpecificHumidity;
      m_parameterName = "Specific Humidity";
      m_fileVarName = "shum";
      m_unit = kg_kg;
    } else if (IsUwindComponent()) {
      m_parameter = Uwind;
      m_parameterName = "U-Wind";
      m_fileVarName = "uwnd";
      m_unit = m_s;
    } else if (IsVwindComponent()) {
      m_parameter = Vwind;
      m_parameterName = "V-Wind";
      m_fileVarName = "vwnd";
      m_unit = m_s;
    } else {
      m_parameter = ParameterUndefined;
      m_parameterName = "Undefined";
      m_fileVarName = m_dataId;
      m_unit = UnitUndefined;
    }
    m_fileNamePattern = m_fileVarName + ".%d.nc";

  } else if (IsSurfaceLevel() || m_product.IsSameAs("monolevel", false)) {
    m_fStr.hasLevelDim = false;
    if (IsPrecipitableWater()) {
      m_parameter = PrecipitableWater;
      m_parameterName = "Precipitable water";
      m_fileNamePattern = "pr_wtr.eatm.%d.nc";
      m_fileVarName = "pr_wtr";
      m_unit = mm;
    } else if (IsSeaLevelPressure()) {
      m_parameter = Pressure;
      m_parameterName = "Sea level pressure";
      m_fileNamePattern = "prmsl.%d.nc";
      m_fileVarName = "prmsl";
      m_unit = Pa;
    } else {
      wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), m_dataId, m_product);
      return false;
    }

  } else if (IsSurfaceFluxesLevel() || m_product.IsSameAs("surface_gauss", false) ||
             m_product.IsSameAs("gauss", false) || m_product.IsSameAs("gaussian", false)) {
    m_fStr.hasLevelDim = false;
    if (IsPrecipitationRate()) {
      m_parameter = PrecipitationRate;
      m_parameterName = "Precipitation rate";
      m_fileNamePattern = "prate.%d.nc";
      m_fileVarName = "prate";
      m_unit = kg_m2_s;
    } else {
      wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), m_dataId, m_product);
      return false;
    }

  } else {
    wxLogError(_("Product type not implemented for this reanalysis dataset."));
    return false;
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

void asPredictorNoaa20Cr2c::ListFiles(asTimeArray &timeArray) {
  for (int iYear = timeArray.GetStartingYear(); iYear <= timeArray.GetEndingYear(); iYear++) {
    m_files.push_back(GetFullDirectoryPath() + wxString::Format(m_fileNamePattern, iYear));
  }
}

double asPredictorNoaa20Cr2c::ConvertToMjd(double timeValue, double refValue) const {
  timeValue = (timeValue / 24.0);           // hours to days
  timeValue += asTime::GetMJD(1800, 1, 1);  // to MJD: add a negative time span

  return timeValue;
}
