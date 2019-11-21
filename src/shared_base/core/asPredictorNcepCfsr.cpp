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

#include "asPredictorNcepCfsr.h"

#include "asAreaCompGrid.h"
#include "asTimeArray.h"

asPredictorNcepCfsr::asPredictorNcepCfsr(const wxString &dataId) : asPredictor(dataId) {
  // Set the basic properties.
  m_datasetId = "NCEP_CFSR";
  m_provider = "NCEP";
  m_datasetName = "CFSR";
  m_fileType = asFile::Grib;
  m_strideAllowed = false;
}

bool asPredictorNcepCfsr::Init() {
  CheckLevelTypeIsDefined();

  // Last element in grib code: level type (http://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_table4-5.shtml)

  // Identify data ID and set the corresponding properties.
  if (IsPressureLevel()) {
    m_fStr.hasLevelDim = true;
    m_fStr.singleLevel = true;
    if (IsGeopotentialHeight()) {
      m_parameter = GeopotentialHeight;
      m_gribCode = {0, 3, 5, 100};
      m_parameterName = "Geopotential height @ Isobaric surface";
      m_unit = gpm;
    } else if (IsPrecipitableWater()) {
      m_parameter = PrecipitableWater;
      m_gribCode = {0, 1, 3, 200};
      m_parameterName = "Precipitable water @ Entire atmosphere layer";
      m_unit = kg_m2;
    } else if (IsSeaLevelPressure()) {
      m_parameter = Pressure;
      m_gribCode = {0, 3, 0, 101};
      m_parameterName = "Pressure @ Mean sea level";
      m_unit = Pa;
    } else if (IsRelativeHumidity()) {
      m_parameter = RelativeHumidity;
      m_gribCode = {0, 1, 1, 100};
      m_parameterName = "Relative humidity @ Isobaric surface";
      m_unit = percent;
    } else if (IsAirTemperature()) {
      m_parameter = AirTemperature;
      m_gribCode = {0, 0, 0, 100};
      m_parameterName = "Temperature @ Isobaric surface";
      m_unit = degK;
    } else {
      wxLogError(_("Parameter '%s' not implemented yet."), m_dataId);
      return false;
    }
    m_fileNamePattern = "%4d/%4d%02d/%4d%02d%02d/pgbhnl.gdas.%4d%02d%02d%02d.grb2";

  } else if (IsIsentropicLevel()) {
    wxLogError(_("Isentropic levels for CFSR are not implemented yet."));
    return false;

  } else if (IsSurfaceFluxesLevel()) {
    wxLogError(_("Surface fluxes grids for CFSR are not implemented yet."));
    return false;

  } else {
    wxLogError(_("level type not implemented for this reanalysis dataset."));
    return false;
  }

  // Check data ID
  if (m_fileNamePattern.IsEmpty() || m_gribCode[2] == asNOT_FOUND) {
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

  wxASSERT(m_gribCode.size() == 4);

  // Set to initialized
  m_initialized = true;

  return true;
}

void asPredictorNcepCfsr::ListFiles(asTimeArray &timeArray) {
  a1d tArray = timeArray.GetTimeArray();

  for (int i = 0; i < tArray.size(); i++) {
    Time t = asTime::GetTimeStruct(tArray[i]);
    m_files.push_back(GetFullDirectoryPath() + wxString::Format(m_fileNamePattern, t.year, t.year, t.month, t.year,
                                                                t.month, t.day, t.year, t.month, t.day, t.hour));
  }
}
