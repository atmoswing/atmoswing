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
 * Portions Copyright 2017-2019 Pascal Horton, University of Bern.
 */

#include "asPredictorEcmwfEraInterim.h"

#include "asAreaCompGrid.h"
#include "asTimeArray.h"
#include <wx/dir.h>
#include <wx/regex.h>

asPredictorEcmwfEraInterim::asPredictorEcmwfEraInterim(const wxString &dataId) : asPredictor(dataId) {
  // Set the basic properties.
  m_datasetId = "ECMWF_ERA_interim";
  m_provider = "ECMWF";
  m_datasetName = "ERA-interim";
  m_fileType = asFile::Netcdf;
  m_strideAllowed = true;
  m_nanValues.push_back(-32767);
  m_fStr.dimLatName = "latitude";
  m_fStr.dimLonName = "longitude";
  m_fStr.dimTimeName = "time";
  m_fStr.dimLevelName = "level";
}

bool asPredictorEcmwfEraInterim::Init() {
  CheckLevelTypeIsDefined();

  // List of variables: http://rda.ucar.edu/datasets/ds627.0/docs/era_interim_grib_table.html

  // Identify data ID and set the corresponding properties.
  if (IsPressureLevel()) {
    m_fStr.hasLevelDim = true;
    if (m_dataId.IsSameAs("d", false)) {
      m_parameter = Divergence;
      m_parameterName = "Divergence";
      m_fileVarName = "d";
      m_unit = per_s;
    } else if (IsPotentialVorticity()) {
      m_parameter = PotentialVorticity;
      m_parameterName = "Potential vorticity";
      m_fileVarName = "pv";
      m_unit = degKm2_kg_s;
    } else if (IsSpecificHumidity()) {
      m_parameter = SpecificHumidity;
      m_parameterName = "Specific humidity";
      m_fileVarName = "q";
      m_unit = kg_kg;
    } else if (IsRelativeHumidity()) {
      m_parameter = RelativeHumidity;
      m_parameterName = "Relative humidity";
      m_fileVarName = "r";
      m_unit = percent;
    } else if (IsAirTemperature()) {
      m_parameter = AirTemperature;
      m_parameterName = "Temperature";
      m_fileVarName = "t";
      m_unit = degK;
    } else if (IsUwindComponent()) {
      m_parameter = Uwind;
      m_parameterName = "U component of wind";
      m_fileVarName = "u";
      m_unit = m_s;
    } else if (IsVwindComponent()) {
      m_parameter = Vwind;
      m_parameterName = "V component of wind";
      m_fileVarName = "v";
      m_unit = m_s;
    } else if (m_dataId.IsSameAs("vo", false)) {
      m_parameter = Vorticity;
      m_parameterName = "Vorticity (relative)";
      m_fileVarName = "vo";
      m_unit = per_s;
    } else if (IsVerticalVelocity()) {
      m_parameter = VerticalVelocity;
      m_parameterName = "Vertical velocity";
      m_fileVarName = "w";
      m_unit = Pa_s;
    } else if (IsGeopotential()) {
      m_parameter = Geopotential;
      m_parameterName = "Geopotential";
      m_fileVarName = "z";
      m_unit = m2_s2;
    } else {
      m_parameter = ParameterUndefined;
      m_parameterName = "Undefined";
      m_fileVarName = m_dataId;
      m_unit = UnitUndefined;
    }

  } else if (IsIsentropicLevel()) {
    m_fStr.hasLevelDim = true;
    if (m_dataId.IsSameAs("d", false)) {
      m_parameter = Divergence;
      m_parameterName = "Divergence";
      m_fileVarName = "d";
      m_unit = per_s;
    } else if (m_dataId.IsSameAs("mont", false)) {
      m_parameter = MontgomeryPotential;
      m_parameterName = "Montgomery potential";
      m_fileVarName = "mont";
      m_unit = m2_s2;
    } else if (IsPressure()) {
      m_parameter = Pressure;
      m_parameterName = "Pressure";
      m_fileVarName = "pres";
      m_unit = Pa;
    } else if (IsPotentialVorticity()) {
      m_parameter = PotentialVorticity;
      m_parameterName = "Potential vorticity";
      m_fileVarName = "pv";
      m_unit = degKm2_kg_s;
    } else if (IsSpecificHumidity()) {
      m_parameter = SpecificHumidity;
      m_parameterName = "Specific humidity";
      m_fileVarName = "q";
      m_unit = kg_kg;
    } else if (IsUwindComponent()) {
      m_parameter = Uwind;
      m_parameterName = "U component of wind";
      m_fileVarName = "u";
      m_unit = m_s;
    } else if (IsVwindComponent()) {
      m_parameter = Vwind;
      m_parameterName = "V component of wind";
      m_fileVarName = "v";
      m_unit = m_s;
    } else if (m_dataId.IsSameAs("vo", false)) {
      m_parameter = Vorticity;
      m_parameterName = "Vorticity (relative)";
      m_fileVarName = "vo";
      m_unit = per_s;
    } else {
      m_parameter = ParameterUndefined;
      m_parameterName = "Undefined";
      m_fileVarName = m_dataId;
      m_unit = UnitUndefined;
    }

  } else if (IsSurfaceLevel() || m_product.IsSameAs("sfa", false) || m_product.IsSameAs("sfan", false) ||
             m_product.IsSameAs("sff", false) || m_product.IsSameAs("sffc", false)) {
    m_fStr.hasLevelDim = false;
    // Surface analysis
    if (m_dataId.IsSameAs("d2m", false)) {
      m_parameter = DewpointTemperature;
      m_parameterName = "2 metre dewpoint temperature";
      m_fileVarName = "d2m";
      m_unit = degK;
    } else if (IsSeaLevelPressure()) {
      m_parameter = Pressure;
      m_parameterName = "Sea level pressure";
      m_fileVarName = "msl";
      m_unit = Pa;
    } else if (m_dataId.IsSameAs("sd", false)) {
      m_parameter = SnowWaterEquivalent;
      m_parameterName = "Snow depth";
      m_fileVarName = "sd";
      m_unit = m;
    } else if (m_dataId.IsSameAs("sst", false)) {
      m_parameter = SeaSurfaceTemperature;
      m_parameterName = "Sea surface temperature";
      m_fileVarName = "sst";
      m_unit = degK;
    } else if (m_dataId.IsSameAs("t2m", false)) {
      m_parameter = AirTemperature;
      m_parameterName = "2 metre temperature";
      m_fileVarName = "t2m";
      m_unit = degK;
    } else if (IsPrecipitableWater()) {
      m_parameter = PrecipitableWater;
      m_parameterName = "Total column water";
      m_fileVarName = "tcw";
      m_unit = kg_m2;
    } else if (m_dataId.IsSameAs("tcwv", false)) {
      m_parameter = WaterVapour;
      m_parameterName = "Total column water vapour";
      m_fileVarName = "tcwv";
      m_unit = kg_m2;
    } else if (m_dataId.IsSameAs("u10", false)) {
      m_parameter = Uwind;
      m_parameterName = "10 metre U wind component";
      m_fileVarName = "u10";
      m_unit = m_s;
    } else if (m_dataId.IsSameAs("v10", false)) {
      m_parameter = Vwind;
      m_parameterName = "10 metre V wind component";
      m_fileVarName = "v10";
      m_unit = m_s;
    } else if (IsTotalPrecipitation()) {
      m_parameter = Precipitation;
      m_parameterName = "Total precipitation";
      m_fileVarName = "tp";
      m_unit = m;
    } else if (m_dataId.IsSameAs("cape", false)) {
      m_parameter = CAPE;
      m_parameterName = "Convective available potential energy";
      m_fileVarName = "cape";
      m_unit = J_kg;
    } else if (m_dataId.IsSameAs("ie", false)) {
      m_parameter = MoistureFlux;
      m_parameterName = "Instantaneous moisture flux";
      m_fileVarName = "ie";
      m_unit = kg_m2_s;
    } else if (m_dataId.IsSameAs("ssr", false)) {
      m_parameter = Radiation;
      m_parameterName = "Surface net solar radiation";
      m_fileVarName = "ssr";
      m_unit = J_m2;
    } else if (m_dataId.IsSameAs("ssrd", false)) {
      m_parameter = Radiation;
      m_parameterName = "Surface solar radiation downwards";
      m_fileVarName = "ssrd";
      m_unit = J_m2;
    } else if (m_dataId.IsSameAs("str", false)) {
      m_parameter = Radiation;
      m_parameterName = "Surface net thermal radiation";
      m_fileVarName = "str";
      m_unit = J_m2;
    } else if (m_dataId.IsSameAs("strd", false)) {
      m_parameter = Radiation;
      m_parameterName = "Surface thermal radiation downwards";
      m_fileVarName = "strd";
      m_unit = J_m2;
    } else {
      m_parameter = ParameterUndefined;
      m_parameterName = "Undefined";
      m_fileVarName = m_dataId;
      m_unit = UnitUndefined;
    }

  } else if (IsPVLevel()) {
    m_fStr.hasLevelDim = false;
    if (IsPressure()) {
      m_parameter = Pressure;
      m_parameterName = "Pressure";
      m_fileVarName = "pres";
      m_unit = Pa;
    } else if (m_dataId.IsSameAs("pt", false)) {
      m_parameter = PotentialTemperature;
      m_parameterName = "Potential temperature";
      m_fileVarName = "pt";
      m_unit = degK;
    } else if (IsUwindComponent()) {
      m_parameter = Uwind;
      m_parameterName = "U component of wind";
      m_fileVarName = "u";
      m_unit = m_s;
    } else if (IsVwindComponent()) {
      m_parameter = Vwind;
      m_parameterName = "V component of wind";
      m_fileVarName = "v";
      m_unit = m_s;
    } else if (IsGeopotential()) {
      m_parameter = Geopotential;
      m_parameterName = "Geopotential";
      m_fileVarName = "z";
      m_unit = m2_s2;
    } else {
      m_parameter = ParameterUndefined;
      m_parameterName = "Undefined";
      m_fileVarName = m_dataId;
      m_unit = UnitUndefined;
    }

  } else {
    wxLogError(_("level type not implemented for this reanalysis dataset."));
    return false;
  }

  // Check data ID
  if (m_fileVarName.IsEmpty()) {
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

void asPredictorEcmwfEraInterim::ListFiles(asTimeArray &timeArray) {
  // Case 1: single file with the variable name
  wxString filePath = GetFullDirectoryPath() + m_fileVarName + ".nc";

  if (wxFileExists(filePath)) {
    m_files.push_back(filePath);
    return;
  }

  // Case 2: yearly files
  wxArrayString listFiles;
  size_t nbFiles = wxDir::GetAllFiles(GetFullDirectoryPath(), &listFiles, "*.nc");

  if (nbFiles == 0) {
    asThrowException(_("No ERA-interim file found."));
  }

  listFiles.Sort();

  double firstYear = timeArray.GetStartingYear();
  double lastYear = timeArray.GetEndingYear();

  for (size_t i = 0; i < listFiles.Count(); ++i) {
    wxRegEx reDates("\\d{4,}", wxRE_ADVANCED);
    if (!reDates.Matches(listFiles.Item(i))) {
      continue;
    }

    wxString datesSrt = reDates.GetMatch(listFiles.Item(i));
    double fileYear = 0;
    datesSrt.ToDouble(&fileYear);

    if (fileYear < firstYear || fileYear > lastYear) {
      if (m_product.IsSameAs("sff", false) || m_product.IsSameAs("sffc", false)) {
        if (fileYear != firstYear - 1) {
          continue;
        }
      } else {
        continue;
      }
    }

    m_files.push_back(listFiles.Item(i));
  }

  if (!m_files.empty()) {
    return;
  }

  // Case 3: list all files from the directory
  for (size_t i = 0; i < listFiles.Count(); ++i) {
    m_files.push_back(listFiles.Item(i));
  }
}

double asPredictorEcmwfEraInterim::ConvertToMjd(double timeValue, double refValue) const {
  timeValue = (timeValue / 24.0);           // hours to days
  timeValue += asTime::GetMJD(1900, 1, 1);  // to MJD: add a negative time span

  return timeValue;
}
