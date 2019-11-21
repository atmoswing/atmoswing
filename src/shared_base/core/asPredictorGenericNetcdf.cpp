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
 * Portions Copyright 2019 Pascal Horton, University of Bern.
 */

#include "asPredictorGenericNetcdf.h"

#include <asAreaCompGrid.h>
#include <asTimeArray.h>
#include <wx/dir.h>
#include <wx/regex.h>

asPredictorGenericNetcdf::asPredictorGenericNetcdf(const wxString &dataId) : asPredictor(dataId) {
  // Set the basic properties.
  m_datasetId = "GenericNetcdf";
  m_provider = "";
  m_datasetName = "Generic Netcdf";
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

bool asPredictorGenericNetcdf::Init() {
  m_parameter = ParameterUndefined;
  m_parameterName = "Undefined";
  m_fileVarName = m_dataId;
  m_unit = UnitUndefined;
  m_fStr.hasLevelDim = true;

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

void asPredictorGenericNetcdf::ListFiles(asTimeArray &timeArray) {
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
    asThrowException(_("No file found for the generic archive."));
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
      continue;
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

double asPredictorGenericNetcdf::ConvertToMjd(double timeValue, double refValue) const {
  return timeValue;
}
