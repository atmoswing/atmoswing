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

#include "asPredictandLightnings.h"

#include <asCatalogPredictands.h>
#include <asFileNetcdf.h>
#include <asTimeArray.h>

asPredictandLightnings::asPredictandLightnings(Parameter dataParameter, TemporalResolution dataTemporalResolution,
                                               SpatialAggregation dataSpatialAggregation)
    : asPredictand(dataParameter, dataTemporalResolution, dataSpatialAggregation) {
  m_hasNormalizedData = false;
  m_hasReferenceValues = false;
}

bool asPredictandLightnings::InitContainers() {
  return InitBaseContainers();
}

bool asPredictandLightnings::Load(const wxString &filePath) {
  // Open the NetCDF file
  wxLogVerbose(_("Opening the file %s"), filePath);
  asFileNetcdf ncFile(filePath, asFileNetcdf::ReadOnly);
  if (!ncFile.Open()) {
    wxLogError(_("Couldn't open file %s"), filePath);
    return false;
  } else {
    wxLogVerbose(_("File successfully opened"));
  }

  // Load common data
  LoadCommonData(ncFile);

  if (m_hasNormalizedData) {
    // Get normalized data
    size_t indexStart[2] = {0, 0};
    size_t indexCount[2] = {size_t(m_timeLength), size_t(m_stationsNb)};
    m_dataNormalized.resize(m_timeLength, m_stationsNb);
    ncFile.GetVarArray("data_normalized", indexStart, indexCount, &m_dataNormalized(0, 0));
  }

  // Close the netCDF file
  ncFile.Close();

  return true;
}

bool asPredictandLightnings::Save(const wxString &destinationDir) const {
  // Get the file path
  wxString predictandDBFilePath = GetDBFilePathSaving(destinationDir);

  // Create netCDF dataset: enter define mode
  asFileNetcdf ncFile(predictandDBFilePath, asFileNetcdf::Replace);
  if (!ncFile.Open()) return false;

  // Set common definitions
  SetCommonDefinitions(ncFile);

  if (m_hasNormalizedData) {
    // Define dimensions
    vstds dimNames2D;
    dimNames2D.push_back("time");
    dimNames2D.push_back("stations");

    // Define variable
    ncFile.DefVar("data_normalized", NC_FLOAT, 2, dimNames2D);

    // Put attributes for the variable
    ncFile.PutAtt("long_name", "Normalized data", "data_normalized");
    ncFile.PutAtt("var_desc", "Normalized data", "data_normalized");
    ncFile.PutAtt("units", "-", "data_normalized");
  }

  // End definitions: leave define mode
  ncFile.EndDef();

  // Save common data
  SaveCommonData(ncFile);

  if (m_hasNormalizedData) {
    // Provide sizes for variable
    size_t start2[] = {0, 0};
    size_t count2[] = {size_t(m_timeLength), size_t(m_stationsNb)};

    // Write data
    ncFile.PutVarArray("data_normalized", start2, count2, &m_dataNormalized(0, 0));
  }

  // Close:save new netCDF dataset
  ncFile.Close();

  return true;
}

bool asPredictandLightnings::BuildPredictandDB(const wxString &catalogFilePath, const wxString &dataDir,
                                               const wxString &patternDir, const wxString &destinationDir) {
  if (!g_unitTesting) {
    wxLogVerbose(_("Building the predictand DB."));
  }

  // Initialize the members
  if (!InitMembers(catalogFilePath)) return false;

  // Resize matrices
  if (!InitContainers()) return false;

  // Load data from files
  if (!ParseData(catalogFilePath, dataDir, patternDir)) return false;

  if (m_hasNormalizedData) {
    if (!BuildDataNormalized()) return false;
  }

  if (!destinationDir.IsEmpty()) {
    if (!Save(destinationDir)) return false;
  }

  if (!g_unitTesting) {
    wxLogVerbose(_("Predictand DB saved."));
  }

#if wxUSE_GUI
  if (!g_silentMode) {
    wxMessageBox(_("Predictand DB saved."));
  }
#endif

  return true;
}

bool asPredictandLightnings::BuildDataNormalized() {
  for (int iStat = 0; iStat < m_stationsNb; iStat++) {
    for (int iTime = 0; iTime < m_timeLength; iTime++) {
      m_dataNormalized(iTime, iStat) = log10(m_dataRaw(iTime, iStat) + 1);
    }
  }
  return true;
}
