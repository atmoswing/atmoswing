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

#include "asPredictorCustomMeteoFvgIfsPacked.h"

#include <asTimeArray.h>
#include <asAreaCompGrid.h>
#include <wx/dir.h>
#include <wx/regex.h>


asPredictorCustomMeteoFvgIfsPacked::asPredictorCustomMeteoFvgIfsPacked(const wxString &dataId)
        : asPredictorCustomMeteoFvgIfs(dataId)
{
    // Set the basic properties.
    m_datasetId = "Custom_MeteoFVG_ECMWF_IFS_GRIB_Packed";
    m_provider = "ECMWF";
    m_transformedBy = "Meteo FVG";
    m_datasetName = "Integrated Forecasting System (IFS) grib files at Meteo FVG";
    m_fStr.hasLevelDim = true;
    m_fStr.singleTimeStep = false;
    m_warnMissingFiles = true;
}

void asPredictorCustomMeteoFvgIfsPacked::ListFiles(asTimeArray &timeArray)
{
    // Case 1: single file with the variable name
    wxString filePath = GetFullDirectoryPath() + m_fileVarName + ".grib";

    if (wxFileExists(filePath)) {
        m_files.push_back(filePath);
        return;
    }

    // Case 2: yearly files
    wxArrayString listFiles;
    size_t nbFiles = wxDir::GetAllFiles(GetFullDirectoryPath(), &listFiles, m_dataId + "_*.grib");

    if (nbFiles == 0) {
        nbFiles = wxDir::GetAllFiles(GetFullDirectoryPath(), &listFiles, m_dataId + ".*.grib");
        if (nbFiles == 0) {
            asThrowException(_("No file found for the FVG packed archive."));
        }
    }

    listFiles.Sort();

    double firstYear = timeArray.GetStartingYear();
    double lastYear = timeArray.GetEndingYear();

    for (size_t i = 0; i < listFiles.Count(); ++i) {
        wxRegEx reDates("\\d{4,}.grib", wxRE_ADVANCED);
        if (!reDates.Matches(listFiles.Item(i))) {
            continue;
        }

        wxString datesSrt = reDates.GetMatch(listFiles.Item(i));
        datesSrt = datesSrt.Left(datesSrt.Length() - 5);
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
}
