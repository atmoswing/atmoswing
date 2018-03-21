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
 * Portions Copyright 2018 Pascal Horton, University of Bern.
 */

#include "asPredictorProjCMIP5.h"

#include <asTimeArray.h>
#include <asAreaCompGrid.h>
#include <wx/dir.h>
#include <wx/regex.h>


asPredictorProjCMIP5::asPredictorProjCMIP5(const wxString &dataId, const wxString &model, const wxString &scenario)
        : asPredictorProj(dataId, model, scenario)
{
    // Downloaded from https://esgf-node.llnl.gov/search/cmip5/
    // Set the basic properties.
    m_datasetId = "CMIP5";
    m_provider = "various";
    m_datasetName = "CFSR Subset";
    m_fileType = asFile::Netcdf;
    m_strideAllowed = true;
    m_fStr.dimLatName = "lat";
    m_fStr.dimLonName = "lon";
    m_fStr.dimTimeName = "time";
    m_fStr.dimLevelName = "plev";
    m_subFolder = wxEmptyString;
}

asPredictorProjCMIP5::~asPredictorProjCMIP5()
{

}

bool asPredictorProjCMIP5::Init()
{
    // Identify data ID and set the corresponding properties.
    if (m_dataId.IsSameAs("hgt", false) || m_dataId.IsSameAs("zg", false)) {
        m_parameter = GeopotentialHeight;
        m_parameterName = "Geopotential height";
        m_fileVarName = "zg";
        m_unit = m;
        m_fStr.hasLevelDim = true;
    } else if (m_dataId.IsSameAs("slp", false) || m_dataId.IsSameAs("psl", false)) {
        m_parameter = Pressure;
        m_parameterName = "Sea level pressure";
        m_fileVarName = "psl";
        m_unit = Pa;
        m_fStr.hasLevelDim = false;
    } else if (m_dataId.IsSameAs("rh", false) || m_dataId.IsSameAs("hur", false)) {
        m_parameter = RelativeHumidity;
        m_parameterName = "Relative humidity";
        m_fileVarName = "hur";
        m_unit = percent;
        m_fStr.hasLevelDim = true;
    } else if (m_dataId.IsSameAs("rhs", false)) {
        m_parameter = RelativeHumidity;
        m_parameterName = "Near-Surface Relative Humidity";
        m_fileVarName = "rhs";
        m_unit = percent;
        m_fStr.hasLevelDim = false;
    } else if (m_dataId.IsSameAs("sh", false) || m_dataId.IsSameAs("hus", false)) {
        m_parameter = SpecificHumidity;
        m_parameterName = "Specific humidity";
        m_fileVarName = "hus";
        m_unit = g_kg;
        m_fStr.hasLevelDim = true;
    } else if (m_dataId.IsSameAs("huss", false)) {
        m_parameter = SpecificHumidity;
        m_parameterName = "Near-Surface Specific Humidity";
        m_fileVarName = "huss";
        m_unit = g_kg;
        m_fStr.hasLevelDim = false;
    } else if (m_dataId.IsSameAs("pr", false) || m_dataId.IsSameAs("precip", false)) {
        m_parameter = Precipitation;
        m_parameterName = "Precipitation";
        m_fileVarName = "pr";
        m_unit = kg_m2_s;
        m_fStr.hasLevelDim = false;
    } else if (m_dataId.IsSameAs("prc", false)) {
        m_parameter = Precipitation;
        m_parameterName = "Convective Precipitation";
        m_fileVarName = "prc";
        m_unit = kg_m2_s;
        m_fStr.hasLevelDim = false;
    } else if (m_dataId.IsSameAs("temp", false) || m_dataId.IsSameAs("ta", false)) {
        m_parameter = AirTemperature;
        m_parameterName = "Air Temperature";
        m_fileVarName = "ta";
        m_unit = degK;
        m_fStr.hasLevelDim = true;
    } else if (m_dataId.IsSameAs("tas", false)) {
        m_parameter = AirTemperature;
        m_parameterName = "Near-Surface Air Temperature";
        m_fileVarName = "tas";
        m_unit = degK;
        m_fStr.hasLevelDim = false;
    } else if (m_dataId.IsSameAs("tasmax", false)) {
        m_parameter = AirTemperature;
        m_parameterName = "Daily Maximum Near-Surface Air Temperature";
        m_fileVarName = "tasmax";
        m_unit = degK;
        m_fStr.hasLevelDim = false;
    } else if (m_dataId.IsSameAs("tasmin", false)) {
        m_parameter = AirTemperature;
        m_parameterName = "Daily Minimum Near-Surface Air Temperature";
        m_fileVarName = "tasmin";
        m_unit = degK;
        m_fStr.hasLevelDim = false;
    } else if (m_dataId.IsSameAs("omega", false) || m_dataId.IsSameAs("wap", false)) {
        m_parameter = VerticalVelocity;
        m_parameterName = "Vertical Velocity";
        m_fileVarName = "wap";
        m_unit = Pa_s;
        m_fStr.hasLevelDim = true;
    } else {
        asThrowException(wxString::Format(_("Parameter '%s' not implemented yet."), m_dataId));
    }
    m_fileNamePattern = m_fileVarName + "*" + m_model + "*" + m_scenario + "*.nc";

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

void asPredictorProjCMIP5::ListFiles(asTimeArray &timeArray)
{
    wxArrayString listFiles;
    size_t nbFiles = wxDir::GetAllFiles(GetFullDirectoryPath(), &listFiles, m_fileNamePattern);

    if (nbFiles == 0) {
        asThrowException(wxString::Format(_("No CMIP5 file found for this pattern : %s."), m_fileNamePattern));
    }

    // Sort the list of files
    listFiles.Sort();

    // Check if file is in time range
    double firstYear = timeArray.GetStartingYear();
    double lastYear = timeArray.GetEndingYear();

    for (int i = 0; i < listFiles.Count(); ++i) {

        wxRegEx reDates("\\d{8}-\\d{8}");
        if (!reDates.Matches(listFiles.Item(i))) {
            asThrowException(wxString::Format(_("The dates sequence was not found in the CMIP5 file : %s."), listFiles.Item(i)));

        }

        wxString datesSrt = reDates.GetMatch(listFiles.Item(i));
        double fileStartYear = 0;
        double fileEndYear = 0;
        datesSrt.Mid(0, 4).ToDouble(&fileStartYear);
        datesSrt.Mid(9, 4).ToDouble(&fileEndYear);

        if (fileEndYear < firstYear || fileStartYear > lastYear) {
            continue;
        }

        m_files.push_back(listFiles.Item(i));
    }
}

double asPredictorProjCMIP5::ConvertToMjd(double timeValue, double refValue) const
{
    wxASSERT(refValue > 30000);
    wxASSERT(refValue < 70000);

    return refValue + timeValue;
}
