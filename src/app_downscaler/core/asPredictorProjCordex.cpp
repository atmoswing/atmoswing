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

#include "asPredictorProjCordex.h"

#include <wx/dir.h>
#include <wx/regex.h>

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorProjCordex::asPredictorProjCordex(const wxString& dataId, const wxString& model, const wxString& scenario)
    : asPredictorProj(dataId, model, scenario) {
    // Downloaded from https://esgf-index1.ceda.ac.uk/search/cordex-ceda/
    // Set the basic properties.
    m_datasetId = "CORDEX";
    m_provider = "various";
    m_datasetName = "CORDEX";
    m_fileType = asFile::Netcdf;
    m_strideAllowed = true;
    m_parseTimeReference = true;
    m_fStr.dimLatName = "rlat";  // y will be considered otherwise
    m_fStr.dimLonName = "rlon";  // x will be considered otherwise
    m_fStr.dimTimeName = "time";
    m_fStr.dimLevelName = "plev";
    m_fStr.hasLevelDim = false;
    m_isLatLon = false;
}

asPredictorProjCordex::~asPredictorProjCordex() {}

bool asPredictorProjCordex::Init() {
    // Identify data ID and set the corresponding properties.
    if (m_dataId.IsSameAs("zg200", false)) {
        m_parameter = GeopotentialHeight;
        m_parameterName = "Geopotential height";
        m_fileVarName = "zg200";
        m_unit = m;
    } else if (m_dataId.IsSameAs("zg500", false)) {
        m_parameter = GeopotentialHeight;
        m_parameterName = "Geopotential height";
        m_fileVarName = "zg500";
        m_unit = m;
    } else if (m_dataId.IsSameAs("zg850", false)) {
        m_parameter = GeopotentialHeight;
        m_parameterName = "Geopotential height";
        m_fileVarName = "zg850";
        m_unit = m;
    } else if (m_dataId.IsSameAs("ua200", false)) {
        m_parameter = Uwind;
        m_parameterName = "Eastward Wind";
        m_fileVarName = "ua200";
        m_unit = m_s;
    } else if (m_dataId.IsSameAs("ua500", false)) {
        m_parameter = Uwind;
        m_parameterName = "Eastward Wind";
        m_fileVarName = "ua500";
        m_unit = m_s;
    } else if (m_dataId.IsSameAs("ua850", false)) {
        m_parameter = Uwind;
        m_parameterName = "Eastward Wind";
        m_fileVarName = "ua850";
        m_unit = m_s;
    } else if (m_dataId.IsSameAs("va200", false)) {
        m_parameter = Vwind;
        m_parameterName = "Eastward Wind";
        m_fileVarName = "va200";
        m_unit = m_s;
    } else if (m_dataId.IsSameAs("va500", false)) {
        m_parameter = Vwind;
        m_parameterName = "Eastward Wind";
        m_fileVarName = "va500";
        m_unit = m_s;
    } else if (m_dataId.IsSameAs("va850", false)) {
        m_parameter = Vwind;
        m_parameterName = "Eastward Wind";
        m_fileVarName = "va850";
        m_unit = m_s;
    } else if (IsSeaLevelPressure()) {
        m_parameter = Pressure;
        m_parameterName = "Sea level pressure";
        m_fileVarName = "psl";
        m_unit = Pa;
    } else if (m_dataId.IsSameAs("hurs", false)) {
        m_parameter = RelativeHumidity;
        m_parameterName = "Near-Surface Relative Humidity";
        m_fileVarName = "hurs";
        m_unit = percent;
    } else if (m_dataId.IsSameAs("hus850", false)) {
        m_parameter = SpecificHumidity;
        m_parameterName = "Specific humidity";
        m_fileVarName = "hus850";
        m_unit = g_kg;
    } else if (m_dataId.IsSameAs("huss", false)) {
        m_parameter = SpecificHumidity;
        m_parameterName = "Near-Surface Specific Humidity";
        m_fileVarName = "huss";
        m_unit = g_kg;
    } else if (m_dataId.IsSameAs("pr", false) || m_dataId.IsSameAs("precip", false)) {
        m_parameter = Precipitation;
        m_parameterName = "Precipitation";
        m_fileVarName = "pr";
        m_unit = kg_m2_s;
    } else if (m_dataId.IsSameAs("prc", false)) {
        m_parameter = Precipitation;
        m_parameterName = "Convective Precipitation";
        m_fileVarName = "prc";
        m_unit = kg_m2_s;
    } else if (m_dataId.IsSameAs("ta200", false)) {
        m_parameter = AirTemperature;
        m_parameterName = "Air Temperature";
        m_fileVarName = "ta200";
        m_unit = degK;
    } else if (m_dataId.IsSameAs("ta500", false)) {
        m_parameter = AirTemperature;
        m_parameterName = "Air Temperature";
        m_fileVarName = "ta500";
        m_unit = degK;
    } else if (m_dataId.IsSameAs("ta850", false)) {
        m_parameter = AirTemperature;
        m_parameterName = "Air Temperature";
        m_fileVarName = "ta850";
        m_unit = degK;
    } else if (m_dataId.IsSameAs("tas", false)) {
        m_parameter = AirTemperature;
        m_parameterName = "Near-Surface Air Temperature";
        m_fileVarName = "tas";
        m_unit = degK;
    } else if (m_dataId.IsSameAs("tasmax", false)) {
        m_parameter = AirTemperature;
        m_parameterName = "Daily Maximum Near-Surface Air Temperature";
        m_fileVarName = "tasmax";
        m_unit = degK;
    } else if (m_dataId.IsSameAs("tasmin", false)) {
        m_parameter = AirTemperature;
        m_parameterName = "Daily Minimum Near-Surface Air Temperature";
        m_fileVarName = "tasmin";
        m_unit = degK;
    } else {
        m_parameter = ParameterUndefined;
        m_parameterName = "Undefined";
        m_fileVarName = m_dataId;
        m_unit = UnitUndefined;
    }
    m_fileNamePattern = m_fileVarName + "*" + m_model + "*" + m_scenario + "*.nc";

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

void asPredictorProjCordex::ListFiles(asTimeArray& timeArray) {
    wxArrayString listFiles;
    size_t nbFiles = wxDir::GetAllFiles(GetFullDirectoryPath(), &listFiles, m_fileNamePattern);

    if (nbFiles == 0) {
        throw runtime_error(asStrF(_("No CORDEX file found for this pattern : %s."), m_fileNamePattern));
    }

    // Sort the list of files
    listFiles.Sort();

    // Check if file is in time range
    double firstYear = timeArray.GetStartingYear();
    double lastYear = timeArray.GetEndingYear();

    for (int i = 0; i < listFiles.Count(); ++i) {
        wxRegEx reDates("\\d{8,}-\\d{8,}", wxRE_ADVANCED);
        if (!reDates.Matches(listFiles.Item(i))) {
            throw runtime_error(asStrF(_("The dates sequence was not found in the CORDEX file name : %s."), listFiles.Item(i)));
        }

        wxString datesSrt = reDates.GetMatch(listFiles.Item(i));
        double fileStartYear = 0;
        double fileEndYear = 0;
        datesSrt.Mid(0, 4).ToDouble(&fileStartYear);
        datesSrt.After('-').Mid(0, 4).ToDouble(&fileEndYear);

        if (fileEndYear < firstYear || fileStartYear > lastYear) {
            continue;
        }

        m_files.push_back(listFiles.Item(i));
    }
}

void asPredictorProjCordex::ConvertToMjd(a1d& time, double refValue) const {
    time += refValue;
}
