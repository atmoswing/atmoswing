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
#include "asIncludes.h"
#include "asTimeArray.h"

asPredictorProjCordex::asPredictorProjCordex(const wxString& dataId, const wxString& model, const wxString& scenario)
    : asPredictorProj(dataId, model, scenario) {
    // Downloaded from https://esgf-index1.ceda.ac.uk/search/cordex-ceda/
    // Set the basic properties.
    _datasetId = "CORDEX";
    _provider = "various";
    _datasetName = "CORDEX";
    _fileType = asFile::Netcdf;
    _strideAllowed = true;
    _parseTimeReference = true;
    _fStr.dimLatName = "rlat";  // y will be considered otherwise
    _fStr.dimLonName = "rlon";  // x will be considered otherwise
    _fStr.dimTimeName = "time";
    _fStr.dimLevelName = "plev";
    _fStr.hasLevelDim = false;
    _isLatLon = false;
}

asPredictorProjCordex::~asPredictorProjCordex() {}

bool asPredictorProjCordex::Init() {
    // Identify data ID and set the corresponding properties.
    if (_dataId.IsSameAs("zg200", false)) {
        _parameter = GeopotentialHeight;
        _parameterName = "Geopotential height";
        _fileVarName = "zg200";
        _unit = m;
    } else if (_dataId.IsSameAs("zg500", false)) {
        _parameter = GeopotentialHeight;
        _parameterName = "Geopotential height";
        _fileVarName = "zg500";
        _unit = m;
    } else if (_dataId.IsSameAs("zg850", false)) {
        _parameter = GeopotentialHeight;
        _parameterName = "Geopotential height";
        _fileVarName = "zg850";
        _unit = m;
    } else if (_dataId.IsSameAs("ua200", false)) {
        _parameter = Uwind;
        _parameterName = "Eastward Wind";
        _fileVarName = "ua200";
        _unit = _s;
    } else if (_dataId.IsSameAs("ua500", false)) {
        _parameter = Uwind;
        _parameterName = "Eastward Wind";
        _fileVarName = "ua500";
        _unit = _s;
    } else if (_dataId.IsSameAs("ua850", false)) {
        _parameter = Uwind;
        _parameterName = "Eastward Wind";
        _fileVarName = "ua850";
        _unit = _s;
    } else if (_dataId.IsSameAs("va200", false)) {
        _parameter = Vwind;
        _parameterName = "Eastward Wind";
        _fileVarName = "va200";
        _unit = _s;
    } else if (_dataId.IsSameAs("va500", false)) {
        _parameter = Vwind;
        _parameterName = "Eastward Wind";
        _fileVarName = "va500";
        _unit = _s;
    } else if (_dataId.IsSameAs("va850", false)) {
        _parameter = Vwind;
        _parameterName = "Eastward Wind";
        _fileVarName = "va850";
        _unit = _s;
    } else if (IsSeaLevelPressure()) {
        _parameter = Pressure;
        _parameterName = "Sea level pressure";
        _fileVarName = "psl";
        _unit = Pa;
    } else if (_dataId.IsSameAs("hurs", false)) {
        _parameter = RelativeHumidity;
        _parameterName = "Near-Surface Relative Humidity";
        _fileVarName = "hurs";
        _unit = percent;
    } else if (_dataId.IsSameAs("hus850", false)) {
        _parameter = SpecificHumidity;
        _parameterName = "Specific humidity";
        _fileVarName = "hus850";
        _unit = g_kg;
    } else if (_dataId.IsSameAs("huss", false)) {
        _parameter = SpecificHumidity;
        _parameterName = "Near-Surface Specific Humidity";
        _fileVarName = "huss";
        _unit = g_kg;
    } else if (_dataId.IsSameAs("pr", false) || _dataId.IsSameAs("precip", false)) {
        _parameter = Precipitation;
        _parameterName = "Precipitation";
        _fileVarName = "pr";
        _unit = kg_m2_s;
    } else if (_dataId.IsSameAs("prc", false)) {
        _parameter = Precipitation;
        _parameterName = "Convective Precipitation";
        _fileVarName = "prc";
        _unit = kg_m2_s;
    } else if (_dataId.IsSameAs("ta200", false)) {
        _parameter = AirTemperature;
        _parameterName = "Air Temperature";
        _fileVarName = "ta200";
        _unit = degK;
    } else if (_dataId.IsSameAs("ta500", false)) {
        _parameter = AirTemperature;
        _parameterName = "Air Temperature";
        _fileVarName = "ta500";
        _unit = degK;
    } else if (_dataId.IsSameAs("ta850", false)) {
        _parameter = AirTemperature;
        _parameterName = "Air Temperature";
        _fileVarName = "ta850";
        _unit = degK;
    } else if (_dataId.IsSameAs("tas", false)) {
        _parameter = AirTemperature;
        _parameterName = "Near-Surface Air Temperature";
        _fileVarName = "tas";
        _unit = degK;
    } else if (_dataId.IsSameAs("tasmax", false)) {
        _parameter = AirTemperature;
        _parameterName = "Daily Maximum Near-Surface Air Temperature";
        _fileVarName = "tasmax";
        _unit = degK;
    } else if (_dataId.IsSameAs("tasmin", false)) {
        _parameter = AirTemperature;
        _parameterName = "Daily Minimum Near-Surface Air Temperature";
        _fileVarName = "tasmin";
        _unit = degK;
    } else {
        _parameter = ParameterUndefined;
        _parameterName = "Undefined";
        _fileVarName = _dataId;
        _unit = UnitUndefined;
    }
    _fileNamePattern = _fileVarName + "*" + _model + "*" + _scenario + "*.nc";

    // Check directory is set
    if (GetDirectoryPath().IsEmpty()) {
        wxLogError(_("The path to the directory has not been set for the data %s from the dataset %s."), _dataId,
                   _datasetName);
        return false;
    }

    // Set to initialized
    _initialized = true;

    return true;
}

void asPredictorProjCordex::ListFiles(asTimeArray& timeArray) {
    wxArrayString listFiles;
    size_t nbFiles = wxDir::GetAllFiles(GetFullDirectoryPath(), &listFiles, _fileNamePattern);

    if (nbFiles == 0) {
        throw std::runtime_error(asStrF(_("No CORDEX file found for this pattern : %s."), _fileNamePattern));
    }

    // Sort the list of files
    listFiles.Sort();

    // Check if file is in time range
    double firstYear = timeArray.GetStartingYear();
    double lastYear = timeArray.GetEndingYear();

    for (int i = 0; i < listFiles.Count(); ++i) {
        wxRegEx reDates("\\d{8,}-\\d{8,}", wxRE_ADVANCED);
        if (!reDates.Matches(listFiles.Item(i))) {
            throw std::runtime_error(
                asStrF(_("The dates sequence was not found in the CORDEX file name : %s."), listFiles.Item(i)));
        }

        wxString datesSrt = reDates.GetMatch(listFiles.Item(i));
        double fileStartYear = 0;
        double fileEndYear = 0;
        datesSrt.Mid(0, 4).ToDouble(&fileStartYear);
        datesSrt.After('-').Mid(0, 4).ToDouble(&fileEndYear);

        if (fileEndYear < firstYear || fileStartYear > lastYear) {
            continue;
        }

        _files.push_back(listFiles.Item(i));
    }
}

void asPredictorProjCordex::ConvertToMjd(a1d& time, double refValue) const {
    time += refValue;
}
