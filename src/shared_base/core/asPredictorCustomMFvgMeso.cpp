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

#include "asPredictorCustomMFvgMeso.h"
#include "asIncludes.h"

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorCustomMFvgMeso::asPredictorCustomMFvgMeso(const wxString& dataId)
    : asPredictorCustomMFvgSynop(dataId) {
    // Set the basic properties.
    _datasetId = "Custom_MeteoFVG_Meso";
    _provider = "ECMWF";
    _transformedBy = "Meteo FVG";
    _datasetName = "Integrated Forecasting System (IFS) grib files at Meteo FVG";
    _fStr.hasLevelDim = true;
    _fStr.singleTimeStep = true;
    _warnMissingFiles = false;
}

bool asPredictorCustomMFvgMeso::Init() {
    if (!asPredictorCustomMFvgSynop::Init()) {
        return false;
    }

    if (_product.IsSameAs("data", false)) {
        if (_dataId.Contains("2t_sfc")) {
            _parameter = AirTemperature;
            _gribCode = {0, 128, 167, 1};
            _unit = degK;
        } else if (_dataId.Contains("10u_sfc")) {
            _parameter = Uwind;
            _gribCode = {0, 128, 165, 1};
            _unit = _s;
        } else if (_dataId.Contains("10v_sfc")) {
            _parameter = Vwind;
            _gribCode = {0, 128, 166, 1};
            _unit = _s;
        } else if (_dataId.Contains("cp_sfc")) {
            _parameter = Precipitation;
            _gribCode = {0, 128, 143, 1};
            _unit = m;
        } else if (_dataId.Contains("msl_sfc")) {
            _parameter = Pressure;
            _gribCode = {0, 128, 151, 1};
            _unit = Pa;
        } else if (_dataId.Contains("tp_sfc")) {
            _parameter = AirTemperature;
            _gribCode = {0, 128, 228, 1};
            _unit = degK;
        } else {
            if (_parameter == ParameterUndefined) {
                wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
                return false;
            }
        }

        _fileNamePattern = _dataId + ".%4d%02d%02d%02d.grib";
    }

    return true;
}

void asPredictorCustomMFvgMeso::ListFiles(asTimeArray& timeArray) {
    // Check product directory
    if (!wxDirExists(GetFullDirectoryPath())) {
        throw runtime_error(asStrF(_("Cannot find predictor directory for FVG data (%s)."), GetFullDirectoryPath()));
    }

    // Check directory structure
    Time t0 = asTime::GetTimeStruct(timeArray[0]);
    bool skipMonthDayInPath = false;
    if (!wxDirExists(GetFullDirectoryPath() + asStrF("%4d/%02d/%02d", t0.year, t0.month, t0.day))) {
        if (wxDirExists(GetFullDirectoryPath() + asStrF("%4d", t0.year))) {
            skipMonthDayInPath = true;
        } else {
            throw runtime_error(_("Cannot find coherent predictor directory structure for FVG data."));
        }
    }

    for (int i = 0; i < timeArray.GetSize(); ++i) {
        Time t = asTime::GetTimeStruct(timeArray[i]);
        wxString path;
        if (t.hour > 0) {
            if (!skipMonthDayInPath) {
                path = GetFullDirectoryPath() + asStrF("%4d/%02d/%02d/", t.year, t.month, t.day);
            } else {
                path = GetFullDirectoryPath() + asStrF("%4d/", t.year);
            }
            _files.push_back(path + asStrF(_fileNamePattern, t.year, t.month, t.day, t.hour));
        } else {
            Time t2 = asTime::GetTimeStruct(timeArray[i] - timeArray.GetTimeStepDays());
            if (!skipMonthDayInPath) {
                path = GetFullDirectoryPath() + asStrF("%4d/%02d/%02d/", t2.year, t2.month, t2.day);
            } else {
                path = GetFullDirectoryPath() + asStrF("%4d/", t2.year);
            }
            _files.push_back(path + asStrF(_fileNamePattern, t2.year, t2.month, t2.day, 24));
        }
    }
}
