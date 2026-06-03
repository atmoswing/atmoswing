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

#include "asPredictorCustomMFvgSynop.h"
#include "asIncludes.h"

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorCustomMFvgSynop::asPredictorCustomMFvgSynop(const wxString& dataId)
    : asPredictorEcmwfIfs(dataId) {
    // Set the basic properties.
    _datasetId = "Custom_MeteoFVG_Synop";
    _provider = "ECMWF";
    _transformedBy = "Meteo FVG";
    _datasetName = "Integrated Forecasting System (IFS) grib files at Meteo FVG";
    _fStr.hasLevelDim = true;
    _fStr.singleTimeStep = true;
    _warnMissingFiles = false;
}

bool asPredictorCustomMFvgSynop::Init() {
    if (_product.IsEmpty()) {
        _product = "data";
    }

    if (_product.IsSameAs("data", false)) {
        if (_dataId.Contains("gh")) {
            _parameter = GeopotentialHeight;
            _gribCode = {0, 128, 156, 100};
            _unit = m;
        } else if (_dataId.Contains("t")) {
            _parameter = AirTemperature;
            _gribCode = {0, 128, 130, 100};
            _unit = degK;
        } else if (_dataId.Contains("w")) {
            _parameter = VerticalVelocity;
            _gribCode = {0, 128, 135, 100};
            _unit = Pa_s;
        } else if (_dataId.Contains("r")) {
            _parameter = RelativeHumidity;
            _gribCode = {0, 128, 157, 100};
            _unit = percent;
        } else if (_dataId.Contains("u")) {
            _parameter = Uwind;
            _gribCode = {0, 128, 131, 100};
            _unit = _s;
        } else if (_dataId.Contains("v")) {
            _parameter = Vwind;
            _gribCode = {0, 128, 132, 100};
            _unit = _s;
        } else {
            if (_datasetId.IsSameAs("Custom_MeteoFVG_meso", false) ||
                _datasetId.IsSameAs("Custom_MeteoFVG_meso_packed", false)) {
                return true;
            }
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
            return false;
        }

        _fileNamePattern = _dataId + ".%4d%02d%02d%02d.grib";

    } else if (_product.IsSameAs("datader", false)) {
        if (_dataId.Contains("thetaES")) {
            _parameter = PotentialTemperature;
            _gribCode = {0, 3, 114, 100};
            _unit = W_m2;
        } else if (_dataId.Contains("thetaE")) {
            _parameter = PotentialTemperature;
            _gribCode = {0, 3, 113, 100};
            _unit = W_m2;
        } else if (_dataId.Contains("vflux")) {
            _parameter = MomentumFlux;
            _gribCode = {0, 3, 125, 100};
            _unit = kg_m2_s;
        } else if (_dataId.Contains("uflux")) {
            _parameter = MomentumFlux;
            _gribCode = {0, 3, 124, 100};
            _unit = kg_m2_s;
        } else if (_dataId.Contains("q")) {
            _parameter = SpecificHumidity;
            _gribCode = {0, 128, 133, 100};
            _unit = percent;
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
            return false;
        }

        _fileNamePattern = _dataId + ".%4d%02d%02d%02d.grib";

    } else if (_product.IsSameAs("vertdiff", false)) {
        _parameter = Other;

        if (_dataId.IsSameAs("DP500925", false)) {
            _gribCode = {0, 3, 113, 100};
        } else if (_dataId.IsSameAs("LRT700500", false)) {
            _gribCode = {0, 128, 130, 100};
        } else if (_dataId.IsSameAs("LRT850500", false)) {
            _gribCode = {0, 128, 130, 100};
        } else if (_dataId.IsSameAs("LRTE700500", false)) {
            _gribCode = {0, 3, 113, 100};
        } else if (_dataId.IsSameAs("LRTE850500", false)) {
            _gribCode = {0, 3, 113, 100};
        } else if (_dataId.IsSameAs("MB500850", false)) {
            _parameter = MaximumBuoyancy;
            _gribCode = {0, 3, 114, 100};
        } else if (_dataId.IsSameAs("MB500925", false)) {
            _parameter = MaximumBuoyancy;
            _gribCode = {0, 3, 114, 100};
        } else if (_dataId.IsSameAs("MB700925", false)) {
            _parameter = MaximumBuoyancy;
            _gribCode = {0, 3, 114, 100};
        } else if (_dataId.IsSameAs("MB850500", false)) {
            _parameter = MaximumBuoyancy;
            _gribCode = {0, 3, 114, 100};
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
            return false;
        }

        _fileNamePattern = _dataId + ".%4d%02d%02d%02d.grib";
    }

    return true;
}

void asPredictorCustomMFvgSynop::ListFiles(asTimeArray& timeArray) {
    // Check product directory
    if (!wxDirExists(GetFullDirectoryPath())) {
        throw std::runtime_error(asStrF(_("Cannot find predictor directory for FVG data (%s)."), GetFullDirectoryPath()));
    }

    // Check directory structure
    Time t0 = asTime::GetTimeStruct(timeArray[0]);
    bool skipMonthDayInPath = false;
    if (!wxDirExists(GetFullDirectoryPath() + asStrF("%4d/%02d/%02d", t0.year, t0.month, t0.day))) {
        if (wxDirExists(GetFullDirectoryPath() + asStrF("%4d", t0.year))) {
            skipMonthDayInPath = true;
        } else {
            throw std::runtime_error(_("Cannot find coherent predictor directory structure for FVG data."));
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
