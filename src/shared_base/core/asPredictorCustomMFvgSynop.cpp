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

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorCustomMFvgSynop::asPredictorCustomMFvgSynop(const wxString& dataId)
    : asPredictorEcmwfIfsGrib(dataId) {
    // Set the basic properties.
    m_datasetId = "Custom_MeteoFVG_Synop";
    m_provider = "ECMWF";
    m_transformedBy = "Meteo FVG";
    m_datasetName = "Integrated Forecasting System (IFS) grib files at Meteo FVG";
    m_fStr.hasLevelDim = true;
    m_fStr.singleTimeStep = true;
    m_warnMissingFiles = false;
}

bool asPredictorCustomMFvgSynop::Init() {
    if (m_product.IsEmpty()) {
        m_product = "data";
    }

    if (m_product.IsSameAs("data", false)) {
        if (m_dataId.Contains("gh")) {
            m_parameter = GeopotentialHeight;
            m_gribCode = {0, 128, 156, 100};
            m_unit = m;
        } else if (m_dataId.Contains("t")) {
            m_parameter = AirTemperature;
            m_gribCode = {0, 128, 130, 100};
            m_unit = degK;
        } else if (m_dataId.Contains("w")) {
            m_parameter = VerticalVelocity;
            m_gribCode = {0, 128, 135, 100};
            m_unit = Pa_s;
        } else if (m_dataId.Contains("r")) {
            m_parameter = RelativeHumidity;
            m_gribCode = {0, 128, 157, 100};
            m_unit = percent;
        } else if (m_dataId.Contains("u")) {
            m_parameter = Uwind;
            m_gribCode = {0, 128, 131, 100};
            m_unit = m_s;
        } else if (m_dataId.Contains("v")) {
            m_parameter = Vwind;
            m_gribCode = {0, 128, 132, 100};
            m_unit = m_s;
        } else {
            if (m_datasetId.IsSameAs("Custom_MeteoFVG_meso", false) ||
                m_datasetId.IsSameAs("Custom_MeteoFVG_meso_packed", false)) {
                return true;
            }
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), m_dataId, m_product);
            return false;
        }

        m_fileNamePattern = m_dataId + ".%4d%02d%02d%02d.grib";

    } else if (m_product.IsSameAs("datader", false)) {
        if (m_dataId.Contains("thetaES")) {
            m_parameter = PotentialTemperature;
            m_gribCode = {0, 3, 114, 100};
            m_unit = W_m2;
        } else if (m_dataId.Contains("thetaE")) {
            m_parameter = PotentialTemperature;
            m_gribCode = {0, 3, 113, 100};
            m_unit = W_m2;
        } else if (m_dataId.Contains("vflux")) {
            m_parameter = MomentumFlux;
            m_gribCode = {0, 3, 125, 100};
            m_unit = kg_m2_s;
        } else if (m_dataId.Contains("uflux")) {
            m_parameter = MomentumFlux;
            m_gribCode = {0, 3, 124, 100};
            m_unit = kg_m2_s;
        } else if (m_dataId.Contains("q")) {
            m_parameter = SpecificHumidity;
            m_gribCode = {0, 128, 133, 100};
            m_unit = percent;
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), m_dataId, m_product);
            return false;
        }

        m_fileNamePattern = m_dataId + ".%4d%02d%02d%02d.grib";

    } else if (m_product.IsSameAs("vertdiff", false)) {
        m_parameter = Other;

        if (m_dataId.IsSameAs("DP500925", false)) {
            m_gribCode = {0, 3, 113, 100};
        } else if (m_dataId.IsSameAs("LRT700500", false)) {
            m_gribCode = {0, 128, 130, 100};
        } else if (m_dataId.IsSameAs("LRT850500", false)) {
            m_gribCode = {0, 128, 130, 100};
        } else if (m_dataId.IsSameAs("LRTE700500", false)) {
            m_gribCode = {0, 3, 113, 100};
        } else if (m_dataId.IsSameAs("LRTE850500", false)) {
            m_gribCode = {0, 3, 113, 100};
        } else if (m_dataId.IsSameAs("MB500850", false)) {
            m_parameter = MaximumBuoyancy;
            m_gribCode = {0, 3, 114, 100};
        } else if (m_dataId.IsSameAs("MB500925", false)) {
            m_parameter = MaximumBuoyancy;
            m_gribCode = {0, 3, 114, 100};
        } else if (m_dataId.IsSameAs("MB700925", false)) {
            m_parameter = MaximumBuoyancy;
            m_gribCode = {0, 3, 114, 100};
        } else if (m_dataId.IsSameAs("MB850500", false)) {
            m_parameter = MaximumBuoyancy;
            m_gribCode = {0, 3, 114, 100};
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), m_dataId, m_product);
            return false;
        }

        m_fileNamePattern = m_dataId + ".%4d%02d%02d%02d.grib";
    }

    return true;
}

void asPredictorCustomMFvgSynop::ListFiles(asTimeArray& timeArray) {
    // Check product directory
    if (!wxDirExists(GetFullDirectoryPath())) {
        asThrowException(asStrF(_("Cannot find predictor directory for FVG data (%s)."), GetFullDirectoryPath()));
    }

    // Check directory structure
    Time t0 = asTime::GetTimeStruct(timeArray[0]);
    bool skipMonthDayInPath = false;
    if (!wxDirExists(GetFullDirectoryPath() + asStrF("%4d/%02d/%02d", t0.year, t0.month, t0.day))) {
        if (wxDirExists(GetFullDirectoryPath() + asStrF("%4d", t0.year))) {
            skipMonthDayInPath = true;
        } else {
            asThrowException(_("Cannot find coherent predictor directory structure for FVG data."));
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
            m_files.push_back(path + asStrF(m_fileNamePattern, t.year, t.month, t.day, t.hour));
        } else {
            Time t2 = asTime::GetTimeStruct(timeArray[i] - timeArray.GetTimeStepDays());
            if (!skipMonthDayInPath) {
                path = GetFullDirectoryPath() + asStrF("%4d/%02d/%02d/", t2.year, t2.month, t2.day);
            } else {
                path = GetFullDirectoryPath() + asStrF("%4d/", t2.year);
            }
            m_files.push_back(path + asStrF(m_fileNamePattern, t2.year, t2.month, t2.day, 24));
        }
    }
}
