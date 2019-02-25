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

#include "asPredictorCustomMeteoFvgIfs.h"

#include <asTimeArray.h>
#include <asAreaCompGrid.h>


asPredictorCustomMeteoFvgIfs::asPredictorCustomMeteoFvgIfs(const wxString &dataId)
        : asPredictorEcmwfIfsGrib(dataId)
{
    // Set the basic properties.
    m_datasetId = "Custom_MeteoFVG_ECMWF_IFS_GRIB";
    m_provider = "ECMWF";
    m_transformedBy = "Meteo FVG";
    m_datasetName = "Integrated Forecasting System (IFS) grib files at Meteo FVG";
}

bool asPredictorCustomMeteoFvgIfs::Init()
{
    if (m_product.IsEmpty()) {
        m_product = "data";
    }

    if (m_product.IsSameAs("data", false)) {

        if (m_dataId.IsSameAs("gh_500", false) ||
            m_dataId.IsSameAs("gh_700", false) ||
            m_dataId.IsSameAs("gh_850", false) ||
            m_dataId.IsSameAs("gh_925", false)) {
            m_parameter = GeopotentialHeight;
            m_gribCode = {0, 128, 156, 100};
            m_unit = m;
            m_fStr.hasLevelDim = true;
        } else if (m_dataId.IsSameAs("t_500", false) ||
                   m_dataId.IsSameAs("t_700", false) ||
                   m_dataId.IsSameAs("t_850", false) ||
                   m_dataId.IsSameAs("t_925", false)) {
            m_parameter = AirTemperature;
            m_gribCode = {0, 128, 130, 100};
            m_unit = degK;
            m_fStr.hasLevelDim = true;
        } else if (m_dataId.IsSameAs("w_500", false) ||
                   m_dataId.IsSameAs("w_700", false) ||
                   m_dataId.IsSameAs("w_850", false) ||
                   m_dataId.IsSameAs("w_925", false)) {
            m_parameter = VerticalVelocity;
            m_gribCode = {0, 128, 135, 100};
            m_unit = Pa_s;
            m_fStr.hasLevelDim = true;
        } else if (m_dataId.IsSameAs("r_500", false) ||
                   m_dataId.IsSameAs("r_700", false) ||
                   m_dataId.IsSameAs("r_850", false) ||
                   m_dataId.IsSameAs("r_925", false)) {
            m_parameter = RelativeHumidity;
            m_gribCode = {0, 128, 157, 100};
            m_unit = percent;
            m_fStr.hasLevelDim = true;
        } else if (m_dataId.IsSameAs("u_500", false) ||
                   m_dataId.IsSameAs("u_700", false) ||
                   m_dataId.IsSameAs("u_850", false) ||
                   m_dataId.IsSameAs("u_925", false)) {
            m_parameter = Uwind;
            m_gribCode = {0, 128, 131, 100};
            m_unit = m_s;
            m_fStr.hasLevelDim = true;
        } else if (m_dataId.IsSameAs("v_500", false) ||
                   m_dataId.IsSameAs("v_700", false) ||
                   m_dataId.IsSameAs("v_850", false) ||
                   m_dataId.IsSameAs("v_925", false)) {
            m_parameter = Vwind;
            m_gribCode = {0, 128, 132, 100};
            m_unit = m_s;
            m_fStr.hasLevelDim = true;
        } else {
            asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                              m_dataId, m_product));
        }

        m_fileNamePattern = m_dataId + ".%4d%02d%02d%02d.grib";

    } else if (m_product.IsSameAs("datader", false)) {

        if (m_dataId.IsSameAs("q_500", false) ||
            m_dataId.IsSameAs("q_700", false) ||
            m_dataId.IsSameAs("q_850", false) ||
            m_dataId.IsSameAs("q_925", false)) {
            m_parameter = SpecificHumidity;
            m_gribCode = {0, 128, 133, 100};
            m_unit = percent;
            m_fStr.hasLevelDim = true;
        } else if (m_dataId.IsSameAs("thetaE_500", false) ||
                   m_dataId.IsSameAs("thetaE_700", false) ||
                   m_dataId.IsSameAs("thetaE_850", false) ||
                   m_dataId.IsSameAs("thetaE_925", false)) {
            m_parameter = Radiation;
            m_gribCode = {0, 3, 113, 100};
            m_unit = W_m2;
            m_fStr.hasLevelDim = true;
        } else if (m_dataId.IsSameAs("thetaES_500", false) ||
                   m_dataId.IsSameAs("thetaES_700", false) ||
                   m_dataId.IsSameAs("thetaES_850", false) ||
                   m_dataId.IsSameAs("thetaES_925", false)) {
            m_parameter = Radiation;
            m_gribCode = {0, 3, 114, 100};
            m_unit = W_m2;
            m_fStr.hasLevelDim = true;
        } else {
            asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                              m_dataId, m_product));
        }

        m_fileNamePattern = m_dataId + ".%4d%02d%02d%02d.grib";

    } else if (m_product.IsSameAs("vertdiff", false)) {

        m_parameter = Other;
        m_fStr.hasLevelDim = true;

        if (m_dataId.IsSameAs("DP500925", false)) {
            m_gribCode = {0, 3, 113, 100};
        } else if (m_dataId.IsSameAs("LRT700500", false)) {
            m_gribCode = {0, 128, 130, 100};
        } else if (m_dataId.IsSameAs("LRT850500", false)) {
            m_gribCode = {0, 128, 130, 100};
        } else if (m_dataId.IsSameAs("LRTE700500", false)) {
            m_gribCode = {0, 3, 113, 100};
        } else if (m_dataId.IsSameAs("LRTE85050", false)) {
            m_gribCode = {0, 3, 113, 100};
        } else if (m_dataId.IsSameAs("MB500850", false)) {
            m_gribCode = {0, 3, 114, 100};
        } else if (m_dataId.IsSameAs("MB500925", false)) {
            m_gribCode = {0, 3, 114, 100};
        } else if (m_dataId.IsSameAs("MB700925", false)) {
            m_gribCode = {0, 3, 114, 100};
        } else {
            asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                              m_dataId, m_product));
        }

        m_fileNamePattern = m_dataId + ".%4d%02d%02d%02d.grib";
    }

    return true;
}

void asPredictorCustomMeteoFvgIfs::ListFiles(asTimeArray &timeArray)
{
    for (int i = 0; i < timeArray.GetSize(); ++i) {
        Time t = asTime::GetTimeStruct(timeArray[i]);
        if (t.hour > 0) {
            m_files.push_back(GetFullDirectoryPath() + wxString::Format("%4d/%02d/%02d/", t.year, t.month, t.day) +
                              wxString::Format(m_fileNamePattern, t.year, t.month, t.day, t.hour));
        } else if (i > 0) {
            Time t2 = asTime::GetTimeStruct(timeArray[i-1]);
            m_files.push_back(GetFullDirectoryPath() + wxString::Format("%4d/%02d/%02d/", t2.year, t2.month, t2.day) +
                              wxString::Format(m_fileNamePattern, t2.year, t2.month, t2.day, 24));
        }
    }
}
