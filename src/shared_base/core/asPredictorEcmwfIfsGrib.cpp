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

#include "asPredictorEcmwfIfsGrib.h"

#include <asTimeArray.h>
#include <asAreaCompGrid.h>


asPredictorEcmwfIfsGrib::asPredictorEcmwfIfsGrib(const wxString &dataId)
        : asPredictor(dataId)
{
    // Set the basic properties.
    m_datasetId = "ECMWF_IFS_GRIB";
    m_provider = "ECMWF";
    m_datasetName = "Integrated Forecasting System (IFS) grib files";
    m_fileType = asFile::Grib;
    m_isEnsemble = false;
    m_strideAllowed = false;
    m_fStr.dimLatName = "lat";
    m_fStr.dimLonName = "lon";
    m_fStr.dimTimeName = "time";
    m_fStr.dimLevelName = "level";
}

bool asPredictorEcmwfIfsGrib::Init()
{
    // Identify data ID and set the corresponding properties.
    if (IsGeopotentialHeight()) {
        m_parameter = GeopotentialHeight;
        m_gribCode = {0, 3, 5, 100};
        m_unit = m;
    } else if (IsAirTemperature()) {
        m_parameter = AirTemperature;
        m_gribCode = {0, 0, 0, 100};
        m_unit = degK;
    } else if (IsVerticalVelocity()) {
        m_parameter = VerticalVelocity;
        m_gribCode = {0, 2, 8, 100};
        m_unit = Pa_s;
    } else if (IsRelativeHumidity()) {
        m_parameter = RelativeHumidity;
        m_gribCode = {0, 1, 1, 100};
        m_unit = percent;
    } else if (IsUwindComponent()) {
        m_parameter = Uwind;
        m_gribCode = {0, 2, 2, 100};
        m_unit = m_s;
    } else if (IsVwindComponent()) {
        m_parameter = Vwind;
        m_gribCode = {0, 2, 3, 100};
        m_unit = m_s;
    } else if (IsPrecipitableWater()) {
        m_parameter = PrecipitableWater;
        m_gribCode = {0, 1, 3, 200};
        m_unit = mm;
    } else {
        asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                          m_dataId, m_product));
    }

    // Check data ID
    if (m_parameter == ParameterUndefined) {
        wxLogError(_("The provided data ID (%s) does not match any possible option in the dataset %s."),
                   m_dataId, m_datasetName);
        return false;
    }

    // Check directory is set
    if (GetDirectoryPath().IsEmpty()) {
        wxLogError(_("The path to the directory has not been set for the data %s from the dataset %s."),
                   m_dataId, m_datasetName);
        return false;
    }

    // Set to initialized
    m_initialized = true;

    return true;
}

double asPredictorEcmwfIfsGrib::ConvertToMjd(double timeValue, double refValue) const
{
    wxASSERT(refValue > 30000);
    wxASSERT(refValue < 70000);

    return refValue + (timeValue / 24.0); // hours to days
}

