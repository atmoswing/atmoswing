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

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorEcmwfIfsGrib::asPredictorEcmwfIfsGrib(const wxString& dataId)
    : asPredictor(dataId) {
    // Set the basic properties.
    m_datasetId = "ECMWF_IFS_GRIB";
    m_provider = "ECMWF";
    m_datasetName = "Integrated Forecasting System (IFS) grib files";
    m_fileType = asFile::Grib;
    m_isEnsemble = false;
    m_strideAllowed = false;
    m_fStr.hasLevelDim = false;
    m_fStr.singleTimeStep = true;
    m_nanValues.push_back(NaNd);
    m_nanValues.push_back(NaNf);
    m_parameter = ParameterUndefined;
}

bool asPredictorEcmwfIfsGrib::Init() {
    // Identify data ID and set the corresponding properties.
    if (m_dataId.IsSameAs("z", false)) {
        m_parameter = Geopotential;
        m_gribCode = {0, 128, 129, 100};
        m_unit = m2_s2;
        m_fStr.hasLevelDim = true;
    } else if (m_dataId.IsSameAs("gh", false)) {
        m_parameter = GeopotentialHeight;
        m_gribCode = {0, 128, 156, 100};
        m_unit = m;
        m_fStr.hasLevelDim = true;
    } else if (IsAirTemperature()) {
        m_parameter = AirTemperature;
        m_gribCode = {0, 128, 130, 100};
        m_unit = degK;
        m_fStr.hasLevelDim = true;
    } else if (IsVerticalVelocity()) {
        m_parameter = VerticalVelocity;
        m_gribCode = {0, 128, 135, 100};
        m_unit = Pa_s;
        m_fStr.hasLevelDim = true;
    } else if (IsRelativeHumidity()) {
        m_parameter = RelativeHumidity;
        m_gribCode = {0, 128, 157, 100};
        m_unit = percent;
        m_fStr.hasLevelDim = true;
    } else if (IsSpecificHumidity()) {
        m_parameter = SpecificHumidity;
        m_gribCode = {0, 128, 133, 100};
        m_unit = percent;
        m_fStr.hasLevelDim = true;
    } else if (IsUwindComponent()) {
        m_parameter = Uwind;
        m_gribCode = {0, 128, 131, 100};
        m_unit = m_s;
        m_fStr.hasLevelDim = true;
    } else if (IsVwindComponent()) {
        m_parameter = Vwind;
        m_gribCode = {0, 128, 132, 100};
        m_unit = m_s;
        m_fStr.hasLevelDim = true;
    } else if (m_dataId.IsSameAs("thetaE", false)) {
        m_parameter = PotentialTemperature;
        m_gribCode = {0, 3, 113, 100};
        m_unit = W_m2;
        m_fStr.hasLevelDim = true;
    } else if (m_dataId.IsSameAs("thetaES", false)) {
        m_parameter = PotentialTemperature;
        m_gribCode = {0, 3, 114, 100};
        m_unit = W_m2;
        m_fStr.hasLevelDim = true;
    } else if (IsTotalColumnWaterVapour()) {
        m_parameter = PrecipitableWater;
        m_gribCode = {0, 128, 137, 200};
        m_unit = mm;
    } else if (IsPrecipitableWater()) {
        m_parameter = PrecipitableWater;
        m_gribCode = {0, 128, 136, 200};
        m_unit = mm;
    } else {
        wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), m_dataId, m_product);
        return false;
    }

    // Set to initialized
    m_initialized = true;

    return true;
}

void asPredictorEcmwfIfsGrib::ConvertToMjd(a1d& time, double refValue) const {
    time = (time / 24.0) + refValue;
}
