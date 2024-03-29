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
 * Portions Copyright 2017 Pascal Horton, University of Bern.
 */

#include "asPredictorEcmwfCera20C.h"

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorEcmwfCera20C::asPredictorEcmwfCera20C(const wxString& dataId)
    : asPredictor(dataId) {
    // Set the basic properties.
    m_datasetId = "ECMWF_CERA_20C";
    m_provider = "ECMWF";
    m_datasetName = "Coupled ERA 20th Century";
    m_fileType = asFile::Netcdf;
    m_isEnsemble = true;
    m_strideAllowed = true;
    m_fStr.dimLatName = "latitude";
    m_fStr.dimLonName = "longitude";
    m_fStr.dimTimeName = "time";
    m_fStr.dimLevelName = "level";
    m_fStr.dimMemberName = "number";
}

bool asPredictorEcmwfCera20C::Init() {
    CheckLevelTypeIsDefined();

    // List of variables: http://rda.ucar.edu/datasets/ds627.0/docs/era_interim_grib_table.html

    // Identify data ID and set the corresponding properties.
    if (IsPressureLevel()) {
        m_fStr.hasLevelDim = true;
        if (IsGeopotential()) {
            m_parameter = Geopotential;
            m_parameterName = "Geopotential";
            m_fileVarName = "z";
            m_unit = m2_s2;
        } else if (IsAirTemperature()) {
            m_parameter = AirTemperature;
            m_parameterName = "Temperature";
            m_fileVarName = "t";
            m_unit = degK;
        } else if (IsRelativeHumidity()) {
            m_parameter = RelativeHumidity;
            m_parameterName = "Relative humidity";
            m_fileVarName = "r";
            m_unit = percent;
        } else if (IsVerticalVelocity()) {
            m_parameter = VerticalVelocity;
            m_parameterName = "Vertical velocity";
            m_fileVarName = "w";
            m_unit = Pa_s;
        } else {
            m_parameter = ParameterUndefined;
            m_parameterName = "Undefined";
            m_fileVarName = m_dataId;
            m_unit = UnitUndefined;
        }
        m_fileNamePattern = m_fileVarName + ".%d.nc";

    } else if (IsSurfaceLevel()) {
        m_fStr.hasLevelDim = false;
        if (IsTotalColumnWater()) {
            m_parameter = TotalColumnWater;
            m_parameterName = "Total column water";
            m_fileVarName = "tcw";
            m_unit = kg_m2;
        } else if (IsTotalPrecipitation()) {
            m_parameter = Precipitation;
            m_parameterName = "Total precipitation";
            m_fileVarName = "tp";
            m_unit = m;
        } else if (IsSeaLevelPressure()) {
            m_parameter = Pressure;
            m_parameterName = "Sea level pressure";
            m_fileVarName = "msl";
            m_unit = Pa;
        } else {
            m_parameter = ParameterUndefined;
            m_parameterName = "Undefined";
            m_fileVarName = m_dataId;
            m_unit = UnitUndefined;
        }
        m_fileNamePattern = m_fileVarName + ".%d.nc";

    } else {
        wxLogError(_("level type not implemented for this reanalysis dataset."));
        return false;
    }

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

void asPredictorEcmwfCera20C::ListFiles(asTimeArray& timeArray) {
    for (int iYear = timeArray.GetStartingYear(); iYear <= timeArray.GetEndingYear(); iYear++) {
        m_files.push_back(GetFullDirectoryPath() + asStrF(m_fileNamePattern, iYear));
    }
}

void asPredictorEcmwfCera20C::ConvertToMjd(a1d& time, double refValue) const {
    time = (time / 24.0) + asTime::GetMJD(1900, 1, 1);
}
