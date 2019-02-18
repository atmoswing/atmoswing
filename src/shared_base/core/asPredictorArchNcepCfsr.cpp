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
 * Portions Copyright 2016 Pascal Horton, University of Bern.
 */

#include "asPredictorArchNcepCfsr.h"

#include <asTimeArray.h>
#include <asAreaCompGrid.h>


asPredictorArchNcepCfsr::asPredictorArchNcepCfsr(const wxString &dataId)
        : asPredictorArch(dataId)
{
    // Set the basic properties.
    m_datasetId = "NCEP_CFSR";
    m_provider = "NCEP";
    m_datasetName = "CFSR";
    m_fileType = asFile::Grib;
    m_strideAllowed = false;
}

bool asPredictorArchNcepCfsr::Init()
{
    CheckLevelTypeIsDefined();

    // Last element in grib code: level type (http://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_table4-5.shtml)

    // Identify data ID and set the corresponding properties.
    if (IsPressureLevel()) {
        m_fStr.hasLevelDim = true;
        m_fStr.singleLevel = true;
        if (IsGeopotentialHeight()) {
            m_parameter = GeopotentialHeight;
            int arr[] = {0, 3, 5, 100};
            AssignGribCode(arr);
            m_parameterName = "Geopotential height @ Isobaric surface";
            m_unit = gpm;
        } else if (IsPrecipitableWater()) {
            m_parameter = PrecipitableWater;
            int arr[] = {0, 1, 3, 200};
            AssignGribCode(arr);
            m_parameterName = "Precipitable water @ Entire atmosphere layer";
            m_unit = kg_m2;
        } else if (IsSeaLevelPressure()) {
            m_parameter = Pressure;
            int arr[] = {0, 3, 0, 101};
            AssignGribCode(arr);
            m_parameterName = "Pressure @ Mean sea level";
            m_unit = Pa;
        } else if (IsRelativeHumidity()) {
            m_parameter = RelativeHumidity;
            int arr[] = {0, 1, 1, 100};
            AssignGribCode(arr);
            m_parameterName = "Relative humidity @ Isobaric surface";
            m_unit = percent;
        } else if (IsAirTemperature()) {
            m_parameter = AirTemperature;
            int arr[] = {0, 0, 0, 100};
            AssignGribCode(arr);
            m_parameterName = "Temperature @ Isobaric surface";
            m_unit = degK;
        } else {
            asThrowException(wxString::Format(_("Parameter '%s' not implemented yet."), m_dataId));
        }
        m_fileNamePattern = "%4d/%4d%02d/%4d%02d%02d/pgbhnl.gdas.%4d%02d%02d%02d.grb2";

    } else if (IsIsentropicLevel()) {
        asThrowException(_("Isentropic levels for CFSR are not implemented yet."));

    } else if (IsSurfaceFluxesLevel()) {
        asThrowException(_("Surface fluxes grids for CFSR are not implemented yet."));

    } else {
        asThrowException(_("level type not implemented for this reanalysis dataset."));
    }

    // Check data ID
    if (m_fileNamePattern.IsEmpty() || m_gribCode[2] == asNOT_FOUND) {
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

    wxASSERT(m_gribCode.size() == 4);

    // Set to initialized
    m_initialized = true;

    return true;
}

void asPredictorArchNcepCfsr::ListFiles(asTimeArray &timeArray)
{
    a1d tArray = timeArray.GetTimeArray();

    for (int i = 0; i < tArray.size(); i++) {
        Time t = asTime::GetTimeStruct(tArray[i]);
        m_files.push_back(GetFullDirectoryPath() + wxString::Format(m_fileNamePattern, t.year, t.year, t.month, t.year,
                                                                    t.month, t.day, t.year, t.month, t.day, t.hour));
    }
}
