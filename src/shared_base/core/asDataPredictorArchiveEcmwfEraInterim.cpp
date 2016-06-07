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

#include "asDataPredictorArchiveEcmwfEraInterim.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileNetcdf.h>


asDataPredictorArchiveEcmwfEraInterim::asDataPredictorArchiveEcmwfEraInterim(const wxString &dataId)
        : asDataPredictorArchiveNcepReanalysis1Subset(dataId)
{
    // Set the basic properties.
    m_initialized = false;
    m_datasetId = "ECMWF_ERA-40";
    m_originalProvider = "ECMWF";
    m_datasetName = "ERA-40";
    m_originalProviderStart = asTime::GetMJD(1957, 9, 1);
    m_originalProviderEnd = asTime::GetMJD(2002, 8, 31);
    m_timeZoneHours = 0;
    m_timeStepHours = 6;
    m_firstTimeStepHours = 0;
    m_nanValues.push_back(-32767);
    m_xAxisShift = 0;
    m_yAxisShift = 0;
    m_subFolder = wxEmptyString;
    m_fileAxisLatName = "latitude";
    m_fileAxisLonName = "longitude";
    m_fileAxisTimeName = "time";
    m_fileAxisLevelName = "level";

    // Identify data ID and set the corresponding properties.
    switch (m_levelType) {
        case PressureLevel:
            m_subFolder = "pressure";
            m_xAxisStep = 2.5;
            m_yAxisStep = 2.5;
            if (m_dataId.IsSameAs("z", false)) {
                m_parameter = GeopotentialHeight;
                m_parameterName = "Geopotential Height";
                m_fileNamePattern = "ECMWF_ERA40_hgt.nc";
                m_fileVariableName = "z";
                m_unit = m;
            } else {
                asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                                  m_dataId, LevelEnumToString(m_levelType)));
            }
            m_fileNamePattern = m_fileVariableName + ".%d.nc";
            break;

        case Surface:
            m_subFolder = "surface";
            m_xAxisStep = 2.5;
            m_yAxisStep = 2.5;
            if (m_dataId.IsSameAs("xxxx", false)) {

            } else {
                asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                                  m_dataId, LevelEnumToString(m_levelType)));
            }
            break;

        case PotentialTemperatureLevel:
            m_subFolder = "surface_gauss";
            m_xAxisStep = NaNFloat;
            m_yAxisStep = NaNFloat;
            if (m_dataId.IsSameAs("xxxx", false)) {

            } else {
                asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                                  m_dataId, LevelEnumToString(m_levelType)));
            }
            break;

        case PotentialVorticityLevel:
            m_subFolder = "surface_gauss";
            m_xAxisStep = NaNFloat;
            m_yAxisStep = NaNFloat;
            if (m_dataId.IsSameAs("xxxx", false)) {

            } else {
                asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                                  m_dataId, LevelEnumToString(m_levelType)));
            }
            break;

        default: asThrowException(_("Level type not implemented for this reanalysis dataset."));
    }
}

asDataPredictorArchiveEcmwfEraInterim::~asDataPredictorArchiveEcmwfEraInterim()
{

}

bool asDataPredictorArchiveEcmwfEraInterim::Init()
{
    // Check data ID
    if (m_fileNamePattern.IsEmpty() || m_fileVariableName.IsEmpty()) {
        asLogError(
                wxString::Format(_("The provided data ID (%s) does not match any possible option in the dataset %s."),
                                 m_dataId, m_datasetName));
        return false;
    }

    // Check directory is set
    if (GetDirectoryPath().IsEmpty()) {
        asLogError(
                wxString::Format(_("The path to the directory has not been set for the data %s from the dataset %s."),
                                 m_dataId, m_datasetName));
        return false;
    }

    // Set to initialized
    m_initialized = true;

    return true;
}

