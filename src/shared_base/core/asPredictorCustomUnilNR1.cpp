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
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#include "asPredictorCustomUnilNR1.h"

#include <asTimeArray.h>
#include <asAreaCompGrid.h>


asPredictorCustomUnilNR1::asPredictorCustomUnilNR1(const wxString &dataId)
        : asPredictor(dataId)
{
    // Set the basic properties.
    m_datasetId = "Custom_Unil_NR1";
    m_provider = "NCEP/NCAR";
    m_transformedBy = "Pascal Horton";
    m_datasetName = "Reanalysis 1 subset";
    m_fileType = asFile::Netcdf;
    m_strideAllowed = true;
    m_nanValues.push_back(32767);
    m_nanValues.push_back(936 * std::pow(10.f, 34.f));
    m_fStr.dimLatName = "lat";
    m_fStr.dimLonName = "lon";
    m_fStr.dimTimeName = "time";
    m_fStr.dimLevelName = "level";
}

bool asPredictorCustomUnilNR1::Init()
{
    // Identify data ID and set the corresponding properties.
    if (IsGeopotentialHeight()) {
        m_fStr.hasLevelDim = true;
        m_parameter = GeopotentialHeight;
        m_parameterName = "Geopotential height";
        m_fileNamePattern = "hgt.nc";
        m_fileVarName = "hgt";
        m_unit = m;
    } else if (IsAirTemperature()) {
        m_fStr.hasLevelDim = true;
        m_parameter = AirTemperature;
        m_parameterName = "Air Temperature";
        m_fileNamePattern = "air.nc";
        m_fileVarName = "air";
        m_unit = degK;
    } else if (IsVerticalVelocity()) {
        m_fStr.hasLevelDim = true;
        m_parameter = VerticalVelocity;
        m_parameterName = "Vertical velocity";
        m_fileNamePattern = "omega.nc";
        m_fileVarName = "omega";
        m_unit = Pa_s;
    } else if (IsRelativeHumidity()) {
        m_fStr.hasLevelDim = true;
        m_parameter = RelativeHumidity;
        m_parameterName = "Relative Humidity";
        m_fileNamePattern = "rhum.nc";
        m_fileVarName = "rhum";
        m_unit = percent;
    } else if (IsSpecificHumidity()) {
        m_fStr.hasLevelDim = true;
        m_parameter = SpecificHumidity;
        m_parameterName = "Specific Humidity";
        m_fileNamePattern = "shum.nc";
        m_fileVarName = "shum";
        m_unit = kg_kg;
    } else if (IsUwindComponent()) {
        m_fStr.hasLevelDim = true;
        m_parameter = Uwind;
        m_parameterName = "U-Wind";
        m_fileNamePattern = "uwnd.nc";
        m_fileVarName = "uwnd";
        m_unit = m_s;
    } else if (IsVwindComponent()) {
        m_fStr.hasLevelDim = true;
        m_parameter = Vwind;
        m_parameterName = "V-Wind";
        m_fileNamePattern = "vwnd.nc";
        m_fileVarName = "vwnd";
        m_unit = m_s;
    } else if (IsPrecipitableWater()) {
        m_fStr.hasLevelDim = false;
        m_parameter = PrecipitableWater;
        m_parameterName = "Precipitable water";
        m_fileNamePattern = "pr_wtr.nc";
        m_fileVarName = "pr_wtr";
        m_unit = mm;
    } else {
        wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), m_dataId, m_product);
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

void asPredictorCustomUnilNR1::ListFiles(asTimeArray &timeArray)
{
    m_files.push_back(GetFullDirectoryPath() + m_fileNamePattern);
}

double asPredictorCustomUnilNR1::ConvertToMjd(double timeValue, double refValue) const
{
    timeValue = (timeValue / 24.0); // hours to days
    if (timeValue < 500 * 365) { // New format
        timeValue += asTime::GetMJD(1800, 1, 1); // to MJD: add a negative time span
    } else { // Old format
        timeValue += asTime::GetMJD(1, 1, 1); // to MJD: add a negative time span
    }

    return timeValue;
}
