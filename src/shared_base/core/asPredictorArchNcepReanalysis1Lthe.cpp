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
 * Portions Copyright 2014-2015 Pascal Horton, Terranum.
 */

#include "asPredictorArchNcepReanalysis1Lthe.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>


asPredictorArchNcepReanalysis1Lthe::asPredictorArchNcepReanalysis1Lthe(const wxString &dataId)
        : asPredictorArchNcepReanalysis1Subset(dataId)
{
    // Set the basic properties.
    m_datasetId = "NCEP_Reanalysis_v1_lthe";
    m_originalProvider = "NCEP/NCAR";
    m_transformedBy = "LTHE";
    m_datasetName = "Reanalysis 1 subset from LTHE";
    m_fileType = asFile::Netcdf;
    m_originalProviderStart = asTime::GetMJD(1948, 1, 1);
    m_originalProviderEnd = NaNd;
    m_timeStepHours = 24;
    m_firstTimeStepHours = 0;
    m_strideAllowed = true;
    m_nanValues.push_back(32767);
    m_nanValues.push_back(936 * std::pow(10.f, 34.f));
    m_xAxisShift = 0;
    m_yAxisShift = 0;
    m_xAxisStep = 2.5;
    m_yAxisStep = 2.5;
    m_subFolder = wxEmptyString;
    m_fileStructure.dimLatName = "lat";
    m_fileStructure.dimLonName = "lon";
    m_fileStructure.dimTimeName = "time";
    m_fileStructure.dimLevelName = "level";
    m_fileStructure.hasLevelDimension = true;
}

asPredictorArchNcepReanalysis1Lthe::~asPredictorArchNcepReanalysis1Lthe()
{

}

bool asPredictorArchNcepReanalysis1Lthe::Init()
{
    // Identify data ID and set the corresponding properties.
    if (m_dataId.IsSameAs("hgt_500hPa", false)) {
        m_parameter = GeopotentialHeight;
        m_parameterName = "Geopotential height";
        m_fileNamePattern = "NCEP_Reanalysis_v1_lthe_hgt_500hPa.nc";
        m_fileVariableName = "hgt";
        m_unit = m;
        m_firstTimeStepHours = 0;
        m_timeStepHours = 24;
    } else if (m_dataId.IsSameAs("hgt_1000hPa", false)) {
        m_parameter = GeopotentialHeight;
        m_parameterName = "Geopotential height";
        m_fileNamePattern = "NCEP_Reanalysis_v1_lthe_hgt_1000hPa.nc";
        m_fileVariableName = "hgt";
        m_unit = m;
        m_firstTimeStepHours = 12;
        m_timeStepHours = 24;
    } else if (m_dataId.IsSameAs("prwtr", false)) {
        m_parameter = PrecipitableWater;
        m_parameterName = "Precipitable water";
        m_fileNamePattern = "NCEP_Reanalysis_v1_lthe_prwtr.nc";
        m_fileVariableName = "pwa";
        m_unit = mm;
        m_firstTimeStepHours = 0;
        m_timeStepHours = 12;
    } else if (m_dataId.IsSameAs("rhum", false)) {
        m_parameter = RelativeHumidity;
        m_parameterName = "Relative Humidity";
        m_fileNamePattern = "NCEP_Reanalysis_v1_lthe_rhum.nc";
        m_fileVariableName = "rhum";
        m_unit = percent;
        m_firstTimeStepHours = 0;
        m_timeStepHours = 12;
    } else {
        asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."), m_dataId,
                                          m_product));
    }

    // Check data ID
    if (m_fileNamePattern.IsEmpty() || m_fileVariableName.IsEmpty()) {
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

void asPredictorArchNcepReanalysis1Lthe::ListFiles(asTimeArray &timeArray)
{
    m_files.push_back(GetFullDirectoryPath() + m_fileNamePattern);
}

double asPredictorArchNcepReanalysis1Lthe::ConvertToMjd(double timeValue, double refValue) const
{
    timeValue = (timeValue / 24.0); // hours to days
    if (timeValue < 500 * 365) { // New format
        timeValue += asTime::GetMJD(1800, 1, 1); // to MJD: add a negative time span
    } else { // Old format
        timeValue += asTime::GetMJD(1, 1, 1); // to MJD: add a negative time span
    }

    return timeValue;
}