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

#include "asPredictorArchNoaaOisst2Subset.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>


asPredictorArchNoaaOisst2Subset::asPredictorArchNoaaOisst2Subset(const wxString &dataId)
        : asPredictorArch(dataId)
{
    // Set the basic properties.
    m_datasetId = "NOAA_OISST_v2_subset";
    m_originalProvider = "NOAA";
    m_transformedBy = "Pascal Horton";
    m_datasetName = "Optimum Interpolation Sea Surface Temperature, version 2, subset";
    m_fileType = asFile::Netcdf;
    m_strideAllowed = true;
    m_nanValues.push_back(32767);
    m_nanValues.push_back(936 * std::pow(10.f, 34.f));
    m_xAxisShift = 0.125;
    m_yAxisShift = 0.125;
    m_xAxisStep = 1;
    m_yAxisStep = 1;
    m_subFolder = wxEmptyString;
    m_fileStructure.dimLatName = "lat";
    m_fileStructure.dimLonName = "lon";
    m_fileStructure.dimTimeName = "time";
    m_fileStructure.hasLevelDimension = false;
}

asPredictorArchNoaaOisst2Subset::~asPredictorArchNoaaOisst2Subset()
{

}

bool asPredictorArchNoaaOisst2Subset::Init()
{
    // Identify data ID and set the corresponding properties.
    if (m_dataId.IsSameAs("sst", false)) {
        m_parameter = SeaSurfaceTemperature;
        m_parameterName = "Sea Surface Temperature";
        m_fileNamePattern = "sst_1deg.nc";
        m_fileVariableName = "sst";
        m_unit = degC;
    } else if (m_dataId.IsSameAs("sst_anom", false)) {
        m_parameter = SeaSurfaceTemperatureAnomaly;
        m_parameterName = "Sea Surface Temperature Anomaly";
        m_fileNamePattern = "sst_anom_1deg.nc";
        m_fileVariableName = "anom";
        m_unit = degC;
    } else {
        asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."), m_dataId,
                                          m_product));
    }

    // Check data ID
    if (m_fileNamePattern.IsEmpty() || m_fileVariableName.IsEmpty()) {
        wxLogError(_("The provided data ID (%s) does not match any possible option in dataset %s."), m_dataId,
                   m_datasetName);
        return false;
    }

    // Check directory is set
    if (GetDirectoryPath().IsEmpty()) {
        wxLogError(_("The path to the directory has not been set for the data %s from dataset %s."), m_dataId,
                   m_datasetName);
        return false;
    }

    // Set to initialized
    m_initialized = true;

    return true;
}

void asPredictorArchNoaaOisst2Subset::ListFiles(asTimeArray &timeArray)
{
    m_files.push_back(GetFullDirectoryPath() + m_fileNamePattern);
}

double asPredictorArchNoaaOisst2Subset::ConvertToMjd(double timeValue, double refValue) const
{
    timeValue = (timeValue / 24.0); // hours to days
    if (timeValue < 500 * 365) { // New format
        timeValue += asTime::GetMJD(1800, 1, 1); // to MJD: add a negative time span
    } else { // Old format
        timeValue += asTime::GetMJD(1, 1, 1); // to MJD: add a negative time span
    }

    return timeValue;
}
