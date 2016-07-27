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

#include "asDataPredictorArchiveNcepReanalysis1Subset.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileNetcdf.h>


asDataPredictorArchiveNcepReanalysis1Subset::asDataPredictorArchiveNcepReanalysis1Subset(const wxString &dataId)
        : asDataPredictorArchive(dataId)
{
    // Set the basic properties.
    m_initialized = false;
    m_datasetId = "NCEP_Reanalysis_v1_terranum";
    m_originalProvider = "NCEP/NCAR";
    m_transformedBy = "Pascal Horton";
    m_datasetName = "Reanalysis 1 subset";
    m_originalProviderStart = asTime::GetMJD(1948, 1, 1);
    m_originalProviderEnd = NaNDouble;
    m_timeZoneHours = 0;
    m_timeStepHours = 6;
    m_firstTimeStepHours = 0;
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
}

asDataPredictorArchiveNcepReanalysis1Subset::~asDataPredictorArchiveNcepReanalysis1Subset()
{

}

bool asDataPredictorArchiveNcepReanalysis1Subset::Init()
{
    // Identify data ID and set the corresponding properties.
    if (m_dataId.IsSameAs("hgt", false)) {
        m_fileStructure.hasLevelDimension = true;
        m_parameter = GeopotentialHeight;
        m_parameterName = "Geopotential height";
        m_fileNamePattern = "hgt.nc";
        m_fileVariableName = "hgt";
        m_unit = m;
    } else if (m_dataId.IsSameAs("air", false)) {
        m_fileStructure.hasLevelDimension = true;
        m_parameter = AirTemperature;
        m_parameterName = "Air Temperature";
        m_fileNamePattern = "air.nc";
        m_fileVariableName = "air";
        m_unit = degK;
    } else if (m_dataId.IsSameAs("omega", false)) {
        m_fileStructure.hasLevelDimension = true;
        m_parameter = Omega;
        m_parameterName = "Omega (Vertical Velocity)";
        m_fileNamePattern = "omega.nc";
        m_fileVariableName = "omega";
        m_unit = Pa_s;
    } else if (m_dataId.IsSameAs("rhum", false)) {
        m_fileStructure.hasLevelDimension = true;
        m_parameter = RelativeHumidity;
        m_parameterName = "Relative Humidity";
        m_fileNamePattern = "rhum.nc";
        m_fileVariableName = "rhum";
        m_unit = percent;
    } else if (m_dataId.IsSameAs("shum", false)) {
        m_fileStructure.hasLevelDimension = true;
        m_parameter = SpecificHumidity;
        m_parameterName = "Specific Humidity";
        m_fileNamePattern = "shum.nc";
        m_fileVariableName = "shum";
        m_unit = kg_kg;
    } else if (m_dataId.IsSameAs("uwnd", false)) {
        m_fileStructure.hasLevelDimension = true;
        m_parameter = Uwind;
        m_parameterName = "U-Wind";
        m_fileNamePattern = "uwnd.nc";
        m_fileVariableName = "uwnd";
        m_unit = m_s;
    } else if (m_dataId.IsSameAs("vwnd", false)) {
        m_fileStructure.hasLevelDimension = true;
        m_parameter = Vwind;
        m_parameterName = "V-Wind";
        m_fileNamePattern = "vwnd.nc";
        m_fileVariableName = "vwnd";
        m_unit = m_s;
    } else if (m_dataId.IsSameAs("prwtr", false)) {
        m_fileStructure.hasLevelDimension = false;
        m_parameter = PrecipitableWater;
        m_parameterName = "Precipitable water";
        m_fileNamePattern = "pr_wtr.nc";
        m_fileVariableName = "pr_wtr";
        m_unit = mm;
    } else {
        asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                          m_dataId, LevelEnumToString(m_product)));
    }

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

VectorString asDataPredictorArchiveNcepReanalysis1Subset::GetListOfFiles(asTimeArray &timeArray) const
{
    VectorString files;

    files.push_back(GetFullDirectoryPath() + m_fileNamePattern);

    return files;
}

bool asDataPredictorArchiveNcepReanalysis1Subset::ExtractFromFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea,
                                                            asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    return ExtractFromNetcdfFile(fileName, dataArea, timeArray, compositeData);
}

double asDataPredictorArchiveNcepReanalysis1Subset::ConvertToMjd(double timeValue, double refValue) const
{
    timeValue = (timeValue / 24.0); // hours to days
    if (timeValue < 500 * 365) { // New format
        timeValue += asTime::GetMJD(1800, 1, 1); // to MJD: add a negative time span
    } else { // Old format
        timeValue += asTime::GetMJD(1, 1, 1); // to MJD: add a negative time span
    }

    return timeValue;
}
