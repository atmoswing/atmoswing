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

#include "asDataPredictorArchiveNoaa20Cr2c.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileNetcdf.h>


asDataPredictorArchiveNoaa20Cr2c::asDataPredictorArchiveNoaa20Cr2c(const wxString &dataId)
        : asDataPredictorArchive(dataId)
{
    // Set the basic properties.
    m_initialized = false;
    m_datasetId = "NOAA_20CR_v2c";
    m_originalProvider = "NOAA";
    m_datasetName = "Twentieth Century Reanalysis (v2c)";
    m_originalProviderStart = asTime::GetMJD(1871, 1, 1);
    m_originalProviderEnd = asTime::GetMJD(2012, 12, 31, 18);
    m_timeZoneHours = 0;
    m_timeStepHours = 6;
    m_firstTimeStepHours = 0;
    m_nanValues.push_back(-9.96921 * std::pow(10.f, 36.f));
    m_xAxisShift = 0;
    m_yAxisShift = 0;
    m_fileStructure.dimLatName = "lat";
    m_fileStructure.dimLonName = "lon";
    m_fileStructure.dimTimeName = "time";
    m_fileStructure.dimLevelName = "level";
}

asDataPredictorArchiveNoaa20Cr2c::~asDataPredictorArchiveNoaa20Cr2c()
{

}

bool asDataPredictorArchiveNoaa20Cr2c::Init()
{
    CheckLevelTypeIsDefined();

    // Identify data ID and set the corresponding properties.
    if (m_product.IsSameAs("pressure", false) || m_product.IsSameAs("press", false)) {
        m_fileStructure.hasLevelDimension = true;
        m_subFolder = "pressure";
        m_xAxisStep = 2;
        m_yAxisStep = 2;
        if (m_dataId.IsSameAs("air", false)) {
            m_parameter = AirTemperature;
            m_parameterName = "Air Temperature";
            m_fileVariableName = "air";
            m_unit = degK;
        } else if (m_dataId.IsSameAs("hgt", false)) {
            m_parameter = GeopotentialHeight;
            m_parameterName = "Geopotential height";
            m_fileVariableName = "hgt";
            m_unit = m;
        } else if (m_dataId.IsSameAs("omega", false)) {
            m_parameter = Omega;
            m_parameterName = "Omega (Vertical Velocity)";
            m_fileVariableName = "omega";
            m_unit = Pa_s;
        } else if (m_dataId.IsSameAs("rhum", false)) {
            m_parameter = RelativeHumidity;
            m_parameterName = "Relative Humidity";
            m_fileVariableName = "rhum";
            m_unit = percent;
        } else if (m_dataId.IsSameAs("shum", false)) {
            m_parameter = SpecificHumidity;
            m_parameterName = "Specific Humidity";
            m_fileVariableName = "shum";
            m_unit = kg_kg;
        } else if (m_dataId.IsSameAs("uwnd", false)) {
            m_parameter = Uwind;
            m_parameterName = "U-Wind";
            m_fileVariableName = "uwnd";
            m_unit = m_s;
        } else if (m_dataId.IsSameAs("vwnd", false)) {
            m_parameter = Vwind;
            m_parameterName = "V-Wind";
            m_fileVariableName = "vwnd";
            m_unit = m_s;
        } else {
            asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                              m_dataId, m_product));
        }
        m_fileNamePattern = m_fileVariableName + ".%d.nc";

    } else {
        asThrowException(_("Product type not implemented for this reanalysis dataset."));
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

VectorString asDataPredictorArchiveNoaa20Cr2c::GetListOfFiles(asTimeArray &timeArray) const
{
    VectorString files;

    for (int i_year = timeArray.GetStartingYear(); i_year <= timeArray.GetEndingYear(); i_year++) {
        files.push_back(GetFullDirectoryPath() + wxString::Format(m_fileNamePattern, i_year));
    }

    return files;
}

bool asDataPredictorArchiveNoaa20Cr2c::ExtractFromFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea,
                                                            asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    return ExtractFromNetcdfFile(fileName, dataArea, timeArray, compositeData);
}

double asDataPredictorArchiveNoaa20Cr2c::ConvertToMjd(double timeValue, double refValue) const
{
    timeValue = (timeValue / 24.0); // hours to days
    timeValue += asTime::GetMJD(1800, 1, 1); // to MJD: add a negative time span

    return timeValue;
}
