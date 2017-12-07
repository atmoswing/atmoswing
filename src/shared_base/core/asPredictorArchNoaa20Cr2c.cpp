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

#include "asPredictorArchNoaa20Cr2c.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>


asPredictorArchNoaa20Cr2c::asPredictorArchNoaa20Cr2c(const wxString &dataId)
        : asPredictorArch(dataId)
{
    // Set the basic properties.
    m_datasetId = "NOAA_20CR_v2c";
    m_originalProvider = "NOAA";
    m_datasetName = "Twentieth Century Reanalysis (v2c)";
    m_originalProviderStart = asTime::GetMJD(1850, 1, 1);
    m_originalProviderEnd = asTime::GetMJD(2014, 12, 31, 18);
    m_timeStepHours = 6;
    m_firstTimeStepHours = 0;
    m_strideAllowed = true;
    m_nanValues.push_back(-9.96921 * std::pow(10.f, 36.f));
    m_xAxisShift = 0;
    m_yAxisShift = 0;
    m_fileStructure.dimLatName = "lat";
    m_fileStructure.dimLonName = "lon";
    m_fileStructure.dimTimeName = "time";
    m_fileStructure.dimLevelName = "level";
}

asPredictorArchNoaa20Cr2c::~asPredictorArchNoaa20Cr2c()
{

}

bool asPredictorArchNoaa20Cr2c::Init()
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
            m_parameter = VerticalVelocity;
            m_parameterName = "Vertical velocity";
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

    } else if (m_product.IsSameAs("surface", false) || m_product.IsSameAs("surf", false) ||
               m_product.IsSameAs("monolevel", false)) {
        m_fileStructure.hasLevelDimension = false;
        m_subFolder = "monolevel";
        m_xAxisStep = 2;
        m_yAxisStep = 2;
        if (m_dataId.IsSameAs("prwtr", false)) {
            m_parameter = PrecipitableWater;
            m_parameterName = "Precipitable water";
            m_fileNamePattern = "pr_wtr.eatm.%d.nc";
            m_fileVariableName = "pr_wtr";
            m_unit = mm;
        } else if (m_dataId.IsSameAs("mslp", false)) {
            m_parameter = Pressure;
            m_parameterName = "Sea level pressure";
            m_fileNamePattern = "prmsl.%d.nc";
            m_fileVariableName = "prmsl";
            m_unit = Pa;
        } else {
            asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                              m_dataId, m_product));
        }

    } else if (m_product.IsSameAs("surface_gauss", false) || m_product.IsSameAs("gauss", false) ||
               m_product.IsSameAs("gaussian", false) || m_product.IsSameAs("flux", false)) {
        m_fileStructure.hasLevelDimension = false;
        m_subFolder = "gaussian";
        m_xAxisStep = NaNf;
        m_yAxisStep = NaNf;
        if (m_dataId.IsSameAs("prate", false)) {
            m_parameter = PrecipitationRate;
            m_parameterName = "Precipitation rate";
            m_fileNamePattern = "prate.%d.nc";
            m_fileVariableName = "prate";
            m_unit = kg_m2_s;
            m_subFolder.Append(DS + "monolevel");
        } else {
            asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                              m_dataId, m_product));
        }

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

vwxs asPredictorArchNoaa20Cr2c::GetListOfFiles(asTimeArray &timeArray) const
{
    vwxs files;

    for (int iYear = timeArray.GetStartingYear(); iYear <= timeArray.GetEndingYear(); iYear++) {
        files.push_back(GetFullDirectoryPath() + wxString::Format(m_fileNamePattern, iYear));
    }

    return files;
}

bool asPredictorArchNoaa20Cr2c::ExtractFromFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea,
                                                asTimeArray &timeArray, vvva2f &compositeData)
{
    return ExtractFromNetcdfFile(fileName, dataArea, timeArray, compositeData);
}

double asPredictorArchNoaa20Cr2c::ConvertToMjd(double timeValue, double refValue) const
{
    timeValue = (timeValue / 24.0); // hours to days
    timeValue += asTime::GetMJD(1800, 1, 1); // to MJD: add a negative time span

    return timeValue;
}
