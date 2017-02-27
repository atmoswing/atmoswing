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

#include "asDataPredictorArchiveNoaa20Cr2cEnsemble.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileNetcdf.h>


asDataPredictorArchiveNoaa20Cr2cEnsemble::asDataPredictorArchiveNoaa20Cr2cEnsemble(const wxString &dataId)
        : asDataPredictorArchive(dataId)
{
    // Set the basic properties.
    m_initialized = false;
    m_datasetId = "NOAA_20CR_v2c_ens";
    m_originalProvider = "NOAA";
    m_datasetName = "Twentieth Century Reanalysis (v2c) Ensemble";
    m_isEnsemble = true;
    m_originalProviderStart = asTime::GetMJD(1850, 1, 1);
    m_originalProviderEnd = asTime::GetMJD(2014, 12, 31, 18);
    m_timeZoneHours = 0;
    m_timeStepHours = 6;
    m_firstTimeStepHours = 0;
    m_nanValues.push_back(-9.96921 * std::pow(10.f, 36.f));
    m_xAxisShift = 0;
    m_yAxisShift = 0;
    m_fileStructure.dimLatName = "lat";
    m_fileStructure.dimLonName = "lon";
    m_fileStructure.dimTimeName = "time";
    m_fileStructure.dimMemberName = "ensemble_member";
    m_fileStructure.hasLevelDimension = false;
}

asDataPredictorArchiveNoaa20Cr2cEnsemble::~asDataPredictorArchiveNoaa20Cr2cEnsemble()
{

}

bool asDataPredictorArchiveNoaa20Cr2cEnsemble::Init()
{
    CheckLevelTypeIsDefined();

    // Identify data ID and set the corresponding properties.
    if (m_product.IsSameAs("analysis", false)) {

        m_subFolder = "analysis";
        m_xAxisStep = 2;
        m_yAxisStep = 2;
        if (m_dataId.IsSameAs("prmsl", false)) {
            m_parameter = Pressure;
            m_parameterName = "Sea level pressure";
            m_fileVariableName = "prmsl";
            m_unit = Pa;
        } else if (m_dataId.IsSameAs("pwat", false)) {
            m_parameter = PrecipitableWater;
            m_parameterName = "Precipitable water";
            m_fileVariableName = "pwat";
            m_unit = kg_m2;
        } else if (m_dataId.IsSameAs("omega500", false)) {
            m_parameter = VerticalVelocity;
            m_parameterName = "Vertical velocity at 500 hPa";
            m_fileVariableName = "omega500";
            m_unit = Pa_s;
        } else if (m_dataId.IsSameAs("rh850", false)) {
            m_parameter = RelativeHumidity;
            m_parameterName = "Relative Humidity at 850 hPa";
            m_fileVariableName = "rh850";
            m_unit = percent;
        } else if (m_dataId.IsSameAs("rh9950", false)) {
            m_parameter = RelativeHumidity;
            m_parameterName = "Relative Humidity at the pressure level 0.995 times the surface pressure";
            m_fileVariableName = "rh850";
            m_unit = percent;
        } else if (m_dataId.IsSameAs("t850", false)) {
            m_parameter = AirTemperature;
            m_parameterName = "Air Temperature at 850 hPa";
            m_fileVariableName = "t850";
            m_unit = degK;
        } else if (m_dataId.IsSameAs("t9950", false)) {
            m_parameter = AirTemperature;
            m_parameterName = "Air Temperature at the pressure level 0.995 times the surface pressure";
            m_fileVariableName = "t9950";
            m_unit = degK;
        } else if (m_dataId.IsSameAs("z200", false)) {
            m_parameter = GeopotentialHeight;
            m_parameterName = "Geopotential height at 200 hPa";
            m_fileVariableName = "z200";
            m_unit = m;
        } else if (m_dataId.IsSameAs("z500", false)) {
            m_parameter = GeopotentialHeight;
            m_parameterName = "Geopotential height at 500 hPa";
            m_fileVariableName = "z500";
            m_unit = m;
        } else if (m_dataId.IsSameAs("z1000", false)) {
            m_parameter = GeopotentialHeight;
            m_parameterName = "Geopotential height at 1000 hPa";
            m_fileVariableName = "z1000";
            m_unit = m;
        } else {
            asThrowException(
                    wxString::Format(_("No '%s' parameter identified for the provided data type (%s)."), m_dataId,
                                     m_product));
        }
        m_fileNamePattern = m_fileVariableName + "_%d.nc";

    } else if (m_product.IsSameAs("first_guess", false)) {

        m_subFolder = "first_guess";
        m_xAxisStep = 2;
        m_yAxisStep = 2;
        if (m_dataId.IsSameAs("prate", false)) {
            m_parameter = PrecipitationRate;
            m_parameterName = "Precipitation rate";
            m_fileVariableName = "prate";
            m_unit = kg_m2_s;
        } else {
            asThrowException(wxString::Format(_("No '%s' parameter identified for the provided data type (%s)."),
                                              m_dataId, m_product));
        }
        m_fileNamePattern = m_fileVariableName + "_%d.nc";

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

VectorString asDataPredictorArchiveNoaa20Cr2cEnsemble::GetListOfFiles(asTimeArray &timeArray) const
{
    VectorString files;

    for (int i_year = timeArray.GetStartingYear(); i_year <= timeArray.GetEndingYear(); i_year++) {
        files.push_back(GetFullDirectoryPath() + wxString::Format(m_fileNamePattern, i_year));
    }

    return files;
}

bool asDataPredictorArchiveNoaa20Cr2cEnsemble::ExtractFromFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea,
                                                            asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    return ExtractFromNetcdfFile(fileName, dataArea, timeArray, compositeData);
}

double asDataPredictorArchiveNoaa20Cr2cEnsemble::ConvertToMjd(double timeValue, double refValue) const
{
    timeValue = (timeValue / 24.0); // hours to days
    timeValue += asTime::GetMJD(1, 1, 1); // to MJD: add a negative time span

    return timeValue;
}
