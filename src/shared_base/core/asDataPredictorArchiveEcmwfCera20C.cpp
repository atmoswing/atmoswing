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

#include "asDataPredictorArchiveEcmwfCera20C.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>


asDataPredictorArchiveEcmwfCera20C::asDataPredictorArchiveEcmwfCera20C(const wxString &dataId)
        : asDataPredictorArchive(dataId)
{
    // Set the basic properties.
    m_initialized = false;
    m_datasetId = "ECMWF_CERA_20C";
    m_originalProvider = "ECMWF";
    m_datasetName = "Coupled ERA 20th Century";
    m_isEnsemble = true;
    m_originalProviderStart = asTime::GetMJD(1901, 1, 1);
    m_originalProviderEnd = asTime::GetMJD(2010, 12, 31);
    m_timeZoneHours = 0;
    m_timeStepHours = 3;
    m_firstTimeStepHours = 0;
    m_xAxisShift = 0;
    m_yAxisShift = 0;
    m_fileStructure.dimLatName = "latitude";
    m_fileStructure.dimLonName = "longitude";
    m_fileStructure.dimTimeName = "time";
    m_fileStructure.dimLevelName = "level";
    m_fileStructure.dimMemberName = "number";
    m_subFolder = wxEmptyString;
}

asDataPredictorArchiveEcmwfCera20C::~asDataPredictorArchiveEcmwfCera20C()
{

}

bool asDataPredictorArchiveEcmwfCera20C::Init()
{
    CheckLevelTypeIsDefined();

    // List of variables: http://rda.ucar.edu/datasets/ds627.0/docs/era_interim_grib_table.html

    // Identify data ID and set the corresponding properties.
    if (m_product.IsSameAs("pressure_level", false) || m_product.IsSameAs("pressure", false) ||
        m_product.IsSameAs("press", false) || m_product.IsSameAs("pl", false)) {
        m_fileStructure.hasLevelDimension = true;
        m_subFolder = "pressure_level";
        m_xAxisStep = 1;
        m_yAxisStep = 1;
        if (m_dataId.IsSameAs("z", false) || m_dataId.IsSameAs("hgt", false)) {
            m_parameter = Geopotential;
            m_parameterName = "Geopotential";
            m_fileVariableName = "z";
            m_unit = m2_s2;
        } else if (m_dataId.IsSameAs("t", false)) {
            m_parameter = AirTemperature;
            m_parameterName = "Temperature";
            m_fileVariableName = "t";
            m_unit = degK;
        } else if (m_dataId.IsSameAs("r", false) || m_dataId.IsSameAs("rh", false)) {
            m_parameter = RelativeHumidity;
            m_parameterName = "Relative humidity";
            m_fileVariableName = "r";
            m_unit = percent;
        } else if (m_dataId.IsSameAs("omega", false) || m_dataId.IsSameAs("w", false)) {
            m_parameter = VerticalVelocity;
            m_parameterName = "Vertical velocity";
            m_fileVariableName = "w";
            m_unit = Pa_s;
        } else {
            asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                              m_dataId, m_product));
        }
        m_fileNamePattern = m_fileVariableName + ".%d.nc";

    } else if (m_product.IsSameAs("surface", false) || m_product.IsSameAs("surf", false) ||
               m_product.IsSameAs("sfc", false)) {
        m_fileStructure.hasLevelDimension = false;
        m_subFolder = "surface";
        m_xAxisStep = 1;
        m_yAxisStep = 1;
        if (m_dataId.IsSameAs("tcw", false)) {
            m_parameter = PrecipitableWater;
            m_parameterName = "Total column water";
            m_fileVariableName = "tcw";
            m_unit = kg_m2;
        } else if (m_dataId.IsSameAs("tp", false)) {
            m_parameter = Precipitation;
            m_parameterName = "Total precipitation";
            m_fileVariableName = "tp";
            m_unit = m;
        } else if (m_dataId.IsSameAs("mslp", false) || m_dataId.IsSameAs("msl", false)) {
            m_parameter = Pressure;
            m_parameterName = "Sea level pressure";
            m_fileVariableName = "msl";
            m_unit = Pa;
        } else {
            asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                              m_dataId, m_product));
        }
        m_fileNamePattern = m_fileVariableName + ".%d.nc";

    } else {
        asThrowException(_("level type not implemented for this reanalysis dataset."));
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

VectorString asDataPredictorArchiveEcmwfCera20C::GetListOfFiles(asTimeArray &timeArray) const
{
    VectorString files;

    for (int i_year = timeArray.GetStartingYear(); i_year <= timeArray.GetEndingYear(); i_year++) {
        files.push_back(GetFullDirectoryPath() + wxString::Format(m_fileNamePattern, i_year));
    }

    return files;
}

bool asDataPredictorArchiveEcmwfCera20C::ExtractFromFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea,
                                                            asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    return ExtractFromNetcdfFile(fileName, dataArea, timeArray, compositeData);
}

double asDataPredictorArchiveEcmwfCera20C::ConvertToMjd(double timeValue, double refValue) const
{
    timeValue = (timeValue / 24.0); // hours to days
    timeValue += asTime::GetMJD(1900, 1, 1); // to MJD: add a negative time span

    return timeValue;
}

