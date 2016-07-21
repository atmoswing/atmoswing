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

#include "asDataPredictorArchiveNcepCfsr2.h"
#include "asTypeDefs.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileNetcdf.h>


asDataPredictorArchiveNcepCfsr2::asDataPredictorArchiveNcepCfsr2(const wxString &dataId)
        : asDataPredictorArchive(dataId)
{
    // Set the basic properties.
    m_initialized = false;
    m_datasetId = "NCEP_CFSR_v2";
    m_originalProvider = "NCEP";
    m_datasetName = "CFSR 2";
    m_originalProviderStart = asTime::GetMJD(1979, 1, 1);
    m_originalProviderEnd = asTime::GetMJD(2011, 3, 1);
    m_timeZoneHours = 0;
    m_timeStepHours = 6;
    m_firstTimeStepHours = NaNDouble;
    m_xAxisShift = 0;
    m_yAxisShift = 0;
}

asDataPredictorArchiveNcepCfsr2::~asDataPredictorArchiveNcepCfsr2()
{

}

bool asDataPredictorArchiveNcepCfsr2::Init()
{
    CheckLevelTypeIsDefined();

    // Identify data ID and set the corresponding properties.
    switch (m_levelType) {
        case PressureLevel:
            m_fileStructure.hasLevelDimension = true;
            m_fileStructure.singleLevel = true;
            m_subFolder = "pgbh";
            m_xAxisStep = 0.5;
            m_yAxisStep = 0.5;
            if (m_dataId.IsSameAs("hgt", false)) {
                m_parameter = GeopotentialHeight;
                m_gribParameterDiscipline = 0;
                m_gribParameterCategory = 3;
                m_gribParameterNum = 5;
                m_parameterName = "Geopotential height";
                m_unit = gpm;
                m_fileStructure.dimLevelName = "isobaric";
            } else {
                asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                                  m_dataId, LevelEnumToString(m_levelType)));
            }
            m_fileNamePattern = "%4d/%4d%02d/%4d%02d%02d/pgbhnl.gdas.%4d%02d%02d%02d.grb2";
            break;

        default: asThrowException(_("level type not implemented for this reanalysis dataset."));
    }

    // Check data ID
    if (m_fileNamePattern.IsEmpty() || m_gribParameterNum == asNOT_FOUND) {
        asLogError(wxString::Format(_("The provided data ID (%s) does not match any possible option in the dataset %s."),
                                 m_dataId, m_datasetName));
        return false;
    }

    // Check directory is set
    if (GetDirectoryPath().IsEmpty()) {
        asLogError(wxString::Format(_("The path to the directory has not been set for the data %s from the dataset %s."),
                                 m_dataId, m_datasetName));
        return false;
    }

    // Set to initialized
    m_initialized = true;

    return true;
}

VectorString asDataPredictorArchiveNcepCfsr2::GetListOfFiles(asTimeArray &timeArray) const
{
    VectorString files;
    Array1DDouble tArray = timeArray.GetTimeArray();

    for (int i = 0; i < tArray.size(); i++) {
        TimeStruct t = asTime::GetTimeStruct(tArray[i]);
        files.push_back(GetFullDirectoryPath() + wxString::Format(m_fileNamePattern, t.year, t.year, t.month, t.year,
                                                                  t.month, t.day, t.year, t.month, t.day, t.hour));
    }

    return files;
}

bool asDataPredictorArchiveNcepCfsr2::ExtractFromFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea,
                                                            asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    return ExtractFromGribFile(fileName, dataArea, timeArray, compositeData);
}
