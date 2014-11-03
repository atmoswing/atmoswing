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
 * The Original Software is AtmoSwing. The Initial Developer of the
 * Original Software is Pascal Horton of the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 * Portions Copyright 2014 Pascal Horton, Terr@num.
 */

#include "asDataPredictorArchiveNcepReanalysis1Lthe.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileNetcdf.h>


asDataPredictorArchiveNcepReanalysis1Lthe::asDataPredictorArchiveNcepReanalysis1Lthe(const wxString &dataId)
:
asDataPredictorArchiveNcepReanalysis1Terranum(dataId)
{
    // Set the basic properties.
    m_Initialized = false;
    m_DataId = dataId;
    m_DatasetId = "NCEP_Reanalysis_v1_lthe";
    m_OriginalProvider = "NCEP/NCAR";
    m_FinalProvider = "LTHE";
    m_FinalProviderWebsite = "http://www.lthe.fr";
    m_FinalProviderFTP = wxEmptyString;
    m_DatasetName = "Reanalysis 1 subset from LTHE";
    m_OriginalProviderStart = asTime::GetMJD(1948, 1, 1);
    m_OriginalProviderEnd = NaNDouble;
    m_TimeZoneHours = 0;
    m_TimeStepHours = 24;
    m_FirstTimeStepHours = 0;
    m_NanValues.push_back(32767);
    m_NanValues.push_back(936*std::pow(10.f,34.f));
    m_CoordinateSystem = WGS84;
    m_UaxisShift = 0;
    m_VaxisShift = 0;
    m_UaxisStep = 2.5;
    m_VaxisStep = 2.5;
    m_SubFolder = wxEmptyString;
    m_FileAxisLatName = "lat";
    m_FileAxisLonName = "lon";
    m_FileAxisTimeName = "time";
    m_FileAxisLevelName = "level";

    // Identify data ID and set the corresponding properties.
    if (m_DataId.IsSameAs("hgt_500hPa", false))
    {
        m_DataParameter = GeopotentialHeight;
        m_FileNamePattern = "NCEP_Reanalysis_v1_lthe_hgt_500hPa.nc";
        m_FileVariableName = "hgt";
        m_Unit = m;
        m_FirstTimeStepHours = 0;
        m_TimeStepHours = 24;
    }
    else if (m_DataId.IsSameAs("hgt_1000hPa", false))
    {
        m_DataParameter = GeopotentialHeight;
        m_FileNamePattern = "NCEP_Reanalysis_v1_lthe_hgt_1000hPa.nc";
        m_FileVariableName = "hgt";
        m_Unit = m;
        m_FirstTimeStepHours = 12;
        m_TimeStepHours = 24;
    }
    else if (m_DataId.IsSameAs("prwtr", false))
    {
        m_DataParameter = GeopotentialHeight;
        m_FileNamePattern = "NCEP_Reanalysis_v1_lthe_prwtr.nc";
        m_FileVariableName = "pwa";
        m_Unit = m;
        m_FirstTimeStepHours = 0;
        m_TimeStepHours = 12;
    }
    else if (m_DataId.IsSameAs("rhum", false))
    {
        m_DataParameter = GeopotentialHeight;
        m_FileNamePattern = "NCEP_Reanalysis_v1_lthe_rhum.nc";
        m_FileVariableName = "rhum";
        m_Unit = m;
        m_FirstTimeStepHours = 0;
        m_TimeStepHours = 12;
    }
    else
    {
        m_DataParameter = NoDataParameter;
        m_FileNamePattern = wxEmptyString;
        m_FileVariableName = wxEmptyString;
        m_Unit = NoDataUnit;
    }
}

asDataPredictorArchiveNcepReanalysis1Lthe::~asDataPredictorArchiveNcepReanalysis1Lthe()
{

}

bool asDataPredictorArchiveNcepReanalysis1Lthe::Init()
{
    // Check data ID
    if (m_FileNamePattern.IsEmpty() || m_FileVariableName.IsEmpty()) {
        asLogError(wxString::Format(_("The provided data ID (%s) does not match any possible option in the dataset %s."), m_DataId.c_str(), m_DatasetName.c_str()));
        return false;
    }

    // Check directory is set
    if (m_DirectoryPath.IsEmpty()) {
        asLogError(wxString::Format(_("The path to the directory has not been set for the data %s from the dataset %s."), m_DataId.c_str(), m_DatasetName.c_str()));
        return false;
    }

    // Set to initialized
    m_Initialized = true;

    return true;
}

VectorString asDataPredictorArchiveNcepReanalysis1Lthe::GetDataIdList()
{
    VectorString list;

    list.push_back("hgt_500hPa_24h"); // Geopotential Height at 500 hPa & 24 h
    list.push_back("hgt_1000hPa_12h"); // Geopotential Height at 1000 hPa & 12 h
    list.push_back("prwtr_12h"); // Precipitable Water at 12 h
    list.push_back("rhum_12h"); // Relative Humidity at 12 h

    return list;
}

VectorString asDataPredictorArchiveNcepReanalysis1Lthe::GetDataIdDescriptionList()
{
    VectorString list;

    list.push_back("Geopotential Height at 500 hPa & 24 h");
    list.push_back("Geopotential Height at 1000 hPa & 12 h");
    list.push_back("Precipitable Water at 12 h");
    list.push_back("Relative Humidity at 12 h");

    return list;
}

