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

#include "asPredictorOperGfsForecast.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>


asPredictorOperGfsForecast::asPredictorOperGfsForecast(const wxString &dataId)
        : asPredictorOper(dataId)
{
    // Set the basic properties.
    m_datasetId = "NWS_GFS_Forecast";
    m_originalProvider = "NWS";
    m_transformedBy = wxEmptyString;
    m_datasetName = "Global Forecast System";
    m_fileType = asFile::Grib2;
    m_forecastLeadTimeStart = 0;
    m_forecastLeadTimeEnd = 240; // After 240h, available in another temporal resolution
    m_forecastLeadTimeStep = 6;
    m_runHourStart = 0;
    m_runUpdate = 6;
    m_strideAllowed = false;
    m_nanValues.push_back(NaNd);
    m_nanValues.push_back(NaNf);
    m_xAxisShift = 0;
    m_yAxisShift = 0;
    m_xAxisStep = 0.5;
    m_yAxisStep = 0.5;
    m_restrictTimeHours = 0;
    m_restrictTimeStepHours = 24;
    m_fileExtension = "grib2";
    m_fileStructure.hasLevelDimension = false;
}

asPredictorOperGfsForecast::~asPredictorOperGfsForecast()
{

}

bool asPredictorOperGfsForecast::Init()
{
    wxConfigBase *pConfig = wxFileConfig::Get();

    // Last element in grib code: level type (http://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_table4-5.shtml)

    // Identify data ID and set the corresponding properties.
    if (m_dataId.IsSameAs("hgt", false)) {
        m_parameter = GeopotentialHeight;
        m_gribCode = {0, 3, 5, 100};
        m_commandDownload = pConfig->Read("/PredictorsUrl/GFS/hgt", m_commandDownload);
        m_fileVariableName = "HGT";
        m_unit = m;
    } else if (m_dataId.IsSameAs("air", false) || m_dataId.IsSameAs("temp", false)) {
        m_parameter = AirTemperature;
        m_gribCode = {0, 0, 0, 100};
        m_commandDownload = pConfig->Read("/PredictorsUrl/GFS/temp", m_commandDownload);
        m_fileVariableName = "TEMP";
        m_unit = degK;
    } else if (m_dataId.IsSameAs("omega", false) || m_dataId.IsSameAs("vvel", false)) {
        m_parameter = VerticalVelocity;
        m_gribCode = {0, 2, 8, 100};
        m_commandDownload = pConfig->Read("/PredictorsUrl/GFS/vvel", m_commandDownload);
        m_fileVariableName = "VVEL";
        m_unit = Pa_s;
    } else if (m_dataId.IsSameAs("rhum", false) || m_dataId.IsSameAs("rh", false)) {
        m_parameter = RelativeHumidity;
        m_gribCode = {0, 1, 1, 100};
        m_commandDownload = pConfig->Read("/PredictorsUrl/GFS/rh", m_commandDownload);
        m_fileVariableName = "RH";
        m_unit = percent;
    } else if (m_dataId.IsSameAs("uwnd", false) || m_dataId.IsSameAs("ugrd", false)) {
        m_parameter = Uwind;
        m_gribCode = {0, 2, 2, 100};
        m_commandDownload = pConfig->Read("/PredictorsUrl/GFS/uwnd", m_commandDownload);
        m_fileVariableName = "UGRD";
        m_unit = m_s;
    } else if (m_dataId.IsSameAs("vwnd", false) || m_dataId.IsSameAs("vgrd", false)) {
        m_parameter = Vwind;
        m_gribCode = {0, 2, 3, 100};
        m_commandDownload = pConfig->Read("/PredictorsUrl/GFS/vwnd", m_commandDownload);
        m_fileVariableName = "VGRD";
        m_unit = m_s;
    } else if (m_dataId.IsSameAs("prwtr", false) || m_dataId.IsSameAs("pwat", false)) {
        m_parameter = PrecipitableWater;
        m_gribCode = {0, 1, 3, 200};
        m_commandDownload = pConfig->Read("/PredictorsUrl/GFS/pwat", m_commandDownload);
        m_fileVariableName = "PWAT";
        m_unit = mm;
    } else {
        asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."), m_dataId,
                                          m_product));
    }

    // Check data ID
    if (m_fileVariableName.IsEmpty()) {
        wxLogError(_("The provided data ID (%s) does not match any possible option in the dataset %s."), m_dataId,
                   m_datasetName);
        return false;
    }

    // Set to initialized
    m_initialized = true;

    return true;
}
