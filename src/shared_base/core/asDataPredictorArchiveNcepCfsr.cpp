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

#include "asDataPredictorArchiveNcepCfsr.h"
#include "asTypeDefs.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileNetcdf.h>


asDataPredictorArchiveNcepCfsr::asDataPredictorArchiveNcepCfsr(const wxString &dataId)
        : asDataPredictorArchive(dataId)
{
    // Set the basic properties.
    m_initialized = false;
    m_datasetId = "NCEP_CFSR";
    m_originalProvider = "NCEP";
    m_datasetName = "CFSR";
    m_originalProviderStart = asTime::GetMJD(1979, 1, 1);
    m_originalProviderEnd = asTime::GetMJD(2011, 3, 1);
    m_timeZoneHours = 0;
    m_timeStepHours = 6;
    m_firstTimeStepHours = NaNDouble;
    m_xAxisShift = 0;
    m_yAxisShift = 0;
}

asDataPredictorArchiveNcepCfsr::~asDataPredictorArchiveNcepCfsr()
{

}

bool asDataPredictorArchiveNcepCfsr::Init()
{
    CheckLevelTypeIsDefined();

    // Last element in grib code: level type (http://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_table4-5.shtml)

    // Identify data ID and set the corresponding properties.
    if (m_product.IsSameAs("pressure_level", false) || m_product.IsSameAs("pressure", false) ||
        m_product.IsSameAs("press", false) || m_product.IsSameAs("pl", false) || m_product.IsSameAs("pgbh", false)  ||
        m_product.IsSameAs("pgb", false)) {
        m_fileStructure.hasLevelDimension = true;
        m_fileStructure.singleLevel = true;
        m_subFolder = "pgbh";
        m_xAxisStep = 0.5;
        m_yAxisStep = 0.5;
        if (m_dataId.IsSameAs("hgt@iso", false)) {
            m_parameter = GeopotentialHeight;
            int arr[] = {0, 3, 5, 100};
            AssignGribCode(arr);
            m_parameterName = "Geopotential height @ Isobaric surface";
            m_unit = gpm;
        } else if (m_dataId.IsSameAs("pwat@eal", false)) {
            m_parameter = PrecipitableWater;
            int arr[] = {0, 1, 3, 200};
            AssignGribCode(arr);
            m_parameterName = "Precipitable water @ Entire atmosphere layer";
            m_unit = kg_m2;
        } else if (m_dataId.IsSameAs("pres@msl", false)) {
            m_parameter = Pressure;
            int arr[] = {0, 3, 0, 101};
            AssignGribCode(arr);
            m_parameterName = "Pressure @ Mean sea level";
            m_unit = Pa;
        } else if (m_dataId.IsSameAs("rh@iso", false)) {
            m_parameter = RelativeHumidity;
            int arr[] = {0, 1, 1, 100};
            AssignGribCode(arr);
            m_parameterName = "Relative humidity @ Isobaric surface";
            m_unit = percent;
        } else if (m_dataId.IsSameAs("tmp@iso", false)) {
            m_parameter = AirTemperature;
            int arr[] = {0, 0, 0, 100};
            AssignGribCode(arr);
            m_parameterName = "Temperature @ Isobaric surface";
            m_unit = degK;
        } else {
            asThrowException(wxString::Format(_("Parameter '%s' not implemented yet."), m_dataId));
        }
        m_fileNamePattern = "%4d/%4d%02d/%4d%02d%02d/pgbhnl.gdas.%4d%02d%02d%02d.grb2";

    } else if (m_product.IsSameAs("isentropic_level", false) || m_product.IsSameAs("isentropic", false) ||
               m_product.IsSameAs("ipvh", false) || m_product.IsSameAs("ipv", false)) {
        asThrowException(_("Isentropic levels for CFSR are not implemented yet."));

    } else if (m_product.IsSameAs("surface_fluxes", false) || m_product.IsSameAs("fluxes", false) ||
               m_product.IsSameAs("flx", false)) {
        asThrowException(_("Surface fluxes grids for CFSR are not implemented yet."));

    } else {
        asThrowException(_("level type not implemented for this reanalysis dataset."));
    }

    // Check data ID
    if (m_fileNamePattern.IsEmpty() || m_gribCode[2] == asNOT_FOUND) {
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

    wxASSERT(m_gribCode.size()==4);

    // Set to initialized
    m_initialized = true;

    return true;
}

VectorString asDataPredictorArchiveNcepCfsr::GetListOfFiles(asTimeArray &timeArray) const
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

bool asDataPredictorArchiveNcepCfsr::ExtractFromFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea,
                                                            asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    return ExtractFromGribFile(fileName, dataArea, timeArray, compositeData);
}
