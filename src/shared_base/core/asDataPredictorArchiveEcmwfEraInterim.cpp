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

#include "asDataPredictorArchiveEcmwfEraInterim.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileNetcdf.h>


asDataPredictorArchiveEcmwfEraInterim::asDataPredictorArchiveEcmwfEraInterim(const wxString &dataId)
        : asDataPredictorArchiveNcepReanalysis1Terranum(dataId)
{
    // Set the basic properties.
    m_initialized = false;
    m_dataId = dataId;
    m_datasetId = "ECMWF_ERA-40";
    m_originalProvider = "ECMWF";
    m_finalProvider = "ECMWF";
    m_finalProviderWebsite = "";
    m_finalProviderFTP = wxEmptyString;
    m_datasetName = "ERA-40";
    m_originalProviderStart = asTime::GetMJD(1957, 9, 1);
    m_originalProviderEnd = asTime::GetMJD(2002, 8, 31);
    m_timeZoneHours = 0;
    m_timeStepHours = 6;
    m_firstTimeStepHours = 0;
    m_nanValues.push_back(-32767);
    m_xAxisShift = 0;
    m_yAxisShift = 0;
    m_subFolder = wxEmptyString;
    m_fileAxisLatName = "latitude";
    m_fileAxisLonName = "longitude";
    m_fileAxisTimeName = "time";
    m_fileAxisLevelName = "level";

    // The axis steps are defined here for default grids and will be overridden by real data resolution.
    m_xAxisStep = 2.5;
    m_yAxisStep = 2.5;



/*
    Potential temparature
    Potential vorticity
    Pressure levels
    Surface
*/



    // Identify data ID and set the corresponding properties.
    if (m_dataId.IsSameAs("hgt", false)) {
        m_dataParameter = GeopotentialHeight;
        m_fileNamePattern = "ECMWF_ERA40_hgt.nc";
        m_fileVariableName = "z";
        m_unit = m;
    }
    /*
    else if (m_dataId.IsSameAs("air", false))
    {
        m_parameter = AirTemperature;
        m_fileNamePattern = "air.%d.nc";
        m_fileVariableName = "air";
        m_unit = degK;
    }
    else if (m_dataId.IsSameAs("omega", false))
    {
        m_parameter = Omega;
        m_subFolder = "pressure";
        m_fileNamePattern = "omega.%d.nc";
        m_fileVariableName = "omega";
        m_unit = PascalsPerSec;
    }
    else if (m_dataId.IsSameAs("rhum", false))
    {
        m_parameter = RelativeHumidity;
        m_subFolder = "pressure";
        m_fileNamePattern = "rhum.%d.nc";
        m_fileVariableName = "rhum";
        m_unit = percent;
    }
    else if (m_dataId.IsSameAs("shum", false))
    {
        m_parameter = SpecificHumidity;
        m_subFolder = "pressure";
        m_fileNamePattern = "shum.%d.nc";
        m_fileVariableName = "shum";
        m_unit = kgPerKg;
    }
    else if (m_dataId.IsSameAs("uwnd", false))
    {
        m_parameter = Uwind;
        m_subFolder = "pressure";
        m_fileNamePattern = "uwnd.%d.nc";
        m_fileVariableName = "uwnd";
        m_unit = mPerSec;
    }
    else if (m_dataId.IsSameAs("vwnd", false))
    {
        m_parameter = Vwind;
        m_subFolder = "pressure";
        m_fileNamePattern = "vwnd.%d.nc";
        m_fileVariableName = "vwnd";
        m_unit = mPerSec;
    }
    else if (m_dataId.IsSameAs("surf_air", false))
    {
        m_parameter = AirTemperature;
        m_subFolder = "surface";
        m_fileNamePattern = "air.sig995.%d.nc";
        m_fileVariableName = "air";
        m_unit = degK;
    }
    else if (m_dataId.IsSameAs("surf_lftx", false))
    {
        m_parameter = SurfaceLiftedIndex;
        m_subFolder = "surface";
        m_fileNamePattern = "lftx.sfc.%d.nc";
        m_fileVariableName = "lftx";
        m_unit = degK;
    }
    else if (m_dataId.IsSameAs("surf_lftx4", false))
    {
        m_parameter = SurfaceLiftedIndex;
        m_subFolder = "surface";
        m_fileNamePattern = "lftx4.sfc.%d.nc";
        m_fileVariableName = "lftx4";
        m_unit = degK;
    }
    else if (m_dataId.IsSameAs("surf_omega", false))
    {
        m_parameter = Omega;
        m_subFolder = "surface";
        m_fileNamePattern = "omega.sig995.%d.nc";
        m_fileVariableName = "omega";
        m_unit = PascalsPerSec;
    }
    else if (m_dataId.IsSameAs("surf_pottmp", false))
    {
        m_parameter = PotentialTemperature;
        m_subFolder = "surface";
        m_fileNamePattern = "pottmp.sig995.%d.nc";
        m_fileVariableName = "pottmp";
        m_unit = degK;
    }
    else if (m_dataId.IsSameAs("surf_prwtr", false))
    {
        m_parameter = PrecipitableWater;
        m_subFolder = "surface";
        m_fileNamePattern = "pr_wtr.eatm.%d.nc";
        m_fileVariableName = "pr_wtr";
        m_unit = mm;
    }
    else if (m_dataId.IsSameAs("surf_pres", false))
    {
        m_parameter = Pressure;
        m_subFolder = "surface";
        m_fileNamePattern = "pres.sfc.%d.nc";
        m_fileVariableName = "pres";
        m_unit = Pascals;
    }
    else if (m_dataId.IsSameAs("surf_rhum", false))
    {
        m_parameter = RelativeHumidity;
        m_subFolder = "surface";
        m_fileNamePattern = "rhum.sig995.%d.nc";
        m_fileVariableName = "rhum";
        m_unit = percent;
    }
    else if (m_dataId.IsSameAs("surf_slp", false))
    {
        m_parameter = Pressure;
        m_subFolder = "surface";
        m_fileNamePattern = "slp.%d.nc";
        m_fileVariableName = "slp";
        m_unit = Pascals;
    }
    else if (m_dataId.IsSameAs("surf_uwnd", false))
    {
        m_parameter = Uwind;
        m_subFolder = "surface";
        m_fileNamePattern = "uwnd.sig995.%d.nc";
        m_fileVariableName = "uwnd";
        m_unit = mPerSec;
    }
    else if (m_dataId.IsSameAs("surf_vwnd", false))
    {
        m_parameter = Vwind;
        m_subFolder = "surface";
        m_fileNamePattern = "vwnd.sig995.%d.nc";
        m_fileVariableName = "vwnd";
        m_unit = mPerSec;
    }
    else
    {
        m_xAxisStep = NaNFloat;
        m_yAxisStep = NaNFloat;

        if (m_dataId.IsSameAs("flux_air2m", false))
        {
            m_parameter = AirTemperature;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "air.2m.gauss.%d.nc";
            m_fileVariableName = "air";
            m_unit = degK;
        }
        else if (m_dataId.IsSameAs("flux_pevpr", false))
        {
            m_parameter = PotentialEvaporation;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "pevpr.sfc.gauss.%d.nc";
            m_fileVariableName = "pevpr";
            m_unit = WPerm2;
        }
        else if (m_dataId.IsSameAs("flux_shum2m", false))
        {
            m_parameter = SpecificHumidity;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "shum.2m.gauss.%d.nc";
            m_fileVariableName = "shum";
            m_unit = kgPerKg;
        }
        else if (m_dataId.IsSameAs("flux_sktmp", false))
        {
            m_parameter = SurfaceTemperature;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "skt.sfc.gauss.%d.nc";
            m_fileVariableName = "skt";
            m_unit = degK;
        }
        else if (m_dataId.IsSameAs("flux_tmp0-10", false))
        {
            m_parameter = SurfaceTemperature;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "tmp.0-10cm.gauss.%d.nc";
            m_fileVariableName = "tmp";
            m_unit = degK;
        }
        else if (m_dataId.IsSameAs("flux_tmp10-200", false))
        {
            m_parameter = SurfaceTemperature;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "tmp.10-200cm.gauss.%d.nc";
            m_fileVariableName = "tmp";
            m_unit = degK;
        }
        else if (m_dataId.IsSameAs("flux_tmp300", false))
        {
            m_parameter = SurfaceTemperature;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "tmp.300cm.gauss.%d.nc";
            m_fileVariableName = "tmp";
            m_unit = degK;
        }
        else if (m_dataId.IsSameAs("flux_uwnd10m", false))
        {
            m_parameter = Uwind;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "uwnd.10m.gauss.%d.nc";
            m_fileVariableName = "uwnd";
            m_unit = mPerSec;
        }
        else if (m_dataId.IsSameAs("flux_vwnd10m", false))
        {
            m_parameter = Vwind;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "vwnd.10m.gauss.%d.nc";
            m_fileVariableName = "vwnd";
            m_unit = mPerSec;
        }
        else if (m_dataId.IsSameAs("flux_cprat", false))
        {
            m_parameter = ConvectivePrecipitation;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "cprat.sfc.gauss.%d.nc";
            m_fileVariableName = "cprat";
            m_unit = kgPerm2Pers;
        }
        else if (m_dataId.IsSameAs("flux_dlwrf", false))
        {
            m_parameter = LongwaveRadiation;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "dlwrf.sfc.gauss.%d.nc";
            m_fileVariableName = "dlwrf";
            m_unit = WPerm2;
        }
        else if (m_dataId.IsSameAs("flux_dswrf", false))
        {
            m_parameter = SolarRadiation;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "dswrf.sfc.gauss.%d.nc";
            m_fileVariableName = "dswrf";
            m_unit = WPerm2;
        }
        else if (m_dataId.IsSameAs("flux_gflux", false))
        {
            m_parameter = GroundHeatFlux;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "gflux.sfc.gauss.%d.nc";
            m_fileVariableName = "gflux";
            m_unit = WPerm2;
        }
        else if (m_dataId.IsSameAs("flux_lhtfl", false))
        {
            m_parameter = LatentHeatFlux;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "lhtfl.sfc.gauss.%d.nc";
            m_fileVariableName = "lhtfl";
            m_unit = WPerm2;
        }
        else if (m_dataId.IsSameAs("flux_nbdsf", false))
        {
            m_parameter = NearIRFlux;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "nbdsf.sfc.gauss.%d.nc";
            m_fileVariableName = "nbdsf";
            m_unit = WPerm2;
        }
        else if (m_dataId.IsSameAs("flux_nddsf", false))
        {
            m_parameter = NearIRFlux;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "nddsf.sfc.gauss.%d.nc";
            m_fileVariableName = "nddsf";
            m_unit = WPerm2;
        }
        else if (m_dataId.IsSameAs("flux_nlwrs", false))
        {
            m_parameter = LongwaveRadiation;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "nlwrs.sfc.gauss.%d.nc";
            m_fileVariableName = "nlwrs";
            m_unit = WPerm2;
        }
        else if (m_dataId.IsSameAs("flux_nswrs", false))
        {
            m_parameter = ShortwaveRadiation;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "nswrs.sfc.gauss.%d.nc";
            m_fileVariableName = "nswrs";
            m_unit = WPerm2;
        }
        else if (m_dataId.IsSameAs("flux_prate", false))
        {
            m_parameter = Precipitation;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "prate.sfc.gauss.%d.nc";
            m_fileVariableName = "prate";
            m_unit = kgPerm2Pers;
        }
        else if (m_dataId.IsSameAs("flux_shtfl", false))
        {
            m_parameter = SensibleHeatFlux;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "shtfl.sfc.gauss.%d.nc";
            m_fileVariableName = "shtfl";
            m_unit = WPerm2;
        }
        else
        {
            m_parameter = NoDataParameter;
            m_subFolder = wxEmptyString;
            m_fileNamePattern = wxEmptyString;
            m_fileVariableName = wxEmptyString;
            m_unit = NoDataUnit;
        }
    }*/

}

asDataPredictorArchiveEcmwfEraInterim::~asDataPredictorArchiveEcmwfEraInterim()
{

}

bool asDataPredictorArchiveEcmwfEraInterim::Init()
{
    // Check data ID
    if (m_fileNamePattern.IsEmpty() || m_fileVariableName.IsEmpty()) {
        asLogError(
                wxString::Format(_("The provided data ID (%s) does not match any possible option in the dataset %s."),
                                 m_dataId, m_datasetName));
        return false;
    }

    // Check directory is set
    if (m_directoryPath.IsEmpty()) {
        asLogError(
                wxString::Format(_("The path to the directory has not been set for the data %s from the dataset %s."),
                                 m_dataId, m_datasetName));
        return false;
    }

    // Set to initialized
    m_initialized = true;

    return true;
}

