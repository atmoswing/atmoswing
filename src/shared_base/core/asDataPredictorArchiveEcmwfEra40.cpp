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
 * Portions Copyright 2014 Pascal Horton, Terr@num.
 */

#include "asDataPredictorArchiveEcmwfEra40.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileNetcdf.h>


asDataPredictorArchiveEcmwfEra40::asDataPredictorArchiveEcmwfEra40(const wxString &dataId)
:
asDataPredictorArchiveNcepReanalysis1Terranum(dataId)
{
    // Set the basic properties.
    m_initialized = false;
    m_dataId = dataId;
    m_datasetId = "ECMWF_ERA-40";
    m_originalProvider = "ECMWF";
    m_finalProvider = "ECMWF";
    m_finalProviderWebsite = "http://apps.ecmwf.int/datasets/data/era40_daily/";
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

    // Identify data ID and set the corresponding properties.
    if (m_dataId.IsSameAs("hgt", false))
    {
        m_dataParameter = GeopotentialHeight;
        m_fileNamePattern = "ECMWF_ERA40_hgt.nc";
        m_fileVariableName = "z";
        m_unit = m;
    }
    /*
    else if (m_dataId.IsSameAs("air", false))
    {
        m_dataParameter = AirTemperature;
        m_fileNamePattern = "air.%d.nc";
        m_fileVariableName = "air";
        m_unit = degK;
    }
    else if (m_dataId.IsSameAs("omega", false))
    {
        m_dataParameter = Omega;
        m_subFolder = "pressure";
        m_fileNamePattern = "omega.%d.nc";
        m_fileVariableName = "omega";
        m_unit = PascalsPerSec;
    }
    else if (m_dataId.IsSameAs("rhum", false))
    {
        m_dataParameter = RelativeHumidity;
        m_subFolder = "pressure";
        m_fileNamePattern = "rhum.%d.nc";
        m_fileVariableName = "rhum";
        m_unit = percent;
    }
    else if (m_dataId.IsSameAs("shum", false))
    {
        m_dataParameter = SpecificHumidity;
        m_subFolder = "pressure";
        m_fileNamePattern = "shum.%d.nc";
        m_fileVariableName = "shum";
        m_unit = kgPerKg;
    }
    else if (m_dataId.IsSameAs("uwnd", false))
    {
        m_dataParameter = Uwind;
        m_subFolder = "pressure";
        m_fileNamePattern = "uwnd.%d.nc";
        m_fileVariableName = "uwnd";
        m_unit = mPerSec;
    }
    else if (m_dataId.IsSameAs("vwnd", false))
    {
        m_dataParameter = Vwind;
        m_subFolder = "pressure";
        m_fileNamePattern = "vwnd.%d.nc";
        m_fileVariableName = "vwnd";
        m_unit = mPerSec;
    }
    else if (m_dataId.IsSameAs("surf_air", false))
    {
        m_dataParameter = AirTemperature;
        m_subFolder = "surface";
        m_fileNamePattern = "air.sig995.%d.nc";
        m_fileVariableName = "air";
        m_unit = degK;
    }
    else if (m_dataId.IsSameAs("surf_lftx", false))
    {
        m_dataParameter = SurfaceLiftedIndex;
        m_subFolder = "surface";
        m_fileNamePattern = "lftx.sfc.%d.nc";
        m_fileVariableName = "lftx";
        m_unit = degK;
    }
    else if (m_dataId.IsSameAs("surf_lftx4", false))
    {
        m_dataParameter = SurfaceLiftedIndex;
        m_subFolder = "surface";
        m_fileNamePattern = "lftx4.sfc.%d.nc";
        m_fileVariableName = "lftx4";
        m_unit = degK;
    }
    else if (m_dataId.IsSameAs("surf_omega", false))
    {
        m_dataParameter = Omega;
        m_subFolder = "surface";
        m_fileNamePattern = "omega.sig995.%d.nc";
        m_fileVariableName = "omega";
        m_unit = PascalsPerSec;
    }
    else if (m_dataId.IsSameAs("surf_pottmp", false))
    {
        m_dataParameter = PotentialTemperature;
        m_subFolder = "surface";
        m_fileNamePattern = "pottmp.sig995.%d.nc";
        m_fileVariableName = "pottmp";
        m_unit = degK;
    }
    else if (m_dataId.IsSameAs("surf_prwtr", false))
    {
        m_dataParameter = PrecipitableWater;
        m_subFolder = "surface";
        m_fileNamePattern = "pr_wtr.eatm.%d.nc";
        m_fileVariableName = "pr_wtr";
        m_unit = mm;
    }
    else if (m_dataId.IsSameAs("surf_pres", false))
    {
        m_dataParameter = Pressure;
        m_subFolder = "surface";
        m_fileNamePattern = "pres.sfc.%d.nc";
        m_fileVariableName = "pres";
        m_unit = Pascals;
    }
    else if (m_dataId.IsSameAs("surf_rhum", false))
    {
        m_dataParameter = RelativeHumidity;
        m_subFolder = "surface";
        m_fileNamePattern = "rhum.sig995.%d.nc";
        m_fileVariableName = "rhum";
        m_unit = percent;
    }
    else if (m_dataId.IsSameAs("surf_slp", false))
    {
        m_dataParameter = Pressure;
        m_subFolder = "surface";
        m_fileNamePattern = "slp.%d.nc";
        m_fileVariableName = "slp";
        m_unit = Pascals;
    }
    else if (m_dataId.IsSameAs("surf_uwnd", false))
    {
        m_dataParameter = Uwind;
        m_subFolder = "surface";
        m_fileNamePattern = "uwnd.sig995.%d.nc";
        m_fileVariableName = "uwnd";
        m_unit = mPerSec;
    }
    else if (m_dataId.IsSameAs("surf_vwnd", false))
    {
        m_dataParameter = Vwind;
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
            m_dataParameter = AirTemperature;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "air.2m.gauss.%d.nc";
            m_fileVariableName = "air";
            m_unit = degK;
        }
        else if (m_dataId.IsSameAs("flux_pevpr", false))
        {
            m_dataParameter = PotentialEvaporation;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "pevpr.sfc.gauss.%d.nc";
            m_fileVariableName = "pevpr";
            m_unit = WPerm2;
        }
        else if (m_dataId.IsSameAs("flux_shum2m", false))
        {
            m_dataParameter = SpecificHumidity;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "shum.2m.gauss.%d.nc";
            m_fileVariableName = "shum";
            m_unit = kgPerKg;
        }
        else if (m_dataId.IsSameAs("flux_sktmp", false))
        {
            m_dataParameter = SurfaceTemperature;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "skt.sfc.gauss.%d.nc";
            m_fileVariableName = "skt";
            m_unit = degK;
        }
        else if (m_dataId.IsSameAs("flux_tmp0-10", false))
        {
            m_dataParameter = SurfaceTemperature;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "tmp.0-10cm.gauss.%d.nc";
            m_fileVariableName = "tmp";
            m_unit = degK;
        }
        else if (m_dataId.IsSameAs("flux_tmp10-200", false))
        {
            m_dataParameter = SurfaceTemperature;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "tmp.10-200cm.gauss.%d.nc";
            m_fileVariableName = "tmp";
            m_unit = degK;
        }
        else if (m_dataId.IsSameAs("flux_tmp300", false))
        {
            m_dataParameter = SurfaceTemperature;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "tmp.300cm.gauss.%d.nc";
            m_fileVariableName = "tmp";
            m_unit = degK;
        }
        else if (m_dataId.IsSameAs("flux_uwnd10m", false))
        {
            m_dataParameter = Uwind;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "uwnd.10m.gauss.%d.nc";
            m_fileVariableName = "uwnd";
            m_unit = mPerSec;
        }
        else if (m_dataId.IsSameAs("flux_vwnd10m", false))
        {
            m_dataParameter = Vwind;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "vwnd.10m.gauss.%d.nc";
            m_fileVariableName = "vwnd";
            m_unit = mPerSec;
        }
        else if (m_dataId.IsSameAs("flux_cprat", false))
        {
            m_dataParameter = ConvectivePrecipitation;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "cprat.sfc.gauss.%d.nc";
            m_fileVariableName = "cprat";
            m_unit = kgPerm2Pers;
        }
        else if (m_dataId.IsSameAs("flux_dlwrf", false))
        {
            m_dataParameter = LongwaveRadiation;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "dlwrf.sfc.gauss.%d.nc";
            m_fileVariableName = "dlwrf";
            m_unit = WPerm2;
        }
        else if (m_dataId.IsSameAs("flux_dswrf", false))
        {
            m_dataParameter = SolarRadiation;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "dswrf.sfc.gauss.%d.nc";
            m_fileVariableName = "dswrf";
            m_unit = WPerm2;
        }
        else if (m_dataId.IsSameAs("flux_gflux", false))
        {
            m_dataParameter = GroundHeatFlux;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "gflux.sfc.gauss.%d.nc";
            m_fileVariableName = "gflux";
            m_unit = WPerm2;
        }
        else if (m_dataId.IsSameAs("flux_lhtfl", false))
        {
            m_dataParameter = LatentHeatFlux;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "lhtfl.sfc.gauss.%d.nc";
            m_fileVariableName = "lhtfl";
            m_unit = WPerm2;
        }
        else if (m_dataId.IsSameAs("flux_nbdsf", false))
        {
            m_dataParameter = NearIRFlux;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "nbdsf.sfc.gauss.%d.nc";
            m_fileVariableName = "nbdsf";
            m_unit = WPerm2;
        }
        else if (m_dataId.IsSameAs("flux_nddsf", false))
        {
            m_dataParameter = NearIRFlux;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "nddsf.sfc.gauss.%d.nc";
            m_fileVariableName = "nddsf";
            m_unit = WPerm2;
        }
        else if (m_dataId.IsSameAs("flux_nlwrs", false))
        {
            m_dataParameter = LongwaveRadiation;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "nlwrs.sfc.gauss.%d.nc";
            m_fileVariableName = "nlwrs";
            m_unit = WPerm2;
        }
        else if (m_dataId.IsSameAs("flux_nswrs", false))
        {
            m_dataParameter = ShortwaveRadiation;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "nswrs.sfc.gauss.%d.nc";
            m_fileVariableName = "nswrs";
            m_unit = WPerm2;
        }
        else if (m_dataId.IsSameAs("flux_prate", false))
        {
            m_dataParameter = Precipitation;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "prate.sfc.gauss.%d.nc";
            m_fileVariableName = "prate";
            m_unit = kgPerm2Pers;
        }
        else if (m_dataId.IsSameAs("flux_shtfl", false))
        {
            m_dataParameter = SensibleHeatFlux;
            m_subFolder = "surface_gauss";
            m_fileNamePattern = "shtfl.sfc.gauss.%d.nc";
            m_fileVariableName = "shtfl";
            m_unit = WPerm2;
        }
        else
        {
            m_dataParameter = NoDataParameter;
            m_subFolder = wxEmptyString;
            m_fileNamePattern = wxEmptyString;
            m_fileVariableName = wxEmptyString;
            m_unit = NoDataUnit;
        }
    }*/

}

asDataPredictorArchiveEcmwfEra40::~asDataPredictorArchiveEcmwfEra40()
{

}

bool asDataPredictorArchiveEcmwfEra40::Init()
{
    // Check data ID
    if (m_fileNamePattern.IsEmpty() || m_fileVariableName.IsEmpty()) {
        asLogError(wxString::Format(_("The provided data ID (%s) does not match any possible option in the dataset %s."), m_dataId.c_str(), m_datasetName.c_str()));
        return false;
    }

    // Check directory is set
    if (m_directoryPath.IsEmpty()) {
        asLogError(wxString::Format(_("The path to the directory has not been set for the data %s from the dataset %s."), m_dataId.c_str(), m_datasetName.c_str()));
        return false;
    }

    // Set to initialized
    m_initialized = true;

    return true;
}

VectorString asDataPredictorArchiveEcmwfEra40::GetDataIdList()
{
    VectorString list;

    list.push_back("z"); // Geopotential Height
   /* list.push_back("air"); // Air Temperature
    list.push_back("omega"); // Omega (Vertical Velocity)
    list.push_back("rhum"); // Relative Humidity
    list.push_back("shum"); // Specific Humidity
    list.push_back("uwnd"); // U-Wind
    list.push_back("vwnd"); // V-Wind
    list.push_back("surf_air"); // Air Temperature at sigma level 995
    list.push_back("surf_lftx"); // Surface lifted index
    list.push_back("surf_lftx4"); // Best (4-layer) lifted index
    list.push_back("surf_omega"); // Omega (Vertical Velocity)
    list.push_back("surf_pottmp"); // Potential Temperature at sigma level 995
    list.push_back("surf_prwtr"); // Precipitable Water
    list.push_back("surf_pres"); // Pressure
    list.push_back("surf_rhum"); // Relative Humidity at sigma level 995
    list.push_back("surf_slp"); // Sea Level Pressure" enable="1
    list.push_back("surf_uwnd"); // U-Wind at sigma level 995
    list.push_back("surf_vwnd"); // V-Wind at sigma level 995
    list.push_back("flux_air2m"); // Air Temperature 2m
    list.push_back("flux_pevpr"); // Potential evaporation rate
    list.push_back("flux_shum2m"); // Specific humidity at 2m
    list.push_back("flux_sktmp"); // Skin Temperature
    list.push_back("flux_tmp0-10"); // Temperature of 0-10cm layer
    list.push_back("flux_tmp10-200"); // Temperature of 10-200cm layer
    list.push_back("flux_tmp300"); // Temperature at 300cm
    list.push_back("flux_uwnd10m"); // U-wind at 10m
    list.push_back("flux_vwnd10m"); // V-wind at 10m
    list.push_back("flux_cprat"); // Convective precipitation rate
    list.push_back("flux_dlwrf"); // Downward longwave radiation flux
    list.push_back("flux_dswrf"); // Downward solar radiation flux
    list.push_back("flux_gflux"); // Ground heat flux
    list.push_back("flux_lhtfl"); // Latent heat net flux
    list.push_back("flux_nbdsf"); // Near IR beam downward solar flux
    list.push_back("flux_nddsf"); // Near IR diffuse downward solar flux
    list.push_back("flux_nlwrs"); // Net longwave radiation
    list.push_back("flux_nswrs"); // Net shortwave radiation
    list.push_back("flux_prate"); // Precipitation rate
    list.push_back("flux_shtfl"); // Sensible heat net flux
    */
    return list;
}

VectorString asDataPredictorArchiveEcmwfEra40::GetDataIdDescriptionList()
{
    VectorString list;

    list.push_back("Geopotential Height");
    /*list.push_back("Air Temperature");
    list.push_back("Omega (Vertical Velocity)");
    list.push_back("Relative Humidity");
    list.push_back("Specific Humidity");
    list.push_back("U-Wind");
    list.push_back("V-Wind");
    list.push_back("Air Temperature at sigma level 995");
    list.push_back("Surface lifted index");
    list.push_back("Best (4-layer) lifted index");
    list.push_back("Omega (Vertical Velocity)");
    list.push_back("Potential Temperature at sigma level 995");
    list.push_back("Precipitable Water");
    list.push_back("Pressure");
    list.push_back("Relative Humidity at sigma level 995");
    list.push_back("Sea Level Pressure");
    list.push_back("U-Wind at sigma level 995");
    list.push_back("V-Wind at sigma level 995");
    list.push_back("Air Temperature 2m");
    list.push_back("Potential evaporation rate");
    list.push_back("Specific humidity at 2m");
    list.push_back("Skin Temperature");
    list.push_back("Temperature of 0-10cm layer");
    list.push_back("Temperature of 10-200cm layer");
    list.push_back("Temperature at 300cm");
    list.push_back("U-wind at 10m");
    list.push_back("V-wind at 10m");
    list.push_back("Convective precipitation rate");
    list.push_back("Downward longwave radiation flux");
    list.push_back("Downward solar radiation flux");
    list.push_back("Ground heat flux");
    list.push_back("Latent heat net flux");
    list.push_back("Near IR beam downward solar flux");
    list.push_back("Near IR diffuse downward solar flux");
    list.push_back("Net longwave radiation");
    list.push_back("Net shortwave radiation");
    list.push_back("Precipitation rate");
    list.push_back("Sensible heat net flux");
    */
    return list;
}
