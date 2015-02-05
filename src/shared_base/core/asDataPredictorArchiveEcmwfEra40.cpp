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
    m_Initialized = false;
    m_DataId = dataId;
    m_DatasetId = "ECMWF_ERA-40";
    m_OriginalProvider = "ECMWF";
    m_FinalProvider = "ECMWF";
    m_FinalProviderWebsite = "http://apps.ecmwf.int/datasets/data/era40_daily/";
    m_FinalProviderFTP = wxEmptyString;
    m_DatasetName = "ERA-40";
    m_OriginalProviderStart = asTime::GetMJD(1957, 9, 1);
    m_OriginalProviderEnd = asTime::GetMJD(2002, 8, 31);
    m_TimeZoneHours = 0;
    m_TimeStepHours = 6;
    m_FirstTimeStepHours = 0;
    m_NanValues.push_back(-32767);
    m_CoordinateSystem = WGS84;
    m_XaxisShift = 0;
    m_YaxisShift = 0;
    m_SubFolder = wxEmptyString;
    m_FileAxisLatName = "latitude";
    m_FileAxisLonName = "longitude";
    m_FileAxisTimeName = "time";
    m_FileAxisLevelName = "level";

    // The axis steps are defined here for default grids and will be overridden by real data resolution.
    m_XaxisStep = 2.5;
    m_YaxisStep = 2.5;

    // Identify data ID and set the corresponding properties.
    if (m_DataId.IsSameAs("hgt", false))
    {
        m_DataParameter = GeopotentialHeight;
        m_FileNamePattern = "ECMWF_ERA40_hgt.nc";
        m_FileVariableName = "z";
        m_Unit = m;
    }
    /*
    else if (m_DataId.IsSameAs("air", false))
    {
        m_DataParameter = AirTemperature;
        m_FileNamePattern = "air.%d.nc";
        m_FileVariableName = "air";
        m_Unit = degK;
    }
    else if (m_DataId.IsSameAs("omega", false))
    {
        m_DataParameter = Omega;
        m_SubFolder = "pressure";
        m_FileNamePattern = "omega.%d.nc";
        m_FileVariableName = "omega";
        m_Unit = PascalsPerSec;
    }
    else if (m_DataId.IsSameAs("rhum", false))
    {
        m_DataParameter = RelativeHumidity;
        m_SubFolder = "pressure";
        m_FileNamePattern = "rhum.%d.nc";
        m_FileVariableName = "rhum";
        m_Unit = percent;
    }
    else if (m_DataId.IsSameAs("shum", false))
    {
        m_DataParameter = SpecificHumidity;
        m_SubFolder = "pressure";
        m_FileNamePattern = "shum.%d.nc";
        m_FileVariableName = "shum";
        m_Unit = kgPerKg;
    }
    else if (m_DataId.IsSameAs("uwnd", false))
    {
        m_DataParameter = Uwind;
        m_SubFolder = "pressure";
        m_FileNamePattern = "uwnd.%d.nc";
        m_FileVariableName = "uwnd";
        m_Unit = mPerSec;
    }
    else if (m_DataId.IsSameAs("vwnd", false))
    {
        m_DataParameter = Vwind;
        m_SubFolder = "pressure";
        m_FileNamePattern = "vwnd.%d.nc";
        m_FileVariableName = "vwnd";
        m_Unit = mPerSec;
    }
    else if (m_DataId.IsSameAs("surf_air", false))
    {
        m_DataParameter = AirTemperature;
        m_SubFolder = "surface";
        m_FileNamePattern = "air.sig995.%d.nc";
        m_FileVariableName = "air";
        m_Unit = degK;
    }
    else if (m_DataId.IsSameAs("surf_lftx", false))
    {
        m_DataParameter = SurfaceLiftedIndex;
        m_SubFolder = "surface";
        m_FileNamePattern = "lftx.sfc.%d.nc";
        m_FileVariableName = "lftx";
        m_Unit = degK;
    }
    else if (m_DataId.IsSameAs("surf_lftx4", false))
    {
        m_DataParameter = SurfaceLiftedIndex;
        m_SubFolder = "surface";
        m_FileNamePattern = "lftx4.sfc.%d.nc";
        m_FileVariableName = "lftx4";
        m_Unit = degK;
    }
    else if (m_DataId.IsSameAs("surf_omega", false))
    {
        m_DataParameter = Omega;
        m_SubFolder = "surface";
        m_FileNamePattern = "omega.sig995.%d.nc";
        m_FileVariableName = "omega";
        m_Unit = PascalsPerSec;
    }
    else if (m_DataId.IsSameAs("surf_pottmp", false))
    {
        m_DataParameter = PotentialTemperature;
        m_SubFolder = "surface";
        m_FileNamePattern = "pottmp.sig995.%d.nc";
        m_FileVariableName = "pottmp";
        m_Unit = degK;
    }
    else if (m_DataId.IsSameAs("surf_prwtr", false))
    {
        m_DataParameter = PrecipitableWater;
        m_SubFolder = "surface";
        m_FileNamePattern = "pr_wtr.eatm.%d.nc";
        m_FileVariableName = "pr_wtr";
        m_Unit = mm;
    }
    else if (m_DataId.IsSameAs("surf_pres", false))
    {
        m_DataParameter = Pressure;
        m_SubFolder = "surface";
        m_FileNamePattern = "pres.sfc.%d.nc";
        m_FileVariableName = "pres";
        m_Unit = Pascals;
    }
    else if (m_DataId.IsSameAs("surf_rhum", false))
    {
        m_DataParameter = RelativeHumidity;
        m_SubFolder = "surface";
        m_FileNamePattern = "rhum.sig995.%d.nc";
        m_FileVariableName = "rhum";
        m_Unit = percent;
    }
    else if (m_DataId.IsSameAs("surf_slp", false))
    {
        m_DataParameter = Pressure;
        m_SubFolder = "surface";
        m_FileNamePattern = "slp.%d.nc";
        m_FileVariableName = "slp";
        m_Unit = Pascals;
    }
    else if (m_DataId.IsSameAs("surf_uwnd", false))
    {
        m_DataParameter = Uwind;
        m_SubFolder = "surface";
        m_FileNamePattern = "uwnd.sig995.%d.nc";
        m_FileVariableName = "uwnd";
        m_Unit = mPerSec;
    }
    else if (m_DataId.IsSameAs("surf_vwnd", false))
    {
        m_DataParameter = Vwind;
        m_SubFolder = "surface";
        m_FileNamePattern = "vwnd.sig995.%d.nc";
        m_FileVariableName = "vwnd";
        m_Unit = mPerSec;
    }
    else
    {
        m_XaxisStep = NaNFloat;
        m_YaxisStep = NaNFloat;

        if (m_DataId.IsSameAs("flux_air2m", false))
        {
            m_DataParameter = AirTemperature;
            m_SubFolder = "surface_gauss";
            m_FileNamePattern = "air.2m.gauss.%d.nc";
            m_FileVariableName = "air";
            m_Unit = degK;
        }
        else if (m_DataId.IsSameAs("flux_pevpr", false))
        {
            m_DataParameter = PotentialEvaporation;
            m_SubFolder = "surface_gauss";
            m_FileNamePattern = "pevpr.sfc.gauss.%d.nc";
            m_FileVariableName = "pevpr";
            m_Unit = WPerm2;
        }
        else if (m_DataId.IsSameAs("flux_shum2m", false))
        {
            m_DataParameter = SpecificHumidity;
            m_SubFolder = "surface_gauss";
            m_FileNamePattern = "shum.2m.gauss.%d.nc";
            m_FileVariableName = "shum";
            m_Unit = kgPerKg;
        }
        else if (m_DataId.IsSameAs("flux_sktmp", false))
        {
            m_DataParameter = SurfaceTemperature;
            m_SubFolder = "surface_gauss";
            m_FileNamePattern = "skt.sfc.gauss.%d.nc";
            m_FileVariableName = "skt";
            m_Unit = degK;
        }
        else if (m_DataId.IsSameAs("flux_tmp0-10", false))
        {
            m_DataParameter = SurfaceTemperature;
            m_SubFolder = "surface_gauss";
            m_FileNamePattern = "tmp.0-10cm.gauss.%d.nc";
            m_FileVariableName = "tmp";
            m_Unit = degK;
        }
        else if (m_DataId.IsSameAs("flux_tmp10-200", false))
        {
            m_DataParameter = SurfaceTemperature;
            m_SubFolder = "surface_gauss";
            m_FileNamePattern = "tmp.10-200cm.gauss.%d.nc";
            m_FileVariableName = "tmp";
            m_Unit = degK;
        }
        else if (m_DataId.IsSameAs("flux_tmp300", false))
        {
            m_DataParameter = SurfaceTemperature;
            m_SubFolder = "surface_gauss";
            m_FileNamePattern = "tmp.300cm.gauss.%d.nc";
            m_FileVariableName = "tmp";
            m_Unit = degK;
        }
        else if (m_DataId.IsSameAs("flux_uwnd10m", false))
        {
            m_DataParameter = Uwind;
            m_SubFolder = "surface_gauss";
            m_FileNamePattern = "uwnd.10m.gauss.%d.nc";
            m_FileVariableName = "uwnd";
            m_Unit = mPerSec;
        }
        else if (m_DataId.IsSameAs("flux_vwnd10m", false))
        {
            m_DataParameter = Vwind;
            m_SubFolder = "surface_gauss";
            m_FileNamePattern = "vwnd.10m.gauss.%d.nc";
            m_FileVariableName = "vwnd";
            m_Unit = mPerSec;
        }
        else if (m_DataId.IsSameAs("flux_cprat", false))
        {
            m_DataParameter = ConvectivePrecipitation;
            m_SubFolder = "surface_gauss";
            m_FileNamePattern = "cprat.sfc.gauss.%d.nc";
            m_FileVariableName = "cprat";
            m_Unit = kgPerm2Pers;
        }
        else if (m_DataId.IsSameAs("flux_dlwrf", false))
        {
            m_DataParameter = LongwaveRadiation;
            m_SubFolder = "surface_gauss";
            m_FileNamePattern = "dlwrf.sfc.gauss.%d.nc";
            m_FileVariableName = "dlwrf";
            m_Unit = WPerm2;
        }
        else if (m_DataId.IsSameAs("flux_dswrf", false))
        {
            m_DataParameter = SolarRadiation;
            m_SubFolder = "surface_gauss";
            m_FileNamePattern = "dswrf.sfc.gauss.%d.nc";
            m_FileVariableName = "dswrf";
            m_Unit = WPerm2;
        }
        else if (m_DataId.IsSameAs("flux_gflux", false))
        {
            m_DataParameter = GroundHeatFlux;
            m_SubFolder = "surface_gauss";
            m_FileNamePattern = "gflux.sfc.gauss.%d.nc";
            m_FileVariableName = "gflux";
            m_Unit = WPerm2;
        }
        else if (m_DataId.IsSameAs("flux_lhtfl", false))
        {
            m_DataParameter = LatentHeatFlux;
            m_SubFolder = "surface_gauss";
            m_FileNamePattern = "lhtfl.sfc.gauss.%d.nc";
            m_FileVariableName = "lhtfl";
            m_Unit = WPerm2;
        }
        else if (m_DataId.IsSameAs("flux_nbdsf", false))
        {
            m_DataParameter = NearIRFlux;
            m_SubFolder = "surface_gauss";
            m_FileNamePattern = "nbdsf.sfc.gauss.%d.nc";
            m_FileVariableName = "nbdsf";
            m_Unit = WPerm2;
        }
        else if (m_DataId.IsSameAs("flux_nddsf", false))
        {
            m_DataParameter = NearIRFlux;
            m_SubFolder = "surface_gauss";
            m_FileNamePattern = "nddsf.sfc.gauss.%d.nc";
            m_FileVariableName = "nddsf";
            m_Unit = WPerm2;
        }
        else if (m_DataId.IsSameAs("flux_nlwrs", false))
        {
            m_DataParameter = LongwaveRadiation;
            m_SubFolder = "surface_gauss";
            m_FileNamePattern = "nlwrs.sfc.gauss.%d.nc";
            m_FileVariableName = "nlwrs";
            m_Unit = WPerm2;
        }
        else if (m_DataId.IsSameAs("flux_nswrs", false))
        {
            m_DataParameter = ShortwaveRadiation;
            m_SubFolder = "surface_gauss";
            m_FileNamePattern = "nswrs.sfc.gauss.%d.nc";
            m_FileVariableName = "nswrs";
            m_Unit = WPerm2;
        }
        else if (m_DataId.IsSameAs("flux_prate", false))
        {
            m_DataParameter = Precipitation;
            m_SubFolder = "surface_gauss";
            m_FileNamePattern = "prate.sfc.gauss.%d.nc";
            m_FileVariableName = "prate";
            m_Unit = kgPerm2Pers;
        }
        else if (m_DataId.IsSameAs("flux_shtfl", false))
        {
            m_DataParameter = SensibleHeatFlux;
            m_SubFolder = "surface_gauss";
            m_FileNamePattern = "shtfl.sfc.gauss.%d.nc";
            m_FileVariableName = "shtfl";
            m_Unit = WPerm2;
        }
        else
        {
            m_DataParameter = NoDataParameter;
            m_SubFolder = wxEmptyString;
            m_FileNamePattern = wxEmptyString;
            m_FileVariableName = wxEmptyString;
            m_Unit = NoDataUnit;
        }
    }*/

}

asDataPredictorArchiveEcmwfEra40::~asDataPredictorArchiveEcmwfEra40()
{

}

bool asDataPredictorArchiveEcmwfEra40::Init()
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
