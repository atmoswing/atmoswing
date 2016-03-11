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

#include "asDataPredictorArchiveNcepReanalysis1.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileNetcdf.h>


asDataPredictorArchiveNcepReanalysis1::asDataPredictorArchiveNcepReanalysis1(const wxString &dataId)
:
asDataPredictorArchive(dataId)
{
    // Set the basic properties.
    m_initialized = false;
    m_dataId = dataId;
    m_datasetId = "NCEP_Reanalysis_v1";
    m_originalProvider = "NCEP/NCAR";
    m_finalProvider = "NCEP/NCAR";
    m_finalProviderWebsite = "http://www.esrl.noaa.gov/psd/data/reanalysis/reanalysis.shtml";
    m_finalProviderFTP = "ftp://ftp.cdc.noaa.gov/DataSets/ncep.reanalysis";
    m_datasetName = "Reanalysis 1";
    m_originalProviderStart = asTime::GetMJD(1948, 1, 1);
    m_originalProviderEnd = NaNDouble;
    m_timeZoneHours = 0;
    m_timeStepHours = 6;
    m_firstTimeStepHours = 0;
    m_nanValues.push_back(32767);
    m_nanValues.push_back(936*std::pow(10.f,34.f));
    m_xAxisShift = 0;
    m_yAxisShift = 0;
    m_fileAxisLatName = "lat";
    m_fileAxisLonName = "lon";
    m_fileAxisTimeName = "time";
    m_fileAxisLevelName = "level";

    // The axis steps are defined here for regular grids and will be overridden for unregular grids.
    m_xAxisStep = 2.5;
    m_yAxisStep = 2.5;

    // Identify data ID and set the corresponding properties.
    if (m_dataId.IsSameAs("hgt", false))
    {
        m_dataParameter = GeopotentialHeight;
        m_subFolder = "pressure";
        m_fileNamePattern = "hgt.%d.nc";
        m_fileVariableName = "hgt";
        m_unit = m;
    }
    else if (m_dataId.IsSameAs("air", false))
    {
        m_dataParameter = AirTemperature;
        m_subFolder = "pressure";
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
    }

}

asDataPredictorArchiveNcepReanalysis1::~asDataPredictorArchiveNcepReanalysis1()
{

}

bool asDataPredictorArchiveNcepReanalysis1::Init()
{
    // Check data ID
    if (m_fileNamePattern.IsEmpty() || m_fileVariableName.IsEmpty()) {
        asLogError(wxString::Format(_("The provided data ID (%s) does not match any possible option in the dataset %s."), m_dataId, m_datasetName));
        return false;
    }

    // Check directory is set
    if (m_directoryPath.IsEmpty()) {
        asLogError(wxString::Format(_("The path to the directory has not been set for the data %s from the dataset %s."), m_dataId, m_datasetName));
        return false;
    }

    // Set to initialized
    m_initialized = true;

    return true;
}

VectorString asDataPredictorArchiveNcepReanalysis1::GetDataIdList()
{
    VectorString list;

    list.push_back("hgt"); // Geopotential Height
    list.push_back("air"); // Air Temperature
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

    return list;
}

VectorString asDataPredictorArchiveNcepReanalysis1::GetDataIdDescriptionList()
{
    VectorString list;

    list.push_back("Geopotential Height");
    list.push_back("Air Temperature");
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

    return list;
}

bool asDataPredictorArchiveNcepReanalysis1::ExtractFromFiles(asGeoAreaCompositeGrid *& dataArea, asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    // Get requested years
    int yearFirst = timeArray.GetFirstDayYear();
    int yearLast = timeArray.GetLastDayYear();

    #if wxUSE_GUI
        asDialogProgressBar progressBar(_("Loading data from files.\n"), yearLast-yearFirst);
    #endif

    // Loop through the files
    for (int i_year=yearFirst; i_year<=yearLast; i_year++)
    {
        // Build the file path
        wxString fileFullPath = m_directoryPath + wxString::Format(m_fileNamePattern, i_year);

        #if wxUSE_GUI
            // Update the progress bar
            wxString fileNameMessage = wxString::Format(_("Loading data from files.\nFile: %s"), wxString::Format(m_fileNamePattern, i_year));
            if(!progressBar.Update(i_year-yearFirst, fileNameMessage))
            {
                asLogWarning(_("The process has been canceled by the user."));
                return false;
            }
        #endif

        // Open the NetCDF file
        ThreadsManager().CritSectionNetCDF().Enter();
        asFileNetcdf ncFile(fileFullPath, asFileNetcdf::ReadOnly);
        if(!ncFile.Open())
        {
            ThreadsManager().CritSectionNetCDF().Leave();
            return false;
        }

        // Number of dimensions
        int nDims = ncFile.GetNDims();
        wxASSERT(nDims>=3);
        wxASSERT(nDims<=4);

        // Get some attributes
        float dataAddOffset = ncFile.GetAttFloat("add_offset", m_fileVariableName);
        if (asTools::IsNaN(dataAddOffset)) dataAddOffset = 0;
        float dataScaleFactor = ncFile.GetAttFloat("scale_factor", m_fileVariableName);
        if (asTools::IsNaN(dataScaleFactor)) dataScaleFactor = 1;
        bool scalingNeeded = true;
        if (dataAddOffset==0 && dataScaleFactor==1) scalingNeeded = false;

        // Get full axes from the netcdf file
        Array1DFloat axisDataLon(ncFile.GetVarLength(m_fileAxisLonName));
        ncFile.GetVar(m_fileAxisLonName, &axisDataLon[0]);
        Array1DFloat axisDataLat(ncFile.GetVarLength(m_fileAxisLatName));
        ncFile.GetVar(m_fileAxisLatName, &axisDataLat[0]);
        Array1DFloat axisDataLevel;
        if (nDims==4)
        {
            axisDataLevel.resize(ncFile.GetVarLength(m_fileAxisLevelName));
            ncFile.GetVar(m_fileAxisLevelName, &axisDataLevel[0]);
        }

        // Adjust axes if necessary
        dataArea = AdjustAxes(dataArea, axisDataLon, axisDataLat, compositeData);
        if(dataArea) 
        {
            wxASSERT(dataArea->GetNbComposites()>0);
        }
            
        // Time array takes ages to load !! Avoid if possible. Get the first value of the time array.
        size_t axisDataTimeLength = ncFile.GetVarLength(m_fileAxisTimeName);
        double valFirstTime = ncFile.GetVarOneDouble(m_fileAxisTimeName, 0);
        valFirstTime = (valFirstTime/24.0); // hours to days
        bool format2003 = false;
        bool format2014 = false;
        if(valFirstTime<500*365) { // New format
            valFirstTime += asTime::GetMJD(1800,1,1); // to MJD: add a negative time span
            format2014 = true;
        } else { // Old format
            valFirstTime += asTime::GetMJD(1,1,1); // to MJD: add a negative time span
            format2003 = true;
        }
            
        // Get start and end of the current year
        double timeStart = asTime::GetMJD(i_year,1,1,0,0);
        double timeEnd = asTime::GetMJD(i_year,12,31,23,59);

        // Get the time length
        double timeArrayIndexStart = timeArray.GetIndexFirstAfter(timeStart);
        double timeArrayIndexEnd = timeArray.GetIndexFirstBefore(timeEnd);
        int indexLengthTime = timeArrayIndexEnd-timeArrayIndexStart+1;
        int indexLengthTimeArray = indexLengthTime;

        // Correct the time start and end
        size_t indexStartTime = 0;
        int cutStart = 0;
        if(i_year==yearFirst)
        {
            cutStart = timeArrayIndexStart;
        }
        int cutEnd = 0;
        while (valFirstTime<timeArray[timeArrayIndexStart])
        {
            valFirstTime += m_timeStepHours/24.0;
            indexStartTime++;
        }
        if (indexStartTime+indexLengthTime>axisDataTimeLength)
        {
            indexLengthTime--;
            cutEnd++;
        }

        // Containers for extraction
        VectorInt vectIndexLengthLat;
        VectorInt vectIndexLengthLon;
        VectorBool vectLoad360;
        VVectorFloat vectData;
        VVectorFloat vectData360;

        for (int i_area = 0; i_area<(int)compositeData.size(); i_area++)
        {
            // Check if necessary to load the data of lon=360 (so lon=0)
            bool load360 = false;

            int indexStartLon, indexStartLat, indexLengthLon, indexLengthLat;
            if (dataArea)
            {
                // Get the spatial extent
                float lonMin = dataArea->GetXaxisCompositeStart(i_area);
                float lonMax = dataArea->GetXaxisCompositeEnd(i_area);
                float latMinStart = dataArea->GetYaxisCompositeStart(i_area);
                float latMinEnd = dataArea->GetYaxisCompositeEnd(i_area);

                // The dimensions lengths
                indexLengthLon = dataArea->GetXaxisCompositePtsnb(i_area);
                indexLengthLat = dataArea->GetYaxisCompositePtsnb(i_area);

                if(lonMax==dataArea->GetAxisXmax())
                {
                    // Correction if the lon 360 degrees is required (doesn't exist)
                    load360 = true;
                    for (int i_check = 0; i_check<(int)compositeData.size(); i_check++)
                    {
                        // If so, already loaded in another composite
                        if(dataArea->GetComposite(i_check).GetXmin() == 0)
                        {
                            load360 = false;
                        }
                    }
                    indexLengthLon--;
                }

                // Get the spatial indices of the desired data
                indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLon.size()-1], lonMin, 0.01f);
                if(indexStartLon==asOUT_OF_RANGE)
                {
                    // If not found, try with negative angles
                    indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLon.size()-1], lonMin-360, 0.01f);
                }
                if(indexStartLon==asOUT_OF_RANGE)
                {
                    // If not found, try with angles above 360 degrees
                    indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLon.size()-1], lonMin+360, 0.01f);
                }
                if(indexStartLon<0)
                {
                    asLogError(wxString::Format("Cannot find lonMin (%f) in the array axisDataLon ([0]=%f -> [%d]=%f) ", lonMin, axisDataLon[0], (int)axisDataLon.size(), axisDataLon[axisDataLon.size()-1]));
                    return false;
                }
                wxASSERT_MSG(indexStartLon>=0, wxString::Format("axisDataLon[0] = %f, &axisDataLon[%d] = %f & lonMin = %f", axisDataLon[0], (int)axisDataLon.size(), axisDataLon[axisDataLon.size()-1], lonMin));

                int indexStartLat1 = asTools::SortedArraySearch(&axisDataLat[0], &axisDataLat[axisDataLat.size()-1], latMinStart, 0.01f);
                int indexStartLat2 = asTools::SortedArraySearch(&axisDataLat[0], &axisDataLat[axisDataLat.size()-1], latMinEnd, 0.01f);
                wxASSERT_MSG(indexStartLat1>=0, wxString::Format("Looking for %g in %g to %g", latMinStart, axisDataLat[0], axisDataLat[axisDataLat.size()-1]));
                wxASSERT_MSG(indexStartLat2>=0, wxString::Format("Looking for %g in %g to %g", latMinEnd, axisDataLat[0], axisDataLat[axisDataLat.size()-1]));
                indexStartLat = wxMin(indexStartLat1, indexStartLat2);
            }
            else
            {
                indexStartLon = 0;
                indexStartLat = 0;
                indexLengthLon = m_lonPtsnb;
                indexLengthLat = m_latPtsnb;
            }
            int indexLevel = 0;
            if (nDims==4)
            {
                indexLevel = asTools::SortedArraySearch(&axisDataLevel[0], &axisDataLevel[axisDataLevel.size()-1], m_level, 0.01f);
            }

            // Create the arrays to receive the data
            VectorFloat data, data360;
            VectorShort dataShort, dataShort360;

            // Resize the arrays to store the new data
            int totLength = indexLengthTimeArray * indexLengthLat * indexLengthLon;
            wxASSERT(totLength>0);
            data.resize(totLength);
            if(format2003) {
                dataShort.resize(totLength);
            }

            // Fill empty begining with NaNs
            int indexBegining = 0;
            if(cutStart>0)
            {
                int latlonlength = indexLengthLat*indexLengthLon;
                for (int i_empty=0; i_empty<cutStart; i_empty++)
                {
                    for (int i_emptylatlon=0; i_emptylatlon<latlonlength; i_emptylatlon++)
                    {
                        data[indexBegining] = NaNFloat;
                        indexBegining++;
                    }
                }
            }

            // Fill empty end with NaNs
            int indexEnd = indexLengthTime * indexLengthLat * indexLengthLon - 1;
            if(cutEnd>0)
            {
                int latlonlength = indexLengthLat*indexLengthLon;
                for (int i_empty=0; i_empty<cutEnd; i_empty++)
                {
                    for (int i_emptylatlon=0; i_emptylatlon<latlonlength; i_emptylatlon++)
                    {
                        indexEnd++;
                        data[indexEnd] = NaNFloat;
                    }
                }
            }

            // Get the indices for data
            size_t indexStartData4[4] = {0,0,0,0};
            size_t indexCountData4[4] = {0,0,0,0};
            ptrdiff_t indexStrideData4[4] = {0,0,0,0};
            size_t indexStartData3[3] = {0,0,0};
            size_t indexCountData3[3] = {0,0,0};
            ptrdiff_t indexStrideData3[3] = {0,0,0};

            if (nDims==4)
            {
                // Set the indices for data
                indexStartData4[0] = indexStartTime;
                indexStartData4[1] = indexLevel;
                indexStartData4[2] = indexStartLat;
                indexStartData4[3] = indexStartLon;
                indexCountData4[0] = indexLengthTime;
                indexCountData4[1] = 1;
                indexCountData4[2] = indexLengthLat;
                indexCountData4[3] = indexLengthLon;
                indexStrideData4[0] = m_timeIndexStep;
                indexStrideData4[1] = 1;
                indexStrideData4[2] = m_latIndexStep;
                indexStrideData4[3] = m_lonIndexStep;

                // In the netCDF Common Data Language, variables are printed with the outermost dimension first and the innermost dimension last.
                if(format2014) {
                    ncFile.GetVarSample(m_fileVariableName, indexStartData4, indexCountData4, indexStrideData4, &data[indexBegining]);
                } else {
                    ncFile.GetVarSample(m_fileVariableName, indexStartData4, indexCountData4, indexStrideData4, &dataShort[indexBegining]);
                }
            }
            else
            {
                // Set the indices for data
                indexStartData3[0] = indexStartTime;
                indexStartData3[1] = indexStartLat;
                indexStartData3[2] = indexStartLon;
                indexCountData3[0] = indexLengthTime;
                indexCountData3[1] = indexLengthLat;
                indexCountData3[2] = indexLengthLon;
                indexStrideData3[0] = m_timeIndexStep;
                indexStrideData3[1] = m_latIndexStep;
                indexStrideData3[2] = m_lonIndexStep;

                // In the netCDF Common Data Language, variables are printed with the outermost dimension first and the innermost dimension last.
                if(format2014) {
                    ncFile.GetVarSample(m_fileVariableName, indexStartData3, indexCountData3, indexStrideData3, &data[indexBegining]);
                } else {
                    ncFile.GetVarSample(m_fileVariableName, indexStartData3, indexCountData3, indexStrideData3, &dataShort[indexBegining]);
                }
            }

            // Convert to float
            if(format2003) {
                for (int i = 0; i < dataShort.size(); i++) {
                    data[i] = (float)dataShort[i];
                }
            }

            // Load data at lon = 360 degrees
            if(load360)
            {
                // Resize the arrays to store the new data
                int totlength360 = indexLengthTimeArray * indexLengthLat * 1;
                data360.resize(totlength360);
                if (format2003) {
                    dataShort360.resize(totlength360);
                }

                // Set the indices
                indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLon.size()-1], 360, 0.01f, asHIDE_WARNINGS);
                if(indexStartLon==asOUT_OF_RANGE)
                {
                    // If not found, try with negative angles
                    indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLon.size()-1], 0, 0.01f);
                }

                if (nDims==4)
                {
                    indexStartData4[3] = indexStartLon;
                    indexCountData4[3] = 1;
                    indexStrideData4[3] = 1;
                }
                else
                {
                    indexStartData3[2] = indexStartLon;
                    indexCountData3[2] = 1;
                    indexStrideData3[2] = 1;
                }

                // Fill empty begining with NaNs
                int indexBegining = 0;
                if(cutStart>0)
                {
                    int latlonlength = indexLengthLat*indexLengthLon;
                    for (int i_empty=0; i_empty<cutStart; i_empty++)
                    {
                        for (int i_emptylatlon=0; i_emptylatlon<latlonlength; i_emptylatlon++)
                        {
                            data360[indexBegining] = NaNFloat;
                            indexBegining++;
                        }
                    }
                }

                // Fill empty end with NaNs
                int indexEnd = (indexLengthTime-1) * (indexLengthLat-1) * (indexLengthLon-1);
                if(cutEnd>0)
                {
                    int latlonlength = indexLengthLat*indexLengthLon;
                    for (int i_empty=0; i_empty<cutEnd; i_empty++)
                    {
                        for (int i_emptylatlon=0; i_emptylatlon<latlonlength; i_emptylatlon++)
                        {
                            indexEnd++;
                            data360[indexEnd] = NaNFloat;
                        }
                    }
                }

                // Load data at 0 degrees (corresponds to 360 degrees)
                if (nDims==4)
                {
                    if(format2014){
                        ncFile.GetVarSample(m_fileVariableName, indexStartData4, indexCountData4, indexStrideData4, &data360[indexBegining]);
                    } else {
                        ncFile.GetVarSample(m_fileVariableName, indexStartData4, indexCountData4, indexStrideData4, &dataShort360[indexBegining]);
                    }
                }
                else
                {
                    if(format2014){
                        ncFile.GetVarSample(m_fileVariableName, indexStartData3, indexCountData3, indexStrideData3, &data360[indexBegining]);
                    } else {
                        ncFile.GetVarSample(m_fileVariableName, indexStartData3, indexCountData3, indexStrideData3, &dataShort360[indexBegining]);
                    }
                }

                // Convert to float
                if(format2003) {
                    for (int i = 0; i < dataShort360.size(); i++) {
                        data360[i] = (float)dataShort360[i];
                    }
                }
            }

            // Keep data for later treatment
            vectIndexLengthLat.push_back(indexLengthLat);
            vectIndexLengthLon.push_back(indexLengthLon);
            vectLoad360.push_back(load360);
            vectData.push_back(data);
            vectData360.push_back(data360);
        }

        // Close the nc file
        ncFile.Close();
        ThreadsManager().CritSectionNetCDF().Leave();

        // Allocate space into compositeData if not already done
        if (compositeData[0].capacity()==0)
        {
            int totSize = 0;
            for (int i_area = 0; i_area<(int)compositeData.size(); i_area++)
            {
                int indexLengthLat = vectIndexLengthLat[i_area];
                int indexLengthLon = vectIndexLengthLon[i_area];
                totSize += m_time.size() * indexLengthLat * (indexLengthLon+1); // +1 in case of a border
            }
            compositeData.reserve(totSize);
        }

        // Transfer data
        for (int i_area = 0; i_area<(int)compositeData.size(); i_area++)
        {
            // Extract data
            int indexLengthLat = vectIndexLengthLat[i_area];
            int indexLengthLon = vectIndexLengthLon[i_area];
            bool load360 = vectLoad360[i_area];
            VectorFloat data = vectData[i_area];
            VectorFloat data360 = vectData360[i_area];

            // Loop to extract the data from the array
            int ind = 0;
            for (int i_time=0; i_time<indexLengthTimeArray; i_time++)
            {
                Array2DFloat latlonData;
                if(load360)
                {
                    latlonData = Array2DFloat(indexLengthLat,indexLengthLon+1);
                }
                else
                {
                    latlonData = Array2DFloat(indexLengthLat,indexLengthLon);
                }

                for (int i_lat=0; i_lat<indexLengthLat; i_lat++)
                {
                    for (int i_lon=0; i_lon<indexLengthLon; i_lon++)
                    {
                        ind = i_lon + i_lat * indexLengthLon + i_time * indexLengthLon * indexLengthLat;

                        if (scalingNeeded)
                        {
                            latlonData(i_lat,i_lon) = data[ind] * dataScaleFactor + dataAddOffset;
                        }
                        else
                        {
                            latlonData(i_lat,i_lon) = data[ind];
                        }

                        // Check if not NaN
                        bool notNan = true;
                        for (size_t i_nan=0; i_nan<m_nanValues.size(); i_nan++)
                        {
                            if (data[ind]==m_nanValues[i_nan] || latlonData(i_lat,i_lon)==m_nanValues[i_nan])
                            {
                                notNan = false;
                            }
                        }
                        if (!notNan)
                        {
                            latlonData(i_lat,i_lon) = NaNFloat;
                        }
                    }

                    if(load360)
                    {
                        ind = i_lat + i_time * indexLengthLat;

                        if (scalingNeeded)
                        {
                            latlonData(i_lat,indexLengthLon) = data360[ind] * dataScaleFactor + dataAddOffset;
                        }
                        else
                        {
                            latlonData(i_lat,indexLengthLon) = data360[ind];
                        }

                        // Check if not NaN
                        bool notNan = true;
                        for (size_t i_nan=0; i_nan<m_nanValues.size(); i_nan++)
                        {
                            if (data360[ind]==m_nanValues[i_nan] || latlonData(i_lat,indexLengthLon)==m_nanValues[i_nan])
                            {
                                notNan = false;
                            }
                        }
                        if (!notNan)
                        {
                            latlonData(i_lat,indexLengthLon) = NaNFloat;
                        }
                    }
                }
                compositeData[i_area].push_back(latlonData);
            }
            data.clear();
            data360.clear();
        }
    }

    return true;
}

