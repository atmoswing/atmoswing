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

#include "asDataPredictorArchiveNcepReanalysis2.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileNetcdf.h>


asDataPredictorArchiveNcepReanalysis2::asDataPredictorArchiveNcepReanalysis2(const wxString &dataId)
        : asDataPredictorArchive(dataId)
{
    // Set the basic properties.
    m_initialized = false;
    m_datasetId = "NCEP_Reanalysis_v2";
    m_originalProvider = "NCEP/DOE";
    m_datasetName = "Reanalysis 2";
    m_originalProviderStart = asTime::GetMJD(1979, 1, 1);
    m_originalProviderEnd = NaNDouble;
    m_timeZoneHours = 0;
    m_timeStepHours = 6;
    m_firstTimeStepHours = 0;
    m_nanValues.push_back(32767);
    m_nanValues.push_back(936 * std::pow(10.f, 34.f));
    m_xAxisShift = 0;
    m_yAxisShift = 0;
    m_fileStructure.dimLatName = "lat";
    m_fileStructure.dimLonName = "lon";
    m_fileStructure.dimTimeName = "time";
    m_fileStructure.dimLevelName = "level";
}

asDataPredictorArchiveNcepReanalysis2::~asDataPredictorArchiveNcepReanalysis2()
{

}

bool asDataPredictorArchiveNcepReanalysis2::Init()
{
    CheckLevelTypeIsDefined();

    // Identify data ID and set the corresponding properties.
    if (m_product.IsSameAs("pressure", false) || m_product.IsSameAs("press", false)) {
        m_fileStructure.hasLevelDimension = true;
        m_subFolder = "pressure";
        m_xAxisStep = 2.5;
        m_yAxisStep = 2.5;
        if (m_dataId.IsSameAs("air", false)) {
            m_parameter = AirTemperature;
            m_parameterName = "Air Temperature";
            m_fileVariableName = "air";
            m_unit = degK;
        } else if (m_dataId.IsSameAs("hgt", false)) {
            m_parameter = GeopotentialHeight;
            m_parameterName = "Geopotential height";
            m_fileVariableName = "hgt";
            m_unit = m;
        } else if (m_dataId.IsSameAs("rhum", false)) {
            m_parameter = RelativeHumidity;
            m_parameterName = "Relative Humidity";
            m_fileVariableName = "rhum";
            m_unit = percent;
        } else if (m_dataId.IsSameAs("omega", false)) {
            m_parameter = VerticalVelocity;
            m_parameterName = "Vertical velocity";
            m_fileVariableName = "omega";
            m_unit = Pa_s;
        } else if (m_dataId.IsSameAs("uwnd", false)) {
            m_parameter = Uwind;
            m_parameterName = "U-Wind";
            m_fileVariableName = "uwnd";
            m_unit = m_s;
        } else if (m_dataId.IsSameAs("vwnd", false)) {
            m_parameter = Vwind;
            m_parameterName = "V-Wind";
            m_fileVariableName = "vwnd";
            m_unit = m_s;
        } else {
            asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                              m_dataId, m_product));
        }
        m_fileNamePattern = m_fileVariableName + ".%d.nc";

    } else if (m_product.IsSameAs("surface", false) || m_product.IsSameAs("surf", false)) {
        m_fileStructure.hasLevelDimension = false;
        m_subFolder = "surface";
        m_xAxisStep = 2.5;
        m_yAxisStep = 2.5;
        if (m_dataId.IsSameAs("prwtr", false)) {
            m_parameter = PrecipitableWater;
            m_parameterName = "Precipitable water";
            m_fileNamePattern = "pr_wtr.eatm.%d.nc";
            m_fileVariableName = "pr_wtr";
            m_unit = mm;
        } else if (m_dataId.IsSameAs("pres", false)) {
            m_parameter = Pressure;
            m_parameterName = "Pressure";
            m_fileNamePattern = "pres.sfc.%d.nc";
            m_fileVariableName = "pres";
            m_unit = Pa;
        } else if (m_dataId.IsSameAs("mslp", false)) {
            m_parameter = Pressure;
            m_parameterName = "Mean Sea level pressure";
            m_fileNamePattern = "mslp.%d.nc";
            m_fileVariableName = "mslp";
            m_unit = Pa;
        } else {
            asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                              m_dataId, m_product));
        }

    } else if (m_product.IsSameAs("surface_gauss", false) || m_product.IsSameAs("gaussian_grid", false) ||
            m_product.IsSameAs("gauss", false) || m_product.IsSameAs("flux", false)) {
        m_fileStructure.hasLevelDimension = false;
        m_subFolder = "gaussian_grid";
        m_xAxisStep = NaNFloat;
        m_yAxisStep = NaNFloat;
        if (m_dataId.IsSameAs("air2m", false)) {
            m_fileStructure.hasLevelDimension = true;
            m_fileStructure.singleLevel = true;
            m_parameter = AirTemperature;
            m_parameterName = "Air Temperature 2m";
            m_fileNamePattern = "air.2m.gauss.%d.nc";
            m_fileVariableName = "air";
            m_unit = degK;
        } else if (m_dataId.IsSameAs("shum2m", false)) {
            m_fileStructure.hasLevelDimension = true;
            m_fileStructure.singleLevel = true;
            m_parameter = SpecificHumidity;
            m_parameterName = "Specific humidity at 2m";
            m_fileNamePattern = "shum.2m.gauss.%d.nc";
            m_fileVariableName = "shum";
            m_unit = kg_kg;
        } else if (m_dataId.IsSameAs("tmax2m", false)) {
            m_fileStructure.hasLevelDimension = true;
            m_fileStructure.singleLevel = true;
            m_parameter = AirTemperature;
            m_parameterName = "Maximum temperature at 2m";
            m_fileNamePattern = "tmax.2m.gauss.%d.nc";
            m_fileVariableName = "tmax";
            m_unit = degK;
        } else if (m_dataId.IsSameAs("tmin2m", false)) {
            m_fileStructure.hasLevelDimension = true;
            m_fileStructure.singleLevel = true;
            m_parameter = AirTemperature;
            m_parameterName = "Minimum temperature at 2m";
            m_fileNamePattern = "tmin.2m.gauss.%d.nc";
            m_fileVariableName = "tmin";
            m_unit = degK;
        } else if (m_dataId.IsSameAs("sktmp", false)) {
            m_parameter = SoilTemperature;
            m_parameterName = "Skin Temperature";
            m_fileNamePattern = "skt.sfc.gauss.%d.nc";
            m_fileVariableName = "skt";
            m_unit = degK;
        } else if (m_dataId.IsSameAs("soilw0-10", false)) {
            m_fileStructure.hasLevelDimension = true;
            m_fileStructure.singleLevel = true;
            m_parameter = SoilMoisture;
            m_parameterName = "Soil moisture (0-10cm)";
            m_fileNamePattern = "soilw.0-10cm.gauss.%d.nc";
            m_fileVariableName = "soilw";
            m_unit = fraction;
        } else if (m_dataId.IsSameAs("soilw10-200", false)) {
            m_fileStructure.hasLevelDimension = true;
            m_fileStructure.singleLevel = true;
            m_parameter = SoilMoisture;
            m_parameterName = "Soil moisture (10-200cm)";
            m_fileNamePattern = "soilw.10-200cm.gauss.%d.nc";
            m_fileVariableName = "soilw";
            m_unit = fraction;
        } else if (m_dataId.IsSameAs("tmp0-10", false)) {
            m_fileStructure.hasLevelDimension = true;
            m_fileStructure.singleLevel = true;
            m_parameter = SoilTemperature;
            m_parameterName = "Temperature of 0-10cm layer";
            m_fileNamePattern = "tmp.0-10cm.gauss.%d.nc";
            m_fileVariableName = "tmp";
            m_unit = degK;
        } else if (m_dataId.IsSameAs("tmp10-200", false)) {
            m_fileStructure.hasLevelDimension = true;
            m_fileStructure.singleLevel = true;
            m_parameter = SoilTemperature;
            m_parameterName = "Temperature of 10-200cm layer";
            m_fileNamePattern = "tmp.10-200cm.gauss.%d.nc";
            m_fileVariableName = "tmp";
            m_unit = degK;
        } else if (m_dataId.IsSameAs("uwnd10m", false)) {
            m_fileStructure.hasLevelDimension = true;
            m_fileStructure.singleLevel = true;
            m_parameter = Uwind;
            m_parameterName = "U-wind at 10 m";
            m_fileNamePattern = "uwnd.10m.gauss.%d.nc";
            m_fileVariableName = "uwnd";
            m_unit = m_s;
        } else if (m_dataId.IsSameAs("vwnd10m", false)) {
            m_fileStructure.hasLevelDimension = true;
            m_fileStructure.singleLevel = true;
            m_parameter = Vwind;
            m_parameterName = "V-wind at 10 m";
            m_fileNamePattern = "vwnd.10m.gauss.%d.nc";
            m_fileVariableName = "vwnd";
            m_unit = m_s;
        } else if (m_dataId.IsSameAs("weasd", false)) {
            m_parameter = SnowWaterEquivalent;
            m_parameterName = "Water equiv. of snow dept";
            m_fileNamePattern = "weasd.sfc.gauss.%d.nc";
            m_fileVariableName = "weasd";
            m_unit = kg_m2;
        } else if (m_dataId.IsSameAs("cprat", false)) {
            m_parameter = PrecipitationRate;
            m_parameterName = "Convective precipitation rate";
            m_fileNamePattern = "cprat.sfc.gauss.%d.nc";
            m_fileVariableName = "cprat";
            m_unit = kg_m2_s;
        } else if (m_dataId.IsSameAs("dlwrf", false)) {
            m_parameter = Radiation;
            m_parameterName = "Downward longwave radiation flux";
            m_fileNamePattern = "dlwrf.sfc.gauss.%d.nc";
            m_fileVariableName = "dlwrf";
            m_unit = W_m2;
        } else if (m_dataId.IsSameAs("dswrf", false)) {
            m_parameter = Radiation;
            m_parameterName = "Downward solar radiation flux";
            m_fileNamePattern = "dswrf.sfc.gauss.%d.nc";
            m_fileVariableName = "dswrf";
            m_unit = W_m2;
        } else if (m_dataId.IsSameAs("gflux", false)) {
            m_parameter = Radiation;
            m_parameterName = "Ground heat flux";
            m_fileNamePattern = "gflux.sfc.gauss.%d.nc";
            m_fileVariableName = "gflux";
            m_unit = W_m2;
        } else if (m_dataId.IsSameAs("lhtfl", false)) {
            m_parameter = Radiation;
            m_parameterName = "Latent heat net flux";
            m_fileNamePattern = "lhtfl.sfc.gauss.%d.nc";
            m_fileVariableName = "lhtfl";
            m_unit = W_m2;
        } else if (m_dataId.IsSameAs("pevpr", false)) {
            m_parameter = PotentialEvaporation;
            m_parameterName = "Potential evaporation rate";
            m_fileNamePattern = "pevpr.sfc.gauss.%d.nc";
            m_fileVariableName = "pevpr";
            m_unit = W_m2;
        } else if (m_dataId.IsSameAs("prate", false)) {
            m_parameter = PrecipitationRate;
            m_parameterName = "Precipitation rate";
            m_fileNamePattern = "prate.sfc.gauss.%d.nc";
            m_fileVariableName = "prate";
            m_unit = kg_m2_s;
        } else if (m_dataId.IsSameAs("shtfl", false)) {
            m_parameter = Radiation;
            m_parameterName = "Sensible heat net flux";
            m_fileNamePattern = "shtfl.sfc.gauss.%d.nc";
            m_fileVariableName = "shtfl";
            m_unit = W_m2;
        } else if (m_dataId.IsSameAs("tcdc", false)) {
            m_parameter = CloudCover;
            m_parameterName = "Total cloud cover";
            m_fileNamePattern = "tcdc.eatm.gauss.%d.nc";
            m_fileVariableName = "tcdc";
            m_unit = percent;
        } else if (m_dataId.IsSameAs("uflx", false)) {
            m_parameter = MomentumFlux;
            m_parameterName = "Momentum flux (zonal)";
            m_fileNamePattern = "uflx.sfc.gauss.%d.nc";
            m_fileVariableName = "uflx";
            m_unit = N_m2;
        } else if (m_dataId.IsSameAs("ugwd", false)) {
            m_parameter = GravityWaveStress;
            m_parameterName = "Zonal gravity wave stress";
            m_fileNamePattern = "ugwd.sfc.gauss.%d.nc";
            m_fileVariableName = "ugwd";
            m_unit = N_m2;
        } else if (m_dataId.IsSameAs("ulwrf", false)) {
            m_parameter = Radiation;
            m_parameterName = "Upward Longwave Radiation Flux";
            m_fileNamePattern = "ulwrf.sfc.gauss.%d.nc";
            m_fileVariableName = "ulwrf";
            m_unit = W_m2;
        } else if (m_dataId.IsSameAs("uswrf", false)) {
            m_parameter = Radiation;
            m_parameterName = "Upward Solar Radiation Flux";
            m_fileNamePattern = "uswrf.sfc.gauss.%d.nc";
            m_fileVariableName = "uswrf";
            m_unit = W_m2;
        } else if (m_dataId.IsSameAs("vflx", false)) {
            m_parameter = MomentumFlux;
            m_parameterName = "Momentum Flux (meridional)";
            m_fileNamePattern = "vflx.sfc.gauss.%d.nc";
            m_fileVariableName = "vflx";
            m_unit = N_m2;
        } else if (m_dataId.IsSameAs("vgwd", false)) {
            m_parameter = GravityWaveStress;
            m_parameterName = "Meridional Gravity Wave Stress";
            m_fileNamePattern = "vgwd.sfc.gauss.%d.nc";
            m_fileVariableName = "vgwd";
            m_unit = N_m2;
        } else if (m_dataId.IsSameAs("vgwd", false)) {
            m_parameter = GravityWaveStress;
            m_parameterName = "Meridional Gravity Wave Stress";
            m_fileNamePattern = "vgwd.sfc.gauss.%d.nc";
            m_fileVariableName = "vgwd";
            m_unit = N_m2;
        } else {
            asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                              m_dataId, m_product));
        }

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

VectorString asDataPredictorArchiveNcepReanalysis2::GetListOfFiles(asTimeArray &timeArray) const
{
    VectorString files;

    for (int i_year = timeArray.GetStartingYear(); i_year <= timeArray.GetEndingYear(); i_year++) {
        files.push_back(GetFullDirectoryPath() + wxString::Format(m_fileNamePattern, i_year));
    }

    return files;
}

bool asDataPredictorArchiveNcepReanalysis2::ExtractFromFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea,
                                                            asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    return ExtractFromNetcdfFile(fileName, dataArea, timeArray, compositeData);
}

double asDataPredictorArchiveNcepReanalysis2::ConvertToMjd(double timeValue, double refValue) const
{
    timeValue = (timeValue / 24.0); // hours to days
    timeValue += asTime::GetMJD(1800, 1, 1); // to MJD: add a negative time span

    return timeValue;
}
