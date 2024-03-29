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

#include "asPredictorNcepR2.h"

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorNcepR2::asPredictorNcepR2(const wxString& dataId)
    : asPredictor(dataId) {
    // Set the basic properties.
    m_datasetId = "NCEP_R2";
    m_provider = "NCEP/DOE";
    m_datasetName = "Reanalysis 2";
    m_fileType = asFile::Netcdf;
    m_strideAllowed = true;
    m_nanValues.push_back(32767);
    m_nanValues.push_back(936 * std::pow(10.f, 34.f));
    m_fStr.dimLatName = "lat";
    m_fStr.dimLonName = "lon";
    m_fStr.dimTimeName = "time";
    m_fStr.dimLevelName = "level";
}

bool asPredictorNcepR2::Init() {
    CheckLevelTypeIsDefined();

    // Identify data ID and set the corresponding properties.
    if (IsPressureLevel()) {
        m_fStr.hasLevelDim = true;
        if (IsAirTemperature()) {
            m_parameter = AirTemperature;
            m_parameterName = "Air Temperature";
            m_fileVarName = "air";
            m_unit = degK;
        } else if (IsGeopotentialHeight()) {
            m_parameter = GeopotentialHeight;
            m_parameterName = "Geopotential height";
            m_fileVarName = "hgt";
            m_unit = m;
        } else if (IsRelativeHumidity()) {
            m_parameter = RelativeHumidity;
            m_parameterName = "Relative Humidity";
            m_fileVarName = "rhum";
            m_unit = percent;
        } else if (IsVerticalVelocity()) {
            m_parameter = VerticalVelocity;
            m_parameterName = "Vertical velocity";
            m_fileVarName = "omega";
            m_unit = Pa_s;
        } else if (IsUwindComponent()) {
            m_parameter = Uwind;
            m_parameterName = "U-Wind";
            m_fileVarName = "uwnd";
            m_unit = m_s;
        } else if (IsVwindComponent()) {
            m_parameter = Vwind;
            m_parameterName = "V-Wind";
            m_fileVarName = "vwnd";
            m_unit = m_s;
        } else {
            m_parameter = ParameterUndefined;
            m_parameterName = "Undefined";
            m_fileVarName = m_dataId;
            m_unit = UnitUndefined;
        }
        m_fileNamePattern = m_fileVarName + ".%d.nc";

    } else if (IsSurfaceLevel()) {
        m_fStr.hasLevelDim = false;
        if (IsPrecipitableWater()) {
            m_parameter = PrecipitableWater;
            m_parameterName = "Precipitable water";
            m_fileNamePattern = "pr_wtr.eatm.%d.nc";
            m_fileVarName = "pr_wtr";
            m_unit = mm;
        } else if (IsPressure()) {
            m_parameter = Pressure;
            m_parameterName = "Pressure";
            m_fileNamePattern = "pres.sfc.%d.nc";
            m_fileVarName = "pres";
            m_unit = Pa;
        } else if (IsSeaLevelPressure()) {
            m_parameter = Pressure;
            m_parameterName = "Mean Sea level pressure";
            m_fileNamePattern = "mslp.%d.nc";
            m_fileVarName = "mslp";
            m_unit = Pa;
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), m_dataId, m_product);
            return false;
        }

    } else if (IsSurfaceFluxesLevel() || m_product.IsSameAs("surface_gauss", false) ||
               m_product.IsSameAs("gaussian_grid", false) || m_product.IsSameAs("gauss", false)) {
        m_fStr.hasLevelDim = false;
        if (IsAirTemperature()) {
            m_fStr.hasLevelDim = true;
            m_fStr.singleLevel = true;
            m_parameter = AirTemperature;
            m_parameterName = "Air Temperature 2m";
            m_fileNamePattern = "air.2m.gauss.%d.nc";
            m_fileVarName = "air";
            m_unit = degK;
        } else if (IsSpecificHumidity()) {
            m_fStr.hasLevelDim = true;
            m_fStr.singleLevel = true;
            m_parameter = SpecificHumidity;
            m_parameterName = "Specific humidity at 2m";
            m_fileNamePattern = "shum.2m.gauss.%d.nc";
            m_fileVarName = "shum";
            m_unit = kg_kg;
        } else if (m_dataId.IsSameAs("tmax2m", false)) {
            m_fStr.hasLevelDim = true;
            m_fStr.singleLevel = true;
            m_parameter = AirTemperature;
            m_parameterName = "Maximum temperature at 2m";
            m_fileNamePattern = "tmax.2m.gauss.%d.nc";
            m_fileVarName = "tmax";
            m_unit = degK;
        } else if (m_dataId.IsSameAs("tmin2m", false)) {
            m_fStr.hasLevelDim = true;
            m_fStr.singleLevel = true;
            m_parameter = AirTemperature;
            m_parameterName = "Minimum temperature at 2m";
            m_fileNamePattern = "tmin.2m.gauss.%d.nc";
            m_fileVarName = "tmin";
            m_unit = degK;
        } else if (m_dataId.IsSameAs("sktmp", false)) {
            m_parameter = SoilTemperature;
            m_parameterName = "Skin Temperature";
            m_fileNamePattern = "skt.sfc.gauss.%d.nc";
            m_fileVarName = "skt";
            m_unit = degK;
        } else if (m_dataId.IsSameAs("soilw0-10", false)) {
            m_fStr.hasLevelDim = true;
            m_fStr.singleLevel = true;
            m_parameter = SoilMoisture;
            m_parameterName = "Soil moisture (0-10cm)";
            m_fileNamePattern = "soilw.0-10cm.gauss.%d.nc";
            m_fileVarName = "soilw";
            m_unit = fraction;
        } else if (m_dataId.IsSameAs("soilw10-200", false)) {
            m_fStr.hasLevelDim = true;
            m_fStr.singleLevel = true;
            m_parameter = SoilMoisture;
            m_parameterName = "Soil moisture (10-200cm)";
            m_fileNamePattern = "soilw.10-200cm.gauss.%d.nc";
            m_fileVarName = "soilw";
            m_unit = fraction;
        } else if (m_dataId.IsSameAs("tmp0-10", false)) {
            m_fStr.hasLevelDim = true;
            m_fStr.singleLevel = true;
            m_parameter = SoilTemperature;
            m_parameterName = "Temperature of 0-10cm layer";
            m_fileNamePattern = "tmp.0-10cm.gauss.%d.nc";
            m_fileVarName = "tmp";
            m_unit = degK;
        } else if (m_dataId.IsSameAs("tmp10-200", false)) {
            m_fStr.hasLevelDim = true;
            m_fStr.singleLevel = true;
            m_parameter = SoilTemperature;
            m_parameterName = "Temperature of 10-200cm layer";
            m_fileNamePattern = "tmp.10-200cm.gauss.%d.nc";
            m_fileVarName = "tmp";
            m_unit = degK;
        } else if (IsUwindComponent()) {
            m_fStr.hasLevelDim = true;
            m_fStr.singleLevel = true;
            m_parameter = Uwind;
            m_parameterName = "U-wind at 10 m";
            m_fileNamePattern = "uwnd.10m.gauss.%d.nc";
            m_fileVarName = "uwnd";
            m_unit = m_s;
        } else if (IsVwindComponent()) {
            m_fStr.hasLevelDim = true;
            m_fStr.singleLevel = true;
            m_parameter = Vwind;
            m_parameterName = "V-wind at 10 m";
            m_fileNamePattern = "vwnd.10m.gauss.%d.nc";
            m_fileVarName = "vwnd";
            m_unit = m_s;
        } else if (m_dataId.IsSameAs("weasd", false)) {
            m_parameter = SnowWaterEquivalent;
            m_parameterName = "Water equiv. of snow dept";
            m_fileNamePattern = "weasd.sfc.gauss.%d.nc";
            m_fileVarName = "weasd";
            m_unit = kg_m2;
        } else if (m_dataId.IsSameAs("cprat", false)) {
            m_parameter = PrecipitationRate;
            m_parameterName = "Convective precipitation rate";
            m_fileNamePattern = "cprat.sfc.gauss.%d.nc";
            m_fileVarName = "cprat";
            m_unit = kg_m2_s;
        } else if (m_dataId.IsSameAs("dlwrf", false)) {
            m_parameter = Radiation;
            m_parameterName = "Downward longwave radiation flux";
            m_fileNamePattern = "dlwrf.sfc.gauss.%d.nc";
            m_fileVarName = "dlwrf";
            m_unit = W_m2;
        } else if (m_dataId.IsSameAs("dswrf", false)) {
            m_parameter = Radiation;
            m_parameterName = "Downward solar radiation flux";
            m_fileNamePattern = "dswrf.sfc.gauss.%d.nc";
            m_fileVarName = "dswrf";
            m_unit = W_m2;
        } else if (m_dataId.IsSameAs("gflux", false)) {
            m_parameter = Radiation;
            m_parameterName = "Ground heat flux";
            m_fileNamePattern = "gflux.sfc.gauss.%d.nc";
            m_fileVarName = "gflux";
            m_unit = W_m2;
        } else if (m_dataId.IsSameAs("lhtfl", false)) {
            m_parameter = Radiation;
            m_parameterName = "Latent heat net flux";
            m_fileNamePattern = "lhtfl.sfc.gauss.%d.nc";
            m_fileVarName = "lhtfl";
            m_unit = W_m2;
        } else if (m_dataId.IsSameAs("pevpr", false)) {
            m_parameter = PotentialEvaporation;
            m_parameterName = "Potential evaporation rate";
            m_fileNamePattern = "pevpr.sfc.gauss.%d.nc";
            m_fileVarName = "pevpr";
            m_unit = W_m2;
        } else if (IsPrecipitationRate()) {
            m_parameter = PrecipitationRate;
            m_parameterName = "Precipitation rate";
            m_fileNamePattern = "prate.sfc.gauss.%d.nc";
            m_fileVarName = "prate";
            m_unit = kg_m2_s;
        } else if (m_dataId.IsSameAs("shtfl", false)) {
            m_parameter = Radiation;
            m_parameterName = "Sensible heat net flux";
            m_fileNamePattern = "shtfl.sfc.gauss.%d.nc";
            m_fileVarName = "shtfl";
            m_unit = W_m2;
        } else if (m_dataId.IsSameAs("tcdc", false)) {
            m_parameter = CloudCover;
            m_parameterName = "Total cloud cover";
            m_fileNamePattern = "tcdc.eatm.gauss.%d.nc";
            m_fileVarName = "tcdc";
            m_unit = percent;
        } else if (m_dataId.IsSameAs("uflx", false)) {
            m_parameter = MomentumFlux;
            m_parameterName = "Momentum flux (zonal)";
            m_fileNamePattern = "uflx.sfc.gauss.%d.nc";
            m_fileVarName = "uflx";
            m_unit = N_m2;
        } else if (m_dataId.IsSameAs("ugwd", false)) {
            m_parameter = GravityWaveStress;
            m_parameterName = "Zonal gravity wave stress";
            m_fileNamePattern = "ugwd.sfc.gauss.%d.nc";
            m_fileVarName = "ugwd";
            m_unit = N_m2;
        } else if (m_dataId.IsSameAs("ulwrf", false)) {
            m_parameter = Radiation;
            m_parameterName = "Upward Longwave Radiation Flux";
            m_fileNamePattern = "ulwrf.sfc.gauss.%d.nc";
            m_fileVarName = "ulwrf";
            m_unit = W_m2;
        } else if (m_dataId.IsSameAs("uswrf", false)) {
            m_parameter = Radiation;
            m_parameterName = "Upward Solar Radiation Flux";
            m_fileNamePattern = "uswrf.sfc.gauss.%d.nc";
            m_fileVarName = "uswrf";
            m_unit = W_m2;
        } else if (m_dataId.IsSameAs("vflx", false)) {
            m_parameter = MomentumFlux;
            m_parameterName = "Momentum Flux (meridional)";
            m_fileNamePattern = "vflx.sfc.gauss.%d.nc";
            m_fileVarName = "vflx";
            m_unit = N_m2;
        } else if (m_dataId.IsSameAs("vgwd", false)) {
            m_parameter = GravityWaveStress;
            m_parameterName = "Meridional Gravity Wave Stress";
            m_fileNamePattern = "vgwd.sfc.gauss.%d.nc";
            m_fileVarName = "vgwd";
            m_unit = N_m2;
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), m_dataId, m_product);
            return false;
        }

    } else {
        wxLogError(_("Product type not implemented for this reanalysis dataset."));
        return false;
    }

    // Check data ID
    if (m_fileNamePattern.IsEmpty() || m_fileVarName.IsEmpty()) {
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

void asPredictorNcepR2::ListFiles(asTimeArray& timeArray) {
    for (int iYear = timeArray.GetStartingYear(); iYear <= timeArray.GetEndingYear(); iYear++) {
        m_files.push_back(GetFullDirectoryPath() + asStrF(m_fileNamePattern, iYear));
    }
}

void asPredictorNcepR2::ConvertToMjd(a1d& time, double refValue) const {
    time = (time / 24.0) + asTime::GetMJD(1800, 1, 1);
}
