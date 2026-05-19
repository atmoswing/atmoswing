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

#include "asPredictorNcepR1.h"

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorNcepR1::asPredictorNcepR1(const wxString& dataId)
    : asPredictor(dataId) {
    // Set the basic properties.
    _datasetId = "NCEP_R1";
    _provider = "NCEP/NCAR";
    _datasetName = "Reanalysis 1";
    _fileType = asFile::Netcdf;
    _strideAllowed = true;
    _nanValues.push_back(32767);
    _nanValues.push_back(936 * std::pow(10.f, 34.f));
    _fStr.dimLatName = "lat";
    _fStr.dimLonName = "lon";
    _fStr.dimTimeName = "time";
    _fStr.dimLevelName = "level";
}

bool asPredictorNcepR1::Init() {
    CheckLevelTypeIsDefined();

    // Identify data ID and set the corresponding properties.
    if (IsPressureLevel()) {
        _fStr.hasLevelDim = true;
        if (IsAirTemperature()) {
            _parameter = AirTemperature;
            _parameterName = "Air Temperature";
            _fileVarName = "air";
            _unit = degK;
        } else if (IsGeopotentialHeight()) {
            _parameter = GeopotentialHeight;
            _parameterName = "Geopotential height";
            _fileVarName = "hgt";
            _unit = m;
        } else if (IsRelativeHumidity()) {
            _parameter = RelativeHumidity;
            _parameterName = "Relative Humidity";
            _fileVarName = "rhum";
            _unit = percent;
        } else if (IsSpecificHumidity()) {
            _parameter = SpecificHumidity;
            _parameterName = "Specific Humidity";
            _fileVarName = "shum";
            _unit = kg_kg;
        } else if (IsVerticalVelocity()) {
            _parameter = VerticalVelocity;
            _parameterName = "Vertical Velocity";
            _fileVarName = "omega";
            _unit = Pa_s;
        } else if (IsUwindComponent()) {
            _parameter = Uwind;
            _parameterName = "U-Wind";
            _fileVarName = "uwnd";
            _unit = _s;
        } else if (IsVwindComponent()) {
            _parameter = Vwind;
            _parameterName = "V-Wind";
            _fileVarName = "vwnd";
            _unit = _s;
        } else {
            _parameter = ParameterUndefined;
            _parameterName = "Undefined";
            _fileVarName = _dataId;
            _unit = UnitUndefined;
        }
        _fileNamePattern = _fileVarName + ".%d.nc";

    } else if (IsSurfaceLevel()) {
        _fStr.hasLevelDim = false;
        if (IsAirTemperature()) {
            _parameter = AirTemperature;
            _parameterName = "Air Temperature";
            _fileNamePattern = "air.sig995.%d.nc";
            _fileVarName = "air";
            _unit = degK;
        } else if (_dataId.IsSameAs("lftx", false)) {
            _parameter = SurfaceLiftedIndex;
            _parameterName = "Surface lifted index";
            _fileNamePattern = "lftx.sfc.%d.nc";
            _fileVarName = "lftx";
            _unit = degK;
        } else if (_dataId.IsSameAs("lftx4", false)) {
            _parameter = SurfaceLiftedIndex;
            _parameterName = "Best (4-layer) lifted index";
            _fileNamePattern = "lftx4.sfc.%d.nc";
            _fileVarName = "lftx4";
            _unit = degK;
        } else if (IsVerticalVelocity()) {
            _parameter = VerticalVelocity;
            _parameterName = "Vertical velocity";
            _fileNamePattern = "omega.sig995.%d.nc";
            _fileVarName = "omega";
            _unit = Pa_s;
        } else if (_dataId.IsSameAs("pottmp", false)) {
            _parameter = PotentialTemperature;
            _parameterName = "Potential temperature";
            _fileNamePattern = "pottmp.sig995.%d.nc";
            _fileVarName = "pottmp";
            _unit = degK;
        } else if (IsPrecipitableWater()) {
            _parameter = PrecipitableWater;
            _parameterName = "Precipitable water";
            _fileNamePattern = "pr_wtr.eatm.%d.nc";
            _fileVarName = "pr_wtr";
            _unit = mm;
        } else if (IsPressure()) {
            _parameter = Pressure;
            _parameterName = "Pressure";
            _fileNamePattern = "pres.sfc.%d.nc";
            _fileVarName = "pres";
            _unit = Pa;
        } else if (IsRelativeHumidity()) {
            _parameter = RelativeHumidity;
            _parameterName = "Relative humidity";
            _fileNamePattern = "rhum.sig995.%d.nc";
            _fileVarName = "rhum";
            _unit = percent;
        } else if (IsSeaLevelPressure()) {
            _parameter = Pressure;
            _parameterName = "Sea level pressure";
            _fileNamePattern = "slp.%d.nc";
            _fileVarName = "slp";
            _unit = Pa;
        } else if (IsUwindComponent()) {
            _parameter = Uwind;
            _parameterName = "U-wind";
            _fileNamePattern = "uwnd.sig995.%d.nc";
            _fileVarName = "uwnd";
            _unit = _s;
        } else if (IsVwindComponent()) {
            _parameter = Vwind;
            _parameterName = "V-wind";
            _fileNamePattern = "vwnd.sig995.%d.nc";
            _fileVarName = "vwnd";
            _unit = _s;
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
            return false;
        }

    } else if (IsSurfaceFluxesLevel() || _product.IsSameAs("surface_gauss", false) ||
               _product.IsSameAs("gauss", false)) {
        _fStr.hasLevelDim = false;
        if (IsAirTemperature()) {
            _parameter = AirTemperature;
            _parameterName = "Air Temperature 2m";
            _fileNamePattern = "air.2m.gauss.%d.nc";
            _fileVarName = "air";
            _unit = degK;
        } else if (_dataId.IsSameAs("pevpr", false)) {
            _parameter = PotentialEvaporation;
            _parameterName = "Potential evaporation rate";
            _fileNamePattern = "pevpr.sfc.gauss.%d.nc";
            _fileVarName = "pevpr";
            _unit = W_m2;
        } else if (IsSpecificHumidity()) {
            _parameter = SpecificHumidity;
            _parameterName = "Specific humidity at 2m";
            _fileNamePattern = "shum.2m.gauss.%d.nc";
            _fileVarName = "shum";
            _unit = kg_kg;
        } else if (_dataId.IsSameAs("soilw0-10", false)) {
            _parameter = SoilMoisture;
            _parameterName = "Soil moisture (0-10cm)";
            _fileNamePattern = "soilw.0-10cm.gauss.%d.nc";
            _fileVarName = "soilw";
            _unit = fraction;
        } else if (_dataId.IsSameAs("soilw10-200", false)) {
            _parameter = SoilMoisture;
            _parameterName = "Soil moisture (10-200cm)";
            _fileNamePattern = "soilw.10-200cm.gauss.%d.nc";
            _fileVarName = "soilw";
            _unit = fraction;
        } else if (_dataId.IsSameAs("sktmp", false)) {
            _parameter = SoilTemperature;
            _parameterName = "Skin Temperature";
            _fileNamePattern = "skt.sfc.gauss.%d.nc";
            _fileVarName = "skt";
            _unit = degK;
        } else if (_dataId.IsSameAs("tmp0-10", false)) {
            _parameter = SoilTemperature;
            _parameterName = "Temperature of 0-10cm layer";
            _fileNamePattern = "tmp.0-10cm.gauss.%d.nc";
            _fileVarName = "tmp";
            _unit = degK;
        } else if (_dataId.IsSameAs("tmp10-200", false)) {
            _parameter = SoilTemperature;
            _parameterName = "Temperature of 10-200cm layer";
            _fileNamePattern = "tmp.10-200cm.gauss.%d.nc";
            _fileVarName = "tmp";
            _unit = degK;
        } else if (_dataId.IsSameAs("tmp300", false)) {
            _parameter = SoilTemperature;
            _parameterName = "Temperature at 300cm";
            _fileNamePattern = "tmp.300cm.gauss.%d.nc";
            _fileVarName = "tmp";
            _unit = degK;
        } else if (IsUwindComponent()) {
            _parameter = Uwind;
            _parameterName = "U-wind at 10 m";
            _fileNamePattern = "uwnd.10m.gauss.%d.nc";
            _fileVarName = "uwnd";
            _unit = _s;
        } else if (IsVwindComponent()) {
            _parameter = Vwind;
            _parameterName = "V-wind at 10 m";
            _fileNamePattern = "vwnd.10m.gauss.%d.nc";
            _fileVarName = "vwnd";
            _unit = _s;
        } else if (_dataId.IsSameAs("weasd", false)) {
            _parameter = SnowWaterEquivalent;
            _parameterName = "Water equiv. of snow dept";
            _fileNamePattern = "weasd.sfc.gauss.%d.nc";
            _fileVarName = "weasd";
            _unit = kg_m2;
        } else if (_dataId.IsSameAs("tmax2m", false)) {
            _parameter = AirTemperature;
            _parameterName = "Maximum temperature at 2m";
            _fileNamePattern = "tmax.2m.gauss.%d.nc";
            _fileVarName = "tmax";
            _unit = degK;
        } else if (_dataId.IsSameAs("tmin2m", false)) {
            _parameter = AirTemperature;
            _parameterName = "Minimum temperature at 2m";
            _fileNamePattern = "tmin.2m.gauss.%d.nc";
            _fileVarName = "tmin";
            _unit = degK;
        } else if (_dataId.IsSameAs("cfnlf", false)) {
            _parameter = Radiation;
            _parameterName = "Cloud forcing net longwave flux";
            _fileNamePattern = "cfnlf.sfc.gauss.%d.nc";
            _fileVarName = "cfnlf";
            _unit = W_m2;
        } else if (_dataId.IsSameAs("cfnsf", false)) {
            _parameter = Radiation;
            _parameterName = "Cloud forcing net solar flux";
            _fileNamePattern = "cfnsf.sfc.gauss.%d.nc";
            _fileVarName = "cfnsf";
            _unit = W_m2;
        } else if (_dataId.IsSameAs("cprat", false)) {
            _parameter = PrecipitationRate;
            _parameterName = "Convective precipitation rate";
            _fileNamePattern = "cprat.sfc.gauss.%d.nc";
            _fileVarName = "cprat";
            _unit = kg_m2_s;
        } else if (_dataId.IsSameAs("csdlf", false)) {
            _parameter = Radiation;
            _parameterName = "Clear sky downward longwave flux";
            _fileNamePattern = "csdlf.sfc.gauss.%d.nc";
            _fileVarName = "csdlf";
            _unit = W_m2;
        } else if (_dataId.IsSameAs("csdsf", false)) {
            _parameter = Radiation;
            _parameterName = "Clear sky downward solar flux";
            _fileNamePattern = "csdsf.sfc.gauss.%d.nc";
            _fileVarName = "csdsf";
            _unit = W_m2;
        } else if (_dataId.IsSameAs("csusf", false)) {
            _parameter = Radiation;
            _parameterName = "Clear sky upward solar flux at surface";
            _fileNamePattern = "csusf.sfc.gauss.%d.nc";
            _fileVarName = "csusf";
            _unit = W_m2;
        } else if (_dataId.IsSameAs("dlwrf", false)) {
            _parameter = Radiation;
            _parameterName = "Downward longwave radiation flux";
            _fileNamePattern = "dlwrf.sfc.gauss.%d.nc";
            _fileVarName = "dlwrf";
            _unit = W_m2;
        } else if (_dataId.IsSameAs("dswrf", false)) {
            _parameter = Radiation;
            _parameterName = "Downward solar radiation flux";
            _fileNamePattern = "dswrf.sfc.gauss.%d.nc";
            _fileVarName = "dswrf";
            _unit = W_m2;
        } else if (_dataId.IsSameAs("gflux", false)) {
            _parameter = Radiation;
            _parameterName = "Ground heat flux";
            _fileNamePattern = "gflux.sfc.gauss.%d.nc";
            _fileVarName = "gflux";
            _unit = W_m2;
        } else if (_dataId.IsSameAs("lhtfl", false)) {
            _parameter = Radiation;
            _parameterName = "Latent heat net flux";
            _fileNamePattern = "lhtfl.sfc.gauss.%d.nc";
            _fileVarName = "lhtfl";
            _unit = W_m2;
        } else if (_dataId.IsSameAs("nbdsf", false)) {
            _parameter = Radiation;
            _parameterName = "Near IR beam downward solar flux";
            _fileNamePattern = "nbdsf.sfc.gauss.%d.nc";
            _fileVarName = "nbdsf";
            _unit = W_m2;
        } else if (_dataId.IsSameAs("nddsf", false)) {
            _parameter = Radiation;
            _parameterName = "Near IR diffuse downward solar flux";
            _fileNamePattern = "nddsf.sfc.gauss.%d.nc";
            _fileVarName = "nddsf";
            _unit = W_m2;
        } else if (_dataId.IsSameAs("nlwrs", false)) {
            _parameter = Radiation;
            _parameterName = "Net longwave radiation";
            _fileNamePattern = "nlwrs.sfc.gauss.%d.nc";
            _fileVarName = "nlwrs";
            _unit = W_m2;
        } else if (_dataId.IsSameAs("nswrs", false)) {
            _parameter = Radiation;
            _parameterName = "Net shortwave radiation";
            _fileNamePattern = "nswrs.sfc.gauss.%d.nc";
            _fileVarName = "nswrs";
            _unit = W_m2;
        } else if (IsPrecipitationRate()) {
            _parameter = PrecipitationRate;
            _parameterName = "Precipitation rate";
            _fileNamePattern = "prate.sfc.gauss.%d.nc";
            _fileVarName = "prate";
            _unit = kg_m2_s;
        } else if (_dataId.IsSameAs("shtfl", false)) {
            _parameter = Radiation;
            _parameterName = "Sensible heat net flux";
            _fileNamePattern = "shtfl.sfc.gauss.%d.nc";
            _fileVarName = "shtfl";
            _unit = W_m2;
        } else if (_dataId.IsSameAs("uflx", false)) {
            _parameter = MomentumFlux;
            _parameterName = "Momentum flux (zonal)";
            _fileNamePattern = "uflx.sfc.gauss.%d.nc";
            _fileVarName = "uflx";
            _unit = N_m2;
        } else if (_dataId.IsSameAs("ugwd", false)) {
            _parameter = GravityWaveStress;
            _parameterName = "Zonal gravity wave stress";
            _fileNamePattern = "ugwd.sfc.gauss.%d.nc";
            _fileVarName = "ugwd";
            _unit = N_m2;
        } else if (_dataId.IsSameAs("ulwrf", false)) {
            _parameter = Radiation;
            _parameterName = "Upward Longwave Radiation Flux";
            _fileNamePattern = "ulwrf.sfc.gauss.%d.nc";
            _fileVarName = "ulwrf";
            _unit = W_m2;
        } else if (_dataId.IsSameAs("uswrf", false)) {
            _parameter = Radiation;
            _parameterName = "Upward Solar Radiation Flux";
            _fileNamePattern = "uswrf.sfc.gauss.%d.nc";
            _fileVarName = "uswrf";
            _unit = W_m2;
        } else if (_dataId.IsSameAs("vbdsf", false)) {
            _parameter = Radiation;
            _parameterName = "Visible Beam Downward Solar Flux";
            _fileNamePattern = "vbdsf.sfc.gauss.%d.nc";
            _fileVarName = "vbdsf";
            _unit = W_m2;
        } else if (_dataId.IsSameAs("vddsf", false)) {
            _parameter = Radiation;
            _parameterName = "Visible Diffuse Downward Solar Flux";
            _fileNamePattern = "vddsf.sfc.gauss.%d.nc";
            _fileVarName = "vddsf";
            _unit = W_m2;
        } else if (_dataId.IsSameAs("vflx", false)) {
            _parameter = MomentumFlux;
            _parameterName = "Momentum Flux, v-component";
            _fileNamePattern = "vflx.sfc.gauss.%d.nc";
            _fileVarName = "vflx";
            _unit = N_m2;
        } else if (_dataId.IsSameAs("vgwd", false)) {
            _parameter = GravityWaveStress;
            _parameterName = "Meridional Gravity Wave Stress";
            _fileNamePattern = "vgwd.sfc.gauss.%d.nc";
            _fileVarName = "vgwd";
            _unit = N_m2;
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
            return false;
        }
    } else {
        wxLogError(_("level type not implemented for this reanalysis dataset."));
        return false;
    }

    // Check data ID
    if (_fileNamePattern.IsEmpty() || _fileVarName.IsEmpty()) {
        wxLogError(_("The provided data ID (%s) does not match any possible option in the dataset %s."), _dataId,
                   _datasetName);
        return false;
    }

    // Check directory is set
    if (GetDirectoryPath().IsEmpty()) {
        wxLogError(_("The path to the directory has not been set for the data %s from the dataset %s."), _dataId,
                   _datasetName);
        return false;
    }

    // Set to initialized
    _initialized = true;

    return true;
}

void asPredictorNcepR1::ListFiles(asTimeArray& timeArray) {
    for (int iYear = timeArray.GetStartingYear(); iYear <= timeArray.GetEndingYear(); iYear++) {
        _files.push_back(GetFullDirectoryPath() + asStrF(_fileNamePattern, iYear));
    }
}

void asPredictorNcepR1::ConvertToMjd(a1d& time, double refValue) const {
    time /= 24.0;
    if (time[0] < 500 * 365) {               // New format
        time += asTime::GetMJD(1800, 1, 1);  // to MJD: add a negative time span
    } else {                                 // Old format
        time += asTime::GetMJD(1, 1, 1);     // to MJD: add a negative time span
    }
}
