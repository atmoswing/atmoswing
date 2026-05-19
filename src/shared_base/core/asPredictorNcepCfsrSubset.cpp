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

#include "asPredictorNcepCfsrSubset.h"

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorNcepCfsrSubset::asPredictorNcepCfsrSubset(const wxString& dataId)
    : asPredictor(dataId) {
    // Downloaded from
    // http://rda.ucar.edu/datasets/ds093.0/index.html#!cgi-bin/datasets/getSubset?dsnum=093.0&action=customize&_da=y
    // Set the basic properties.
    _datasetId = "NCEP_CFSR_subset";
    _provider = "NCEP";
    _datasetName = "CFSR Subset";
    _fileType = asFile::Netcdf;
    _strideAllowed = true;
    _nanValues.push_back(3.4E38f);
    _parseTimeReference = true;
    _fStr.dimLatName = "lat";
    _fStr.dimLonName = "lon";
    _fStr.dimTimeName = "time";
    _fStr.dimLevelName = "level0";
}

bool asPredictorNcepCfsrSubset::Init() {
    CheckLevelTypeIsDefined();

    // Identify data ID and set the corresponding properties.
    if (IsPressureLevel()) {
        _fStr.hasLevelDim = true;
        if (IsGeopotentialHeight() || _dataId.IsSameAs("HGT_L100", false)) {
            _parameter = GeopotentialHeight;
            _parameterName = "Geopotential height";
            _fileVarName = "HGT_L100";
            _unit = gpm;
        } else if (_dataId.IsSameAs("gpa", false) || _dataId.IsSameAs("GP_A_L100", false)) {
            _parameter = GeopotentialHeightAnomaly;
            _parameterName = "Geopotential height anomaly";
            _fileVarName = "GP_A_L100";
            _unit = gpm;
            _fStr.dimLevelName = "level2";
        } else if (IsRelativeHumidity() || _dataId.IsSameAs("R_H_L100", false)) {
            _parameter = RelativeHumidity;
            _parameterName = "Relative humidity";
            _fileVarName = "R_H_L100";
            _unit = percent;
        } else if (IsSpecificHumidity() || _dataId.IsSameAs("SPF_H_L100", false)) {
            _parameter = SpecificHumidity;
            _parameterName = "Specific humidity";
            _fileVarName = "SPF_H_L100";
            _unit = kg_kg;
        } else if (IsAirTemperature() || _dataId.IsSameAs("TMP_L100", false)) {
            _parameter = AirTemperature;
            _parameterName = "Temperature";
            _fileVarName = "TMP_L100";
            _unit = degK;
        } else if (IsVerticalVelocity() || _dataId.IsSameAs("V_VEL_L100", false)) {
            _parameter = VerticalVelocity;
            _parameterName = "Vertical Velocity";
            _fileVarName = "V_VEL_L100";
            _unit = Pa_s;
        } else if (IsUwindComponent() || _dataId.IsSameAs("U_GRD_L100", false)) {
            _parameter = Uwind;
            _parameterName = "Eastward wind";
            _fileVarName = "U_GRD_L100";
            _unit = _s;
        } else if (IsVwindComponent() || _dataId.IsSameAs("V_GRD_L100", false)) {
            _parameter = Vwind;
            _parameterName = "Northward wind";
            _fileVarName = "V_GRD_L100";
            _unit = _s;
        } else if (_dataId.IsSameAs("vpot", false) || _dataId.IsSameAs("V_POT_L100", false)) {
            _parameter = VelocityPotential;
            _parameterName = "Atmosphere horizontal velocity potential";
            _fileVarName = "V_POT_L100";
            _unit = m2_s;
            _fStr.dimLevelName = "level1";
        } else if (_dataId.IsSameAs("5wavh", false) || _dataId.IsSameAs("5WAVH_L100", false)) {
            _parameter = GeopotentialHeight;
            _parameterName = "5-wave geopotential height";
            _fileVarName = "5WAVH_L100";
            _unit = gpm;
            _fStr.hasLevelDim = false;
        } else if (_dataId.IsSameAs("5wava", false) || _dataId.IsSameAs("5WAVA_L100", false)) {
            _parameter = GeopotentialHeightAnomaly;
            _parameterName = "5-wave geopotential height anomaly";
            _fileVarName = "5WAVA_L100";
            _unit = gpm;
            _fStr.hasLevelDim = false;
        } else if (_dataId.IsSameAs("absv", false) || _dataId.IsSameAs("ABS_V_L100", false)) {
            _parameter = AbsoluteVorticity;
            _parameterName = "Atmosphere absolute vorticity";
            _fileVarName = "ABS_V_L100";
            _unit = per_s;
        } else if (_dataId.IsSameAs("clwmr", false) || _dataId.IsSameAs("CLWMR_L100", false)) {
            _parameter = CloudWater;
            _parameterName = "Cloud water mixing ratio";
            _fileVarName = "CLWMR_L100";
            _unit = kg_kg;
        } else if (_dataId.IsSameAs("strm", false) || _dataId.IsSameAs("STRM_L100", false)) {
            _parameter = StreamFunction;
            _parameterName = "Atmosphere horizontal streamfunction";
            _fileVarName = "STRM_L100";
            _unit = m2_s;
            _fStr.dimLevelName = "level1";
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
            return false;
        }
        _fileNamePattern = "pgbhnl.gdas.%4d%02d%02d-%4d%02d%02d.grb2.nc";

    } else if (IsTotalColumnLevel()) {
        _fStr.hasLevelDim = false;
        if (IsRelativeHumidity() || _dataId.IsSameAs("R_H_L200", false)) {
            _parameter = RelativeHumidity;
            _parameterName = "Relative_humidity";
            _fileVarName = "R_H_L200";
            _unit = percent;
        } else if (_dataId.IsSameAs("cwat", false) || _dataId.IsSameAs("c_wat", false) ||
                   _dataId.IsSameAs("C_WAT_L200", false)) {
            _parameter = CloudWater;
            _parameterName = "Cloud water";
            _fileVarName = "C_WAT_L200";
            _unit = kg_m2;
        } else if (IsPrecipitableWater() || _dataId.IsSameAs("P_WAT_L200", false)) {
            _parameter = PrecipitableWater;
            _parameterName = "Atmosphere water vapor content";
            _fileVarName = "P_WAT_L200";
            _unit = kg_m2;
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
            return false;
        }
        _fileNamePattern = "pgbhnl.gdas.%4d%02d%02d-%4d%02d%02d.grb2.nc";

    } else if (IsSurfaceLevel()) {
        _fStr.hasLevelDim = false;
        if (IsPressure()) {
            _parameter = Pressure;
            _parameterName = "Pressure";
            _fileVarName = "PRES_L1";
            _unit = Pa;
        } else if (_dataId.IsSameAs("4lftx", false) || _dataId.IsSameAs("4LFTX_L1", false)) {
            _parameter = SurfaceLiftedIndex;
            _parameterName = "Best (4 layer) lifted index";
            _fileVarName = "4LFTX_L1";
            _unit = degK;
        } else if (_dataId.IsSameAs("lftx", false) || _dataId.IsSameAs("LFT_X_L1", false)) {
            _parameter = SurfaceLiftedIndex;
            _parameterName = "Surface lifted index";
            _fileVarName = "LFT_X_L1";
            _unit = degK;
        } else if (_dataId.IsSameAs("cape", false) || _dataId.IsSameAs("CAPE_L1", false)) {
            _parameter = CAPE;
            _parameterName = "Convective available potential energy";
            _fileVarName = "CAPE_L1";
            _unit = J_kg;
        } else if (_dataId.IsSameAs("cin", false) || _dataId.IsSameAs("CIN_L1", false)) {
            _parameter = CIN;
            _parameterName = "Convective inhibition";
            _fileVarName = "CIN_L1";
            _unit = J_kg;
        } else if (IsGeopotentialHeight() || _dataId.IsSameAs("HGT_L1", false)) {
            _parameter = GeopotentialHeight;
            _parameterName = "Geopotential height";
            _fileVarName = "HGT_L1";
            _unit = gpm;
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
            return false;
        }
        _fileNamePattern = "pgbhnl.gdas.%4d%02d%02d-%4d%02d%02d.grb2.nc";

    } else if (_product.IsSameAs("msl", false)) {
        _fStr.hasLevelDim = false;
        if (IsPressure() || _dataId.IsSameAs("PRES_L101", false)) {
            _parameter = Pressure;
            _parameterName = "Pressure";
            _fileVarName = "PRES_L101";
            _unit = Pa;
        } else if (IsSeaLevelPressure() || _dataId.IsSameAs("PRMSL_L101", false)) {
            _parameter = Pressure;
            _parameterName = "Mean sea level pressure";
            _fileVarName = "PRMSL_L101";
            _unit = Pa;
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
            return false;
        }
        _fileNamePattern = "pgbhnl.gdas.%4d%02d%02d-%4d%02d%02d.grb2.nc";

    } else if (IsIsentropicLevel()) {
        _fStr.hasLevelDim = true;
        if (_dataId.IsSameAs("lapr", false) || _dataId.IsSameAs("LAPR_L107", false)) {
            _parameter = LapseRate;
            _parameterName = "Air temperature lapse rate";
            _fileVarName = "LAPR_L107";
            _unit = degK_m;
        } else if (_dataId.IsSameAs("msf", false) || _dataId.IsSameAs("MNTSF_L107", false)) {
            _parameter = StreamFunction;
            _parameterName = "Atmosphere horizontal montgomery streamfunction";
            _fileVarName = "MNTSF_L107";
            _unit = m2_s;
        } else if (IsPotentialVorticity() || _dataId.IsSameAs("PVORT_L107", false)) {
            _parameter = PotentialVorticity;
            _parameterName = "Potential vorticity";
            _fileVarName = "PVORT_L107";
            _unit = degKm2_kg_s;
        } else if (IsRelativeHumidity() || _dataId.IsSameAs("R_H_L107", false)) {
            _parameter = RelativeHumidity;
            _parameterName = "Relative humidity";
            _fileVarName = "R_H_L107";
            _unit = percent;
        } else if (IsAirTemperature() || _dataId.IsSameAs("TMP_L107", false)) {
            _parameter = AirTemperature;
            _parameterName = "Air temperature";
            _fileVarName = "TMP_L107";
            _unit = degK;
        } else if (IsUwindComponent() || _dataId.IsSameAs("U_GRD_L107", false)) {
            _parameter = Uwind;
            _parameterName = "Eastward wind";
            _fileVarName = "U_GRD_L107";
            _unit = _s;
        } else if (IsVwindComponent() || _dataId.IsSameAs("V_GRD_L107", false)) {
            _parameter = Vwind;
            _parameterName = "Northward wind";
            _fileVarName = "V_GRD_L107";
            _unit = _s;
        } else if (IsVerticalVelocity() || _dataId.IsSameAs("V_VEL_L107", false)) {
            _parameter = VerticalVelocity;
            _parameterName = "Vertical velocity";
            _fileVarName = "V_VEL_L107";
            _unit = Pa_s;
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
            return false;
        }
        _fileNamePattern = "pgbhnl.gdas.%4d%02d%02d-%4d%02d%02d.grb2.nc";

    } else if (IsPVLevel()) {
        _fStr.hasLevelDim = true;
        if (IsGeopotentialHeight() || _dataId.IsSameAs("HGT_L109", false)) {
            _parameter = GeopotentialHeight;
            _parameterName = "Geopotential height";
            _fileVarName = "HGT_L109";
            _unit = gpm;
        } else if (IsPressure() || _dataId.IsSameAs("PRES_L109", false)) {
            _parameter = Pressure;
            _parameterName = "Pressure";
            _fileVarName = "PRES_L109";
            _unit = Pa;
        } else if (IsAirTemperature() || _dataId.IsSameAs("TMP_L109", false)) {
            _parameter = AirTemperature;
            _parameterName = "Air temperature";
            _fileVarName = "TMP_L109";
            _unit = degK;
        } else if (IsUwindComponent() || _dataId.IsSameAs("U_GRD_L109", false)) {
            _parameter = Uwind;
            _parameterName = "Eastward wind";
            _fileVarName = "U_GRD_L109";
            _unit = _s;
        } else if (IsVwindComponent() || _dataId.IsSameAs("V_GRD_L109", false)) {
            _parameter = Vwind;
            _parameterName = "Northward wind";
            _fileVarName = "V_GRD_L109";
            _unit = _s;
        } else if (_dataId.IsSameAs("ws", false) || _dataId.IsSameAs("VW_SH_L109", false)) {
            _parameter = WindShear;
            _parameterName = "Wind speed shear";
            _fileVarName = "VW_SH_L109";
            _unit = per_s;
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
            return false;
        }
        _fileNamePattern = "pgbhnl.gdas.%4d%02d%02d-%4d%02d%02d.grb2.nc";

    } else if (IsSurfaceFluxesLevel()) {
        _fStr.hasLevelDim = false;
        if (IsPrecipitationRate()) {
            _parameter = PrecipitationRate;
            _parameterName = "Precipitation rate";
            _fileVarName = "PRATE_L1_Avg_1";
            _unit = kg_m2_s;
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
            return false;
        }
        _fileNamePattern = "flxf06.gdas.%4d%02d%02d-%4d%02d%02d.grb2.nc";

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

void asPredictorNcepCfsrSubset::ListFiles(asTimeArray& timeArray) {
    auto firstDay = int(std::floor((timeArray.GetStartingDay() - 1.0) / 5.0) * 5.0 + 1.0);
    double fileStart = asTime::GetMJD(timeArray.GetStartingYear(), timeArray.GetStartingMonth(), firstDay);
    double fileEnd = fileStart + 4;

    while (true) {
        Time t1 = asTime::GetTimeStruct(fileStart);
        Time t2 = asTime::GetTimeStruct(fileEnd);
        _files.push_back(GetFullDirectoryPath() +
                         asStrF(_fileNamePattern, t1.year, t1.month, t1.day, t2.year, t2.month, t2.day));
        fileStart = fileEnd + 1;
        fileEnd = fileStart + 4;

        // Have to be in the same month
        if (asTime::GetMonth(fileStart) != asTime::GetMonth(fileEnd)) {
            while (asTime::GetMonth(fileStart) != asTime::GetMonth(fileEnd)) {
                fileEnd--;
            }
        }

        // If following day is a 31st, it is also included
        if (asTime::GetDay(fileEnd + 1) == 31) {
            fileEnd++;
        }

        // Exit condition
        if (fileStart >= timeArray.GetEnd()) {
            break;
        }
    }
}

void asPredictorNcepCfsrSubset::ConvertToMjd(a1d& time, double refValue) const {
    time = (time / 24.0) + refValue;
}
