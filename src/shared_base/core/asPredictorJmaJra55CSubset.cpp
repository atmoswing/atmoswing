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

#include "asPredictorJmaJra55CSubset.h"

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorJmaJra55CSubset::asPredictorJmaJra55CSubset(const wxString& dataId)
    : asPredictorJmaJra55Subset(dataId) {
    // Set the basic properties.
    _datasetId = "JMA_JRA_55C_subset";
    _datasetName = "Japanese 55-year Reanalysis Conventional";
    _fileType = asFile::Netcdf;
}

bool asPredictorJmaJra55CSubset::Init() {
    CheckLevelTypeIsDefined();

    // Get data:
    // http://rda.ucar.edu/datasets/ds628.2/index.html#!cgi-bin/datasets/getSubset?dsnum=628.2&listAction=customize&_da=y

    // Identify data ID and set the corresponding properties.
    if (IsPressureLevel() || _product.IsSameAs("anl_p125", false)) {
        // JRA-55 6-Hourly 1.25 Degree Isobaric Analysis Fields
        _fStr.hasLevelDim = true;
        _fileNamePattern = _product + ".C.";
        _fStr.dimLatName = "g0_lat_2";
        _fStr.dimLonName = "g0_lon_3";
        _fStr.dimTimeName = "initial_time0_hours";
        _fStr.dimLevelName = "lv_ISBL1";
        _monthlyFiles = true;
        if (IsGeopotentialHeight()) {
            _parameter = GeopotentialHeight;
            _parameterName = "Geopotential Height";
            _fileVarName = "HGT_GDS0_ISBL";
            _unit = gpm;
            _fileNamePattern.Append("007_hgt");
        } else if (IsRelativeHumidity()) {
            _parameter = RelativeHumidity;
            _parameterName = "Relative humidity";
            _fileVarName = "RH_GDS0_ISBL";
            _unit = percent;
            _fileNamePattern.Append("052_rh");
        } else if (IsAirTemperature()) {
            _parameter = AirTemperature;
            _parameterName = "Temperature";
            _fileVarName = "TMP_GDS0_ISBL";
            _unit = degK;
            _fileNamePattern.Append("011_tmp");
        } else if (IsVerticalVelocity()) {
            _parameter = VerticalVelocity;
            _parameterName = "Vertical velocity";
            _fileVarName = "VVEL_GDS0_ISBL";
            _unit = Pa_s;
            _fileNamePattern.Append("039_vvel");
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
            return false;
        }
        _fileNamePattern.Append(".%4d%02d01*.nc");

    } else if (IsSurfaceLevel() || _product.IsSameAs("anl_surf125", false)) {
        // JRA-55 6-Hourly 1.25 Degree Surface Analysis Fields
        _fStr.hasLevelDim = false;
        _fileNamePattern = _product + ".C.";
        _fStr.dimLatName = "g0_lat_1";
        _fStr.dimLonName = "g0_lon_2";
        _fStr.dimTimeName = "initial_time0_hours";
        _monthlyFiles = false;
        if (IsSeaLevelPressure()) {
            _parameter = Pressure;
            _parameterName = "Pressure reduced to MSL";
            _fileVarName = "PRMSL_GDS0_MSL";
            _unit = Pa;
            _fileNamePattern.Append("002_prmsl");
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
            return false;
        }
        _fileNamePattern.Append(".%4d%02d01*.nc");

    } else if (IsTotalColumnLevel() || _product.IsSameAs("anl_column125", false)) {
        // JRA-55 6-Hourly 1.25 Degree Total Column Analysis Fields
        _fStr.hasLevelDim = false;
        _fileNamePattern = _product + ".C.";
        _fStr.dimLatName = "g0_lat_1";
        _fStr.dimLonName = "g0_lon_2";
        _fStr.dimTimeName = "initial_time0_hours";
        _monthlyFiles = false;
        if (IsPrecipitableWater()) {
            _parameter = PrecipitableWater;
            _parameterName = "Precipitable water";
            _fileVarName = "PWAT_GDS0_EATM";
            _unit = kg_m2;
            _fileNamePattern.Append("054_pwat");
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
            return false;
        }
        _fileNamePattern.Append(".%4d%02d01*.nc");

    } else if (_product.IsSameAs("fcst_phy2m125", false)) {
        // JRA-55 3-Hourly 1.25 Degree 2-Dimensional Average Diagnostic Fields
        _fStr.hasLevelDim = false;
        _fStr.dimLatName = "g0_lat_1";
        _fStr.dimLonName = "g0_lon_2";
        _fStr.dimTimeName = "initial_time0_hours";
        _monthlyFiles = false;
        if (_dataId.IsSameAs("tprat3h", false)) {
            _parameter = Precipitation;
            _parameterName = "Total precipitation";
            _fileVarName = "TPRAT_GDS0_SFC_ave3h";
            _unit = mm_d;
            _fileNamePattern.Append("fcst_phy2m125.C.061_tprat");
        } else if (_dataId.IsSameAs("tprat6h", false)) {
            _parameter = Precipitation;
            _parameterName = "Total precipitation";
            _fileVarName = "TPRAT_GDS0_SFC_ave3h";
            _unit = mm_d;
            _fileNamePattern.Append("fcst_phy2m125.C.061_tprat");
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
            return false;
        }
        _fileNamePattern.Append(".%4d%02d01*.nc");

    } else if (IsIsentropicLevel() || _product.IsSameAs("anl_isentrop125", false)) {
        // JRA-55 6-Hourly 1.25 Degree Isentropic Analysis Fields
        _fStr.hasLevelDim = true;
        _fileNamePattern = _product + ".C.";
        _fStr.dimLatName = "g0_lat_2";
        _fStr.dimLonName = "g0_lon_3";
        _fStr.dimTimeName = "initial_time0_hours";
        _fStr.dimLevelName = "lv_THEL1";
        _monthlyFiles = true;
        if (IsPotentialVorticity()) {
            _parameter = PotentialVorticity;
            _parameterName = "Potential vorticity";
            _fileVarName = "pVOR_GDS0_THEL";
            _unit = degKm2_kg_s;
            _fileNamePattern.Append("004_pvor");
        } else if (IsGeopotentialHeight()) {
            _parameter = GeopotentialHeight;
            _parameterName = "Geopotential Height";
            _fileVarName = "HGT_GDS0_THEL";
            _unit = gpm;
            _fileNamePattern.Append("007_hgt");
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), _dataId, _product);
            return false;
        }
        _fileNamePattern.Append(".%4d%02d01*.nc");

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
