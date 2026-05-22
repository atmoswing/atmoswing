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

#include "asPredictorNasaMerra2Subset.h"
#include "asIncludes.h"

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorNasaMerra2Subset::asPredictorNasaMerra2Subset(const wxString& dataId)
    : asPredictorNasaMerra2(dataId) {
    // Set the basic properties.
    _initialized = false;
    _datasetId = "NASA_MERRA_2_subset";
    _provider = "NASA";
    _transformedBy = "MDISC Data Subset";
    _datasetName = "Modern-Era Retrospective analysis for Research and Applications, Version 2, subset";
}

bool asPredictorNasaMerra2Subset::Init() {
    CheckLevelTypeIsDefined();

    // Get data: http://disc.sci.gsfc.nasa.gov/daac-bin/FTPSubset2.pl
    // Data may not be available for lower layers !!

    // Identify data ID and set the corresponding properties.
    if (_product.IsSameAs("inst6_3d_ana_Np", false) || _product.IsSameAs("ana", false) ||
        _product.IsSameAs("M2I6NPANA", false)) {
        // inst6_3d_ana_Np: 3d,6-Hourly,Instantaneous,Pressure-Level,Analysis,Analyzed Meteorological Fields
        _fStr.hasLevelDim = true;
        if (IsGeopotentialHeight()) {
            _parameter = GeopotentialHeight;
            _parameterName = "Geopotential height";
            _fileVarName = "H";
            _unit = m;
        } else if (IsSpecificHumidity()) {
            _parameter = SpecificHumidity;
            _parameterName = "Specific humidity";
            _fileVarName = "QV";
            _unit = kg_kg;
        } else if (IsAirTemperature()) {
            _parameter = AirTemperature;
            _parameterName = "Air temperature";
            _fileVarName = "T";
            _unit = degK;
        } else if (IsSeaLevelPressure()) {
            _parameter = Pressure;
            _parameterName = "Sea-level pressure";
            _fileVarName = "SLP";
            _unit = Pa;
            _fStr.hasLevelDim = false;
        } else if (IsUwindComponent()) {
            _parameter = Uwind;
            _parameterName = "Eastward wind component";
            _fileVarName = "U";
            _unit = _s;
        } else if (IsVwindComponent()) {
            _parameter = Vwind;
            _parameterName = "Northward wind component";
            _fileVarName = "V";
            _unit = _s;
        } else if (_dataId.IsSameAs("ps", false)) {
            _parameter = Pressure;
            _parameterName = "Surface pressure";
            _fileVarName = "PS";
            _unit = Pa;
            _fStr.hasLevelDim = false;
        } else {
            _parameter = ParameterUndefined;
            _parameterName = "Undefined";
            _fileVarName = _dataId;
            _unit = UnitUndefined;
        }
        _fileNamePattern = _fileVarName + "/MERRA2_*00.inst6_3d_ana_Np.%4d%02d%02d.SUB.nc";

    } else if (_product.IsSameAs("inst3_3d_asm_Np", false) || _product.IsSameAs("asm", false) ||
               _product.IsSameAs("M2I3NPASM", false)) {
        // inst3_3d_asm_Np: 3d,3-Hourly,Instantaneous,Pressure-Level,Assimilation,Assimilated Meteorological Fields
        _fStr.hasLevelDim = true;
        if (IsPotentialVorticity()) {
            _parameter = PotentialVorticity;
            _parameterName = "Ertel's potential vorticity";
            _fileVarName = "EPV";
            _unit = degKm2_kg_s;
        } else if (IsVerticalVelocity()) {
            _parameter = VerticalVelocity;
            _parameterName = "Vertical pressure velocity";
            _fileVarName = "OMEGA";
            _unit = Pa_s;
        } else if (IsRelativeHumidity()) {
            _parameter = RelativeHumidity;
            _parameterName = "Relative humidity after moist";
            _fileVarName = "RH";
            _unit = unitary;
        } else if (IsSeaLevelPressure()) {
            _parameter = Pressure;
            _parameterName = "Sea level pressure";
            _fileVarName = "SLP";
            _unit = Pa;
            _fStr.hasLevelDim = false;
        } else if (IsAirTemperature()) {
            _parameter = AirTemperature;
            _parameterName = "Air temperature";
            _fileVarName = "T";
            _unit = degK;
        } else {
            _parameter = ParameterUndefined;
            _parameterName = "Undefined";
            _fileVarName = _dataId;
            _unit = UnitUndefined;
        }
        _fileNamePattern = _fileVarName + "/MERRA2_*00.inst3_3d_asm_Np.%4d%02d%02d.SUB.nc";

    } else if (_product.IsSameAs("inst1_2d_int_Nx", false) || _product.IsSameAs("M2I1NXINT", false)) {
        // inst1_2d_int_Nx: 2d,1-Hourly,Instantaneous,Single-Level,Assimilation,Vertically Integrated Diagnostics
        _fStr.hasLevelDim = false;
        if (_dataId.IsSameAs("tqi", false)) {
            _parameter = PrecipitableWater;
            _parameterName = "Total precipitable ice water";
            _fileVarName = "TQI";
            _unit = kg_m2;
        } else if (_dataId.IsSameAs("tql", false)) {
            _parameter = PrecipitableWater;
            _parameterName = "Total precipitable liquid water";
            _fileVarName = "TQL";
            _unit = kg_m2;
        } else if (_dataId.IsSameAs("tqv", false)) {
            _parameter = PrecipitableWater;
            _parameterName = "Total precipitable water vapor";
            _fileVarName = "TQV";
            _unit = kg_m2;
        } else {
            _parameter = ParameterUndefined;
            _parameterName = "Undefined";
            _fileVarName = _dataId;
            _unit = UnitUndefined;
        }
        _fileNamePattern = _fileVarName + "/MERRA2_*00.inst1_2d_int_Nx.%4d%02d%02d.SUB.nc";

    } else if (_product.IsSameAs("inst1_2d_asm_Nx", false) || _product.IsSameAs("M2I1NXASM", false)) {
        // inst1_2d_asm_Nx: 2d,3-Hourly,Instantaneous,Single-Level,Assimilation,Single-Level Diagnostics
        _fStr.hasLevelDim = false;
        if (_dataId.IsSameAs("tqi", false)) {
            _parameter = PrecipitableWater;
            _parameterName = "Total precipitable ice water";
            _fileVarName = "TQI";
            _unit = kg_m2;
        } else if (_dataId.IsSameAs("tql", false)) {
            _parameter = PrecipitableWater;
            _parameterName = "Total precipitable liquid water";
            _fileVarName = "TQL";
            _unit = kg_m2;
        } else if (_dataId.IsSameAs("tqv", false)) {
            _parameter = PrecipitableWater;
            _parameterName = "Total precipitable water vapor";
            _fileVarName = "TQV";
            _unit = kg_m2;
        } else if (_dataId.IsSameAs("t10m", false)) {
            _parameter = AirTemperature;
            _parameterName = "10-meter air temperature";
            _fileVarName = "T10M";
            _unit = degK;
        } else {
            _parameter = ParameterUndefined;
            _parameterName = "Undefined";
            _fileVarName = _dataId;
            _unit = UnitUndefined;
        }
        _fileNamePattern = _fileVarName + "/MERRA2_*00.inst1_2d_asm_Nx.%4d%02d%02d.SUB.nc4";

    } else if (_product.IsSameAs("tavg1_2d_flx_Nx", false) || _product.IsSameAs("M2T1NXFLX", false)) {
        // tavg1_2d_flx_Nx:  2d,1-Hourly,Time-Averaged,Single-Level,Assimilation,Surface Flux Diagnostics
        _fStr.hasLevelDim = false;
        if (IsTotalPrecipitation()) {
            _parameter = Precipitation;
            _parameterName = "Total surface precipitation flux";
            _fileVarName = "PRECTOT";
            _unit = kg_m2_s;
        } else {
            _parameter = ParameterUndefined;
            _parameterName = "Undefined";
            _fileVarName = _dataId;
            _unit = UnitUndefined;
        }
        _fileNamePattern = _fileVarName + "/MERRA2_*00.tavg1_2d_flx_Nx.%4d%02d%02d.SUB.nc4";

    } else if (_product.IsSameAs("tavg1_2d_lnd_Nx", false) || _product.IsSameAs("M2T1NXLND", false)) {
        // tavg1_2d_lnd_Nx:
        _fStr.hasLevelDim = false;
        if (IsTotalPrecipitation()) {
            _parameter = Precipitation;
            _parameterName = "Total precipitation land; bias corrected";
            _fileVarName = "PRECTOTLAND";
            _unit = kg_m2_s;
        } else {
            _parameter = ParameterUndefined;
            _parameterName = "Undefined";
            _fileVarName = _dataId;
            _unit = UnitUndefined;
        }
        _fileNamePattern = _fileVarName + "/MERRA2_*00.tavg1_2d_lnd_Nx.%4d%02d%02d.SUB.nc4";

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

void asPredictorNasaMerra2Subset::ListFiles(asTimeArray& timeArray) {
    a1d tArray = timeArray.GetTimeArray();

    Time tLast = asTime::GetTimeStruct(20000);

    for (int i = 0; i < tArray.size(); i++) {
        Time t = asTime::GetTimeStruct(tArray[i]);
        if (tLast.year != t.year || tLast.month != t.month || tLast.day != t.day) {
            wxString path = GetFullDirectoryPath() + asStrF(_fileNamePattern, t.year, t.month, t.day);
            if (t.year < 1992) {
                path.Replace("MERRA2_*00", "MERRA2_100");
            } else if (t.year < 2001) {
                path.Replace("MERRA2_*00", "MERRA2_200");
            } else if (t.year < 2011) {
                path.Replace("MERRA2_*00", "MERRA2_300");
            } else {
                path.Replace("MERRA2_*00", "MERRA2_400");
            }

            _files.push_back(path);
            tLast = t;
        }
    }
}
