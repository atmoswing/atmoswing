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

#include <asTimeArray.h>
#include <asAreaCompGrid.h>


asPredictorNcepCfsrSubset::asPredictorNcepCfsrSubset(const wxString &dataId)
        : asPredictor(dataId)
{
    // Downloaded from http://rda.ucar.edu/datasets/ds093.0/index.html#!cgi-bin/datasets/getSubset?dsnum=093.0&action=customize&_da=y
    // Set the basic properties.
    m_datasetId = "NCEP_CFSR_subset";
    m_provider = "NCEP";
    m_datasetName = "CFSR Subset";
    m_fileType = asFile::Netcdf;
    m_strideAllowed = true;
    m_nanValues.push_back(3.4E38f);
    m_parseTimeReference = true;
    m_fStr.dimLatName = "lat";
    m_fStr.dimLonName = "lon";
    m_fStr.dimTimeName = "time";
    m_fStr.dimLevelName = "level0";
}

bool asPredictorNcepCfsrSubset::Init()
{
    CheckLevelTypeIsDefined();

    // Identify data ID and set the corresponding properties.
    if (IsPressureLevel()) {
        m_fStr.hasLevelDim = true;
        if (IsGeopotentialHeight() || m_dataId.IsSameAs("HGT_L100", false)) {
            m_parameter = GeopotentialHeight;
            m_parameterName = "Geopotential height";
            m_fileVarName = "HGT_L100";
            m_unit = gpm;
        } else if (m_dataId.IsSameAs("gpa", false) || m_dataId.IsSameAs("GP_A_L100", false)) {
            m_parameter = GeopotentialHeight;
            m_parameterName = "Geopotential height anomaly";
            m_fileVarName = "GP_A_L100";
            m_unit = gpm;
            m_fStr.dimLevelName = "level2";
        } else if (IsRelativeHumidity() || m_dataId.IsSameAs("R_H_L100", false)) {
            m_parameter = RelativeHumidity;
            m_parameterName = "Relative humidity";
            m_fileVarName = "R_H_L100";
            m_unit = percent;
        } else if (IsSpecificHumidity() || m_dataId.IsSameAs("SPF_H_L100", false)) {
            m_parameter = SpecificHumidity;
            m_parameterName = "Specific humidity";
            m_fileVarName = "SPF_H_L100";
            m_unit = kg_kg;
        } else if (IsAirTemperature() || m_dataId.IsSameAs("TMP_L100", false)) {
            m_parameter = AirTemperature;
            m_parameterName = "Temperature";
            m_fileVarName = "TMP_L100";
            m_unit = degK;
        } else if (IsVerticalVelocity() || m_dataId.IsSameAs("V_VEL_L100", false)) {
            m_parameter = VerticalVelocity;
            m_parameterName = "Vertical Velocity";
            m_fileVarName = "V_VEL_L100";
            m_unit = Pa_s;
        } else if (IsUwindComponent() || m_dataId.IsSameAs("U_GRD_L100", false)) {
            m_parameter = Uwind;
            m_parameterName = "Eastward wind";
            m_fileVarName = "U_GRD_L100";
            m_unit = m_s;
        } else if (IsVwindComponent() || m_dataId.IsSameAs("V_GRD_L100", false)) {
            m_parameter = Uwind;
            m_parameterName = "Northward wind";
            m_fileVarName = "V_GRD_L100";
            m_unit = m_s;
        } else if (m_dataId.IsSameAs("vpot", false) || m_dataId.IsSameAs("V_POT_L100", false)) {
            m_parameter = VelocityPotential;
            m_parameterName = "Atmosphere horizontal velocity potential";
            m_fileVarName = "V_POT_L100";
            m_unit = m2_s;
            m_fStr.dimLevelName = "level1";
        } else if (m_dataId.IsSameAs("5wavh", false) || m_dataId.IsSameAs("5WAVH_L100", false)) {
            m_parameter = GeopotentialHeight;
            m_parameterName = "5-wave geopotential height";
            m_fileVarName = "5WAVH_L100";
            m_unit = gpm;
            m_fStr.hasLevelDim = false;
        } else if (m_dataId.IsSameAs("5wava", false) || m_dataId.IsSameAs("5WAVA_L100", false)) {
            m_parameter = GeopotentialHeightAnomaly;
            m_parameterName = "5-wave geopotential height anomaly";
            m_fileVarName = "5WAVA_L100";
            m_unit = gpm;
            m_fStr.hasLevelDim = false;
        } else if (m_dataId.IsSameAs("absv", false) || m_dataId.IsSameAs("ABS_V_L100", false)) {
            m_parameter = AbsoluteVorticity;
            m_parameterName = "Atmosphere absolute vorticity";
            m_fileVarName = "ABS_V_L100";
            m_unit = per_s;
        } else if (m_dataId.IsSameAs("clwmr", false) || m_dataId.IsSameAs("CLWMR_L100", false)) {
            m_parameter = CloudWater;
            m_parameterName = "Cloud water mixing ratio";
            m_fileVarName = "CLWMR_L100";
            m_unit = kg_kg;
        } else if (m_dataId.IsSameAs("strm", false) || m_dataId.IsSameAs("STRM_L100", false)) {
            m_parameter = StreamFunction;
            m_parameterName = "Atmosphere horizontal streamfunction";
            m_fileVarName = "STRM_L100";
            m_unit = m2_s;
            m_fStr.dimLevelName = "level1";
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), m_dataId, m_product);
            return false;
        }
        m_fileNamePattern = "pgbhnl.gdas.%4d%02d%02d-%4d%02d%02d.grb2.nc";

    } else if (IsTotalColumnLevel()) {
        m_fStr.hasLevelDim = false;
        if (IsRelativeHumidity() || m_dataId.IsSameAs("R_H_L200", false)) {
            m_parameter = RelativeHumidity;
            m_parameterName = "Relative_humidity";
            m_fileVarName = "R_H_L200";
            m_unit = percent;
        } else if (m_dataId.IsSameAs("cwat", false) || m_dataId.IsSameAs("c_wat", false) || m_dataId.IsSameAs("C_WAT_L200", false)) {
            m_parameter = CloudWater;
            m_parameterName = "Cloud water";
            m_fileVarName = "C_WAT_L200";
            m_unit = kg_m2;
        } else if (IsPrecipitableWater() || m_dataId.IsSameAs("P_WAT_L200", false)) {
            m_parameter = PrecipitableWater;
            m_parameterName = "Atmosphere water vapor content";
            m_fileVarName = "P_WAT_L200";
            m_unit = kg_m2;
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), m_dataId, m_product);
            return false;
        }
        m_fileNamePattern = "pgbhnl.gdas.%4d%02d%02d-%4d%02d%02d.grb2.nc";

    } else if (IsSurfaceLevel()) {
        m_fStr.hasLevelDim = false;
        if (IsPressure()) {
            m_parameter = Pressure;
            m_parameterName = "Pressure";
            m_fileVarName = "PRES_L1";
            m_unit = Pa;
        } else if (m_dataId.IsSameAs("4lftx", false) || m_dataId.IsSameAs("4LFTX_L1", false)) {
            m_parameter = SurfaceLiftedIndex;
            m_parameterName = "Best (4 layer) lifted index";
            m_fileVarName = "4LFTX_L1";
            m_unit = degK;
        } else if (m_dataId.IsSameAs("lftx", false) || m_dataId.IsSameAs("LFT_X_L1", false)) {
            m_parameter = SurfaceLiftedIndex;
            m_parameterName = "Surface lifted index";
            m_fileVarName = "LFT_X_L1";
            m_unit = degK;
        } else if (m_dataId.IsSameAs("cape", false) || m_dataId.IsSameAs("CAPE_L1", false)) {
            m_parameter = CAPE;
            m_parameterName = "Convective available potential energy";
            m_fileVarName = "CAPE_L1";
            m_unit = J_kg;
        } else if (m_dataId.IsSameAs("cin", false) || m_dataId.IsSameAs("CIN_L1", false)) {
            m_parameter = CIN;
            m_parameterName = "Convective inhibition";
            m_fileVarName = "CIN_L1";
            m_unit = J_kg;
        } else if (IsGeopotentialHeight() || m_dataId.IsSameAs("HGT_L1", false)) {
            m_parameter = GeopotentialHeight;
            m_parameterName = "Geopotential height";
            m_fileVarName = "HGT_L1";
            m_unit = gpm;
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), m_dataId, m_product);
            return false;
        }
        m_fileNamePattern = "pgbhnl.gdas.%4d%02d%02d-%4d%02d%02d.grb2.nc";

    } else if (m_product.IsSameAs("msl", false)) {
        m_fStr.hasLevelDim = false;
        if (IsPressure() || m_dataId.IsSameAs("PRES_L101", false)) {
            m_parameter = Pressure;
            m_parameterName = "Pressure";
            m_fileVarName = "PRES_L101";
            m_unit = Pa;
        } else if (IsSeaLevelPressure() || m_dataId.IsSameAs("PRMSL_L101", false)) {
            m_parameter = Pressure;
            m_parameterName = "Mean sea level pressure";
            m_fileVarName = "PRMSL_L101";
            m_unit = Pa;
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), m_dataId, m_product);
            return false;
        }
        m_fileNamePattern = "pgbhnl.gdas.%4d%02d%02d-%4d%02d%02d.grb2.nc";

    } else if (IsIsentropicLevel()) {
        m_fStr.hasLevelDim = true;
        if (m_dataId.IsSameAs("lapr", false) || m_dataId.IsSameAs("LAPR_L107", false)) {
            m_parameter = LapseRate;
            m_parameterName = "Air temperature lapse rate";
            m_fileVarName = "LAPR_L107";
            m_unit = degK_m;
        } else if (m_dataId.IsSameAs("msf", false) || m_dataId.IsSameAs("MNTSF_L107", false)) {
            m_parameter = StreamFunction;
            m_parameterName = "Atmosphere horizontal montgomery streamfunction";
            m_fileVarName = "MNTSF_L107";
            m_unit = m2_s;
        } else if (IsPotentialVorticity() || m_dataId.IsSameAs("PVORT_L107", false)) {
            m_parameter = PotentialVorticity;
            m_parameterName = "Potential vorticity";
            m_fileVarName = "PVORT_L107";
            m_unit = degKm2_kg_s;
        } else if (IsRelativeHumidity() || m_dataId.IsSameAs("R_H_L107", false)) {
            m_parameter = RelativeHumidity;
            m_parameterName = "Relative humidity";
            m_fileVarName = "R_H_L107";
            m_unit = percent;
        } else if (IsAirTemperature() || m_dataId.IsSameAs("TMP_L107", false)) {
            m_parameter = AirTemperature;
            m_parameterName = "Air temperature";
            m_fileVarName = "TMP_L107";
            m_unit = degK;
        } else if (IsUwindComponent() || m_dataId.IsSameAs("U_GRD_L107", false)) {
            m_parameter = Uwind;
            m_parameterName = "Eastward wind";
            m_fileVarName = "U_GRD_L107";
            m_unit = m_s;
        } else if (IsVwindComponent() || m_dataId.IsSameAs("V_GRD_L107", false)) {
            m_parameter = Vwind;
            m_parameterName = "Northward wind";
            m_fileVarName = "V_GRD_L107";
            m_unit = m_s;
        } else if (IsVerticalVelocity() || m_dataId.IsSameAs("V_VEL_L107", false)) {
            m_parameter = VerticalVelocity;
            m_parameterName = "Vertical velocity";
            m_fileVarName = "V_VEL_L107";
            m_unit = Pa_s;
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), m_dataId, m_product);
            return false;
        }
        m_fileNamePattern = "pgbhnl.gdas.%4d%02d%02d-%4d%02d%02d.grb2.nc";

    } else if (IsPVLevel()) {
        m_fStr.hasLevelDim = true;
        if (IsGeopotentialHeight() || m_dataId.IsSameAs("HGT_L109", false)) {
            m_parameter = GeopotentialHeight;
            m_parameterName = "Geopotential height";
            m_fileVarName = "HGT_L109";
            m_unit = gpm;
        } else if (IsPressure() || m_dataId.IsSameAs("PRES_L109", false)) {
            m_parameter = Pressure;
            m_parameterName = "Pressure";
            m_fileVarName = "PRES_L109";
            m_unit = Pa;
        } else if (IsAirTemperature() || m_dataId.IsSameAs("TMP_L109", false)) {
            m_parameter = AirTemperature;
            m_parameterName = "Air temperature";
            m_fileVarName = "TMP_L109";
            m_unit = degK;
        } else if (IsUwindComponent() || m_dataId.IsSameAs("U_GRD_L109", false)) {
            m_parameter = Uwind;
            m_parameterName = "Eastward wind";
            m_fileVarName = "U_GRD_L109";
            m_unit = m_s;
        } else if (IsVwindComponent() || m_dataId.IsSameAs("V_GRD_L109", false)) {
            m_parameter = Vwind;
            m_parameterName = "Northward wind";
            m_fileVarName = "V_GRD_L109";
            m_unit = m_s;
        } else if (m_dataId.IsSameAs("ws", false) || m_dataId.IsSameAs("VW_SH_L109", false)) {
            m_parameter = WindShear;
            m_parameterName = "Wind speed shear";
            m_fileVarName = "VW_SH_L109";
            m_unit = per_s;
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), m_dataId, m_product);
            return false;
        }
        m_fileNamePattern = "pgbhnl.gdas.%4d%02d%02d-%4d%02d%02d.grb2.nc";

    } else if (IsSurfaceFluxesLevel()) {
        m_fStr.hasLevelDim = false;
        if (IsPrecipitationRate()) {
            m_parameter = PrecipitationRate;
            m_parameterName = "Precipitation rate";
            m_fileVarName = "PRATE_L1_Avg_1";
            m_unit = kg_m2_s;
        } else {
            wxLogError(_("No '%s' parameter identified for the provided level type (%s)."), m_dataId, m_product);
            return false;
        }
        m_fileNamePattern = "flxf06.gdas.%4d%02d%02d-%4d%02d%02d.grb2.nc";

    } else if (IsIsentropicLevel()) {
        wxLogError(_("Isentropic levels for CFSR are not implemented yet."));
        return false;

    } else {
        wxLogError(_("level type not implemented for this reanalysis dataset."));
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

void asPredictorNcepCfsrSubset::ListFiles(asTimeArray &timeArray)
{
    auto firstDay = int(std::floor((timeArray.GetStartingDay() - 1.0) / 5.0) * 5.0 + 1.0);
    double fileStart = asTime::GetMJD(timeArray.GetStartingYear(), timeArray.GetStartingMonth(), firstDay);
    double fileEnd = fileStart + 4;

    while (true) {
        Time t1 = asTime::GetTimeStruct(fileStart);
        Time t2 = asTime::GetTimeStruct(fileEnd);
        m_files.push_back(GetFullDirectoryPath() +
                        wxString::Format(m_fileNamePattern, t1.year, t1.month, t1.day, t2.year, t2.month, t2.day));
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

double asPredictorNcepCfsrSubset::ConvertToMjd(double timeValue, double refValue) const
{
    wxASSERT(refValue > 30000);
    wxASSERT(refValue < 70000);

    return refValue + (timeValue / 24.0); // hours to days
}
