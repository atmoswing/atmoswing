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

#include "asPredictorArchNasaMerra2Subset.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>


asPredictorArchNasaMerra2Subset::asPredictorArchNasaMerra2Subset(const wxString &dataId)
        : asPredictorArchNasaMerra2(dataId)
{
    // Set the basic properties.
    m_initialized = false;
    m_datasetId = "NASA_MERRA_2_subset";
    m_originalProvider = "NASA";
    m_transformedBy = "MDISC Data Subset";
    m_datasetName = "Modern-Era Retrospective analysis for Research and Applications, Version 2, subset";
}

asPredictorArchNasaMerra2Subset::~asPredictorArchNasaMerra2Subset()
{

}

bool asPredictorArchNasaMerra2Subset::Init()
{
    CheckLevelTypeIsDefined();

    // Get data: http://disc.sci.gsfc.nasa.gov/daac-bin/FTPSubset2.pl
    // Data may not be available for lower layers !!

    // Identify data ID and set the corresponding properties.
    if (m_product.IsSameAs("inst6_3d_ana_Np", false) || m_product.IsSameAs("M2I6NPANA", false)) {
        // inst6_3d_ana_Np: 3d,6-Hourly,Instantaneous,Pressure-Level,Analysis,Analyzed Meteorological Fields V5.12.4 (M2I6NPANA.5.12.4)
        m_fileStructure.hasLevelDimension = true;
        m_subFolder = "inst6_3d_ana_Np";
        m_xAxisStep = 0.625;
        m_yAxisStep = 0.5;
        m_timeStepHours = 6;
        if (m_dataId.IsSameAs("h", false)) {
            m_parameter = GeopotentialHeight;
            m_parameterName = "Geopotential height";
            m_fileVariableName = "H";
            m_unit = m;
        } else if (m_dataId.IsSameAs("qv", false)) {
            m_parameter = SpecificHumidity;
            m_parameterName = "Specific humidity";
            m_fileVariableName = "QV";
            m_unit = kg_kg;
        } else if (m_dataId.IsSameAs("t", false)) {
            m_parameter = AirTemperature;
            m_parameterName = "Air temperature";
            m_fileVariableName = "T";
            m_unit = degK;
        } else if (m_dataId.IsSameAs("slp", false)) {
            m_parameter = Pressure;
            m_parameterName = "Sea-level pressure";
            m_fileVariableName = "SLP";
            m_unit = Pa;
            m_fileStructure.hasLevelDimension = false;
        } else if (m_dataId.IsSameAs("u", false)) {
            m_parameter = Uwind;
            m_parameterName = "Eastward wind component";
            m_fileVariableName = "U";
            m_unit = m_s;
        } else if (m_dataId.IsSameAs("v", false)) {
            m_parameter = Vwind;
            m_parameterName = "Northward wind component";
            m_fileVariableName = "V";
            m_unit = m_s;
        } else {
            asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                              m_dataId, m_product));
        }
        m_fileNamePattern = m_fileVariableName + "/MERRA2_*00.inst6_3d_ana_Np.%4d%02d%02d.SUB.nc";

    } else if (m_product.IsSameAs("inst3_3d_asm_Np", false) || m_product.IsSameAs("M2I3NPASM", false)) {
        // inst3_3d_asm_Np: 3d,3-Hourly,Instantaneous,Pressure-Level,Assimilation,Assimilated Meteorological Fields V5.12.4 (M2I3NPASM.5.12.4)
        m_fileStructure.hasLevelDimension = true;
        m_subFolder = "inst3_3d_asm_Np";
        m_xAxisStep = 0.625;
        m_yAxisStep = 0.5;
        m_timeStepHours = 3;
        if (m_dataId.IsSameAs("epv", false)) {
            m_parameter = PotentialVorticity;
            m_parameterName = "Ertel's potential vorticity";
            m_fileVariableName = "EPV";
            m_unit = degKm2_kg_s;
        } else if (m_dataId.IsSameAs("omega", false)) {
            m_parameter = VerticalVelocity;
            m_parameterName = "Vertical pressure velocity";
            m_fileVariableName = "OMEGA";
            m_unit = Pa_s;
        } else if (m_dataId.IsSameAs("rh", false)) {
            m_parameter = RelativeHumidity;
            m_parameterName = "Relative humidity after moist";
            m_fileVariableName = "RH";
            m_unit = unitary;
        } else if (m_dataId.IsSameAs("slp", false)) {
            m_parameter = Pressure;
            m_parameterName = "Sea level pressure";
            m_fileVariableName = "SLP";
            m_unit = Pa;
            m_fileStructure.hasLevelDimension = false;
        } else if (m_dataId.IsSameAs("t", false)) {
            m_parameter = AirTemperature;
            m_parameterName = "Air temperature";
            m_fileVariableName = "T";
            m_unit = degK;
        } else {
            asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                              m_dataId, m_product));
        }
        m_fileNamePattern = m_fileVariableName + "/MERRA2_*00.inst3_3d_asm_Np.%4d%02d%02d.SUB.nc";

    } else if (m_product.IsSameAs("inst1_2d_int_Nx", false) || m_product.IsSameAs("M2I1NXINT", false)) {
        // inst1_2d_int_Nx: 2d,1-Hourly,Instantaneous,Single-Level,Assimilation,Vertically Integrated Diagnostics V5.12.4 (M2I1NXINT.5.12.4)
        m_fileStructure.hasLevelDimension = false;
        m_subFolder = "inst1_2d_int_Nx";
        m_xAxisStep = 0.625;
        m_yAxisStep = 0.5;
        m_timeStepHours = 1;
        if (m_dataId.IsSameAs("tqi", false)) {
            m_parameter = PrecipitableWater;
            m_parameterName = "Total precipitable ice water";
            m_fileVariableName = "TQI";
            m_unit = kg_m2;
        } else if (m_dataId.IsSameAs("tql", false)) {
            m_parameter = PrecipitableWater;
            m_parameterName = "Total precipitable liquid water";
            m_fileVariableName = "TQL";
            m_unit = kg_m2;
        } else if (m_dataId.IsSameAs("tqv", false)) {
            m_parameter = PrecipitableWater;
            m_parameterName = "Total precipitable water vapor";
            m_fileVariableName = "TQV";
            m_unit = kg_m2;
        } else {
            asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                              m_dataId, m_product));
        }
        m_fileNamePattern = m_fileVariableName + "/MERRA2_*00.inst1_2d_int_Nx.%4d%02d%02d.SUB.nc";

    } else if (m_product.IsSameAs("inst1_2d_asm_Nx", false) || m_product.IsSameAs("M2I1NXASM", false)) {
        // inst1_2d_asm_Nx: 2d,3-Hourly,Instantaneous,Single-Level,Assimilation,Single-Level Diagnostics V5.12.4 (M2I1NXASM)
        m_fileStructure.hasLevelDimension = false;
        m_subFolder = "inst1_2d_asm_Nx";
        m_xAxisStep = 0.625;
        m_yAxisStep = 0.5;
        m_timeStepHours = 1;
        if (m_dataId.IsSameAs("tqi", false)) {
            m_parameter = PrecipitableWater;
            m_parameterName = "Total precipitable ice water";
            m_fileVariableName = "TQI";
            m_unit = kg_m2;
        } else if (m_dataId.IsSameAs("tql", false)) {
            m_parameter = PrecipitableWater;
            m_parameterName = "Total precipitable liquid water";
            m_fileVariableName = "TQL";
            m_unit = kg_m2;
        } else if (m_dataId.IsSameAs("tqv", false)) {
            m_parameter = PrecipitableWater;
            m_parameterName = "Total precipitable water vapor";
            m_fileVariableName = "TQV";
            m_unit = kg_m2;
        } else if (m_dataId.IsSameAs("t10m", false)) {
            m_parameter = AirTemperature;
            m_parameterName = "10-meter air temperature";
            m_fileVariableName = "T10M";
            m_unit = degK;
        } else {
            asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                              m_dataId, m_product));
        }
        m_fileNamePattern = m_fileVariableName + "/MERRA2_*00.inst1_2d_asm_Nx.%4d%02d%02d.SUB.nc4";

    } else if (m_product.IsSameAs("tavg1_2d_flx_Nx", false) || m_product.IsSameAs("M2T1NXFLX", false)) {
        // tavg1_2d_flx_Nx:  2d,1-Hourly,Time-Averaged,Single-Level,Assimilation,Surface Flux Diagnostics V5.12.4 (M2T1NXFLX)
        m_fileStructure.hasLevelDimension = false;
        m_subFolder = "tavg1_2d_flx_Nx";
        m_xAxisStep = 0.625;
        m_yAxisStep = 0.5;
        m_timeStepHours = 1;
        m_firstTimeStepHours = 0.5;
        if (m_dataId.IsSameAs("prectot", false)) {
            m_parameter = Precipitation;
            m_parameterName = "Total surface precipitation flux";
            m_fileVariableName = "PRECTOT";
            m_unit = kg_m2_s;
        } else {
            asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                              m_dataId, m_product));
        }
        m_fileNamePattern = m_fileVariableName + "/MERRA2_*00.tavg1_2d_flx_Nx.%4d%02d%02d.SUB.nc4";

    } else if (m_product.IsSameAs("tavg1_2d_lnd_Nx", false) || m_product.IsSameAs("M2T1NXLND", false)) {
        // tavg1_2d_lnd_Nx:
        m_fileStructure.hasLevelDimension = false;
        m_subFolder = "tavg1_2d_lnd_Nx";
        m_xAxisStep = 0.625;
        m_yAxisStep = 0.5;
        m_timeStepHours = 1;
        m_firstTimeStepHours = 0.5;
        if (m_dataId.IsSameAs("prectot", false)) {
            m_parameter = Precipitation;
            m_parameterName = "Total precipitation land; bias corrected";
            m_fileVariableName = "PRECTOTLAND";
            m_unit = kg_m2_s;
        } else {
            asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                              m_dataId, m_product));
        }
        m_fileNamePattern = m_fileVariableName + "/MERRA2_*00.tavg1_2d_lnd_Nx.%4d%02d%02d.SUB.nc4";

    } else {
        asThrowException(_("level type not implemented for this reanalysis dataset."));
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

void asPredictorArchNasaMerra2Subset::ListFiles(asTimeArray &timeArray)
{
    a1d tArray = timeArray.GetTimeArray();

    Time tLast = asTime::GetTimeStruct(20000);

    for (int i = 0; i < tArray.size(); i++) {
        Time t = asTime::GetTimeStruct(tArray[i]);
        if (tLast.year != t.year || tLast.month != t.month || tLast.day != t.day) {

            wxString path = GetFullDirectoryPath() + wxString::Format(m_fileNamePattern, t.year, t.month, t.day);
            if (t.year < 1992) {
                path.Replace("MERRA2_*00", "MERRA2_100");
            } else if (t.year < 2001) {
                path.Replace("MERRA2_*00", "MERRA2_200");
            } else if (t.year < 2011) {
                path.Replace("MERRA2_*00", "MERRA2_300");
            } else {
                path.Replace("MERRA2_*00", "MERRA2_400");
            }

            m_files.push_back(path);
            tLast = t;
        }
    }
}
