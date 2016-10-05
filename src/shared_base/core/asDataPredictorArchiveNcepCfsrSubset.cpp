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

#include "asDataPredictorArchiveNcepCfsrSubset.h"
#include "asTypeDefs.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileNetcdf.h>


asDataPredictorArchiveNcepCfsrSubset::asDataPredictorArchiveNcepCfsrSubset(const wxString &dataId)
        : asDataPredictorArchive(dataId)
{
    // Downloaded from http://rda.ucar.edu/datasets/ds093.0/index.html#!cgi-bin/datasets/getSubset?dsnum=093.0&action=customize&_da=y
    // Set the basic properties.
    m_initialized = false;
    m_datasetId = "NCEP_CFSR_subset";
    m_originalProvider = "NCEP";
    m_datasetName = "CFSR Subset";
    m_originalProviderStart = asTime::GetMJD(1979, 1, 1);
    m_originalProviderEnd = asTime::GetMJD(2011, 3, 1);
    m_timeZoneHours = 0;
    m_timeStepHours = 6;
    m_firstTimeStepHours = NaNDouble;
    m_xAxisShift = 0;
    m_yAxisShift = 0;
    m_fileStructure.dimLatName = "lat";
    m_fileStructure.dimLonName = "lon";
    m_fileStructure.dimTimeName = "time";
    m_fileStructure.dimLevelName = "level0";
}

asDataPredictorArchiveNcepCfsrSubset::~asDataPredictorArchiveNcepCfsrSubset()
{

}

bool asDataPredictorArchiveNcepCfsrSubset::Init()
{
    CheckLevelTypeIsDefined();

    // Identify data ID and set the corresponding properties.
    if (m_product.IsSameAs("pressure_level", false) || m_product.IsSameAs("pressure", false) ||
        m_product.IsSameAs("press", false) || m_product.IsSameAs("pl", false) || m_product.IsSameAs("pgbh", false) ||
        m_product.IsSameAs("pgb", false) || m_product.IsSameAs("pgbhnl", false)) {
        m_fileStructure.hasLevelDimension = true;
        m_subFolder = "pgbhnl";
        m_xAxisStep = 0.5;
        m_yAxisStep = 0.5;
        if (m_dataId.IsSameAs("hgt", false)) {
            m_parameter = GeopotentialHeight;
            m_parameterName = "Geopotential height";
            m_fileVariableName = "HGT_L100";
            m_unit = gpm;
            m_subFolder.Append(DS + "hgt");
        } else if (m_dataId.IsSameAs("gpa", false)) {
            m_parameter = GeopotentialHeight;
            m_parameterName = "Geopotential height anomaly";
            m_fileVariableName = "GP_A_L100";
            m_unit = gpm;
            m_subFolder.Append(DS + "hgt");
            m_fileStructure.dimLevelName = "level1";
        } else if (m_dataId.IsSameAs("mslp", false)) {
            m_parameter = Pressure;
            m_parameterName = "Mean sea level pressure";
            m_fileVariableName = "PRES_L101";
            m_unit = Pa;
            m_subFolder.Append(DS + "mslp");
            m_fileStructure.hasLevelDimension = false;
        } else if (m_dataId.IsSameAs("pwat", false)) {
            m_parameter = PrecipitableWater;
            m_parameterName = "Precipitable water";
            m_fileVariableName = "P_WAT_L200";
            m_unit = kg_m2;
            m_subFolder.Append(DS + "pwat");
            m_fileStructure.hasLevelDimension = false;
        } else if (m_dataId.IsSameAs("rh", false)) {
            m_parameter = RelativeHumidity;
            m_parameterName = "Relative humidity";
            m_fileVariableName = "R_H_L100";
            m_unit = percent;
            m_subFolder.Append(DS + "rh");
        } else if (m_dataId.IsSameAs("temp", false)) {
            m_parameter = AirTemperature;
            m_parameterName = "Temperature";
            m_fileVariableName = "TMP_L100";
            m_unit = degK;
            m_subFolder.Append(DS + "temp");
        } else {
            asThrowException(wxString::Format(_("Parameter '%s' not implemented yet."), m_dataId));
        }
        m_fileNamePattern = "pgbhnl.gdas.%4d%02d%02d-%4d%02d%02d.grb2.nc";

    } else if (m_product.IsSameAs("surface_fluxes", false) || m_product.IsSameAs("fluxes", false) ||
               m_product.IsSameAs("flx", false)) {
        m_fileStructure.hasLevelDimension = false;
        m_subFolder = "flxf06";
        m_xAxisStep = NaNFloat;
        m_yAxisStep = NaNFloat;
        if (m_dataId.IsSameAs("prate", false)) {
            m_parameter = PrecipitationRate;
            m_parameterName = "Precipitation rate";
            m_fileVariableName = "PRATE_L1_Avg_1";
            m_unit = kg_m2_s;
            m_subFolder.Append(DS + "prate");
        } else {
            asThrowException(wxString::Format(_("Parameter '%s' not implemented yet."), m_dataId));
        }
        m_fileNamePattern = "flxf06.gdas.%4d%02d%02d-%4d%02d%02d.grb2.nc";

    } else if (m_product.IsSameAs("isentropic_level", false) || m_product.IsSameAs("isentropic", false) ||
               m_product.IsSameAs("ipvh", false) || m_product.IsSameAs("ipv", false)) {
        asThrowException(_("Isentropic levels for CFSR are not implemented yet."));

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

VectorString asDataPredictorArchiveNcepCfsrSubset::GetListOfFiles(asTimeArray &timeArray) const
{
    VectorString files;

    int firstDay = int(std::floor((timeArray.GetStartingDay() - 1.0) / 5.0) * 5.0 + 1.0);
    double fileStart = asTime::GetMJD(timeArray.GetStartingYear(), timeArray.GetStartingMonth(), firstDay);
    double fileEnd = fileStart + 4;

    while (true) {
        TimeStruct t1 = asTime::GetTimeStruct(fileStart);
        TimeStruct t2 = asTime::GetTimeStruct(fileEnd);
        files.push_back(GetFullDirectoryPath() +
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

        // Do not go beyond the requested time series
        if (fileEnd > timeArray.GetEnd()) {
            while (fileEnd > timeArray.GetEnd()) {
                fileEnd--;
            }
        }

        // Exit condition
        if (fileStart >= timeArray.GetEnd()) {
            break;
        }
    }

    return files;
}

bool asDataPredictorArchiveNcepCfsrSubset::ExtractFromFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea,
                                                           asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    return ExtractFromNetcdfFile(fileName, dataArea, timeArray, compositeData);
}

double asDataPredictorArchiveNcepCfsrSubset::ConvertToMjd(double timeValue, double refValue) const
{
    wxASSERT(refValue>30000);
    wxASSERT(refValue<70000);

    return refValue + (timeValue / 24.0); // hours to days
}
