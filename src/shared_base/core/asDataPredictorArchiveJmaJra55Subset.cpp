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

#include "asDataPredictorArchiveJmaJra55Subset.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileNetcdf.h>
#include <wx/dir.h>


asDataPredictorArchiveJmaJra55Subset::asDataPredictorArchiveJmaJra55Subset(const wxString &dataId)
        : asDataPredictorArchive(dataId)
{
    // Set the basic properties.
    m_initialized = false;
    m_datasetId = "JMA_JRA_55_subset";
    m_originalProvider = "JMA";
    m_transformedBy = "NCAR/UCAR Data Subset";
    m_datasetName = "Japanese 55-year Reanalysis";
    m_originalProviderStart = asTime::GetMJD(1958, 1, 1);
    m_originalProviderEnd = NaNDouble;
    m_timeZoneHours = 0;
    m_timeStepHours = 6;
    m_firstTimeStepHours = 0;
    m_nanValues.push_back(32767);
    m_nanValues.push_back(std::pow(10.f, 20.f));
    m_xAxisShift = 0;
    m_yAxisShift = 0;
    m_fileStructure.dimLatName = "g0_lat_2";
    m_fileStructure.dimLonName = "g0_lon_3";
    m_fileStructure.dimTimeName = "initial_time0_hours";
    m_fileStructure.dimLevelName = "lv_ISBL1";
}

asDataPredictorArchiveJmaJra55Subset::~asDataPredictorArchiveJmaJra55Subset()
{

}

bool asDataPredictorArchiveJmaJra55Subset::Init()
{
    CheckLevelTypeIsDefined();

    // Identify data ID and set the corresponding properties.
    if (m_product.IsSameAs("anl_p125", false)) {
        m_fileStructure.hasLevelDimension = true;
        m_subFolder = "anl_p125";
        m_xAxisStep = 1.250;
        m_yAxisStep = 1.250;
        m_fileNamePattern = m_subFolder + ".";
        if (m_dataId.IsSameAs("hgt", false)) {
            m_parameter = GeopotentialHeight;
            m_parameterName = "Geopotential Height";
            m_fileVariableName = "HGT_GDS0_ISBL";
            m_unit = gpm;
            m_fileNamePattern.Append("007_hgt");
        } else {
            asThrowException(wxString::Format(_("No '%s' parameter identified for the provided level type (%s)."),
                                              m_dataId, m_product));
        }
        m_fileNamePattern.Append(".%4d%02d01*.nc");

    } else {
        asThrowException(_("level type not implemented for this reanalysis dataset."));
    }

    // Check data ID
    if (m_fileNamePattern.IsEmpty() || m_fileVariableName.IsEmpty()) {
        asLogError(wxString::Format(_("The provided data ID (%s) does not match any possible option in the dataset %s."),
                                    m_dataId, m_datasetName));
        return false;
    }

    // Check directory is set
    if (GetDirectoryPath().IsEmpty()) {
        asLogError(wxString::Format(_("The path to the directory has not been set for the data %s from the dataset %s."),
                                    m_dataId, m_datasetName));
        return false;
    }

    // Set to initialized
    m_initialized = true;

    return true;
}

VectorString asDataPredictorArchiveJmaJra55Subset::GetListOfFiles(asTimeArray &timeArray) const
{
    VectorString files;

    for (int i_year = timeArray.GetFirstDayYear(); i_year <= timeArray.GetLastDayYear(); i_year++) {
        int firstMonth = 1;
        int lastMonth = 12;
        if (i_year==timeArray.GetFirstDayYear()) {
            firstMonth = timeArray.GetFirstDayMonth();
        }
        if (i_year==timeArray.GetLastDayYear()) {
            lastMonth = timeArray.GetLastDayMonth();
        }

        for (int i_month = firstMonth; i_month <= lastMonth; ++i_month) {
            wxString filePattern = wxString::Format(m_fileNamePattern, i_year, i_month);
            wxArrayString listFiles;
            size_t nbFiles = wxDir::GetAllFiles(GetFullDirectoryPath(), &listFiles, filePattern);

            if (nbFiles==0) {
                asThrowException(wxString::Format(_("No JRA-55 file found for this pattern : %s."), filePattern));
            } else if (nbFiles>1) {
                asThrowException(wxString::Format(_("Multiple JRA-55 files found for this pattern : %s."), filePattern));
            }

            files.push_back(listFiles.Item(0));
        }
    }

    return files;
}

bool asDataPredictorArchiveJmaJra55Subset::ExtractFromFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea,
                                                            asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    return ExtractFromNetcdfFile(fileName, dataArea, timeArray, compositeData);
}

double asDataPredictorArchiveJmaJra55Subset::ConvertToMjd(double timeValue, double refValue) const
{
    timeValue = (timeValue / 24.0); // hours to days
    timeValue += asTime::GetMJD(1800, 1, 1); // to MJD: add a negative time span

    return timeValue;
}
