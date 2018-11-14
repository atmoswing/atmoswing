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

#include "asPredictorArchNasaMerra2.h"

#include <asTimeArray.h>
#include <asAreaCompGrid.h>


asPredictorArchNasaMerra2::asPredictorArchNasaMerra2(const wxString &dataId)
        : asPredictorArch(dataId)
{
    // Downloaded from http://disc.sci.gsfc.nasa.gov/daac-bin/FTPSubset2.pl
    // Set the basic properties.
    m_datasetId = "NASA_MERRA_2";
    m_provider = "NASA";
    m_datasetName = "Modern-Era Retrospective analysis for Research and Applications, Version 2";
    m_fileType = asFile::Netcdf;
    m_strideAllowed = true;
    m_parseTimeReference = true;
    m_nanValues.push_back(std::pow(10.f, 15.f));
    m_nanValues.push_back(std::pow(10.f, 15.f) - 1);
    m_fStr.dimLatName = "lat";
    m_fStr.dimLonName = "lon";
    m_fStr.dimTimeName = "time";
    m_fStr.dimLevelName = "lev";
}

bool asPredictorArchNasaMerra2::Init()
{
    CheckLevelTypeIsDefined();

    // Identify data ID and set the corresponding properties.
    if (m_product.IsSameAs("inst6_3d_ana_Np", false) || m_product.IsSameAs("M2I6NPANA", false)) {
        m_fStr.hasLevelDim = true;
        m_subFolder = "inst6_3d_ana_Np";
        if (m_dataId.IsSameAs("h", false)) {
            m_parameter = GeopotentialHeight;
            m_parameterName = "Geopotential height";
            m_fileVarName = "H";
            m_unit = m;
        } else if (m_dataId.IsSameAs("t", false)) {
            m_parameter = AirTemperature;
            m_parameterName = "Air temperature";
            m_fileVarName = "T";
            m_unit = degK;
        } else if (m_dataId.IsSameAs("slp", false)) {
            m_parameter = Pressure;
            m_parameterName = "Sea-level pressure";
            m_fileVarName = "SLP";
            m_unit = Pa;
        } else {
            m_parameter = ParameterUndefined;
            m_parameterName = "Undefined";
            m_fileVarName = m_dataId;
            m_unit = UnitUndefined;
        }
        m_fileNamePattern = "%4d/%02d/MERRA2_*00.inst6_3d_ana_Np.%4d%02d%02d.nc4";

    } else {
        asThrowException(_("level type not implemented for this reanalysis dataset."));
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

void asPredictorArchNasaMerra2::ListFiles(asTimeArray &timeArray)
{
    a1d tArray = timeArray.GetTimeArray();

    Time tLast = asTime::GetTimeStruct(20000);

    for (int i = 0; i < tArray.size(); i++) {
        Time t = asTime::GetTimeStruct(tArray[i]);
        if (tLast.year != t.year || tLast.month != t.month || tLast.day != t.day) {

            wxString path = GetFullDirectoryPath() + wxString::Format(m_fileNamePattern, t.year, t.month, t.year,
                                                                      t.month, t.day);
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

double asPredictorArchNasaMerra2::ConvertToMjd(double timeValue, double refValue) const
{
    wxASSERT(refValue > 30000);
    wxASSERT(refValue < 70000);

    return refValue + (timeValue / 1440.0); // minutes to days
}
