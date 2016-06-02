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

#include "asDataPredictorArchiveNoaaOisst2.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileNetcdf.h>


asDataPredictorArchiveNoaaOisst2::asDataPredictorArchiveNoaaOisst2(const wxString &dataId)
        : asDataPredictorArchive(dataId)
{
    // Set the basic properties.
    m_initialized = false;
    m_dataId = dataId;
    m_datasetId = "NOAA_OISST_v2";
    m_originalProvider = "NOAA";
    m_finalProvider = "NOAA";
    m_finalProviderWebsite = "http://www.esrl.noaa.gov/psd/data/gridded/data.noaa.oisst.v2.html";
    m_finalProviderFTP = "ftp://eclipse.ncdc.noaa.gov/pub/OI-daily-v2";
    m_datasetName = "Optimum Interpolation Sea Surface Temperature, version 2";
    m_originalProviderStart = asTime::GetMJD(1982, 1, 1);
    m_originalProviderEnd = NaNDouble;
    m_timeZoneHours = 0;
    m_timeStepHours = 24;
    m_firstTimeStepHours = 12;
    m_nanValues.push_back(32767);
    m_nanValues.push_back(936 * std::pow(10.f, 34.f));
    m_xAxisShift = 0.125;
    m_yAxisShift = 0.125;
    m_xAxisStep = 0.25;
    m_yAxisStep = 0.25;
    m_subFolder = wxEmptyString;
    m_fileNamePattern = "%d/AVHRR/sst4-path-eot.%4d%02d%02d.nc";
    m_fileAxisLatName = "lat";
    m_fileAxisLonName = "lon";

    // Identify data ID and set the corresponding properties.
    if (m_dataId.IsSameAs("sst", false)) {
        m_dataParameter = SeaSurfaceTemperature;
        m_fileVariableName = "sst";
        m_unit = degC;
    } else if (m_dataId.IsSameAs("sst_anom", false)) {
        m_dataParameter = SeaSurfaceTemperatureAnomaly;
        m_fileVariableName = "anom";
        m_unit = degC;
    } else {
        m_dataParameter = NoParameter;
        m_fileVariableName = wxEmptyString;
        m_unit = NoUnit;
    }

}

asDataPredictorArchiveNoaaOisst2::~asDataPredictorArchiveNoaaOisst2()
{

}

bool asDataPredictorArchiveNoaaOisst2::Init()
{
    // Check data ID
    if (m_fileNamePattern.IsEmpty() || m_fileVariableName.IsEmpty()) {
        asLogError(
                wxString::Format(_("The provided data ID (%s) does not match any possible option in the dataset %s."),
                                 m_dataId, m_datasetName));
        return false;
    }

    // Check directory is set
    if (m_directoryPath.IsEmpty()) {
        asLogError(
                wxString::Format(_("The path to the directory has not been set for the data %s from the dataset %s."),
                                 m_dataId, m_datasetName));
        return false;
    }

    // Set to initialized
    m_initialized = true;

    return true;
}

bool asDataPredictorArchiveNoaaOisst2::ExtractFromFiles(asGeoAreaCompositeGrid *&dataArea, asTimeArray &timeArray,
                                                        VVArray2DFloat &compositeData)
{
    // Get requested dates
    double dateFirst = timeArray.GetFirst();
    double dateLast = timeArray.GetLast();

#if wxUSE_GUI
    asDialogProgressBar progressBar(_("Loading data from files.\n"), dateLast - dateFirst);
#endif

    // Loop through the files
    for (double date = dateFirst; date <= dateLast; date++) {
        // Build the file path (ex: %d/AVHRR/sst4-path-eot.%4d%02d%02d.nc)
        wxString fileName = wxString::Format(m_fileNamePattern, asTime::GetYear(date), asTime::GetYear(date),
                                             asTime::GetMonth(date), asTime::GetDay(date));
        wxString fileFullPath = m_directoryPath + fileName;

#if wxUSE_GUI
        // Update the progress bar
        wxString fileNameMessage = wxString::Format(_("Loading data from files.\nFile: %s"), fileName);
        if (!progressBar.Update(date - dateFirst, fileNameMessage)) {
            asLogWarning(_("The process has been canceled by the user."));
            return false;
        }
#endif

        // Open the NetCDF file
        ThreadsManager().CritSectionNetCDF().Enter();
        asFileNetcdf ncFile(fileFullPath, asFileNetcdf::ReadOnly);
        if (!ncFile.Open()) {
            ThreadsManager().CritSectionNetCDF().Leave();
            return false;
        }

        // Get some attributes
        float dataAddOffset = ncFile.GetAttFloat("add_offset", m_fileVariableName);
        if (asTools::IsNaN(dataAddOffset))
            dataAddOffset = 0;
        float dataScaleFactor = ncFile.GetAttFloat("scale_factor", m_fileVariableName);
        if (asTools::IsNaN(dataScaleFactor))
            dataScaleFactor = 1;
        bool scalingNeeded = true;
        if (dataAddOffset == 0 && dataScaleFactor == 1)
            scalingNeeded = false;

        // Get full axes from the netcdf file
        Array1DFloat axisDataLon(ncFile.GetVarLength(m_fileAxisLonName));
        ncFile.GetVar(m_fileAxisLonName, &axisDataLon[0]);
        Array1DFloat axisDataLat(ncFile.GetVarLength(m_fileAxisLatName));
        ncFile.GetVar(m_fileAxisLatName, &axisDataLat[0]);

        // Adjust axes if necessary
        dataArea = AdjustAxes(dataArea, axisDataLon, axisDataLat, compositeData);

        // Containers for extraction
        VectorInt vectIndexLengthLat;
        VectorInt vectIndexLengthLon;
        VectorBool vectLoad360;
        VVectorShort vectData;
        VVectorShort vectData360;

        for (int i_area = 0; i_area < (int) compositeData.size(); i_area++) {
            // Check if necessary to load the data of lon=360 (so lon=0)
            bool load360 = false;

            int indexStartLon, indexStartLat, indexLengthLon, indexLengthLat;
            if (dataArea) {
                // Get the spatial extent
                float lonMin = dataArea->GetXaxisCompositeStart(i_area);
                float lonMax = dataArea->GetXaxisCompositeEnd(i_area);
                float latMinStart = dataArea->GetYaxisCompositeStart(i_area);
                float latMinEnd = dataArea->GetYaxisCompositeEnd(i_area);

                // The dimensions lengths
                indexLengthLon = dataArea->GetXaxisCompositePtsnb(i_area);
                indexLengthLat = dataArea->GetYaxisCompositePtsnb(i_area);

                if (lonMax == dataArea->GetAxisXmax()) {
                    // Correction if the lon 360 degrees is required (doesn't exist)
                    load360 = true;
                    for (int i_check = 0; i_check < (int) compositeData.size(); i_check++) {
                        // If so, already loaded in another composite
                        if (dataArea->GetComposite(i_check).GetXmin() == 0) {
                            load360 = false;
                        }
                    }
                    indexLengthLon--;
                }

                // Get the spatial indices of the desired data
                indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLon.size() - 1],
                                                           lonMin, 0.01f);
                if (indexStartLon == asOUT_OF_RANGE) {
                    // If not found, try with negative angles
                    indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLon.size() - 1],
                                                               lonMin - 360, 0.01f);
                }
                if (indexStartLon == asOUT_OF_RANGE) {
                    // If not found, try with angles above 360 degrees
                    indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLon.size() - 1],
                                                               lonMin + 360, 0.01f);
                }
                if (indexStartLon < 0) {
                    asLogError(wxString::Format("Cannot find lonMin (%f) in the array axisDataLon ([0]=%f -> [%d]=%f) ",
                                                lonMin, axisDataLon[0], (int) axisDataLon.size(),
                                                axisDataLon[axisDataLon.size() - 1]));
                    ncFile.Close();
                    ThreadsManager().CritSectionNetCDF().Leave();
                    return false;
                }
                wxASSERT_MSG(indexStartLon >= 0,
                             wxString::Format("axisDataLon[0] = %f, &axisDataLon[%d] = %f & lonMin = %f",
                                              axisDataLon[0], (int) axisDataLon.size(),
                                              axisDataLon[axisDataLon.size() - 1], lonMin));

                int indexStartLat1 = asTools::SortedArraySearch(&axisDataLat[0], &axisDataLat[axisDataLat.size() - 1],
                                                                latMinStart, 0.01f);
                int indexStartLat2 = asTools::SortedArraySearch(&axisDataLat[0], &axisDataLat[axisDataLat.size() - 1],
                                                                latMinEnd, 0.01f);
                wxASSERT_MSG(indexStartLat1 >= 0,
                             wxString::Format("Looking for %g in %g to %g", latMinStart, axisDataLat[0],
                                              axisDataLat[axisDataLat.size() - 1]));
                wxASSERT_MSG(indexStartLat2 >= 0,
                             wxString::Format("Looking for %g in %g to %g", latMinEnd, axisDataLat[0],
                                              axisDataLat[axisDataLat.size() - 1]));
                indexStartLat = wxMin(indexStartLat1, indexStartLat2);
            } else {
                indexStartLon = 0;
                indexStartLat = 0;
                indexLengthLon = m_lonPtsnb;
                indexLengthLat = m_latPtsnb;
            }

            // Create the arrays to receive the data
            VectorShort data, data360;

            // Resize the arrays to store the new data
            int totLength = indexLengthLat * indexLengthLon;
            wxASSERT(totLength > 0);
            data.resize(totLength);

            // Get the indices for data
            size_t indexStartData[2] = {0, 0};
            size_t indexCountData[2] = {0, 0};
            ptrdiff_t indexStrideData[2] = {0, 0};

            // Set the indices for data
            indexStartData[0] = indexStartLat;
            indexStartData[1] = indexStartLon;
            indexCountData[0] = indexLengthLat;
            indexCountData[1] = indexLengthLon;
            indexStrideData[0] = m_latIndexStep;
            indexStrideData[1] = m_lonIndexStep;

            // In the netCDF Common Data Language, variables are printed with the outermost dimension first and the innermost dimension last.
            ncFile.GetVarSample(m_fileVariableName, indexStartData, indexCountData, indexStrideData, &data[0]);

            // Load data at lon = 360 degrees
            if (load360) {
                // Resize the arrays to store the new data
                int totlength360 = indexLengthLat * 1;
                data360.resize(totlength360);

                // Set the indices
                indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLon.size() - 1], 360,
                                                           0.01f);
                if (indexStartLon == asOUT_OF_RANGE) {
                    // If not found, try with negative angles
                    indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLon.size() - 1], 0,
                                                               0.01f);
                }

                indexStartData[1] = indexStartLon;
                indexCountData[1] = 1;
                indexStrideData[1] = 1;

                // Load data at 0 degrees (corresponds to 360 degrees)
                ncFile.GetVarSample(m_fileVariableName, indexStartData, indexCountData, indexStrideData, &data360[0]);
            }

            // Keep data for later treatment
            vectIndexLengthLat.push_back(indexLengthLat);
            vectIndexLengthLon.push_back(indexLengthLon);
            vectLoad360.push_back(load360);
            vectData.push_back(data);
            vectData360.push_back(data360);
        }

        // Close the nc file
        ncFile.Close();
        ThreadsManager().CritSectionNetCDF().Leave();

        // Allocate space into compositeData if not already done
        if (compositeData[0].capacity() == 0) {
            int totSize = 0;
            for (int i_area = 0; i_area < (int) compositeData.size(); i_area++) {
                int indexLengthLat = vectIndexLengthLat[i_area];
                int indexLengthLon = vectIndexLengthLon[i_area];
                totSize += m_time.size() * indexLengthLat * (indexLengthLon + 1); // +1 in case of a border
            }
            compositeData.reserve(totSize);
        }

        // Transfer data
        for (int i_area = 0; i_area < (int) compositeData.size(); i_area++) {
            // Extract data
            int indexLengthLat = vectIndexLengthLat[i_area];
            int indexLengthLon = vectIndexLengthLon[i_area];
            bool load360 = vectLoad360[i_area];
            VectorShort data = vectData[i_area];
            VectorShort data360 = vectData360[i_area];

            // Loop to extract the data from the array
            int ind = 0;
            Array2DFloat latlonData;
            if (load360) {
                latlonData = Array2DFloat(indexLengthLat, indexLengthLon + 1);
            } else {
                latlonData = Array2DFloat(indexLengthLat, indexLengthLon);
            }

            for (int i_lat = 0; i_lat < indexLengthLat; i_lat++) {
                for (int i_lon = 0; i_lon < indexLengthLon; i_lon++) {
                    ind = i_lon + i_lat * indexLengthLon;

                    if (scalingNeeded) {
                        latlonData(i_lat, i_lon) = (float) data[ind] * dataScaleFactor + dataAddOffset;
                    } else {
                        latlonData(i_lat, i_lon) = (float) data[ind];
                    }

                    // Check if not NaN
                    bool notNan = true;
                    for (size_t i_nan = 0; i_nan < m_nanValues.size(); i_nan++) {
                        if ((float) data[ind] == m_nanValues[i_nan] || latlonData(i_lat, i_lon) == m_nanValues[i_nan]) {
                            notNan = false;
                        }
                    }
                    if (!notNan) {
                        latlonData(i_lat, i_lon) = NaNFloat;
                    }
                }

                if (load360) {
                    ind = i_lat;

                    if (scalingNeeded) {
                        latlonData(i_lat, indexLengthLon) = (float) data360[ind] * dataScaleFactor + dataAddOffset;
                    } else {
                        latlonData(i_lat, indexLengthLon) = (float) data360[ind];
                    }

                    // Check if not NaN
                    bool notNan = true;
                    for (size_t i_nan = 0; i_nan < m_nanValues.size(); i_nan++) {
                        if ((float) data360[ind] == m_nanValues[i_nan] ||
                            latlonData(i_lat, indexLengthLon) == m_nanValues[i_nan]) {
                            notNan = false;
                        }
                    }
                    if (!notNan) {
                        latlonData(i_lat, indexLengthLon) = NaNFloat;
                    }
                }
            }
            compositeData[i_area].push_back(latlonData);

            data.clear();
            data360.clear();
        }
    }

    return true;
}

