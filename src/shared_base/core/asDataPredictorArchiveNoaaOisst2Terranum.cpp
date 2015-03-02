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
 * The Original Software is AtmoSwing. The Initial Developer of the
 * Original Software is Pascal Horton of the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 * Portions Copyright 2013 Pascal Horton, Terr@num.
 */

#include "asDataPredictorArchiveNoaaOisst2Terranum.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileNetcdf.h>


asDataPredictorArchiveNoaaOisst2Terranum::asDataPredictorArchiveNoaaOisst2Terranum(const wxString &dataId)
:
asDataPredictorArchive(dataId)
{
    // Set the basic properties.
    m_initialized = false;
    m_dataId = dataId;
    m_datasetId = "NOAA_OISST_v2_terranum";
    m_originalProvider = "NOAA";
    m_finalProvider = "Terranum";
    m_finalProviderWebsite = "http://www.terranum.ch";
    m_finalProviderFTP = "";
    m_datasetName = "Optimum Interpolation Sea Surface Temperature, version 2, subset from terranum";
    m_originalProviderStart = asTime::GetMJD(1982, 1, 1);
    m_originalProviderEnd = NaNDouble;
    m_timeZoneHours = 0;
    m_timeStepHours = 24;
    m_firstTimeStepHours = 12;
    m_nanValues.push_back(32767);
    m_nanValues.push_back(936*std::pow(10.f,34.f));
    m_xaxisShift = 0.125;
    m_yaxisShift = 0.125;
    m_xaxisStep = 1;
    m_yaxisStep = 1;
    m_subFolder = wxEmptyString;
    m_fileAxisLatName = "lat";
    m_fileAxisLonName = "lon";
    m_fileAxisTimeName = "time";

    // Identify data ID and set the corresponding properties.
    if (m_dataId.IsSameAs("sst", false))
    {
        m_dataParameter = SeaSurfaceTemperature;
        m_fileNamePattern = "sst_1deg.nc";
        m_fileVariableName = "sst";
        m_unit = degC;
    }
    else if (m_dataId.IsSameAs("sst_anom", false))
    {
        m_dataParameter = SeaSurfaceTemperatureAnomaly;
        m_fileNamePattern = "sst_anom_1deg.nc";
        m_fileVariableName = "anom";
        m_unit = degC;
    }
    else
    {
        m_dataParameter = NoDataParameter;
        m_fileVariableName = wxEmptyString;
        m_unit = NoDataUnit;
    }
}

asDataPredictorArchiveNoaaOisst2Terranum::~asDataPredictorArchiveNoaaOisst2Terranum()
{

}

bool asDataPredictorArchiveNoaaOisst2Terranum::Init()
{
    // Check data ID
    if (m_fileNamePattern.IsEmpty() || m_fileVariableName.IsEmpty()) {
        asLogError(wxString::Format(_("The provided data ID (%s) does not match any possible option in the dataset %s."), m_dataId.c_str(), m_datasetName.c_str()));
        return false;
    }

    // Check directory is set
    if (m_directoryPath.IsEmpty()) {
        asLogError(wxString::Format(_("The path to the directory has not been set for the data %s from the dataset %s."), m_dataId.c_str(), m_datasetName.c_str()));
        return false;
    }

    // Set to initialized
    m_initialized = true;

    return true;
}

VectorString asDataPredictorArchiveNoaaOisst2Terranum::GetDataIdList()
{
    VectorString list;

    list.push_back("sst"); // Sea surface temperature
    list.push_back("sst_anom"); // Anomaly

    return list;
}

VectorString asDataPredictorArchiveNoaaOisst2Terranum::GetDataIdDescriptionList()
{
    VectorString list;

    list.push_back("Sea surface temperature");
    list.push_back("Sea surface temperature anomaly");

    return list;
}

bool asDataPredictorArchiveNoaaOisst2Terranum::ExtractFromFiles(asGeoAreaCompositeGrid *& dataArea, asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    // Build the file path
    wxString fileFullPath = m_directoryPath + m_fileNamePattern;

    // Open the NetCDF file
    ThreadsManager().CritSectionNetCDF().Enter();
    asFileNetcdf ncFile(fileFullPath, asFileNetcdf::ReadOnly);
    if(!ncFile.Open())
    {
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Get some attributes
    float dataAddOffset = ncFile.GetAttFloat("add_offset", m_fileVariableName);
    if (asTools::IsNaN(dataAddOffset)) dataAddOffset = 0;
    float dataScaleFactor = ncFile.GetAttFloat("scale_factor", m_fileVariableName);
    if (asTools::IsNaN(dataScaleFactor)) dataScaleFactor = 1;
    bool scalingNeeded = true;
    if (dataAddOffset==0 && dataScaleFactor==1) scalingNeeded = false;

    // Get full axes from the netcdf file
    Array1DFloat axisDataLon(ncFile.GetVarLength(m_fileAxisLonName));
    ncFile.GetVar(m_fileAxisLonName, &axisDataLon[0]);
    Array1DFloat axisDataLat(ncFile.GetVarLength(m_fileAxisLatName));
    ncFile.GetVar(m_fileAxisLatName, &axisDataLat[0]);
    Array1DFloat axisDataLevel;
    
    // Adjust axes if necessary
    dataArea = AdjustAxes(dataArea, axisDataLon, axisDataLat, compositeData);
        
    // Time array takes ages to load !! Avoid if possible. Get the first value of the time array.
    size_t axisDataTimeLength = ncFile.GetVarLength(m_fileAxisTimeName);
    double valFirstTime = ncFile.GetVarOneDouble(m_fileAxisTimeName, 0);
    valFirstTime = (valFirstTime/24.0); // hours to days
    valFirstTime += asTime::GetMJD(1,1,1); // to MJD: add a negative time span
    double valLastTime = ncFile.GetVarOneDouble(m_fileAxisTimeName, axisDataTimeLength-1);
    valLastTime = (valLastTime/24.0); // hours to days
    valLastTime += asTime::GetMJD(1,1,1); // to MJD: add a negative time span

    // Check requested time array
    if(timeArray.GetFirst()<valFirstTime)
    {
        asLogError(wxString::Format(_("The requested data starts before (%s) the actual dataset (%s)"), asTime::GetStringTime(timeArray.GetFirst()).c_str(), asTime::GetStringTime(valFirstTime).c_str()));
        return false;
    }
    if(timeArray.GetLast()>valLastTime)
    {
        asLogError(wxString::Format(_("The requested data ends after (%s) the actual dataset (%s)"), asTime::GetStringTime(timeArray.GetLast()).c_str(), asTime::GetStringTime(valLastTime).c_str()));
        return false;
    }

    // Get start and end of the serie
    double timeStart = timeArray.GetFirst();
    double timeEnd = timeArray.GetLast();

    // Get the time length
    double timeArrayIndexStart = timeArray.GetIndexFirstAfter(timeStart);
    double timeArrayIndexEnd = timeArray.GetIndexFirstBefore(timeEnd);
    int indexLengthTime = timeArrayIndexEnd-timeArrayIndexStart+1;
    int indexLengthTimeArray = indexLengthTime;

    // Correct the time start and end
    size_t indexStartTime = 0;
    int cutStart = timeArrayIndexStart;
    int cutEnd = 0;
    while (valFirstTime<timeArray[timeArrayIndexStart])
    {
        valFirstTime += m_timeStepHours/24.0;
        indexStartTime++;
    }
    if (indexStartTime+indexLengthTime>axisDataTimeLength)
    {
        indexLengthTime--;
        cutEnd++;
    }

    // Containers for extraction
    VectorInt vectIndexLengthLat;
    VectorInt vectIndexLengthLon;
    VVectorShort vectData;

    for (int i_area = 0; i_area<(int)compositeData.size(); i_area++)
    {
        int indexStartLon, indexStartLat, indexLengthLon, indexLengthLat;
        if (dataArea)
        {
            // Get the spatial extent
            float lonMin = dataArea->GetXaxisCompositeStart(i_area);
            float latMinStart = dataArea->GetYaxisCompositeStart(i_area);
            float latMinEnd = dataArea->GetYaxisCompositeEnd(i_area);

            // The dimensions lengths
            indexLengthLon = dataArea->GetXaxisCompositePtsnb(i_area);
            indexLengthLat = dataArea->GetYaxisCompositePtsnb(i_area);

            // Get the spatial indices of the desired data
            indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLon.size()-1], lonMin, 0.01f);
            if(indexStartLon==asOUT_OF_RANGE)
            {
                // If not found, try with negative angles
                indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLon.size()-1], lonMin-360, 0.01f);
            }
            if(indexStartLon==asOUT_OF_RANGE)
            {
                // If not found, try with angles above 360 degrees
                indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLon.size()-1], lonMin+360, 0.01f);
            }
            if(indexStartLon<0)
            {
                asLogError(wxString::Format("Cannot find lonMin (%f) in the array axisDataLon ([0]=%f -> [%d]=%f) ", lonMin, axisDataLon[0], (int)axisDataLon.size(), axisDataLon[axisDataLon.size()-1]));
                return false;
            }
            wxASSERT_MSG(indexStartLon>=0, wxString::Format("axisDataLon[0] = %f, &axisDataLon[%d] = %f & lonMin = %f", axisDataLon[0], (int)axisDataLon.size(), axisDataLon[axisDataLon.size()-1], lonMin));

            int indexStartLat1 = asTools::SortedArraySearch(&axisDataLat[0], &axisDataLat[axisDataLat.size()-1], latMinStart, 0.01f);
            int indexStartLat2 = asTools::SortedArraySearch(&axisDataLat[0], &axisDataLat[axisDataLat.size()-1], latMinEnd, 0.01f);
            wxASSERT_MSG(indexStartLat1>=0, wxString::Format("Looking for %g in %g to %g", latMinStart, axisDataLat[0], axisDataLat[axisDataLat.size()-1]));
            wxASSERT_MSG(indexStartLat2>=0, wxString::Format("Looking for %g in %g to %g", latMinEnd, axisDataLat[0], axisDataLat[axisDataLat.size()-1]));
            indexStartLat = wxMin(indexStartLat1, indexStartLat2);
        }
        else
        {
            indexStartLon = 0;
            indexStartLat = 0;
            indexLengthLon = m_lonPtsnb;
            indexLengthLat = m_latPtsnb;
        }

        // Create the arrays to receive the data
        VectorShort data;

        // Resize the arrays to store the new data
        int totLength = indexLengthTimeArray * indexLengthLat * indexLengthLon;
        wxASSERT(totLength>0);
        data.resize(totLength);

        // Fill empty begining with NaNs
        int indexBegining = 0;
        if(cutStart>0)
        {
            int latlonlength = indexLengthLat*indexLengthLon;
            for (int i_empty=0; i_empty<cutStart; i_empty++)
            {
                for (int i_emptylatlon=0; i_emptylatlon<latlonlength; i_emptylatlon++)
                {
                    data[indexBegining] = NaNFloat;
                    indexBegining++;
                }
            }
        }

        // Fill empty end with NaNs
        int indexEnd = indexLengthTime * indexLengthLat * indexLengthLon - 1;
        if(cutEnd>0)
        {
            int latlonlength = indexLengthLat*indexLengthLon;
            for (int i_empty=0; i_empty<cutEnd; i_empty++)
            {
                for (int i_emptylatlon=0; i_emptylatlon<latlonlength; i_emptylatlon++)
                {
                    indexEnd++;
                    data[indexEnd] = NaNFloat;
                }
            }
        }

        // Get the indices for data
        size_t indexStartData[3] = {0,0,0};
        indexStartData[0] = indexStartTime;
        indexStartData[1] = indexStartLat;
        indexStartData[2] = indexStartLon;
        size_t indexCountData[3] = {0,0,0};
        indexCountData[0] = indexLengthTime;
        indexCountData[1] = indexLengthLat;
        indexCountData[2] = indexLengthLon;
        ptrdiff_t indexStrideData[3] = {0,0,0};
        indexStrideData[0] = m_timeIndexStep;
        indexStrideData[1] = m_latIndexStep;
        indexStrideData[2] = m_lonIndexStep;

        // In the netCDF Common Data Language, variables are printed with the outermost dimension first and the innermost dimension last.
        ncFile.GetVarSample(m_fileVariableName, indexStartData, indexCountData, indexStrideData, &data[indexBegining]);

        // Keep data for later treatment
        vectIndexLengthLat.push_back(indexLengthLat);
        vectIndexLengthLon.push_back(indexLengthLon);
        vectData.push_back(data);
    }

    // Close the nc file
    ncFile.Close();
    ThreadsManager().CritSectionNetCDF().Leave();

    // Allocate space into compositeData if not already done
    if (compositeData[0].capacity()==0)
    {
        int totSize = 0;
        for (int i_area = 0; i_area<(int)compositeData.size(); i_area++)
        {
            int indexLengthLat = vectIndexLengthLat[i_area];
            int indexLengthLon = vectIndexLengthLon[i_area];
            totSize += m_time.size() * indexLengthLat * indexLengthLon;
        }
        compositeData.reserve(totSize);
    }

    // Transfer data
    for (int i_area = 0; i_area<(int)compositeData.size(); i_area++)
    {
        // Extract data
        int indexLengthLat = vectIndexLengthLat[i_area];
        int indexLengthLon = vectIndexLengthLon[i_area];
        VectorShort data = vectData[i_area];

        // Loop to extract the data from the array
        int ind = 0;
        for (int i_time=0; i_time<indexLengthTimeArray; i_time++)
        {
            Array2DFloat latlonData(indexLengthLat,indexLengthLon);
            for (int i_lat=0; i_lat<indexLengthLat; i_lat++)
            {
                for (int i_lon=0; i_lon<indexLengthLon; i_lon++)
                {
                    ind = i_lon;
                    ind += i_lat * indexLengthLon;
                    ind += i_time * indexLengthLon * indexLengthLat;

                    if (scalingNeeded)
                    {
                        latlonData(i_lat,i_lon) = (float)data[ind] * dataScaleFactor + dataAddOffset;
                    }
                    else
                    {
                        latlonData(i_lat,i_lon) = (float)data[ind];
                    }

                    // Check if not NaN
                    bool notNan = true;
                    for (size_t i_nan=0; i_nan<m_nanValues.size(); i_nan++)
                    {
                        if ((float)data[ind]==m_nanValues[i_nan] || latlonData(i_lat,i_lon)==m_nanValues[i_nan])
                        {
                            notNan = false;
                        }
                    }
                    if (!notNan)
                    {
                        latlonData(i_lat,i_lon) = NaNFloat;
                    }
                }
            }
            compositeData[i_area].push_back(latlonData);
        }
        data.clear();
    }

    return true;
}

