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

#include "asDataPredictorArchiveNcepReanalysis2.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileNetcdf.h>


asDataPredictorArchiveNcepReanalysis2::asDataPredictorArchiveNcepReanalysis2(const wxString &dataId)
:
asDataPredictorArchive(dataId)
{
    // Set the basic properties.
    m_Initialized = false;
    m_DataId = dataId;
    m_DatasetId = "NCEP_Reanalysis_v2";
    m_OriginalProvider = "NCEP/DOE";
    m_FinalProvider = "NCEP/DOE";
    m_FinalProviderWebsite = "http://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis2.html";
    m_FinalProviderFTP = "ftp://ftp.cdc.noaa.gov/DataSets/ncep.reanalysis2";
    m_DatasetName = "Reanalysis 2";
    m_OriginalProviderStart = asTime::GetMJD(1979, 1, 1);
    m_OriginalProviderEnd = NaNDouble;
    m_TimeZoneHours = 0;
    m_TimeStepHours = 6;
    m_FirstTimeStepHours = 0;
    m_NanValues.push_back(32767);
    m_NanValues.push_back(936*std::pow(10.f,34.f));
    m_CoordinateSystem = WGS84;
    m_XaxisShift = 0;
    m_YaxisShift = 0;
    m_XaxisStep = 2.5;
    m_YaxisStep = 2.5;
    m_FileAxisLatName = "lat";
    m_FileAxisLonName = "lon";
    m_FileAxisTimeName = "time";
    m_FileAxisLevelName = "level";

    // Identify data ID and set the corresponding properties.
    if (m_DataId.IsSameAs("hgt", false))
    {
        m_DataParameter = GeopotentialHeight;
        m_SubFolder = "pressure";
        m_FileNamePattern = "hgt.%d.nc";
        m_FileVariableName = "hgt";
        m_Unit = m;
    }
    else if (m_DataId.IsSameAs("air", false))
    {
        m_DataParameter = AirTemperature;
        m_SubFolder = "pressure";
        m_FileNamePattern = "air.%d.nc";
        m_FileVariableName = "air";
        m_Unit = degK;
    }
    else if (m_DataId.IsSameAs("omega", false))
    {
        m_DataParameter = Omega;
        m_SubFolder = "pressure";
        m_FileNamePattern = "omega.%d.nc";
        m_FileVariableName = "omega";
        m_Unit = PascalsPerSec;
    }
    else if (m_DataId.IsSameAs("rhum", false))
    {
        m_DataParameter = RelativeHumidity;
        m_SubFolder = "pressure";
        m_FileNamePattern = "rhum.%d.nc";
        m_FileVariableName = "rhum";
        m_Unit = percent;
    }
    else if (m_DataId.IsSameAs("shum", false))
    {
        m_DataParameter = SpecificHumidity;
        m_SubFolder = "pressure";
        m_FileNamePattern = "shum.%d.nc";
        m_FileVariableName = "shum";
        m_Unit = kgPerKg;
    }
    else if (m_DataId.IsSameAs("uwnd", false))
    {
        m_DataParameter = Uwind;
        m_SubFolder = "pressure";
        m_FileNamePattern = "uwnd.%d.nc";
        m_FileVariableName = "uwnd";
        m_Unit = mPerSec;
    }
    else if (m_DataId.IsSameAs("vwnd", false))
    {
        m_DataParameter = Vwind;
        m_SubFolder = "pressure";
        m_FileNamePattern = "vwnd.%d.nc";
        m_FileVariableName = "vwnd";
        m_Unit = mPerSec;
    }
    else if (m_DataId.IsSameAs("surf_prwtr", false))
    {
        m_DataParameter = PrecipitableWater;
        m_SubFolder = "surface";
        m_FileNamePattern = "pr_wtr.eatm.%d.nc";
        m_FileVariableName = "pr_wtr";
        m_Unit = mm;
    }
    else
    {
        m_DataParameter = NoDataParameter;
        m_SubFolder = wxEmptyString;
        m_FileNamePattern = wxEmptyString;
        m_FileVariableName = wxEmptyString;
        m_Unit = NoDataUnit;
    }
}

asDataPredictorArchiveNcepReanalysis2::~asDataPredictorArchiveNcepReanalysis2()
{

}

bool asDataPredictorArchiveNcepReanalysis2::Init()
{
    // Check data ID
    if (m_FileNamePattern.IsEmpty() || m_FileVariableName.IsEmpty()) {
        asLogError(wxString::Format(_("The provided data ID (%s) does not match any possible option in the dataset %s."), m_DataId.c_str(), m_DatasetName.c_str()));
        return false;
    }

    // Check directory is set
    if (m_DirectoryPath.IsEmpty()) {
        asLogError(wxString::Format(_("The path to the directory has not been set for the data %s from the dataset %s."), m_DataId.c_str(), m_DatasetName.c_str()));
        return false;
    }

    // Set to initialized
    m_Initialized = true;

    return true;
}

VectorString asDataPredictorArchiveNcepReanalysis2::GetDataIdList()
{
    VectorString list;

    list.push_back("hgt"); // Geopotential Height
    list.push_back("air"); // Air Temperature
    list.push_back("omega"); // Omega (Vertical Velocity)
    list.push_back("rhum"); // Relative Humidity
    list.push_back("shum"); // Specific Humidity
    list.push_back("uwnd"); // U-Wind
    list.push_back("vwnd"); // V-Wind
    list.push_back("surf_prwtr"); // Precipitable Water

    return list;
}

VectorString asDataPredictorArchiveNcepReanalysis2::GetDataIdDescriptionList()
{
    VectorString list;

    list.push_back("Geopotential Height");
    list.push_back("Air Temperature");
    list.push_back("Omega (Vertical Velocity)");
    list.push_back("Relative Humidity");
    list.push_back("Specific Humidity");
    list.push_back("U-Wind");
    list.push_back("V-Wind");
    list.push_back("Precipitable Water");

    return list;
}

bool asDataPredictorArchiveNcepReanalysis2::ExtractFromFiles(asGeoAreaCompositeGrid *& dataArea, asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    // Get requested years
    int yearFirst = timeArray.GetFirstDayYear();
    int yearLast = timeArray.GetLastDayYear();

    #if wxUSE_GUI
        asDialogProgressBar progressBar(_("Loading data from files.\n"), yearLast-yearFirst);
    #endif

    // Loop through the files
    for (int i_year=yearFirst; i_year<=yearLast; i_year++)
    {
        // Build the file path
        wxString fileFullPath = m_DirectoryPath + wxString::Format(m_FileNamePattern, i_year);

        #if wxUSE_GUI
            // Update the progress bar
            wxString fileNameMessage = wxString::Format(_("Loading data from files.\nFile: %s"), wxString::Format(m_FileNamePattern, i_year).c_str());
            if(!progressBar.Update(i_year-yearFirst, fileNameMessage))
            {
                asLogWarning(_("The process has been canceled by the user."));
                return false;
            }
        #endif

        // Open the NetCDF file
        ThreadsManager().CritSectionNetCDF().Enter();
        asFileNetcdf ncFile(fileFullPath, asFileNetcdf::ReadOnly);
        if(!ncFile.Open())
        {
            ThreadsManager().CritSectionNetCDF().Leave();
            return false;
        }

        // Number of dimensions
        int nDims = ncFile.GetNDims();
        wxASSERT(nDims>=3);
        wxASSERT(nDims<=4);

        // Get some attributes
        float dataAddOffset = ncFile.GetAttFloat("add_offset", m_FileVariableName);
        if (asTools::IsNaN(dataAddOffset)) dataAddOffset = 0;
        float dataScaleFactor = ncFile.GetAttFloat("scale_factor", m_FileVariableName);
        if (asTools::IsNaN(dataScaleFactor)) dataScaleFactor = 1;
        bool scalingNeeded = true;
        if (dataAddOffset==0 && dataScaleFactor==1) scalingNeeded = false;

        // Get full axes from the netcdf file
        Array1DFloat axisDataLon(ncFile.GetVarLength(m_FileAxisLonName));
        ncFile.GetVar(m_FileAxisLonName, &axisDataLon[0]);
        Array1DFloat axisDataLat(ncFile.GetVarLength(m_FileAxisLatName));
        ncFile.GetVar(m_FileAxisLatName, &axisDataLat[0]);
        Array1DFloat axisDataLevel;
        if (nDims==4)
        {
            axisDataLevel.resize(ncFile.GetVarLength(m_FileAxisLevelName));
            ncFile.GetVar(m_FileAxisLevelName, &axisDataLevel[0]);
        }
        
        // Adjust axes if necessary
        dataArea = AdjustAxes(dataArea, axisDataLon, axisDataLat, compositeData);

        // Time array takes ages to load !! Avoid if possible. Get the first value of the time array.
        size_t axisDataTimeLength = ncFile.GetVarLength(m_FileAxisTimeName);
        double valFirstTime = ncFile.GetVarOneDouble(m_FileAxisTimeName, 0);
        valFirstTime = (valFirstTime/24.0); // hours to days
        valFirstTime += asTime::GetMJD(1800,1,1); // to MJD: add a negative time span

        // Get start and end of the current year
        double timeStart = asTime::GetMJD(i_year,1,1,0,0);
        double timeEnd = asTime::GetMJD(i_year,12,31,23,59);

        // Get the time length
        double timeArrayIndexStart = timeArray.GetIndexFirstAfter(timeStart);
        double timeArrayIndexEnd = timeArray.GetIndexFirstBefore(timeEnd);
        int indexLengthTime = timeArrayIndexEnd-timeArrayIndexStart+1;
        int indexLengthTimeArray = indexLengthTime;

        // Correct the time start and end
        size_t indexStartTime = 0;
        int cutStart = 0;
        if(i_year==yearFirst)
        {
            cutStart = timeArrayIndexStart;
        }
        int cutEnd = 0;
        while (valFirstTime<timeArray[timeArrayIndexStart])
        {
            valFirstTime += m_TimeStepHours/24.0;
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
        VectorBool vectLoad360;
        VVectorShort vectData;
        VVectorShort vectData360;

        for (int i_area = 0; i_area<compositeData.size(); i_area++)
        {
            // Check if necessary to load the data of lon=360 (so lon=0)
            bool load360 = false;

            int indexStartLon, indexStartLat, indexLengthLon, indexLengthLat;
            if (dataArea)
            {
                // Get the spatial extent
                float lonMin = dataArea->GetXaxisCompositeStart(i_area);
                float lonMax = dataArea->GetXaxisCompositeEnd(i_area);
                float latMinStart = dataArea->GetYaxisCompositeStart(i_area);
                float latMinEnd = dataArea->GetYaxisCompositeEnd(i_area);

                // The dimensions lengths
                indexLengthLon = dataArea->GetXaxisCompositePtsnb(i_area);
                indexLengthLat = dataArea->GetYaxisCompositePtsnb(i_area);

                if(lonMax==dataArea->GetAxisXmax())
                {
                    // Correction if the lon 360 degrees is required (doesn't exist)
                    load360 = true;
                    for (int i_check = 0; i_check<compositeData.size(); i_check++)
                    {
                        // If so, already loaded in another composite
                        if(dataArea->GetComposite(i_check).GetXmin() == 0)
                        {
                            load360 = false;
                        }
                    }
                    indexLengthLon--;
                }

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
                indexLengthLon = m_LonPtsnb;
                indexLengthLat = m_LatPtsnb;
            }
            int indexLevel = 0;
            if (nDims==4)
            {
                indexLevel = asTools::SortedArraySearch(&axisDataLevel[0], &axisDataLevel[axisDataLevel.size()-1], m_Level, 0.01f);
            }

            // Create the arrays to receive the data
            VectorShort data, data360;

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
            size_t indexStartData4[4] = {0,0,0,0};
            size_t indexCountData4[4] = {0,0,0,0};
            ptrdiff_t indexStrideData4[4] = {0,0,0,0};
            size_t indexStartData3[3] = {0,0,0};
            size_t indexCountData3[3] = {0,0,0};
            ptrdiff_t indexStrideData3[3] = {0,0,0};

            if (nDims==4)
            {
                // Set the indices for data
                indexStartData4[0] = indexStartTime;
                indexStartData4[1] = indexLevel;
                indexStartData4[2] = indexStartLat;
                indexStartData4[3] = indexStartLon;
                indexCountData4[0] = indexLengthTime;
                indexCountData4[1] = 1;
                indexCountData4[2] = indexLengthLat;
                indexCountData4[3] = indexLengthLon;
                indexStrideData4[0] = m_TimeIndexStep;
                indexStrideData4[1] = 1;
                indexStrideData4[2] = m_LatIndexStep;
                indexStrideData4[3] = m_LonIndexStep;

                // In the netCDF Common Data Language, variables are printed with the outermost dimension first and the innermost dimension last.
                ncFile.GetVarSample(m_FileVariableName, indexStartData4, indexCountData4, indexStrideData4, &data[indexBegining]);
            }
            else
            {
                // Set the indices for data
                indexStartData3[0] = indexStartTime;
                indexStartData3[1] = indexStartLat;
                indexStartData3[2] = indexStartLon;
                indexCountData3[0] = indexLengthTime;
                indexCountData3[1] = indexLengthLat;
                indexCountData3[2] = indexLengthLon;
                indexStrideData3[0] = m_TimeIndexStep;
                indexStrideData3[1] = m_LatIndexStep;
                indexStrideData3[2] = m_LonIndexStep;

                // In the netCDF Common Data Language, variables are printed with the outermost dimension first and the innermost dimension last.
                ncFile.GetVarSample(m_FileVariableName, indexStartData3, indexCountData3, indexStrideData3, &data[indexBegining]);
            }

            // Load data at lon = 360 degrees
            if(load360)
            {
                // Resize the arrays to store the new data
                int totlength360 = indexLengthTimeArray * indexLengthLat * 1;
                data360.resize(totlength360);

                // Set the indices
                indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLon.size()-1], 360, 0.01f);
                if(indexStartLon==asOUT_OF_RANGE)
                {
                    // If not found, try with negative angles
                    indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLon.size()-1], 0, 0.01f);
                }

                if (nDims==4)
                {
                    indexStartData4[3] = indexStartLon;
                    indexCountData4[3] = 1;
                    indexStrideData4[3] = 1;
                }
                else
                {
                    indexStartData3[2] = indexStartLon;
                    indexCountData3[2] = 1;
                    indexStrideData3[2] = 1;
                }

                // Fill empty begining with NaNs
                int indexBegining = 0;
                if(cutStart>0)
                {
                    int latlonlength = indexLengthLat*indexLengthLon;
                    for (int i_empty=0; i_empty<cutStart; i_empty++)
                    {
                        for (int i_emptylatlon=0; i_emptylatlon<latlonlength; i_emptylatlon++)
                        {
                            data360[indexBegining] = NaNFloat;
                            indexBegining++;
                        }
                    }
                }

                // Fill empty end with NaNs
                int indexEnd = (indexLengthTime-1) * (indexLengthLat-1) * (indexLengthLon-1);
                if(cutEnd>0)
                {
                    int latlonlength = indexLengthLat*indexLengthLon;
                    for (int i_empty=0; i_empty<cutEnd; i_empty++)
                    {
                        for (int i_emptylatlon=0; i_emptylatlon<latlonlength; i_emptylatlon++)
                        {
                            indexEnd++;
                            data360[indexEnd] = NaNFloat;
                        }
                    }
                }

                // Load data at 0 degrees (corresponds to 360 degrees)
                if (nDims==4)
                {
                    ncFile.GetVarSample(m_FileVariableName, indexStartData4, indexCountData4, indexStrideData4, &data360[indexBegining]);
                }
                else
                {
                    ncFile.GetVarSample(m_FileVariableName, indexStartData3, indexCountData3, indexStrideData3, &data360[indexBegining]);
                }
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
        if (compositeData[0].capacity()==0)
        {
            int totSize = 0;
            for (int i_area = 0; i_area<compositeData.size(); i_area++)
            {
                int indexLengthLat = vectIndexLengthLat[i_area];
                int indexLengthLon = vectIndexLengthLon[i_area];
                totSize += m_Time.size() * indexLengthLat * (indexLengthLon+1); // +1 in case of a border
            }
            compositeData.reserve(totSize);
        }

        // Transfer data
        for (int i_area = 0; i_area<compositeData.size(); i_area++)
        {
            // Extract data
            int indexLengthLat = vectIndexLengthLat[i_area];
            int indexLengthLon = vectIndexLengthLon[i_area];
            bool load360 = vectLoad360[i_area];
            VectorShort data = vectData[i_area];
            VectorShort data360 = vectData360[i_area];

            // Loop to extract the data from the array
            int ind = 0;
            for (int i_time=0; i_time<indexLengthTimeArray; i_time++)
            {
                Array2DFloat latlonData;
                if(load360)
                {
                    latlonData = Array2DFloat(indexLengthLat,indexLengthLon+1);
                }
                else
                {
                    latlonData = Array2DFloat(indexLengthLat,indexLengthLon);
                }

                for (int i_lat=0; i_lat<indexLengthLat; i_lat++)
                {
                    for (int i_lon=0; i_lon<indexLengthLon; i_lon++)
                    {
                        ind = i_lon + i_lat * indexLengthLon + i_time * indexLengthLon * indexLengthLat;

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
                        for (size_t i_nan=0; i_nan<m_NanValues.size(); i_nan++)
                        {
                            if ((float)data[ind]==m_NanValues[i_nan] || latlonData(i_lat,i_lon)==m_NanValues[i_nan])
                            {
                                notNan = false;
                            }
                        }
                        if (!notNan)
                        {
                            latlonData(i_lat,i_lon) = NaNFloat;
                        }
                    }

                    if(load360)
                    {
                        ind = i_lat + i_time * indexLengthLat;

                        if (scalingNeeded)
                        {
                            latlonData(i_lat,indexLengthLon) = (float)data360[ind] * dataScaleFactor + dataAddOffset;
                        }
                        else
                        {
                            latlonData(i_lat,indexLengthLon) = (float)data360[ind];
                        }

                        // Check if not NaN
                        bool notNan = true;
                        for (size_t i_nan=0; i_nan<m_NanValues.size(); i_nan++)
                        {
                            if ((float)data360[ind]==m_NanValues[i_nan] || latlonData(i_lat,indexLengthLon)==m_NanValues[i_nan])
                            {
                                notNan = false;
                            }
                        }
                        if (!notNan)
                        {
                            latlonData(i_lat,indexLengthLon) = NaNFloat;
                        }
                    }
                }
                compositeData[i_area].push_back(latlonData);
            }
            data.clear();
            data360.clear();
        }
    }

    return true;
}

