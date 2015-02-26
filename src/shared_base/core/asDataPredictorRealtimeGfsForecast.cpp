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

#include "asDataPredictorRealtimeGfsForecast.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileGrib2.h>


asDataPredictorRealtimeGfsForecast::asDataPredictorRealtimeGfsForecast(const wxString &dataId)
:
asDataPredictorRealtime(dataId)
{
    // Set the basic properties.
    m_Initialized = false;
    m_DataId = dataId;
    m_DatasetId = "NWS_GFS_Forecast";
    m_OriginalProvider = "NWS";
    m_FinalProvider = "NWS";
    m_FinalProviderWebsite = "http://www.emc.ncep.noaa.gov/GFS/";
    m_FinalProviderFTP = "http://nomads.ncep.noaa.gov/";
    m_DatasetName = "Global Forecast System";
    m_TimeZoneHours = 0;
    m_ForecastLeadTimeStart = 0;
    m_ForecastLeadTimeEnd = 240; // After 240h, available in another temporal resolution
    m_ForecastLeadTimeStep = 6;
    m_RunHourStart = 0;
    m_RunUpdate = 6;
    m_FirstTimeStepHours = 0;
    m_NanValues.push_back(NaNDouble);
    m_NanValues.push_back(NaNFloat);
    m_XaxisShift = 0;
    m_YaxisShift = 0;
    m_XaxisStep = 0.5;
    m_YaxisStep = 0.5;
    m_RestrictTimeHours = 0;
    m_RestrictTimeStepHours = 24;
    m_FileExtension = "grib2";

    // Identify data ID and set the corresponding properties.
    if (m_DataId.IsSameAs("hgt", false))
    {
        m_DataParameter = GeopotentialHeight;
        m_CommandDownload = "http://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p50.pl?file=gfs.t[CURRENTDATE-hh]z.pgrb2full.0p50.f[LEADTIME-hhh]&lev_400_mb=on&lev_500_mb=on&lev_600_mb=on&lev_700_mb=on&lev_850_mb=on&lev_925_mb=on&lev_1000_mb=on&var_HGT=on&subregion=&leftlon=-20&rightlon=30&toplat=60&bottomlat=30&dir=%2Fgfs.[CURRENTDATE-YYYYMMDDhh]";
        m_FileVariableName = "HGT";
        m_Unit = m;
    }
    else if (m_DataId.IsSameAs("air", false))
    {
        m_DataParameter = AirTemperature;
        m_CommandDownload = "http://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p50.pl?file=gfs.t[CURRENTDATE-hh]z.pgrb2full.0p50.f[LEADTIME-hhh]&lev_400_mb=on&lev_500_mb=on&lev_600_mb=on&lev_700_mb=on&lev_850_mb=on&lev_925_mb=on&lev_1000_mb=on&var_TMP=on&subregion=&leftlon=-20&rightlon=30&toplat=60&bottomlat=30&dir=%2Fgfs.[CURRENTDATE-YYYYMMDDhh]";
        m_FileVariableName = "TEMP";
        m_Unit = degK;
    }
    else if (m_DataId.IsSameAs("omega", false))
    {
        m_DataParameter = Omega;
        m_CommandDownload = "http://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p50.pl?file=gfs.t[CURRENTDATE-hh]z.pgrb2full.0p50.f[LEADTIME-hhh]&lev_400_mb=on&lev_500_mb=on&lev_600_mb=on&lev_700_mb=on&lev_850_mb=on&lev_925_mb=on&lev_1000_mb=on&var_VVEL=on&subregion=&leftlon=-20&rightlon=30&toplat=60&bottomlat=30&dir=%2Fgfs.[CURRENTDATE-YYYYMMDDhh]";
        m_FileVariableName = "VVEL";
        m_Unit = PascalsPerSec;
    }
    else if (m_DataId.IsSameAs("rhum", false))
    {
        m_DataParameter = RelativeHumidity;
        m_CommandDownload = "http://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p50.pl?file=gfs.t[CURRENTDATE-hh]z.pgrb2full.0p50.f[LEADTIME-hhh]&lev_400_mb=on&lev_500_mb=on&lev_600_mb=on&lev_700_mb=on&lev_850_mb=on&lev_925_mb=on&lev_1000_mb=on&var_RH=on&subregion=&leftlon=-20&rightlon=30&toplat=60&bottomlat=30&dir=%2Fgfs.[CURRENTDATE-YYYYMMDDhh]";
        m_FileVariableName = "RH";
        m_Unit = percent;
    }
    else if (m_DataId.IsSameAs("uwnd", false))
    {
        m_DataParameter = Uwind;
        m_CommandDownload = "http://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p50.pl?file=gfs.t[CURRENTDATE-hh]z.pgrb2full.0p50.f[LEADTIME-hhh]&lev_400_mb=on&lev_500_mb=on&lev_600_mb=on&lev_700_mb=on&lev_850_mb=on&lev_925_mb=on&lev_1000_mb=on&var_UGRD=on&subregion=&leftlon=-20&rightlon=30&toplat=60&bottomlat=30&dir=%2Fgfs.[CURRENTDATE-YYYYMMDDhh]";
        m_FileVariableName = "UGRD";
        m_Unit = mPerSec;
    }
    else if (m_DataId.IsSameAs("vwnd", false))
    {
        m_DataParameter = Vwind;
        m_CommandDownload = "http://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p50.pl?file=gfs.t[CURRENTDATE-hh]z.pgrb2full.0p50.f[LEADTIME-hhh]&lev_400_mb=on&lev_500_mb=on&lev_600_mb=on&lev_700_mb=on&lev_850_mb=on&lev_925_mb=on&lev_1000_mb=on&var_VGRD=on&subregion=&leftlon=-20&rightlon=30&toplat=60&bottomlat=30&dir=%2Fgfs.[CURRENTDATE-YYYYMMDDhh]";
        m_FileVariableName = "VGRD";
        m_Unit = mPerSec;
    }
    else if (m_DataId.IsSameAs("surf_prwtr", false))
    {
        m_DataParameter = PrecipitableWater;
        m_CommandDownload = "http://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p50.pl?file=gfs.t[CURRENTDATE-hh]z.pgrb2full.0p50.f[LEADTIME-hhh]&lev_entire_atmosphere_%5C%28considered_as_a_single_layer%5C%29=on&var_PWAT=on&subregion=&leftlon=-20&rightlon=30&toplat=60&bottomlat=30&dir=%2Fgfs.[CURRENTDATE-YYYYMMDDhh]";
        m_FileVariableName = "PWAT";
        m_Unit = mm;
    }
    else
    {
        m_DataParameter = NoDataParameter;
        m_CommandDownload = wxEmptyString;
        m_FileVariableName = wxEmptyString;
        m_Unit = NoDataUnit;
    }
}

asDataPredictorRealtimeGfsForecast::~asDataPredictorRealtimeGfsForecast()
{

}

bool asDataPredictorRealtimeGfsForecast::Init()
{
    // Check data ID
    if (m_CommandDownload.IsEmpty() || m_FileVariableName.IsEmpty()) {
        asLogError(wxString::Format(_("The provided data ID (%s) does not match any possible option in the dataset %s."), m_DataId.c_str(), m_DatasetName.c_str()));
        return false;
    }

    // Set to initialized
    m_Initialized = true;

    return true;
}

VectorString asDataPredictorRealtimeGfsForecast::GetDataIdList()
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

VectorString asDataPredictorRealtimeGfsForecast::GetDataIdDescriptionList()
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

bool asDataPredictorRealtimeGfsForecast::ExtractFromFiles(asGeoAreaCompositeGrid *& dataArea, asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    // File path
    VectorString filePaths = GetFileNames();
    wxASSERT(filePaths.size()>=(unsigned)timeArray.GetSize());

    // Load files
    for (int i_file=0; i_file<timeArray.GetSize(); i_file++)
    {
        // Extract file path
        wxString filePath = wxEmptyString;

        // Check if the volume is present
        wxFileName fileName(filePaths[i_file]);
        if (!fileName.HasVolume() && !m_PredictorsRealtimeDirectory.IsEmpty())
        {
            filePath = m_PredictorsRealtimeDirectory;
            filePath.Append(DS);
        }
        filePath.Append(filePaths[i_file]);

        // Open the Grib2 file
        asFileGrib2 g2File(filePath, asFileGrib2::ReadOnly);
        if(!g2File.Open())
        {
            return false;
        }

        // Get some attributes
        float dataAddOffset = g2File.GetOffset();
        float dataScaleFactor = g2File.GetScale();
        bool scalingNeeded = true;
        if (dataAddOffset==0 && dataScaleFactor==1) scalingNeeded = false;

        // Get full axes from the grib file
        Array1DFloat axisDataLon, axisDataLat;
        int axisDataLonLength = g2File.GetXPtsnb();
        g2File.GetXaxis(axisDataLon);
        int axisDataLatLength = g2File.GetYPtsnb();
        g2File.GetYaxis(axisDataLat);

        // Adjust axes if necessary
        dataArea = AdjustAxes(dataArea, axisDataLon, axisDataLat, compositeData);

        for (int i_area = 0; i_area<compositeData.size(); i_area++)
        {
            // Check if necessary to load the data of lon=360 (so lon=0)
            bool load360 = false;

            int indexStartLon, indexStartLat, indexLengthLon, indexLengthLat;
            int indexLengthTimeArray = 1; // For 1 file
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

                if(lonMax==360)
                {
                    // Correction if the lon 360° is required (doesn't exist)
                    load360 = true;
                    for (int i_check = 0; i_check<dataArea->GetNbComposites(); i_check++)
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
                indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLonLength-1], lonMin, 0.0001f, asHIDE_WARNINGS);
                if(indexStartLon==asOUT_OF_RANGE)
                {
                    // If not found, try with negative angles
                    indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLonLength-1], lonMin-360, 0.0001f, asHIDE_WARNINGS);
                }
                if(indexStartLon==asOUT_OF_RANGE)
                {
                    // If not found, try with angles above 360°
                    indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLonLength-1], lonMin+360, 0.0001f);
                }
                wxASSERT(indexStartLon>=0);
                int indexStartLat1 = asTools::SortedArraySearch(&axisDataLat[0], &axisDataLat[axisDataLatLength-1], latMinStart, 0.0001f);
                int indexStartLat2 = asTools::SortedArraySearch(&axisDataLat[0], &axisDataLat[axisDataLatLength-1], latMinEnd, 0.0001f);
                wxASSERT(indexStartLat1>=0);
                wxASSERT(indexStartLat2>=0);
                indexStartLat = wxMin(indexStartLat1, indexStartLat2);
            }
            else
            {
                indexStartLon = 0;
                indexStartLat = 0;
                indexLengthLon = m_LonPtsnb;
                indexLengthLat = m_LatPtsnb;
            }

            // Create the arrays to receive the data
            VectorFloat data, data360;

            // Resize the arrays to store the new data
            int totLength = indexLengthTimeArray * indexLengthLat * indexLengthLon;
            wxASSERT(totLength>0);
            data.resize(totLength);

            // Get the indices for data
            int indexStartData[] = {0,0};
            int indexCountData[] = {0,0};

            // Set the indices for data
            indexStartData[0] = indexStartLon;
            indexStartData[1] = indexStartLat;
            indexCountData[0] = indexLengthLon;
            indexCountData[1] = indexLengthLat;

            // Get data from file
            g2File.GetVarArray(m_FileVariableName, indexStartData, indexCountData, m_Level, &data[0]);

            // Load data at lon = 360°
            if(load360)
            {
                // Resize the arrays to store the new data
                int totlength360 = indexLengthLat;
                data360.resize(totlength360);

                // Set the indices
                indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLonLength-1], 360, 0.0001f, asHIDE_WARNINGS);
                if(indexStartLon==asOUT_OF_RANGE)
                {
                    // If not found, try with negative angles
                    indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLonLength-1], 0, 0.0001f);
                }
                indexStartData[0] = indexStartLon;
                indexCountData[0] = 1;

                // Get data from file
                g2File.GetVarArray(m_FileVariableName, indexStartData, indexCountData, m_Level, &data360[0]);
            }

            // Containers for results
            Array2DFloat latlonData;
            if(load360)
            {
                latlonData = Array2DFloat(indexLengthLat,indexLengthLon+1);
            }
            else
            {
                latlonData = Array2DFloat(indexLengthLat,indexLengthLon);
            }

            int ind = 0;

            // Loop to extract the data from the array
            for (int i_lat=0; i_lat<indexLengthLat; i_lat++)
            {
                for (int i_lon=0; i_lon<indexLengthLon; i_lon++)
                {
                    ind = i_lon + i_lat * indexLengthLon;
                    if (scalingNeeded)
                    {
                        latlonData(i_lat,i_lon) = data[ind] * dataScaleFactor + dataAddOffset;
                    }
                    else
                    {
                        latlonData(i_lat,i_lon) = data[ind];
                    }
                }

                if(load360)
                {
                    ind = i_lat;
                    if (scalingNeeded)
                    {
                        latlonData(i_lat,indexLengthLon) = data360[ind] * dataScaleFactor + dataAddOffset;
                    }
                    else
                    {
                        latlonData(i_lat,indexLengthLon) = data360[ind];
                    }
                }
            }

            compositeData[i_area].push_back(latlonData);
            data.clear();
            data360.clear();
        }

        // Close the file
        g2File.Close();
    }

    return true;
}
