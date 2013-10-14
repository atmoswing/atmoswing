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
 */
 
#include "asDataPredictorRealtime.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asInternet.h>
#include <asFileGrib2.h>


asDataPredictorRealtime::asDataPredictorRealtime(asCatalogPredictorsRealtime &catalog)
:
asDataPredictor()
{
    m_Catalog = catalog;
    m_IsPreprocessed = false;
}

asDataPredictorRealtime::~asDataPredictorRealtime()
{

}

int asDataPredictorRealtime::Download(asCatalogPredictorsRealtime &catalog)
{
    // Directory
    wxConfigBase *pConfig = wxFileConfig::Get();
    wxString realtimePredictorSavingDir = pConfig->Read("/StandardPaths/RealtimePredictorSavingDir", wxEmptyString);

    // Internet (cURL)
    asInternet internet;

    return internet.Download(catalog.GetUrls(), catalog.GetFileNames(), realtimePredictorSavingDir);
}

bool asDataPredictorRealtime::LoadFullArea(double date, float level, const VectorString &AlternatePredictorDataPath)
{
    asTimeArray timeArray(date, asTimeArray::SingleDay);
    timeArray.Init();
    m_Level = level;

    return Load(NULL, timeArray, AlternatePredictorDataPath);
}

bool asDataPredictorRealtime::Load(asGeoAreaCompositeGrid &desiredArea, asTimeArray &timeArray, const VectorString &AlternatePredictorDataPath)
{
    return Load(&desiredArea, timeArray, AlternatePredictorDataPath);
}

bool asDataPredictorRealtime::Load(asGeoAreaCompositeGrid *desiredArea, asTimeArray &timeArray, const VectorString &AlternatePredictorDataPath)
{
    // Configuration
    wxConfigBase *pConfig = wxFileConfig::Get();
    wxString realtimePredictorDir = pConfig->Read("/StandardPaths/RealtimePredictorSavingDir", wxEmptyString);

    // File path
    VectorString filePaths;
    if(AlternatePredictorDataPath.size()==0)
    {
        filePaths = m_Catalog.GetFileNames();
    }
    else
    {
        filePaths = AlternatePredictorDataPath;
    }
    wxASSERT(filePaths.size()>=(unsigned)timeArray.GetSize());

    // The data name
    wxString dataName = m_Catalog.GetDataInfo().FileVarName;
    if (dataName.IsEmpty())
    {
        asLogError(_("The real-time predictor variable name is not defined."));
        return false;
    }

    asGeoAreaCompositeGrid* dataArea = NULL;
    if (desiredArea)
    {
        // Create a new area matching the dataset
        double dataUstepCat = m_Catalog.GetDataUaxisStep();
        double dataVstepCat = m_Catalog.GetDataVaxisStep();
        double dataUmin = floor(desiredArea->GetAbsoluteUmin()/dataUstepCat)*dataUstepCat;
        double dataVmin = floor(desiredArea->GetAbsoluteVmin()/dataVstepCat)*dataVstepCat;
        double dataUmax = ceil(desiredArea->GetAbsoluteUmax()/dataUstepCat)*dataUstepCat;
        double dataVmax = ceil(desiredArea->GetAbsoluteVmax()/dataVstepCat)*dataVstepCat;
        double dataUstep = dataUstepCat;
        double dataVstep = dataVstepCat;
        int dataUptsnb = (dataUmax-dataUmin)/dataUstep+1;
        int dataVptsnb = (dataVmax-dataVmin)/dataVstep+1;
        wxString gridType = desiredArea->GetGridTypeString();
        dataArea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, dataUmin, dataUptsnb, dataUstep, dataVmin, dataVptsnb, dataVstep, desiredArea->GetLevel(), asNONE, asFLAT_ALLOWED);

        // Get axes length for preallocation
        GetSizes(*dataArea, timeArray);
        InitContainers();
    }
    else
    {
        m_SizeTime = timeArray.GetSize();
        m_Time.resize(m_SizeTime);
    }

    // Add dates to m_Time
    m_Time = timeArray.GetTimeArray();

    // The desired level
    if (desiredArea)
    {
        m_Level = desiredArea->GetComposite(0).GetLevel();
    }

    // Containers for the axes
    Array1DFloat axisDataLon, axisDataLat;

    // Load files
    for (int i_file=0; i_file<timeArray.GetSize(); i_file++)
    {
        // Extract file path
        wxString filePath;
        if(AlternatePredictorDataPath.size()==0)
        {
            filePath = realtimePredictorDir;
            filePath.Append(DS);
            filePath.Append(filePaths[i_file]);
        }
        else
        {
            filePath = filePaths[i_file];
        }

        // Open the Grib2 file
        asFileGrib2 g2File(filePath, asFileGrib2::ReadOnly);
        if(!g2File.Open())
        {
            wxDELETE(dataArea);
            return false;
        }

        // Get some attributes
        float dataAddOffset = g2File.GetOffset();
        float dataScaleFactor = g2File.GetScale();

        // Get full axes from the netcdf file
            // Longitudes
        int axisDataLonLength = g2File.GetUPtsnb();
        g2File.GetUaxis(axisDataLon);
            // Latitudes
        int axisDataLatLength = g2File.GetVPtsnb();
        g2File.GetVaxis(axisDataLat);

        if (desiredArea==NULL && i_file==0)
        {
            // Get axes length for preallocation
            m_LonPtsnb = axisDataLonLength;
            m_LatPtsnb = axisDataLatLength;
            m_AxisLon.resize(axisDataLon.size());
            m_AxisLon = axisDataLon;
            m_AxisLat.resize(axisDataLat.size());
            m_AxisLat = axisDataLat;
            m_Data.reserve(m_SizeTime*m_LonPtsnb*m_LatPtsnb);
        }
        else if(desiredArea!=NULL && i_file==0)
        {
            Array1DDouble axisLon = dataArea->GetUaxis();
            m_AxisLon.resize(axisLon.size());
            for (int i=0; i<axisLon.size(); i++)
            {
                m_AxisLon[i] = (float)axisLon[i];
            }
            m_LonPtsnb = dataArea->GetUaxisPtsnb();
            wxASSERT_MSG(m_AxisLon.size()==m_LonPtsnb, wxString::Format("m_AxisLon.size()=%d, m_LonPtsnb=%d",(int)m_AxisLon.size(),m_LonPtsnb));

            Array1DDouble axisLat = dataArea->GetVaxis();
            m_AxisLat.resize(axisLat.size());
            for (int i=0; i<axisLat.size(); i++)
            {
                m_AxisLat[i] = (float)axisLat[i];
            }
            m_LatPtsnb = dataArea->GetVaxisPtsnb();
            wxASSERT_MSG(m_AxisLat.size()==m_LatPtsnb, wxString::Format("m_AxisLat.size()=%d, m_LatPtsnb=%d",(int)m_AxisLat.size(),m_LatPtsnb));

            m_Data.reserve(m_SizeTime*m_LonPtsnb*m_LatPtsnb);
        }

        // The container for extracted data from every composite
        VVArray2DFloat compositeData;
        int iterationNb = 1;
        if (desiredArea)
        {
            iterationNb = dataArea->GetNbComposites();
        }

        for (int i_area = 0; i_area<iterationNb; i_area++)
        {
            // Check if necessary to load the data of lon=360 (so lon=0)
            bool load360 = false;

            int indexStartLon, indexStartLat, indexLengthLon, indexLengthLat;
            int indexLengthTimeArray = 1; // For 1 file
            if (desiredArea)
            {

                // Get the spatial extent
                float lonMin = dataArea->GetUaxisCompositeStart(i_area);
                float lonMax = dataArea->GetUaxisCompositeEnd(i_area);
                float latMinStart = dataArea->GetVaxisCompositeStart(i_area);
                float latMinEnd = dataArea->GetVaxisCompositeEnd(i_area);

                // The dimensions lengths
                indexLengthLon = dataArea->GetUaxisCompositePtsnb(i_area);
                indexLengthLat = dataArea->GetVaxisCompositePtsnb(i_area);

                if(lonMax==360)
                {
                    // Correction if the lon 360° is required (doesn't exist)
                    load360 = true;
                    for (int i_check = 0; i_check<dataArea->GetNbComposites(); i_check++)
                    {
                        // If so, already loaded in another composite
                        if(dataArea->GetComposite(i_check).GetUmin() == 0)
                        {
                            load360 = false;
                        }
                    }
                    lonMax -= dataArea->GetUstep();
                    indexLengthLon--;
                }

                // Get the spatial indices of the desired data
                indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLonLength-1], lonMin, 0.0001f);
                if(indexStartLon==asOUT_OF_RANGE)
                {
                    // If not found, try with negative angles
                    indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLonLength-1], lonMin-360, 0.0001f);
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
            g2File.GetVarArray(dataName, indexStartData, indexCountData, m_Level, &data[0]);

            // Load data at lon = 360°
            if(load360)
            {
                // Resize the arrays to store the new data
                int totlength360 = indexLengthLat;
                data360.resize(totlength360);

                // Set the indices
                indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLonLength-1], 360, 0.0001f);
                if(indexStartLon==asOUT_OF_RANGE)
                {
                    // If not found, try with negative angles
                    indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLonLength-1], 0, 0.0001f);
                }
                indexStartData[0] = indexStartLon;
                indexCountData[0] = 1;

                // Get data from file
                g2File.GetVarArray(dataName, indexStartData, indexCountData, m_Level, &data360[0]);
            }

            // Containers for results
            Array2DFloat latlonData(indexLengthLat,indexLengthLon);

            if(load360)
            {
                latlonData.resize(indexLengthLat,indexLengthLon+1);
            }

            VArray2DFloat latlonTimeData;
            latlonTimeData.reserve(totLength);
            int ind = 0;

            // Loop to extract the data from the array
            for (int i_lat=0; i_lat<indexLengthLat; i_lat++)
            {
                for (int i_lon=0; i_lon<indexLengthLon; i_lon++)
                {
                    ind = i_lon;
                    ind += i_lat * indexLengthLon;
                    // Add the Offset and multiply by the Scale Factor
                    latlonData(i_lat,i_lon) = data[ind] * dataScaleFactor + dataAddOffset;
                }

                if(load360)
                {
                    ind = i_lat;
                    // Add the Offset and multiply by the Scale Factor
                    latlonData(i_lat,indexLengthLon) = data360[ind] * dataScaleFactor + dataAddOffset;
                }
            }

            latlonTimeData.push_back(latlonData);

            if(load360)
            {
                latlonData.setZero(indexLengthLat,indexLengthLon+1);
            }
            else
            {
                latlonData.setZero(indexLengthLat,indexLengthLon);
            }

            compositeData.push_back(latlonTimeData);
            latlonTimeData.clear();
            data.clear();
            data360.clear();
        }

        // Close the nc file
        g2File.Close();

        // Merge the composites into m_Data
        if (!MergeComposites(compositeData, dataArea))
        {
            wxDELETE(dataArea);
            return false;
        }

    }

    // Interpolate the loaded data on the desired grid
    if (desiredArea)
    {
        if (!InterpolateOnGrid(dataArea, desiredArea))
        {
            wxDELETE(dataArea);
            return false;
        }
        wxASSERT_MSG(m_Data[0].cols()==desiredArea->GetUaxisPtsnb(), wxString::Format("m_Data[0].cols()=%d, desiredArea->GetUaxisPtsnb()=%d", (int)m_Data[0].cols(), (int)desiredArea->GetUaxisPtsnb()));
        wxASSERT_MSG(m_Data[0].rows()==desiredArea->GetVaxisPtsnb(), wxString::Format("m_Data[0].rows()=%d, desiredArea->GetVaxisPtsnb()=%d", (int)m_Data[0].rows(), (int)desiredArea->GetVaxisPtsnb()));
    }

    wxDELETE(dataArea);

    return true;
}
