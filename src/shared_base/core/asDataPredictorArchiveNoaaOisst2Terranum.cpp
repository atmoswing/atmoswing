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
    m_Initialized = false;
    m_DataId = dataId;
    m_DatasetId = "NOAA_OISST_v2_terranum";
    m_OriginalProvider = "NOAA";
    m_FinalProvider = "Terranum";
    m_FinalProviderWebsite = "http://www.terranum.ch";
    m_FinalProviderFTP = "";
    m_DatasetName = "Optimum Interpolation Sea Surface Temperature, version 2, subset from terranum";
    m_OriginalProviderStart = asTime::GetMJD(1982, 1, 1);
    m_OriginalProviderEnd = NaNDouble;
    m_TimeZoneHours = 0;
    m_TimeStepHours = 24;
    m_FirstTimeStepHours = 12;
    m_NanValues.push_back(32767);
    m_NanValues.push_back(936*std::pow(10.f,34.f));
    m_CoordinateSystem = WGS84;
    m_UaxisShift = 0.125;
    m_VaxisShift = 0.125;
    m_UaxisStep = 1;
    m_VaxisStep = 1;
    m_SubFolder = wxEmptyString;

    // Identify data ID and set the corresponding properties.
    if (m_DataId.IsSameAs("sst", false))
    {
        m_DataParameter = SeaSurfaceTemperature;
        m_FileNamePattern = "sst_1deg.nc";
        m_FileVariableName = "sst";
        m_Unit = degC;
    }
    else if (m_DataId.IsSameAs("sst_anom", false))
    {
        m_DataParameter = SeaSurfaceTemperatureAnomaly;
        m_FileNamePattern = "sst_anom_1deg.nc";
        m_FileVariableName = "anom";
        m_Unit = degC;
    }
    else
    {
        m_DataParameter = NoDataParameter;
        m_FileVariableName = wxEmptyString;
        m_Unit = NoDataUnit;
    }
}

asDataPredictorArchiveNoaaOisst2Terranum::~asDataPredictorArchiveNoaaOisst2Terranum()
{

}

bool asDataPredictorArchiveNoaaOisst2Terranum::Init()
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

bool asDataPredictorArchiveNoaaOisst2Terranum::Load(asGeoAreaCompositeGrid *desiredArea, asTimeArray &timeArray)
{
    if (!m_Initialized)
    {
        if (!Init()) {
            asLogError(wxString::Format(_("Error at initialization of the predictor dataset %s."), m_DatasetName.c_str()));
            return false;
        }
    }

    try
    {
        // Check the time array
        if(!CheckTimeArray(timeArray))
        {
            asLogError(_("The time array is not valid to load data."));
            return false;
        }

        // Create a new area matching the dataset
        asGeoAreaCompositeGrid* dataArea = NULL;
        size_t indexStepLon = 1, indexStepLat = 1, indexStepTime = 1;
        if (desiredArea)
        {
            double dataUmin, dataVmin, dataUmax, dataVmax, dataUstep, dataVstep;
            int dataUptsnb, dataVptsnb;
            wxString gridType = desiredArea->GetGridTypeString();
            if (gridType.IsSameAs("Regular", false))
            {
                dataUmin = floor((desiredArea->GetAbsoluteUmin()-m_UaxisShift)/m_UaxisStep)*m_UaxisStep+m_UaxisShift;
                dataVmin = floor((desiredArea->GetAbsoluteVmin()-m_VaxisShift)/m_VaxisStep)*m_VaxisStep+m_VaxisShift;
                dataUmax = ceil((desiredArea->GetAbsoluteUmax()-m_UaxisShift)/m_UaxisStep)*m_UaxisStep+m_UaxisShift;
                dataVmax = ceil((desiredArea->GetAbsoluteVmax()-m_VaxisShift)/m_VaxisStep)*m_VaxisStep+m_VaxisShift;
                dataUstep = floor(desiredArea->GetUstep()/m_UaxisStep)*m_UaxisStep; // NetCDF allows to use strides
                dataVstep = floor(desiredArea->GetVstep()/m_VaxisStep)*m_VaxisStep; // NetCDF allows to use strides
                dataUptsnb = (dataUmax-dataUmin)/dataUstep+1;
                dataVptsnb = (dataVmax-dataVmin)/dataVstep+1;
            }
            else
            {
                dataUmin = desiredArea->GetAbsoluteUmin();
                dataVmin = desiredArea->GetAbsoluteVmin();
                dataUstep = desiredArea->GetUstep();
                dataVstep = desiredArea->GetVstep();
                dataUptsnb = desiredArea->GetUaxisPtsnb();
                dataVptsnb = desiredArea->GetVaxisPtsnb();
            }

            dataArea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, dataUmin, dataUptsnb, dataUstep, dataVmin, dataVptsnb, dataVstep, desiredArea->GetLevel(), asNONE, asFLAT_ALLOWED);

            // Get indexes steps
            if (gridType.IsSameAs("Regular", false))
            {
                indexStepLon = dataArea->GetUstep()/m_UaxisStep;
                indexStepLat = dataArea->GetVstep()/m_VaxisStep;
            }
            else
            {
                indexStepLon = 1;
                indexStepLat = 1;
            }

            // Get axes length for preallocation
            GetSizes(*dataArea, timeArray);
            InitContainers();
        }
        else
        {
            m_SizeTime = timeArray.GetSize();
            m_Time.resize(m_SizeTime);
        }

        indexStepTime = timeArray.GetTimeStepHours()/m_TimeStepHours;
        indexStepTime = wxMax(indexStepTime,1);

        // Add dates to m_Time
        m_Time = timeArray.GetTimeArray();

        // Check of the array length
        int counterTime = 0;

        // The desired level
        if (desiredArea)
        {
            m_Level = desiredArea->GetComposite(0).GetLevel();
        }

        // Containers for the axes
        Array1DFloat axisDataLon, axisDataLat;

        // Build the file path
        wxString fileFullPath = m_DirectoryPath + m_FileNamePattern;

        ThreadsManager().CritSectionNetCDF().Enter();

        // Open the NetCDF file
        asFileNetcdf ncFile(fileFullPath, asFileNetcdf::ReadOnly);
        if(!ncFile.Open())
        {
            wxDELETE(dataArea);
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
            // Longitudes
        size_t axisDataLonLength = ncFile.GetVarLength("lon");
        axisDataLon.resize(axisDataLonLength);
        ncFile.GetVar("lon", &axisDataLon[0]);
            // Latitudes
        size_t axisDataLatLength = ncFile.GetVarLength("lat");
        axisDataLat.resize(axisDataLatLength);
        ncFile.GetVar("lat", &axisDataLat[0]);
            // Levels
        size_t axisDataLevelLength = 0;
        Array1DFloat axisDataLevel;
        if (nDims==4)
        {
            axisDataLevelLength = ncFile.GetVarLength("level");
            axisDataLevel.resize(axisDataLevelLength);
            ncFile.GetVar("level", &axisDataLevel[0]);
        }
            // Time
        size_t axisDataTimeLength = ncFile.GetVarLength("time");
        // Time array takes ages to load !! Avoid if possible. Get the first value of the time array.
        double valFirstTime = ncFile.GetVarOneDouble("time", 0);
        valFirstTime = (valFirstTime/24.0); // hours to days
        valFirstTime += asTime::GetMJD(1,1,1); // to MJD: add a negative time span
        double valLastTime = ncFile.GetVarOneDouble("time", axisDataTimeLength-1);
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


        if (desiredArea==NULL)
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
        else if(desiredArea!=NULL)
        {
            // Check that requested data do not overtake the file
            for (int i_comp=0; i_comp<dataArea->GetNbComposites(); i_comp++)
            {
                Array1DDouble axisLonComp = dataArea->GetUaxisComposite(i_comp);

                if (axisDataLon[axisDataLonLength-1]>axisDataLon[0])
                {
                    wxASSERT(axisLonComp[axisLonComp.size()-1]>=axisLonComp[0]);

                    // Condition for change: The composite must not be fully outside (considered as handled) and the limit is not the coordinate grid border.
                    if (axisLonComp[axisLonComp.size()-1]>axisDataLon[axisDataLonLength-1] && axisLonComp[0]<axisDataLon[axisDataLonLength-1] && axisLonComp[axisLonComp.size()-1]!=dataArea->GetAxisUmax())
                    {
                        asLogMessage(_("Correcting the longitude extent according to the file limits."));
                        double Uwidth = axisDataLon[axisDataLonLength-1]-dataArea->GetAbsoluteUmin();
                        wxASSERT(Uwidth>=0);
                        int Uptsnb = 1+Uwidth/dataArea->GetUstep();
                        asLogMessage(wxString::Format(_("Uptsnb = %d."), Uptsnb));
                        asGeoAreaCompositeGrid* newdataArea = asGeoAreaCompositeGrid::GetInstance(WGS84, dataArea->GetGridTypeString(),
                                                                            dataArea->GetAbsoluteUmin(), Uptsnb,
                                                                            dataArea->GetUstep(), dataArea->GetAbsoluteVmin(),
                                                                            dataArea->GetVaxisPtsnb(), dataArea->GetVstep(),
                                                                            dataArea->GetLevel(), asNONE, asFLAT_ALLOWED);

                        wxDELETE(dataArea);
                        dataArea = newdataArea;
                    }
                }
                else
                {
                    wxASSERT(axisLonComp[axisLonComp.size()-1]>=axisLonComp[0]);

                    // Condition for change: The composite must not be fully outside (considered as handled) and the limit is not the coordinate grid border.
                    if (axisLonComp[axisLonComp.size()-1]>axisDataLon[0] && axisLonComp[0]<axisDataLon[0] && axisLonComp[axisLonComp.size()-1]!=dataArea->GetAxisUmax())
                    {
                        asLogMessage(_("Correcting the longitude extent according to the file limits."));
                        double Uwidth = axisDataLon[0]-dataArea->GetAbsoluteUmin();
                        wxASSERT(Uwidth>=0);
                        int Uptsnb = 1+Uwidth/dataArea->GetUstep();
                        asLogMessage(wxString::Format(_("Uptsnb = %d."), Uptsnb));
                        asGeoAreaCompositeGrid* newdataArea = asGeoAreaCompositeGrid::GetInstance(WGS84, dataArea->GetGridTypeString(),
                                                                            dataArea->GetAbsoluteUmin(), Uptsnb,
                                                                            dataArea->GetUstep(), dataArea->GetAbsoluteVmin(),
                                                                            dataArea->GetVaxisPtsnb(), dataArea->GetVstep(),
                                                                            dataArea->GetLevel(), asNONE, asFLAT_ALLOWED);

                        wxDELETE(dataArea);
                        dataArea = newdataArea;
                    }
                }
            }

            Array1DDouble axisLon = dataArea->GetUaxis();
            m_AxisLon.resize(axisLon.size());
            for (int i=0; i<axisLon.size(); i++)
            {
                m_AxisLon[i] = (float)axisLon[i];
            }
            m_LonPtsnb = dataArea->GetUaxisPtsnb();
            wxASSERT_MSG(m_AxisLon.size()==m_LonPtsnb, wxString::Format("m_AxisLon.size()=%d, m_LonPtsnb=%d",(int)m_AxisLon.size(),m_LonPtsnb));

            // Check that requested data do not overtake the file
            for (int i_comp=0; i_comp<dataArea->GetNbComposites(); i_comp++)
            {
                Array1DDouble axisLatComp = dataArea->GetVaxisComposite(i_comp);

                if (axisDataLat[axisDataLatLength-1]>axisDataLat[0])
                {
                    wxASSERT(axisLatComp[axisLatComp.size()-1]>=axisLatComp[0]);

                    // Condition for change: The composite must not be fully outside (considered as handled).
                    if (axisLatComp[axisLatComp.size()-1]>axisDataLat[axisDataLatLength-1] && axisLatComp[0]<axisDataLat[axisDataLatLength-1])
                    {
                        asLogMessage(_("Correcting the latitude extent according to the file limits."));
                        double Vwidth = axisDataLat[axisDataLatLength-1]-dataArea->GetAbsoluteVmin();
                        wxASSERT(Vwidth>=0);
                        int Vptsnb = 1+Vwidth/dataArea->GetVstep();
                        asLogMessage(wxString::Format(_("Vptsnb = %d."), Vptsnb));
                        asGeoAreaCompositeGrid* newdataArea = asGeoAreaCompositeGrid::GetInstance(WGS84, dataArea->GetGridTypeString(),
                                                                            dataArea->GetAbsoluteUmin(), dataArea->GetUaxisPtsnb(),
                                                                            dataArea->GetUstep(), dataArea->GetAbsoluteVmin(),
                                                                            Vptsnb, dataArea->GetVstep(),
                                                                            dataArea->GetLevel(), asNONE, asFLAT_ALLOWED);

                        wxDELETE(dataArea);
                        dataArea = newdataArea;
                    }

                }
                else
                {
                    wxASSERT(axisLatComp[axisLatComp.size()-1]>=axisLatComp[0]);

                    // Condition for change: The composite must not be fully outside (considered as handled).
                    if (axisLatComp[axisLatComp.size()-1]>axisDataLat[0] && axisLatComp[0]<axisDataLat[0])
                    {
                        asLogMessage(_("Correcting the latitude extent according to the file limits."));
                        double Vwidth = axisDataLat[0]-dataArea->GetAbsoluteVmin();
                        wxASSERT(Vwidth>=0);
                        int Vptsnb = 1+Vwidth/dataArea->GetVstep();
                        asLogMessage(wxString::Format(_("Vptsnb = %d."), Vptsnb));
                        asGeoAreaCompositeGrid* newdataArea = asGeoAreaCompositeGrid::GetInstance(WGS84, dataArea->GetGridTypeString(),
                                                                            dataArea->GetAbsoluteUmin(), dataArea->GetUaxisPtsnb(),
                                                                            dataArea->GetUstep(), dataArea->GetAbsoluteVmin(),
                                                                            Vptsnb, dataArea->GetVstep(),
                                                                            dataArea->GetLevel(), asNONE, asFLAT_ALLOWED);

                        wxDELETE(dataArea);
                        dataArea = newdataArea;
                    }
                }
            }

            Array1DDouble axisLat = dataArea->GetVaxis();
            m_AxisLat.resize(axisLat.size());
            for (int i=0; i<axisLat.size(); i++)
            {
                // Latitude axis in reverse order
                m_AxisLat[i] = (float)axisLat[axisLat.size()-1-i];
            }
            m_LatPtsnb = dataArea->GetVaxisPtsnb();
            wxASSERT_MSG(m_AxisLat.size()==m_LatPtsnb, wxString::Format("m_AxisLat.size()=%d, m_LatPtsnb=%d",(int)m_AxisLat.size(),m_LatPtsnb));

            m_Data.reserve(m_SizeTime*m_LonPtsnb*m_LatPtsnb);
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
            valFirstTime += m_TimeStepHours/24.0;
            indexStartTime++;
        }
        if (indexStartTime+indexLengthTime>axisDataTimeLength)
        {
            indexLengthTime--;
            cutEnd++;
        }

        // Get the number of iterations
        int iterationNb = 1;
        if (desiredArea)
        {
            iterationNb = dataArea->GetNbComposites();
        }

        // Containers for extraction
        VectorInt vectIndexLengthLat;
        VectorInt vectIndexLengthLon;
        VectorInt vectTotLength;
        VVectorShort vectData;

        for (int i_area = 0; i_area<iterationNb; i_area++)
        {
            int indexStartLon, indexStartLat, indexLengthLon, indexLengthLat;
            if (desiredArea)
            {
                // Get the spatial extent
                float lonMin = dataArea->GetUaxisCompositeStart(i_area);
                //float lonMax = dataArea->GetUaxisCompositeEnd(i_area);
                float latMinStart = dataArea->GetVaxisCompositeStart(i_area);
                float latMinEnd = dataArea->GetVaxisCompositeEnd(i_area);

                // The dimensions lengths
                indexLengthLon = dataArea->GetUaxisCompositePtsnb(i_area);
                indexLengthLat = dataArea->GetVaxisCompositePtsnb(i_area);

                // Get the spatial indices of the desired data
                indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLonLength-1], lonMin, 0.01f);
                if(indexStartLon==asOUT_OF_RANGE)
                {
                    // If not found, try with negative angles
                    indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLonLength-1], lonMin-360, 0.01f);
                }
                if(indexStartLon==asOUT_OF_RANGE)
                {
                    // If not found, try with angles above 360 degrees
                    indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLonLength-1], lonMin+360, 0.01f);
                }
                if(indexStartLon<0)
                {
                    asLogError(wxString::Format("Cannot find lonMin (%f) in the array axisDataLon ([0]=%f -> [%d]=%f) ", lonMin, axisDataLon[0], (int)axisDataLonLength, axisDataLon[axisDataLonLength-1]));
                    return false;
                }
                wxASSERT_MSG(indexStartLon>=0, wxString::Format("axisDataLon[0] = %f, &axisDataLon[%d] = %f & lonMin = %f", axisDataLon[0], (int)axisDataLonLength, axisDataLon[axisDataLonLength-1], lonMin));

                int indexStartLat1 = asTools::SortedArraySearch(&axisDataLat[0], &axisDataLat[axisDataLatLength-1], latMinStart, 0.01f);
                int indexStartLat2 = asTools::SortedArraySearch(&axisDataLat[0], &axisDataLat[axisDataLatLength-1], latMinEnd, 0.01f);
                wxASSERT_MSG(indexStartLat1>=0, wxString::Format("Looking for %g in %g to %g", latMinStart, axisDataLat[0], axisDataLat[axisDataLatLength-1]));
                wxASSERT_MSG(indexStartLat2>=0, wxString::Format("Looking for %g in %g to %g", latMinEnd, axisDataLat[0], axisDataLat[axisDataLatLength-1]));
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
                indexLevel = asTools::SortedArraySearch(&axisDataLevel[0], &axisDataLevel[axisDataLevelLength-1], m_Level, 0.01f);
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
            if (nDims==4)
            {
                // Set the indices for data
                size_t indexStartData[4] = {0,0,0,0};
                indexStartData[0] = indexStartTime;
                indexStartData[1] = indexLevel;
                indexStartData[2] = indexStartLat;
                indexStartData[3] = indexStartLon;
                size_t indexCountData[4] = {0,0,0,0};
                indexCountData[0] = indexLengthTime;
                indexCountData[1] = 1;
                indexCountData[2] = indexLengthLat;
                indexCountData[3] = indexLengthLon;
                ptrdiff_t indexStrideData[4] = {0,0,0,0};
                indexStrideData[0] = indexStepTime;
                indexStrideData[1] = 1;
                indexStrideData[2] = indexStepLat;
                indexStrideData[3] = indexStepLon;

                // In the netCDF Common Data Language, variables are printed with the outermost dimension first and the innermost dimension last.
                ncFile.GetVarSample(m_FileVariableName, indexStartData, indexCountData, indexStrideData, &data[indexBegining]);
            }
            else
            {
                // Set the indices for data
                size_t indexStartData[3] = {0,0,0};
                indexStartData[0] = indexStartTime;
                indexStartData[1] = indexStartLat;
                indexStartData[2] = indexStartLon;
                size_t indexCountData[3] = {0,0,0};
                indexCountData[0] = indexLengthTime;
                indexCountData[1] = indexLengthLat;
                indexCountData[2] = indexLengthLon;
                ptrdiff_t indexStrideData[3] = {0,0,0};
                indexStrideData[0] = indexStepTime;
                indexStrideData[1] = indexStepLat;
                indexStrideData[2] = indexStepLon;

                // In the netCDF Common Data Language, variables are printed with the outermost dimension first and the innermost dimension last.
                ncFile.GetVarSample(m_FileVariableName, indexStartData, indexCountData, indexStrideData, &data[indexBegining]);
            }

            // Keep data for later treatment
            vectIndexLengthLat.push_back(indexLengthLat);
            vectIndexLengthLon.push_back(indexLengthLon);
            vectTotLength.push_back(totLength);
            vectData.push_back(data);
        }

        // Close the nc file
        ncFile.Close();

        ThreadsManager().CritSectionNetCDF().Leave();

        // The container for extracted data from every composite
        VVArray2DFloat compositeData;

        // Treat data
        for (int i_area = 0; i_area<iterationNb; i_area++)
        {
            // Extract data
            int indexLengthLat = vectIndexLengthLat[i_area];
            int indexLengthLon = vectIndexLengthLon[i_area];
            int totLength = vectTotLength[i_area];
            VectorShort data = vectData[i_area];

            // Containers for results
            Array2DFloat latlonData(indexLengthLat,indexLengthLon);
            VArray2DFloat latlonTimeData;
            latlonTimeData.reserve(totLength);
            int ind = 0;

            // Loop to extract the data from the array
            for (int i_time=0; i_time<indexLengthTimeArray; i_time++)
            {
                for (int i_lat=0; i_lat<indexLengthLat; i_lat++)
                {
                    for (int i_lon=0; i_lon<indexLengthLon; i_lon++)
                    {
                        ind = i_lon;
                        ind += i_lat * indexLengthLon;
                        ind += i_time * indexLengthLon * indexLengthLat;

                        if (scalingNeeded)
                        {
                            // Add the Offset and multiply by the Scale Factor
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
                }

                latlonTimeData.push_back(latlonData);
                latlonData.setZero(indexLengthLat,indexLengthLon);

                counterTime++;
            }

            compositeData.push_back(latlonTimeData);
            latlonTimeData.clear();
            data.clear();
        }

        // Merge the composites into m_Data
        if (!MergeComposites(compositeData, dataArea))
        {
            wxDELETE(dataArea);
            return false;
        }

        if (desiredArea)
        {
            // Interpolate the loaded data on the desired grid
            if (!InterpolateOnGrid(dataArea, desiredArea))
            {
                wxDELETE(dataArea);
                return false;
            }
        }

        // Check the time dimension
        int compositesNb = 1;
        if (desiredArea)
        {
            compositesNb = dataArea->GetNbComposites();
        }
        counterTime /= compositesNb;
        if (!CheckTimeLength(counterTime))
        {
            wxDELETE(dataArea);
            return false;
        }

        wxDELETE(dataArea);
    }
    catch(bad_alloc& ba)
    {
        wxString msg(ba.what(), wxConvUTF8);
        asLogError(wxString::Format(_("Bad allocation caught when loading archive data: %s"), msg.c_str()));
        return false;
    }
    catch(asException& e)
    {
        wxString fullMessage = e.GetFullMessage();
        if (!fullMessage.IsEmpty())
        {
            asLogError(fullMessage);
        }
        asLogError(_("Failed to load data."));
        return false;
    }

    return true;
}

