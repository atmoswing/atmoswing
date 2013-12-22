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
 
#include "asDataPredictorArchiveNoaaOisst2.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileNetcdf.h>


asDataPredictorArchiveNoaaOisst2::asDataPredictorArchiveNoaaOisst2(const wxString &dataId)
:
asDataPredictorArchive(dataId)
{
    // Set the basic properties.
    m_Initialized = false;
    m_DataId = dataId;
    m_DatasetId = "NOAA_OISST_v2";
    m_OriginalProvider = "NOAA";
    m_FinalProvider = "NOAA";
    m_FinalProviderWebsite = "http://www.esrl.noaa.gov/psd/data/gridded/data.noaa.oisst.v2.html";
    m_FinalProviderFTP = "ftp://eclipse.ncdc.noaa.gov/pub/OI-daily-v2";
    m_DatasetName = "Optimum Interpolation Sea Surface Temperature, version 2";
    m_OriginalProviderStart = asTime::GetMJD(1982, 1, 1);
    m_OriginalProviderEnd = NaNDouble;
    m_TimeZoneHours = 0;
    m_TimeStepHours = 24;
    m_FirstTimeStepHours = 12;
    m_NanValues.push_back(32767);
    m_NanValues.push_back(936*10^34);
    m_CoordinateSystem = CoordSys::WGS84;
    m_UaxisShift = 0.125;
    m_VaxisShift = 0.125;
    m_UaxisStep = 0.25;
    m_VaxisStep = 0.25;
    m_SubFolder = wxEmptyString;
    m_FileNamePattern = "%d/AVHRR/sst4-path-eot.%4d%02d%02d.nc";

    // Identify data ID and set the corresponding properties.
    if (m_DataId.IsSameAs("sst", false))
    {
        m_DataParameter = SeaSurfaceTemperature;
        m_FileVariableName = "sst";
        m_Unit = DataUnit::degC;
    }
    else if (m_DataId.IsSameAs("sst_anom", false))
    {
        m_DataParameter = SeaSurfaceTemperatureAnomaly;
        m_FileVariableName = "anom";
        m_Unit = DataUnit::degC;
    }
    else
    {
        m_DataParameter = NoDataParameter;
        m_FileVariableName = wxEmptyString;
        m_Unit = DataUnit::NoDataUnit;
    }

}

asDataPredictorArchiveNoaaOisst2::~asDataPredictorArchiveNoaaOisst2()
{

}

bool asDataPredictorArchiveNoaaOisst2::Init()
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

VectorString asDataPredictorArchiveNoaaOisst2::GetDataIdList()
{
    VectorString list;

    list.push_back("sst"); // Sea surface temperature
    list.push_back("sst_anom"); // Anomaly

    return list;
}

VectorString asDataPredictorArchiveNoaaOisst2::GetDataIdDescriptionList()
{
    VectorString list;

    list.push_back("Sea surface temperature");
    list.push_back("Sea surface temperature anomaly");

    return list;
}

bool asDataPredictorArchiveNoaaOisst2::Load(asGeoAreaCompositeGrid *desiredArea, asTimeArray &timeArray)
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

        // Get requested dates
        double dateFirst = timeArray.GetFirst();
        double dateLast = timeArray.GetLast();

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
                dataUmax = desiredArea->GetAbsoluteUmax();
                dataVmax = desiredArea->GetAbsoluteVmax();
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

        #if wxUSE_GUI
            // The progress bar
            asDialogProgressBar progressBar(_("Loading data from files.\n"), dateLast-dateFirst);
        #endif

        // Loop through the files
        for (double date=dateFirst; date<=dateLast; date++)
        {
            // Build the file path (ex: %d/AVHRR/sst4-path-eot.%4d%02d%02d.nc)
            wxString fileName = wxString::Format(m_FileNamePattern, asTime::GetYear(date), asTime::GetYear(date), asTime::GetMonth(date), asTime::GetDay(date));
            wxString fileFullPath = m_DirectoryPath + fileName;

            #if wxUSE_GUI
                // Update the progress bar
                wxString fileNameMessage = wxString::Format(_("Loading data from files.\nFile: %s"), fileName.c_str());
                if(!progressBar.Update(date-dateFirst, fileNameMessage))
                {
                    asLogWarning(_("The process has been canceled by the user."));
                    wxDELETE(dataArea);
                    return false;
                }
            #endif

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

            if (desiredArea==NULL && date==dateFirst)
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
            else if(desiredArea!=NULL && date==dateFirst)
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

            // Get the time length
            double timeArrayIndex = timeArray.GetIndexFirstAfter(date);
            int indexLengthTime = 1;
            int indexLengthTimeArray = indexLengthTime;

            // Correct the time start and end
            size_t indexStartTime = 0;
            int cutStart = 0;
            if(date==dateFirst)
            {
                cutStart = timeArrayIndex;
            }
            int cutEnd = 0;
            while (valFirstTime<timeArray[timeArrayIndex])
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
            VectorBool vectLoad360;
            VectorInt vectTotLength;
            VVectorShort vectData;
            VVectorShort vectData360;

            for (int i_area = 0; i_area<iterationNb; i_area++)
            {
                // Check if necessary to load the data of lon=360 (so lon=0)
                bool load360 = false;

                int indexStartLon, indexStartLat, indexLengthLon, indexLengthLat;
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

                    if(lonMax==dataArea->GetAxisUmax())
                    {
                        // Correction if the lon 360 degrees is required (doesn't exist)
                        load360 = true;
                        for (int i_check = 0; i_check<iterationNb; i_check++)
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
                    indexStrideData4[0] = indexStepTime;
                    indexStrideData4[1] = 1;
                    indexStrideData4[2] = indexStepLat;
                    indexStrideData4[3] = indexStepLon;

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
                    indexStrideData3[0] = indexStepTime;
                    indexStrideData3[1] = indexStepLat;
                    indexStrideData3[2] = indexStepLon;

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
                    indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLonLength-1], 360, 0.01f);
                    if(indexStartLon==asOUT_OF_RANGE)
                    {
                        // If not found, try with negative angles
                        indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLonLength-1], 0, 0.01f);
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
                vectTotLength.push_back(totLength);
                vectData.push_back(data);
                vectData360.push_back(data360);
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
                bool load360 = vectLoad360[i_area];
                int totLength = vectTotLength[i_area];
                VectorShort data = vectData[i_area];
                VectorShort data360 = vectData360[i_area];

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

                        if(load360)
                        {
                            ind = i_lat;
                            ind += i_time * indexLengthLat;

                            if (scalingNeeded)
                            {
                                // Add the Offset and multiply by the Scale Factor
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

                    latlonTimeData.push_back(latlonData);

                    if(load360)
                    {
                        latlonData.setZero(indexLengthLat,indexLengthLon+1);
                    }
                    else
                    {
                        latlonData.setZero(indexLengthLat,indexLengthLon);
                    }
                    counterTime++;
                }

                compositeData.push_back(latlonTimeData);
                latlonTimeData.clear();
                data.clear();
                data360.clear();
            }

            // Merge the composites into m_Data
            if (!MergeComposites(compositeData, dataArea))
            {
                wxDELETE(dataArea);
                return false;
            }
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

        #if wxUSE_GUI
            progressBar.Destroy();
        #endif

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

