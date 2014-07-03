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

#include "asDataPredictand.h"

#include <asCatalogPredictands.h>
#include <asFileDat.h>
#include <asFileNetcdf.h>
#include <asTimeArray.h>
#include <asDataPredictandPrecipitation.h>
#include <asDataPredictandLightnings.h>
#include <asDataPredictandTemperature.h>


asDataPredictand::asDataPredictand(DataParameter dataParameter, DataTemporalResolution dataTemporalResolution, DataSpatialAggregation dataSpatialAggregation)
{
    m_DataParameter = dataParameter;
    m_DataTemporalResolution = dataTemporalResolution;
    m_DataSpatialAggregation = dataSpatialAggregation;
    m_FileVersion = 1.2f;
    m_HasNormalizedData = false;
    m_HasReferenceValues = false;
    m_DatasetId = wxEmptyString;
    m_TimeStepDays = 0.0;
    m_TimeLength = 0;
    m_StationsNb = 0;
    m_DateProcessed = 0.0;
    m_DateStart = 0.0;
    m_DateEnd = 0.0;
}

asDataPredictand::~asDataPredictand()
{
    //dtor
}

asDataPredictand* asDataPredictand::GetInstance(const wxString& dataParameterStr, const wxString& dataTemporalResolutionStr, const wxString& dataSpatialAggregationStr)
{
    DataParameter dataParameter = asGlobEnums::StringToDataParameterEnum(dataParameterStr);
    DataTemporalResolution dataTemporalResolution = asGlobEnums::StringToDataTemporalResolutionEnum(dataTemporalResolutionStr);
    DataSpatialAggregation dataSpatialAggregation = asGlobEnums::StringToDataSpatialAggregationEnum(dataSpatialAggregationStr);

    if (dataParameter==NoDataParameter)
    {
        asLogError(_("The given data parameter is unknown. Cannot get an instance of the predictand DB."));
        return NULL;
    }

    if (dataTemporalResolution==NoDataTemporalResolution)
    {
        asLogError(_("The given data temporal resolution is unknown. Cannot get an instance of the predictand DB."));
        return NULL;
    }

    if (dataSpatialAggregation==NoDataSpatialAggregation)
    {
        asLogError(_("The given data spatial aggregation is unknown. Cannot get an instance of the predictand DB."));
        return NULL;
    }

    asDataPredictand* db = asDataPredictand::GetInstance(dataParameter, dataTemporalResolution, dataSpatialAggregation);
    return db;
}

asDataPredictand* asDataPredictand::GetInstance(DataParameter dataParameter, DataTemporalResolution dataTemporalResolution, DataSpatialAggregation dataSpatialAggregation)
{
    switch (dataParameter)
    {
        case (Precipitation):
        {
            asDataPredictand* db = new asDataPredictandPrecipitation(dataParameter, dataTemporalResolution, dataSpatialAggregation);
            return db;
        }
        case (AirTemperature):
        {
            asDataPredictand* db = new asDataPredictandTemperature(dataParameter, dataTemporalResolution, dataSpatialAggregation);
            return db;
        }
        case (Lightnings):
        {
            asDataPredictand* db = new asDataPredictandLightnings(dataParameter, dataTemporalResolution, dataSpatialAggregation);
            return db;
        }
        default:
            asLogError(_("The predictand parameter is not listed in the asDataPredictand instance factory."));
            return NULL;
    }
}

asDataPredictand* asDataPredictand::GetInstance(const wxString& filePath)
{
    // Open the NetCDF file
    asLogMessage(wxString::Format(_("Opening the file %s"), filePath.c_str()));
    asFileNetcdf ncFile(filePath, asFileNetcdf::ReadOnly);
    if(!ncFile.Open())
    {
        asLogError(wxString::Format(_("Couldn't open file %s"), filePath.c_str()));
        return NULL;
    }
    else
    {
        asLogMessage(_("File successfully opened"));
    }

    // Check version
    float version = ncFile.GetAttFloat("version");
    if (asTools::IsNaN(version) || version<1.1)
    {
        asLogError(_("The predictand DB file was made with an older version of AtmoSwing that is no longer supported. Please generate the file with the actual version."));
        return NULL;
    }

    // Get basic information
    DataParameter dataParameter = (DataParameter)ncFile.GetAttInt("data_parameter");
    DataTemporalResolution dataTemporalResolution = (DataTemporalResolution)ncFile.GetAttInt("data_temporal_resolution");
    DataSpatialAggregation dataSpatialAggregation = (DataSpatialAggregation)ncFile.GetAttInt("data_spatial_aggregation");

    // Close the netCDF file
    ncFile.Close();

    // Get instance
    asDataPredictand* db = asDataPredictand::GetInstance(dataParameter, dataTemporalResolution, dataSpatialAggregation);
    return db;
}

wxString asDataPredictand::GetDBFilePathSaving(const wxString &AlternateDestinationDir)
{
    wxString PredictandDBFilePath;

    wxString dataParameterStr = asGlobEnums::DataParameterEnumToString(m_DataParameter);
    wxString dataTemporalResolutionStr = asGlobEnums::DataTemporalResolutionEnumToString(m_DataTemporalResolution);
    wxString dataSpatialAggregationStr = asGlobEnums::DataSpatialAggregationEnumToString(m_DataSpatialAggregation);
    wxString FileName = dataParameterStr + "-" + dataTemporalResolutionStr + "-" + dataSpatialAggregationStr + "-" + m_DatasetId;

    if (AlternateDestinationDir.IsEmpty())
    {
        ThreadsManager().CritSectionConfig().Enter();
        PredictandDBFilePath = wxFileConfig::Get()->Read("/StandardPaths/DataPredictandDBDir", asConfig::GetDefaultUserWorkingDir()) + DS + FileName + ".nc";
        ThreadsManager().CritSectionConfig().Leave();
    }
    else
    {
        PredictandDBFilePath = AlternateDestinationDir + DS + FileName + ".nc";
    }

    return PredictandDBFilePath;
}

bool asDataPredictand::InitMembers(const wxString &catalogFilePath)
{
    // Starting and ending date of the DB, to be overwritten
    m_DateStart = asTime::GetMJD(2100,1,1);
    m_DateEnd = asTime::GetMJD(1800,1,1);

    // Get the catalog information
    asCatalogPredictands catalog(catalogFilePath);
    if(!catalog.Load()) return false;

    // Get first and last date
    if (catalog.GetStart()<m_DateStart)
        m_DateStart = catalog.GetStart();
    if (catalog.GetEnd()>m_DateEnd)
        m_DateEnd = catalog.GetEnd();

    // Get dataset ID
    m_DatasetId = catalog.GetSetId();

    // Get the number of stations
    asCatalog::DataIdListInt datListCheck = asCatalog::GetDataIdListInt(Predictand, wxEmptyString, catalogFilePath);
    m_StationsNb = datListCheck.Id.size();

    // Get the timestep
    m_TimeStepDays = catalog.GetTimeStepDays();

    // Get the time length
    m_TimeLength = ((m_DateEnd-m_DateStart) / m_TimeStepDays) + 1;

    // Get time array
    asTimeArray timeArray(m_DateStart, m_DateEnd, m_TimeStepDays*24.0, asTimeArray::Simple);
    timeArray.Init();
    m_Time = timeArray.GetTimeArray();

    return true;
}

bool asDataPredictand::InitBaseContainers()
{
    if (m_StationsNb<1)
    {
        asLogError(_("The stations number is inferior to 1."));
        return false;
    }
    if (m_TimeLength<1)
    {
        asLogError(_("The time length is inferior to 1."));
        return false;
    }
    m_StationsName.resize(m_StationsNb);
    m_StationsIds.resize(m_StationsNb);
    m_StationsLocCoordU.resize(m_StationsNb);
    m_StationsLocCoordV.resize(m_StationsNb);
    m_StationsLon.resize(m_StationsNb);
    m_StationsLat.resize(m_StationsNb);
    m_StationsHeight.resize(m_StationsNb);
    m_StationsStart.resize(m_StationsNb);
    m_StationsEnd.resize(m_StationsNb);
    m_Time.resize(m_TimeLength);
    m_DataGross.resize(m_TimeLength, m_StationsNb);
    m_DataGross.fill(NaNFloat);
    if(m_HasNormalizedData)
    {
        m_DataNormalized.resize(m_TimeLength, m_StationsNb);
        m_DataNormalized.fill(NaNFloat);
    }

    return true;
}

bool asDataPredictand::LoadCommonData(asFileNetcdf &ncFile)
{
    // Check version
    float version = ncFile.GetAttFloat("version");
    if (asTools::IsNaN(version) || version<1.2)
    {
        asLogError(_("The predictand DB file was made with an older version of AtmoSwing that is no longer supported. Please generate the file with the actual version."));
        return false;
    }

    // Get global attributes
    m_DataParameter = (DataParameter)ncFile.GetAttInt("data_parameter");
    m_DataTemporalResolution = (DataTemporalResolution)ncFile.GetAttInt("data_temporal_resolution");
    m_DataSpatialAggregation = (DataSpatialAggregation)ncFile.GetAttInt("data_spatial_aggregation");
    m_DatasetId = ncFile.GetAttString("dataset_id");

    // Get time
    m_TimeLength = ncFile.GetDimLength("time");
    m_Time.resize( m_TimeLength );
    ncFile.GetVar("time", &m_Time[0]);

    // Get stations properties
    m_StationsNb = ncFile.GetDimLength("stations");
    wxASSERT(m_StationsNb>0);
    m_StationsName.resize( m_StationsNb );
    ncFile.GetVar("stations_name", &m_StationsName[0], m_StationsNb);
    m_StationsIds.resize( m_StationsNb );
    ncFile.GetVar("stations_ids", &m_StationsIds[0]);
    m_StationsHeight.resize( m_StationsNb );
    ncFile.GetVar("stations_height", &m_StationsHeight[0]);
    m_StationsLon.resize( m_StationsNb );
    ncFile.GetVar("lon", &m_StationsLon[0]);
    m_StationsLat.resize( m_StationsNb );
    ncFile.GetVar("lat", &m_StationsLat[0]);
    m_StationsLocCoordU.resize( m_StationsNb );
    ncFile.GetVar("loc_coord_u", &m_StationsLocCoordU[0]);
    m_StationsLocCoordV.resize( m_StationsNb );
    ncFile.GetVar("loc_coord_v", &m_StationsLocCoordV[0]);
    m_StationsStart.resize( m_StationsNb );
    ncFile.GetVar("start", &m_StationsStart[0]);
    m_StationsEnd.resize( m_StationsNb );
    ncFile.GetVar("end", &m_StationsEnd[0]);

    // Get data
    size_t IndexStart[2] = {0,0};
    size_t IndexCount[2] = {size_t(m_TimeLength), size_t(m_StationsNb)};
    m_DataGross.resize( m_TimeLength, m_StationsNb );
    ncFile.GetVarArray("data_gross", IndexStart, IndexCount, &m_DataGross(0,0));

    return true;
}

void asDataPredictand::SetCommonDefinitions(asFileNetcdf &ncFile)
{
     // Define dimensions. Time is the unlimited dimension.
    ncFile.DefDim("stations", m_StationsNb);
    ncFile.DefDim("time");

    // The dimensions name array is used to pass the dimensions to the variable.
    VectorStdString DimNameTime;
    DimNameTime.push_back("time");
    VectorStdString DimNameStations;
    DimNameStations.push_back("stations");
    VectorStdString DimNames2D;
    DimNames2D.push_back("time");
    DimNames2D.push_back("stations");

    // Put general attributes
    ncFile.PutAtt("version", &m_FileVersion);
    int dataParameter = (int)m_DataParameter;
    ncFile.PutAtt("data_parameter", &dataParameter);
    int dataTemporalResolution = (int)m_DataTemporalResolution;
    ncFile.PutAtt("data_temporal_resolution", &dataTemporalResolution);
    int dataSpatialAggregation = (int)m_DataSpatialAggregation;
    ncFile.PutAtt("data_spatial_aggregation", &dataSpatialAggregation);
    ncFile.PutAtt("dataset_id", m_DatasetId);

    // Define variables: the scores and the corresponding dates
    ncFile.DefVar("time", NC_DOUBLE, 1, DimNameTime);
    ncFile.DefVar("data_gross", NC_FLOAT, 2, DimNames2D);
    ncFile.DefVar("stations_name", NC_STRING, 1, DimNameStations);
    ncFile.DefVar("stations_ids", NC_INT, 1, DimNameStations);
    ncFile.DefVar("stations_height", NC_FLOAT, 1, DimNameStations);
    ncFile.DefVar("lon", NC_DOUBLE, 1, DimNameStations);
    ncFile.DefVar("lat", NC_DOUBLE, 1, DimNameStations);
    ncFile.DefVar("loc_coord_u", NC_DOUBLE, 1, DimNameStations);
    ncFile.DefVar("loc_coord_v", NC_DOUBLE, 1, DimNameStations);
    ncFile.DefVar("start", NC_DOUBLE, 1, DimNameStations);
    ncFile.DefVar("end", NC_DOUBLE, 1, DimNameStations);

    // Put attributes for the stations
    ncFile.PutAtt("long_name", "Stations names", "stations_name");
    ncFile.PutAtt("var_desc", "Name of the predictand stations", "stations_name");

    // Put attributes for the stations
    ncFile.PutAtt("long_name", "Stations IDs", "stations_ids");
    ncFile.PutAtt("var_desc", "Internal IDs of the predictand stations", "stations_ids");

    // Put attributes for the stations
    ncFile.PutAtt("long_name", "Stations height", "stations_height");
    ncFile.PutAtt("var_desc", "Altitude of the predictand stations", "stations_height");
    ncFile.PutAtt("units", "m", "stations_height");

    // Put attributes for the lon variable
    ncFile.PutAtt("long_name", "Longitude", "lon");
    ncFile.PutAtt("var_desc", "Longitudes of the stations positions", "lon");
    ncFile.PutAtt("units", "degrees", "lon");

    // Put attributes for the lat variable
    ncFile.PutAtt("long_name", "Latitude", "lat");
    ncFile.PutAtt("var_desc", "Latitudes of the stations positions", "lat");
    ncFile.PutAtt("units", "degrees", "lat");

    // Put attributes for the loccoordu variable
    ncFile.PutAtt("long_name", "Local coordinate U", "loc_coord_u");
    ncFile.PutAtt("var_desc", "Local coordinate for the U axis (west-east)", "loc_coord_u");
    ncFile.PutAtt("units", "m", "loc_coord_u");

    // Put attributes for the loccoordv variable
    ncFile.PutAtt("long_name", "Local coordinate V", "loc_coord_v");
    ncFile.PutAtt("var_desc", "Local coordinate for the V axis (west-east)", "loc_coord_v");
    ncFile.PutAtt("units", "m", "loc_coord_v");

    // Put attributes for the start variable
    ncFile.PutAtt("long_name", "Start", "start");
    ncFile.PutAtt("var_desc", "Start of the stations data", "start");
    ncFile.PutAtt("units", "Modified Julian Day Number (MJD)", "start");

    // Put attributes for the end variable
    ncFile.PutAtt("long_name", "End", "end");
    ncFile.PutAtt("var_desc", "End of the stations data", "end");
    ncFile.PutAtt("units", "Modified Julian Day Number (MJD)", "end");

    // Put attributes for the data variable
    ncFile.PutAtt("long_name", "Gross data", "data_gross");
    ncFile.PutAtt("var_desc", "Gross data, whithout any treatment", "data_gross");

}

bool asDataPredictand::SaveCommonData(asFileNetcdf &ncFile)
{
    // Provide sizes for variables
    size_t startTime[] = {0};
    size_t countTime[] = {size_t(m_TimeLength)};
    size_t startStations[] = {0};
    size_t countStations[] = {size_t(m_StationsNb)};
    size_t start2[] = {0, 0};
    size_t count2[] = {size_t(m_TimeLength), size_t(m_StationsNb)};

    // Write data
    ncFile.PutVarArray("time", startTime, countTime, &m_Time(0));
    ncFile.PutVarArray("stations_name", startStations, countStations, &m_StationsName[0], m_StationsName.size());
    ncFile.PutVarArray("stations_ids", startStations, countStations, &m_StationsIds(0));
    ncFile.PutVarArray("stations_height", startStations, countStations, &m_StationsHeight(0));
    ncFile.PutVarArray("lon", startStations, countStations, &m_StationsLon(0));
    ncFile.PutVarArray("lat", startStations, countStations, &m_StationsLat(0));
    ncFile.PutVarArray("loc_coord_u", startStations, countStations, &m_StationsLocCoordU(0));
    ncFile.PutVarArray("loc_coord_v", startStations, countStations, &m_StationsLocCoordV(0));
    ncFile.PutVarArray("start", startStations, countStations, &m_StationsStart(0));
    ncFile.PutVarArray("end", startStations, countStations, &m_StationsEnd(0));
    ncFile.PutVarArray("data_gross", start2, count2, &m_DataGross(0,0));

    return true;
}

bool asDataPredictand::SetStationProperties(asCatalogPredictands &currentData, size_t stationIndex)
{
    m_StationsName[stationIndex] = currentData.GetStationName();
    m_StationsIds(stationIndex) = currentData.GetStationId();
    m_StationsLocCoordU(stationIndex) = currentData.GetStationCoord().u;
    m_StationsLocCoordV(stationIndex) = currentData.GetStationCoord().v;
// FIXME (Pascal#1#): Implement lon/lat
    m_StationsLon(stationIndex) = NaNDouble;
    m_StationsLat(stationIndex) = NaNDouble;
    m_StationsHeight(stationIndex) = currentData.GetStationHeight();
    m_StationsStart(stationIndex) = currentData.GetStationStart();
    m_StationsEnd(stationIndex) = currentData.GetStationEnd();

    return true;
}

bool asDataPredictand::ParseData(const wxString &catalogFilePath, const wxString &AlternateDataDir, const wxString &AlternatePatternDir)
{
    // Index for stations
    int stationIndex = 0;

    #if wxUSE_GUI
        // The progress bar
        asDialogProgressBar ProgressBar(_("Loading data from files.\n"), m_StationsNb);
    #endif

    // Get catalog
    asCatalogPredictands catalog(catalogFilePath);

    // Get the stations list
    asCatalog::DataIdListInt stationsList = catalog.GetDataIdListInt(Predictand, m_DatasetId, catalogFilePath);

    for (size_t i_station=0; i_station<stationsList.Id.size(); i_station++)
    {
        // The station ID
        int stationId = stationsList.Id[i_station];

        // Load data properties
        if(!catalog.Load(stationId)) return false;

        #if wxUSE_GUI
            // Update the progress bar.
            wxString fileNameMessage = wxString::Format(_("Loading data from files.\nFile: %s"), catalog.GetStationFilename().c_str());
            if(!ProgressBar.Update(stationIndex, fileNameMessage))
            {
                asLogError(_("The process has been canceled by the user."));
                return false;
            }
        #endif

        // Get station information
        if(!SetStationProperties(catalog, stationIndex)) return false;

        // Get file content
        if(!GetFileContent(catalog, stationIndex, AlternateDataDir, AlternatePatternDir)) return false;

        stationIndex++;
    }

    #if wxUSE_GUI
        ProgressBar.Destroy();
    #endif

    return true;
}

bool asDataPredictand::GetFileContent(asCatalogPredictands &currentData, size_t stationIndex, const wxString &AlternateDataDir, const wxString &AlternatePatternDir)
{
    // Load file
    wxString fileFullPath;
    if (!AlternateDataDir.IsEmpty())
    {
        fileFullPath = AlternateDataDir + DS + currentData.GetStationFilename();
    }
    else
    {
        fileFullPath = currentData.GetDataPath() + currentData.GetStationFilename();
    }
    asFileDat datFile(fileFullPath, asFile::ReadOnly);
    if(!datFile.Open()) return false;

    // Get the parsing format
    wxString stationFilePattern = currentData.GetStationFilepattern();
    asFileDat::Pattern filePattern = asFileDat::GetPattern(stationFilePattern, AlternatePatternDir);
    size_t maxCharWidth = asFileDat::GetPatternLineMaxCharWidth(filePattern);

    // Jump the header
    datFile.SkipLines(filePattern.HeaderLines);

    // Get first index on the tima axis
    int startIndex = asTools::SortedArraySearch(&m_Time[0], &m_Time[m_Time.size()-1], currentData.GetStationStart());
    if (startIndex==asOUT_OF_RANGE || startIndex==asNOT_FOUND)
    {
        asLogError(wxString::Format(_("The given start date for \"%s\" is out of the catalog range."), currentData.GetStationName().c_str()));
        return false;
    }

    int timeIndex = startIndex;

    // Parse every line until the end of the file
    while (!datFile.EndOfFile())
    {
        // Get current line
        wxString lineContent = datFile.GetLineContent();

        // Check the line width
        if (lineContent.Len()>=maxCharWidth)
        {
            // Check the size of the array
            if(timeIndex>=m_TimeLength)
            {
                asLogError(wxString::Format(_("The time index is larger than the matrix (timeIndex = %d, m_TimeLength = %d)."), (int)timeIndex, (int)m_TimeLength));
                return false;
            }

            switch (filePattern.StructType)
            {
                case (asFileDat::ConstantWidth):
                {
                    if(filePattern.ParseTime)
                    {
                        // Containers. Must be a double to use wxString::ToDouble
                        double valTimeYear=0, valTimeMonth=0, valTimeDay=0, valTimeHour=0, valTimeMinute=0;

                        // Get time value
                        if (filePattern.TimeYearBegin!=0 && filePattern.TimeYearEnd!=0 && filePattern.TimeMonthBegin!=0 && filePattern.TimeMonthEnd!=0 && filePattern.TimeDayBegin!=0 && filePattern.TimeDayEnd!=0)
                        {
                            lineContent.Mid(filePattern.TimeYearBegin-1, filePattern.TimeYearEnd-filePattern.TimeYearBegin+1).ToDouble(&valTimeYear);
                            lineContent.Mid(filePattern.TimeMonthBegin-1, filePattern.TimeMonthEnd-filePattern.TimeMonthBegin+1).ToDouble(&valTimeMonth);
                            lineContent.Mid(filePattern.TimeDayBegin-1, filePattern.TimeDayEnd-filePattern.TimeDayBegin+1).ToDouble(&valTimeDay);
                        } else {
                            asLogError(_("The data file pattern is not correctly defined."));
                            return false;
                        }

                        if (filePattern.TimeHourBegin!=0 && filePattern.TimeHourEnd!=0)
                        {
                            lineContent.Mid(filePattern.TimeHourBegin-1, filePattern.TimeHourEnd-filePattern.TimeHourBegin+1).ToDouble(&valTimeHour);
                        }
                        if (filePattern.TimeMinuteBegin!=0 && filePattern.TimeMinuteEnd!=0)
                        {
                            lineContent.Mid(filePattern.TimeMinuteBegin-1, filePattern.TimeMinuteEnd-filePattern.TimeMinuteBegin+1).ToDouble(&valTimeMinute);
                        }

                        double dateData = asTime::GetMJD(valTimeYear, valTimeMonth, valTimeDay, valTimeHour, valTimeMinute, 0);

                        // Check again date vector
                        if ( abs(dateData - m_Time(timeIndex)) > 0.0001)
                        {
                            wxString errorMessage = wxString::Format(_("Value in data : %6.4f (%s), value in time array : %6.4f (%s). In file %s"), dateData, asTime::GetStringTime(dateData,"DD.MM.YYYY").c_str(), m_Time(timeIndex), asTime::GetStringTime(m_Time(timeIndex),"DD.MM.YYYY").c_str(), currentData.GetStationFilename().c_str());
                            asLogError(wxString::Format(_("The time value doesn't match: %s"), errorMessage.c_str() ));
                            return false;
                        }
                    }

                    // Get Precipitation value
                    double valPrecipitationGross=0;
                    lineContent.Mid(filePattern.DataBegin-1, filePattern.DataEnd-filePattern.DataBegin+1).ToDouble(&valPrecipitationGross);

                    // Check if not NaN and store
                    bool notNan = true;
                    for (size_t i_nan=0; i_nan<currentData.GetNan().size(); i_nan++)
                    {
                        if (valPrecipitationGross==currentData.GetNan()[i_nan]) notNan = false;
                    }
                    if (notNan)
                    {
                        // Put value in the matrix
                        m_DataGross(timeIndex,stationIndex) = valPrecipitationGross;
                    }
                    timeIndex++;
                    break;
                }

                case (asFileDat::TabsDelimited):
                {
                    // Parse into a vector
                    VectorString vColumns;
                    wxString tmpLineContent = lineContent;
                    while( tmpLineContent.Find("\t") != wxNOT_FOUND )
                    {
                        int foundCol = tmpLineContent.Find("\t");
                        vColumns.push_back(tmpLineContent.Mid(0,foundCol));
                        tmpLineContent = tmpLineContent.Mid(foundCol+1);
                    }
                    if (!tmpLineContent.IsEmpty())
                    {
                        vColumns.push_back(tmpLineContent);
                    }

                    if(filePattern.ParseTime)
                    {
                        // Containers. Must be a double to use wxString::ToDouble
                        double valTimeYear=0, valTimeMonth=0, valTimeDay=0, valTimeHour=0, valTimeMinute=0;

                        // Get time value
                        if (filePattern.TimeYearBegin!=0 && filePattern.TimeMonthBegin!=0 && filePattern.TimeDayBegin!=0)
                        {
                            if ((unsigned)filePattern.TimeYearBegin>vColumns.size() || (unsigned)filePattern.TimeMonthBegin>vColumns.size() || (unsigned)filePattern.TimeDayBegin>vColumns.size())
                            {
                                asLogError(_("The data file pattern is not correctly defined. Trying to access an element (date) after the line width."));
                                return false;
                            }
                            vColumns[filePattern.TimeYearBegin-1].ToDouble(&valTimeYear);
                            vColumns[filePattern.TimeMonthBegin-1].ToDouble(&valTimeMonth);
                            vColumns[filePattern.TimeDayBegin-1].ToDouble(&valTimeDay);
                        } else {
                            asLogError(_("The data file pattern is not correctly defined."));
                            return false;
                        }

                        if (filePattern.TimeHourBegin!=0)
                        {
                            if ((unsigned)filePattern.TimeHourBegin>vColumns.size())
                            {
                                asLogError(_("The data file pattern is not correctly defined. Trying to access an element (hour) after the line width."));
                                return false;
                            }
                            vColumns[filePattern.TimeHourBegin-1].ToDouble(&valTimeHour);
                        }
                        if (filePattern.TimeMinuteBegin!=0)
                        {
                            if ((unsigned)filePattern.TimeMinuteBegin>vColumns.size())
                            {
                                asLogError(_("The data file pattern is not correctly defined. Trying to access an element (minute) after the line width."));
                                return false;
                            }
                            vColumns[filePattern.TimeMinuteBegin-1].ToDouble(&valTimeMinute);
                        }

                        double dateData = asTime::GetMJD(valTimeYear, valTimeMonth, valTimeDay, valTimeHour, valTimeMinute, 0);

                        // Check again date vector
                        if ( abs(dateData - m_Time(timeIndex)) > 0.001)
                        {
                            wxString errorMessage = wxString::Format(_("Value in data : %6.4f (%s), value in time array : %6.4f (%s). In file %s"), dateData, asTime::GetStringTime(dateData,"DD.MM.YYYY").c_str(), m_Time(timeIndex), asTime::GetStringTime(m_Time(timeIndex),"DD.MM.YYYY").c_str(), currentData.GetStationFilename().c_str());
                            asLogError(wxString::Format(_("The time value doesn't match: %s"), errorMessage.c_str() ));
                            return false;
                        }
                    }

                    // Get Precipitation value
                    double valPrecipitationGross=0;
                    vColumns[filePattern.DataBegin-1].ToDouble(&valPrecipitationGross);

                    // Check if not NaN and store
                    bool notNan = true;
                    for (size_t i_nan=0; i_nan<currentData.GetNan().size(); i_nan++)
                    {
                        if (valPrecipitationGross==currentData.GetNan()[i_nan]) notNan = false;
                    }
                    if (notNan)
                    {
                        // Put value in the matrix
                        m_DataGross(timeIndex,stationIndex) = valPrecipitationGross;
                    }
                    timeIndex++;
                    break;
                }
            }
        }
        else
        {
            if(lineContent.Len()>1)
            {
                asLogError(_("The line length doesn't match."));
                return false;
            }
        }
    }
    datFile.Close();

    // Get end index
    int endIndex = asTools::SortedArraySearch(&m_Time[0], &m_Time[m_Time.size()-1], currentData.GetStationEnd());
    if (endIndex==asOUT_OF_RANGE || endIndex==asNOT_FOUND)
    {
        asLogError(wxString::Format(_("The given end date for \"%s\" is out of the catalog range."), currentData.GetStationName().c_str()));
        return false;
    }

    // Check time width
    if (endIndex-startIndex!=timeIndex-startIndex-1)
    {
        wxString messageTime = wxString::Format(_("The length of the data in \"%s / %s\" is not coherent"), currentData.GetName().c_str(), currentData.GetStationName().c_str());
        asLogError(messageTime);
        return false;
    }

    return true;
}

Array2DFloat asDataPredictand::GetAnnualMax(double timeStepDays, int nansNbMax)
{
    // Flag to check the need of aggregation (timeStepDays>m_TimeStepDays)
    bool aggregate = false;
    int indexTimeSpanUp = 0;
    int indexTimeSpanDown = 0;

    if(timeStepDays==m_TimeStepDays)
    {
        aggregate = false;
    }
    else if(timeStepDays>m_TimeStepDays)
    {
        if(fmod(timeStepDays,m_TimeStepDays)>0.0000001)
        {
            asLogError(_("The timestep for the extraction of the predictands maximums has to be a multiple of the data timestep."));
            Array2DFloat emptyMatrix;
            emptyMatrix << NaNFloat;
            return emptyMatrix;
        }

        // Aggragation necessary
        aggregate = true;

        // indices to add or substract around the mid value
        indexTimeSpanUp = floor((timeStepDays/m_TimeStepDays)/2);
        indexTimeSpanDown = ceil((timeStepDays/m_TimeStepDays)/2)-1;
    }
    else
    {
        asLogError(_("The timestep for the extraction of the predictands maximums cannot be lower than the data timestep."));
        Array2DFloat emptyMatrix;
        emptyMatrix << NaNFloat;
        return emptyMatrix;
    }

    // Keep the real indices of years
    int indYearStart = 0;
    int indYearEnd = 0;

    // Get catalog beginning and end
    int yearStart = asTime::GetYear(m_DateStart);
    if (asTime::GetMonth(m_DateStart)!=1 || asTime::GetDay(m_DateStart)!=1)
    {
        yearStart++;
        indYearStart++;
    }
    int yearEnd = asTime::GetYear(m_DateEnd);
    indYearEnd = yearEnd-yearStart+indYearStart;
    if (asTime::GetMonth(m_DateEnd)!=12 || asTime::GetDay(m_DateEnd)!=31)
    {
        yearEnd--;
    }

    // Create the container
    Array2DFloat maxMatrix = Array2DFloat::Constant(m_StationsNb, indYearEnd+1, NaNFloat);

    // Look for maximums
    for (int i_stnb=0; i_stnb<m_StationsNb; i_stnb++)
    {
        for (int i_year=yearStart; i_year<=yearEnd; i_year++)
        {
            // The maximum value and a flag for accepted NaNs
            float annualmax = -99999;
            int nansNb = 0;

            // Find begining and end of the year
            int rowstart = asTools::SortedArraySearchFloor(&m_Time[0], &m_Time[m_TimeLength-1], asTime::GetMJD(i_year, 1, 1), asHIDE_WARNINGS );
            int rowend = asTools::SortedArraySearchFloor(&m_Time[0], &m_Time[m_TimeLength-1], asTime::GetMJD(i_year, 12, 31, 59, 59), asHIDE_WARNINGS);
            if ( (rowend==asOUT_OF_RANGE) | (rowend==asNOT_FOUND) )
            {
                if (i_year==yearEnd)
                {
                    rowend = m_TimeLength-1;
                }
                else
                {
                    annualmax = NaNFloat;
                }
            }
            rowend -= 1;

            // Get max
            if(!aggregate)
            {
                for (int i_row=rowstart; i_row<=rowend; i_row++)
                {
                    if (!asTools::IsNaN(m_DataGross(i_row, i_stnb)))
                    {
                        annualmax = wxMax(m_DataGross(i_row, i_stnb),annualmax);
                    }
                    else
                    {
                        nansNb++;
                    }
                }
                if (nansNb>nansNbMax)
                {
                    annualmax = NaNFloat;
                }
            }
            else
            {
                // Correction for both extremes
                rowstart = wxMax(rowstart-indexTimeSpanDown, 0);
                rowstart += indexTimeSpanDown;
                rowend = wxMin(rowend+indexTimeSpanUp, (int)m_DataGross.rows()-1);
                rowend -= indexTimeSpanUp;

                // Loop within the new limits
                for (int i_row=rowstart; i_row<=rowend; i_row++)
                {
                    float timeStepSum = 0;
                    for (int i_element=i_row-indexTimeSpanDown; i_element<=i_row+indexTimeSpanUp; i_element++)
                    {
                        if (!asTools::IsNaN(m_DataGross(i_element, i_stnb)))
                        {
                            timeStepSum += m_DataGross(i_element, i_stnb);
                        }
                        else
                        {
                            timeStepSum = NaNFloat;
                            break;
                        }
                    }

                    if (!asTools::IsNaN(timeStepSum))
                    {
                        annualmax = wxMax(timeStepSum,annualmax);
                    }
                    else
                    {
                        nansNb++;
                    }
                }
                if (nansNb>nansNbMax)
                {
                    annualmax = NaNFloat;
                }
            }

            maxMatrix(i_stnb, i_year-yearStart+indYearStart) = annualmax;
        }
    }

    return maxMatrix;
}

int asDataPredictand::GetStationIndex(int stationId)
{
    return asTools::SortedArraySearch(&m_StationsIds[0], &m_StationsIds[m_StationsNb-1], stationId);
}
