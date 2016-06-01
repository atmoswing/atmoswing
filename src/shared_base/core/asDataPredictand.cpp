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

#include "asDataPredictand.h"

#include <asCatalogPredictands.h>
#include <asFileDat.h>
#include <asFileNetcdf.h>
#include <asTimeArray.h>
#include <asDataPredictandPrecipitation.h>
#include <asDataPredictandLightnings.h>
#include <asDataPredictandTemperature.h>


asDataPredictand::asDataPredictand(DataParameter dataParameter, DataTemporalResolution dataTemporalResolution,
                                   DataSpatialAggregation dataSpatialAggregation)
{
    m_dataParameter = dataParameter;
    m_dataTemporalResolution = dataTemporalResolution;
    m_dataSpatialAggregation = dataSpatialAggregation;
    m_fileVersion = 1.4f;
    m_hasNormalizedData = false;
    m_hasReferenceValues = false;
    m_datasetId = wxEmptyString;
    m_timeStepDays = 0.0;
    m_timeLength = 0;
    m_stationsNb = 0;
    m_dateProcessed = 0.0;
    m_dateStart = 0.0;
    m_dateEnd = 0.0;
}

asDataPredictand::~asDataPredictand()
{
    //dtor
}

asDataPredictand *asDataPredictand::GetInstance(const wxString &dataParameterStr,
                                                const wxString &dataTemporalResolutionStr,
                                                const wxString &dataSpatialAggregationStr)
{
    DataParameter dataParameter = asGlobEnums::StringToDataParameterEnum(dataParameterStr);
    DataTemporalResolution dataTemporalResolution = asGlobEnums::StringToDataTemporalResolutionEnum(
            dataTemporalResolutionStr);
    DataSpatialAggregation dataSpatialAggregation = asGlobEnums::StringToDataSpatialAggregationEnum(
            dataSpatialAggregationStr);

    if (dataParameter == NoDataParameter) {
        asLogError(_("The given data parameter is unknown. Cannot get an instance of the predictand DB."));
        return NULL;
    }

    if (dataTemporalResolution == NoDataTemporalResolution) {
        asLogError(_("The given data temporal resolution is unknown. Cannot get an instance of the predictand DB."));
        return NULL;
    }

    if (dataSpatialAggregation == NoDataSpatialAggregation) {
        asLogError(_("The given data spatial aggregation is unknown. Cannot get an instance of the predictand DB."));
        return NULL;
    }

    asDataPredictand *db = asDataPredictand::GetInstance(dataParameter, dataTemporalResolution, dataSpatialAggregation);
    return db;
}

asDataPredictand *asDataPredictand::GetInstance(DataParameter dataParameter,
                                                DataTemporalResolution dataTemporalResolution,
                                                DataSpatialAggregation dataSpatialAggregation)
{
    switch (dataParameter) {
        case (Precipitation): {
            asDataPredictand *db = new asDataPredictandPrecipitation(dataParameter, dataTemporalResolution,
                                                                     dataSpatialAggregation);
            return db;
        }
        case (AirTemperature): {
            asDataPredictand *db = new asDataPredictandTemperature(dataParameter, dataTemporalResolution,
                                                                   dataSpatialAggregation);
            return db;
        }
        case (Lightnings): {
            asDataPredictand *db = new asDataPredictandLightnings(dataParameter, dataTemporalResolution,
                                                                  dataSpatialAggregation);
            return db;
        }
        default:
            asLogError(_("The predictand parameter is not listed in the asDataPredictand instance factory."));
            return NULL;
    }
}

asDataPredictand *asDataPredictand::GetInstance(const wxString &filePath)
{
    // Open the NetCDF file
    asLogMessage(wxString::Format(_("Opening the file %s"), filePath));
    asFileNetcdf ncFile(filePath, asFileNetcdf::ReadOnly);
    if (!ncFile.Open()) {
        asLogError(wxString::Format(_("Couldn't open file %s"), filePath));
        return NULL;
    } else {
        asLogMessage(_("File successfully opened"));
    }

    // Check version
    float version = ncFile.GetAttFloat("version");
    if (asTools::IsNaN(version) || version <= 1.0) {
        asLogError(
                _("The predictand DB file was made with an older version of AtmoSwing that is no longer supported. Please generate the file with the actual version."));
        return NULL;
    }

    // Get basic information
    DataParameter dataParameter = (DataParameter) ncFile.GetAttInt("data_parameter");
    DataTemporalResolution dataTemporalResolution = (DataTemporalResolution) ncFile.GetAttInt(
            "data_temporal_resolution");
    DataSpatialAggregation dataSpatialAggregation = (DataSpatialAggregation) ncFile.GetAttInt(
            "data_spatial_aggregation");

    // Close the netCDF file
    ncFile.Close();

    // Get instance
    asDataPredictand *db = asDataPredictand::GetInstance(dataParameter, dataTemporalResolution, dataSpatialAggregation);
    return db;
}

wxString asDataPredictand::GetDBFilePathSaving(const wxString &destinationDir) const
{
    wxString dataParameterStr = asGlobEnums::DataParameterEnumToString(m_dataParameter);
    wxString dataTemporalResolutionStr = asGlobEnums::DataTemporalResolutionEnumToString(m_dataTemporalResolution);
    wxString dataSpatialAggregationStr = asGlobEnums::DataSpatialAggregationEnumToString(m_dataSpatialAggregation);
    wxString fileName =
            dataParameterStr + "-" + dataTemporalResolutionStr + "-" + dataSpatialAggregationStr + "-" + m_datasetId;

    wxString predictandDBFilePath = destinationDir + DS + fileName + ".nc";

    return predictandDBFilePath;
}

bool asDataPredictand::InitMembers(const wxString &catalogFilePath)
{
    // Starting and ending date of the DB, to be overwritten
    m_dateStart = asTime::GetMJD(2100, 1, 1);
    m_dateEnd = asTime::GetMJD(1800, 1, 1);

    // Get the catalog information
    asCatalogPredictands catalog(catalogFilePath);
    if (!catalog.Load())
        return false;

    // Get first and last date
    if (catalog.GetStart() < m_dateStart)
        m_dateStart = catalog.GetStart();
    if (catalog.GetEnd() > m_dateEnd)
        m_dateEnd = catalog.GetEnd();

    // Get dataset ID
    m_datasetId = catalog.GetSetId();

    // Get the number of stations
    m_stationsNb = catalog.GetStationsNb();

    // Get the timestep
    m_timeStepDays = catalog.GetTimeStepDays();

    // Get the time length
    m_timeLength = ((m_dateEnd - m_dateStart) / m_timeStepDays) + 1;

    // Get time array
    asTimeArray timeArray(m_dateStart, m_dateEnd, m_timeStepDays * 24.0, asTimeArray::Simple);
    timeArray.Init();
    m_time = timeArray.GetTimeArray();

    return true;
}

bool asDataPredictand::InitBaseContainers()
{
    if (m_stationsNb < 1) {
        asLogError(_("The stations number is inferior to 1."));
        return false;
    }
    if (m_timeLength < 1) {
        asLogError(_("The time length is inferior to 1."));
        return false;
    }
    m_stationNames.resize(m_stationsNb);
    m_stationIds.resize(m_stationsNb);
    m_stationOfficialIds.resize(m_stationsNb);
    m_stationXCoords.resize(m_stationsNb);
    m_stationYCoords.resize(m_stationsNb);
    m_stationHeights.resize(m_stationsNb);
    m_stationStarts.resize(m_stationsNb);
    m_stationEnds.resize(m_stationsNb);
    m_time.resize(m_timeLength);
    m_dataGross.resize(m_timeLength, m_stationsNb);
    m_dataGross.fill(NaNFloat);
    if (m_hasNormalizedData) {
        m_dataNormalized.resize(m_timeLength, m_stationsNb);
        m_dataNormalized.fill(NaNFloat);
    }

    return true;
}

bool asDataPredictand::LoadCommonData(asFileNetcdf &ncFile)
{
    // Check version
    float version = ncFile.GetAttFloat("version");
    if (asTools::IsNaN(version) || version <= 1.1) {
        asLogError(
                _("The predictand DB file was made with an older version of AtmoSwing that is no longer supported. Please generate the file with the actual version."));
        return false;
    }

    // Get global attributes
    m_dataParameter = (DataParameter) ncFile.GetAttInt("data_parameter");
    m_dataTemporalResolution = (DataTemporalResolution) ncFile.GetAttInt("data_temporal_resolution");
    m_dataSpatialAggregation = (DataSpatialAggregation) ncFile.GetAttInt("data_spatial_aggregation");
    m_datasetId = ncFile.GetAttString("dataset_id");

    // Get time
    m_timeLength = ncFile.GetDimLength("time");
    m_time.resize(m_timeLength);
    ncFile.GetVar("time", &m_time[0]);

    // Get stations properties
    m_stationsNb = ncFile.GetDimLength("stations");
    wxASSERT(m_stationsNb > 0);
    m_stationNames.resize(m_stationsNb);
    m_stationIds.resize(m_stationsNb);
    m_stationOfficialIds.resize(m_stationsNb);
    m_stationHeights.resize(m_stationsNb);
    m_stationXCoords.resize(m_stationsNb);
    m_stationYCoords.resize(m_stationsNb);
    m_stationStarts.resize(m_stationsNb);
    m_stationEnds.resize(m_stationsNb);

    if (version <= 1.2) {
        ncFile.GetVar("stations_name", &m_stationNames[0], m_stationsNb);
        ncFile.GetVar("stations_ids", &m_stationIds[0]);
        ncFile.GetVar("stations_height", &m_stationHeights[0]);
        ncFile.GetVar("loc_coord_u", &m_stationXCoords[0]);
        ncFile.GetVar("loc_coord_v", &m_stationYCoords[0]);
        ncFile.GetVar("start", &m_stationStarts[0]);
        ncFile.GetVar("end", &m_stationEnds[0]);
    } else if (version <= 1.3) {
        ncFile.GetVar("stations_name", &m_stationNames[0], m_stationsNb);
        ncFile.GetVar("stations_ids", &m_stationIds[0]);
        ncFile.GetVar("stations_height", &m_stationHeights[0]);
        ncFile.GetVar("loc_coord_x", &m_stationXCoords[0]);
        ncFile.GetVar("loc_coord_y", &m_stationYCoords[0]);
        ncFile.GetVar("start", &m_stationStarts[0]);
        ncFile.GetVar("end", &m_stationEnds[0]);
    } else {
        ncFile.GetVar("station_names", &m_stationNames[0], m_stationsNb);
        ncFile.GetVar("station_ids", &m_stationIds[0]);
        ncFile.GetVar("station_official_ids", &m_stationOfficialIds[0], m_stationsNb);
        ncFile.GetVar("station_heights", &m_stationHeights[0]);
        ncFile.GetVar("station_x_coords", &m_stationXCoords[0]);
        ncFile.GetVar("station_y_coords", &m_stationYCoords[0]);
        ncFile.GetVar("station_starts", &m_stationStarts[0]);
        ncFile.GetVar("station_ends", &m_stationEnds[0]);
    }

    // Get data
    size_t IndexStart[2] = {0, 0};
    size_t IndexCount[2] = {size_t(m_timeLength), size_t(m_stationsNb)};
    m_dataGross.resize(m_timeLength, m_stationsNb);

    if (asTools::IsNaN(version) || version <= 1.3) {
        ncFile.GetVarArray("data_gross", IndexStart, IndexCount, &m_dataGross(0, 0));
    } else {
        ncFile.GetVarArray("data", IndexStart, IndexCount, &m_dataGross(0, 0));
    }

    return true;
}

void asDataPredictand::SetCommonDefinitions(asFileNetcdf &ncFile) const
{
    // Define dimensions. Time is the unlimited dimension.
    ncFile.DefDim("stations", m_stationsNb);
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
    ncFile.PutAtt("version", &m_fileVersion);
    int dataParameter = (int) m_dataParameter;
    ncFile.PutAtt("data_parameter", &dataParameter);
    int dataTemporalResolution = (int) m_dataTemporalResolution;
    ncFile.PutAtt("data_temporal_resolution", &dataTemporalResolution);
    int dataSpatialAggregation = (int) m_dataSpatialAggregation;
    ncFile.PutAtt("data_spatial_aggregation", &dataSpatialAggregation);
    ncFile.PutAtt("dataset_id", m_datasetId);

    // Define variables: the scores and the corresponding dates
    ncFile.DefVar("time", NC_DOUBLE, 1, DimNameTime);
    ncFile.DefVar("data", NC_FLOAT, 2, DimNames2D);
    ncFile.DefVarDeflate("data");
    ncFile.DefVar("station_names", NC_STRING, 1, DimNameStations);
    ncFile.DefVar("station_official_ids", NC_STRING, 1, DimNameStations);
    ncFile.DefVar("station_ids", NC_INT, 1, DimNameStations);
    ncFile.DefVar("station_heights", NC_FLOAT, 1, DimNameStations);
    ncFile.DefVar("station_x_coords", NC_DOUBLE, 1, DimNameStations);
    ncFile.DefVar("station_y_coords", NC_DOUBLE, 1, DimNameStations);
    ncFile.DefVar("station_starts", NC_DOUBLE, 1, DimNameStations);
    ncFile.DefVar("station_ends", NC_DOUBLE, 1, DimNameStations);

    // Put attributes for station_names
    ncFile.PutAtt("long_name", "Stations names", "station_names");
    ncFile.PutAtt("var_desc", "Name of the predictand stations", "station_names");

    // Put attributes for station_ids
    ncFile.PutAtt("long_name", "Stations IDs", "station_ids");
    ncFile.PutAtt("var_desc", "Internal IDs of the predictand stations", "station_ids");

    // Put attributes for station_official_ids
    ncFile.PutAtt("long_name", "Stations official IDs", "station_official_ids");
    ncFile.PutAtt("var_desc", "Official IDs of the predictand stations", "station_official_ids");

    // Put attributes for station_heights
    ncFile.PutAtt("long_name", "Stations height", "station_heights");
    ncFile.PutAtt("var_desc", "Altitude of the predictand stations", "station_heights");
    ncFile.PutAtt("units", "m", "station_heights");

    // Put attributes for station_x_coords
    ncFile.PutAtt("long_name", "X coordinate", "station_x_coords");
    ncFile.PutAtt("var_desc", "X coordinate", "station_x_coords");

    // Put attributes for station_y_coords
    ncFile.PutAtt("long_name", "Y coordinate", "station_y_coords");
    ncFile.PutAtt("var_desc", "Y coordinate", "station_y_coords");

    // Put attributes for station_starts
    ncFile.PutAtt("long_name", "Start", "station_starts");
    ncFile.PutAtt("var_desc", "Start of the stations data", "station_starts");
    ncFile.PutAtt("units", "Modified Julian Day Number (MJD)", "station_starts");

    // Put attributes for station_ends
    ncFile.PutAtt("long_name", "End", "station_ends");
    ncFile.PutAtt("var_desc", "End of the stations data", "station_ends");
    ncFile.PutAtt("units", "Modified Julian Day Number (MJD)", "station_ends");

    // Put attributes for data
    ncFile.PutAtt("long_name", "Data", "data");
    ncFile.PutAtt("var_desc", "Data (whithout any treatment)", "data");

}

bool asDataPredictand::SaveCommonData(asFileNetcdf &ncFile) const
{
    // Provide sizes for variables
    size_t startTime[] = {0};
    size_t countTime[] = {size_t(m_timeLength)};
    size_t startStations[] = {0};
    size_t countStations[] = {size_t(m_stationsNb)};
    size_t start2[] = {0, 0};
    size_t count2[] = {size_t(m_timeLength), size_t(m_stationsNb)};

    // Write data
    ncFile.PutVarArray("time", startTime, countTime, &m_time(0));
    ncFile.PutVarArray("station_names", startStations, countStations, &m_stationNames[0], m_stationNames.size());
    ncFile.PutVarArray("station_official_ids", startStations, countStations, &m_stationOfficialIds[0],
                       m_stationOfficialIds.size());
    ncFile.PutVarArray("station_ids", startStations, countStations, &m_stationIds(0));
    ncFile.PutVarArray("station_heights", startStations, countStations, &m_stationHeights(0));
    ncFile.PutVarArray("station_x_coords", startStations, countStations, &m_stationXCoords(0));
    ncFile.PutVarArray("station_y_coords", startStations, countStations, &m_stationYCoords(0));
    ncFile.PutVarArray("station_starts", startStations, countStations, &m_stationStarts(0));
    ncFile.PutVarArray("station_ends", startStations, countStations, &m_stationEnds(0));
    ncFile.PutVarArray("data", start2, count2, &m_dataGross(0, 0));

    return true;
}

bool asDataPredictand::SetStationProperties(asCatalogPredictands &currentData, size_t stationIndex)
{
    m_stationNames[stationIndex] = currentData.GetStationName(stationIndex);
    m_stationIds(stationIndex) = currentData.GetStationId(stationIndex);
    m_stationOfficialIds[stationIndex] = currentData.GetStationOfficialId(stationIndex);
    m_stationXCoords(stationIndex) = currentData.GetStationCoord(stationIndex).x;
    m_stationYCoords(stationIndex) = currentData.GetStationCoord(stationIndex).y;
    m_stationHeights(stationIndex) = currentData.GetStationHeight(stationIndex);
    m_stationStarts(stationIndex) = currentData.GetStationStart(stationIndex);
    m_stationEnds(stationIndex) = currentData.GetStationEnd(stationIndex);

    return true;
}

bool asDataPredictand::ParseData(const wxString &catalogFilePath, const wxString &AlternateDataDir,
                                 const wxString &AlternatePatternDir)
{
#if wxUSE_GUI
    // The progress bar
    asDialogProgressBar ProgressBar(_("Loading data from files.\n"), m_stationsNb);
#endif

    // Get catalog
    asCatalogPredictands catalog(catalogFilePath);
    catalog.Load();

    // Get the stations list
    for (int i_station = 0; i_station < catalog.GetStationsNb(); i_station++) {
#if wxUSE_GUI
        // Update the progress bar.
        wxString fileNameMessage = wxString::Format(_("Loading data from files.\nFile: %s"),
                                                    catalog.GetStationFilename(i_station));
        if (!ProgressBar.Update(i_station, fileNameMessage)) {
            asLogError(_("The process has been canceled by the user."));
            return false;
        }
#endif

        // Get station information
        if (!SetStationProperties(catalog, i_station))
            return false;

        // Get file content
        if (!GetFileContent(catalog, i_station, AlternateDataDir, AlternatePatternDir))
            return false;
    }

#if wxUSE_GUI
    ProgressBar.Destroy();
#endif

    return true;
}

bool asDataPredictand::GetFileContent(asCatalogPredictands &currentData, size_t stationIndex,
                                      const wxString &AlternateDataDir, const wxString &AlternatePatternDir)
{
    // Load file
    wxString fileFullPath;
    if (!AlternateDataDir.IsEmpty()) {
        fileFullPath = AlternateDataDir + DS + currentData.GetStationFilename(stationIndex);
    } else {
        fileFullPath = currentData.GetDataPath() + currentData.GetStationFilename(stationIndex);
    }
    asFileDat datFile(fileFullPath, asFile::ReadOnly);
    if (!datFile.Open())
        return false;

    // Get the parsing format
    wxString stationFilePattern = currentData.GetStationFilepattern(stationIndex);
    asFileDat::Pattern filePattern = asFileDat::GetPattern(stationFilePattern, AlternatePatternDir);
    size_t maxCharWidth = asFileDat::GetPatternLineMaxCharWidth(filePattern);

    // Jump the header
    datFile.SkipLines(filePattern.HeaderLines);

    // Get first index on the tima axis
    int startIndex = asTools::SortedArraySearch(&m_time[0], &m_time[m_time.size() - 1],
                                                currentData.GetStationStart(stationIndex));
    if (startIndex == asOUT_OF_RANGE || startIndex == asNOT_FOUND) {
        asLogError(wxString::Format(_("The given start date for \"%s\" is out of the catalog range."),
                                    currentData.GetStationName(stationIndex)));
        return false;
    }

    int timeIndex = startIndex;

    // Parse every line until the end of the file
    while (!datFile.EndOfFile()) {
        // Get current line
        wxString lineContent = datFile.GetLineContent();

        // Check the line width
        if (lineContent.Len() >= maxCharWidth) {
            // Check the size of the array
            if (timeIndex >= m_timeLength) {
                asLogError(wxString::Format(
                        _("The time index is larger than the matrix (timeIndex = %d, m_timeLength = %d)."),
                        (int) timeIndex, (int) m_timeLength));
                return false;
            }

            switch (filePattern.StructType) {
                case (asFileDat::ConstantWidth): {
                    if (filePattern.ParseTime) {
                        // Containers. Must be a double to use wxString::ToDouble
                        double valTimeYear = 0, valTimeMonth = 0, valTimeDay = 0, valTimeHour = 0, valTimeMinute = 0;

                        // Get time value
                        if (filePattern.TimeYearBegin != 0 && filePattern.TimeYearEnd != 0 &&
                            filePattern.TimeMonthBegin != 0 && filePattern.TimeMonthEnd != 0 &&
                            filePattern.TimeDayBegin != 0 && filePattern.TimeDayEnd != 0) {
                            lineContent.Mid(filePattern.TimeYearBegin - 1,
                                            filePattern.TimeYearEnd - filePattern.TimeYearBegin + 1).ToDouble(
                                    &valTimeYear);
                            lineContent.Mid(filePattern.TimeMonthBegin - 1,
                                            filePattern.TimeMonthEnd - filePattern.TimeMonthBegin + 1).ToDouble(
                                    &valTimeMonth);
                            lineContent.Mid(filePattern.TimeDayBegin - 1,
                                            filePattern.TimeDayEnd - filePattern.TimeDayBegin + 1).ToDouble(
                                    &valTimeDay);
                        } else {
                            asLogError(_("The data file pattern is not correctly defined."));
                            return false;
                        }

                        if (filePattern.TimeHourBegin != 0 && filePattern.TimeHourEnd != 0) {
                            lineContent.Mid(filePattern.TimeHourBegin - 1,
                                            filePattern.TimeHourEnd - filePattern.TimeHourBegin + 1).ToDouble(
                                    &valTimeHour);
                        }
                        if (filePattern.TimeMinuteBegin != 0 && filePattern.TimeMinuteEnd != 0) {
                            lineContent.Mid(filePattern.TimeMinuteBegin - 1,
                                            filePattern.TimeMinuteEnd - filePattern.TimeMinuteBegin + 1).ToDouble(
                                    &valTimeMinute);
                        }

                        double dateData = asTime::GetMJD(valTimeYear, valTimeMonth, valTimeDay, valTimeHour,
                                                         valTimeMinute, 0);

                        // Check again date vector
                        if (std::abs(dateData - m_time(timeIndex)) > 0.0001) {
                            wxString errorMessage = wxString::Format(
                                    _("Value in data : %6.4f (%s), value in time array : %6.4f (%s). In file %s"),
                                    dateData, asTime::GetStringTime(dateData, "DD.MM.YYYY"), m_time(timeIndex),
                                    asTime::GetStringTime(m_time(timeIndex), "DD.MM.YYYY"),
                                    currentData.GetStationFilename(stationIndex));
                            asLogError(wxString::Format(_("The time value doesn't match: %s"), errorMessage));
                            return false;
                        }
                    }

                    // Get Precipitation value
                    wxString dataStr = lineContent.Mid(filePattern.DataBegin - 1,
                                                       filePattern.DataEnd - filePattern.DataBegin + 1);

                    // Put value in the matrix
                    m_dataGross(timeIndex, stationIndex) = ParseAndCheckDataValue(currentData, dataStr);

                    timeIndex++;
                    break;
                }

                case (asFileDat::TabsDelimited): {
                    // Parse into a vector
                    VectorString vColumns;
                    wxString tmpLineContent = lineContent;
                    while (tmpLineContent.Find("\t") != wxNOT_FOUND) {
                        int foundCol = tmpLineContent.Find("\t");
                        vColumns.push_back(tmpLineContent.Mid(0, foundCol));
                        tmpLineContent = tmpLineContent.Mid(foundCol + 1);
                    }
                    if (!tmpLineContent.IsEmpty()) {
                        vColumns.push_back(tmpLineContent);
                    }

                    if (filePattern.ParseTime) {
                        // Containers. Must be a double to use wxString::ToDouble
                        double valTimeYear = 0, valTimeMonth = 0, valTimeDay = 0, valTimeHour = 0, valTimeMinute = 0;

                        // Get time value
                        if (filePattern.TimeYearBegin != 0 && filePattern.TimeMonthBegin != 0 &&
                            filePattern.TimeDayBegin != 0) {
                            if ((unsigned) filePattern.TimeYearBegin > vColumns.size() ||
                                (unsigned) filePattern.TimeMonthBegin > vColumns.size() ||
                                (unsigned) filePattern.TimeDayBegin > vColumns.size()) {
                                asLogError(
                                        _("The data file pattern is not correctly defined. Trying to access an element (date) after the line width."));
                                return false;
                            }
                            vColumns[filePattern.TimeYearBegin - 1].ToDouble(&valTimeYear);
                            vColumns[filePattern.TimeMonthBegin - 1].ToDouble(&valTimeMonth);
                            vColumns[filePattern.TimeDayBegin - 1].ToDouble(&valTimeDay);
                        } else {
                            asLogError(_("The data file pattern is not correctly defined."));
                            return false;
                        }

                        if (filePattern.TimeHourBegin != 0) {
                            if ((unsigned) filePattern.TimeHourBegin > vColumns.size()) {
                                asLogError(
                                        _("The data file pattern is not correctly defined. Trying to access an element (hour) after the line width."));
                                return false;
                            }
                            vColumns[filePattern.TimeHourBegin - 1].ToDouble(&valTimeHour);
                        }
                        if (filePattern.TimeMinuteBegin != 0) {
                            if ((unsigned) filePattern.TimeMinuteBegin > vColumns.size()) {
                                asLogError(
                                        _("The data file pattern is not correctly defined. Trying to access an element (minute) after the line width."));
                                return false;
                            }
                            vColumns[filePattern.TimeMinuteBegin - 1].ToDouble(&valTimeMinute);
                        }

                        double dateData = asTime::GetMJD(valTimeYear, valTimeMonth, valTimeDay, valTimeHour,
                                                         valTimeMinute, 0);

                        // Check again date vector
                        if (std::abs(dateData - m_time(timeIndex)) > 0.001) {
                            wxString errorMessage = wxString::Format(
                                    _("Value in data : %6.4f (%s), value in time array : %6.4f (%s). In file %s"),
                                    dateData, asTime::GetStringTime(dateData, "DD.MM.YYYY"), m_time(timeIndex),
                                    asTime::GetStringTime(m_time(timeIndex), "DD.MM.YYYY"),
                                    currentData.GetStationFilename(stationIndex));
                            asLogError(wxString::Format(_("The time value doesn't match: %s"), errorMessage));
                            return false;
                        }
                    }

                    // Get Precipitation value
                    wxString dataStr = vColumns[filePattern.DataBegin - 1];

                    // Put value in the matrix
                    m_dataGross(timeIndex, stationIndex) = ParseAndCheckDataValue(currentData, dataStr);

                    timeIndex++;
                    break;
                }
            }
        } else {
            if (lineContent.Len() > 1) {
                asLogError(_("The line length doesn't match."));
                return false;
            }
        }
    }
    datFile.Close();

    // Get end index
    int endIndex = asTools::SortedArraySearch(&m_time[0], &m_time[m_time.size() - 1],
                                              currentData.GetStationEnd(stationIndex));
    if (endIndex == asOUT_OF_RANGE || endIndex == asNOT_FOUND) {
        asLogError(wxString::Format(_("The given end date for \"%s\" is out of the catalog range."),
                                    currentData.GetStationName(stationIndex)));
        return false;
    }

    // Check time width
    if (endIndex - startIndex != timeIndex - startIndex - 1) {
        wxString messageTime = wxString::Format(_("The length of the data in \"%s / %s\" is not coherent"),
                                                currentData.GetName(), currentData.GetStationName(stationIndex));
        asLogError(messageTime);
        return false;
    }

    return true;
}

float asDataPredictand::ParseAndCheckDataValue(asCatalogPredictands &currentData, wxString &dataStr) const
{
    // Trim
    dataStr = dataStr.Trim();
    dataStr = dataStr.Trim(true);

    // Check if not NaN
    for (size_t i_nan = 0; i_nan < currentData.GetNan().size(); i_nan++) {
        if (dataStr.IsSameAs(currentData.GetNan()[i_nan], false)) {
            return NaNFloat;
        }
    }

    // Convert
    double dataGross = 0;
    dataStr.ToDouble(&dataGross);

    return (float) dataGross;
}

Array2DFloat asDataPredictand::GetAnnualMax(double timeStepDays, int nansNbMax) const
{
    // Flag to check the need of aggregation (timeStepDays>m_timeStepDays)
    bool aggregate = false;
    int indexTimeSpanUp = 0;
    int indexTimeSpanDown = 0;

    if (timeStepDays == m_timeStepDays) {
        aggregate = false;
    } else if (timeStepDays > m_timeStepDays) {
        if (std::fmod(timeStepDays, m_timeStepDays) > 0.0000001) {
            asLogError(
                    _("The timestep for the extraction of the predictands maximums has to be a multiple of the data timestep."));
            Array2DFloat emptyMatrix;
            emptyMatrix << NaNFloat;
            return emptyMatrix;
        }

        // Aggregation necessary
        aggregate = true;

        // indices to add or substract around the mid value
        indexTimeSpanUp = floor((timeStepDays / m_timeStepDays) / 2);
        indexTimeSpanDown = ceil((timeStepDays / m_timeStepDays) / 2) - 1;
    } else {
        asLogError(
                _("The timestep for the extraction of the predictands maximums cannot be lower than the data timestep."));
        Array2DFloat emptyMatrix;
        emptyMatrix << NaNFloat;
        return emptyMatrix;
    }

    // Keep the real indices of years
    int indYearStart = 0;
    int indYearEnd = 0;

    // Get catalog beginning and end
    int yearStart = asTime::GetYear(m_dateStart);
    if (asTime::GetMonth(m_dateStart) != 1 || asTime::GetDay(m_dateStart) != 1) {
        yearStart++;
        indYearStart++;
    }
    int yearEnd = asTime::GetYear(m_dateEnd);
    indYearEnd = yearEnd - yearStart + indYearStart;
    if (asTime::GetMonth(m_dateEnd) != 12 || asTime::GetDay(m_dateEnd) != 31) {
        yearEnd--;
    }

    // Create the container
    Array2DFloat maxMatrix = Array2DFloat::Constant(m_stationsNb, indYearEnd + 1, NaNFloat);

    // Look for maximums
    for (int i_stnb = 0; i_stnb < m_stationsNb; i_stnb++) {
        for (int i_year = yearStart; i_year <= yearEnd; i_year++) {
            // The maximum value and a flag for accepted NaNs
            float annualmax = -99999;
            int nansNb = 0;

            // Find begining and end of the year
            int rowstart = asTools::SortedArraySearchFloor(&m_time[0], &m_time[m_timeLength - 1],
                                                           asTime::GetMJD(i_year, 1, 1), asHIDE_WARNINGS);
            int rowend = asTools::SortedArraySearchFloor(&m_time[0], &m_time[m_timeLength - 1],
                                                         asTime::GetMJD(i_year, 12, 31, 59, 59), asHIDE_WARNINGS);
            if ((rowend == asOUT_OF_RANGE) | (rowend == asNOT_FOUND)) {
                if (i_year == yearEnd) {
                    rowend = m_timeLength - 1;
                } else {
                    annualmax = NaNFloat;
                }
            }
            rowend -= 1;

            // Get max
            if (!aggregate) {
                for (int i_row = rowstart; i_row <= rowend; i_row++) {
                    if (!asTools::IsNaN(m_dataGross(i_row, i_stnb))) {
                        annualmax = wxMax(m_dataGross(i_row, i_stnb), annualmax);
                    } else {
                        nansNb++;
                    }
                }
                if (nansNb > nansNbMax) {
                    annualmax = NaNFloat;
                }
            } else {
                // Correction for both extremes
                rowstart = wxMax(rowstart - indexTimeSpanDown, 0);
                rowstart += indexTimeSpanDown;
                rowend = wxMin(rowend + indexTimeSpanUp, (int) m_dataGross.rows() - 1);
                rowend -= indexTimeSpanUp;

                // Loop within the new limits
                for (int i_row = rowstart; i_row <= rowend; i_row++) {
                    float timeStepSum = 0;
                    for (int i_element = i_row - indexTimeSpanDown; i_element <= i_row + indexTimeSpanUp; i_element++) {
                        if (!asTools::IsNaN(m_dataGross(i_element, i_stnb))) {
                            timeStepSum += m_dataGross(i_element, i_stnb);
                        } else {
                            timeStepSum = NaNFloat;
                            break;
                        }
                    }

                    if (!asTools::IsNaN(timeStepSum)) {
                        annualmax = wxMax(timeStepSum, annualmax);
                    } else {
                        nansNb++;
                    }
                }
                if (nansNb > nansNbMax) {
                    annualmax = NaNFloat;
                }
            }

            maxMatrix(i_stnb, i_year - yearStart + indYearStart) = annualmax;
        }
    }

    return maxMatrix;
}

int asDataPredictand::GetStationIndex(int stationId) const
{
    return asTools::SortedArraySearch(&m_stationIds[0], &m_stationIds[m_stationsNb - 1], stationId);
}
