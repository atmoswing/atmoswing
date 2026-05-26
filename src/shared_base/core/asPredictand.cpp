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

#include "asPredictand.h"
#include "asIncludes.h"

#include "asCatalogPredictands.h"
#include "asFileDat.h"
#include "asFileNetcdf.h"
#include "asPredictandLightning.h"
#include "asPredictandPrecipitation.h"
#include "asPredictandTemperature.h"
#include "asTimeArray.h"

asPredictand::asPredictand(Parameter dataParameter, TemporalResolution dataTemporalResolution,
                           SpatialAggregation dataSpatialAggregation)
    : _fileVersion(1.4f),
      _parameter(dataParameter),
      _temporalResolution(dataTemporalResolution),
      _spatialAggregation(dataSpatialAggregation),
      _timeStepDays(0.0),
      _timeLength(0),
      _stationsNb(0),
      _dateProcessed(0.0),
      _dateStart(0.0),
      _dateEnd(0.0),
      _hasNormalizedData(false),
      _hasReferenceValues(false) {}

asPredictand::Parameter asPredictand::StringToParameterEnum(const wxString& parameterStr) {
    if (parameterStr.CmpNoCase("Precipitation") == 0) {
        return Precipitation;
    } else if (parameterStr.CmpNoCase("AirTemperature") == 0) {
        return AirTemperature;
    } else if (parameterStr.CmpNoCase("Lightning") == 0 || parameterStr.CmpNoCase("Lightnings") == 0) {
        return Lightning;
    } else if (parameterStr.CmpNoCase("Wind") == 0) {
        return Wind;
    } else {
        wxLogError(_("The Parameter enumeration (%s) entry doesn't exists"), parameterStr);
    }
    return Precipitation;
}

wxString asPredictand::ParameterEnumToString(asPredictand::Parameter parameter) {
    switch (parameter) {
        case (Precipitation):
            return "Precipitation";
        case (AirTemperature):
            return "AirTemperature";
        case (Lightning):
            return "Lightning";
        case (Wind):
            return "Wind";
        default:
            wxLogError(_("The given data parameter type in unknown."));
    }
    return wxEmptyString;
}

asPredictand::Unit asPredictand::StringToUnitEnum(const wxString& unitStr) {
    if (unitStr.CmpNoCase("nb") == 0 || unitStr.CmpNoCase("number") == 0) {
        return nb;
    } else if (unitStr.CmpNoCase("mm") == 0) {
        return mm;
    } else if (unitStr.CmpNoCase("m") == 0) {
        return m;
    } else if (unitStr.CmpNoCase("inches") == 0 || unitStr.CmpNoCase("in")) {
        return in;
    } else if (unitStr.CmpNoCase("percent") == 0 || unitStr.CmpNoCase("%") == 0) {
        return percent;
    } else if (unitStr.CmpNoCase("degC") == 0) {
        return degC;
    } else if (unitStr.CmpNoCase("degK") == 0) {
        return degK;
    } else {
        wxLogError(_("The Unit enumeration (%s) entry doesn't exists"), unitStr);
    }
    return mm;
}

asPredictand::TemporalResolution asPredictand::StringToTemporalResolutionEnum(const wxString& temporalResolution) {
    if (temporalResolution.CmpNoCase("Daily") == 0) {
        return Daily;
    } else if (temporalResolution.CmpNoCase("1 day") == 0) {
        return Daily;
    } else if (temporalResolution.CmpNoCase("SixHourly") == 0) {
        return SixHourly;
    } else if (temporalResolution.CmpNoCase("6 hours") == 0) {
        return SixHourly;
    } else if (temporalResolution.CmpNoCase("Hourly") == 0) {
        return Hourly;
    } else if (temporalResolution.CmpNoCase("1 hour") == 0) {
        return Hourly;
    } else if (temporalResolution.CmpNoCase("OneHourlyMTW") == 0) {
        return OneHourlyMTW;
    } else if (temporalResolution.CmpNoCase("ThreeHourlyMTW") == 0) {
        return ThreeHourlyMTW;
    } else if (temporalResolution.CmpNoCase("SixHourlyMTW") == 0) {
        return SixHourlyMTW;
    } else if (temporalResolution.CmpNoCase("TwelveHourlyMTW") == 0) {
        return TwelveHourlyMTW;
    } else {
        wxLogError(_("The temporalResolution enumeration (%s) entry doesn't exists"), temporalResolution);
    }
    return Daily;
}

wxString asPredictand::TemporalResolutionEnumToString(asPredictand::TemporalResolution temporalResolution) {
    switch (temporalResolution) {
        case (Daily):
            return "Daily";
        case (SixHourly):
            return "SixHourly";
        case (Hourly):
            return "Hourly";
        case (OneHourlyMTW):
            return "OneHourlyMTW";
        case (ThreeHourlyMTW):
            return "ThreeHourlyMTW";
        case (SixHourlyMTW):
            return "SixHourlyMTW";
        case (TwelveHourlyMTW):
            return "TwelveHourlyMTW";
        default:
            wxLogError(_("The given data temporal resolution type in unknown."));
    }
    return wxEmptyString;
}

asPredictand::SpatialAggregation asPredictand::StringToSpatialAggregationEnum(const wxString& spatialAggregation) {
    if (spatialAggregation.CmpNoCase("Station") == 0) {
        return Station;
    } else if (spatialAggregation.CmpNoCase("Groupment") == 0) {
        return Groupment;
    } else if (spatialAggregation.CmpNoCase("Catchment") == 0 || spatialAggregation.CmpNoCase("Basin") == 0) {
        return Catchment;
    } else if (spatialAggregation.CmpNoCase("Region") == 0) {
        return Region;
    } else {
        wxLogError(_("The spatialAggregation enumeration (%s) entry doesn't exists"), spatialAggregation);
    }
    return Station;
}

wxString asPredictand::SpatialAggregationEnumToString(asPredictand::SpatialAggregation spatialAggregation) {
    switch (spatialAggregation) {
        case (Station):
            return "Station";
        case (Groupment):
            return "Groupment";
        case (Catchment):
            return "Catchment";
        case (Region):
            return "Region";
        default:
            wxLogError(_("The given data spatial aggregation type in unknown."));
    }
    return wxEmptyString;
}

asPredictand* asPredictand::GetInstance(const wxString& parameterStr, const wxString& temporalResolutionStr,
                                        const wxString& spatialAggregationStr) {
    Parameter parameter = StringToParameterEnum(parameterStr);
    TemporalResolution temporalResolution = StringToTemporalResolutionEnum(temporalResolutionStr);
    SpatialAggregation spatialAggregation = StringToSpatialAggregationEnum(spatialAggregationStr);

    asPredictand* db = asPredictand::GetInstance(parameter, temporalResolution, spatialAggregation);
    return db;
}

asPredictand* asPredictand::GetInstance(Parameter parameter, TemporalResolution temporalResolution,
                                        SpatialAggregation spatialAggregation) {
    switch (parameter) {
        case (Precipitation): {
            asPredictand* db = new asPredictandPrecipitation(parameter, temporalResolution, spatialAggregation);
            return db;
        }
        case (AirTemperature): {
            asPredictand* db = new asPredictandTemperature(parameter, temporalResolution, spatialAggregation);
            return db;
        }
        case (Lightning): {
            asPredictand* db = new asPredictandLightning(parameter, temporalResolution, spatialAggregation);
            return db;
        }
        default:
            wxLogError(_("The predictand parameter is not listed in the asPredictand instance factory."));
            return nullptr;
    }
}

asPredictand* asPredictand::GetInstance(const wxString& filePath) {
    // Open the NetCDF file
    wxLogVerbose(_("Opening the file %s"), filePath);
    asFileNetcdf ncFile(filePath, asFileNetcdf::ReadOnly);
    if (!ncFile.Open()) {
        wxLogError(_("Couldn't open file %s"), filePath);
        return nullptr;
    }

    // Check version
    float version = ncFile.GetAttFloat("version");
    if (isnan(version) || version <= 1.0) {
        wxLogError(
            _("The predictand DB file was made with an older version of AtmoSwing that is no longer supported. Please "
              "generate the file with the actual version."));
        return nullptr;
    }

    // Get basic information
    Parameter dataParameter = (Parameter)ncFile.GetAttInt("data_parameter");
    TemporalResolution dataTemporalResolution = (TemporalResolution)ncFile.GetAttInt("data_temporal_resolution");
    SpatialAggregation dataSpatialAggregation = (SpatialAggregation)ncFile.GetAttInt("data_spatial_aggregation");

    // Close the netCDF file
    if (!ncFile.Close()) {
        wxLogError(_("Couldn't close file %s"), filePath);
        return nullptr;
    }

    // Get instance
    asPredictand* db = asPredictand::GetInstance(dataParameter, dataTemporalResolution, dataSpatialAggregation);
    return db;
}

wxString asPredictand::GetDBFilePathSaving(const wxString& destinationDir) const {
    wxString parameter = ParameterEnumToString(_parameter);
    wxString temporalResolution = asPredictand::TemporalResolutionEnumToString(_temporalResolution);
    wxString spatialAggregation = asPredictand::SpatialAggregationEnumToString(_spatialAggregation);
    wxString fileName = parameter + "-" + temporalResolution + "-" + spatialAggregation + "-" + _datasetId;

    wxString predictandDBFilePath = destinationDir + DS + fileName + ".nc";

    return predictandDBFilePath;
}

bool asPredictand::InitMembers(const wxString& catalogFilePath) {
    // Starting and ending date of the DB, to be overwritten
    _dateStart = asTime::GetMJD(2100, 1, 1);
    _dateEnd = asTime::GetMJD(1800, 1, 1);

    // Get the catalog information
    asCatalogPredictands catalog(catalogFilePath);
    if (!catalog.Load()) return false;

    // Get first and last date
    if (catalog.GetStart() < _dateStart) _dateStart = catalog.GetStart();
    if (catalog.GetEnd() > _dateEnd) _dateEnd = catalog.GetEnd();

    // Get other catalog data
    _datasetId = catalog.GetSetId();
    _coordSys = catalog.GetCoordSys();
    _stationsNb = catalog.GetStationsNb();
    _timeStepDays = catalog.GetTimeStepDays();

    // Get the time length
    _timeLength = ((_dateEnd - _dateStart) / _timeStepDays) + 1;

    // Get time array
    asTimeArray timeArray(_dateStart, _dateEnd, _timeStepDays * 24.0, asTimeArray::Simple);
    timeArray.Init();
    _time = timeArray.GetTimeArray();

    return true;
}

bool asPredictand::InitBaseContainers() {
    if (_stationsNb < 1) {
        wxLogError(_("The stations number is inferior to 1."));
        return false;
    }
    if (_timeLength < 1) {
        wxLogError(_("The time length is inferior to 1."));
        return false;
    }
    _stationNames.resize(_stationsNb);
    _stationIds.resize(_stationsNb);
    _stationOfficialIds.resize(_stationsNb);
    _stationXCoords.resize(_stationsNb);
    _stationYCoords.resize(_stationsNb);
    _stationHeights.resize(_stationsNb);
    _stationStarts.resize(_stationsNb);
    _stationEnds.resize(_stationsNb);
    _time.resize(_timeLength);
    _dataRaw.resize(_timeLength, _stationsNb);
    _dataRaw.fill(NAN);
    if (_hasNormalizedData) {
        _dataNormalized.resize(_timeLength, _stationsNb);
        _dataNormalized.fill(NAN);
    }

    return true;
}

bool asPredictand::LoadCommonData(asFileNetcdf& ncFile) {
    // Check version
    float version = ncFile.GetAttFloat("version");
    if (isnan(version) || version <= 1.1) {
        wxLogError(
            _("The predictand DB file was made with an older version of AtmoSwing that is no longer supported. Please "
              "generate the file with the actual version."));
        return false;
    }

    // Get global attributes
    _parameter = (Parameter)ncFile.GetAttInt("data_parameter");
    _temporalResolution = (TemporalResolution)ncFile.GetAttInt("data_temporal_resolution");
    _spatialAggregation = (SpatialAggregation)ncFile.GetAttInt("data_spatial_aggregation");
    _datasetId = ncFile.GetAttString("dataset_id");
    _hasNormalizedData = ncFile.HasVariable("data_normalized");
    _hasReferenceValues = _hasNormalizedData;
    if (ncFile.HasAttribute("coordinate_system")) {
        _coordSys = ncFile.GetAttString("coordinate_system");
    }

    // Get time
    _timeLength = ncFile.GetDimLength("time");
    _time.resize(_timeLength);
    ncFile.GetVar("time", &_time[0]);

    // Get stations properties
    _stationsNb = ncFile.GetDimLength("stations");
    wxASSERT(_stationsNb > 0);
    _stationNames.resize(_stationsNb);
    _stationIds.resize(_stationsNb);
    _stationOfficialIds.resize(_stationsNb);
    _stationHeights.resize(_stationsNb);
    _stationXCoords.resize(_stationsNb);
    _stationYCoords.resize(_stationsNb);
    _stationStarts.resize(_stationsNb);
    _stationEnds.resize(_stationsNb);

    if (version <= 1.2) {
        ncFile.GetVar("stations_name", &_stationNames[0], _stationsNb);
        ncFile.GetVar("stations_ids", &_stationIds[0]);
        ncFile.GetVar("stations_height", &_stationHeights[0]);
        ncFile.GetVar("loc_coord_u", &_stationXCoords[0]);
        ncFile.GetVar("loc_coord_v", &_stationYCoords[0]);
        ncFile.GetVar("start", &_stationStarts[0]);
        ncFile.GetVar("end", &_stationEnds[0]);
    } else if (version <= 1.3) {
        ncFile.GetVar("stations_name", &_stationNames[0], _stationsNb);
        ncFile.GetVar("stations_ids", &_stationIds[0]);
        ncFile.GetVar("stations_height", &_stationHeights[0]);
        ncFile.GetVar("loc_coord_x", &_stationXCoords[0]);
        ncFile.GetVar("loc_coord_y", &_stationYCoords[0]);
        ncFile.GetVar("start", &_stationStarts[0]);
        ncFile.GetVar("end", &_stationEnds[0]);
    } else {
        ncFile.GetVar("station_names", &_stationNames[0], _stationsNb);
        ncFile.GetVar("station_ids", &_stationIds[0]);
        ncFile.GetVar("station_official_ids", &_stationOfficialIds[0], _stationsNb);
        ncFile.GetVar("station_heights", &_stationHeights[0]);
        ncFile.GetVar("station_x_coords", &_stationXCoords[0]);
        ncFile.GetVar("station_y_coords", &_stationYCoords[0]);
        ncFile.GetVar("station_starts", &_stationStarts[0]);
        ncFile.GetVar("station_ends", &_stationEnds[0]);
    }

    // Get data
    size_t indexStart[2] = {0, 0};
    size_t indexCount[2] = {size_t(_timeLength), size_t(_stationsNb)};
    _dataRaw.resize(_timeLength, _stationsNb);

    if (isnan(version) || version <= 1.3) {
        ncFile.GetVarArray("data_gross", indexStart, indexCount, &_dataRaw(0, 0));
    } else {
        ncFile.GetVarArray("data", indexStart, indexCount, &_dataRaw(0, 0));
    }

    return true;
}

void asPredictand::SetCommonDefinitions(asFileNetcdf& ncFile) const {
    // Define dimensions. Time is the unlimited dimension.
    ncFile.DefDim("stations", _stationsNb);
    ncFile.DefDim("time");

    // The dimensions name array is used to pass the dimensions to the variable.
    vstds dimNameTime;
    dimNameTime.push_back("time");
    vstds dimNameStations;
    dimNameStations.push_back("stations");
    vstds dimNames2D;
    dimNames2D.push_back("time");
    dimNames2D.push_back("stations");

    // Put general attributes
    ncFile.PutAtt("version", &_fileVersion);
    auto dataParameter = (int)_parameter;
    ncFile.PutAtt("data_parameter", &dataParameter);
    auto dataTemporalResolution = (int)_temporalResolution;
    ncFile.PutAtt("data_temporal_resolution", &dataTemporalResolution);
    auto dataSpatialAggregation = (int)_spatialAggregation;
    ncFile.PutAtt("data_spatial_aggregation", &dataSpatialAggregation);
    ncFile.PutAtt("dataset_id", _datasetId);
    ncFile.PutAtt("coordinate_system", _coordSys);

    // Define variables: the scores and the corresponding dates
    ncFile.DefVar("time", NC_DOUBLE, 1, dimNameTime);
    ncFile.DefVar("data", NC_FLOAT, 2, dimNames2D);
    ncFile.DefVarDeflate("data");
    ncFile.DefVar("station_names", NC_STRING, 1, dimNameStations);
    ncFile.DefVar("station_official_ids", NC_STRING, 1, dimNameStations);
    ncFile.DefVar("station_ids", NC_INT, 1, dimNameStations);
    ncFile.DefVar("station_heights", NC_FLOAT, 1, dimNameStations);
    ncFile.DefVar("station_x_coords", NC_DOUBLE, 1, dimNameStations);
    ncFile.DefVar("station_y_coords", NC_DOUBLE, 1, dimNameStations);
    ncFile.DefVar("station_starts", NC_DOUBLE, 1, dimNameStations);
    ncFile.DefVar("station_ends", NC_DOUBLE, 1, dimNameStations);

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
    ncFile.PutAtt("units", "days since 1858-11-17 00:00:00.0", "station_starts");
    ncFile.PutAtt("units_note", "Modified Julian Day Number (MJD)", "station_ends");

    // Put attributes for station_ends
    ncFile.PutAtt("long_name", "End", "station_ends");
    ncFile.PutAtt("var_desc", "End of the stations data", "station_ends");
    ncFile.PutAtt("units", "days since 1858-11-17 00:00:00.0", "station_ends");
    ncFile.PutAtt("units_note", "Modified Julian Day Number (MJD)", "station_ends");

    // Put attributes for data
    ncFile.PutAtt("long_name", "Data", "data");
    ncFile.PutAtt("var_desc", "Data (without any treatment)", "data");

    // Put attributes for time
    ncFile.PutAtt("long_name", "Time", "time");
    ncFile.PutAtt("units", "days since 1858-11-17 00:00:00.0", "time");
    ncFile.PutAtt("units_note", "Modified Julian Day Number (MJD)", "time");
}

bool asPredictand::SaveCommonData(asFileNetcdf& ncFile) const {
    // Provide sizes for variables
    size_t startTime[] = {0};
    size_t countTime[] = {size_t(_timeLength)};
    size_t startStations[] = {0};
    size_t countStations[] = {size_t(_stationsNb)};
    size_t start2[] = {0, 0};
    size_t count2[] = {size_t(_timeLength), size_t(_stationsNb)};

    // Write data
    ncFile.PutVarArray("time", startTime, countTime, &_time(0));
    ncFile.PutVarArray("station_names", startStations, countStations, &_stationNames[0], _stationNames.size());
    ncFile.PutVarArray("station_official_ids", startStations, countStations, &_stationOfficialIds[0],
                       _stationOfficialIds.size());
    ncFile.PutVarArray("station_ids", startStations, countStations, &_stationIds(0));
    ncFile.PutVarArray("station_heights", startStations, countStations, &_stationHeights(0));
    ncFile.PutVarArray("station_x_coords", startStations, countStations, &_stationXCoords(0));
    ncFile.PutVarArray("station_y_coords", startStations, countStations, &_stationYCoords(0));
    ncFile.PutVarArray("station_starts", startStations, countStations, &_stationStarts(0));
    ncFile.PutVarArray("station_ends", startStations, countStations, &_stationEnds(0));
    ncFile.PutVarArray("data", start2, count2, &_dataRaw(0, 0));

    return true;
}

bool asPredictand::SetStationProperties(asCatalogPredictands& currentData, size_t stationIndex) {
    _stationNames[stationIndex] = currentData.GetStationName(stationIndex);
    _stationIds(stationIndex) = currentData.GetStationId(stationIndex);
    _stationOfficialIds[stationIndex] = currentData.GetStationOfficialId(stationIndex);
    _stationXCoords(stationIndex) = currentData.GetStationCoord(stationIndex).x;
    _stationYCoords(stationIndex) = currentData.GetStationCoord(stationIndex).y;
    _stationHeights(stationIndex) = currentData.GetStationHeight(stationIndex);
    _stationStarts(stationIndex) = currentData.GetStationStart(stationIndex);
    _stationEnds(stationIndex) = currentData.GetStationEnd(stationIndex);

    return true;
}

bool asPredictand::ParseData(const wxString& catalogFile, const wxString& directory, const wxString& patternDir) {
#if USE_GUI
    // The progress bar
    asDialogProgressBar ProgressBar(_("Loading data from files.\n"), _stationsNb);
#endif

    // Get catalog
    asCatalogPredictands catalog(catalogFile);
    if (!catalog.Load()) {
        wxLogError(_("Cannot load catalog file %s"), catalogFile);
        return false;
    }

    // Get the stations list
    for (int iStat = 0; iStat < catalog.GetStationsNb(); iStat++) {
#if USE_GUI
        // Update the progress bar.
        wxString fileNameMessage = asStrF(_("Loading data from files.\nFile: %s"), catalog.GetStationFilename(iStat));
        if (!ProgressBar.Update(iStat, fileNameMessage)) {
            wxLogError(_("The process has been canceled by the user."));
            return false;
        }
#endif

        // Get station information
        if (!SetStationProperties(catalog, iStat)) return false;

        // Get file content
        if (!GetFileContent(catalog, iStat, directory, patternDir)) return false;
    }

#if USE_GUI
    ProgressBar.Destroy();
#endif

    return true;
}

bool asPredictand::GetFileContent(asCatalogPredictands& currentData, int stationIndex, const wxString& directory,
                                  const wxString& patternDir) {
    // Load file
    wxString fileFullPath;
    if (!directory.IsEmpty()) {
        fileFullPath = directory + DS + currentData.GetStationFilename(stationIndex);
    } else {
        fileFullPath = currentData.GetDataPath() + currentData.GetStationFilename(stationIndex);
    }
    asFileDat datFile(fileFullPath, asFile::ReadOnly);
    if (!datFile.Open()) return false;

    // Get the parsing format
    wxString stationFilePattern = currentData.GetStationFilepattern(stationIndex);
    asFileDat::Pattern filePattern = asFileDat::GetPattern(stationFilePattern, patternDir);
    size_t maxCharWidth = asFileDat::GetPatternLineMaxCharWidth(filePattern);

    // Jump the header
    if (!datFile.SkipLines(filePattern.headerLines)) {
        wxLogError(_("Cannot skip header lines in %s"), fileFullPath);
        return false;
    }

    // Get first index on the tima axis
    int startIndex = asFind(&_time[0], &_time[_time.size() - 1], currentData.GetStationStart(stationIndex));
    if (startIndex == asOUT_OF_RANGE || startIndex == asNOT_FOUND) {
        wxLogError(_("The given start date for \"%s\" is out of the catalog range."),
                   currentData.GetStationName(stationIndex));
        return false;
    }

    int timeIndex = startIndex;

    // Parse every line until the end of the file
    while (!datFile.EndOfFile()) {
        // Get current line
        wxString lineContent = datFile.GetNextLine();

        // Check the line width
        if (lineContent.Len() < maxCharWidth) {
            if (lineContent.Len() > 1) {
                wxLogError(_("The line length doesn't match."));
                return false;
            }
            continue;
        }

        // Check the size of the array
        if (timeIndex >= _timeLength) {
            wxLogError(_("The time index is larger than the matrix (timeIndex = %d, _timeLength = %d)."), timeIndex,
                       _timeLength);
            return false;
        }

        switch (filePattern.structType) {
            case (asFileDat::ConstantWidth): {
                if (!ParseConstantWidthContent(stationIndex, filePattern, lineContent, currentData, timeIndex)) {
                    return false;
                }
                break;
            }

            case (asFileDat::TabsDelimited): {
                ParseTabsDelimitedContent(stationIndex, filePattern, lineContent, currentData, timeIndex);

                break;
            }
        }
    }
    if (!datFile.Close()) {
        wxLogError(_("Cannot close file %s"), fileFullPath);
        return false;
    }

    // Get end index
    int endIndex = asFind(&_time[0], &_time[_time.size() - 1], currentData.GetStationEnd(stationIndex));
    if (endIndex == asOUT_OF_RANGE || endIndex == asNOT_FOUND) {
        wxLogError(_("The given end date for \"%s\" is out of the catalog range."),
                   currentData.GetStationName(stationIndex));
        return false;
    }

    // Check time width
    if (endIndex - startIndex != timeIndex - startIndex - 1) {
        wxString messageTime = asStrF(_("The length of the data in \"%s / %s\" is not coherent"), currentData.GetName(),
                                      currentData.GetStationName(stationIndex));
        wxLogError(messageTime);
        return false;
    }

    return true;
}

bool asPredictand::ParseTabsDelimitedContent(int stationIndex, const asFileDat::Pattern& pattern,
                                             const wxString& lineContent, asCatalogPredictands& currentData,
                                             int& timeIndex) {
    // Parse into a vector
    vwxs vColumns;
    wxString tmpLineContent = lineContent;
    while (tmpLineContent.Find("\t") != wxNOT_FOUND) {
        int foundCol = tmpLineContent.Find("\t");
        vColumns.push_back(tmpLineContent.Mid(0, foundCol));
        tmpLineContent = tmpLineContent.Mid(foundCol + 1);
    }
    if (!tmpLineContent.IsEmpty()) {
        vColumns.push_back(tmpLineContent);
    }

    if (pattern.parseTime) {
        // Containers. Must be a double to use wxString::ToDouble
        int valTimeYear = 0, valTimeMonth = 0, valTimeDay = 0, valTimeHour = 0, valTimeMinute = 0;

        // Get time value
        if (pattern.timeYearBegin != 0 && pattern.timeMonthBegin != 0 && pattern.timeDayBegin != 0) {
            if (pattern.timeYearBegin > vColumns.size() || pattern.timeMonthBegin > vColumns.size() ||
                pattern.timeDayBegin > vColumns.size()) {
                wxLogError(
                    _("The data file pattern is not correctly defined. "
                      "Trying to access an element (date) after the line width."));
                return false;
            }
            vColumns[pattern.timeYearBegin - 1].ToInt(&valTimeYear);
            vColumns[pattern.timeMonthBegin - 1].ToInt(&valTimeMonth);
            vColumns[pattern.timeDayBegin - 1].ToInt(&valTimeDay);
        } else {
            wxLogError(_("The data file pattern is not correctly defined."));
            return false;
        }

        if (pattern.timeHourBegin != 0) {
            if (pattern.timeHourBegin > vColumns.size()) {
                wxLogError(
                    _("The data file pattern is not correctly defined."
                      "Trying to access an element (hour) after the line width."));
                return false;
            }
            vColumns[pattern.timeHourBegin - 1].ToInt(&valTimeHour);
        }
        if (pattern.timeMinuteBegin != 0) {
            if (pattern.timeMinuteBegin > vColumns.size()) {
                wxLogError(
                    _("The data file pattern is not correctly defined."
                      "Trying to access an element (minute) after the line width."));
                return false;
            }
            vColumns[pattern.timeMinuteBegin - 1].ToInt(&valTimeMinute);
        }

        double dateData = asTime::GetMJD(valTimeYear, valTimeMonth, valTimeDay, valTimeHour, valTimeMinute, 0);

        // Find matching date
        while (dateData - _time(timeIndex) > 0.0001) {
            timeIndex++;
        }
    }

    // Get Precipitation value
    wxString dataStr = vColumns[pattern.dataBegin - 1];

    // Put value in the matrix
    _dataRaw(timeIndex, stationIndex) = ParseAndCheckDataValue(currentData, dataStr);

    timeIndex++;

    return true;
}

bool asPredictand::ParseConstantWidthContent(int stationIndex, const asFileDat::Pattern& pattern,
                                             const wxString& lineContent, asCatalogPredictands& currentData,
                                             int& timeIndex) {
    if (pattern.parseTime) {
        // Containers. Must be a double to use wxString::ToDouble
        int valTimeYear = 0, valTimeMonth = 0, valTimeDay = 0, valTimeHour = 0, valTimeMinute = 0;

        // Get time value
        if (pattern.timeYearBegin == 0 || pattern.timeYearEnd == 0 || pattern.timeMonthBegin == 0 ||
            pattern.timeMonthEnd == 0 || pattern.timeDayBegin == 0 || pattern.timeDayEnd == 0) {
            wxLogError(_("The data file pattern is not correctly defined."));
            return false;
        }

        lineContent.Mid(pattern.timeYearBegin - 1, pattern.timeYearEnd - pattern.timeYearBegin + 1).ToInt(&valTimeYear);
        lineContent.Mid(pattern.timeMonthBegin - 1, pattern.timeMonthEnd - pattern.timeMonthBegin + 1)
            .ToInt(&valTimeMonth);
        lineContent.Mid(pattern.timeDayBegin - 1, pattern.timeDayEnd - pattern.timeDayBegin + 1).ToInt(&valTimeDay);

        if (pattern.timeHourBegin != 0 && pattern.timeHourEnd != 0) {
            lineContent.Mid(pattern.timeHourBegin - 1, pattern.timeHourEnd - pattern.timeHourBegin + 1)
                .ToInt(&valTimeHour);
        }
        if (pattern.timeMinuteBegin != 0 && pattern.timeMinuteEnd != 0) {
            lineContent.Mid(pattern.timeMinuteBegin - 1, pattern.timeMinuteEnd - pattern.timeMinuteBegin + 1)
                .ToInt(&valTimeMinute);
        }

        double dateData = asTime::GetMJD(valTimeYear, valTimeMonth, valTimeDay, valTimeHour, valTimeMinute, 0);

        // Find matching date
        while (dateData - _time(timeIndex) > 0.0001) {
            timeIndex++;
        }
    }

    // Get predictand value
    wxString dataStr = lineContent.Mid(pattern.dataBegin - 1, pattern.dataEnd - pattern.dataBegin + 1);

    // Put value in the matrix
    _dataRaw(timeIndex, stationIndex) = ParseAndCheckDataValue(currentData, dataStr);

    timeIndex++;

    return true;
}

float asPredictand::ParseAndCheckDataValue(asCatalogPredictands& currentData, wxString& dataStr) const {
    // Trim
    dataStr = dataStr.Trim();
    dataStr = dataStr.Trim(true);

    // Check if not NaN
    for (size_t iNan = 0; iNan < currentData.GetNan().size(); iNan++) {
        if (dataStr.IsSameAs(currentData.GetNan()[iNan], false)) {
            return NAN;
        }
    }

    // Convert
    double dataRaw = 0;
    dataStr.ToDouble(&dataRaw);

    return (float)dataRaw;
}

a2f asPredictand::GetAnnualMax(double timeStepDays, int nansNbMax) const {
    // Flag to check the need of aggregation (timeStepDays>_timeStepDays)
    bool aggregate = false;
    int indexTimeSpanUp = 0;
    int indexTimeSpanDown = 0;

    if (timeStepDays == _timeStepDays) {
        aggregate = false;
    } else if (timeStepDays > _timeStepDays) {
        if (std::fmod(timeStepDays, _timeStepDays) > 0.0000001) {
            wxLogError(
                _("The timestep for the extraction of the predictands maximums has to be a multiple of the data "
                  "timestep."));
            a2f emptyMatrix;
            emptyMatrix << NAN;
            return emptyMatrix;
        }

        // Aggregation necessary
        aggregate = true;

        // indices to add or substract around the mid value
        indexTimeSpanUp = floor((timeStepDays / _timeStepDays) / 2);
        indexTimeSpanDown = ceil((timeStepDays / _timeStepDays) / 2) - 1;
    } else {
        wxLogError(
            _("The timestep for the extraction of the predictands maximums cannot be lower than the data timestep."));
        a2f emptyMatrix;
        emptyMatrix << NAN;
        return emptyMatrix;
    }

    // Keep the real indices of years
    int indYearStart = 0;
    int indYearEnd = 0;

    // Get catalog beginning and end
    int yearStart = asTime::GetYear(_dateStart);
    if (asTime::GetMonth(_dateStart) != 1 || asTime::GetDay(_dateStart) != 1) {
        yearStart++;
        indYearStart++;
    }
    int yearEnd = asTime::GetYear(_dateEnd);
    indYearEnd = yearEnd - yearStart + indYearStart;
    if (asTime::GetMonth(_dateEnd) != 12 || asTime::GetDay(_dateEnd) != 31) {
        yearEnd--;
    }

    // Create the container
    a2f maxMatrix = a2f::Constant(_stationsNb, indYearEnd + 1, NAN);

    // Look for maximums
    for (int iStat = 0; iStat < _stationsNb; iStat++) {
        for (int iYear = yearStart; iYear <= yearEnd; iYear++) {
            // The maximum value and a flag for accepted NaNs
            float annualmax = -99999;
            int nansNb = 0;

            // Find begining and end of the year
            int rowstart = asFindFloor(&_time[0], &_time[_timeLength - 1], asTime::GetMJD(iYear, 1, 1),
                                       asHIDE_WARNINGS);
            int rowend = asFindFloor(&_time[0], &_time[_timeLength - 1], asTime::GetMJD(iYear, 12, 31, 59, 59),
                                     asHIDE_WARNINGS);
            if ((rowend == asOUT_OF_RANGE) | (rowend == asNOT_FOUND)) {
                if (iYear == yearEnd) {
                    rowend = _timeLength - 1;
                } else {
                    annualmax = NAN;
                }
            }
            rowend -= 1;

            // Get max
            if (!aggregate) {
                for (int iRow = rowstart; iRow <= rowend; iRow++) {
                    if (!isnan(_dataRaw(iRow, iStat))) {
                        annualmax = std::max(_dataRaw(iRow, iStat), annualmax);
                    } else {
                        nansNb++;
                    }
                }
                if (nansNb > nansNbMax) {
                    annualmax = NAN;
                }
            } else {
                // Correction for both extremes
                rowstart = std::max(rowstart - indexTimeSpanDown, 0);
                rowstart += indexTimeSpanDown;
                rowend = std::min(rowend + indexTimeSpanUp, (int)_dataRaw.rows() - 1);
                rowend -= indexTimeSpanUp;

                // Loop within the new limits
                for (int iRow = rowstart; iRow <= rowend; iRow++) {
                    float timeStepSum = 0;
                    for (int iEl = iRow - indexTimeSpanDown; iEl <= iRow + indexTimeSpanUp; iEl++) {
                        if (!isnan(_dataRaw(iEl, iStat))) {
                            timeStepSum += _dataRaw(iEl, iStat);
                        } else {
                            timeStepSum = NAN;
                            break;
                        }
                    }

                    if (!isnan(timeStepSum)) {
                        annualmax = std::max(timeStepSum, annualmax);
                    } else {
                        nansNb++;
                    }
                }
                if (nansNb > nansNbMax) {
                    annualmax = NAN;
                }
            }

            maxMatrix(iStat, iYear - yearStart + indYearStart) = annualmax;
        }
    }

    return maxMatrix;
}

int asPredictand::GetStationIndex(int stationId) const {
    return asFind(&_stationIds[0], &_stationIds[_stationsNb - 1], stationId);
}
