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

#include "asCatalogPredictands.h"
#include "asFileDat.h"
#include "asFileNetcdf.h"
#include "asPredictandLightnings.h"
#include "asPredictandPrecipitation.h"
#include "asPredictandTemperature.h"
#include "asTimeArray.h"

asPredictand::asPredictand(Parameter dataParameter, TemporalResolution dataTemporalResolution,
                           SpatialAggregation dataSpatialAggregation)
    : m_fileVersion(1.4f),
      m_parameter(dataParameter),
      m_temporalResolution(dataTemporalResolution),
      m_spatialAggregation(dataSpatialAggregation),
      m_timeStepDays(0.0),
      m_timeLength(0),
      m_stationsNb(0),
      m_dateProcessed(0.0),
      m_dateStart(0.0),
      m_dateEnd(0.0),
      m_hasNormalizedData(false),
      m_hasReferenceValues(false) {}

asPredictand::Parameter asPredictand::StringToParameterEnum(const wxString &parameterStr) {
  if (parameterStr.CmpNoCase("Precipitation") == 0) {
    return Precipitation;
  } else if (parameterStr.CmpNoCase("AirTemperature") == 0) {
    return AirTemperature;
  } else if (parameterStr.CmpNoCase("Lightnings") == 0) {
    return Lightnings;
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
    case (Lightnings):
      return "Lightnings";
    case (Wind):
      return "Wind";
    default:
      wxLogError(_("The given data parameter type in unknown."));
  }
  return wxEmptyString;
}

asPredictand::Unit asPredictand::StringToUnitEnum(const wxString &unitStr) {
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

asPredictand::TemporalResolution asPredictand::StringToTemporalResolutionEnum(const wxString &temporalResolution) {
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
  } else if (temporalResolution.CmpNoCase("TwoDays") == 0) {
    return TwoDays;
  } else if (temporalResolution.CmpNoCase("2 days") == 0) {
    return TwoDays;
  } else if (temporalResolution.CmpNoCase("ThreeDays") == 0) {
    return ThreeDays;
  } else if (temporalResolution.CmpNoCase("3 days") == 0) {
    return ThreeDays;
  } else if (temporalResolution.CmpNoCase("Weekly") == 0) {
    return Weekly;
  } else if (temporalResolution.CmpNoCase("1 week") == 0) {
    return Weekly;
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
    case (TwoDays):
      return "TwoDays";
    case (ThreeDays):
      return "ThreeDays";
    case (Weekly):
      return "Weekly";
    default:
      wxLogError(_("The given data temporal resolution type in unknown."));
  }
  return wxEmptyString;
}

asPredictand::SpatialAggregation asPredictand::StringToSpatialAggregationEnum(const wxString &spatialAggregation) {
  if (spatialAggregation.CmpNoCase("Station") == 0) {
    return Station;
  } else if (spatialAggregation.CmpNoCase("Groupment") == 0) {
    return Groupment;
  } else if (spatialAggregation.CmpNoCase("Catchment") == 0) {
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

asPredictand *asPredictand::GetInstance(const wxString &parameterStr, const wxString &temporalResolutionStr,
                                        const wxString &spatialAggregationStr) {
  Parameter parameter = StringToParameterEnum(parameterStr);
  TemporalResolution temporalResolution = StringToTemporalResolutionEnum(temporalResolutionStr);
  SpatialAggregation spatialAggregation = StringToSpatialAggregationEnum(spatialAggregationStr);

  asPredictand *db = asPredictand::GetInstance(parameter, temporalResolution, spatialAggregation);
  return db;
}

asPredictand *asPredictand::GetInstance(Parameter parameter, TemporalResolution temporalResolution,
                                        SpatialAggregation spatialAggregation) {
  switch (parameter) {
    case (Precipitation): {
      asPredictand *db = new asPredictandPrecipitation(parameter, temporalResolution, spatialAggregation);
      return db;
    }
    case (AirTemperature): {
      asPredictand *db = new asPredictandTemperature(parameter, temporalResolution, spatialAggregation);
      return db;
    }
    case (Lightnings): {
      asPredictand *db = new asPredictandLightnings(parameter, temporalResolution, spatialAggregation);
      return db;
    }
    default:
      wxLogError(_("The predictand parameter is not listed in the asPredictand instance factory."));
      return nullptr;
  }
}

asPredictand *asPredictand::GetInstance(const wxString &filePath) {
  // Open the NetCDF file
  wxLogVerbose(_("Opening the file %s"), filePath);
  asFileNetcdf ncFile(filePath, asFileNetcdf::ReadOnly);
  if (!ncFile.Open()) {
    wxLogError(_("Couldn't open file %s"), filePath);
    return nullptr;
  } else {
    wxLogVerbose(_("File successfully opened"));
  }

  // Check version
  float version = ncFile.GetAttFloat("version");
  if (asIsNaN(version) || version <= 1.0) {
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
  ncFile.Close();

  // Get instance
  asPredictand *db = asPredictand::GetInstance(dataParameter, dataTemporalResolution, dataSpatialAggregation);
  return db;
}

wxString asPredictand::GetDBFilePathSaving(const wxString &destinationDir) const {
  wxString dataParameterStr = ParameterEnumToString(m_parameter);
  wxString dataTemporalResolutionStr = asPredictand::TemporalResolutionEnumToString(m_temporalResolution);
  wxString dataSpatialAggregationStr = asPredictand::SpatialAggregationEnumToString(m_spatialAggregation);
  wxString fileName =
      dataParameterStr + "-" + dataTemporalResolutionStr + "-" + dataSpatialAggregationStr + "-" + m_datasetId;

  wxString predictandDBFilePath = destinationDir + DS + fileName + ".nc";

  return predictandDBFilePath;
}

bool asPredictand::InitMembers(const wxString &catalogFilePath) {
  // Starting and ending date of the DB, to be overwritten
  m_dateStart = asTime::GetMJD(2100, 1, 1);
  m_dateEnd = asTime::GetMJD(1800, 1, 1);

  // Get the catalog information
  asCatalogPredictands catalog(catalogFilePath);
  if (!catalog.Load()) return false;

  // Get first and last date
  if (catalog.GetStart() < m_dateStart) m_dateStart = catalog.GetStart();
  if (catalog.GetEnd() > m_dateEnd) m_dateEnd = catalog.GetEnd();

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

bool asPredictand::InitBaseContainers() {
  if (m_stationsNb < 1) {
    wxLogError(_("The stations number is inferior to 1."));
    return false;
  }
  if (m_timeLength < 1) {
    wxLogError(_("The time length is inferior to 1."));
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
  m_dataRaw.resize(m_timeLength, m_stationsNb);
  m_dataRaw.fill(NaNf);
  if (m_hasNormalizedData) {
    m_dataNormalized.resize(m_timeLength, m_stationsNb);
    m_dataNormalized.fill(NaNf);
  }

  return true;
}

bool asPredictand::LoadCommonData(asFileNetcdf &ncFile) {
  // Check version
  float version = ncFile.GetAttFloat("version");
  if (asIsNaN(version) || version <= 1.1) {
    wxLogError(
        _("The predictand DB file was made with an older version of AtmoSwing that is no longer supported. Please "
          "generate the file with the actual version."));
    return false;
  }

  // Get global attributes
  m_parameter = (Parameter)ncFile.GetAttInt("data_parameter");
  m_temporalResolution = (TemporalResolution)ncFile.GetAttInt("data_temporal_resolution");
  m_spatialAggregation = (SpatialAggregation)ncFile.GetAttInt("data_spatial_aggregation");
  m_datasetId = ncFile.GetAttString("dataset_id");
  m_hasNormalizedData = ncFile.HasVariable("data_normalized");
  m_hasReferenceValues = m_hasNormalizedData;

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
  size_t indexStart[2] = {0, 0};
  size_t indexCount[2] = {size_t(m_timeLength), size_t(m_stationsNb)};
  m_dataRaw.resize(m_timeLength, m_stationsNb);

  if (asIsNaN(version) || version <= 1.3) {
    ncFile.GetVarArray("data_gross", indexStart, indexCount, &m_dataRaw(0, 0));
  } else {
    ncFile.GetVarArray("data", indexStart, indexCount, &m_dataRaw(0, 0));
  }

  return true;
}

void asPredictand::SetCommonDefinitions(asFileNetcdf &ncFile) const {
  // Define dimensions. Time is the unlimited dimension.
  ncFile.DefDim("stations", m_stationsNb);
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
  ncFile.PutAtt("version", &m_fileVersion);
  auto dataParameter = (int)m_parameter;
  ncFile.PutAtt("data_parameter", &dataParameter);
  auto dataTemporalResolution = (int)m_temporalResolution;
  ncFile.PutAtt("data_temporal_resolution", &dataTemporalResolution);
  auto dataSpatialAggregation = (int)m_spatialAggregation;
  ncFile.PutAtt("data_spatial_aggregation", &dataSpatialAggregation);
  ncFile.PutAtt("dataset_id", m_datasetId);

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
  ncFile.PutAtt("units", "Modified Julian Day Number (MJD)", "station_starts");

  // Put attributes for station_ends
  ncFile.PutAtt("long_name", "End", "station_ends");
  ncFile.PutAtt("var_desc", "End of the stations data", "station_ends");
  ncFile.PutAtt("units", "Modified Julian Day Number (MJD)", "station_ends");

  // Put attributes for data
  ncFile.PutAtt("long_name", "Data", "data");
  ncFile.PutAtt("var_desc", "Data (whithout any treatment)", "data");
}

bool asPredictand::SaveCommonData(asFileNetcdf &ncFile) const {
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
  ncFile.PutVarArray("data", start2, count2, &m_dataRaw(0, 0));

  return true;
}

bool asPredictand::SetStationProperties(asCatalogPredictands &currentData, size_t stationIndex) {
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

bool asPredictand::ParseData(const wxString &catalogFile, const wxString &directory, const wxString &patternDir) {
#if wxUSE_GUI
  // The progress bar
  asDialogProgressBar ProgressBar(_("Loading data from files.\n"), m_stationsNb);
#endif

  // Get catalog
  asCatalogPredictands catalog(catalogFile);
  catalog.Load();

  // Get the stations list
  for (int iStat = 0; iStat < catalog.GetStationsNb(); iStat++) {
#if wxUSE_GUI
    // Update the progress bar.
    wxString fileNameMessage =
        wxString::Format(_("Loading data from files.\nFile: %s"), catalog.GetStationFilename(iStat));
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

#if wxUSE_GUI
  ProgressBar.Destroy();
#endif

  return true;
}

bool asPredictand::GetFileContent(asCatalogPredictands &currentData, size_t stationIndex, const wxString &directory,
                                  const wxString &patternDir) {
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
  datFile.SkipLines(filePattern.headerLines);

  // Get first index on the tima axis
  int startIndex = asFind(&m_time[0], &m_time[m_time.size() - 1], currentData.GetStationStart(stationIndex));
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
    if (lineContent.Len() >= maxCharWidth) {
      // Check the size of the array
      if (timeIndex >= m_timeLength) {
        wxLogError(_("The time index is larger than the matrix (timeIndex = %d, m_timeLength = %d)."), timeIndex,
                   m_timeLength);
        return false;
      }

      switch (filePattern.structType) {
        case (asFileDat::ConstantWidth): {
          if (filePattern.parseTime) {
            // Containers. Must be a double to use wxString::ToDouble
            double valTimeYear = 0, valTimeMonth = 0, valTimeDay = 0, valTimeHour = 0, valTimeMinute = 0;

            // Get time value
            if (filePattern.timeYearBegin != 0 && filePattern.timeYearEnd != 0 && filePattern.timeMonthBegin != 0 &&
                filePattern.timeMonthEnd != 0 && filePattern.timeDayBegin != 0 && filePattern.timeDayEnd != 0) {
              lineContent.Mid(filePattern.timeYearBegin - 1, filePattern.timeYearEnd - filePattern.timeYearBegin + 1)
                  .ToDouble(&valTimeYear);
              lineContent.Mid(filePattern.timeMonthBegin - 1, filePattern.timeMonthEnd - filePattern.timeMonthBegin + 1)
                  .ToDouble(&valTimeMonth);
              lineContent.Mid(filePattern.timeDayBegin - 1, filePattern.timeDayEnd - filePattern.timeDayBegin + 1)
                  .ToDouble(&valTimeDay);
            } else {
              wxLogError(_("The data file pattern is not correctly defined."));
              return false;
            }

            if (filePattern.timeHourBegin != 0 && filePattern.timeHourEnd != 0) {
              lineContent.Mid(filePattern.timeHourBegin - 1, filePattern.timeHourEnd - filePattern.timeHourBegin + 1)
                  .ToDouble(&valTimeHour);
            }
            if (filePattern.timeMinuteBegin != 0 && filePattern.timeMinuteEnd != 0) {
              lineContent
                  .Mid(filePattern.timeMinuteBegin - 1, filePattern.timeMinuteEnd - filePattern.timeMinuteBegin + 1)
                  .ToDouble(&valTimeMinute);
            }

            double dateData = asTime::GetMJD(valTimeYear, valTimeMonth, valTimeDay, valTimeHour, valTimeMinute, 0);

            // Find matching date
            while (dateData - m_time(timeIndex) > 0.0001) {
              timeIndex++;
            }
          }

          // Get predictand value
          wxString dataStr =
              lineContent.Mid(filePattern.dataBegin - 1, filePattern.dataEnd - filePattern.dataBegin + 1);

          // Put value in the matrix
          m_dataRaw(timeIndex, stationIndex) = ParseAndCheckDataValue(currentData, dataStr);

          timeIndex++;
          break;
        }

        case (asFileDat::TabsDelimited): {
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

          if (filePattern.parseTime) {
            // Containers. Must be a double to use wxString::ToDouble
            double valTimeYear = 0, valTimeMonth = 0, valTimeDay = 0, valTimeHour = 0, valTimeMinute = 0;

            // Get time value
            if (filePattern.timeYearBegin != 0 && filePattern.timeMonthBegin != 0 && filePattern.timeDayBegin != 0) {
              if (filePattern.timeYearBegin > vColumns.size() || filePattern.timeMonthBegin > vColumns.size() ||
                  filePattern.timeDayBegin > vColumns.size()) {
                wxLogError(
                    _("The data file pattern is not correctly defined. Trying to access an element "
                      "(date) after the line width."));
                return false;
              }
              vColumns[filePattern.timeYearBegin - 1].ToDouble(&valTimeYear);
              vColumns[filePattern.timeMonthBegin - 1].ToDouble(&valTimeMonth);
              vColumns[filePattern.timeDayBegin - 1].ToDouble(&valTimeDay);
            } else {
              wxLogError(_("The data file pattern is not correctly defined."));
              return false;
            }

            if (filePattern.timeHourBegin != 0) {
              if (filePattern.timeHourBegin > vColumns.size()) {
                wxLogError(
                    _("The data file pattern is not correctly defined. Trying to access an element "
                      "(hour) after the line width."));
                return false;
              }
              vColumns[filePattern.timeHourBegin - 1].ToDouble(&valTimeHour);
            }
            if (filePattern.timeMinuteBegin != 0) {
              if (filePattern.timeMinuteBegin > vColumns.size()) {
                wxLogError(
                    _("The data file pattern is not correctly defined. Trying to access an element "
                      "(minute) after the line width."));
                return false;
              }
              vColumns[filePattern.timeMinuteBegin - 1].ToDouble(&valTimeMinute);
            }

            double dateData = asTime::GetMJD(valTimeYear, valTimeMonth, valTimeDay, valTimeHour, valTimeMinute, 0);

            // Find matching date
            while (dateData - m_time(timeIndex) > 0.0001) {
              timeIndex++;
            }
          }

          // Get Precipitation value
          wxString dataStr = vColumns[filePattern.dataBegin - 1];

          // Put value in the matrix
          m_dataRaw(timeIndex, stationIndex) = ParseAndCheckDataValue(currentData, dataStr);

          timeIndex++;
          break;
        }
      }
    } else {
      if (lineContent.Len() > 1) {
        wxLogError(_("The line length doesn't match."));
        return false;
      }
    }
  }
  datFile.Close();

  // Get end index
  int endIndex = asFind(&m_time[0], &m_time[m_time.size() - 1], currentData.GetStationEnd(stationIndex));
  if (endIndex == asOUT_OF_RANGE || endIndex == asNOT_FOUND) {
    wxLogError(_("The given end date for \"%s\" is out of the catalog range."),
               currentData.GetStationName(stationIndex));
    return false;
  }

  // Check time width
  if (endIndex - startIndex != timeIndex - startIndex - 1) {
    wxString messageTime = wxString::Format(_("The length of the data in \"%s / %s\" is not coherent"),
                                            currentData.GetName(), currentData.GetStationName(stationIndex));
    wxLogError(messageTime);
    return false;
  }

  return true;
}

float asPredictand::ParseAndCheckDataValue(asCatalogPredictands &currentData, wxString &dataStr) const {
  // Trim
  dataStr = dataStr.Trim();
  dataStr = dataStr.Trim(true);

  // Check if not NaN
  for (size_t iNan = 0; iNan < currentData.GetNan().size(); iNan++) {
    if (dataStr.IsSameAs(currentData.GetNan()[iNan], false)) {
      return NaNf;
    }
  }

  // Convert
  double dataRaw = 0;
  dataStr.ToDouble(&dataRaw);

  return (float)dataRaw;
}

a2f asPredictand::GetAnnualMax(double timeStepDays, int nansNbMax) const {
  // Flag to check the need of aggregation (timeStepDays>m_timeStepDays)
  bool aggregate = false;
  int indexTimeSpanUp = 0;
  int indexTimeSpanDown = 0;

  if (timeStepDays == m_timeStepDays) {
    aggregate = false;
  } else if (timeStepDays > m_timeStepDays) {
    if (std::fmod(timeStepDays, m_timeStepDays) > 0.0000001) {
      wxLogError(
          _("The timestep for the extraction of the predictands maximums has to be a multiple of the data "
            "timestep."));
      a2f emptyMatrix;
      emptyMatrix << NaNf;
      return emptyMatrix;
    }

    // Aggregation necessary
    aggregate = true;

    // indices to add or substract around the mid value
    indexTimeSpanUp = floor((timeStepDays / m_timeStepDays) / 2);
    indexTimeSpanDown = ceil((timeStepDays / m_timeStepDays) / 2) - 1;
  } else {
    wxLogError(
        _("The timestep for the extraction of the predictands maximums cannot be lower than the data timestep."));
    a2f emptyMatrix;
    emptyMatrix << NaNf;
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
  a2f maxMatrix = a2f::Constant(m_stationsNb, indYearEnd + 1, NaNf);

  // Look for maximums
  for (int iStat = 0; iStat < m_stationsNb; iStat++) {
    for (int iYear = yearStart; iYear <= yearEnd; iYear++) {
      // The maximum value and a flag for accepted NaNs
      float annualmax = -99999;
      int nansNb = 0;

      // Find begining and end of the year
      int rowstart = asFindFloor(&m_time[0], &m_time[m_timeLength - 1], asTime::GetMJD(iYear, 1, 1), asHIDE_WARNINGS);
      int rowend =
          asFindFloor(&m_time[0], &m_time[m_timeLength - 1], asTime::GetMJD(iYear, 12, 31, 59, 59), asHIDE_WARNINGS);
      if ((rowend == asOUT_OF_RANGE) | (rowend == asNOT_FOUND)) {
        if (iYear == yearEnd) {
          rowend = m_timeLength - 1;
        } else {
          annualmax = NaNf;
        }
      }
      rowend -= 1;

      // Get max
      if (!aggregate) {
        for (int iRow = rowstart; iRow <= rowend; iRow++) {
          if (!asIsNaN(m_dataRaw(iRow, iStat))) {
            annualmax = wxMax(m_dataRaw(iRow, iStat), annualmax);
          } else {
            nansNb++;
          }
        }
        if (nansNb > nansNbMax) {
          annualmax = NaNf;
        }
      } else {
        // Correction for both extremes
        rowstart = wxMax(rowstart - indexTimeSpanDown, 0);
        rowstart += indexTimeSpanDown;
        rowend = wxMin(rowend + indexTimeSpanUp, (int)m_dataRaw.rows() - 1);
        rowend -= indexTimeSpanUp;

        // Loop within the new limits
        for (int iRow = rowstart; iRow <= rowend; iRow++) {
          float timeStepSum = 0;
          for (int iEl = iRow - indexTimeSpanDown; iEl <= iRow + indexTimeSpanUp; iEl++) {
            if (!asIsNaN(m_dataRaw(iEl, iStat))) {
              timeStepSum += m_dataRaw(iEl, iStat);
            } else {
              timeStepSum = NaNf;
              break;
            }
          }

          if (!asIsNaN(timeStepSum)) {
            annualmax = wxMax(timeStepSum, annualmax);
          } else {
            nansNb++;
          }
        }
        if (nansNb > nansNbMax) {
          annualmax = NaNf;
        }
      }

      maxMatrix(iStat, iYear - yearStart + indYearStart) = annualmax;
    }
  }

  return maxMatrix;
}

int asPredictand::GetStationIndex(int stationId) const {
  return asFind(&m_stationIds[0], &m_stationIds[m_stationsNb - 1], stationId);
}
