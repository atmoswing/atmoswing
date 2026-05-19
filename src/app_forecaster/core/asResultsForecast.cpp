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

#include "asResultsForecast.h"

#include <vector>
#include <numeric>
#include <wx/tokenzr.h>

#include "asFileNetcdf.h"

asResultsForecast::asResultsForecast()
    : asResults(),
      _predictandParameter(asPredictand::Precipitation),
      _predictandTemporalResolution(asPredictand::Daily),
      _predictandSpatialAggregation(asPredictand::Station),
      _hasReferenceValues(false),
      _leadTimeOrigin(0.0) {}

void asResultsForecast::Init(asParametersForecast& params, double leadTimeOrigin) {
    // Resize to 0 to avoid keeping old results
    _targetDates.resize(0);
    _stationNames.resize(0);
    _stationOfficialIds.resize(0);
    _stationIds.resize(0);
    _stationHeights.resize(0);
    _analogsNb.resize(0);
    _analogsCriteria.resize(0);
    _analogsDates.resize(0);
    _analogsValuesRaw.resize(0);
    _analogsValuesNorm.resize(0);
    _stationXCoords.resize(0);
    _stationYCoords.resize(0);
    _referenceAxis.resize(0);
    _referenceValues.resize(0, 0);
    _predictorDatasetIdsOper.resize(0);
    _predictorDatasetIdsArchive.resize(0);
    _predictorDataIdsOper.resize(0);
    _predictorDataIdsArchive.resize(0);
    _predictorLevels.resize(0);
    _predictorHours.resize(0);
    _predictorLonMin.resize(0);
    _predictorLonMax.resize(0);
    _predictorLatMin.resize(0);
    _predictorLatMax.resize(0);

    _methodId = params.GetMethodId();
    _methodIdDisplay = params.GetMethodIdDisplay();
    _specificTag = params.GetSpecificTag();
    _specificTagDisplay = params.GetSpecificTagDisplay();
    _description = params.GetDescription();
    _predictandDatabase = params.GetPredictandDatabase();
    _predictandStationIds = params.GetPredictandStationIds();

    _leadTimeOrigin = leadTimeOrigin;
    _dateProcessed = asTime::NowMJD(asUTM);

    // Set the analogs number
    _analogsNb.resize(params.GetLeadTimeNb());
    for (int i = 0; i < params.GetLeadTimeNb(); i++) {
        _analogsNb[i] = params.GetAnalogsNumberLeadTime(_currentStep, i);
    }

    BuildFileName();
}

void asResultsForecast::BuildFileName() {
    wxASSERT(!_forecastsDir.IsEmpty());

    if (_methodId.IsEmpty() || _specificTag.IsEmpty()) {
        wxLogError(_("The provided ID or the tag is empty, which isn't allowed !"));
    }

    // Base directory
    _filePath = _forecastsDir;
    _filePath.Append(DS);
    if (!_subFolder.IsEmpty()) {
        _filePath.Append(DS);
        _filePath.Append(_subFolder);
    }

    // Directory
    wxString dirstructure = "YYYY";
    dirstructure.Append(DS);
    dirstructure.Append("MM");
    dirstructure.Append(DS);
    dirstructure.Append("DD");
    wxString directory = asTime::GetStringTime(_leadTimeOrigin, dirstructure);
    _filePath.Append(directory);
    _filePath.Append(DS);

    // Filename
    wxString forecastname = _methodId + '.' + _specificTag;
    wxString nowstr = asTime::GetStringTime(_leadTimeOrigin, "YYYY-MM-DD_hh");
    wxString ext = "nc";
    wxString filename = asStrF("%s.%s.%s", nowstr, forecastname, ext);
    _filePath.Append(filename);
}

bool asResultsForecast::Save() {
    wxASSERT(!_filePath.IsEmpty());
    wxASSERT(_targetDates.size() > 0);
    wxASSERT(_analogsNb.size() > 0);
    wxASSERT(!_stationNames.empty());
    wxASSERT(!_stationOfficialIds.empty());
    wxASSERT(_stationHeights.size() > 0);
    wxASSERT(_stationIds.size() > 0);
    wxASSERT(!_analogsCriteria.empty());
    wxASSERT(!_analogsDates.empty());
    wxASSERT(!_analogsValuesRaw.empty());
    wxASSERT(!_analogsValuesNorm.empty());
    wxASSERT(_stationXCoords.size() > 0);
    wxASSERT(_stationYCoords.size() > 0);

    if (_hasReferenceValues) {
        wxASSERT(_referenceAxis.size() > 0);
        wxASSERT(_referenceValues.cols() > 0);
        wxASSERT(_referenceValues.rows() > 0);
    }

    // Get the elements size
    size_t nLeadtime = _targetDates.size();
    size_t nAnalogsTot = _analogsNb.sum();
    size_t nStations = _stationIds.size();
    size_t nReferenceAxis = _referenceAxis.size();
    size_t nPredictors = _predictorDatasetIdsOper.size();

    ThreadsManager().CritSectionNetCDF().Enter();

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(_filePath, asFileNetcdf::Replace);
    if (!ncFile.Open()) {
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Set general attributes
    ncFile.PutAtt("version_major", &_fileVersionMajor);
    ncFile.PutAtt("version_minor", &_fileVersionMinor);
    ncFile.PutAtt("predictand_parameter", asPredictand::ParameterEnumToString(_predictandParameter));
    ncFile.PutAtt("predictand_temporal_resolution",
                  asPredictand::TemporalResolutionEnumToString(_predictandTemporalResolution));
    ncFile.PutAtt("predictand_spatial_aggregation",
                  asPredictand::SpatialAggregationEnumToString(_predictandSpatialAggregation));
    ncFile.PutAtt("predictand_dataset_id", _predictandDatasetId);
    ncFile.PutAtt("predictand_database", _predictandDatabase);
    ncFile.PutAtt("predictand_station_ids", GetPredictandStationIdsString());
    ncFile.PutAtt("method_id", _methodId);
    ncFile.PutAtt("method_id_display", _methodIdDisplay);
    ncFile.PutAtt("specific_tag", _specificTag);
    ncFile.PutAtt("specific_tag_display", _specificTagDisplay);
    ncFile.PutAtt("description", _description);
    ncFile.PutAtt("date_processed", &_dateProcessed);
    ncFile.PutAtt("lead_time_origin", &_leadTimeOrigin);
    ncFile.PutAtt("coordinate_system", _coordinateSystem);
    short hasReferenceValues = 0;
    if (_hasReferenceValues) {
        hasReferenceValues = 1;
    }
    ncFile.PutAtt("has_reference_values", &hasReferenceValues);

    // Define dimensions. No unlimited dimension.
    ncFile.DefDim("lead_time", nLeadtime);
    ncFile.DefDim("analogs_tot", nAnalogsTot);
    ncFile.DefDim("stations", nStations);
    if (_hasReferenceValues) {
        ncFile.DefDim("reference_axis", nReferenceAxis);
    }
    ncFile.DefDim("predictors", nPredictors);

    // The dimensions name array is used to pass the dimensions to the variable.
    vstds dimNamesLeadTime;
    dimNamesLeadTime.emplace_back("lead_time");
    vstds dimNamesAnalogsTot;
    dimNamesAnalogsTot.emplace_back("analogs_tot");
    vstds dimNamesStations;
    dimNamesStations.emplace_back("stations");
    vstds dimNamesAnalogsStations;
    dimNamesAnalogsStations.emplace_back("stations");
    dimNamesAnalogsStations.emplace_back("analogs_tot");
    vstds dimNameReferenceAxis;
    vstds dimNameReferenceValues;
    if (_hasReferenceValues) {
        dimNameReferenceAxis.emplace_back("reference_axis");
        dimNameReferenceValues.emplace_back("stations");
        dimNameReferenceValues.emplace_back("reference_axis");
    }
    vstds dimNamesPredictors;
    dimNamesPredictors.emplace_back("predictors");

    // Define variables
    ncFile.DefVar("target_dates", NC_FLOAT, 1, dimNamesLeadTime);
    ncFile.DefVar("analogs_nb", NC_INT, 1, dimNamesLeadTime);
    ncFile.DefVar("station_names", NC_STRING, 1, dimNamesStations);
    ncFile.DefVar("station_ids", NC_INT, 1, dimNamesStations);
    ncFile.DefVar("station_official_ids", NC_STRING, 1, dimNamesStations);
    ncFile.DefVar("station_heights", NC_FLOAT, 1, dimNamesStations);
    ncFile.DefVar("station_x_coords", NC_DOUBLE, 1, dimNamesStations);
    ncFile.DefVar("station_y_coords", NC_DOUBLE, 1, dimNamesStations);
    ncFile.DefVar("analog_criteria", NC_FLOAT, 1, dimNamesAnalogsTot);
    ncFile.DefVar("analog_dates", NC_FLOAT, 1, dimNamesAnalogsTot);
    ncFile.DefVar("analog_values_raw", NC_FLOAT, 2, dimNamesAnalogsStations);
    ncFile.DefVar("analog_values_norm", NC_FLOAT, 2, dimNamesAnalogsStations);
    ncFile.DefVarDeflate("analog_values_raw");
    ncFile.DefVarDeflate("analog_values_norm");
    if (_hasReferenceValues) {
        ncFile.DefVar("reference_axis", NC_FLOAT, 1, dimNameReferenceAxis);
        ncFile.DefVar("reference_values", NC_FLOAT, 2, dimNameReferenceValues);
    }
    ncFile.DefVar("predictor_dataset_ids_realtime", NC_STRING, 1, dimNamesPredictors);
    ncFile.DefVar("predictor_dataset_ids_archive", NC_STRING, 1, dimNamesPredictors);
    ncFile.DefVar("predictor_data_ids_realtime", NC_STRING, 1, dimNamesPredictors);
    ncFile.DefVar("predictor_data_ids_archive", NC_STRING, 1, dimNamesPredictors);
    ncFile.DefVar("predictor_levels", NC_FLOAT, 1, dimNamesPredictors);
    ncFile.DefVar("predictor_hours", NC_FLOAT, 1, dimNamesPredictors);
    ncFile.DefVar("predictor_lon_min", NC_FLOAT, 1, dimNamesPredictors);
    ncFile.DefVar("predictor_lon_max", NC_FLOAT, 1, dimNamesPredictors);
    ncFile.DefVar("predictor_lat_min", NC_FLOAT, 1, dimNamesPredictors);
    ncFile.DefVar("predictor_lat_max", NC_FLOAT, 1, dimNamesPredictors);

    // Put attributes
    DefTargetDatesAttributes(ncFile);
    DefStationIdsAttributes(ncFile);
    DefStationOfficialIdsAttributes(ncFile);
    DefAnalogsNbAttributes(ncFile);
    DefAnalogsCriteriaAttributes(ncFile);
    DefAnalogsValuesRawAttributes(ncFile);
    DefAnalogsValuesNormAttributes(ncFile);
    DefAnalogsDatesAttributes(ncFile);

    ncFile.PutAtt("long_name", "Station names", "station_names");
    ncFile.PutAtt("var_desc", "Name of the weather stations", "station_names");
    ncFile.PutAtt("long_name", "Station heights", "station_heights");
    ncFile.PutAtt("var_desc", "Altitude of the weather stations", "station_heights");
    ncFile.PutAtt("units", "m", "station_heights");
    ncFile.PutAtt("long_name", "X coordinate", "station_x_coords");
    ncFile.PutAtt("var_desc", "X coordinate (west-east)", "station_x_coords");
    ncFile.PutAtt("units", "m", "station_x_coords");
    ncFile.PutAtt("long_name", "Y coordinate", "station_y_coords");
    ncFile.PutAtt("var_desc", "Y coordinate (west-east)", "station_y_coords");
    ncFile.PutAtt("units", "m", "station_y_coords");
    if (_hasReferenceValues) {
        ncFile.PutAtt("long_name", "Reference axis", "reference_axis");
        ncFile.PutAtt("var_desc", "Reference axis", "reference_axis");
        ncFile.PutAtt("long_name", "Reference values", "reference_values");
        ncFile.PutAtt("var_desc", "Reference values", "reference_values");
    }
    ncFile.PutAtt("long_name", "Realtime predictor dataset IDs", "predictor_dataset_ids_realtime");
    ncFile.PutAtt("var_desc", "Realtime (NWP) predictor dataset IDs used to compute the forecast.",
                  "predictor_dataset_ids_realtime");
    ncFile.PutAtt("long_name", "Archive predictor dataset IDs", "predictor_dataset_ids_archive");
    ncFile.PutAtt("var_desc", "Archive predictor dataset IDs used to compute the forecast.",
                  "predictor_dataset_ids_archive");
    ncFile.PutAtt("long_name", "Realtime predictor data IDs", "predictor_data_ids_realtime");
    ncFile.PutAtt("var_desc", "Realtime (NWP) predictor data IDs used to compute the forecast.",
                  "predictor_data_ids_realtime");
    ncFile.PutAtt("long_name", "Archive predictor data IDs", "predictor_data_ids_archive");
    ncFile.PutAtt("var_desc", "Archive predictor data IDs used to compute the forecast.", "predictor_data_ids_archive");
    ncFile.PutAtt("long_name", "Predictor levels", "predictor_levels");
    ncFile.PutAtt("var_desc", "Predictor levels used to compute the forecast.", "predictor_levels");
    ncFile.PutAtt("long_name", "Predictor hours", "predictor_hours");
    ncFile.PutAtt("var_desc", "Predictor hours used to compute the forecast.", "predictor_hours");
    ncFile.PutAtt("long_name", "Predictor min longitudes", "predictor_lon_min");
    ncFile.PutAtt("var_desc", "Predictor minimum longitudes", "predictor_lon_min");
    ncFile.PutAtt("long_name", "Predictor max longitudes", "predictor_lon_max");
    ncFile.PutAtt("var_desc", "Predictor maximum longitudes", "predictor_lon_max");
    ncFile.PutAtt("long_name", "Predictor min latitudes", "predictor_lat_min");
    ncFile.PutAtt("var_desc", "Predictor minimum latitudes", "predictor_lat_min");
    ncFile.PutAtt("long_name", "Predictor max latitudes", "predictor_lat_max");
    ncFile.PutAtt("var_desc", "Predictor maximum latitudes", "predictor_lat_max");

    // End definitions: leave define mode
    ncFile.EndDef();

    // Provide sizes for variables
    size_t startLeadTime[] = {0};
    size_t countLeadTime[] = {nLeadtime};
    size_t startAnalogsTot[] = {0};
    size_t countAnalogsTot[] = {nAnalogsTot};
    size_t startStations[] = {0};
    size_t countStations[] = {nStations};
    size_t startAnalogsStations[] = {0, 0};
    size_t countAnalogsStations[] = {nStations, nAnalogsTot};
    size_t startPredictors[] = {0};
    size_t countPredictors[] = {nPredictors};

    // Set the matrices in vectors
    vf analogsCriteria(nAnalogsTot);
    vf analogsDates(nAnalogsTot);
    vf analogsValuesRaw(nAnalogsTot * nStations);
    vf analogsValuesNorm(nAnalogsTot * nStations);

    int ind = 0;
    for (int iTime = 0; iTime < nLeadtime; iTime++) {
        for (int iAnalog = 0; iAnalog < _analogsNb[iTime]; iAnalog++) {
            analogsCriteria[ind] = _analogsCriteria[iTime][iAnalog];
            analogsDates[ind] = _analogsDates[iTime][iAnalog];
            ind++;
        }
    }

    int indVal = 0;
    for (int iStat = 0; iStat < nStations; iStat++) {
        for (int iTime = 0; iTime < nLeadtime; iTime++) {
            for (int iAnalog = 0; iAnalog < _analogsNb[iTime]; iAnalog++) {
                analogsValuesRaw[indVal] = _analogsValuesRaw[iTime](iStat, iAnalog);
                analogsValuesNorm[indVal] = _analogsValuesNorm[iTime](iStat, iAnalog);
                indVal++;
            }
        }
    }

    // Write data
    ncFile.PutVarArray("target_dates", startLeadTime, countLeadTime, &_targetDates[0]);
    ncFile.PutVarArray("analogs_nb", startLeadTime, countLeadTime, &_analogsNb[0]);
    ncFile.PutVarArray("station_names", startStations, countStations, &_stationNames[0], nStations);
    ncFile.PutVarArray("station_official_ids", startStations, countStations, &_stationOfficialIds[0], nStations);
    ncFile.PutVarArray("station_ids", startStations, countStations, &_stationIds[0]);
    ncFile.PutVarArray("station_heights", startStations, countStations, &_stationHeights[0]);
    ncFile.PutVarArray("station_x_coords", startStations, countStations, &_stationXCoords(0));
    ncFile.PutVarArray("station_y_coords", startStations, countStations, &_stationYCoords(0));
    ncFile.PutVarArray("analog_criteria", startAnalogsTot, countAnalogsTot, &analogsCriteria[0]);
    ncFile.PutVarArray("analog_dates", startAnalogsTot, countAnalogsTot, &analogsDates[0]);
    ncFile.PutVarArray("analog_values_raw", startAnalogsStations, countAnalogsStations, &analogsValuesRaw[0]);
    ncFile.PutVarArray("analog_values_norm", startAnalogsStations, countAnalogsStations, &analogsValuesNorm[0]);
    if (_hasReferenceValues) {
        size_t startReferenceAxis[] = {0};
        size_t countReferenceAxis[] = {nReferenceAxis};
        size_t startReferenceValues[] = {0, 0};
        size_t countReferenceValues[] = {nStations, nReferenceAxis};
        ncFile.PutVarArray("reference_axis", startReferenceAxis, countReferenceAxis, &_referenceAxis(0));
        ncFile.PutVarArray("reference_values", startReferenceValues, countReferenceValues, &_referenceValues(0, 0));
    }
    ncFile.PutVarArray("predictor_dataset_ids_realtime", startPredictors, countPredictors,
                       &_predictorDatasetIdsOper[0], nPredictors);
    ncFile.PutVarArray("predictor_dataset_ids_archive", startPredictors, countPredictors,
                       &_predictorDatasetIdsArchive[0], nPredictors);
    ncFile.PutVarArray("predictor_data_ids_realtime", startPredictors, countPredictors, &_predictorDataIdsOper[0],
                       nPredictors);
    ncFile.PutVarArray("predictor_data_ids_archive", startPredictors, countPredictors, &_predictorDataIdsArchive[0],
                       nPredictors);
    ncFile.PutVarArray("predictor_levels", startPredictors, countPredictors, &_predictorLevels[0]);
    ncFile.PutVarArray("predictor_hours", startPredictors, countPredictors, &_predictorHours[0]);
    ncFile.PutVarArray("predictor_lon_min", startPredictors, countPredictors, &_predictorLonMin[0]);
    ncFile.PutVarArray("predictor_lon_max", startPredictors, countPredictors, &_predictorLonMax[0]);
    ncFile.PutVarArray("predictor_lat_min", startPredictors, countPredictors, &_predictorLatMin[0]);
    ncFile.PutVarArray("predictor_lat_max", startPredictors, countPredictors, &_predictorLatMax[0]);

    // Close:save new netCDF dataset
    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();

    return true;
}

bool asResultsForecast::Load() {
    if (!Exists()) return false;
    if (_currentStep != 0) return false;

    ThreadsManager().CritSectionNetCDF().Enter();

    int nLeadtime = 0;
    int nStations = 0;
    int nPredictors = 0;
    int versionMajor, versionMinor;
    vf analogsCriteria, analogsDates, analogsValuesRaw, analogsValuesNorm;

    asFileNetcdf ncFile(_filePath, asFileNetcdf::ReadOnly);

    try {
        // Open the NetCDF file
        if (!ncFile.Open()) return false;

        // Get global attributes
        versionMajor = ncFile.GetAttInt("version_major");
        versionMinor = ncFile.GetAttInt("version_minor");
        if (versionMajor == 0) {
            float version = ncFile.GetAttFloat("version");
            if (isnan(version)) {
                versionMajor = 1;
                versionMinor = 0;
            } else {
                versionMajor = std::floor(version);
                versionMinor = asRound(10 * (version - versionMajor));
            }
        }

        if (versionMajor > _fileVersionMajor ||
            (versionMajor >= _fileVersionMajor && versionMinor > _fileVersionMinor)) {
            wxLogError(
                _("The forecast file was made with more recent version of AtmoSwing (file version %d.%d). It cannot "
                  "be opened here."),
                versionMajor, versionMinor);
            return false;
        }

        if (versionMajor == 1 && versionMinor == 0) {
            wxLogWarning(_("The forecast file was made with an older version of AtmoSwing."));
            _predictandParameter = asPredictand::Precipitation;
            _predictandTemporalResolution = asPredictand::Daily;
            _predictandSpatialAggregation = asPredictand::Station;
            _predictandDatasetId = "MeteoSwiss-Rhone";
            _methodId = ncFile.GetAttString("modelName");
            _methodIdDisplay = ncFile.GetAttString("modelName");
            _specificTag = wxEmptyString;
            _specificTagDisplay = wxEmptyString;
            _description = wxEmptyString;
            _dateProcessed = ncFile.GetAttDouble("dateProcessed");
            _leadTimeOrigin = ncFile.GetAttDouble("leadTimeOrigin");
            _hasReferenceValues = true;
        } else {
            if (versionMajor == 1 && versionMinor <= 4) {
                _methodId = ncFile.GetAttString("model_name");
                _methodIdDisplay = ncFile.GetAttString("model_name");
                _specificTag = wxEmptyString;
                _specificTagDisplay = wxEmptyString;
                _description = wxEmptyString;
            } else {
                _methodId = ncFile.GetAttString("method_id");
                _methodIdDisplay = ncFile.GetAttString("method_id_display");
                _specificTag = ncFile.GetAttString("specific_tag");
                _specificTagDisplay = ncFile.GetAttString("specific_tag_display");
                _description = ncFile.GetAttString("description");
            }

            if (versionMajor == 1 && versionMinor <= 7) {
                _predictandParameter = asPredictand::Precipitation;
                _predictandTemporalResolution = asPredictand::Daily;
                if (ncFile.GetAttInt("predictand_spatial_aggregation") == 0) {
                    _predictandSpatialAggregation = asPredictand::Station;
                } else if (ncFile.GetAttInt("predictand_spatial_aggregation") == 1) {
                    _predictandSpatialAggregation = asPredictand::Groupment;
                } else {
                    wxLogError(_("The spatial aggregation could not be converted."));
                    return false;
                }
            } else {
                _predictandParameter = asPredictand::StringToParameterEnum(
                    ncFile.GetAttString("predictand_parameter"));
                _predictandTemporalResolution = asPredictand::StringToTemporalResolutionEnum(
                    ncFile.GetAttString("predictand_temporal_resolution"));
                _predictandSpatialAggregation = asPredictand::StringToSpatialAggregationEnum(
                    ncFile.GetAttString("predictand_spatial_aggregation"));
            }

            _predictandDatasetId = ncFile.GetAttString("predictand_dataset_id");

            if (versionMajor > 1 || (versionMajor == 1 && versionMinor >= 5)) {
                _predictandDatabase = ncFile.GetAttString("predictand_database");
                SetPredictandStationIds(ncFile.GetAttString("predictand_station_ids"));
            }

            _dateProcessed = ncFile.GetAttDouble("date_processed");
            _leadTimeOrigin = ncFile.GetAttDouble("lead_time_origin");
            _hasReferenceValues = false;
            if (ncFile.GetAttShort("has_reference_values") == 1) {
                _hasReferenceValues = true;
            }

            if (ncFile.HasAttribute("coordinate_system")) {
                _coordinateSystem = ncFile.GetAttString("coordinate_system");
            }
        }

        // Get the elements size
        int nAnalogsTot;
        if (versionMajor == 1 && versionMinor == 0) {
            nLeadtime = ncFile.GetDimLength("leadtime");
            nAnalogsTot = ncFile.GetDimLength("analogstot");
            nStations = ncFile.GetDimLength("stations");
        } else {
            nLeadtime = ncFile.GetDimLength("lead_time");
            nAnalogsTot = ncFile.GetDimLength("analogs_tot");
            nStations = ncFile.GetDimLength("stations");
        }
        if (versionMajor >= 3) {
            nPredictors = ncFile.GetDimLength("predictors");
        }

        // Get lead time data
        _targetDates.resize(nLeadtime);
        _analogsNb.resize(nLeadtime);
        _stationNames.resize(nStations);
        _stationOfficialIds.resize(nStations);
        _stationIds.resize(nStations);
        _stationHeights.resize(nStations);
        _stationXCoords.resize(nStations);
        _stationYCoords.resize(nStations);

        if (versionMajor == 1 && versionMinor == 0) {
            ncFile.GetVar("targetdates", &_targetDates[0]);
            ncFile.GetVar("analogsnb", &_analogsNb[0]);
            ncFile.GetVar("stationsnames", &_stationNames[0], nStations);
            ncFile.GetVar("stationsids", &_stationIds[0]);
            ncFile.GetVar("stationsheights", &_stationHeights[0]);
            ncFile.GetVar("loccoordu", &_stationXCoords[0]);
            ncFile.GetVar("loccoordv", &_stationYCoords[0]);
        } else if (versionMajor == 1 && versionMinor <= 3) {
            ncFile.GetVar("target_dates", &_targetDates[0]);
            ncFile.GetVar("analogs_nb", &_analogsNb[0]);
            ncFile.GetVar("stations_names", &_stationNames[0], nStations);
            ncFile.GetVar("stations_ids", &_stationIds[0]);
            ncFile.GetVar("stations_heights", &_stationHeights[0]);
            ncFile.GetVar("loc_coord_u", &_stationXCoords[0]);
            ncFile.GetVar("loc_coord_v", &_stationYCoords[0]);
        } else if (versionMajor == 1 && versionMinor <= 5) {
            ncFile.GetVar("target_dates", &_targetDates[0]);
            ncFile.GetVar("analogs_nb", &_analogsNb[0]);
            ncFile.GetVar("stations_names", &_stationNames[0], nStations);
            ncFile.GetVar("stations_ids", &_stationIds[0]);
            ncFile.GetVar("stations_heights", &_stationHeights[0]);
            ncFile.GetVar("loc_coord_x", &_stationXCoords[0]);
            ncFile.GetVar("loc_coord_y", &_stationYCoords[0]);
        } else {
            ncFile.GetVar("target_dates", &_targetDates[0]);
            ncFile.GetVar("analogs_nb", &_analogsNb[0]);
            ncFile.GetVar("station_names", &_stationNames[0], nStations);
            ncFile.GetVar("station_official_ids", &_stationOfficialIds[0], nStations);
            ncFile.GetVar("station_ids", &_stationIds[0]);
            ncFile.GetVar("station_heights", &_stationHeights[0]);
            ncFile.GetVar("station_x_coords", &_stationXCoords[0]);
            ncFile.GetVar("station_y_coords", &_stationYCoords[0]);
        }

        // Get return periods properties
        if (_hasReferenceValues) {
            if (versionMajor == 1 && versionMinor == 0) {
                int referenceAxisLength = ncFile.GetDimLength("returnperiods");
                _referenceAxis.resize(referenceAxisLength);
                ncFile.GetVar("returnperiods", &_referenceAxis[0]);
                size_t startReferenceValues[2] = {0, 0};
                size_t countReferenceValues[2] = {size_t(referenceAxisLength), size_t(nStations)};
                _referenceValues.resize(nStations, referenceAxisLength);
                ncFile.GetVarArray("dailyprecipitationsforreturnperiods", startReferenceValues, countReferenceValues,
                                   &_referenceValues(0, 0));
            } else {
                int referenceAxisLength = ncFile.GetDimLength("reference_axis");
                _referenceAxis.resize(referenceAxisLength);
                ncFile.GetVar("reference_axis", &_referenceAxis[0]);
                size_t startReferenceValues[2] = {0, 0};
                size_t countReferenceValues[2] = {0, 0};
                if (versionMajor == 1 && versionMinor == 1) {
                    countReferenceValues[0] = size_t(referenceAxisLength);
                    countReferenceValues[1] = size_t(nStations);
                } else {
                    countReferenceValues[0] = size_t(nStations);
                    countReferenceValues[1] = size_t(referenceAxisLength);
                }
                _referenceValues.resize(nStations, referenceAxisLength);
                ncFile.GetVarArray("reference_values", startReferenceValues, countReferenceValues,
                                   &_referenceValues(0, 0));
            }
        }

        // Get predictors info
        if (versionMajor >= 3) {
            _predictorDatasetIdsOper.resize(nPredictors);
            _predictorDatasetIdsArchive.resize(nPredictors);
            _predictorDataIdsOper.resize(nPredictors);
            _predictorDataIdsArchive.resize(nPredictors);
            _predictorLevels.resize(nPredictors);
            _predictorHours.resize(nPredictors);
            _predictorLonMin.resize(nPredictors);
            _predictorLonMax.resize(nPredictors);
            _predictorLatMin.resize(nPredictors);
            _predictorLatMax.resize(nPredictors);

            ncFile.GetVar("predictor_dataset_ids_realtime", &_predictorDatasetIdsOper[0], nPredictors);
            ncFile.GetVar("predictor_dataset_ids_archive", &_predictorDatasetIdsArchive[0], nPredictors);
            ncFile.GetVar("predictor_data_ids_realtime", &_predictorDataIdsOper[0], nPredictors);
            ncFile.GetVar("predictor_data_ids_archive", &_predictorDataIdsArchive[0], nPredictors);
            ncFile.GetVar("predictor_levels", &_predictorLevels[0]);
            ncFile.GetVar("predictor_hours", &_predictorHours[0]);
            ncFile.GetVar("predictor_lon_min", &_predictorLonMin[0]);
            ncFile.GetVar("predictor_lon_max", &_predictorLonMax[0]);
            ncFile.GetVar("predictor_lat_min", &_predictorLatMin[0]);
            ncFile.GetVar("predictor_lat_max", &_predictorLatMax[0]);
        }

        // Create vectors for matrices data
        analogsCriteria.resize(nAnalogsTot);
        analogsDates.resize(nAnalogsTot);
        analogsValuesRaw.resize(nAnalogsTot * nStations);
        analogsValuesNorm.resize(nAnalogsTot * nStations);

        // Get data
        size_t indexStart1D[] = {0};
        size_t indexCount1D[] = {size_t(nAnalogsTot)};
        size_t indexStart2D[] = {0, 0};
        size_t indexCount2D[] = {size_t(nStations), size_t(nAnalogsTot)};
        if (versionMajor == 1 && versionMinor == 0) {
            ncFile.GetVarArray("analogscriteria", indexStart1D, indexCount1D, &analogsCriteria[0]);
            ncFile.GetVarArray("analogsdates", indexStart1D, indexCount1D, &analogsDates[0]);
            ncFile.GetVarArray("analogsvaluesgross", indexStart2D, indexCount2D, &analogsValuesRaw[0]);
        } else if (versionMajor == 1 && versionMinor <= 5) {
            ncFile.GetVarArray("analogs_criteria", indexStart1D, indexCount1D, &analogsCriteria[0]);
            ncFile.GetVarArray("analogs_dates", indexStart1D, indexCount1D, &analogsDates[0]);
            ncFile.GetVarArray("analogs_values_gross", indexStart2D, indexCount2D, &analogsValuesRaw[0]);
        } else if ((versionMajor < 2) || (versionMajor == 2 && versionMinor == 0)) {
            ncFile.GetVarArray("analog_criteria", indexStart1D, indexCount1D, &analogsCriteria[0]);
            ncFile.GetVarArray("analog_dates", indexStart1D, indexCount1D, &analogsDates[0]);
            ncFile.GetVarArray("analog_values", indexStart2D, indexCount2D, &analogsValuesRaw[0]);
        } else {
            ncFile.GetVarArray("analog_criteria", indexStart1D, indexCount1D, &analogsCriteria[0]);
            ncFile.GetVarArray("analog_dates", indexStart1D, indexCount1D, &analogsDates[0]);
            ncFile.GetVarArray("analog_values_raw", indexStart2D, indexCount2D, &analogsValuesRaw[0]);
            ncFile.GetVarArray("analog_values_norm", indexStart2D, indexCount2D, &analogsValuesNorm[0]);
        }

        ncFile.Close();

    } catch (runtime_error& e) {
        wxString msg(e.what(), wxConvUTF8);
        wxLogError(_("Exception caught: %s"), msg);

        ncFile.ForceClose();
        ThreadsManager().CritSectionNetCDF().Leave();

        return false;
    }

    ThreadsManager().CritSectionNetCDF().Leave();

    // Set data into the matrices
    int ind = 0;
    for (int iTime = 0; iTime < (int)nLeadtime; iTime++) {
        a1f analogsCriteriaLeadTime(_analogsNb[iTime]);
        a1f analogsDatesLeadTime(_analogsNb[iTime]);

        for (int iAnalog = 0; iAnalog < _analogsNb[iTime]; iAnalog++) {
            analogsCriteriaLeadTime(iAnalog) = analogsCriteria[ind];
            analogsDatesLeadTime(iAnalog) = analogsDates[ind];
            ind++;
        }

        _analogsCriteria.push_back(analogsCriteriaLeadTime);
        _analogsDates.push_back(analogsDatesLeadTime);
    }

    int indVal = 0;
    if (versionMajor == 1 && versionMinor == 0) {
        for (int iTime = 0; iTime < nLeadtime; iTime++) {
            a2f analogsValuesRawLeadTime(nStations, _analogsNb[iTime]);
            a2f analogsValuesNormLeadTime(nStations, _analogsNb[iTime]);

            for (int iAnalog = 0; iAnalog < _analogsNb[iTime]; iAnalog++) {
                for (int iStat = 0; iStat < nStations; iStat++) {
                    analogsValuesRawLeadTime(iStat, iAnalog) = analogsValuesRaw[indVal];
                    analogsValuesNormLeadTime(iStat, iAnalog) = analogsValuesNorm[indVal];
                    indVal++;
                }
            }

            _analogsValuesRaw.push_back(analogsValuesRawLeadTime);
            _analogsValuesNorm.push_back(analogsValuesNormLeadTime);
        }
    } else {
        // Create containers
        for (int iTime = 0; iTime < nLeadtime; iTime++) {
            a2f analogsValuesLeadTime(nStations, _analogsNb[iTime]);
            analogsValuesLeadTime.fill(NAN);
            _analogsValuesRaw.push_back(analogsValuesLeadTime);
            _analogsValuesNorm.push_back(analogsValuesLeadTime);
        }

        for (int iStat = 0; iStat < nStations; iStat++) {
            for (int iTime = 0; iTime < nLeadtime; iTime++) {
                for (int iAnalog = 0; iAnalog < _analogsNb[iTime]; iAnalog++) {
                    _analogsValuesRaw[iTime](iStat, iAnalog) = analogsValuesRaw[indVal];
                    _analogsValuesNorm[iTime](iStat, iAnalog) = analogsValuesNorm[indVal];
                    indVal++;
                }
            }
        }
    }

    wxASSERT(!_filePath.IsEmpty());
    wxASSERT(!_predictandDatasetId.IsEmpty());
    wxASSERT(_targetDates.size() > 0);
    wxASSERT(_analogsNb.size() > 0);
    wxASSERT(_stationIds.size() > 0);
    wxASSERT(!_stationNames.empty());
    wxASSERT(_stationHeights.size() > 0);
    wxASSERT(!_analogsCriteria.empty());
    wxASSERT(!_analogsDates.empty());
    wxASSERT(!_analogsValuesRaw.empty());
    wxASSERT(_stationXCoords.size() > 0);
    wxASSERT(_stationYCoords.size() > 0);
    if (_hasReferenceValues) {
        wxASSERT(_referenceAxis.size() > 0);
        wxASSERT(_referenceValues.cols() > 0);
        wxASSERT(_referenceValues.rows() > 0);
    }

    return true;
}

wxArrayString asResultsForecast::GetStationNamesWxArray() const {
    wxArrayString stationsNames;
    for (const auto& stationName : _stationNames) {
        stationsNames.Add(stationName);
    }
    return stationsNames;
}

wxArrayString asResultsForecast::GetStationNamesAndHeightsWxArray() const {
    wxArrayString stationsNames;
    for (int i = 0; i < _stationNames.size(); i++) {
        wxString label;
        if (std::isfinite(_stationHeights[i]) && _stationHeights[i] != 0 && _stationHeights[i] != -1) {
            label = asStrF("%s (%4.0fm)", _stationNames[i], _stationHeights[i]);
        } else {
            label = asStrF("%s", _stationNames[i]);
        }
        stationsNames.Add(label);
    }
    return stationsNames;
}

void asResultsForecast::LimitDataToHours(int hours) {
    LimitDataToNbTimeSteps(1 + int(hours / GetForecastTimeStepHours()));
}

void asResultsForecast::LimitDataToDays(int days) {
    LimitDataToNbTimeSteps(days);
}

void asResultsForecast::LimitDataToNbTimeSteps(int length) {
    if (length >= _targetDates.size()) return;

    _targetDates = _targetDates.head(length);
    _analogsNb = _analogsNb.head(length);

    _analogsDates = va1f(_analogsDates.begin(), _analogsDates.begin() + length);
    _analogsCriteria = va1f(_analogsCriteria.begin(), _analogsCriteria.begin() + length);
    _analogsValuesRaw = va2f(_analogsValuesRaw.begin(), _analogsValuesRaw.begin() + length);
    _analogsValuesNorm = va2f(_analogsValuesNorm.begin(), _analogsValuesNorm.begin() + length);
}

wxString asResultsForecast::GetDateFormatting() const {
    wxString format = "DD.MM.YYYY";
    if (GetPredictandTemporalResolution() != asPredictand::Daily) {
        format = "DD.MM.YYYY hh";
    }

    return format;
}

double asResultsForecast::GetForecastTimeStepHours() const {
    if (GetPredictandTemporalResolution() == asPredictand::Hourly) return 1;
    if (GetPredictandTemporalResolution() == asPredictand::SixHourly) return 6;
    if (GetPredictandTemporalResolution() == asPredictand::Daily) return 24;

    return 24;
}

bool asResultsForecast::IsSubDaily() const {
    return GetForecastTimeStepHours() < 24;
}

wxArrayString asResultsForecast::GetTargetDatesWxArray() const {
    wxArrayString dates;
    wxString format = GetDateFormatting();
    for (float date : _targetDates) {
        dates.Add(asTime::GetStringTime(date, format));
    }

    return dates;
}

wxString asResultsForecast::GetStationNameAndHeight(int iStat) const {
    wxString stationName;
    if (std::isfinite(_stationHeights[iStat]) && _stationHeights[iStat] != 0 && _stationHeights[iStat] != -1) {
        stationName = asStrF("%s (%4.0fm)", _stationNames[iStat], _stationHeights[iStat]);
    } else {
        stationName = asStrF("%s", _stationNames[iStat]);
    }
    return stationName;
}

wxString asResultsForecast::GetPredictandStationIdsString() const {
    wxString ids;

    for (int i = 0; i < (int)_predictandStationIds.size(); i++) {
        ids << _predictandStationIds[i];

        if (i < (int)_predictandStationIds.size() - 1) {
            ids.Append(",");
        }
    }

    return ids;
}

void asResultsForecast::SetPredictandStationIds(const wxString& val) {
    wxStringTokenizer tokenizer(val, ":,; ");
    while (tokenizer.HasMoreTokens()) {
        wxString token = tokenizer.GetNextToken();
        long stationId;
        if (token.ToLong(&stationId)) {
            _predictandStationIds.push_back(int(stationId));
        }
    }
}

bool asResultsForecast::IsCompatibleWith(asResultsForecast* otherForecast) const {
    bool compatible = true;

    if (!_methodId.IsSameAs(otherForecast->GetMethodId(), false)) compatible = false;
    if (_predictandParameter != otherForecast->GetPredictandParameter()) compatible = false;
    if (_predictandTemporalResolution != otherForecast->GetPredictandTemporalResolution()) compatible = false;
    if (_predictandSpatialAggregation != otherForecast->GetPredictandSpatialAggregation()) compatible = false;
    if (!_predictandDatasetId.IsSameAs(otherForecast->GetPredictandDatasetId(), false)) compatible = false;
    if (!_predictandDatabase.IsSameAs(otherForecast->GetPredictandDatabase(), false)) compatible = false;
    if (_hasReferenceValues != otherForecast->HasReferenceValues()) compatible = false;
    if (_leadTimeOrigin != otherForecast->GetLeadTimeOrigin()) compatible = false;

    a1f targetDates = otherForecast->GetTargetDates();
    if (_targetDates.size() != targetDates.size()) {
        compatible = false;
    } else {
        for (int i = 0; i < _targetDates.size(); i++) {
            if (_targetDates[i] != targetDates[i]) compatible = false;
        }
    }

    a1i stationsIds = otherForecast->GetStationIds();
    if (_stationIds.size() != stationsIds.size()) {
        compatible = false;
    } else {
        for (int i = 0; i < _stationIds.size(); i++) {
            if (_stationIds[i] != stationsIds[i]) compatible = false;
        }
    }

    a1f referenceAxis = otherForecast->GetReferenceAxis();
    if (_referenceAxis.size() != referenceAxis.size()) {
        compatible = false;
    } else {
        for (int i = 0; i < _referenceAxis.size(); i++) {
            if (!isnan(_referenceAxis[i]) && _referenceAxis[i] != referenceAxis[i]) {
                compatible = false;
            }
        }
    }

    if (!compatible) {
        wxLogError(_("The forecasts \"%s\" and \"%s\" are not compatible"), _specificTagDisplay,
                   otherForecast->GetSpecificTagDisplay());
        return false;
    }

    return true;
}

bool asResultsForecast::IsSameAs(asResultsForecast* otherForecast) const {
    if (!IsCompatibleWith(otherForecast)) return false;

    if (!_specificTag.IsSameAs(otherForecast->GetSpecificTag(), false)) return false;

    vi predictandStationIds = otherForecast->GetPredictandStationIds();
    if (_predictandStationIds.size() != predictandStationIds.size()) {
        return false;
    }

    for (int i = 0; i < _predictandStationIds.size(); i++) {
        if (_predictandStationIds[i] != predictandStationIds[i]) return false;
    }

    a1f targetDates = otherForecast->GetTargetDates();
    if (_targetDates.size() != targetDates.size()) {
        return false;
    }

    for (int i = 0; i < _targetDates.size(); i++) {
        if (_targetDates[i] != targetDates[i]) return false;
        if (_analogsNb[i] != otherForecast->GetAnalogsNumber(i)) return false;
        if (_analogsCriteria[i].size() != otherForecast->GetAnalogsCriteria(i).size()) return false;
        if (_analogsDates[i].size() != otherForecast->GetAnalogsDates(i).size()) return false;
        if (_analogsValuesRaw[i].size() != otherForecast->GetAnalogsValuesRaw(i).size()) return false;

        for (int j = 0; j < _analogsCriteria[i].size(); j++) {
            if (_analogsCriteria[i][j] != otherForecast->GetAnalogsCriteria(i)[j]) return false;
        }
        for (int j = 0; j < _analogsDates[i].size(); j++) {
            if (_analogsDates[i][j] != otherForecast->GetAnalogsDates(i)[j]) return false;
        }
        for (int j = 0; j < _analogsDates[i].size(); j++) {
            if (_analogsDates[i][j] != otherForecast->GetAnalogsDates(i)[j]) return false;
        }
        for (int j = 0; j < _analogsValuesRaw[i].size(); j++) {
            if (_analogsValuesRaw[i].rows() != otherForecast->GetAnalogsValuesRaw(i).rows()) return false;
            if (_analogsValuesRaw[i].cols() != otherForecast->GetAnalogsValuesRaw(i).cols()) return false;
        }
    }

    return true;
}

bool asResultsForecast::IsSpecificForStationId(int stationId) const {
    for (int predictandStationId : _predictandStationIds) {
        if (predictandStationId == stationId) {
            return true;
        }
    }
    return false;
}

int asResultsForecast::GetStationRowFromId(int stationId) const {
    for (int i = 0; i < _stationIds.size(); i++) {
        if (_stationIds[i] == stationId) {
            return i;
        }
    }

    wxFAIL;
    wxLogError(_("The station ID %d was not found in the forecast results."), stationId);
    return -1;
}

Coo asResultsForecast::GetStationsMeanCoordinates() {
    vd xs, ys;
    for (int id : _predictandStationIds) {
        int i = GetStationRowFromId(id);
        wxASSERT(i >= 0);
        xs.push_back(_stationXCoords[i]);
        ys.push_back(_stationYCoords[i]);
    }

    return {std::reduce(xs.begin(), xs.end()) / xs.size(), std::reduce(ys.begin(), ys.end()) / ys.size()};
}
