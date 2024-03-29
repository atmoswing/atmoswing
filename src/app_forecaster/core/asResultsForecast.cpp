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
      m_predictandParameter(asPredictand::Precipitation),
      m_predictandTemporalResolution(asPredictand::Daily),
      m_predictandSpatialAggregation(asPredictand::Station),
      m_hasReferenceValues(false),
      m_leadTimeOrigin(0.0) {}

void asResultsForecast::Init(asParametersForecast& params, double leadTimeOrigin) {
    // Resize to 0 to avoid keeping old results
    m_targetDates.resize(0);
    m_stationNames.resize(0);
    m_stationOfficialIds.resize(0);
    m_stationIds.resize(0);
    m_stationHeights.resize(0);
    m_analogsNb.resize(0);
    m_analogsCriteria.resize(0);
    m_analogsDates.resize(0);
    m_analogsValuesRaw.resize(0);
    m_analogsValuesNorm.resize(0);
    m_stationXCoords.resize(0);
    m_stationYCoords.resize(0);
    m_referenceAxis.resize(0);
    m_referenceValues.resize(0, 0);
    m_predictorDatasetIdsOper.resize(0);
    m_predictorDatasetIdsArchive.resize(0);
    m_predictorDataIdsOper.resize(0);
    m_predictorDataIdsArchive.resize(0);
    m_predictorLevels.resize(0);
    m_predictorHours.resize(0);
    m_predictorLonMin.resize(0);
    m_predictorLonMax.resize(0);
    m_predictorLatMin.resize(0);
    m_predictorLatMax.resize(0);

    m_methodId = params.GetMethodId();
    m_methodIdDisplay = params.GetMethodIdDisplay();
    m_specificTag = params.GetSpecificTag();
    m_specificTagDisplay = params.GetSpecificTagDisplay();
    m_description = params.GetDescription();
    m_predictandDatabase = params.GetPredictandDatabase();
    m_predictandStationIds = params.GetPredictandStationIds();

    m_leadTimeOrigin = leadTimeOrigin;
    m_dateProcessed = asTime::NowMJD(asUTM);

    // Set the analogs number
    m_analogsNb.resize(params.GetLeadTimeNb());
    for (int i = 0; i < params.GetLeadTimeNb(); i++) {
        m_analogsNb[i] = params.GetAnalogsNumberLeadTime(m_currentStep, i);
    }

    BuildFileName();
}

void asResultsForecast::BuildFileName() {
    wxASSERT(!m_forecastsDir.IsEmpty());

    if (m_methodId.IsEmpty() || m_specificTag.IsEmpty()) {
        wxLogError(_("The provided ID or the tag is empty, which isn't allowed !"));
    }

    // Base directory
    m_filePath = m_forecastsDir;
    m_filePath.Append(DS);
    if (!m_subFolder.IsEmpty()) {
        m_filePath.Append(DS);
        m_filePath.Append(m_subFolder);
    }

    // Directory
    wxString dirstructure = "YYYY";
    dirstructure.Append(DS);
    dirstructure.Append("MM");
    dirstructure.Append(DS);
    dirstructure.Append("DD");
    wxString directory = asTime::GetStringTime(m_leadTimeOrigin, dirstructure);
    m_filePath.Append(directory);
    m_filePath.Append(DS);

    // Filename
    wxString forecastname = m_methodId + '.' + m_specificTag;
    wxString nowstr = asTime::GetStringTime(m_leadTimeOrigin, "YYYY-MM-DD_hh");
    wxString ext = "nc";
    wxString filename = asStrF("%s.%s.%s", nowstr, forecastname, ext);
    m_filePath.Append(filename);
}

bool asResultsForecast::Save() {
    wxASSERT(!m_filePath.IsEmpty());
    wxASSERT(m_targetDates.size() > 0);
    wxASSERT(m_analogsNb.size() > 0);
    wxASSERT(!m_stationNames.empty());
    wxASSERT(!m_stationOfficialIds.empty());
    wxASSERT(m_stationHeights.size() > 0);
    wxASSERT(m_stationIds.size() > 0);
    wxASSERT(!m_analogsCriteria.empty());
    wxASSERT(!m_analogsDates.empty());
    wxASSERT(!m_analogsValuesRaw.empty());
    wxASSERT(!m_analogsValuesNorm.empty());
    wxASSERT(m_stationXCoords.size() > 0);
    wxASSERT(m_stationYCoords.size() > 0);

    if (m_hasReferenceValues) {
        wxASSERT(m_referenceAxis.size() > 0);
        wxASSERT(m_referenceValues.cols() > 0);
        wxASSERT(m_referenceValues.rows() > 0);
    }

    // Get the elements size
    size_t nLeadtime = m_targetDates.size();
    size_t nAnalogsTot = m_analogsNb.sum();
    size_t nStations = m_stationIds.size();
    size_t nReferenceAxis = m_referenceAxis.size();
    size_t nPredictors = m_predictorDatasetIdsOper.size();

    ThreadsManager().CritSectionNetCDF().Enter();

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(m_filePath, asFileNetcdf::Replace);
    if (!ncFile.Open()) {
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Set general attributes
    ncFile.PutAtt("version_major", &m_fileVersionMajor);
    ncFile.PutAtt("version_minor", &m_fileVersionMinor);
    ncFile.PutAtt("predictand_parameter", asPredictand::ParameterEnumToString(m_predictandParameter));
    ncFile.PutAtt("predictand_temporal_resolution",
                  asPredictand::TemporalResolutionEnumToString(m_predictandTemporalResolution));
    ncFile.PutAtt("predictand_spatial_aggregation",
                  asPredictand::SpatialAggregationEnumToString(m_predictandSpatialAggregation));
    ncFile.PutAtt("predictand_dataset_id", m_predictandDatasetId);
    ncFile.PutAtt("predictand_database", m_predictandDatabase);
    ncFile.PutAtt("predictand_station_ids", GetPredictandStationIdsString());
    ncFile.PutAtt("method_id", m_methodId);
    ncFile.PutAtt("method_id_display", m_methodIdDisplay);
    ncFile.PutAtt("specific_tag", m_specificTag);
    ncFile.PutAtt("specific_tag_display", m_specificTagDisplay);
    ncFile.PutAtt("description", m_description);
    ncFile.PutAtt("date_processed", &m_dateProcessed);
    ncFile.PutAtt("lead_time_origin", &m_leadTimeOrigin);
    ncFile.PutAtt("coordinate_system", m_coordinateSystem);
    short hasReferenceValues = 0;
    if (m_hasReferenceValues) {
        hasReferenceValues = 1;
    }
    ncFile.PutAtt("has_reference_values", &hasReferenceValues);

    // Define dimensions. No unlimited dimension.
    ncFile.DefDim("lead_time", nLeadtime);
    ncFile.DefDim("analogs_tot", nAnalogsTot);
    ncFile.DefDim("stations", nStations);
    if (m_hasReferenceValues) {
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
    if (m_hasReferenceValues) {
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
    if (m_hasReferenceValues) {
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
    if (m_hasReferenceValues) {
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
        for (int iAnalog = 0; iAnalog < m_analogsNb[iTime]; iAnalog++) {
            analogsCriteria[ind] = m_analogsCriteria[iTime][iAnalog];
            analogsDates[ind] = m_analogsDates[iTime][iAnalog];
            ind++;
        }
    }

    int indVal = 0;
    for (int iStat = 0; iStat < nStations; iStat++) {
        for (int iTime = 0; iTime < nLeadtime; iTime++) {
            for (int iAnalog = 0; iAnalog < m_analogsNb[iTime]; iAnalog++) {
                analogsValuesRaw[indVal] = m_analogsValuesRaw[iTime](iStat, iAnalog);
                analogsValuesNorm[indVal] = m_analogsValuesNorm[iTime](iStat, iAnalog);
                indVal++;
            }
        }
    }

    // Write data
    ncFile.PutVarArray("target_dates", startLeadTime, countLeadTime, &m_targetDates[0]);
    ncFile.PutVarArray("analogs_nb", startLeadTime, countLeadTime, &m_analogsNb[0]);
    ncFile.PutVarArray("station_names", startStations, countStations, &m_stationNames[0], nStations);
    ncFile.PutVarArray("station_official_ids", startStations, countStations, &m_stationOfficialIds[0], nStations);
    ncFile.PutVarArray("station_ids", startStations, countStations, &m_stationIds[0]);
    ncFile.PutVarArray("station_heights", startStations, countStations, &m_stationHeights[0]);
    ncFile.PutVarArray("station_x_coords", startStations, countStations, &m_stationXCoords(0));
    ncFile.PutVarArray("station_y_coords", startStations, countStations, &m_stationYCoords(0));
    ncFile.PutVarArray("analog_criteria", startAnalogsTot, countAnalogsTot, &analogsCriteria[0]);
    ncFile.PutVarArray("analog_dates", startAnalogsTot, countAnalogsTot, &analogsDates[0]);
    ncFile.PutVarArray("analog_values_raw", startAnalogsStations, countAnalogsStations, &analogsValuesRaw[0]);
    ncFile.PutVarArray("analog_values_norm", startAnalogsStations, countAnalogsStations, &analogsValuesNorm[0]);
    if (m_hasReferenceValues) {
        size_t startReferenceAxis[] = {0};
        size_t countReferenceAxis[] = {nReferenceAxis};
        size_t startReferenceValues[] = {0, 0};
        size_t countReferenceValues[] = {nStations, nReferenceAxis};
        ncFile.PutVarArray("reference_axis", startReferenceAxis, countReferenceAxis, &m_referenceAxis(0));
        ncFile.PutVarArray("reference_values", startReferenceValues, countReferenceValues, &m_referenceValues(0, 0));
    }
    ncFile.PutVarArray("predictor_dataset_ids_realtime", startPredictors, countPredictors,
                       &m_predictorDatasetIdsOper[0], nPredictors);
    ncFile.PutVarArray("predictor_dataset_ids_archive", startPredictors, countPredictors,
                       &m_predictorDatasetIdsArchive[0], nPredictors);
    ncFile.PutVarArray("predictor_data_ids_realtime", startPredictors, countPredictors, &m_predictorDataIdsOper[0],
                       nPredictors);
    ncFile.PutVarArray("predictor_data_ids_archive", startPredictors, countPredictors, &m_predictorDataIdsArchive[0],
                       nPredictors);
    ncFile.PutVarArray("predictor_levels", startPredictors, countPredictors, &m_predictorLevels[0]);
    ncFile.PutVarArray("predictor_hours", startPredictors, countPredictors, &m_predictorHours[0]);
    ncFile.PutVarArray("predictor_lon_min", startPredictors, countPredictors, &m_predictorLonMin[0]);
    ncFile.PutVarArray("predictor_lon_max", startPredictors, countPredictors, &m_predictorLonMax[0]);
    ncFile.PutVarArray("predictor_lat_min", startPredictors, countPredictors, &m_predictorLatMin[0]);
    ncFile.PutVarArray("predictor_lat_max", startPredictors, countPredictors, &m_predictorLatMax[0]);

    // Close:save new netCDF dataset
    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();

    return true;
}

bool asResultsForecast::Load() {
    if (!Exists()) return false;
    if (m_currentStep != 0) return false;

    ThreadsManager().CritSectionNetCDF().Enter();

    int nLeadtime = 0;
    int nStations = 0;
    int nPredictors = 0;
    int versionMajor, versionMinor;
    vf analogsCriteria, analogsDates, analogsValuesRaw, analogsValuesNorm;

    asFileNetcdf ncFile(m_filePath, asFileNetcdf::ReadOnly);

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

        if (versionMajor > m_fileVersionMajor ||
            (versionMajor >= m_fileVersionMajor && versionMinor > m_fileVersionMinor)) {
            wxLogError(
                _("The forecast file was made with more recent version of AtmoSwing (file version %d.%d). It cannot "
                  "be opened here."),
                versionMajor, versionMinor);
            return false;
        }

        if (versionMajor == 1 && versionMinor == 0) {
            wxLogWarning(_("The forecast file was made with an older version of AtmoSwing."));
            m_predictandParameter = asPredictand::Precipitation;
            m_predictandTemporalResolution = asPredictand::Daily;
            m_predictandSpatialAggregation = asPredictand::Station;
            m_predictandDatasetId = "MeteoSwiss-Rhone";
            m_methodId = ncFile.GetAttString("modelName");
            m_methodIdDisplay = ncFile.GetAttString("modelName");
            m_specificTag = wxEmptyString;
            m_specificTagDisplay = wxEmptyString;
            m_description = wxEmptyString;
            m_dateProcessed = ncFile.GetAttDouble("dateProcessed");
            m_leadTimeOrigin = ncFile.GetAttDouble("leadTimeOrigin");
            m_hasReferenceValues = true;
        } else {
            if (versionMajor == 1 && versionMinor <= 4) {
                m_methodId = ncFile.GetAttString("model_name");
                m_methodIdDisplay = ncFile.GetAttString("model_name");
                m_specificTag = wxEmptyString;
                m_specificTagDisplay = wxEmptyString;
                m_description = wxEmptyString;
            } else {
                m_methodId = ncFile.GetAttString("method_id");
                m_methodIdDisplay = ncFile.GetAttString("method_id_display");
                m_specificTag = ncFile.GetAttString("specific_tag");
                m_specificTagDisplay = ncFile.GetAttString("specific_tag_display");
                m_description = ncFile.GetAttString("description");
            }

            if (versionMajor == 1 && versionMinor <= 7) {
                m_predictandParameter = asPredictand::Precipitation;
                m_predictandTemporalResolution = asPredictand::Daily;
                if (ncFile.GetAttInt("predictand_spatial_aggregation") == 0) {
                    m_predictandSpatialAggregation = asPredictand::Station;
                } else if (ncFile.GetAttInt("predictand_spatial_aggregation") == 1) {
                    m_predictandSpatialAggregation = asPredictand::Groupment;
                } else {
                    wxLogError(_("The spatial aggregation could not be converted."));
                    return false;
                }
            } else {
                m_predictandParameter = asPredictand::StringToParameterEnum(
                    ncFile.GetAttString("predictand_parameter"));
                m_predictandTemporalResolution = asPredictand::StringToTemporalResolutionEnum(
                    ncFile.GetAttString("predictand_temporal_resolution"));
                m_predictandSpatialAggregation = asPredictand::StringToSpatialAggregationEnum(
                    ncFile.GetAttString("predictand_spatial_aggregation"));
            }

            m_predictandDatasetId = ncFile.GetAttString("predictand_dataset_id");

            if (versionMajor > 1 || (versionMajor == 1 && versionMinor >= 5)) {
                m_predictandDatabase = ncFile.GetAttString("predictand_database");
                SetPredictandStationIds(ncFile.GetAttString("predictand_station_ids"));
            }

            m_dateProcessed = ncFile.GetAttDouble("date_processed");
            m_leadTimeOrigin = ncFile.GetAttDouble("lead_time_origin");
            m_hasReferenceValues = false;
            if (ncFile.GetAttShort("has_reference_values") == 1) {
                m_hasReferenceValues = true;
            }

            if (ncFile.HasAttribute("coordinate_system")) {
                m_coordinateSystem = ncFile.GetAttString("coordinate_system");
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
        m_targetDates.resize(nLeadtime);
        m_analogsNb.resize(nLeadtime);
        m_stationNames.resize(nStations);
        m_stationOfficialIds.resize(nStations);
        m_stationIds.resize(nStations);
        m_stationHeights.resize(nStations);
        m_stationXCoords.resize(nStations);
        m_stationYCoords.resize(nStations);

        if (versionMajor == 1 && versionMinor == 0) {
            ncFile.GetVar("targetdates", &m_targetDates[0]);
            ncFile.GetVar("analogsnb", &m_analogsNb[0]);
            ncFile.GetVar("stationsnames", &m_stationNames[0], nStations);
            ncFile.GetVar("stationsids", &m_stationIds[0]);
            ncFile.GetVar("stationsheights", &m_stationHeights[0]);
            ncFile.GetVar("loccoordu", &m_stationXCoords[0]);
            ncFile.GetVar("loccoordv", &m_stationYCoords[0]);
        } else if (versionMajor == 1 && versionMinor <= 3) {
            ncFile.GetVar("target_dates", &m_targetDates[0]);
            ncFile.GetVar("analogs_nb", &m_analogsNb[0]);
            ncFile.GetVar("stations_names", &m_stationNames[0], nStations);
            ncFile.GetVar("stations_ids", &m_stationIds[0]);
            ncFile.GetVar("stations_heights", &m_stationHeights[0]);
            ncFile.GetVar("loc_coord_u", &m_stationXCoords[0]);
            ncFile.GetVar("loc_coord_v", &m_stationYCoords[0]);
        } else if (versionMajor == 1 && versionMinor <= 5) {
            ncFile.GetVar("target_dates", &m_targetDates[0]);
            ncFile.GetVar("analogs_nb", &m_analogsNb[0]);
            ncFile.GetVar("stations_names", &m_stationNames[0], nStations);
            ncFile.GetVar("stations_ids", &m_stationIds[0]);
            ncFile.GetVar("stations_heights", &m_stationHeights[0]);
            ncFile.GetVar("loc_coord_x", &m_stationXCoords[0]);
            ncFile.GetVar("loc_coord_y", &m_stationYCoords[0]);
        } else {
            ncFile.GetVar("target_dates", &m_targetDates[0]);
            ncFile.GetVar("analogs_nb", &m_analogsNb[0]);
            ncFile.GetVar("station_names", &m_stationNames[0], nStations);
            ncFile.GetVar("station_official_ids", &m_stationOfficialIds[0], nStations);
            ncFile.GetVar("station_ids", &m_stationIds[0]);
            ncFile.GetVar("station_heights", &m_stationHeights[0]);
            ncFile.GetVar("station_x_coords", &m_stationXCoords[0]);
            ncFile.GetVar("station_y_coords", &m_stationYCoords[0]);
        }

        // Get return periods properties
        if (m_hasReferenceValues) {
            if (versionMajor == 1 && versionMinor == 0) {
                int referenceAxisLength = ncFile.GetDimLength("returnperiods");
                m_referenceAxis.resize(referenceAxisLength);
                ncFile.GetVar("returnperiods", &m_referenceAxis[0]);
                size_t startReferenceValues[2] = {0, 0};
                size_t countReferenceValues[2] = {size_t(referenceAxisLength), size_t(nStations)};
                m_referenceValues.resize(nStations, referenceAxisLength);
                ncFile.GetVarArray("dailyprecipitationsforreturnperiods", startReferenceValues, countReferenceValues,
                                   &m_referenceValues(0, 0));
            } else {
                int referenceAxisLength = ncFile.GetDimLength("reference_axis");
                m_referenceAxis.resize(referenceAxisLength);
                ncFile.GetVar("reference_axis", &m_referenceAxis[0]);
                size_t startReferenceValues[2] = {0, 0};
                size_t countReferenceValues[2] = {0, 0};
                if (versionMajor == 1 && versionMinor == 1) {
                    countReferenceValues[0] = size_t(referenceAxisLength);
                    countReferenceValues[1] = size_t(nStations);
                } else {
                    countReferenceValues[0] = size_t(nStations);
                    countReferenceValues[1] = size_t(referenceAxisLength);
                }
                m_referenceValues.resize(nStations, referenceAxisLength);
                ncFile.GetVarArray("reference_values", startReferenceValues, countReferenceValues,
                                   &m_referenceValues(0, 0));
            }
        }

        // Get predictors info
        if (versionMajor >= 3) {
            m_predictorDatasetIdsOper.resize(nPredictors);
            m_predictorDatasetIdsArchive.resize(nPredictors);
            m_predictorDataIdsOper.resize(nPredictors);
            m_predictorDataIdsArchive.resize(nPredictors);
            m_predictorLevels.resize(nPredictors);
            m_predictorHours.resize(nPredictors);
            m_predictorLonMin.resize(nPredictors);
            m_predictorLonMax.resize(nPredictors);
            m_predictorLatMin.resize(nPredictors);
            m_predictorLatMax.resize(nPredictors);

            ncFile.GetVar("predictor_dataset_ids_realtime", &m_predictorDatasetIdsOper[0], nPredictors);
            ncFile.GetVar("predictor_dataset_ids_archive", &m_predictorDatasetIdsArchive[0], nPredictors);
            ncFile.GetVar("predictor_data_ids_realtime", &m_predictorDataIdsOper[0], nPredictors);
            ncFile.GetVar("predictor_data_ids_archive", &m_predictorDataIdsArchive[0], nPredictors);
            ncFile.GetVar("predictor_levels", &m_predictorLevels[0]);
            ncFile.GetVar("predictor_hours", &m_predictorHours[0]);
            ncFile.GetVar("predictor_lon_min", &m_predictorLonMin[0]);
            ncFile.GetVar("predictor_lon_max", &m_predictorLonMax[0]);
            ncFile.GetVar("predictor_lat_min", &m_predictorLatMin[0]);
            ncFile.GetVar("predictor_lat_max", &m_predictorLatMax[0]);
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
        a1f analogsCriteriaLeadTime(m_analogsNb[iTime]);
        a1f analogsDatesLeadTime(m_analogsNb[iTime]);

        for (int iAnalog = 0; iAnalog < m_analogsNb[iTime]; iAnalog++) {
            analogsCriteriaLeadTime(iAnalog) = analogsCriteria[ind];
            analogsDatesLeadTime(iAnalog) = analogsDates[ind];
            ind++;
        }

        m_analogsCriteria.push_back(analogsCriteriaLeadTime);
        m_analogsDates.push_back(analogsDatesLeadTime);
    }

    int indVal = 0;
    if (versionMajor == 1 && versionMinor == 0) {
        for (int iTime = 0; iTime < nLeadtime; iTime++) {
            a2f analogsValuesRawLeadTime(nStations, m_analogsNb[iTime]);
            a2f analogsValuesNormLeadTime(nStations, m_analogsNb[iTime]);

            for (int iAnalog = 0; iAnalog < m_analogsNb[iTime]; iAnalog++) {
                for (int iStat = 0; iStat < nStations; iStat++) {
                    analogsValuesRawLeadTime(iStat, iAnalog) = analogsValuesRaw[indVal];
                    analogsValuesNormLeadTime(iStat, iAnalog) = analogsValuesNorm[indVal];
                    indVal++;
                }
            }

            m_analogsValuesRaw.push_back(analogsValuesRawLeadTime);
            m_analogsValuesNorm.push_back(analogsValuesNormLeadTime);
        }
    } else {
        // Create containers
        for (int iTime = 0; iTime < nLeadtime; iTime++) {
            a2f analogsValuesLeadTime(nStations, m_analogsNb[iTime]);
            analogsValuesLeadTime.fill(NAN);
            m_analogsValuesRaw.push_back(analogsValuesLeadTime);
            m_analogsValuesNorm.push_back(analogsValuesLeadTime);
        }

        for (int iStat = 0; iStat < nStations; iStat++) {
            for (int iTime = 0; iTime < nLeadtime; iTime++) {
                for (int iAnalog = 0; iAnalog < m_analogsNb[iTime]; iAnalog++) {
                    m_analogsValuesRaw[iTime](iStat, iAnalog) = analogsValuesRaw[indVal];
                    m_analogsValuesNorm[iTime](iStat, iAnalog) = analogsValuesNorm[indVal];
                    indVal++;
                }
            }
        }
    }

    wxASSERT(!m_filePath.IsEmpty());
    wxASSERT(!m_predictandDatasetId.IsEmpty());
    wxASSERT(m_targetDates.size() > 0);
    wxASSERT(m_analogsNb.size() > 0);
    wxASSERT(m_stationIds.size() > 0);
    wxASSERT(!m_stationNames.empty());
    wxASSERT(m_stationHeights.size() > 0);
    wxASSERT(!m_analogsCriteria.empty());
    wxASSERT(!m_analogsDates.empty());
    wxASSERT(!m_analogsValuesRaw.empty());
    wxASSERT(m_stationXCoords.size() > 0);
    wxASSERT(m_stationYCoords.size() > 0);
    if (m_hasReferenceValues) {
        wxASSERT(m_referenceAxis.size() > 0);
        wxASSERT(m_referenceValues.cols() > 0);
        wxASSERT(m_referenceValues.rows() > 0);
    }

    return true;
}

wxArrayString asResultsForecast::GetStationNamesWxArray() const {
    wxArrayString stationsNames;
    for (const auto& stationName : m_stationNames) {
        stationsNames.Add(stationName);
    }
    return stationsNames;
}

wxArrayString asResultsForecast::GetStationNamesAndHeightsWxArray() const {
    wxArrayString stationsNames;
    for (int i = 0; i < m_stationNames.size(); i++) {
        wxString label;
        if (std::isfinite(m_stationHeights[i]) && m_stationHeights[i] != 0 && m_stationHeights[i] != -1) {
            label = asStrF("%s (%4.0fm)", m_stationNames[i], m_stationHeights[i]);
        } else {
            label = asStrF("%s", m_stationNames[i]);
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
    if (length >= m_targetDates.size()) return;

    m_targetDates = m_targetDates.head(length);
    m_analogsNb = m_analogsNb.head(length);

    m_analogsDates = va1f(m_analogsDates.begin(), m_analogsDates.begin() + length);
    m_analogsCriteria = va1f(m_analogsCriteria.begin(), m_analogsCriteria.begin() + length);
    m_analogsValuesRaw = va2f(m_analogsValuesRaw.begin(), m_analogsValuesRaw.begin() + length);
    m_analogsValuesNorm = va2f(m_analogsValuesNorm.begin(), m_analogsValuesNorm.begin() + length);
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
    for (float date : m_targetDates) {
        dates.Add(asTime::GetStringTime(date, format));
    }

    return dates;
}

wxString asResultsForecast::GetStationNameAndHeight(int iStat) const {
    wxString stationName;
    if (std::isfinite(m_stationHeights[iStat]) && m_stationHeights[iStat] != 0 && m_stationHeights[iStat] != -1) {
        stationName = asStrF("%s (%4.0fm)", m_stationNames[iStat], m_stationHeights[iStat]);
    } else {
        stationName = asStrF("%s", m_stationNames[iStat]);
    }
    return stationName;
}

wxString asResultsForecast::GetPredictandStationIdsString() const {
    wxString ids;

    for (int i = 0; i < (int)m_predictandStationIds.size(); i++) {
        ids << m_predictandStationIds[i];

        if (i < (int)m_predictandStationIds.size() - 1) {
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
            m_predictandStationIds.push_back(int(stationId));
        }
    }
}

bool asResultsForecast::IsCompatibleWith(asResultsForecast* otherForecast) const {
    bool compatible = true;

    if (!m_methodId.IsSameAs(otherForecast->GetMethodId(), false)) compatible = false;
    if (m_predictandParameter != otherForecast->GetPredictandParameter()) compatible = false;
    if (m_predictandTemporalResolution != otherForecast->GetPredictandTemporalResolution()) compatible = false;
    if (m_predictandSpatialAggregation != otherForecast->GetPredictandSpatialAggregation()) compatible = false;
    if (!m_predictandDatasetId.IsSameAs(otherForecast->GetPredictandDatasetId(), false)) compatible = false;
    if (!m_predictandDatabase.IsSameAs(otherForecast->GetPredictandDatabase(), false)) compatible = false;
    if (m_hasReferenceValues != otherForecast->HasReferenceValues()) compatible = false;
    if (m_leadTimeOrigin != otherForecast->GetLeadTimeOrigin()) compatible = false;

    a1f targetDates = otherForecast->GetTargetDates();
    if (m_targetDates.size() != targetDates.size()) {
        compatible = false;
    } else {
        for (int i = 0; i < m_targetDates.size(); i++) {
            if (m_targetDates[i] != targetDates[i]) compatible = false;
        }
    }

    a1i stationsIds = otherForecast->GetStationIds();
    if (m_stationIds.size() != stationsIds.size()) {
        compatible = false;
    } else {
        for (int i = 0; i < m_stationIds.size(); i++) {
            if (m_stationIds[i] != stationsIds[i]) compatible = false;
        }
    }

    a1f referenceAxis = otherForecast->GetReferenceAxis();
    if (m_referenceAxis.size() != referenceAxis.size()) {
        compatible = false;
    } else {
        for (int i = 0; i < m_referenceAxis.size(); i++) {
            if (!isnan(m_referenceAxis[i]) && m_referenceAxis[i] != referenceAxis[i]) {
                compatible = false;
            }
        }
    }

    if (!compatible) {
        wxLogError(_("The forecasts \"%s\" and \"%s\" are not compatible"), m_specificTagDisplay,
                   otherForecast->GetSpecificTagDisplay());
        return false;
    }

    return true;
}

bool asResultsForecast::IsSameAs(asResultsForecast* otherForecast) const {
    if (!IsCompatibleWith(otherForecast)) return false;

    if (!m_specificTag.IsSameAs(otherForecast->GetSpecificTag(), false)) return false;

    vi predictandStationIds = otherForecast->GetPredictandStationIds();
    if (m_predictandStationIds.size() != predictandStationIds.size()) {
        return false;
    }

    for (int i = 0; i < m_predictandStationIds.size(); i++) {
        if (m_predictandStationIds[i] != predictandStationIds[i]) return false;
    }

    a1f targetDates = otherForecast->GetTargetDates();
    if (m_targetDates.size() != targetDates.size()) {
        return false;
    }

    for (int i = 0; i < m_targetDates.size(); i++) {
        if (m_targetDates[i] != targetDates[i]) return false;
        if (m_analogsNb[i] != otherForecast->GetAnalogsNumber(i)) return false;
        if (m_analogsCriteria[i].size() != otherForecast->GetAnalogsCriteria(i).size()) return false;
        if (m_analogsDates[i].size() != otherForecast->GetAnalogsDates(i).size()) return false;
        if (m_analogsValuesRaw[i].size() != otherForecast->GetAnalogsValuesRaw(i).size()) return false;

        for (int j = 0; j < m_analogsCriteria[i].size(); j++) {
            if (m_analogsCriteria[i][j] != otherForecast->GetAnalogsCriteria(i)[j]) return false;
        }
        for (int j = 0; j < m_analogsDates[i].size(); j++) {
            if (m_analogsDates[i][j] != otherForecast->GetAnalogsDates(i)[j]) return false;
        }
        for (int j = 0; j < m_analogsDates[i].size(); j++) {
            if (m_analogsDates[i][j] != otherForecast->GetAnalogsDates(i)[j]) return false;
        }
        for (int j = 0; j < m_analogsValuesRaw[i].size(); j++) {
            if (m_analogsValuesRaw[i].rows() != otherForecast->GetAnalogsValuesRaw(i).rows()) return false;
            if (m_analogsValuesRaw[i].cols() != otherForecast->GetAnalogsValuesRaw(i).cols()) return false;
        }
    }

    return true;
}

bool asResultsForecast::IsSpecificForStationId(int stationId) const {
    for (int predictandStationId : m_predictandStationIds) {
        if (predictandStationId == stationId) {
            return true;
        }
    }
    return false;
}

int asResultsForecast::GetStationRowFromId(int stationId) const {
    for (int i = 0; i < m_stationIds.size(); i++) {
        if (m_stationIds[i] == stationId) {
            return i;
        }
    }

    wxFAIL;
    wxLogError(_("The station ID %d was not found in the forecast results."), stationId);
    return -1;
}

Coo asResultsForecast::GetStationsMeanCoordinates() {
    vd xs, ys;
    for (int id : m_predictandStationIds) {
        int i = GetStationRowFromId(id);
        wxASSERT(i >= 0);
        xs.push_back(m_stationXCoords[i]);
        ys.push_back(m_stationYCoords[i]);
    }

    return {std::reduce(xs.begin(), xs.end()) / xs.size(), std::reduce(ys.begin(), ys.end()) / ys.size()};
}
