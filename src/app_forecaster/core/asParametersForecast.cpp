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

#include "asParametersForecast.h"

#include "asFileParametersForecast.h"

asParametersForecast::asParametersForecast()
    : asParameters() {}

asParametersForecast::~asParametersForecast() {}

void asParametersForecast::AddStep() {
    asParameters::AddStep();
    ParamsStepForecast stepForecast;
    m_stepsForecast.push_back(stepForecast);
}

void asParametersForecast::AddPredictorForecast(ParamsStepForecast& step) {
    ParamsPredictorForecast predictor;
    step.predictors.push_back(predictor);
}

bool asParametersForecast::LoadFromFile(const wxString& filePath) {
    wxLogVerbose(_("Loading parameters file."));

    if (filePath.IsEmpty()) {
        wxLogError(_("The given path to the parameters file is empty."));
        return false;
    }

    asFileParametersForecast fileParams(filePath, asFile::ReadOnly);
    if (!fileParams.Open()) return false;

    if (!fileParams.CheckRootElement()) return false;

    int iStep = 0;
    wxXmlNode* nodeProcess = fileParams.GetRoot()->GetChildren();
    while (nodeProcess) {
        if (nodeProcess->GetName() == "description") {
            if (!ParseDescription(fileParams, nodeProcess)) return false;

        } else if (nodeProcess->GetName() == "time_properties") {
            if (!ParseTimeProperties(fileParams, nodeProcess)) return false;

        } else if (nodeProcess->GetName() == "analog_dates") {
            AddStep();
            if (!ParseAnalogDatesParams(fileParams, iStep, nodeProcess)) return false;
            iStep++;

        } else if (nodeProcess->GetName() == "analog_values") {
            if (!ParseAnalogValuesParams(fileParams, nodeProcess)) return false;

        } else {
            fileParams.UnknownNode(nodeProcess);
        }

        nodeProcess = nodeProcess->GetNext();
    }

    // Set properties
    if (!PreprocessingPropertiesOk()) return false;
    SetSpatialWindowProperties();
    SetPreloadingProperties();

    // Check inputs and init parameters
    if (!InputsOK()) return false;
    InitValues();

    // Fixes
    FixTimeLimits();
    FixWeights();
    FixCoordinates();

    wxLogVerbose(_("Parameters file loaded."));

    return true;
}

bool asParametersForecast::ParseDescription(asFileParametersForecast& fileParams, const wxXmlNode* nodeProcess) {
    wxXmlNode* nodeParam = nodeProcess->GetChildren();
    while (nodeParam) {
        if (nodeParam->GetName() == "method_id") {
            SetMethodId(fileParams.GetString(nodeParam));
        } else if (nodeParam->GetName() == "method_id_display") {
            SetMethodIdDisplay(fileParams.GetString(nodeParam));
            wxASSERT(!GetMethodIdDisplay().empty());
        } else if (nodeParam->GetName() == "specific_tag") {
            SetSpecificTag(fileParams.GetString(nodeParam));
        } else if (nodeParam->GetName() == "specific_tag_display") {
            SetSpecificTagDisplay(fileParams.GetString(nodeParam));
            wxASSERT(!GetSpecificTagDisplay().empty());
        } else if (nodeParam->GetName() == "description") {
            SetDescription(fileParams.GetString(nodeParam));
            wxASSERT(!GetDescription().empty());
        } else {
            fileParams.UnknownNode(nodeParam);
        }
        nodeParam = nodeParam->GetNext();
    }

    return true;
}

bool asParametersForecast::ParseTimeProperties(asFileParametersForecast& fileParams, const wxXmlNode* nodeProcess) {
    wxXmlNode* nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "archive_period") {
            wxXmlNode* nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "start_year") {
                    SetArchiveYearStart(fileParams.GetInt(nodeParam));
                } else if (nodeParam->GetName() == "end_year") {
                    SetArchiveYearEnd(fileParams.GetInt(nodeParam));
                } else if (nodeParam->GetName() == "start") {
                    SetArchiveStart(fileParams.GetString(nodeParam));
                } else if (nodeParam->GetName() == "end") {
                    SetArchiveEnd(fileParams.GetString(nodeParam));
                } else if (nodeParam->GetName() == "time_step") {
                    SetAnalogsTimeStepHours(fileParams.GetDouble(nodeParam));
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
        } else if (nodeParamBlock->GetName() == "lead_time") {
            wxXmlNode* nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "lead_time_days") {
                    SetLeadTimeDaysVector(fileParams.GetVectorDouble(nodeParam));
                } else if (nodeParam->GetName() == "lead_time_hours") {
                    SetLeadTimeHoursVector(fileParams.GetVectorDouble(nodeParam));
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
        } else if (nodeParamBlock->GetName() == "time_step") {
            SetTargetTimeStepHours(fileParams.GetDouble(nodeParamBlock));
            SetAnalogsTimeStepHours(fileParams.GetDouble(nodeParamBlock));
        } else if (nodeParamBlock->GetName() == "time_array_analogs") {
            wxXmlNode* nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "time_array") {
                    SetTimeArrayAnalogsMode(fileParams.GetString(nodeParam));
                } else if (nodeParam->GetName() == "interval_days") {
                    SetAnalogsIntervalDays(fileParams.GetInt(nodeParam));
                } else if (nodeParam->GetName() == "exclude_days") {
                    SetAnalogsExcludeDays(fileParams.GetInt(nodeParam));
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
        } else {
            fileParams.UnknownNode(nodeParamBlock);
        }
        nodeParamBlock = nodeParamBlock->GetNext();
    }
    return true;
}

bool asParametersForecast::ParseAnalogDatesParams(asFileParametersForecast& fileParams, int iStep,
                                                  const wxXmlNode* nodeProcess) {
    int iPtor = 0;
    wxXmlNode* nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "analogs_number") {
            SetAnalogsNumberLeadTimeVector(iStep, fileParams.GetVectorInt(nodeParamBlock));
        } else if (nodeParamBlock->GetName() == "predictor") {
            AddPredictor(iStep);
            AddPredictorForecast(m_stepsForecast[iStep]);
            SetPreprocess(iStep, iPtor, false);
            SetPreload(iStep, iPtor, false);
            wxXmlNode* nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "preload") {
                    SetPreload(iStep, iPtor, fileParams.GetBool(nodeParam));
                } else if (nodeParam->GetName() == "realtime_standardize") {
                    SetRealtimeStandardize(iStep, iPtor, fileParams.GetBool(nodeParam));
                } else if (nodeParam->GetName() == "realtime_standardize_mean") {
                    SetRealtimeStandardizeMean(iStep, iPtor, fileParams.GetDouble(nodeParam));
                } else if (nodeParam->GetName() == "realtime_standardize_sd") {
                    SetRealtimeStandardizeSd(iStep, iPtor, fileParams.GetDouble(nodeParam));
                } else if (nodeParam->GetName() == "archive_standardize") {
                    SetArchiveStandardize(iStep, iPtor, fileParams.GetBool(nodeParam));
                } else if (nodeParam->GetName() == "archive_standardize_mean") {
                    SetArchiveStandardizeMean(iStep, iPtor, fileParams.GetDouble(nodeParam));
                } else if (nodeParam->GetName() == "archive_standardize_sd") {
                    SetArchiveStandardizeSd(iStep, iPtor, fileParams.GetDouble(nodeParam));
                } else if (nodeParam->GetName() == "preprocessing") {
                    SetPreprocess(iStep, iPtor, true);
                    if (!ParsePreprocessedPredictors(fileParams, iStep, iPtor, nodeParam)) return false;
                } else if (nodeParam->GetName() == "realtime_dataset_id") {
                    SetPredictorRealtimeDatasetId(iStep, iPtor, fileParams.GetString(nodeParam));
                } else if (nodeParam->GetName() == "realtime_data_id") {
                    SetPredictorRealtimeDataId(iStep, iPtor, fileParams.GetString(nodeParam));
                } else if (nodeParam->GetName() == "archive_dataset_id") {
                    SetPredictorArchiveDatasetId(iStep, iPtor, fileParams.GetString(nodeParam));
                } else if (nodeParam->GetName() == "archive_data_id") {
                    SetPredictorArchiveDataId(iStep, iPtor, fileParams.GetString(nodeParam));
                } else if (nodeParam->GetName() == "level") {
                    SetPredictorLevel(iStep, iPtor, fileParams.GetFloat(nodeParam));
                } else if (nodeParam->GetName() == "time") {
                    SetPredictorHour(iStep, iPtor, fileParams.GetDouble(nodeParam));
                } else if (nodeParam->GetName() == "members") {
                    SetPredictorMembersNb(iStep, iPtor, fileParams.GetInt(nodeParam));
                } else if (nodeParam->GetName() == "spatial_window") {
                    wxXmlNode* nodeWindow = nodeParam->GetChildren();
                    while (nodeWindow) {
                        if (nodeWindow->GetName() == "grid_type") {
                            SetPredictorGridType(iStep, iPtor, fileParams.GetString(nodeWindow, "regular"));
                        } else if (nodeWindow->GetName() == "x_min") {
                            SetPredictorXmin(iStep, iPtor, fileParams.GetDouble(nodeWindow));
                        } else if (nodeWindow->GetName() == "x_points_nb") {
                            SetPredictorXptsnb(iStep, iPtor, fileParams.GetInt(nodeWindow));
                        } else if (nodeWindow->GetName() == "x_step") {
                            SetPredictorXstep(iStep, iPtor, fileParams.GetDouble(nodeWindow));
                        } else if (nodeWindow->GetName() == "y_min") {
                            SetPredictorYmin(iStep, iPtor, fileParams.GetDouble(nodeWindow));
                        } else if (nodeWindow->GetName() == "y_points_nb") {
                            SetPredictorYptsnb(iStep, iPtor, fileParams.GetInt(nodeWindow));
                        } else if (nodeWindow->GetName() == "y_step") {
                            SetPredictorYstep(iStep, iPtor, fileParams.GetDouble(nodeWindow));
                        } else {
                            fileParams.UnknownNode(nodeWindow);
                        }
                        nodeWindow = nodeWindow->GetNext();
                    }
                } else if (nodeParam->GetName() == "criteria") {
                    SetPredictorCriteria(iStep, iPtor, fileParams.GetString(nodeParam));
                } else if (nodeParam->GetName() == "weight") {
                    SetPredictorWeight(iStep, iPtor, fileParams.GetFloat(nodeParam));
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
            iPtor++;
        } else {
            fileParams.UnknownNode(nodeParamBlock);
        }
        nodeParamBlock = nodeParamBlock->GetNext();
    }
    return true;
}

bool asParametersForecast::ParsePreprocessedPredictors(asFileParametersForecast& fileParams, int iStep, int iPtor,
                                                       const wxXmlNode* nodeParam) {
    int iPre = 0;
    wxXmlNode* nodePreprocess = nodeParam->GetChildren();
    while (nodePreprocess) {
        if (nodePreprocess->GetName() == "preprocessing_method") {
            SetPreprocessMethod(iStep, iPtor, fileParams.GetString(nodePreprocess));
        } else if (nodePreprocess->GetName() == "preprocessing_data") {
            wxXmlNode* nodeParamPreprocess = nodePreprocess->GetChildren();
            while (nodeParamPreprocess) {
                if (nodeParamPreprocess->GetName() == "realtime_dataset_id") {
                    SetPreprocessRealtimeDatasetId(iStep, iPtor, iPre, fileParams.GetString(nodeParamPreprocess));
                } else if (nodeParamPreprocess->GetName() == "realtime_data_id") {
                    SetPreprocessRealtimeDataId(iStep, iPtor, iPre, fileParams.GetString(nodeParamPreprocess));
                } else if (nodeParamPreprocess->GetName() == "archive_dataset_id") {
                    SetPreprocessArchiveDatasetId(iStep, iPtor, iPre, fileParams.GetString(nodeParamPreprocess));
                } else if (nodeParamPreprocess->GetName() == "archive_data_id") {
                    SetPreprocessArchiveDataId(iStep, iPtor, iPre, fileParams.GetString(nodeParamPreprocess));
                } else if (nodeParamPreprocess->GetName() == "level") {
                    SetPreprocessLevel(iStep, iPtor, iPre, fileParams.GetFloat(nodeParamPreprocess));
                } else if (nodeParamPreprocess->GetName() == "time") {
                    SetPreprocessHour(iStep, iPtor, iPre, fileParams.GetDouble(nodeParamPreprocess));
                } else if (nodeParamPreprocess->GetName() == "members") {
                    SetPreprocessMembersNb(iStep, iPtor, iPre, fileParams.GetInt(nodeParamPreprocess));
                } else {
                    fileParams.UnknownNode(nodeParamPreprocess);
                }
                nodeParamPreprocess = nodeParamPreprocess->GetNext();
            }
            iPre++;
        } else {
            fileParams.UnknownNode(nodePreprocess);
        }
        nodePreprocess = nodePreprocess->GetNext();
    }
    return true;
}

bool asParametersForecast::ParseAnalogValuesParams(asFileParametersForecast& fileParams, const wxXmlNode* nodeProcess) {
    wxXmlNode* nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "predictand") {
            wxXmlNode* nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "station_id" || nodeParam->GetName() == "station_ids") {
                    SetPredictandStationIds(fileParams.GetString(nodeParam));
                } else if (nodeParam->GetName() == "database") {
                    SetPredictandDatabase(fileParams.GetString(nodeParam));
                } else if (nodeParam->GetName() == "time") {
                    SetPredictandTimeHours(fileParams.GetDouble(nodeParam));
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
        } else {
            fileParams.UnknownNode(nodeParamBlock);
        }
        nodeParamBlock = nodeParamBlock->GetNext();
    }
    return true;
}

bool asParametersForecast::InputsOK() const {
    // Time properties
    if (GetLeadTimeDaysVector().empty()) {
        wxLogError(_("The lead times were not provided in the parameters file."));
        return false;
    }

    if (asIsNaN(GetArchiveStart())) {
        wxLogError(_("The beginning of the archive period was not provided in the parameters file."));
        return false;
    }

    if (asIsNaN(GetArchiveEnd())) {
        wxLogError(_("The end of the archive period was not provided in the parameters file."));
        return false;
    }

    if (GetTargetTimeStepHours() <= 0) {
        wxLogError(_("The time step was not provided in the parameters file."));
        return false;
    }

    if (GetAnalogsTimeStepHours() <= 0) {
        wxLogError(_("The time step was not provided in the parameters file."));
        return false;
    }

    if (GetTimeArrayAnalogsMode().CmpNoCase("interval_days") == 0 ||
        GetTimeArrayAnalogsMode().CmpNoCase("IntervalDays") == 0) {
        if (GetAnalogsIntervalDays() <= 0) {
            wxLogError(
                _("The interval days for the analogs preselection "
                  "was not provided in the parameters file."));
            return false;
        }
        if (GetAnalogsExcludeDays() <= 0) {
            wxLogError(
                _("The number of days to exclude around the target "
                  "date was not provided in the parameters file."));
            return false;
        }
    }

    // Analog dates
    for (int i = 0; i < GetStepsNb(); i++) {
        if (GetAnalogsNumberLeadTimeVector(i).size() != GetLeadTimeDaysVector().size()) {
            wxLogError(_("The length of the analogs numbers (step %d) "
                         "do not match the number of lead times."),
                       i);
            return false;
        }

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                if (GetPreprocessMethod(i, j).IsEmpty()) {
                    wxLogError(_("The preprocessing method (step %d, predictor %d) was not "
                                 "provided in the parameters file."),
                               i, j);
                    return false;
                }

                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (GetPreprocessRealtimeDatasetId(i, j, k).IsEmpty()) {
                        wxLogError(_("The realtime dataset for preprocessing (step %d, predictor %d) "
                                     "was not provided in the parameters file."),
                                   i, j);
                        return false;
                    }
                    if (GetPreprocessRealtimeDataId(i, j, k).IsEmpty()) {
                        wxLogError(_("The realtime data for preprocessing (step %d, predictor %d) "
                                     "was not provided in the parameters file."),
                                   i, j);
                        return false;
                    }
                    if (GetPreprocessArchiveDatasetId(i, j, k).IsEmpty()) {
                        wxLogError(_("The archive dataset for preprocessing (step %d, predictor %d) "
                                     "was not provided in the parameters file."),
                                   i, j);
                        return false;
                    }
                    if (GetPreprocessArchiveDataId(i, j, k).IsEmpty()) {
                        wxLogError(_("The archive data for preprocessing (step %d, predictor %d) "
                                     "was not provided in the parameters file."),
                                   i, j);
                        return false;
                    }
                }
            } else {
                if (GetPredictorRealtimeDatasetId(i, j).IsEmpty()) {
                    wxLogError(_("The realtime dataset (step %d, predictor %d) was not "
                                 "provided in the parameters file."),
                               i, j);
                    return false;
                }
                if (GetPredictorRealtimeDataId(i, j).IsEmpty()) {
                    wxLogError(_("The realtime data (step %d, predictor %d) was "
                                 "not provided in the parameters file."),
                               i, j);
                    return false;
                }
                if (GetPredictorArchiveDatasetId(i, j).IsEmpty()) {
                    wxLogError(_("The archive dataset (step %d, predictor %d) was not "
                                 "provided in the parameters file."),
                               i, j);
                    return false;
                }
                if (GetPredictorArchiveDataId(i, j).IsEmpty()) {
                    wxLogError(_("The archive data (step %d, predictor %d) was "
                                 "not provided in the parameters file."),
                               i, j);
                    return false;
                }
            }

            if (GetPredictorGridType(i, j).IsEmpty()) {
                wxLogError(_("The grid type (step %d, predictor %d) "
                             "is empty in the parameters file."),
                           i, j);
                return false;
            }
            if (GetPredictorXptsnb(i, j) == 0) {
                wxLogError(_("The X points nb value (step %d, predictor %d) "
                             "was not provided in the parameters file."),
                           i, j);
                return false;
            }
            if (GetPredictorYptsnb(i, j) == 0) {
                wxLogError(_("The Y points nb value (step %d, predictor %d) "
                             "was not provided in the parameters file."),
                           i, j);
                return false;
            }
            if (GetPredictorCriteria(i, j).IsEmpty()) {
                wxLogError(_("The criteria (step %d, predictor %d) was "
                             "not provided in the parameters file."),
                           i, j);
                return false;
            }
        }
    }

    return true;
}

void asParametersForecast::InitValues() {
    // Initialize the parameters values with the first values of the vectors
    for (int i = 0; i < GetStepsNb(); i++) {
        SetAnalogsNumber(i, m_stepsForecast[i].analogsNumberLeadTime[0]);
    }

    // Fixes and checks
    FixWeights();
    FixCoordinates();
    FixAnalogsNb();
}

void asParametersForecast::SetLeadTimeDaysVector(vd val) {
    wxASSERT(val.size() > 0);
    m_leadTimeDaysVect = val;
}

void asParametersForecast::SetLeadTimeHoursVector(vd val) {
    wxASSERT(val.size() > 0);
    for (float hour : val) {
        m_leadTimeDaysVect.push_back(hour / 24.0);
    }
}

void asParametersForecast::SetAnalogsNumberLeadTimeVector(int iStep, vi val) {
    wxASSERT(val.size() > 0);

    if (val.size() == GetLeadTimeDaysVector().size()) {
        m_stepsForecast[iStep].analogsNumberLeadTime = val;
    } else if (val.size() == 1) {
        for (int i = 0; i < GetLeadTimeDaysVector().size(); i++) {
            m_stepsForecast[iStep].analogsNumberLeadTime.push_back(val[0]);
        }
    } else {
        wxLogError(_("The lengths of the lead time and the number of analogs arrays are not consistent."));
    }
}

void asParametersForecast::SetPredictorArchiveDatasetId(int iStep, int iPtor, const wxString& val) {
    wxASSERT(!val.IsEmpty());
    m_stepsForecast[iStep].predictors[iPtor].archiveDatasetId = val;
}

void asParametersForecast::SetPredictorArchiveDataId(int iStep, int iPtor, const wxString& val) {
    wxASSERT(!val.IsEmpty());
    m_stepsForecast[iStep].predictors[iPtor].archiveDataId = val;
}

void asParametersForecast::SetPredictorRealtimeDatasetId(int iStep, int iPtor, const wxString& val) {
    wxASSERT(!val.IsEmpty());
    m_stepsForecast[iStep].predictors[iPtor].realtimeDatasetId = val;
}

void asParametersForecast::SetPredictorRealtimeDataId(int iStep, int iPtor, const wxString& val) {
    wxASSERT(!val.IsEmpty());
    m_stepsForecast[iStep].predictors[iPtor].realtimeDataId = val;
}

wxString asParametersForecast::GetPreprocessArchiveDatasetId(int iStep, int iPtor, int iPre) const {
    wxASSERT(m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveDatasetIds.size() > iPre);
    return m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveDatasetIds[iPre];
}

void asParametersForecast::SetPreprocessArchiveDatasetId(int iStep, int iPtor, int iPre, const wxString& val) {
    wxASSERT(!val.IsEmpty());
    if (m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveDatasetIds.size() >= iPre + 1) {
        m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveDatasetIds[iPre] = val;
    } else {
        m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveDatasetIds.push_back(val);
    }
}

wxString asParametersForecast::GetPreprocessArchiveDataId(int iStep, int iPtor, int iPre) const {
    wxASSERT(m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveDataIds.size() > iPre);
    return m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveDataIds[iPre];
}

void asParametersForecast::SetPreprocessArchiveDataId(int iStep, int iPtor, int iPre, const wxString& val) {
    wxASSERT(!val.IsEmpty());
    if (m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveDataIds.size() >= iPre + 1) {
        m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveDataIds[iPre] = val;
    } else {
        m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveDataIds.push_back(val);
    }
}

wxString asParametersForecast::GetPreprocessRealtimeDatasetId(int iStep, int iPtor, int iPre) const {
    wxASSERT(m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeDatasetIds.size() > iPre);
    return m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeDatasetIds[iPre];
}

void asParametersForecast::SetPreprocessRealtimeDatasetId(int iStep, int iPtor, int iPre, const wxString& val) {
    wxASSERT(!val.IsEmpty());
    if (m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeDatasetIds.size() >= iPre + 1) {
        m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeDatasetIds[iPre] = val;
    } else {
        m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeDatasetIds.push_back(val);
    }
}

wxString asParametersForecast::GetPreprocessRealtimeDataId(int iStep, int iPtor, int iPre) const {
    wxASSERT(m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeDataIds.size() >= iPre + 1);
    return m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeDataIds[iPre];
}

void asParametersForecast::SetPreprocessRealtimeDataId(int iStep, int iPtor, int iPre, const wxString& val) {
    wxASSERT(!val.IsEmpty());
    if (m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeDataIds.size() >= iPre + 1) {
        m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeDataIds[iPre] = val;
    } else {
        m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeDataIds.push_back(val);
    }
}

void asParametersForecast::SetPredictandDatabase(const wxString& val) {
    wxASSERT(!val.IsEmpty());
    m_predictandDatabase = val;
}
