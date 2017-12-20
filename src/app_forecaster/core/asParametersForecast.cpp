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
        : asParameters()
{
    //ctor
}

asParametersForecast::~asParametersForecast()
{
    //dtor
}

void asParametersForecast::AddStep()
{
    asParameters::AddStep();
    ParamsStepForecast stepForecast;
    stepForecast.analogsNumberLeadTime.push_back(0);
    m_stepsForecast.push_back(stepForecast);
}

void asParametersForecast::AddPredictorForecast(ParamsStepForecast &step)
{
    ParamsPredictorForecast predictor;

    predictor.archiveDatasetId = wxEmptyString;
    predictor.archiveDataId = wxEmptyString;
    predictor.realtimeDatasetId = wxEmptyString;
    predictor.realtimeDataId = wxEmptyString;

    step.predictors.push_back(predictor);
}

bool asParametersForecast::LoadFromFile(const wxString &filePath)
{
    wxLogVerbose(_("Loading parameters file."));

    if (filePath.IsEmpty()) {
        wxLogError(_("The given path to the parameters file is empty."));
        return false;
    }

    asFileParametersForecast fileParams(filePath, asFile::ReadOnly);
    if (!fileParams.Open())
        return false;

    if (!fileParams.CheckRootElement())
        return false;

    int iStep = 0;
    wxXmlNode *nodeProcess = fileParams.GetRoot()->GetChildren();
    while (nodeProcess) {

        if (nodeProcess->GetName() == "description") {
            if (!ParseDescription(fileParams, nodeProcess))
                return false;

        } else if (nodeProcess->GetName() == "time_properties") {
            if (!ParseTimeProperties(fileParams, nodeProcess))
                return false;

        } else if (nodeProcess->GetName() == "analog_dates") {
            AddStep();
            if (!ParseAnalogDatesParams(fileParams, iStep, nodeProcess))
                return false;
            iStep++;

        } else if (nodeProcess->GetName() == "analog_values") {
            if (!ParseAnalogValuesParams(fileParams, nodeProcess))
                return false;

        } else {
            fileParams.UnknownNode(nodeProcess);
        }

        nodeProcess = nodeProcess->GetNext();
    }

    // Set properties
    if (!PreprocessingPropertiesOk())
        return false;
    SetSpatialWindowProperties();
    SetPreloadingProperties();

    // Check inputs and init parameters
    if (!InputsOK())
        return false;
    InitValues();

    // Fixes
    FixTimeLimits();
    FixWeights();
    FixCoordinates();

    wxLogVerbose(_("Parameters file loaded."));

    return true;
}

bool asParametersForecast::ParseDescription(asFileParametersForecast &fileParams, const wxXmlNode *nodeProcess)
{
    wxXmlNode *nodeParam = nodeProcess->GetChildren();
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

bool asParametersForecast::ParseTimeProperties(asFileParametersForecast &fileParams, const wxXmlNode *nodeProcess)
{
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "archive_period") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "start_year") {
                    if (!SetArchiveYearStart(fileParams.GetInt(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "end_year") {
                    if (!SetArchiveYearEnd(fileParams.GetInt(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "start") {
                    if (!SetArchiveStart(fileParams.GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "end") {
                    if (!SetArchiveEnd(fileParams.GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "time_step") {
                    if (!SetTimeArrayAnalogsTimeStepHours(fileParams.GetDouble(nodeParam)))
                        return false;
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
        } else if (nodeParamBlock->GetName() == "lead_time") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "lead_time_days") {
                    if (!SetLeadTimeDaysVector(fileParams.GetVectorInt(nodeParam)))
                        return false;
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
        } else if (nodeParamBlock->GetName() == "time_step") {
            if (!SetTimeArrayTargetTimeStepHours(fileParams.GetDouble(nodeParamBlock)))
                return false;
            if (!SetTimeArrayAnalogsTimeStepHours(fileParams.GetDouble(nodeParamBlock)))
                return false;
        } else if (nodeParamBlock->GetName() == "time_array_analogs") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "time_array") {
                    if (!SetTimeArrayAnalogsMode(fileParams.GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "interval_days") {
                    if (!SetTimeArrayAnalogsIntervalDays(fileParams.GetInt(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "exclude_days") {
                    if (!SetTimeArrayAnalogsExcludeDays(fileParams.GetInt(nodeParam)))
                        return false;
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

bool asParametersForecast::ParseAnalogDatesParams(asFileParametersForecast &fileParams, int iStep,
                                                  const wxXmlNode *nodeProcess)
{
    int iPtor = 0;
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "analogs_number") {
            if (!SetAnalogsNumberLeadTimeVector(iStep, fileParams.GetVectorInt(nodeParamBlock)))
                return false;
        } else if (nodeParamBlock->GetName() == "predictor") {
            AddPredictor(iStep);
            AddPredictorForecast(m_stepsForecast[iStep]);
            SetPreprocess(iStep, iPtor, false);
            SetPreload(iStep, iPtor, false);
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "preload") {
                    SetPreload(iStep, iPtor, fileParams.GetBool(nodeParam));
                } else if (nodeParam->GetName() == "preprocessing") {
                    SetPreprocess(iStep, iPtor, true);
                    if (!ParsePreprocessedPredictors(fileParams, iStep, iPtor, nodeParam))
                        return false;
                } else if (nodeParam->GetName() == "realtime_dataset_id") {
                    if (!SetPredictorRealtimeDatasetId(iStep, iPtor, fileParams.GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "realtime_data_id") {
                    if (!SetPredictorRealtimeDataId(iStep, iPtor, fileParams.GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "archive_dataset_id") {
                    if (!SetPredictorArchiveDatasetId(iStep, iPtor, fileParams.GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "archive_data_id") {
                    if (!SetPredictorArchiveDataId(iStep, iPtor, fileParams.GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "level") {
                    if (!SetPredictorLevel(iStep, iPtor, fileParams.GetFloat(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "time") {
                    if (!SetPredictorTimeHours(iStep, iPtor, fileParams.GetDouble(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "members") {
                    if (!SetPredictorMembersNb(iStep, iPtor, fileParams.GetInt(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "spatial_window") {
                    wxXmlNode *nodeWindow = nodeParam->GetChildren();
                    while (nodeWindow) {
                        if (nodeWindow->GetName() == "grid_type") {
                            if (!SetPredictorGridType(iStep, iPtor, fileParams.GetString(nodeWindow, "regular")))
                                return false;
                        } else if (nodeWindow->GetName() == "x_min") {
                            if (!SetPredictorXmin(iStep, iPtor, fileParams.GetDouble(nodeWindow)))
                                return false;
                        } else if (nodeWindow->GetName() == "x_points_nb") {
                            if (!SetPredictorXptsnb(iStep, iPtor, fileParams.GetInt(nodeWindow)))
                                return false;
                        } else if (nodeWindow->GetName() == "x_step") {
                            if (!SetPredictorXstep(iStep, iPtor, fileParams.GetDouble(nodeWindow)))
                                return false;
                        } else if (nodeWindow->GetName() == "y_min") {
                            if (!SetPredictorYmin(iStep, iPtor, fileParams.GetDouble(nodeWindow)))
                                return false;
                        } else if (nodeWindow->GetName() == "y_points_nb") {
                            if (!SetPredictorYptsnb(iStep, iPtor, fileParams.GetInt(nodeWindow)))
                                return false;
                        } else if (nodeWindow->GetName() == "y_step") {
                            if (!SetPredictorYstep(iStep, iPtor, fileParams.GetDouble(nodeWindow)))
                                return false;
                        } else {
                            fileParams.UnknownNode(nodeWindow);
                        }
                        nodeWindow = nodeWindow->GetNext();
                    }
                } else if (nodeParam->GetName() == "criteria") {
                    if (!SetPredictorCriteria(iStep, iPtor, fileParams.GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "weight") {
                    if (!SetPredictorWeight(iStep, iPtor, fileParams.GetFloat(nodeParam)))
                        return false;
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

bool asParametersForecast::ParsePreprocessedPredictors(asFileParametersForecast &fileParams, int iStep, int iPtor,
                                                       const wxXmlNode *nodeParam)
{
    int iPre = 0;
    wxXmlNode *nodePreprocess = nodeParam->GetChildren();
    while (nodePreprocess) {
        if (nodePreprocess->GetName() == "preprocessing_method") {
            if (!SetPreprocessMethod(iStep, iPtor, fileParams.GetString(nodePreprocess)))
                return false;
        } else if (nodePreprocess->GetName() == "preprocessing_data") {
            wxXmlNode *nodeParamPreprocess = nodePreprocess->GetChildren();
            while (nodeParamPreprocess) {
                if (nodeParamPreprocess->GetName() == "realtime_dataset_id") {
                    if (!SetPreprocessRealtimeDatasetId(iStep, iPtor, iPre, fileParams.GetString(nodeParamPreprocess)))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "realtime_data_id") {
                    if (!SetPreprocessRealtimeDataId(iStep, iPtor, iPre, fileParams.GetString(nodeParamPreprocess)))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "archive_dataset_id") {
                    if (!SetPreprocessArchiveDatasetId(iStep, iPtor, iPre, fileParams.GetString(nodeParamPreprocess)))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "archive_data_id") {
                    if (!SetPreprocessArchiveDataId(iStep, iPtor, iPre, fileParams.GetString(nodeParamPreprocess)))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "level") {
                    if (!SetPreprocessLevel(iStep, iPtor, iPre, fileParams.GetFloat(nodeParamPreprocess)))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "time") {
                    if (!SetPreprocessTimeHours(iStep, iPtor, iPre, fileParams.GetDouble(nodeParamPreprocess)))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "members") {
                    if (!SetPreprocessMembersNb(iStep, iPtor, iPre, fileParams.GetInt(nodeParamPreprocess)))
                        return false;
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

bool asParametersForecast::ParseAnalogValuesParams(asFileParametersForecast &fileParams, const wxXmlNode *nodeProcess)
{
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "predictand") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "station_id" || nodeParam->GetName() == "station_ids") {
                    if (!SetPredictandStationIds(fileParams.GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "database") {
                    SetPredictandDatabase(fileParams.GetString(nodeParam));
                } else if (nodeParam->GetName() == "time") {
                    if (!SetPredictandTimeHours(fileParams.GetDouble(nodeParam)))
                        return false;
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

bool asParametersForecast::InputsOK() const
{
    // Time properties
    if (GetLeadTimeDaysVector().empty()) {
        wxLogError(_("The lead times were not provided in the parameters file."));
        return false;
    }

    if (GetArchiveStart() <= 0) {
        wxLogError(_("The beginning of the archive period was not provided in the parameters file."));
        return false;
    }

    if (GetArchiveEnd() <= 0) {
        wxLogError(_("The end of the archive period was not provided in the parameters file."));
        return false;
    }

    if (GetTimeArrayTargetTimeStepHours() <= 0) {
        wxLogError(_("The time step was not provided in the parameters file."));
        return false;
    }

    if (GetTimeArrayAnalogsTimeStepHours() <= 0) {
        wxLogError(_("The time step was not provided in the parameters file."));
        return false;
    }

    if (GetTimeArrayAnalogsMode().CmpNoCase("interval_days") == 0 ||
        GetTimeArrayAnalogsMode().CmpNoCase("IntervalDays") == 0) {
        if (GetTimeArrayAnalogsIntervalDays() <= 0) {
            wxLogError(_("The interval days for the analogs preselection was not provided in the parameters file."));
            return false;
        }
        if (GetTimeArrayAnalogsExcludeDays() <= 0) {
            wxLogError(_("The number of days to exclude around the target date was not provided in the parameters file."));
            return false;
        }
    }

    // Analog dates
    for (int i = 0; i < GetStepsNb(); i++) {
        if (GetAnalogsNumberLeadTimeVector(i).size() != GetLeadTimeDaysVector().size()) {
            wxLogError(_("The length of the analogs numbers (step %d) do not match the number of lead times."), i);
            return false;
        }

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                if (GetPreprocessMethod(i, j).IsEmpty()) {
                    wxLogError(_("The preprocessing method (step %d, predictor %d) was not provided in the parameters file."),
                               i, j);
                    return false;
                }

                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (GetPreprocessRealtimeDatasetId(i, j, k).IsEmpty()) {
                        wxLogError(_("The realtime dataset for preprocessing (step %d, predictor %d) was not provided in the parameters file."),
                                   i, j);
                        return false;
                    }
                    if (GetPreprocessRealtimeDataId(i, j, k).IsEmpty()) {
                        wxLogError(_("The realtime data for preprocessing (step %d, predictor %d) was not provided in the parameters file."),
                                   i, j);
                        return false;
                    }
                    if (GetPreprocessArchiveDatasetId(i, j, k).IsEmpty()) {
                        wxLogError(_("The archive dataset for preprocessing (step %d, predictor %d) was not provided in the parameters file."),
                                   i, j);
                        return false;
                    }
                    if (GetPreprocessArchiveDataId(i, j, k).IsEmpty()) {
                        wxLogError(_("The archive data for preprocessing (step %d, predictor %d) was not provided in the parameters file."),
                                   i, j);
                        return false;
                    }
                }
            } else {
                if (GetPredictorRealtimeDatasetId(i, j).IsEmpty()) {
                    wxLogError(_("The realtime dataset (step %d, predictor %d) was not provided in the parameters file."),
                               i, j);
                    return false;
                }
                if (GetPredictorRealtimeDataId(i, j).IsEmpty()) {
                    wxLogError(_("The realtime data (step %d, predictor %d) was not provided in the parameters file."),
                               i, j);
                    return false;
                }
                if (GetPredictorArchiveDatasetId(i, j).IsEmpty()) {
                    wxLogError(_("The archive dataset (step %d, predictor %d) was not provided in the parameters file."),
                               i, j);
                    return false;
                }
                if (GetPredictorArchiveDataId(i, j).IsEmpty()) {
                    wxLogError(_("The archive data (step %d, predictor %d) was not provided in the parameters file."),
                               i, j);
                    return false;
                }
            }

            if (GetPredictorGridType(i, j).IsEmpty()) {
                wxLogError(_("The grid type (step %d, predictor %d) is empty in the parameters file."), i, j);
                return false;
            }
            if (GetPredictorXptsnb(i, j) == 0) {
                wxLogError(_("The X points nb value (step %d, predictor %d) was not provided in the parameters file."),
                           i, j);
                return false;
            }
            if (GetPredictorYptsnb(i, j) == 0) {
                wxLogError(_("The Y points nb value (step %d, predictor %d) was not provided in the parameters file."),
                           i, j);
                return false;
            }
            if (GetPredictorCriteria(i, j).IsEmpty()) {
                wxLogError(_("The criteria (step %d, predictor %d) was not provided in the parameters file."), i, j);
                return false;
            }
        }
    }

    return true;
}

void asParametersForecast::InitValues()
{
    // Initialize the parameters values with the first values of the vectors
    for (int i = 0; i < GetStepsNb(); i++) {
        SetAnalogsNumber(i, m_stepsForecast[i].analogsNumberLeadTime[0]);
    }

    // Fixes and checks
    FixWeights();
    FixCoordinates();
    FixAnalogsNb();
}

bool asParametersForecast::SetLeadTimeDaysVector(vi val)
{
    if (val.size() < 1) {
        wxLogError(_("The provided 'lead time (days)' vector is empty."));
        return false;
    } else {
        for (int i = 0; i < (int) val.size(); i++) {
            if (asTools::IsNaN(val[i])) {
                wxLogError(_("There are NaN values in the provided 'lead time (days)' vector."));
                return false;
            }
        }
    }
    m_leadTimeDaysVect = val;
    return true;
}

bool asParametersForecast::SetAnalogsNumberLeadTimeVector(int iStep, vi val)
{
    if (val.size() < 1) {
        wxLogError(_("The provided analogs numbers vector (fct of the lead time) is empty."));
        return false;
    } else {
        for (int i = 0; i < (int) val.size(); i++) {
            if (asTools::IsNaN(val[i])) {
                wxLogError(_("There are NaN values in the provided analogs numbers vector (fct of the lead time)."));
                return false;
            }
        }
    }
    m_stepsForecast[iStep].analogsNumberLeadTime = val;
    return true;
}

bool asParametersForecast::SetPredictorArchiveDatasetId(int iStep, int iPtor, const wxString &val)
{
    if (val.IsEmpty()) {
        wxLogError(_("The provided value for the predictor archive dataset ID is null"));
        return false;
    }
    m_stepsForecast[iStep].predictors[iPtor].archiveDatasetId = val;
    return true;
}

bool asParametersForecast::SetPredictorArchiveDataId(int iStep, int iPtor, const wxString &val)
{
    if (val.IsEmpty()) {
        wxLogError(_("The provided value for the predictor archive data ID is null"));
        return false;
    }
    m_stepsForecast[iStep].predictors[iPtor].archiveDataId = val;
    return true;
}

bool asParametersForecast::SetPredictorRealtimeDatasetId(int iStep, int iPtor, const wxString &val)
{
    if (val.IsEmpty()) {
        wxLogError(_("The provided value for the predictor realtime dataset ID is null"));
        return false;
    }
    m_stepsForecast[iStep].predictors[iPtor].realtimeDatasetId = val;
    return true;
}

bool asParametersForecast::SetPredictorRealtimeDataId(int iStep, int iPtor, const wxString &val)
{
    if (val.IsEmpty()) {
        wxLogError(_("The provided value for the predictor realtime data ID is null"));
        return false;
    }
    m_stepsForecast[iStep].predictors[iPtor].realtimeDataId = val;
    return true;
}

wxString asParametersForecast::GetPreprocessArchiveDatasetId(int iStep, int iPtor, int iPre) const
{
    if (m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveDatasetIds.size() >= (unsigned) (iPre + 1)) {
        return m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveDatasetIds[iPre];
    } else {
        wxLogError(_("Trying to access to an element outside of preprocessArchiveDatasetIds in the parameters object."));
        return wxEmptyString;
    }
}

bool asParametersForecast::SetPreprocessArchiveDatasetId(int iStep, int iPtor, int iPre, const wxString &val)
{
    if (val.IsEmpty()) {
        wxLogError(_("The provided value for the preprocess archive dataset ID is null"));
        return false;
    }

    if (m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveDatasetIds.size() >= (unsigned) (iPre + 1)) {
        m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveDatasetIds[iPre] = val;
    } else {
        m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveDatasetIds.push_back(val);
    }

    return true;
}

wxString asParametersForecast::GetPreprocessArchiveDataId(int iStep, int iPtor, int iPre) const
{
    if (m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveDataIds.size() >= (unsigned) (iPre + 1)) {
        return m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveDataIds[iPre];
    } else {
        wxLogError(_("Trying to access to an element outside of preprocessArchiveDatasetIds in the parameters object."));
        return wxEmptyString;
    }
}

bool asParametersForecast::SetPreprocessArchiveDataId(int iStep, int iPtor, int iPre, const wxString &val)
{
    if (val.IsEmpty()) {
        wxLogError(_("The provided value for the preprocess archive data ID is null"));
        return false;
    }

    if (m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveDataIds.size() >= (unsigned) (iPre + 1)) {
        m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveDataIds[iPre] = val;
    } else {
        m_stepsForecast[iStep].predictors[iPtor].preprocessArchiveDataIds.push_back(val);
    }

    return true;
}

wxString asParametersForecast::GetPreprocessRealtimeDatasetId(int iStep, int iPtor, int iPre) const
{
    if (m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeDatasetIds.size() >= (unsigned) (iPre + 1)) {
        return m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeDatasetIds[iPre];
    } else {
        wxLogError(_("Trying to access to an element outside of preprocessRealtimeDatasetIds in the parameters object."));
        return wxEmptyString;
    }
}

bool asParametersForecast::SetPreprocessRealtimeDatasetId(int iStep, int iPtor, int iPre, const wxString &val)
{
    if (val.IsEmpty()) {
        wxLogError(_("The provided value for the preprocess realtime dataset ID is null"));
        return false;
    }

    if (m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeDatasetIds.size() >= (unsigned) (iPre + 1)) {
        m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeDatasetIds[iPre] = val;
    } else {
        m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeDatasetIds.push_back(val);
    }

    return true;
}

wxString asParametersForecast::GetPreprocessRealtimeDataId(int iStep, int iPtor, int iPre) const
{
    if (m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeDataIds.size() >= (unsigned) (iPre + 1)) {
        return m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeDataIds[iPre];
    } else {
        wxLogError(_("Trying to access to an element outside of preprocessRealtimeDatasetIds in the parameters object."));
        return wxEmptyString;
    }
}

bool asParametersForecast::SetPreprocessRealtimeDataId(int iStep, int iPtor, int iPre, const wxString &val)
{
    if (val.IsEmpty()) {
        wxLogError(_("The provided value for the preprocess realtime data ID is null"));
        return false;
    }

    if (m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeDataIds.size() >= (unsigned) (iPre + 1)) {
        m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeDataIds[iPre] = val;
    } else {
        m_stepsForecast[iStep].predictors[iPtor].preprocessRealtimeDataIds.push_back(val);
    }

    return true;
}
