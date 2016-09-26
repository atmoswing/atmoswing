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

#include <asFileParametersForecast.h>

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
    asLogMessage(_("Loading parameters file."));

    if (filePath.IsEmpty()) {
        asLogError(_("The given path to the parameters file is empty."));
        return false;
    }

    asFileParametersForecast fileParams(filePath, asFile::ReadOnly);
    if (!fileParams.Open())
        return false;

    if (!fileParams.CheckRootElement())
        return false;

    int i_step = 0;
    wxXmlNode *nodeProcess = fileParams.GetRoot()->GetChildren();
    while (nodeProcess) {

        // Description
        if (nodeProcess->GetName() == "description") {
            if(!ParseDescription(fileParams, nodeProcess))
                return false;

        } else if (nodeProcess->GetName() == "time_properties") {
            if(!ParseTimeProperties(fileParams, nodeProcess))
                return false;

        } else if (nodeProcess->GetName() == "analog_dates") {
            AddStep();
            if(!ParseAnalogDatesParams(fileParams, i_step, nodeProcess))
                return false;
            i_step++;

        } else if (nodeProcess->GetName() == "analog_values") {
            if(!ParseAnalogValuesParams(fileParams, nodeProcess))
                return false;

        } else {
            fileParams.UnknownNode(nodeProcess);
        }

        nodeProcess = nodeProcess->GetNext();
    }

    // Set properties
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

    asLogMessage(_("Parameters file loaded."));

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
            wxASSERT(GetMethodIdDisplay().size() > 0);
        } else if (nodeParam->GetName() == "specific_tag") {
            SetSpecificTag(fileParams.GetString(nodeParam));
        } else if (nodeParam->GetName() == "specific_tag_display") {
            SetSpecificTagDisplay(fileParams.GetString(nodeParam));
            wxASSERT(GetSpecificTagDisplay().size() > 0);
        } else if (nodeParam->GetName() == "description") {
            SetDescription(fileParams.GetString(nodeParam));
            wxASSERT(GetDescription().size() > 0);
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

bool asParametersForecast::ParseAnalogDatesParams(asFileParametersForecast &fileParams, int i_step,
                                                  const wxXmlNode *nodeProcess)
{
    int i_ptor = 0;
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "analogs_number") {
            if (!SetAnalogsNumberLeadTimeVector(i_step, fileParams.GetVectorInt(nodeParamBlock)))
                return false;
        } else if (nodeParamBlock->GetName() == "predictor") {
            AddPredictor(i_step);
            AddPredictorForecast(m_stepsForecast[i_step]);
            SetPreprocess(i_step, i_ptor, false);
            SetPreload(i_step, i_ptor, false);
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "preload") {
                    SetPreload(i_step, i_ptor, fileParams.GetBool(nodeParam));
                } else if (nodeParam->GetName() == "preprocessing") {
                    SetPreprocess(i_step, i_ptor, true);
                    if(!ParsePreprocessedPredictors(fileParams, i_step, i_ptor, nodeParam))
                        return false;
                } else if (nodeParam->GetName() == "realtime_dataset_id") {
                    if (!SetPredictorRealtimeDatasetId(i_step, i_ptor, fileParams.GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "realtime_data_id") {
                    if (!SetPredictorRealtimeDataId(i_step, i_ptor, fileParams.GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "archive_dataset_id") {
                    if (!SetPredictorArchiveDatasetId(i_step, i_ptor, fileParams.GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "archive_data_id") {
                    if (!SetPredictorArchiveDataId(i_step, i_ptor, fileParams.GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "level") {
                    if (!SetPredictorLevel(i_step, i_ptor, fileParams.GetFloat(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "time") {
                    if (!SetPredictorTimeHours(i_step, i_ptor, fileParams.GetDouble(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "spatial_window") {
                    wxXmlNode *nodeWindow = nodeParam->GetChildren();
                    while (nodeWindow) {
                        if (nodeWindow->GetName() == "grid_type") {
                            if (!SetPredictorGridType(i_step, i_ptor, fileParams.GetString(nodeWindow, "regular")))
                                return false;
                        } else if (nodeWindow->GetName() == "x_min") {
                            if (!SetPredictorXmin(i_step, i_ptor, fileParams.GetDouble(nodeWindow)))
                                return false;
                        } else if (nodeWindow->GetName() == "x_points_nb") {
                            if (!SetPredictorXptsnb(i_step, i_ptor, fileParams.GetInt(nodeWindow)))
                                return false;
                        } else if (nodeWindow->GetName() == "x_step") {
                            if (!SetPredictorXstep(i_step, i_ptor, fileParams.GetDouble(nodeWindow)))
                                return false;
                        } else if (nodeWindow->GetName() == "y_min") {
                            if (!SetPredictorYmin(i_step, i_ptor, fileParams.GetDouble(nodeWindow)))
                                return false;
                        } else if (nodeWindow->GetName() == "y_points_nb") {
                            if (!SetPredictorYptsnb(i_step, i_ptor, fileParams.GetInt(nodeWindow)))
                                return false;
                        } else if (nodeWindow->GetName() == "y_step") {
                            if (!SetPredictorYstep(i_step, i_ptor, fileParams.GetDouble(nodeWindow)))
                                return false;
                        } else {
                            fileParams.UnknownNode(nodeWindow);
                        }
                        nodeWindow = nodeWindow->GetNext();
                    }
                } else if (nodeParam->GetName() == "criteria") {
                    if (!SetPredictorCriteria(i_step, i_ptor, fileParams.GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "weight") {
                    if (!SetPredictorWeight(i_step, i_ptor, fileParams.GetFloat(nodeParam)))
                        return false;
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
            i_ptor++;
        } else {
            fileParams.UnknownNode(nodeParamBlock);
        }
        nodeParamBlock = nodeParamBlock->GetNext();
    }
    return true;
}

bool asParametersForecast::ParsePreprocessedPredictors(asFileParametersForecast &fileParams, int i_step, int i_ptor,
                                                       const wxXmlNode *nodeParam)
{
    int i_dataset = 0;
    wxXmlNode *nodePreprocess = nodeParam->GetChildren();
    while (nodePreprocess) {
        if (nodePreprocess->GetName() == "preprocessing_method") {
            if (!SetPreprocessMethod(i_step, i_ptor, fileParams.GetString(nodePreprocess)))
                return false;
        } else if (nodePreprocess->GetName() == "preprocessing_data") {
            wxXmlNode *nodeParamPreprocess = nodePreprocess->GetChildren();
            while (nodeParamPreprocess) {
                if (nodeParamPreprocess->GetName() == "realtime_dataset_id") {
                    if (!SetPreprocessRealtimeDatasetId(i_step, i_ptor, i_dataset,
                                                        fileParams.GetString(nodeParamPreprocess)))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "realtime_data_id") {
                    if (!SetPreprocessRealtimeDataId(i_step, i_ptor, i_dataset,
                                                     fileParams.GetString(nodeParamPreprocess)))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "archive_dataset_id") {
                    if (!SetPreprocessArchiveDatasetId(i_step, i_ptor, i_dataset,
                                                       fileParams.GetString(nodeParamPreprocess)))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "archive_data_id") {
                    if (!SetPreprocessArchiveDataId(i_step, i_ptor, i_dataset,
                                                    fileParams.GetString(nodeParamPreprocess)))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "level") {
                    if (!SetPreprocessLevel(i_step, i_ptor, i_dataset, fileParams.GetFloat(nodeParamPreprocess)))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "time") {
                    if (!SetPreprocessTimeHours(i_step, i_ptor, i_dataset, fileParams.GetDouble(nodeParamPreprocess)))
                        return false;
                } else {
                    fileParams.UnknownNode(nodeParamPreprocess);
                }
                nodeParamPreprocess = nodeParamPreprocess->GetNext();
            }
            i_dataset++;
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
    if (GetLeadTimeDaysVector().size() == 0) {
        asLogError(_("The lead times were not provided in the parameters file."));
        return false;
    }

    if (GetArchiveStart() <= 0) {
        asLogError(_("The beginning of the archive period was not provided in the parameters file."));
        return false;
    }

    if (GetArchiveEnd() <= 0) {
        asLogError(_("The end of the archive period was not provided in the parameters file."));
        return false;
    }

    if (GetTimeArrayTargetTimeStepHours() <= 0) {
        asLogError(_("The time step was not provided in the parameters file."));
        return false;
    }

    if (GetTimeArrayAnalogsTimeStepHours() <= 0) {
        asLogError(_("The time step was not provided in the parameters file."));
        return false;
    }

    if (GetTimeArrayAnalogsMode().CmpNoCase("interval_days") == 0 ||
        GetTimeArrayAnalogsMode().CmpNoCase("IntervalDays") == 0) {
        if (GetTimeArrayAnalogsIntervalDays() <= 0) {
            asLogError(_("The interval days for the analogs preselection was not provided in the parameters file."));
            return false;
        }
        if (GetTimeArrayAnalogsExcludeDays() <= 0) {
            asLogError(
                    _("The number of days to exclude around the target date was not provided in the parameters file."));
            return false;
        }
    }

    // Analog dates
    for (int i = 0; i < GetStepsNb(); i++) {
        if (GetAnalogsNumberLeadTimeVector(i).size() != GetLeadTimeDaysVector().size()) {
            asLogError(wxString::Format(
                    _("The length of the analogs numbers (step %d) do not match the number of lead times."), i));
            return false;
        }

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                if (GetPreprocessMethod(i, j).IsEmpty()) {
                    asLogError(wxString::Format(
                            _("The preprocessing method (step %d, predictor %d) was not provided in the parameters file."),
                            i, j));
                    return false;
                }

                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (GetPreprocessRealtimeDatasetId(i, j, k).IsEmpty()) {
                        asLogError(wxString::Format(
                                _("The realtime dataset for preprocessing (step %d, predictor %d) was not provided in the parameters file."),
                                i, j));
                        return false;
                    }
                    if (GetPreprocessRealtimeDataId(i, j, k).IsEmpty()) {
                        asLogError(wxString::Format(
                                _("The realtime data for preprocessing (step %d, predictor %d) was not provided in the parameters file."),
                                i, j));
                        return false;
                    }
                    if (GetPreprocessArchiveDatasetId(i, j, k).IsEmpty()) {
                        asLogError(wxString::Format(
                                _("The archive dataset for preprocessing (step %d, predictor %d) was not provided in the parameters file."),
                                i, j));
                        return false;
                    }
                    if (GetPreprocessArchiveDataId(i, j, k).IsEmpty()) {
                        asLogError(wxString::Format(
                                _("The archive data for preprocessing (step %d, predictor %d) was not provided in the parameters file."),
                                i, j));
                        return false;
                    }
                }
            } else {
                if (GetPredictorRealtimeDatasetId(i, j).IsEmpty()) {
                    asLogError(wxString::Format(
                            _("The realtime dataset (step %d, predictor %d) was not provided in the parameters file."),
                            i, j));
                    return false;
                }
                if (GetPredictorRealtimeDataId(i, j).IsEmpty()) {
                    asLogError(wxString::Format(
                            _("The realtime data (step %d, predictor %d) was not provided in the parameters file."), i,
                            j));
                    return false;
                }
                if (GetPredictorArchiveDatasetId(i, j).IsEmpty()) {
                    asLogError(wxString::Format(
                            _("The archive dataset (step %d, predictor %d) was not provided in the parameters file."),
                            i, j));
                    return false;
                }
                if (GetPredictorArchiveDataId(i, j).IsEmpty()) {
                    asLogError(wxString::Format(
                            _("The archive data (step %d, predictor %d) was not provided in the parameters file."), i,
                            j));
                    return false;
                }
            }

            if (GetPredictorGridType(i, j).IsEmpty()) {
                asLogError(
                        wxString::Format(_("The grid type (step %d, predictor %d) is empty in the parameters file."), i,
                                         j));
                return false;
            }
            if (GetPredictorXptsnb(i, j) == 0) {
                asLogError(wxString::Format(
                        _("The X points nb value (step %d, predictor %d) was not provided in the parameters file."), i,
                        j));
                return false;
            }
            if (GetPredictorYptsnb(i, j) == 0) {
                asLogError(wxString::Format(
                        _("The Y points nb value (step %d, predictor %d) was not provided in the parameters file."), i,
                        j));
                return false;
            }
            if (GetPredictorCriteria(i, j).IsEmpty()) {
                asLogError(wxString::Format(
                        _("The criteria (step %d, predictor %d) was not provided in the parameters file."), i, j));
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

bool asParametersForecast::SetLeadTimeDaysVector(VectorInt val)
{
    if (val.size() < 1) {
        asLogError(_("The provided 'lead time (days)' vector is empty."));
        return false;
    } else {
        for (int i = 0; i < (int) val.size(); i++) {
            if (asTools::IsNaN(val[i])) {
                asLogError(_("There are NaN values in the provided 'lead time (days)' vector."));
                return false;
            }
        }
    }
    m_leadTimeDaysVect = val;
    return true;
}

bool asParametersForecast::SetAnalogsNumberLeadTimeVector(int i_step, VectorInt val)
{
    if (val.size() < 1) {
        asLogError(_("The provided analogs numbers vector (fct of the lead time) is empty."));
        return false;
    } else {
        for (int i = 0; i < (int) val.size(); i++) {
            if (asTools::IsNaN(val[i])) {
                asLogError(_("There are NaN values in the provided analogs numbers vector (fct of the lead time)."));
                return false;
            }
        }
    }
    m_stepsForecast[i_step].analogsNumberLeadTime = val;
    return true;
}

bool asParametersForecast::SetPredictorArchiveDatasetId(int i_step, int i_predictor, const wxString &val)
{
    if (val.IsEmpty()) {
        asLogError(_("The provided value for the predictor archive dataset ID is null"));
        return false;
    }
    m_stepsForecast[i_step].predictors[i_predictor].archiveDatasetId = val;
    return true;
}

bool asParametersForecast::SetPredictorArchiveDataId(int i_step, int i_predictor, const wxString &val)
{
    if (val.IsEmpty()) {
        asLogError(_("The provided value for the predictor archive data ID is null"));
        return false;
    }
    m_stepsForecast[i_step].predictors[i_predictor].archiveDataId = val;
    return true;
}

bool asParametersForecast::SetPredictorRealtimeDatasetId(int i_step, int i_predictor, const wxString &val)
{
    if (val.IsEmpty()) {
        asLogError(_("The provided value for the predictor realtime dataset ID is null"));
        return false;
    }
    m_stepsForecast[i_step].predictors[i_predictor].realtimeDatasetId = val;
    return true;
}

bool asParametersForecast::SetPredictorRealtimeDataId(int i_step, int i_predictor, const wxString &val)
{
    if (val.IsEmpty()) {
        asLogError(_("The provided value for the predictor realtime data ID is null"));
        return false;
    }
    m_stepsForecast[i_step].predictors[i_predictor].realtimeDataId = val;
    return true;
}

wxString asParametersForecast::GetPreprocessArchiveDatasetId(int i_step, int i_predictor, int i_dataset) const
{
    if (m_stepsForecast[i_step].predictors[i_predictor].preprocessArchiveDatasetIds.size() >=
        (unsigned) (i_dataset + 1)) {
        return m_stepsForecast[i_step].predictors[i_predictor].preprocessArchiveDatasetIds[i_dataset];
    } else {
        asLogError(
                _("Trying to access to an element outside of preprocessArchiveDatasetIds in the parameters object."));
        return wxEmptyString;
    }
}

bool asParametersForecast::SetPreprocessArchiveDatasetId(int i_step, int i_predictor, int i_dataset,
                                                         const wxString &val)
{
    if (val.IsEmpty()) {
        asLogError(_("The provided value for the preprocess archive dataset ID is null"));
        return false;
    }

    if (m_stepsForecast[i_step].predictors[i_predictor].preprocessArchiveDatasetIds.size() >=
        (unsigned) (i_dataset + 1)) {
        m_stepsForecast[i_step].predictors[i_predictor].preprocessArchiveDatasetIds[i_dataset] = val;
    } else {
        m_stepsForecast[i_step].predictors[i_predictor].preprocessArchiveDatasetIds.push_back(val);
    }

    return true;
}

wxString asParametersForecast::GetPreprocessArchiveDataId(int i_step, int i_predictor, int i_dataset) const
{
    if (m_stepsForecast[i_step].predictors[i_predictor].preprocessArchiveDataIds.size() >= (unsigned) (i_dataset + 1)) {
        return m_stepsForecast[i_step].predictors[i_predictor].preprocessArchiveDataIds[i_dataset];
    } else {
        asLogError(
                _("Trying to access to an element outside of preprocessArchiveDatasetIds in the parameters object."));
        return wxEmptyString;
    }
}

bool asParametersForecast::SetPreprocessArchiveDataId(int i_step, int i_predictor, int i_dataset, const wxString &val)
{
    if (val.IsEmpty()) {
        asLogError(_("The provided value for the preprocess archive data ID is null"));
        return false;
    }

    if (m_stepsForecast[i_step].predictors[i_predictor].preprocessArchiveDataIds.size() >= (unsigned) (i_dataset + 1)) {
        m_stepsForecast[i_step].predictors[i_predictor].preprocessArchiveDataIds[i_dataset] = val;
    } else {
        m_stepsForecast[i_step].predictors[i_predictor].preprocessArchiveDataIds.push_back(val);
    }

    return true;
}

wxString asParametersForecast::GetPreprocessRealtimeDatasetId(int i_step, int i_predictor, int i_dataset) const
{
    if (m_stepsForecast[i_step].predictors[i_predictor].preprocessRealtimeDatasetIds.size() >=
        (unsigned) (i_dataset + 1)) {
        return m_stepsForecast[i_step].predictors[i_predictor].preprocessRealtimeDatasetIds[i_dataset];
    } else {
        asLogError(
                _("Trying to access to an element outside of preprocessRealtimeDatasetIds in the parameters object."));
        return wxEmptyString;
    }
}

bool asParametersForecast::SetPreprocessRealtimeDatasetId(int i_step, int i_predictor, int i_dataset,
                                                          const wxString &val)
{
    if (val.IsEmpty()) {
        asLogError(_("The provided value for the preprocess realtime dataset ID is null"));
        return false;
    }

    if (m_stepsForecast[i_step].predictors[i_predictor].preprocessRealtimeDatasetIds.size() >=
        (unsigned) (i_dataset + 1)) {
        m_stepsForecast[i_step].predictors[i_predictor].preprocessRealtimeDatasetIds[i_dataset] = val;
    } else {
        m_stepsForecast[i_step].predictors[i_predictor].preprocessRealtimeDatasetIds.push_back(val);
    }

    return true;
}

wxString asParametersForecast::GetPreprocessRealtimeDataId(int i_step, int i_predictor, int i_dataset) const
{
    if (m_stepsForecast[i_step].predictors[i_predictor].preprocessRealtimeDataIds.size() >=
        (unsigned) (i_dataset + 1)) {
        return m_stepsForecast[i_step].predictors[i_predictor].preprocessRealtimeDataIds[i_dataset];
    } else {
        asLogError(
                _("Trying to access to an element outside of preprocessRealtimeDatasetIds in the parameters object."));
        return wxEmptyString;
    }
}

bool asParametersForecast::SetPreprocessRealtimeDataId(int i_step, int i_predictor, int i_dataset, const wxString &val)
{
    if (val.IsEmpty()) {
        asLogError(_("The provided value for the preprocess realtime data ID is null"));
        return false;
    }

    if (m_stepsForecast[i_step].predictors[i_predictor].preprocessRealtimeDataIds.size() >=
        (unsigned) (i_dataset + 1)) {
        m_stepsForecast[i_step].predictors[i_predictor].preprocessRealtimeDataIds[i_dataset] = val;
    } else {
        m_stepsForecast[i_step].predictors[i_predictor].preprocessRealtimeDataIds.push_back(val);
    }

    return true;
}
