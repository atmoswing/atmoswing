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
 * Portions Copyright 2017 Pascal Horton, University of Bern.
 */

#include "asParametersDownscaling.h"

#include "asAreaCompGrid.h"
#include "asFileParametersDownscaling.h"

asParametersDownscaling::asParametersDownscaling() : asParameters(), m_downscalingStart(NaNd), m_downscalingEnd(NaNd) {}

asParametersDownscaling::~asParametersDownscaling() {}

void asParametersDownscaling::AddStep() {
    asParameters::AddStep();
    ParamsStepProj stepVect;
    m_stepsProj.push_back(stepVect);
}

void asParametersDownscaling::AddPredictorProj(ParamsStepProj &step) {
    ParamsPredictorProj predictor;

    predictor.datasetId = wxEmptyString;
    predictor.dataId = wxEmptyString;
    predictor.membersNb = 1;

    step.predictors.push_back(predictor);
}

bool asParametersDownscaling::LoadFromFile(const wxString &filePath) {
    wxLogVerbose(_("Loading parameters file."));

    if (filePath.IsEmpty()) {
        wxLogError(_("The given path to the parameters file is empty."));
        return false;
    }

    asFileParametersDownscaling fileParams(filePath, asFile::ReadOnly);
    if (!fileParams.Open()) return false;

    if (!fileParams.CheckRootElement()) return false;

    int iStep = 0;
    wxXmlNode *nodeProcess = fileParams.GetRoot()->GetChildren();
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

bool asParametersDownscaling::ParseDescription(asFileParametersDownscaling &fileParams, const wxXmlNode *nodeProcess) {
    wxXmlNode *nodeParam = nodeProcess->GetChildren();
    while (nodeParam) {
        if (nodeParam->GetName() == "method_id") {
            SetMethodId(fileParams.GetString(nodeParam));
        } else if (nodeParam->GetName() == "method_id_display") {
            SetMethodIdDisplay(fileParams.GetString(nodeParam));
        } else if (nodeParam->GetName() == "model") {
            SetModel(fileParams.GetString(nodeParam));
        } else if (nodeParam->GetName() == "scenario") {
            SetScenario(fileParams.GetString(nodeParam));
        } else if (nodeParam->GetName() == "specific_tag") {
            SetSpecificTag(fileParams.GetString(nodeParam));
        } else if (nodeParam->GetName() == "specific_tag_display") {
            SetSpecificTagDisplay(fileParams.GetString(nodeParam));
        } else if (nodeParam->GetName() == "description") {
            SetDescription(fileParams.GetString(nodeParam));
        } else {
            fileParams.UnknownNode(nodeParam);
        }
        nodeParam = nodeParam->GetNext();
    }

    return true;
}

bool asParametersDownscaling::ParseTimeProperties(asFileParametersDownscaling &fileParams,
                                                  const wxXmlNode *nodeProcess) {
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "archive_period") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "start_year") {
                    if (!SetArchiveYearStart(fileParams.GetInt(nodeParam))) return false;
                } else if (nodeParam->GetName() == "end_year") {
                    if (!SetArchiveYearEnd(fileParams.GetInt(nodeParam))) return false;
                } else if (nodeParam->GetName() == "start") {
                    if (!SetArchiveStart(fileParams.GetString(nodeParam))) return false;
                } else if (nodeParam->GetName() == "end") {
                    if (!SetArchiveEnd(fileParams.GetString(nodeParam))) return false;
                } else if (nodeParam->GetName() == "time_step") {
                    if (!SetAnalogsTimeStepHours(fileParams.GetDouble(nodeParam))) return false;
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
        } else if (nodeParamBlock->GetName() == "downscaling_period") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "start_year") {
                    if (!SetDownscalingYearStart(fileParams.GetInt(nodeParam))) return false;
                } else if (nodeParam->GetName() == "end_year") {
                    if (!SetDownscalingYearEnd(fileParams.GetInt(nodeParam))) return false;
                } else if (nodeParam->GetName() == "start") {
                    if (!SetDownscalingStart(fileParams.GetString(nodeParam))) return false;
                } else if (nodeParam->GetName() == "end") {
                    if (!SetDownscalingEnd(fileParams.GetString(nodeParam))) return false;
                } else if (nodeParam->GetName() == "time_step") {
                    if (!SetTargetTimeStepHours(fileParams.GetDouble(nodeParam))) return false;
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
        } else if (nodeParamBlock->GetName() == "time_step") {
            if (!SetTargetTimeStepHours(fileParams.GetDouble(nodeParamBlock))) return false;
            if (!SetAnalogsTimeStepHours(fileParams.GetDouble(nodeParamBlock))) return false;
        } else if (nodeParamBlock->GetName() == "time_array_target") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "time_array") {
                    if (!SetTimeArrayTargetMode(fileParams.GetString(nodeParam))) return false;
                } else if (nodeParam->GetName() == "predictand_serie_name") {
                    if (!SetTimeArrayTargetPredictandSerieName(fileParams.GetString(nodeParam))) return false;
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
        } else if (nodeParamBlock->GetName() == "time_array_analogs") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "time_array") {
                    if (!SetTimeArrayAnalogsMode(fileParams.GetString(nodeParam))) return false;
                } else if (nodeParam->GetName() == "interval_days") {
                    if (!SetAnalogsIntervalDays(fileParams.GetInt(nodeParam))) return false;
                } else if (nodeParam->GetName() == "exclude_days") {
                    if (!SetAnalogsExcludeDays(fileParams.GetInt(nodeParam))) return false;
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

bool asParametersDownscaling::ParseAnalogDatesParams(asFileParametersDownscaling &fileParams, int iStep,
                                                     const wxXmlNode *nodeProcess) {
    int iPtor = 0;
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "analogs_number") {
            if (!SetAnalogsNumber(iStep, asFileParametersDownscaling::GetInt(nodeParamBlock))) return false;
        } else if (nodeParamBlock->GetName() == "predictor") {
            AddPredictor(iStep);
            AddPredictorProj(m_stepsProj[iStep]);
            SetPreprocess(iStep, iPtor, false);
            SetPreload(iStep, iPtor, false);
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "preload") {
                    SetPreload(iStep, iPtor, asFileParametersDownscaling::GetBool(nodeParam));
                } else if (nodeParam->GetName() == "standardize") {
                    SetStandardize(iStep, iPtor, asFileParametersDownscaling::GetBool(nodeParam));
                } else if (nodeParam->GetName() == "standardize_mean") {
                    SetStandardizeMean(iStep, iPtor, asFileParametersDownscaling::GetDouble(nodeParam));
                } else if (nodeParam->GetName() == "standardize_sd") {
                    SetStandardizeSd(iStep, iPtor, asFileParametersDownscaling::GetDouble(nodeParam));
                } else if (nodeParam->GetName() == "preprocessing") {
                    SetPreprocess(iStep, iPtor, true);
                    if (!ParsePreprocessedPredictors(fileParams, iStep, iPtor, nodeParam)) return false;
                } else if (nodeParam->GetName() == "proj_dataset_id") {
                    if (!SetPredictorProjDatasetId(iStep, iPtor, asFileParametersDownscaling::GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "proj_data_id") {
                    if (!SetPredictorProjDataId(iStep, iPtor, asFileParametersDownscaling::GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "archive_dataset_id") {
                    if (!SetPredictorDatasetId(iStep, iPtor, asFileParametersDownscaling::GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "archive_data_id") {
                    if (!SetPredictorDataId(iStep, iPtor, asFileParametersDownscaling::GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "level") {
                    if (!SetPredictorLevel(iStep, iPtor, asFileParametersDownscaling::GetFloat(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "time") {
                    if (!SetPredictorHour(iStep, iPtor, asFileParametersDownscaling::GetDouble(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "members") {
                    if (!SetPredictorMembersNb(iStep, iPtor, asFileParametersDownscaling::GetInt(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "spatial_window") {
                    wxXmlNode *nodeWindow = nodeParam->GetChildren();
                    while (nodeWindow) {
                        if (nodeWindow->GetName() == "grid_type") {
                            if (!SetPredictorGridType(iStep, iPtor,
                                                      asFileParametersDownscaling::GetString(nodeWindow, "regular")))
                                return false;
                        } else if (nodeWindow->GetName() == "x_min") {
                            if (!SetPredictorXmin(iStep, iPtor, asFileParametersDownscaling::GetDouble(nodeWindow)))
                                return false;
                        } else if (nodeWindow->GetName() == "x_points_nb") {
                            if (!SetPredictorXptsnb(iStep, iPtor, asFileParametersDownscaling::GetInt(nodeWindow)))
                                return false;
                        } else if (nodeWindow->GetName() == "x_step") {
                            if (!SetPredictorXstep(iStep, iPtor, asFileParametersDownscaling::GetDouble(nodeWindow)))
                                return false;
                        } else if (nodeWindow->GetName() == "y_min") {
                            if (!SetPredictorYmin(iStep, iPtor, asFileParametersDownscaling::GetDouble(nodeWindow)))
                                return false;
                        } else if (nodeWindow->GetName() == "y_points_nb") {
                            if (!SetPredictorYptsnb(iStep, iPtor, asFileParametersDownscaling::GetInt(nodeWindow)))
                                return false;
                        } else if (nodeWindow->GetName() == "y_step") {
                            if (!SetPredictorYstep(iStep, iPtor, asFileParametersDownscaling::GetDouble(nodeWindow)))
                                return false;
                        } else {
                            fileParams.UnknownNode(nodeWindow);
                        }
                        nodeWindow = nodeWindow->GetNext();
                    }
                } else if (nodeParam->GetName() == "criteria") {
                    if (!SetPredictorCriteria(iStep, iPtor, asFileParametersDownscaling::GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "weight") {
                    if (!SetPredictorWeight(iStep, iPtor, asFileParametersDownscaling::GetFloat(nodeParam)))
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

bool asParametersDownscaling::ParsePreprocessedPredictors(asFileParametersDownscaling &fileParams, int iStep, int iPtor,
                                                          const wxXmlNode *nodeParam) {
    int iPre = 0;
    wxXmlNode *nodePreprocess = nodeParam->GetChildren();
    while (nodePreprocess) {
        if (nodePreprocess->GetName() == "preprocessing_method") {
            if (!SetPreprocessMethod(iStep, iPtor, fileParams.GetString(nodePreprocess))) return false;
        } else if (nodePreprocess->GetName() == "preprocessing_data") {
            wxXmlNode *nodeParamPreprocess = nodePreprocess->GetChildren();
            while (nodeParamPreprocess) {
                if (nodeParamPreprocess->GetName() == "proj_dataset_id") {
                    if (!SetPreprocessProjDatasetId(iStep, iPtor, iPre, fileParams.GetString(nodeParamPreprocess)))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "proj_data_id") {
                    if (!SetPreprocessProjDataId(iStep, iPtor, iPre, fileParams.GetString(nodeParamPreprocess)))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "archive_dataset_id") {
                    if (!SetPreprocessDatasetId(iStep, iPtor, iPre, fileParams.GetString(nodeParamPreprocess)))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "archive_data_id") {
                    if (!SetPreprocessDataId(iStep, iPtor, iPre, fileParams.GetString(nodeParamPreprocess)))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "level") {
                    if (!SetPreprocessLevel(iStep, iPtor, iPre, fileParams.GetFloat(nodeParamPreprocess))) return false;
                } else if (nodeParamPreprocess->GetName() == "time") {
                    if (!SetPreprocessHour(iStep, iPtor, iPre, fileParams.GetDouble(nodeParamPreprocess))) return false;
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

bool asParametersDownscaling::ParseAnalogValuesParams(asFileParametersDownscaling &fileParams,
                                                      const wxXmlNode *nodeProcess) {
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "predictand") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "station_id" || nodeParam->GetName() == "station_ids") {
                    if (!SetPredictandStationIdsVector(fileParams.GetStationIdsVector(nodeParam))) return false;
                } else if (nodeParam->GetName() == "time") {
                    if (!SetPredictandTimeHours(fileParams.GetDouble(nodeParam))) return false;
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

bool asParametersDownscaling::InputsOK() const {
    // Time properties
    if (asIsNaN(GetArchiveStart())) {
        wxLogError(_("The beginning of the archive period was not provided in the parameters file."));
        return false;
    }

    if (asIsNaN(GetArchiveEnd())) {
        wxLogError(_("The end of the archive period was not provided in the parameters file."));
        return false;
    }

    if (asIsNaN(GetDownscalingStart())) {
        wxLogError(_("The beginning of the downscaling period was not provided in the parameters file."));
        return false;
    }

    if (asIsNaN(GetDownscalingEnd())) {
        wxLogError(_("The end of the downscaling period was not provided in the parameters file."));
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
            wxLogError(_("The interval days for the analogs preselection was not provided in the parameters file."));
            return false;
        }
        if (GetAnalogsExcludeDays() <= 0) {
            wxLogError(
                _("The number of days to exclude around the target date was not provided in the parameters file."));
            return false;
        }
    }

    // Analog dates
    for (int i = 0; i < GetStepsNb(); i++) {
        if (GetAnalogsNumber(i) == 0) {
            wxLogError(
                wxString::Format(_("The number of analogs (step %d) was not provided in the parameters file."), i));
            return false;
        }

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                if (GetPreprocessMethod(i, j).IsEmpty()) {
                    wxLogError(
                        _("The preprocessing method (step %d, predictor %d) was not provided in the parameters file."),
                        i, j);
                    return false;
                }

                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (GetPreprocessProjDatasetId(i, j, k).IsEmpty()) {
                        wxLogError(_("The projection dataset for preprocessing (step %d, predictor %d) was not "
                                     "provided in the parameters file."),
                                   i, j);
                        return false;
                    }
                    if (GetPreprocessProjDataId(i, j, k).IsEmpty()) {
                        wxLogError(_("The projection data for preprocessing (step %d, predictor %d) was not provided "
                                     "in the parameters file."),
                                   i, j);
                        return false;
                    }
                    if (GetPreprocessProjDataId(i, j, k).IsEmpty()) {
                        wxLogError(_("The projection data for preprocessing (step %d, predictor %d) was not provided "
                                     "in the parameters file."),
                                   i, j);
                        return false;
                    }
                    if (GetPreprocessDatasetId(i, j, k).IsEmpty()) {
                        wxLogError(_("The archive dataset for preprocessing (step %d, predictor %d) was not provided "
                                     "in the parameters file."),
                                   i, j);
                        return false;
                    }
                    if (GetPreprocessDataId(i, j, k).IsEmpty()) {
                        wxLogError(_("The archive data for preprocessing (step %d, predictor %d) was not provided in "
                                     "the parameters file."),
                                   i, j);
                        return false;
                    }
                }
            } else {
                if (GetPredictorProjDatasetId(i, j).IsEmpty()) {
                    wxLogError(
                        _("The projection dataset (step %d, predictor %d) was not provided in the parameters file."), i,
                        j);
                    return false;
                }
                if (GetPredictorProjDataId(i, j).IsEmpty()) {
                    wxLogError(
                        _("The projection data (step %d, predictor %d) was not provided in the parameters file."), i,
                        j);
                    return false;
                }
                if (GetPredictorDatasetId(i, j).IsEmpty()) {
                    wxLogError(
                        _("The archive dataset (step %d, predictor %d) was not provided in the parameters file."), i,
                        j);
                    return false;
                }
                if (GetPredictorDataId(i, j).IsEmpty()) {
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

    // Analog values
    if (GetPredictandStationIdsVector().empty()) {
        wxLogWarning(_("The station ID was not provided in the parameters file (it can be on purpose)."));
        // allowed
    }

    return true;
}

bool asParametersDownscaling::FixTimeLimits() {
    double minHour = 200.0, maxHour = -50.0;
    for (int i = 0; i < GetStepsNb(); i++) {
        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    minHour = wxMin(GetPreprocessHour(i, j, k), minHour);
                    maxHour = wxMax(GetPreprocessHour(i, j, k), maxHour);
                }
            } else {
                minHour = wxMin(GetPredictorHour(i, j), minHour);
                maxHour = wxMax(GetPredictorHour(i, j), maxHour);
            }
        }
    }

    m_timeMinHours = minHour;
    m_timeMaxHours = maxHour;

    return true;
}

void asParametersDownscaling::InitValues() {
    wxASSERT(!m_predictandStationIdsVect.empty());

    // Initialize the parameters values with the first values of the vectors
    m_predictandStationIds = m_predictandStationIdsVect[0];

    // Fixes and checks
    FixWeights();
    FixCoordinates();
    FixAnalogsNb();
}

bool asParametersDownscaling::SetPredictandStationIdsVector(vvi val) {
    if (val.empty()) {
        wxLogError(_("The provided predictand ID vector is empty."));
        return false;
    } else {
        if (val[0].empty()) {
            wxLogError(_("The provided predictand ID vector is empty."));
            return false;
        }

        for (auto &i : val) {
            for (int j : i) {
                if (asIsNaN(j)) {
                    wxLogError(_("There are NaN values in the provided predictand ID vector."));
                    return false;
                }
            }
        }
    }

    m_predictandStationIdsVect = val;

    return true;
}

bool asParametersDownscaling::SetPredictorProjDatasetId(int iStep, int iPtor, const wxString &val) {
    if (val.IsEmpty()) {
        wxLogError(_("The provided value for the predictor projection dataset ID is null"));
        return false;
    }
    m_stepsProj[iStep].predictors[iPtor].datasetId = val;
    return true;
}

bool asParametersDownscaling::SetPredictorProjDataId(int iStep, int iPtor, const wxString &val) {
    if (val.IsEmpty()) {
        wxLogError(_("The provided value for the predictor projection data ID is null"));
        return false;
    }
    m_stepsProj[iStep].predictors[iPtor].dataId = val;
    return true;
}

wxString asParametersDownscaling::GetPreprocessProjDatasetId(int iStep, int iPtor, int iPre) const {
    if (m_stepsProj[iStep].predictors[iPtor].preprocessDatasetIds.size() >= iPre + 1) {
        return m_stepsProj[iStep].predictors[iPtor].preprocessDatasetIds[iPre];
    } else {
        wxLogError(_("Trying to access to an element outside of preprocessDatasetIds in the parameters object."));
        return wxEmptyString;
    }
}

bool asParametersDownscaling::SetPreprocessProjDatasetId(int iStep, int iPtor, int iPre, const wxString &val) {
    if (val.IsEmpty()) {
        wxLogError(_("The provided value for the preprocess projection dataset ID is null"));
        return false;
    }

    if (m_stepsProj[iStep].predictors[iPtor].preprocessDatasetIds.size() >= iPre + 1) {
        m_stepsProj[iStep].predictors[iPtor].preprocessDatasetIds[iPre] = val;
    } else {
        m_stepsProj[iStep].predictors[iPtor].preprocessDatasetIds.push_back(val);
    }

    return true;
}

wxString asParametersDownscaling::GetPreprocessProjDataId(int iStep, int iPtor, int iPre) const {
    if (m_stepsProj[iStep].predictors[iPtor].preprocessDataIds.size() >= iPre + 1) {
        return m_stepsProj[iStep].predictors[iPtor].preprocessDataIds[iPre];
    } else {
        wxLogError(_("Trying to access to an element outside of preprocessDatasetIds in the parameters object."));
        return wxEmptyString;
    }
}

bool asParametersDownscaling::SetPreprocessProjDataId(int iStep, int iPtor, int iPre, const wxString &val) {
    if (val.IsEmpty()) {
        wxLogError(_("The provided value for the preprocess proj data ID is null"));
        return false;
    }

    if (m_stepsProj[iStep].predictors[iPtor].preprocessDataIds.size() >= iPre + 1) {
        m_stepsProj[iStep].predictors[iPtor].preprocessDataIds[iPre] = val;
    } else {
        m_stepsProj[iStep].predictors[iPtor].preprocessDataIds.push_back(val);
    }

    return true;
}
