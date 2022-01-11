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

#include "asParameters.h"

#include <wx/tokenzr.h>

#include "asFileParameters.h"
#include "asFileText.h"

asParameters::asParameters()
    : m_archiveStart(NaNd),
      m_archiveEnd(NaNd),
      m_analogsIntervalDays(200),
      m_predictandStationIds(),
      m_timeMinHours(0),
      m_timeMaxHours(0),
      m_dateProcessed(asTime::GetStringTime(asTime::NowTimeStruct(asLOCAL))),
      m_timeArrayTargetMode("simple"),
      m_targetTimeStepHours(0),
      m_timeArrayTargetPredictandMinThreshold(0),
      m_timeArrayTargetPredictandMaxThreshold(0),
      m_timeArrayAnalogsMode("days_interval"),
      m_analogsTimeStepHours(0),
      m_analogsExcludeDays(0),
      m_predictandParameter(asPredictand::Precipitation),
      m_predictandTemporalResolution(asPredictand::Daily),
      m_predictandSpatialAggregation(asPredictand::Station),
      m_predictandTimeHours(0) {}

void asParameters::AddStep() {
    ParamsStep step;

    step.analogsNumber = 0;

    m_steps.push_back(step);
}

void asParameters::RemoveStep(int iStep) {
    wxASSERT(m_steps.size() > iStep);
    m_steps.erase(m_steps.begin() + iStep);
}

void asParameters::AddPredictor() {
    AddPredictor(m_steps[m_steps.size() - 1]);
}

void asParameters::AddPredictor(ParamsStep &step) {
    ParamsPredictor predictor;
    step.predictors.push_back(predictor);
}

void asParameters::AddPredictor(int iStep) {
    ParamsPredictor predictor;
    m_steps[iStep].predictors.push_back(predictor);
}

void asParameters::RemovePredictor(int iStep, int iPtor) {
    wxASSERT(m_steps.size() > iStep);
    wxASSERT(m_steps[iStep].predictors.size() > iPtor);
    m_steps[iStep].predictors.erase(m_steps[iStep].predictors.begin() + iPtor);
}

bool asParameters::LoadFromFile(const wxString &filePath) {
    wxLogVerbose(_("Loading parameters file."));

    if (filePath.IsEmpty()) {
        wxLogError(_("The given path to the parameters file is empty."));
        return false;
    }

    asFileParameters fileParams(filePath, asFile::ReadOnly);
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
    SetSpatialWindowProperties();
    SetPreloadingProperties();

    // Check inputs and init parameters
    if (!InputsOK()) return false;

    // Fixes
    FixTimeLimits();
    FixWeights();
    FixCoordinates();

    wxLogVerbose(_("Parameters file loaded."));

    return true;
}

bool asParameters::ParseDescription(asFileParameters &fileParams, const wxXmlNode *nodeProcess) {
    wxXmlNode *nodeParam = nodeProcess->GetChildren();
    while (nodeParam) {
        if (nodeParam->GetName() == "method_id") {
            SetMethodId(asFileParameters::GetString(nodeParam));
        } else if (nodeParam->GetName() == "method_id_display") {
            SetMethodIdDisplay(asFileParameters::GetString(nodeParam));
        } else if (nodeParam->GetName() == "specific_tag") {
            SetSpecificTag(asFileParameters::GetString(nodeParam));
        } else if (nodeParam->GetName() == "specific_tag_display") {
            SetSpecificTagDisplay(asFileParameters::GetString(nodeParam));
        } else if (nodeParam->GetName() == "description") {
            SetDescription(asFileParameters::GetString(nodeParam));
        } else {
            fileParams.UnknownNode(nodeParam);
        }
        nodeParam = nodeParam->GetNext();
    }
    return true;
}

bool asParameters::ParseTimeProperties(asFileParameters &fileParams, const wxXmlNode *nodeProcess) {
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "archive_period") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "start_year") {
                    SetArchiveYearStart(asFileParameters::GetInt(nodeParam));
                } else if (nodeParam->GetName() == "end_year") {
                    SetArchiveYearEnd(asFileParameters::GetInt(nodeParam));
                } else if (nodeParam->GetName() == "start") {
                    SetArchiveStart(asFileParameters::GetString(nodeParam));
                } else if (nodeParam->GetName() == "end") {
                    SetArchiveEnd(asFileParameters::GetString(nodeParam));
                } else if (nodeParam->GetName() == "time_step") {
                    SetAnalogsTimeStepHours(asFileParameters::GetDouble(nodeParam));
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
        } else if (nodeParamBlock->GetName() == "time_step") {
            SetTargetTimeStepHours(asFileParameters::GetDouble(nodeParamBlock));
            SetAnalogsTimeStepHours(asFileParameters::GetDouble(nodeParamBlock));
        } else if (nodeParamBlock->GetName() == "time_array_target") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "time_array") {
                    SetTimeArrayTargetMode(asFileParameters::GetString(nodeParam));
                } else if (nodeParam->GetName() == "predictand_serie_name") {
                    SetTimeArrayTargetPredictandSerieName(asFileParameters::GetString(nodeParam));
                } else if (nodeParam->GetName() == "predictand_min_threshold") {
                    SetTimeArrayTargetPredictandMinThreshold(asFileParameters::GetFloat(nodeParam));
                } else if (nodeParam->GetName() == "predictand_max_threshold") {
                    SetTimeArrayTargetPredictandMaxThreshold(asFileParameters::GetFloat(nodeParam));
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
        } else if (nodeParamBlock->GetName() == "time_array_analogs") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "time_array") {
                    SetTimeArrayAnalogsMode(asFileParameters::GetString(nodeParam));
                } else if (nodeParam->GetName() == "interval_days") {
                    SetAnalogsIntervalDays(asFileParameters::GetInt(nodeParam));
                } else if (nodeParam->GetName() == "exclude_days") {
                    SetAnalogsExcludeDays(asFileParameters::GetInt(nodeParam));
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

bool asParameters::ParseAnalogDatesParams(asFileParameters &fileParams, int iStep, const wxXmlNode *nodeProcess) {
    int iPtor = 0;
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "analogs_number") {
            SetAnalogsNumber(iStep, asFileParameters::GetInt(nodeParamBlock));
        } else if (nodeParamBlock->GetName() == "predictor") {
            AddPredictor(iStep);
            SetPreprocess(iStep, iPtor, false);
            SetPreload(iStep, iPtor, false);
            SetStandardize(iStep, iPtor, false);
            if (!ParsePredictors(fileParams, iStep, iPtor, nodeParamBlock)) return false;
            iPtor++;
        } else {
            fileParams.UnknownNode(nodeParamBlock);
        }
        nodeParamBlock = nodeParamBlock->GetNext();
    }

    return true;
}

bool asParameters::ParsePredictors(asFileParameters &fileParams, int iStep, int iPtor,
                                   const wxXmlNode *nodeParamBlock) {
    wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
    while (nodeParam) {
        if (nodeParam->GetName() == "preload") {
            SetPreload(iStep, iPtor, asFileParameters::GetBool(nodeParam));
        } else if (nodeParam->GetName() == "standardize") {
            SetStandardize(iStep, iPtor, asFileParameters::GetBool(nodeParam));
        } else if (nodeParam->GetName() == "standardize_mean") {
            SetStandardizeMean(iStep, iPtor, asFileParameters::GetDouble(nodeParam));
        } else if (nodeParam->GetName() == "standardize_sd") {
            SetStandardizeSd(iStep, iPtor, asFileParameters::GetDouble(nodeParam));
        } else if (nodeParam->GetName() == "preprocessing") {
            SetPreprocess(iStep, iPtor, true);
            if (!ParsePreprocessedPredictors(fileParams, iStep, iPtor, nodeParam)) return false;
        } else if (nodeParam->GetName() == "dataset_id") {
            SetPredictorDatasetId(iStep, iPtor, asFileParameters::GetString(nodeParam));
        } else if (nodeParam->GetName() == "data_id") {
            SetPredictorDataId(iStep, iPtor, asFileParameters::GetString(nodeParam));
        } else if (nodeParam->GetName() == "level") {
            SetPredictorLevel(iStep, iPtor, asFileParameters::GetFloat(nodeParam));
        } else if (nodeParam->GetName() == "time") {
            SetPredictorHour(iStep, iPtor, asFileParameters::GetDouble(nodeParam));
        } else if (nodeParam->GetName() == "members") {
            SetPredictorMembersNb(iStep, iPtor, asFileParameters::GetInt(nodeParam));
        } else if (nodeParam->GetName() == "spatial_window") {
            wxXmlNode *nodeWindow = nodeParam->GetChildren();
            while (nodeWindow) {
                if (nodeWindow->GetName() == "grid_type") {
                    SetPredictorGridType(iStep, iPtor, asFileParameters::GetString(nodeWindow, "regular"));
                } else if (nodeWindow->GetName() == "x_min") {
                    SetPredictorXmin(iStep, iPtor, asFileParameters::GetDouble(nodeWindow));
                } else if (nodeWindow->GetName() == "x_points_nb") {
                    SetPredictorXptsnb(iStep, iPtor, asFileParameters::GetInt(nodeWindow));
                } else if (nodeWindow->GetName() == "x_step") {
                    SetPredictorXstep(iStep, iPtor, asFileParameters::GetDouble(nodeWindow));
                } else if (nodeWindow->GetName() == "y_min") {
                    SetPredictorYmin(iStep, iPtor, asFileParameters::GetDouble(nodeWindow));
                } else if (nodeWindow->GetName() == "y_points_nb") {
                    SetPredictorYptsnb(iStep, iPtor, asFileParameters::GetInt(nodeWindow));
                } else if (nodeWindow->GetName() == "y_step") {
                    SetPredictorYstep(iStep, iPtor, asFileParameters::GetDouble(nodeWindow));
                } else {
                    fileParams.UnknownNode(nodeWindow);
                }
                nodeWindow = nodeWindow->GetNext();
            }
        } else if (nodeParam->GetName() == "criteria") {
            SetPredictorCriteria(iStep, iPtor, asFileParameters::GetString(nodeParam));
        } else if (nodeParam->GetName() == "weight") {
            SetPredictorWeight(iStep, iPtor, asFileParameters::GetFloat(nodeParam));
        } else {
            fileParams.UnknownNode(nodeParam);
        }
        nodeParam = nodeParam->GetNext();
    }

    return true;
}

bool asParameters::ParsePreprocessedPredictors(asFileParameters &fileParams, int iStep, int iPtor,
                                               const wxXmlNode *nodeParam) {
    int iPre = 0;
    wxXmlNode *nodePreprocess = nodeParam->GetChildren();
    while (nodePreprocess) {
        if (nodePreprocess->GetName() == "preprocessing_method") {
            SetPreprocessMethod(iStep, iPtor, asFileParameters::GetString(nodePreprocess));
        } else if (nodePreprocess->GetName() == "preprocessing_data") {
            wxXmlNode *nodeParamPreprocess = nodePreprocess->GetChildren();
            while (nodeParamPreprocess) {
                if (nodeParamPreprocess->GetName() == "dataset_id") {
                    SetPreprocessDatasetId(iStep, iPtor, iPre, asFileParameters::GetString(nodeParamPreprocess));
                } else if (nodeParamPreprocess->GetName() == "data_id") {
                    SetPreprocessDataId(iStep, iPtor, iPre, asFileParameters::GetString(nodeParamPreprocess));
                } else if (nodeParamPreprocess->GetName() == "level") {
                    SetPreprocessLevel(iStep, iPtor, iPre, asFileParameters::GetFloat(nodeParamPreprocess));
                } else if (nodeParamPreprocess->GetName() == "time") {
                    SetPreprocessHour(iStep, iPtor, iPre, asFileParameters::GetDouble(nodeParamPreprocess));
                } else if (nodeParamPreprocess->GetName() == "members") {
                    SetPreprocessMembersNb(iStep, iPtor, iPre, asFileParameters::GetInt(nodeParamPreprocess));
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

bool asParameters::ParseAnalogValuesParams(asFileParameters &fileParams, const wxXmlNode *nodeProcess) {
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "predictand") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "station_id") {
                    SetPredictandStationIds(asFileParameters::GetStationIds(asFileParameters::GetString(nodeParam)));
                } else if (nodeParam->GetName() == "time") {
                    SetPredictandTimeHours(asFileParameters::GetDouble(nodeParam));
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

bool asParameters::SetSpatialWindowProperties() {
    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        for (int iPtor = 0; iPtor < GetPredictorsNb(iStep); iPtor++) {
            if (GetPredictorXptsnb(iStep, iPtor) == 0) SetPredictorXptsnb(iStep, iPtor, 1);
            if (GetPredictorYptsnb(iStep, iPtor) == 0) SetPredictorYptsnb(iStep, iPtor, 1);

            double xShift = std::fmod(GetPredictorXmin(iStep, iPtor), GetPredictorXstep(iStep, iPtor));
            if (xShift < 0) xShift += GetPredictorXstep(iStep, iPtor);
            SetPredictorXshift(iStep, iPtor, xShift);

            double yShift = std::fmod(GetPredictorYmin(iStep, iPtor), GetPredictorYstep(iStep, iPtor));
            if (yShift < 0) yShift += GetPredictorYstep(iStep, iPtor);
            SetPredictorYshift(iStep, iPtor, yShift);

            if (GetPredictorXptsnb(iStep, iPtor) == 1 || GetPredictorYptsnb(iStep, iPtor) == 1) {
                SetPredictorFlatAllowed(iStep, iPtor, asFLAT_ALLOWED);
            }
        }
    }

    return true;
}

bool asParameters::SetPreloadingProperties() {
    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        for (int iPtor = 0; iPtor < GetPredictorsNb(iStep); iPtor++) {
            // Set maximum extent
            if (NeedsPreloading(iStep, iPtor)) {
                SetPreloadXmin(iStep, iPtor, GetPredictorXmin(iStep, iPtor));
                SetPreloadYmin(iStep, iPtor, GetPredictorYmin(iStep, iPtor));
                SetPreloadXptsnb(iStep, iPtor, GetPredictorXptsnb(iStep, iPtor));
                SetPreloadYptsnb(iStep, iPtor, GetPredictorYptsnb(iStep, iPtor));
            }

            // Change predictor properties when preprocessing
            if (NeedsPreprocessing(iStep, iPtor)) {
                if (GetPreprocessSize(iStep, iPtor) == 1) {
                    SetPredictorDatasetId(iStep, iPtor, GetPreprocessDatasetId(iStep, iPtor, 0));
                    SetPredictorDataId(iStep, iPtor, GetPreprocessDataId(iStep, iPtor, 0));
                    SetPredictorLevel(iStep, iPtor, GetPreprocessLevel(iStep, iPtor, 0));
                    SetPredictorHour(iStep, iPtor, GetPreprocessHour(iStep, iPtor, 0));
                } else {
                    SetPredictorDatasetId(iStep, iPtor, "mix");
                    SetPredictorDataId(iStep, iPtor, "mix");
                    SetPredictorLevel(iStep, iPtor, 0);
                    SetPredictorHour(iStep, iPtor, 0);
                }
            }

            // Set levels and time for preloading
            if (NeedsPreloading(iStep, iPtor) && !NeedsPreprocessing(iStep, iPtor)) {
                SetPreloadDataIds(iStep, iPtor, GetPredictorDataId(iStep, iPtor));
                SetPreloadLevels(iStep, iPtor, GetPredictorLevel(iStep, iPtor));
                SetPreloadHours(iStep, iPtor, GetPredictorHour(iStep, iPtor));
            } else if (NeedsPreloading(iStep, iPtor) && NeedsPreprocessing(iStep, iPtor)) {
                // Check the preprocessing method
                wxString method = GetPreprocessMethod(iStep, iPtor);
                vf preprocLevels;
                vd preprocHours;
                int preprocSize = GetPreprocessSize(iStep, iPtor);

                // Different actions depending on the preprocessing method.
                wxString msg =
                    _("The size of the provided predictors (%d) does not match the requirements (%d) in the "
                      "preprocessing %s method.");
                if (NeedsGradientPreprocessing(iStep, iPtor)) {
                    if (preprocSize != 1) {
                        wxLogError(msg, preprocSize, 1, "Gradient");
                        return false;
                    }
                    preprocLevels.push_back(GetPreprocessLevel(iStep, iPtor, 0));
                    preprocHours.push_back(GetPreprocessHour(iStep, iPtor, 0));
                } else if (method.IsSameAs("HumidityFlux")) {
                    if (preprocSize != 4) {
                        wxLogError(msg, preprocSize, 4, "HumidityFlux");
                        return false;
                    }
                    preprocLevels.push_back(GetPreprocessLevel(iStep, iPtor, 0));
                    preprocHours.push_back(GetPreprocessHour(iStep, iPtor, 0));
                } else if (method.IsSameAs("Multiplication") || method.IsSameAs("Multiply") ||
                           method.IsSameAs("HumidityIndex")) {
                    if (preprocSize != 2) {
                        wxLogError(msg, preprocSize, 2, "HumidityIndex");
                        return false;
                    }
                    preprocLevels.push_back(GetPreprocessLevel(iStep, iPtor, 0));
                    preprocHours.push_back(GetPreprocessHour(iStep, iPtor, 0));
                } else if (method.IsSameAs("FormerHumidityIndex")) {
                    if (preprocSize != 4) {
                        wxLogError(msg, preprocSize, 4, "FormerHumidityIndex");
                        return false;
                    }
                    preprocLevels.push_back(GetPreprocessLevel(iStep, iPtor, 0));
                    preprocHours.push_back(GetPreprocessHour(iStep, iPtor, 0));
                    preprocHours.push_back(GetPreprocessHour(iStep, iPtor, 1));
                } else {
                    wxLogWarning(_("The %s preprocessing method is not yet handled with the preload option."), method);
                }

                SetPreloadLevels(iStep, iPtor, preprocLevels);
                SetPreloadHours(iStep, iPtor, preprocHours);
            }
        }
    }

    return true;
}

bool asParameters::InputsOK() const {
    // Time properties
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

    if (GetTimeArrayTargetMode().CmpNoCase("predictand_thresholds") == 0 ||
        GetTimeArrayTargetMode().CmpNoCase("PredictandThresholds") == 0) {
        if (GetTimeArrayTargetPredictandSerieName().IsEmpty()) {
            wxLogError(_("The predictand time series (for the threshold preselection) "
                  "was not provided in the parameters  file."));
            return false;
        }
        if (GetTimeArrayTargetPredictandMinThreshold() == GetTimeArrayTargetPredictandMaxThreshold()) {
            wxLogError(_("The provided min/max predictand thresholds are equal in the parameters file."));
            return false;
        }
    }

    if (GetTimeArrayAnalogsMode().CmpNoCase("interval_days") == 0 ||
        GetTimeArrayAnalogsMode().CmpNoCase("IntervalDays") == 0) {
        if (GetAnalogsIntervalDays() <= 0) {
            wxLogError(_("The interval days for the analogs preselection "
                "was not provided in the parameters file."));
            return false;
        }
        if (GetAnalogsExcludeDays() <= 0) {
            wxLogError(_("The number of days to exclude around the target date "
                  "was not provided in the parameters file."));
            return false;
        }
    }

    // Analog dates
    for (int i = 0; i < GetStepsNb(); i++) {
        if (GetAnalogsNumber(i) <= 0) {
            wxLogError(_("The number of analogs (step %d) was not provided in the parameters file."), i);
            return false;
        }

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                if (GetPreprocessMethod(i, j).IsEmpty()) {
                    wxLogError(_("The preprocessing method (step %d, predictor %d) "
                                 "was not provided in the parameters file."), i, j);
                    return false;
                }

                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (GetPreprocessDatasetId(i, j, k).IsEmpty()) {
                        wxLogError(_("The dataset for preprocessing (step %d, predictor %d) "
                                     "was not provided in the parameters file."), i, j);
                        return false;
                    }
                    if (GetPreprocessDataId(i, j, k).IsEmpty()) {
                        wxLogError(_("The data for preprocessing (step %d, predictor %d) "
                                     "was not provided in the parameters file."), i, j);
                        return false;
                    }
                }
            } else {
                if (GetPredictorDatasetId(i, j).IsEmpty()) {
                    wxLogError(_("The dataset (step %d, predictor %d) was not "
                                 "provided in the parameters file."), i, j);
                    return false;
                }
                if (GetPredictorDataId(i, j).IsEmpty()) {
                    wxLogError(_("The data (step %d, predictor %d) was not "
                                 "provided in the parameters file."), i, j);
                    return false;
                }
            }

            if (GetPredictorGridType(i, j).IsEmpty()) {
                wxLogError(_("The grid type (step %d, predictor %d) is "
                             "empty in the parameters file."), i, j);
                return false;
            }
            if (GetPredictorXptsnb(i, j) == 0) {
                wxLogError(_("The X points nb value (step %d, predictor %d) "
                             "was not provided in the parameters file."), i, j);
                return false;
            }
            if (GetPredictorYptsnb(i, j) == 0) {
                wxLogError(_("The Y points nb value (step %d, predictor %d) "
                             "was not provided in the parameters file."), i, j);
                return false;
            }
            if (GetPredictorCriteria(i, j).IsEmpty()) {
                wxLogError(_("The criteria (step %d, predictor %d) was not provided "
                             "in the parameters file."), i, j);
                return false;
            }
        }
    }

    return true;
}

bool asParameters::PreprocessingPropertiesOk() const {
    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        for (int iPtor = 0; iPtor < GetPredictorsNb(iStep); iPtor++) {
            if (NeedsPreloading(iStep, iPtor) && NeedsPreprocessing(iStep, iPtor)) {
                // Check the preprocessing method
                wxString method = GetPreprocessMethod(iStep, iPtor);
                int preprocSize = GetPreprocessSize(iStep, iPtor);

                // Different actions depending on the preprocessing method.
                wxString msg = _("The size of the provided predictors (%d) does not match the requirements (%d) "
                    "in the preprocessing %s method.");
                if (method.IsSameAs("Multiplication") || method.IsSameAs("Multiply") ||
                    method.IsSameAs("Addition") || method.IsSameAs("Average")) {
                    // No constraints
                } else if (NeedsGradientPreprocessing(iStep, iPtor)) {
                    if (preprocSize != 1) {
                        wxLogError(msg, preprocSize, 1, "Gradient");
                        return false;
                    }
                } else if (method.IsSameAs("HumidityFlux")) {
                    if (preprocSize != 4) {
                        wxLogError(msg, preprocSize, 4, "HumidityFlux");
                        return false;
                    }
                } else if (method.IsSameAs("HumidityIndex")) {
                    if (preprocSize != 2) {
                        wxLogError(msg, preprocSize, 2, "HumidityIndex");
                        return false;
                    }
                } else if (method.IsSameAs("FormerHumidityIndex")) {
                    if (preprocSize != 4) {
                        wxLogError(msg, preprocSize, 4, "FormerHumidityIndex");
                        return false;
                    }
                } else {
                    wxLogWarning(_("The %s preprocessing method is not yet handled with the preload option."), method);
                }
            }
        }
    }

    return true;
}

bool asParameters::FixAnalogsNb() {
    // Check analogs number coherence
    int analogsNb = GetAnalogsNumber(0);
    for (int iStep = 1; iStep < m_steps.size(); iStep++) {
        if (GetAnalogsNumber(iStep) > analogsNb) {
            SetAnalogsNumber(iStep, analogsNb);
        } else {
            analogsNb = GetAnalogsNumber(iStep);
        }
    }

    return true;
}

void asParameters::SortLevelsAndTime() {
    // Sort levels on every analogy level
    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        // Get the predictors vector
        VectorParamsPredictors oldPtors = GetVectorParamsPredictors(iStep);
        VectorParamsPredictors newPtors;

        // Sort
        while (true) {
            if (oldPtors.empty()) {
                break;
            }

            // Find the smallest level and hour combination
            int lowestIndex = 0;
            float level;
            double hour;
            if (oldPtors[0].preprocess) {
                level = oldPtors[0].preprocessLevels[0];
                hour = oldPtors[0].preprocessHours[0];
            } else {
                level = oldPtors[0].level;
                hour = oldPtors[0].hour;
            }

            for (int i = 1; i < oldPtors.size(); i++) {
                // Get next level and hour
                float nextLevel;
                double nextHour;
                if (oldPtors[i].preprocess) {
                    nextLevel = oldPtors[i].preprocessLevels[0];
                    nextHour = oldPtors[i].preprocessHours[0];
                } else {
                    nextLevel = oldPtors[i].level;
                    nextHour = oldPtors[i].hour;
                }

                // Compare to previous one
                if (nextLevel < level) {
                    lowestIndex = i;
                    level = nextLevel;
                    hour = nextHour;
                } else if (nextLevel == level) {
                    if (nextHour < hour) {
                        lowestIndex = i;
                        level = nextLevel;
                        hour = nextHour;
                    }
                }
            }

            // Store in the new container and remove from the old one
            newPtors.push_back(oldPtors[lowestIndex]);
            oldPtors.erase(oldPtors.begin() + lowestIndex);

            // Store the sorted vector
            SetVectorParamsPredictors(iStep, newPtors);
        }
    }
}

vi asParameters::GetFileStationIds(wxString stationIdsString) {
    // Trim
    stationIdsString.Trim(true);
    stationIdsString.Trim(false);

    vi ids;

    if (stationIdsString.IsEmpty()) {
        wxLogError(_("The station ID was not provided."));
        return ids;
    }

    // Multivariate
    if (stationIdsString.SubString(0, 0).IsSameAs("(") ||
        stationIdsString.SubString(0, 1).IsSameAs("'(")) {
        wxString subStr = wxEmptyString;
        if (stationIdsString.SubString(0, 0).IsSameAs("(")) {
            subStr = stationIdsString.SubString(1, stationIdsString.Len() - 1);
        } else {
            subStr = stationIdsString.SubString(2, stationIdsString.Len() - 1);
        }

        // Check that it contains only 1 opening bracket
        if (subStr.Find("(") != wxNOT_FOUND) {
            wxLogError(_("The format of the station ID is not correct (more than one opening bracket)."));
            return ids;
        }

        // Check that it contains 1 closing bracket at the end
        if (subStr.Find(")") != subStr.size() - 1 && subStr.Find(")'") != subStr.size() - 2) {
            wxLogError(_("The format of the station ID is not correct (location of the closing bracket)."));
            return ids;
        }

        // Extract content
        wxChar separator = ',';
        while (subStr.Find(separator) != wxNOT_FOUND) {
            wxString strBefore = subStr.BeforeFirst(separator);
            subStr = subStr.AfterFirst(separator);
            int id = wxAtoi(strBefore);
            ids.push_back(id);
        }
        if (!subStr.IsEmpty()) {
            int id = wxAtoi(subStr);
            ids.push_back(id);
        }
    } else {
        // Check for single value
        if (stationIdsString.Find("(") != wxNOT_FOUND || stationIdsString.Find(")") != wxNOT_FOUND ||
            stationIdsString.Find(",") != wxNOT_FOUND) {
            wxLogError(_("The format of the station ID is not correct (should be only digits)."));
            return ids;
        }
        int id = wxAtoi(stationIdsString);
        ids.push_back(id);
    }

    return ids;
}

wxString asParameters::GetPredictandStationIdsString() const {
    return PredictandStationIdsToString(m_predictandStationIds);
}

wxString asParameters::PredictandStationIdsToString(const vi &predictandStationIds) {
    wxString ids;

    if (predictandStationIds.size() == 1) {
        ids << predictandStationIds[0];
    } else {
        ids = "(";

        for (int i = 0; i < (int)predictandStationIds.size(); i++) {
            ids << predictandStationIds[i];

            if (i < (int)predictandStationIds.size() - 1) {
                ids.Append(",");
            }
        }

        ids.Append(")");
    }

    return ids;
}

bool asParameters::FixTimeLimits() {
    double minHour = 1000.0, maxHour = -1000.0;
    for (int i = 0; i < GetStepsNb(); i++) {
        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                double minHourPredictor = 1000.0, maxHourPredictor = -1000.0;

                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    minHour = wxMin(m_steps[i].predictors[j].preprocessHours[k], minHour);
                    maxHour = wxMax(m_steps[i].predictors[j].preprocessHours[k], maxHour);
                    minHourPredictor = wxMin(m_steps[i].predictors[j].preprocessHours[k], minHourPredictor);
                    maxHourPredictor = wxMax(m_steps[i].predictors[j].preprocessHours[k], maxHourPredictor);
                    m_steps[i].predictors[j].hour = minHourPredictor;
                }
            } else {
                minHour = wxMin(m_steps[i].predictors[j].hour, minHour);
                maxHour = wxMax(m_steps[i].predictors[j].hour, maxHour);
            }
        }
    }

    m_timeMinHours = minHour;
    m_timeMaxHours = maxHour;

    return true;
}

bool asParameters::FixWeights() {
    for (int i = 0; i < GetStepsNb(); i++) {
        // Sum the weights
        float totWeight = 0;
        for (int j = 0; j < GetPredictorsNb(i); j++) {
            totWeight += m_steps[i].predictors[j].weight;
        }

        // Correct to set the total to 1
        for (int j = 0; j < GetPredictorsNb(i); j++) {
            m_steps[i].predictors[j].weight /= totWeight;
        }
    }

    return true;
}

bool asParameters::FixCoordinates() {
    for (int i = 0; i < GetStepsNb(); i++) {
        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (GetPredictorGridType(i, j).IsSameAs("regular", false)) {
                // Check that the coordinates are a multiple of the steps
                if (std::abs(std::fmod(m_steps[i].predictors[j].xMin - m_steps[i].predictors[j].xShift,
                                       m_steps[i].predictors[j].xStep)) > 0) {
                    double factor = (m_steps[i].predictors[j].xMin - m_steps[i].predictors[j].xShift) /
                                    m_steps[i].predictors[j].xStep;
                    factor = asRound(factor);
                    m_steps[i].predictors[j].xMin =
                        factor * m_steps[i].predictors[j].xStep + m_steps[i].predictors[j].xShift;
                }

                if (std::abs(std::fmod(m_steps[i].predictors[j].yMin - m_steps[i].predictors[j].yShift,
                                       m_steps[i].predictors[j].yStep)) > 0) {
                    double factor = (m_steps[i].predictors[j].yMin - m_steps[i].predictors[j].yShift) /
                                    m_steps[i].predictors[j].yStep;
                    factor = asRound(factor);
                    m_steps[i].predictors[j].yMin =
                        factor * m_steps[i].predictors[j].yStep + m_steps[i].predictors[j].yShift;
                }
            }

            if (m_steps[i].predictors[j].flatAllowed == asFLAT_FORBIDDEN) {
                // Check that the size is larger than 1 point
                if (m_steps[i].predictors[j].xPtsNb < 2) {
                    m_steps[i].predictors[j].xPtsNb = 2;
                }

                if (m_steps[i].predictors[j].yPtsNb < 2) {
                    m_steps[i].predictors[j].yPtsNb = 2;
                }
            } else {
                // Check that the size is larger than 0
                if (m_steps[i].predictors[j].xPtsNb < 1) {
                    m_steps[i].predictors[j].xPtsNb = 1;
                }

                if (m_steps[i].predictors[j].yPtsNb < 1) {
                    m_steps[i].predictors[j].yPtsNb = 1;
                }
            }
        }
    }

    return true;
}

wxString asParameters::Print() const {
    // Create content string
    wxString content = wxEmptyString;

    content.Append(wxString::Format("Station\t%s\t", GetPredictandStationIdsString()));
    content.Append(wxString::Format("DaysInt\t%d\t", GetAnalogsIntervalDays()));
    content.Append(wxString::Format("ExcludeDays\t%d\t", GetAnalogsExcludeDays()));

    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        content.Append(wxString::Format("|||| Step(%d)\t", iStep));
        content.Append(wxString::Format("Anb\t%d\t", GetAnalogsNumber(iStep)));

        for (int iPtor = 0; iPtor < GetPredictorsNb(iStep); iPtor++) {
            content.Append(wxString::Format("|| Ptor(%d)\t", iPtor));

            if (NeedsPreprocessing(iStep, iPtor)) {
                content.Append(wxString::Format("%s\t", GetPreprocessMethod(iStep, iPtor)));

                for (int iPre = 0; iPre < GetPreprocessSize(iStep, iPtor); iPre++) {
                    content.Append(wxString::Format("| %s %s\t", GetPreprocessDatasetId(iStep, iPtor, iPre),
                                                    GetPreprocessDataId(iStep, iPtor, iPre)));
                    content.Append(wxString::Format("Level\t%g\t", GetPreprocessLevel(iStep, iPtor, iPre)));
                    content.Append(wxString::Format("Time\t%g\t", GetPreprocessHour(iStep, iPtor, iPre)));
                }
            } else {
                content.Append(
                    wxString::Format("%s %s\t", GetPredictorDatasetId(iStep, iPtor), GetPredictorDataId(iStep, iPtor)));
                content.Append(wxString::Format("Level\t%g\t", GetPredictorLevel(iStep, iPtor)));
                content.Append(wxString::Format("Time\t%g\t", GetPredictorHour(iStep, iPtor)));
            }

            content.Append(wxString::Format("GridType\t%s\t", GetPredictorGridType(iStep, iPtor)));
            content.Append(wxString::Format("xMin\t%g\t", GetPredictorXmin(iStep, iPtor)));
            content.Append(wxString::Format("xPtsNb\t%d\t", GetPredictorXptsnb(iStep, iPtor)));
            content.Append(wxString::Format("xStep\t%g\t", GetPredictorXstep(iStep, iPtor)));
            content.Append(wxString::Format("yMin\t%g\t", GetPredictorYmin(iStep, iPtor)));
            content.Append(wxString::Format("yPtsNb\t%d\t", GetPredictorYptsnb(iStep, iPtor)));
            content.Append(wxString::Format("yStep\t%g\t", GetPredictorYstep(iStep, iPtor)));
            content.Append(wxString::Format("Weight\t%e\t", GetPredictorWeight(iStep, iPtor)));
            if (!GetPreprocessMethod(iStep, iPtor).IsEmpty()) {
                content.Append(wxString::Format("%s\t", GetPreprocessMethod(iStep, iPtor)));
            } else {
                content.Append("NoPreprocessing\t");
            }
            content.Append(wxString::Format("Criteria\t%s\t", GetPredictorCriteria(iStep, iPtor)));
        }
    }

    return content;
}

bool asParameters::IsSameAs(const asParameters &params) const {
    return IsSameAs(params.GetParameters(), params.GetPredictandStationIds(), params.GetAnalogsIntervalDays());
}

bool asParameters::IsSameAs(const VectorParamsStep &params, const vi &predictandStationIds,
                            int analogsIntervalDays) const {
    if (!GetPredictandStationIdsString().IsSameAs(PredictandStationIdsToString(predictandStationIds))) return false;

    if (GetAnalogsIntervalDays() != analogsIntervalDays) return false;

    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        if (GetAnalogsNumber(iStep) != params[iStep].analogsNumber) return false;

        for (int iPtor = 0; iPtor < GetPredictorsNb(iStep); iPtor++) {
            if (NeedsPreprocessing(iStep, iPtor) != params[iStep].predictors[iPtor].preprocess) return false;

            if (NeedsPreprocessing(iStep, iPtor)) {
                if (!GetPreprocessMethod(iStep, iPtor).IsSameAs(params[iStep].predictors[iPtor].preprocessMethod))
                    return false;

                for (int iPre = 0; iPre < GetPreprocessSize(iStep, iPtor); iPre++) {
                    if (!GetPreprocessDatasetId(iStep, iPtor, iPre)
                             .IsSameAs(params[iStep].predictors[iPtor].preprocessDatasetIds[iPre]))
                        return false;

                    if (!GetPreprocessDataId(iStep, iPtor, iPre)
                             .IsSameAs(params[iStep].predictors[iPtor].preprocessDataIds[iPre]))
                        return false;

                    if (GetPreprocessLevel(iStep, iPtor, iPre) !=
                        params[iStep].predictors[iPtor].preprocessLevels[iPre])
                        return false;

                    if (GetPreprocessHour(iStep, iPtor, iPre) != params[iStep].predictors[iPtor].preprocessHours[iPre])
                        return false;
                }
            } else {
                if (!GetPredictorDatasetId(iStep, iPtor).IsSameAs(params[iStep].predictors[iPtor].datasetId))
                    return false;

                if (!GetPredictorDataId(iStep, iPtor).IsSameAs(params[iStep].predictors[iPtor].dataId)) return false;

                if (GetPredictorLevel(iStep, iPtor) != params[iStep].predictors[iPtor].level) return false;

                if (GetPredictorHour(iStep, iPtor) != params[iStep].predictors[iPtor].hour) return false;
            }

            if (!GetPredictorGridType(iStep, iPtor).IsSameAs(params[iStep].predictors[iPtor].gridType)) return false;

            if (GetPredictorXmin(iStep, iPtor) != params[iStep].predictors[iPtor].xMin) return false;

            if (GetPredictorXptsnb(iStep, iPtor) != params[iStep].predictors[iPtor].xPtsNb) return false;

            if (GetPredictorXstep(iStep, iPtor) != params[iStep].predictors[iPtor].xStep) return false;

            if (GetPredictorYmin(iStep, iPtor) != params[iStep].predictors[iPtor].yMin) return false;

            if (GetPredictorYptsnb(iStep, iPtor) != params[iStep].predictors[iPtor].yPtsNb) return false;

            if (GetPredictorYstep(iStep, iPtor) != params[iStep].predictors[iPtor].yStep) return false;

            if (GetPredictorWeight(iStep, iPtor) != params[iStep].predictors[iPtor].weight) return false;

            if (!GetPredictorCriteria(iStep, iPtor).IsSameAs(params[iStep].predictors[iPtor].criteria)) return false;
        }
    }

    return true;
}

bool asParameters::IsCloseTo(const asParameters &params) const {
    return IsCloseTo(params.GetParameters(), params.GetPredictandStationIds(), params.GetAnalogsIntervalDays());
}

bool asParameters::IsCloseTo(const VectorParamsStep &params, const vi &predictandStationIds,
                             int analogsIntervalDays) const {
    if (!GetPredictandStationIdsString().IsSameAs(PredictandStationIdsToString(predictandStationIds))) return false;

    if (abs(GetAnalogsIntervalDays() - analogsIntervalDays) > 0.1 * GetAnalogsIntervalDays()) return false;

    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        if (abs(GetAnalogsNumber(iStep) - params[iStep].analogsNumber) > 0.1 * GetAnalogsNumber(iStep)) return false;

        for (int iPtor = 0; iPtor < GetPredictorsNb(iStep); iPtor++) {
            if (NeedsPreprocessing(iStep, iPtor) != params[iStep].predictors[iPtor].preprocess) return false;

            if (NeedsPreprocessing(iStep, iPtor)) {
                if (!GetPreprocessMethod(iStep, iPtor).IsSameAs(params[iStep].predictors[iPtor].preprocessMethod))
                    return false;

                for (int iPre = 0; iPre < GetPreprocessSize(iStep, iPtor); iPre++) {
                    if (!GetPreprocessDatasetId(iStep, iPtor, iPre)
                             .IsSameAs(params[iStep].predictors[iPtor].preprocessDatasetIds[iPre]))
                        return false;

                    if (!GetPreprocessDataId(iStep, iPtor, iPre)
                             .IsSameAs(params[iStep].predictors[iPtor].preprocessDataIds[iPre]))
                        return false;

                    if (GetPreprocessLevel(iStep, iPtor, iPre) !=
                        params[iStep].predictors[iPtor].preprocessLevels[iPre])
                        return false;

                    if (GetPreprocessHour(iStep, iPtor, iPre) != params[iStep].predictors[iPtor].preprocessHours[iPre])
                        return false;
                }
            } else {
                if (!GetPredictorDatasetId(iStep, iPtor).IsSameAs(params[iStep].predictors[iPtor].datasetId))
                    return false;

                if (!GetPredictorDataId(iStep, iPtor).IsSameAs(params[iStep].predictors[iPtor].dataId)) return false;

                if (GetPredictorLevel(iStep, iPtor) != params[iStep].predictors[iPtor].level) return false;

                if (GetPredictorHour(iStep, iPtor) != params[iStep].predictors[iPtor].hour) return false;
            }

            if (!GetPredictorGridType(iStep, iPtor).IsSameAs(params[iStep].predictors[iPtor].gridType)) return false;

            if (fabs(GetPredictorXmin(iStep, iPtor) - params[iStep].predictors[iPtor].xMin) > 2) return false;

            if (abs(GetPredictorXptsnb(iStep, iPtor) - params[iStep].predictors[iPtor].xPtsNb) >
                0.1 * GetPredictorXptsnb(iStep, iPtor))
                return false;

            if (GetPredictorXstep(iStep, iPtor) != params[iStep].predictors[iPtor].xStep) return false;

            if (fabs(GetPredictorYmin(iStep, iPtor) - params[iStep].predictors[iPtor].yMin) > 2) return false;

            if (abs(GetPredictorYptsnb(iStep, iPtor) - params[iStep].predictors[iPtor].yPtsNb) >
                0.1 * GetPredictorYptsnb(iStep, iPtor))
                return false;

            if (GetPredictorYstep(iStep, iPtor) != params[iStep].predictors[iPtor].yStep) return false;

            if (fabs(GetPredictorWeight(iStep, iPtor) - params[iStep].predictors[iPtor].weight) > 0.1) return false;

            if (!GetPredictorCriteria(iStep, iPtor).IsSameAs(params[iStep].predictors[iPtor].criteria)) return false;
        }
    }

    return true;
}

bool asParameters::PrintAndSaveTemp(const wxString &filePath) const {
    wxString saveFilePath;

    if (filePath.IsEmpty()) {
        saveFilePath = asConfig::GetTempDir() + "/AtmoSwingCurrentParameters.txt";
    } else {
        saveFilePath = filePath;
    }

    asFileText fileRes(saveFilePath, asFileText::Replace);
    if (!fileRes.Open()) return false;

    wxString content = Print();

    wxString header;
    header = _("AtmoSwing current parameters\n");
    fileRes.AddContent(header);
    fileRes.AddContent(content);
    fileRes.Close();

    return true;
}

bool asParameters::GetValuesFromString(wxString stringVals) {
    wxString strVal;
    double dVal;
    long lVal;

    wxString errMsg(_("Error when parsing the parameters file"));

    strVal = asExtractParamValueAndCut(stringVals, "DaysInt");
    if (!strVal.ToLong(&lVal)) {
        wxLogError(errMsg);
        return false;
    }
    SetAnalogsIntervalDays(int(lVal));

    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        strVal = asExtractParamValueAndCut(stringVals, "Anb");
        if (!strVal.ToLong(&lVal)) {
            wxLogError(errMsg);
            return false;
        }
        SetAnalogsNumber(iStep, int(lVal));

        for (int iPtor = 0; iPtor < GetPredictorsNb(iStep); iPtor++) {
            if (NeedsPreprocessing(iStep, iPtor)) {
                for (int iPre = 0; iPre < GetPreprocessSize(iStep, iPtor); iPre++) {
                    strVal = asExtractParamValueAndCut(stringVals, "Level");
                    if (!strVal.ToDouble(&dVal)) {
                        wxLogError(errMsg);
                        return false;
                    }
                    SetPreprocessLevel(iStep, iPtor, iPre, float(dVal));

                    strVal = asExtractParamValueAndCut(stringVals, "Time");
                    if (!strVal.ToDouble(&dVal)) {
                        wxLogError(errMsg);
                        return false;
                    }
                    SetPreprocessHour(iStep, iPtor, iPre, float(dVal));
                }
            } else {
                strVal = asExtractParamValueAndCut(stringVals, "Ptor");
                stringVals = stringVals.AfterFirst('\t');
                strVal = stringVals.AfterFirst(' ');
                strVal = strVal.BeforeFirst('\t');
                SetPredictorDataId(iStep, iPtor, strVal);

                strVal = asExtractParamValueAndCut(stringVals, "Level");
                if (!strVal.ToDouble(&dVal)) {
                    wxLogError(errMsg);
                    return false;
                }
                SetPredictorLevel(iStep, iPtor, float(dVal));

                strVal = asExtractParamValueAndCut(stringVals, "Time");
                if (!strVal.ToDouble(&dVal)) {
                    wxLogError(errMsg);
                    return false;
                }
                SetPredictorHour(iStep, iPtor, float(dVal));
            }

            strVal = asExtractParamValueAndCut(stringVals, "xMin");
            if (!strVal.ToDouble(&dVal)) {
                wxLogError(errMsg);
                return false;
            }
            SetPredictorXmin(iStep, iPtor, dVal);

            strVal = asExtractParamValueAndCut(stringVals, "xPtsNb");
            if (!strVal.ToLong(&lVal)) {
                wxLogError(errMsg);
                return false;
            }
            SetPredictorXptsnb(iStep, iPtor, int(lVal));

            strVal = asExtractParamValueAndCut(stringVals, "xStep");
            if (!strVal.ToDouble(&dVal)) {
                wxLogError(errMsg);
                return false;
            }
            SetPredictorXstep(iStep, iPtor, dVal);

            strVal = asExtractParamValueAndCut(stringVals, "yMin");
            if (!strVal.ToDouble(&dVal)) {
                wxLogError(errMsg);
                return false;
            }
            SetPredictorYmin(iStep, iPtor, dVal);

            strVal = asExtractParamValueAndCut(stringVals, "yPtsNb");
            if (!strVal.ToLong(&lVal)) {
                wxLogError(errMsg);
                return false;
            }
            SetPredictorYptsnb(iStep, iPtor, int(lVal));

            strVal = asExtractParamValueAndCut(stringVals, "yStep");
            if (!strVal.ToDouble(&dVal)) {
                wxLogError(errMsg);
                return false;
            }
            SetPredictorYstep(iStep, iPtor, dVal);

            strVal = asExtractParamValueAndCut(stringVals, "Weight");
            if (!strVal.ToDouble(&dVal)) {
                wxLogError(errMsg);
                return false;
            }
            SetPredictorWeight(iStep, iPtor, dVal);

            stringVals = stringVals.AfterFirst('\t');
            strVal = stringVals.BeforeFirst('\t');
            SetPreprocessMethod(iStep, iPtor, strVal);

            strVal = asExtractParamValueAndCut(stringVals, "Criteria");
            SetPredictorCriteria(iStep, iPtor, strVal);
        }
    }

    return true;
}

void asParameters::SetTargetTimeStepHours(double val) {
    wxASSERT(!asIsNaN(val));
    m_targetTimeStepHours = val;
}

void asParameters::SetAnalogsTimeStepHours(double val) {
    wxASSERT(!asIsNaN(val));
    m_analogsTimeStepHours = val;
}

void asParameters::SetTimeArrayTargetMode(const wxString &val) {
    wxASSERT(!val.IsEmpty());
    m_timeArrayTargetMode = val;
}

void asParameters::SetTimeArrayTargetPredictandSerieName(const wxString &val) {
    wxASSERT(!val.IsEmpty());
    m_timeArrayTargetPredictandSerieName = val;
}

void asParameters::SetTimeArrayTargetPredictandMinThreshold(float val) {
    wxASSERT(!asIsNaN(val));
    m_timeArrayTargetPredictandMinThreshold = val;
}

void asParameters::SetTimeArrayTargetPredictandMaxThreshold(float val) {
    wxASSERT(!asIsNaN(val));
    m_timeArrayTargetPredictandMaxThreshold = val;
}

void asParameters::SetTimeArrayAnalogsMode(const wxString &val) {
    wxASSERT(!val.IsEmpty());
    m_timeArrayAnalogsMode = val;
}

void asParameters::SetAnalogsExcludeDays(int val) {
    wxASSERT(!asIsNaN(val));
    m_analogsExcludeDays = val;
}

void asParameters::SetAnalogsIntervalDays(int val) {
    wxASSERT(!asIsNaN(val));
    m_analogsIntervalDays = val;
}

void asParameters::SetPredictandStationIds(vi val) {
    m_predictandStationIds = val;
}

void asParameters::SetPredictandStationIds(wxString val) {
    wxStringTokenizer tokenizer(val, ":,; ");
    while (tokenizer.HasMoreTokens()) {
        wxString token = tokenizer.GetNextToken();
        long stationId;
        if (token.ToLong(&stationId)) {
            m_predictandStationIds.push_back(int(stationId));
        }
    }
}

void asParameters::SetPredictandTimeHours(double val) {
    wxASSERT(!asIsNaN(val));
    m_predictandTimeHours = val;
}

void asParameters::SetAnalogsNumber(int iStep, int val) {
    wxASSERT(!asIsNaN(val));
    m_steps[iStep].analogsNumber = val;
}

bool asParameters::SetPreloadDataIds(int iStep, int iPtor, vwxs val) {
    if (val.empty()) {
        wxLogError(_("The provided preload data IDs vector is empty."));
        return false;
    } else {
        for (auto &v : val) {
            if (v.IsEmpty()) {
                wxLogError(_("There are empty values in the provided preload data IDs vector."));
                return false;
            }
        }
    }
    m_steps[iStep].predictors[iPtor].preloadDataIds.clear();
    for (auto &v : val) {
        m_steps[iStep].predictors[iPtor].preloadDataIds.push_back(v.ToStdString());
    }
    return true;
}

void asParameters::SetPreloadDataIds(int iStep, int iPtor, wxString val) {
    wxASSERT(!val.IsEmpty());
    m_steps[iStep].predictors[iPtor].preloadDataIds.clear();
    m_steps[iStep].predictors[iPtor].preloadDataIds.push_back(val.ToStdString());
}

bool asParameters::SetPreloadHours(int iStep, int iPtor, vd val) {
    if (val.empty()) {
        wxLogError(_("The provided preload time (hours) vector is empty."));
        return false;
    } else {
        for (double v : val) {
            if (asIsNaN(v)) {
                wxLogError(_("There are NaN values in the provided preload time (hours) vector."));
                return false;
            }
        }
    }
    m_steps[iStep].predictors[iPtor].preloadHours = val;

    return true;
}

void asParameters::SetPreloadHours(int iStep, int iPtor, double val) {
    wxASSERT(!asIsNaN(val));
    m_steps[iStep].predictors[iPtor].preloadHours.clear();
    m_steps[iStep].predictors[iPtor].preloadHours.push_back(val);
}

bool asParameters::SetPreloadLevels(int iStep, int iPtor, vf val) {
    if (val.empty()) {
        wxLogError(_("The provided 'preload levels' vector is empty."));
        return false;
    } else {
        for (float v : val) {
            if (asIsNaN(v)) {
                wxLogError(_("There are NaN values in the provided 'preload levels' vector."));
                return false;
            }
        }
    }
    m_steps[iStep].predictors[iPtor].preloadLevels = val;
    return true;
}

void asParameters::SetPreloadLevels(int iStep, int iPtor, float val) {
    wxASSERT(!asIsNaN(val));
    m_steps[iStep].predictors[iPtor].preloadLevels.clear();
    m_steps[iStep].predictors[iPtor].preloadLevels.push_back(val);
}

void asParameters::SetPreloadXmin(int iStep, int iPtor, double val) {
    wxASSERT(!asIsNaN(val));
    m_steps[iStep].predictors[iPtor].preloadXmin = val;
}

void asParameters::SetPreloadXptsnb(int iStep, int iPtor, int val) {
    wxASSERT(!asIsNaN(val));
    m_steps[iStep].predictors[iPtor].preloadXptsnb = val;
}

void asParameters::SetPreloadYmin(int iStep, int iPtor, double val) {
    wxASSERT(!asIsNaN(val));
    m_steps[iStep].predictors[iPtor].preloadYmin = val;
}

void asParameters::SetPreloadYptsnb(int iStep, int iPtor, int val) {
    wxASSERT(!asIsNaN(val));
    m_steps[iStep].predictors[iPtor].preloadYptsnb = val;
}

void asParameters::SetPreprocessMethod(int iStep, int iPtor, const wxString &val) {
    wxASSERT(!val.IsEmpty());
    m_steps[iStep].predictors[iPtor].preprocessMethod = val.ToStdString();
}

bool asParameters::NeedsGradientPreprocessing(int iStep, int iPtor) const {
    wxString method = m_steps[iStep].predictors[iPtor].preprocessMethod;

    return method.IsSameAs("Gradients", false) ||
           method.IsSameAs("SimpleGradients", false) ||
           method.IsSameAs("RealGradients", false) ||
           method.IsSameAs("SimpleGradientsWithGaussianWeights", false) ||
           method.IsSameAs("RealGradientsWithGaussianWeights", false);
}

bool asParameters::IsCriteriaUsingGradients(int iStep, int iPtor) const {
    if (GetPredictorCriteria(iStep, iPtor).IsSameAs("S1grads") ||
        GetPredictorCriteria(iStep, iPtor).IsSameAs("S2grads")) {
        return true;
    }

    return false;
}

void asParameters::FixCriteriaIfGradientsPreprocessed(int iStep, int iPtor) {
    if (NeedsGradientPreprocessing(iStep, iPtor)) {
        if (GetPredictorCriteria(iStep, iPtor).IsSameAs("S1") ||
            GetPredictorCriteria(iStep, iPtor).IsSameAs("S1r") ||
            GetPredictorCriteria(iStep, iPtor).IsSameAs("S1s") ||
            GetPredictorCriteria(iStep, iPtor).IsSameAs("S1G") ||
            GetPredictorCriteria(iStep, iPtor).IsSameAs("S1rG") ||
            GetPredictorCriteria(iStep, iPtor).IsSameAs("S1sG")) {
            SetPredictorCriteria(iStep, iPtor, "S1grads");
        } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("S2") ||
                   GetPredictorCriteria(iStep, iPtor).IsSameAs("S2r") ||
                   GetPredictorCriteria(iStep, iPtor).IsSameAs("S2s") ||
                   GetPredictorCriteria(iStep, iPtor).IsSameAs("S2G") ||
                   GetPredictorCriteria(iStep, iPtor).IsSameAs("S2rG") ||
                   GetPredictorCriteria(iStep, iPtor).IsSameAs("S2sG")) {
            SetPredictorCriteria(iStep, iPtor, "S2grads");
        }
    }
}

void asParameters::ForceUsingGradientsPreprocessing(int iStep, int iPtor) {
    // Gradients - S1
    if (GetPredictorCriteria(iStep, iPtor).IsSameAs("S1") ||
        GetPredictorCriteria(iStep, iPtor).IsSameAs("S1r")) {
        SetPredictorCriteria(iStep, iPtor, "S1grads");
        SetPreprocessMethod(iStep, iPtor, "RealGradients");
    } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("S1s")) {
        SetPredictorCriteria(iStep, iPtor, "S1grads");
        SetPreprocessMethod(iStep, iPtor, "SimpleGradients");
    } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("S1G") ||
               GetPredictorCriteria(iStep, iPtor).IsSameAs("S1rG")) {
        SetPredictorCriteria(iStep, iPtor, "S1grads");
        SetPreprocessMethod(iStep, iPtor, "RealGradientsWithGaussianWeights");
    } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("S1sG")) {
        SetPredictorCriteria(iStep, iPtor, "S1grads");
        SetPreprocessMethod(iStep, iPtor, "SimpleGradientsWithGaussianWeights");
    }

    // Curvature - S2
    if (GetPredictorCriteria(iStep, iPtor).IsSameAs("S2") ||
        GetPredictorCriteria(iStep, iPtor).IsSameAs("S2r")) {
        SetPredictorCriteria(iStep, iPtor, "S2grads");
        SetPreprocessMethod(iStep, iPtor, "RealCurvature");
    } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("S2s")) {
        SetPredictorCriteria(iStep, iPtor, "S2grads");
        SetPreprocessMethod(iStep, iPtor, "SimpleCurvature");
    } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("S2G") ||
               GetPredictorCriteria(iStep, iPtor).IsSameAs("S2rG")) {
        SetPredictorCriteria(iStep, iPtor, "S2grads");
        SetPreprocessMethod(iStep, iPtor, "RealCurvatureWithGaussianWeights");
    } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("S2sG")) {
        SetPredictorCriteria(iStep, iPtor, "S2grads");
        SetPreprocessMethod(iStep, iPtor, "SimpleCurvatureWithGaussianWeights");
    }
}

wxString asParameters::GetPreprocessDatasetId(int iStep, int iPtor, int iPre) const {
    wxASSERT(m_steps[iStep].predictors[iPtor].preprocessDatasetIds.size() > iPre);
    return m_steps[iStep].predictors[iPtor].preprocessDatasetIds[iPre];
}

void asParameters::SetPreprocessDatasetId(int iStep, int iPtor, int iPre, const wxString &val) {
    wxASSERT(!val.IsEmpty());
    if (m_steps[iStep].predictors[iPtor].preprocessDatasetIds.size() >= iPre + 1) {
        m_steps[iStep].predictors[iPtor].preprocessDatasetIds[iPre] = val.ToStdString();
    } else {
        wxASSERT((int)m_steps[iStep].predictors[iPtor].preprocessDatasetIds.size() == iPre);
        m_steps[iStep].predictors[iPtor].preprocessDatasetIds.push_back(val.ToStdString());
    }
}

wxString asParameters::GetPreprocessDataId(int iStep, int iPtor, int iPre) const {
    wxASSERT(m_steps[iStep].predictors[iPtor].preprocessDataIds.size() > iPre);
    return m_steps[iStep].predictors[iPtor].preprocessDataIds[iPre];
}

void asParameters::SetPreprocessDataId(int iStep, int iPtor, int iPre, const wxString &val) {
    wxASSERT(!val.IsEmpty());
    if (m_steps[iStep].predictors[iPtor].preprocessDataIds.size() >= iPre + 1) {
        m_steps[iStep].predictors[iPtor].preprocessDataIds[iPre] = val.ToStdString();
    } else {
        wxASSERT((int)m_steps[iStep].predictors[iPtor].preprocessDataIds.size() == iPre);
        m_steps[iStep].predictors[iPtor].preprocessDataIds.push_back(val.ToStdString());
    }
}

float asParameters::GetPreprocessLevel(int iStep, int iPtor, int iPre) const {
    if (m_steps[iStep].predictors[iPtor].preprocessLevels.size() >= iPre + 1) {
        return m_steps[iStep].predictors[iPtor].preprocessLevels[iPre];
    } else {
        wxLogError(_("Trying to access to an element outside of preprocessLevels in the parameters object."));
        return NaNf;
    }
}

void asParameters::SetPreprocessLevel(int iStep, int iPtor, int iPre, float val) {
    wxASSERT(!asIsNaN(val));
    if (m_steps[iStep].predictors[iPtor].preprocessLevels.size() >= iPre + 1) {
        m_steps[iStep].predictors[iPtor].preprocessLevels[iPre] = val;
    } else {
        wxASSERT((int)m_steps[iStep].predictors[iPtor].preprocessLevels.size() == iPre);
        m_steps[iStep].predictors[iPtor].preprocessLevels.push_back(val);
    }
}

double asParameters::GetPreprocessHour(int iStep, int iPtor, int iPre) const {
    wxASSERT(m_steps[iStep].predictors[iPtor].preprocessHours.size() > iPre);
    return m_steps[iStep].predictors[iPtor].preprocessHours[iPre];
}

double asParameters::GetPreprocessTimeAsDays(int iStep, int iPtor, int iPre) const {
    return GetPreprocessHour(iStep, iPtor, iPre) / 24.0;
}

void asParameters::SetPreprocessHour(int iStep, int iPtor, int iPre, double val) {
    wxASSERT(!asIsNaN(val));
    if (m_steps[iStep].predictors[iPtor].preprocessHours.size() >= iPre + 1) {
        m_steps[iStep].predictors[iPtor].preprocessHours[iPre] = val;
    } else {
        wxASSERT((int)m_steps[iStep].predictors[iPtor].preprocessHours.size() == iPre);
        m_steps[iStep].predictors[iPtor].preprocessHours.push_back(val);
    }
}

int asParameters::GetPreprocessMembersNb(int iStep, int iPtor, int iPre) const {
    wxASSERT(m_steps[iStep].predictors[iPtor].preprocessMembersNb.size() > iPre);
    return m_steps[iStep].predictors[iPtor].preprocessMembersNb[iPre];
}

void asParameters::SetPreprocessMembersNb(int iStep, int iPtor, int iPre, int val) {
    wxASSERT(!asIsNaN(val));
    if (m_steps[iStep].predictors[iPtor].preprocessMembersNb.size() >= iPre + 1) {
        m_steps[iStep].predictors[iPtor].preprocessMembersNb[iPre] = val;
    } else {
        wxASSERT((int)m_steps[iStep].predictors[iPtor].preprocessMembersNb.size() == iPre);
        m_steps[iStep].predictors[iPtor].preprocessMembersNb.push_back(val);
    }
}

void asParameters::SetPredictorDatasetId(int iStep, int iPtor, const wxString &val) {
    wxASSERT(!val.IsEmpty());
    m_steps[iStep].predictors[iPtor].datasetId = val.ToStdString();
}

void asParameters::SetPredictorDataId(int iStep, int iPtor, const wxString &val) {
    wxASSERT(!val.IsEmpty());
    m_steps[iStep].predictors[iPtor].dataId = val.ToStdString();
}

void asParameters::SetPredictorLevel(int iStep, int iPtor, float val) {
    wxASSERT(!asIsNaN(val));
    m_steps[iStep].predictors[iPtor].level = val;
}

void asParameters::SetPredictorGridType(int iStep, int iPtor, const wxString &val) {
    wxASSERT(!val.IsEmpty());
    m_steps[iStep].predictors[iPtor].gridType = val.ToStdString();
}

void asParameters::SetPredictorXmin(int iStep, int iPtor, double val) {
    wxASSERT(!asIsNaN(val));
    m_steps[iStep].predictors[iPtor].xMin = val;
}

void asParameters::SetPredictorXptsnb(int iStep, int iPtor, int val) {
    wxASSERT(!asIsNaN(val));
    m_steps[iStep].predictors[iPtor].xPtsNb = val;
}

void asParameters::SetPredictorXstep(int iStep, int iPtor, double val) {
    wxASSERT(!asIsNaN(val));
    m_steps[iStep].predictors[iPtor].xStep = val;
}

void asParameters::SetPredictorXshift(int iStep, int iPtor, double val) {
    wxASSERT(!asIsNaN(val));
    m_steps[iStep].predictors[iPtor].xShift = val;
}

void asParameters::SetPredictorYmin(int iStep, int iPtor, double val) {
    wxASSERT(!asIsNaN(val));
    m_steps[iStep].predictors[iPtor].yMin = val;
}

void asParameters::SetPredictorYptsnb(int iStep, int iPtor, int val) {
    wxASSERT(!asIsNaN(val));
    m_steps[iStep].predictors[iPtor].yPtsNb = val;
}

void asParameters::SetPredictorYstep(int iStep, int iPtor, double val) {
    wxASSERT(!asIsNaN(val));
    m_steps[iStep].predictors[iPtor].yStep = val;
}

void asParameters::SetPredictorYshift(int iStep, int iPtor, double val) {
    wxASSERT(!asIsNaN(val));
    m_steps[iStep].predictors[iPtor].yShift = val;
}

void asParameters::SetPredictorFlatAllowed(int iStep, int iPtor, int val) {
    wxASSERT(!asIsNaN(val));
    m_steps[iStep].predictors[iPtor].flatAllowed = val;
}

void asParameters::SetPredictorHour(int iStep, int iPtor, double val) {
    wxASSERT(!asIsNaN(val));
    m_steps[iStep].predictors[iPtor].hour = val;
}

void asParameters::SetPredictorMembersNb(int iStep, int iPtor, int val) {
    wxASSERT(!asIsNaN(val));
    m_steps[iStep].predictors[iPtor].membersNb = val;
}

void asParameters::SetPredictorCriteria(int iStep, int iPtor, const wxString &val) {
    wxASSERT(!val.IsEmpty());
    m_steps[iStep].predictors[iPtor].criteria = val.ToStdString();
}

void asParameters::SetPredictorWeight(int iStep, int iPtor, float val) {
    wxASSERT(!asIsNaN(val));
    m_steps[iStep].predictors[iPtor].weight = wxMax(val, 0);
}
