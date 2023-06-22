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

#include "asParametersOptimization.h"

#include "asFileParametersOptimization.h"

asParametersOptimization::asParametersOptimization()
    : asParametersScoring(),
      m_variableParamsNb(0),
      m_timeArrayAnalogsIntervalDaysIteration(1),
      m_timeArrayAnalogsIntervalDaysUpperLimit(183),
      m_timeArrayAnalogsIntervalDaysLowerLimit(10),
      m_timeArrayAnalogsIntervalDaysLocks(true) {}

asParametersOptimization::~asParametersOptimization() {}

void asParametersOptimization::AddStep() {
    asParameters::AddStep();

    ParamsStep stepIteration;
    ParamsStep stepUpperLimit;
    ParamsStep stepLowerLimit;
    ParamsStepBool stepLocks;
    ParamsStepVect stepVect;

    stepIteration.analogsNumber = 1;
    stepUpperLimit.analogsNumber = 1000;
    stepLowerLimit.analogsNumber = 5;
    stepLocks.analogsNumber = true;
    stepVect.analogsNumber.push_back(0);

    m_stepsIteration.push_back(stepIteration);
    m_stepsUpperLimit.push_back(stepUpperLimit);
    m_stepsLowerLimit.push_back(stepLowerLimit);
    m_stepsLocks.push_back(stepLocks);
    m_stepsVect.push_back(stepVect);
}

void asParametersOptimization::AddPredictorIteration(ParamsStep& step) {
    ParamsPredictor predictor;

    predictor.xMin = 1;
    predictor.xPtsNb = 1;
    predictor.yMin = 1;
    predictor.yPtsNb = 1;
    predictor.hour = 6;
    predictor.weight = 0.01f;

    step.predictors.push_back(predictor);
}

void asParametersOptimization::AddPredictorUpperLimit(ParamsStep& step) {
    ParamsPredictor predictor;

    predictor.xMin = 717.5;
    predictor.yMin = 87.5;
    predictor.hour = 36;
    predictor.weight = 1;

    step.predictors.push_back(predictor);
}

void asParametersOptimization::AddPredictorLowerLimit(ParamsStep& step) {
    ParamsPredictor predictor;

    predictor.xMin = 0;
    predictor.xPtsNb = 1;
    predictor.yMin = 0;
    predictor.yPtsNb = 1;
    predictor.hour = 0;
    predictor.weight = 0;

    step.predictors.push_back(predictor);
}

void asParametersOptimization::AddPredictorLocks(ParamsStepBool& step) {
    ParamsPredictorBool predictor;

    step.predictors.push_back(predictor);
}

bool asParametersOptimization::LoadFromFile(const wxString& filePath) {
    if (filePath.IsEmpty()) {
        wxLogError(_("The given path to the parameters file is empty."));
        return false;
    }

    asFileParametersOptimization fileParams(filePath, asFile::ReadOnly);
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

        } else if (nodeProcess->GetName() == "evaluation") {
            if (!ParseScore(fileParams, nodeProcess)) return false;

        } else {
            fileParams.UnknownNode(nodeProcess);
        }

        nodeProcess = nodeProcess->GetNext();
    }

    // Set properties
    if (!PreprocessingDataIdsOk()) return false;
    if (!PreprocessingPropertiesOk()) return false;
    SetSpatialWindowProperties();
    SetPreloadingProperties();

    // Check inputs and init parameters
    InitRandomValues();
    if (!InputsOK()) return false;

    // Fixes
    FixTimeLimits();
    FixWeights();
    FixCoordinates();

    return true;
}

bool asParametersOptimization::ParseDescription(asFileParametersOptimization& fileParams,
                                                const wxXmlNode* nodeProcess) {
    wxXmlNode* nodeParam = nodeProcess->GetChildren();
    while (nodeParam) {
        if (nodeParam->GetName() == "method_id") {
            SetMethodId(fileParams.GetString(nodeParam));
        } else if (nodeParam->GetName() == "method_id_display") {
            SetMethodIdDisplay(fileParams.GetString(nodeParam));
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

bool asParametersOptimization::ParseTimeProperties(asFileParametersOptimization& fileParams,
                                                   const wxXmlNode* nodeProcess) {
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
        } else if (nodeParamBlock->GetName() == "calibration_period") {
            wxXmlNode* nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "start_year") {
                    SetCalibrationYearStart(fileParams.GetInt(nodeParam));
                } else if (nodeParam->GetName() == "end_year") {
                    SetCalibrationYearEnd(fileParams.GetInt(nodeParam));
                } else if (nodeParam->GetName() == "start") {
                    SetCalibrationStart(fileParams.GetString(nodeParam));
                } else if (nodeParam->GetName() == "end") {
                    SetCalibrationEnd(fileParams.GetString(nodeParam));
                } else if (nodeParam->GetName() == "time_step") {
                    SetTargetTimeStepHours(fileParams.GetDouble(nodeParam));
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
        } else if (nodeParamBlock->GetName() == "validation_period") {
            wxXmlNode* nodeParam = nodeParamBlock->GetChildren();
            int yStart = 0, yEnd = 0;
            while (nodeParam) {
                if (nodeParam->GetName() == "years") {
                    SetValidationYearsVector(fileParams.GetVectorInt(nodeParam));
                } else if (nodeParam->GetName() == "start_year") {
                    yStart = fileParams.GetInt(nodeParam);
                } else if (nodeParam->GetName() == "end_year") {
                    yEnd = fileParams.GetInt(nodeParam);
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
            if (yStart > 0 && yEnd > 0) {
                vi vect = asFileParameters::BuildVectorInt(yStart, yEnd, 1);
                SetValidationYearsVector(vect);
            }
        } else if (nodeParamBlock->GetName() == "time_step") {
            SetTargetTimeStepHours(fileParams.GetDouble(nodeParamBlock));
            SetAnalogsTimeStepHours(fileParams.GetDouble(nodeParamBlock));
        } else if (nodeParamBlock->GetName() == "time_array_target") {
            wxXmlNode* nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "time_array") {
                    SetTimeArrayTargetMode(fileParams.GetString(nodeParam));
                } else if (nodeParam->GetName() == "predictand_serie_name") {
                    SetTimeArrayTargetPredictandSerieName(fileParams.GetString(nodeParam));
                } else if (nodeParam->GetName() == "predictand_min_threshold") {
                    SetTimeArrayTargetPredictandMinThreshold(fileParams.GetFloat(nodeParam));
                } else if (nodeParam->GetName() == "predictand_max_threshold") {
                    SetTimeArrayTargetPredictandMaxThreshold(fileParams.GetFloat(nodeParam));
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
        } else if (nodeParamBlock->GetName() == "time_array_analogs") {
            wxXmlNode* nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "time_array") {
                    SetTimeArrayAnalogsMode(fileParams.GetString(nodeParam));
                } else if (nodeParam->GetName() == "interval_days") {
                    SetTimeArrayAnalogsIntervalDaysLock(fileParams.GetAttributeBool(nodeParam, "lock", true, false));
                    if (IsTimeArrayAnalogsIntervalDaysLocked()) {
                        SetAnalogsIntervalDays(fileParams.GetInt(nodeParam));
                        SetTimeArrayAnalogsIntervalDaysLowerLimit(GetAnalogsIntervalDays());
                        SetTimeArrayAnalogsIntervalDaysUpperLimit(GetAnalogsIntervalDays());
                        SetTimeArrayAnalogsIntervalDaysIteration(1);
                    } else {
                        SetTimeArrayAnalogsIntervalDaysLowerLimit(fileParams.GetAttributeInt(nodeParam, "lowerlimit"));
                        SetTimeArrayAnalogsIntervalDaysUpperLimit(fileParams.GetAttributeInt(nodeParam, "upperlimit"));
                        SetTimeArrayAnalogsIntervalDaysIteration(fileParams.GetAttributeInt(nodeParam, "iteration"));
                    }
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

bool asParametersOptimization::ParseAnalogDatesParams(asFileParametersOptimization& fileParams, int iStep,
                                                      const wxXmlNode* nodeProcess) {
    int iPtor = 0;
    wxXmlNode* nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "analogs_number") {
            SetAnalogsNumberLock(iStep, fileParams.GetAttributeBool(nodeParamBlock, "lock"));
            if (IsAnalogsNumberLocked(iStep)) {
                SetAnalogsNumber(iStep, fileParams.GetInt(nodeParamBlock));
                SetAnalogsNumberLowerLimit(iStep, GetAnalogsNumber(iStep));
                SetAnalogsNumberUpperLimit(iStep, GetAnalogsNumber(iStep));
                SetAnalogsNumberIteration(iStep, 1);
            } else {
                SetAnalogsNumberLowerLimit(iStep, fileParams.GetAttributeInt(nodeParamBlock, "lowerlimit"));
                SetAnalogsNumberUpperLimit(iStep, fileParams.GetAttributeInt(nodeParamBlock, "upperlimit"));
                SetAnalogsNumberIteration(iStep, fileParams.GetAttributeInt(nodeParamBlock, "iteration"));
            }
        } else if (nodeParamBlock->GetName() == "predictor") {
            AddPredictor(iStep);
            AddPredictorVect(m_stepsVect[iStep]);
            AddPredictorIteration(m_stepsIteration[iStep]);
            AddPredictorUpperLimit(m_stepsUpperLimit[iStep]);
            AddPredictorLowerLimit(m_stepsLowerLimit[iStep]);
            AddPredictorLocks(m_stepsLocks[iStep]);
            SetPreprocess(iStep, iPtor, false);
            SetPreload(iStep, iPtor, true);
            if (!ParsePredictors(fileParams, iStep, iPtor, nodeParamBlock)) return false;
            iPtor++;
        } else {
            fileParams.UnknownNode(nodeParamBlock);
        }
        nodeParamBlock = nodeParamBlock->GetNext();
    }
    return true;
}

bool asParametersOptimization::ParsePredictors(asFileParametersOptimization& fileParams, int iStep, int iPtor,
                                               const wxXmlNode* nodeParamBlock) {
    wxXmlNode* nodeParam = nodeParamBlock->GetChildren();
    while (nodeParam) {
        if (nodeParam->GetName() == "preload") {
            SetPreload(iStep, iPtor, fileParams.GetBool(nodeParam));
            if (!fileParams.GetBool(nodeParam)) {
                wxLogWarning(_("The preload option has been disabled. This can result in very long computation time."));
            }
        } else if (nodeParam->GetName() == "standardize") {
            SetStandardize(iStep, iPtor, fileParams.GetBool(nodeParam));
        } else if (nodeParam->GetName() == "standardize_mean") {
            SetStandardizeMean(iStep, iPtor, fileParams.GetDouble(nodeParam));
        } else if (nodeParam->GetName() == "standardize_sd") {
            SetStandardizeSd(iStep, iPtor, fileParams.GetDouble(nodeParam));
        } else if (nodeParam->GetName() == "preprocessing") {
            SetPreprocess(iStep, iPtor, true);
            if (!ParsePreprocessedPredictors(fileParams, iStep, iPtor, nodeParam)) return false;
        } else if (nodeParam->GetName() == "dataset_id") {
            SetPredictorDatasetId(iStep, iPtor, fileParams.GetString(nodeParam));
        } else if (nodeParam->GetName() == "data_id") {
            SetPredictorDataIdVector(iStep, iPtor, fileParams.GetVectorString(nodeParam));
            SetPredictorDataIdLock(iStep, iPtor, fileParams.GetAttributeBool(nodeParam, "lock", true, false));
            if (IsPredictorDataIdLocked(iStep, iPtor)) {
                SetPredictorDataId(iStep, iPtor, fileParams.GetString(nodeParam));
            }
        } else if (nodeParam->GetName() == "level") {
            SetPredictorLevelVector(iStep, iPtor, fileParams.GetVectorFloat(nodeParam));
            SetPredictorLevelLock(iStep, iPtor, fileParams.GetAttributeBool(nodeParam, "lock", true, false));
            if (IsPredictorLevelLocked(iStep, iPtor)) {
                SetPredictorLevel(iStep, iPtor, fileParams.GetFloat(nodeParam));
            }
        } else if (nodeParam->GetName() == "time") {
            SetPredictorHourLock(iStep, iPtor, fileParams.GetAttributeBool(nodeParam, "lock", true, false));
            if (IsPredictorHourLocked(iStep, iPtor)) {
                SetPredictorHour(iStep, iPtor, fileParams.GetDouble(nodeParam));
                vd vHours;
                vHours.push_back(GetPredictorHour(iStep, iPtor));
                if (!SetPreloadHours(iStep, iPtor, vHours)) return false;
                SetPredictorHoursLowerLimit(iStep, iPtor, GetPredictorHour(iStep, iPtor));
                SetPredictorHoursUpperLimit(iStep, iPtor, GetPredictorHour(iStep, iPtor));
                SetPredictorHoursIteration(iStep, iPtor, 6);
            } else {
                SetPredictorHoursLowerLimit(iStep, iPtor, fileParams.GetAttributeDouble(nodeParam, "lowerlimit"));
                SetPredictorHoursUpperLimit(iStep, iPtor, fileParams.GetAttributeDouble(nodeParam, "upperlimit"));
                SetPredictorHoursIteration(iStep, iPtor, fileParams.GetAttributeDouble(nodeParam, "iteration"));
                // Initialize to ensure correct array sizes
                SetPredictorHour(iStep, iPtor, GetPredictorHoursLowerLimit(iStep, iPtor));
            }
        } else if (nodeParam->GetName() == "members") {
            SetPredictorMembersNb(iStep, iPtor, fileParams.GetInt(nodeParam));
        } else if (nodeParam->GetName() == "spatial_window") {
            if (!ParseSpatialWindow(fileParams, iStep, iPtor, nodeParam)) return false;
        } else if (nodeParam->GetName() == "criteria") {
            SetPredictorCriteriaVector(iStep, iPtor, fileParams.GetVectorString(nodeParam));
            SetPredictorCriteriaLock(iStep, iPtor, fileParams.GetAttributeBool(nodeParam, "lock", true, false));
            if (IsPredictorCriteriaLocked(iStep, iPtor)) {
                SetPredictorCriteria(iStep, iPtor, fileParams.GetString(nodeParam));
            }
        } else if (nodeParam->GetName() == "weight") {
            SetPredictorWeightLock(iStep, iPtor, fileParams.GetAttributeBool(nodeParam, "lock", true, false));
            if (IsPredictorWeightLocked(iStep, iPtor)) {
                SetPredictorWeight(iStep, iPtor, fileParams.GetFloat(nodeParam));
                SetPredictorWeightLowerLimit(iStep, iPtor, GetPredictorWeight(iStep, iPtor));
                SetPredictorWeightUpperLimit(iStep, iPtor, GetPredictorWeight(iStep, iPtor));
                SetPredictorWeightIteration(iStep, iPtor, 1);
            } else {
                SetPredictorWeightLowerLimit(iStep, iPtor, fileParams.GetAttributeFloat(nodeParam, "lowerlimit"));
                SetPredictorWeightUpperLimit(iStep, iPtor, fileParams.GetAttributeFloat(nodeParam, "upperlimit"));
                SetPredictorWeightIteration(iStep, iPtor, fileParams.GetAttributeFloat(nodeParam, "iteration"));
            }
        } else {
            fileParams.UnknownNode(nodeParam);
        }
        nodeParam = nodeParam->GetNext();
    }
    return true;
}

bool asParametersOptimization::ParseSpatialWindow(asFileParametersOptimization& fileParams, int iStep, int iPtor,
                                                  const wxXmlNode* nodeParam) {
    wxXmlNode* nodeWindow = nodeParam->GetChildren();
    while (nodeWindow) {
        if (nodeWindow->GetName() == "grid_type") {
            SetPredictorGridType(iStep, iPtor, fileParams.GetString(nodeWindow, "regular"));
        } else if (nodeWindow->GetName() == "x_min") {
            SetPredictorXminLock(iStep, iPtor, fileParams.GetAttributeBool(nodeWindow, "lock"));
            if (IsPredictorXminLocked(iStep, iPtor)) {
                SetPredictorXmin(iStep, iPtor, fileParams.GetDouble(nodeWindow));
                SetPredictorXminLowerLimit(iStep, iPtor, GetPredictorXmin(iStep, iPtor));
                SetPredictorXminUpperLimit(iStep, iPtor, GetPredictorXmin(iStep, iPtor));
                SetPredictorXminIteration(iStep, iPtor, 1);
            } else {
                SetPredictorXminLowerLimit(iStep, iPtor, fileParams.GetAttributeDouble(nodeWindow, "lowerlimit"));
                SetPredictorXminUpperLimit(iStep, iPtor, fileParams.GetAttributeDouble(nodeWindow, "upperlimit"));
                SetPredictorXminIteration(iStep, iPtor, fileParams.GetAttributeDouble(nodeWindow, "iteration"));
            }
        } else if (nodeWindow->GetName() == "x_points_nb") {
            SetPredictorXptsnbLock(iStep, iPtor, fileParams.GetAttributeBool(nodeWindow, "lock"));
            if (IsPredictorXptsnbLocked(iStep, iPtor)) {
                SetPredictorXptsnb(iStep, iPtor, fileParams.GetDouble(nodeWindow));
                SetPredictorXptsnbLowerLimit(iStep, iPtor, GetPredictorXptsnb(iStep, iPtor));
                SetPredictorXptsnbUpperLimit(iStep, iPtor, GetPredictorXptsnb(iStep, iPtor));
                SetPredictorXptsnbIteration(iStep, iPtor, 1);
            } else {
                SetPredictorXptsnbLowerLimit(iStep, iPtor, fileParams.GetAttributeDouble(nodeWindow, "lowerlimit"));
                SetPredictorXptsnbUpperLimit(iStep, iPtor, fileParams.GetAttributeDouble(nodeWindow, "upperlimit"));
                SetPredictorXptsnbIteration(iStep, iPtor, fileParams.GetAttributeDouble(nodeWindow, "iteration"));
            }
        } else if (nodeWindow->GetName() == "x_step") {
            SetPredictorXstep(iStep, iPtor, fileParams.GetDouble(nodeWindow));
        } else if (nodeWindow->GetName() == "y_min") {
            SetPredictorYminLock(iStep, iPtor, fileParams.GetAttributeBool(nodeWindow, "lock"));
            if (IsPredictorYminLocked(iStep, iPtor)) {
                SetPredictorYmin(iStep, iPtor, fileParams.GetDouble(nodeWindow));
                SetPredictorYminLowerLimit(iStep, iPtor, GetPredictorYmin(iStep, iPtor));
                SetPredictorYminUpperLimit(iStep, iPtor, GetPredictorYmin(iStep, iPtor));
                SetPredictorYminIteration(iStep, iPtor, 1);
            } else {
                SetPredictorYminLowerLimit(iStep, iPtor, fileParams.GetAttributeDouble(nodeWindow, "lowerlimit"));
                SetPredictorYminUpperLimit(iStep, iPtor, fileParams.GetAttributeDouble(nodeWindow, "upperlimit"));
                SetPredictorYminIteration(iStep, iPtor, fileParams.GetAttributeDouble(nodeWindow, "iteration"));
            }
        } else if (nodeWindow->GetName() == "y_points_nb") {
            SetPredictorYptsnbLock(iStep, iPtor, fileParams.GetAttributeBool(nodeWindow, "lock"));
            if (IsPredictorYptsnbLocked(iStep, iPtor)) {
                SetPredictorYptsnb(iStep, iPtor, fileParams.GetDouble(nodeWindow));
                SetPredictorYptsnbLowerLimit(iStep, iPtor, GetPredictorYptsnb(iStep, iPtor));
                SetPredictorYptsnbUpperLimit(iStep, iPtor, GetPredictorYptsnb(iStep, iPtor));
                SetPredictorYptsnbIteration(iStep, iPtor, 1);
            } else {
                SetPredictorYptsnbLowerLimit(iStep, iPtor, fileParams.GetAttributeDouble(nodeWindow, "lowerlimit"));
                SetPredictorYptsnbUpperLimit(iStep, iPtor, fileParams.GetAttributeDouble(nodeWindow, "upperlimit"));
                SetPredictorYptsnbIteration(iStep, iPtor, fileParams.GetAttributeDouble(nodeWindow, "iteration"));
            }
        } else if (nodeWindow->GetName() == "y_step") {
            SetPredictorYstep(iStep, iPtor, fileParams.GetDouble(nodeWindow));
        } else {
            fileParams.UnknownNode(nodeWindow);
        }
        nodeWindow = nodeWindow->GetNext();
    }
    return true;
}

bool asParametersOptimization::ParsePreprocessedPredictors(asFileParametersOptimization& fileParams, int iStep,
                                                           int iPtor, const wxXmlNode* nodeParam) {
    int iPre = 0;
    wxXmlNode* nodePreprocess = nodeParam->GetChildren();
    while (nodePreprocess) {
        if (nodePreprocess->GetName() == "preprocessing_method") {
            SetPreprocessMethod(iStep, iPtor, fileParams.GetString(nodePreprocess));
        } else if (nodePreprocess->GetName() == "preprocessing_data") {
            if (!ParsePreprocessedPredictorDataset(fileParams, iStep, iPtor, iPre, nodePreprocess)) return false;
            iPre++;
        } else {
            fileParams.UnknownNode(nodePreprocess);
        }
        nodePreprocess = nodePreprocess->GetNext();
    }
    return true;
}

bool asParametersOptimization::ParsePreprocessedPredictorDataset(asFileParametersOptimization& fileParams, int iStep,
                                                                 int iPtor, int iPre, const wxXmlNode* nodePreprocess) {
    wxXmlNode* nodeParamPreprocess = nodePreprocess->GetChildren();
    while (nodeParamPreprocess) {
        if (nodeParamPreprocess->GetName() == "dataset_id") {
            SetPreprocessDatasetId(iStep, iPtor, iPre, fileParams.GetString(nodeParamPreprocess));
        } else if (nodeParamPreprocess->GetName() == "data_id") {
            SetPreprocessDataIdVector(iStep, iPtor, iPre, fileParams.GetVectorString(nodeParamPreprocess));
            SetPreprocessDataIdLock(iStep, iPtor, iPre,
                                    fileParams.GetAttributeBool(nodeParamPreprocess, "lock", true, false));
            if (IsPreprocessDataIdLocked(iStep, iPtor, iPre)) {
                SetPreprocessDataId(iStep, iPtor, iPre, fileParams.GetString(nodeParamPreprocess));
            } else {
                // Initialize to ensure correct array sizes
                SetPreprocessDataId(iStep, iPtor, iPre, GetPreprocessDataIdVector(iStep, iPtor, iPre)[0]);
            }
        } else if (nodeParamPreprocess->GetName() == "level") {
            SetPreprocessLevelVector(iStep, iPtor, iPre, fileParams.GetVectorFloat(nodeParamPreprocess));
            SetPreprocessLevelLock(iStep, iPtor, iPre,
                                   fileParams.GetAttributeBool(nodeParamPreprocess, "lock", true, false));
            if (IsPreprocessLevelLocked(iStep, iPtor, iPre)) {
                SetPreprocessLevel(iStep, iPtor, iPre, fileParams.GetFloat(nodeParamPreprocess));
            } else {
                // Initialize to ensure correct array sizes
                SetPreprocessLevel(iStep, iPtor, iPre, GetPreprocessLevelVector(iStep, iPtor, iPre)[0]);
            }
        } else if (nodeParamPreprocess->GetName() == "time") {
            SetPreprocessHourLock(iStep, iPtor, iPre,
                                  fileParams.GetAttributeBool(nodeParamPreprocess, "lock", true, false));
            if (IsPreprocessHourLocked(iStep, iPtor, iPre)) {
                SetPreprocessHour(iStep, iPtor, iPre, fileParams.GetDouble(nodeParamPreprocess));
                SetPreprocessHoursLowerLimit(iStep, iPtor, iPre, GetPreprocessHour(iStep, iPtor, iPre));
                SetPreprocessHoursUpperLimit(iStep, iPtor, iPre, GetPreprocessHour(iStep, iPtor, iPre));
                SetPreprocessHoursIteration(iStep, iPtor, iPre, 6);
            } else {
                SetPreprocessHoursLowerLimit(iStep, iPtor, iPre,
                                             fileParams.GetAttributeDouble(nodeParamPreprocess, "lowerlimit"));
                SetPreprocessHoursUpperLimit(iStep, iPtor, iPre,
                                             fileParams.GetAttributeDouble(nodeParamPreprocess, "upperlimit"));
                SetPreprocessHoursIteration(iStep, iPtor, iPre,
                                            fileParams.GetAttributeDouble(nodeParamPreprocess, "iteration"));
                // Initialize to ensure correct array sizes
                SetPreprocessHour(iStep, iPtor, iPre, GetPreprocessHoursLowerLimit(iStep, iPtor, iPre));
            }
        } else if (nodeParamPreprocess->GetName() == "members") {
            SetPreprocessMembersNb(iStep, iPtor, iPre, fileParams.GetInt(nodeParamPreprocess));
        } else {
            fileParams.UnknownNode(nodeParamPreprocess);
        }
        nodeParamPreprocess = nodeParamPreprocess->GetNext();
    }
    return true;
}

bool asParametersOptimization::ParseAnalogValuesParams(asFileParametersOptimization& fileParams,
                                                       const wxXmlNode* nodeProcess) {
    wxXmlNode* nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "predictand") {
            wxXmlNode* nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "station_id") {
                    SetPredictandStationIds(fileParams.GetStationIds(fileParams.GetString(nodeParam)));
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

bool asParametersOptimization::ParseScore(asFileParametersOptimization& fileParams, const wxXmlNode* nodeProcess) {
    wxXmlNode* nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "score") {
            SetScoreName(fileParams.GetString(nodeParamBlock));
        } else if (nodeParamBlock->GetName() == "threshold") {
            SetScoreThreshold(fileParams.GetFloat(nodeParamBlock));
        } else if (nodeParamBlock->GetName() == "on_mean") {
            SetOnMean(fileParams.GetBool(nodeParamBlock));
        } else if (nodeParamBlock->GetName() == "quantile") {
            SetScoreQuantile(fileParams.GetFloat(nodeParamBlock));
        } else if (nodeParamBlock->GetName() == "time_array") {
            SetScoreTimeArrayMode(fileParams.GetString(nodeParamBlock));
        } else if (nodeParamBlock->GetName() == "postprocessing") {
            wxLogError(_("The postptocessing is not yet fully implemented."));
        } else {
            fileParams.UnknownNode(nodeParamBlock);
        }
        nodeParamBlock = nodeParamBlock->GetNext();
    }
    return true;
}

bool asParametersOptimization::SetSpatialWindowProperties() {
    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        for (int iPtor = 0; iPtor < GetPredictorsNb(iStep); iPtor++) {
            double xShift = std::fmod(GetPredictorXminLowerLimit(iStep, iPtor), GetPredictorXstep(iStep, iPtor));
            if (xShift < 0) xShift += GetPredictorXstep(iStep, iPtor);
            SetPredictorXshift(iStep, iPtor, xShift);

            double yShift = std::fmod(GetPredictorYminLowerLimit(iStep, iPtor), GetPredictorYstep(iStep, iPtor));
            if (yShift < 0) yShift += GetPredictorYstep(iStep, iPtor);
            SetPredictorYshift(iStep, iPtor, yShift);

            if (GetPredictorXptsnbLowerLimit(iStep, iPtor) <= 1 || GetPredictorYptsnbLowerLimit(iStep, iPtor) <= 1)
                SetPredictorFlatAllowed(iStep, iPtor, asFLAT_ALLOWED);
            if (IsPredictorXptsnbLocked(iStep, iPtor) && GetPredictorXptsnb(iStep, iPtor) <= 1)
                SetPredictorFlatAllowed(iStep, iPtor, asFLAT_ALLOWED);
            if (IsPredictorYptsnbLocked(iStep, iPtor) && GetPredictorYptsnb(iStep, iPtor) <= 1)
                SetPredictorFlatAllowed(iStep, iPtor, asFLAT_ALLOWED);
        }
    }

    return true;
}

bool asParametersOptimization::SetPreloadingProperties() {
    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        for (int iPtor = 0; iPtor < GetPredictorsNb(iStep); iPtor++) {
            // Set maximum extent
            if (NeedsPreloading(iStep, iPtor)) {
                // Set maximum extent
                if (!IsPredictorXminLocked(iStep, iPtor)) {
                    SetPreloadXmin(iStep, iPtor, GetPredictorXminLowerLimit(iStep, iPtor));
                } else {
                    SetPreloadXmin(iStep, iPtor, GetPredictorXmin(iStep, iPtor));
                }

                if (!IsPredictorYminLocked(iStep, iPtor)) {
                    SetPreloadYmin(iStep, iPtor, GetPredictorYminLowerLimit(iStep, iPtor));
                } else {
                    SetPreloadYmin(iStep, iPtor, GetPredictorYmin(iStep, iPtor));
                }

                if (!IsPredictorXptsnbLocked(iStep, iPtor)) {
                    int xBasePtsNb = std::abs(GetPredictorXminUpperLimit(iStep, iPtor) -
                                              GetPredictorXminLowerLimit(iStep, iPtor)) /
                                     GetPredictorXstep(iStep, iPtor);
                    SetPreloadXptsnb(iStep, iPtor,
                                     xBasePtsNb + GetPredictorXptsnbUpperLimit(iStep, iPtor));  // No need to add +1
                } else {
                    SetPreloadXptsnb(iStep, iPtor, GetPredictorXptsnb(iStep, iPtor));
                }

                if (!IsPredictorYptsnbLocked(iStep, iPtor)) {
                    int yBasePtsNb = std::abs(GetPredictorYminUpperLimit(iStep, iPtor) -
                                              GetPredictorYminLowerLimit(iStep, iPtor)) /
                                     GetPredictorYstep(iStep, iPtor);
                    SetPreloadYptsnb(iStep, iPtor,
                                     yBasePtsNb + GetPredictorYptsnbUpperLimit(iStep, iPtor));  // No need to add +1
                } else {
                    SetPreloadYptsnb(iStep, iPtor, GetPredictorYptsnb(iStep, iPtor));
                }
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
                if (!SetPreloadDataIds(iStep, iPtor, GetPredictorDataIdVector(iStep, iPtor))) return false;
                if (!SetPreloadLevels(iStep, iPtor, GetPredictorLevelVector(iStep, iPtor))) return false;
                vd vHours;
                for (double h = GetPredictorHoursLowerLimit(iStep, iPtor);
                     h <= GetPredictorHoursUpperLimit(iStep, iPtor); h += GetPredictorHoursIteration(iStep, iPtor)) {
                    vHours.push_back(h);
                }
                if (!SetPreloadHours(iStep, iPtor, vHours)) return false;
            } else if (NeedsPreloading(iStep, iPtor) && NeedsPreprocessing(iStep, iPtor)) {
                // Check the preprocessing method
                wxString method = GetPreprocessMethod(iStep, iPtor);
                vf preprocLevels;
                vd preprocHours;

                // Different actions depending on the preprocessing method.
                if (NeedsGradientPreprocessing(iStep, iPtor)) {
                    preprocLevels = GetPreprocessLevelVector(iStep, iPtor, 0);

                    for (double h = GetPreprocessHoursLowerLimit(iStep, iPtor, 0);
                         h <= GetPreprocessHoursUpperLimit(iStep, iPtor, 0);
                         h += GetPreprocessHoursIteration(iStep, iPtor, 0)) {
                        preprocHours.push_back(h);
                    }
                } else if (method.IsSameAs("HumidityFlux")) {
                    preprocLevels = GetPreprocessLevelVector(iStep, iPtor, 0);

                    for (double h = GetPreprocessHoursLowerLimit(iStep, iPtor, 0);
                         h <= GetPreprocessHoursUpperLimit(iStep, iPtor, 0);
                         h += GetPreprocessHoursIteration(iStep, iPtor, 0)) {
                        preprocHours.push_back(h);
                    }
                } else if (method.IsSameAs("Multiplication") || method.IsSameAs("Multiply") ||
                           method.IsSameAs("HumidityIndex")) {
                    preprocLevels = GetPreprocessLevelVector(iStep, iPtor, 0);

                    for (double h = GetPreprocessHoursLowerLimit(iStep, iPtor, 0);
                         h <= GetPreprocessHoursUpperLimit(iStep, iPtor, 0);
                         h += GetPreprocessHoursIteration(iStep, iPtor, 0)) {
                        preprocHours.push_back(h);
                    }
                } else if (method.IsSameAs("FormerHumidityIndex")) {
                    wxLogWarning(_("The %s preprocessing method is not handled in the optimizer."), method);
                    return false;
                } else {
                    wxLogWarning(_("The %s preprocessing method is not yet handled with the preload option."), method);
                }

                if (!SetPreloadLevels(iStep, iPtor, preprocLevels)) return false;
                if (!SetPreloadHours(iStep, iPtor, preprocHours)) return false;
            }
        }
    }

    return true;
}

void asParametersOptimization::InitRandomValues() {
    if (!m_timeArrayAnalogsIntervalDaysLocks) {
        m_analogsIntervalDays = asRandom(m_timeArrayAnalogsIntervalDaysLowerLimit,
                                         m_timeArrayAnalogsIntervalDaysUpperLimit,
                                         m_timeArrayAnalogsIntervalDaysIteration);
    }

    for (int i = 0; i < GetStepsNb(); i++) {
        if (!m_stepsLocks[i].analogsNumber) {
            SetAnalogsNumber(i, asRandom(m_stepsLowerLimit[i].analogsNumber, m_stepsUpperLimit[i].analogsNumber,
                                         m_stepsIteration[i].analogsNumber));
        }

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (!m_stepsLocks[i].predictors[j].preprocessDataId[k]) {
                        int length = m_stepsVect[i].predictors[j].preprocessDataId[k].size();
                        int row = asRandom(0, length - 1);
                        wxASSERT(m_stepsVect[i].predictors[j].preprocessDataId[k].size() > row);

                        SetPreprocessDataId(i, j, k, m_stepsVect[i].predictors[j].preprocessDataId[k][row]);
                    }

                    if (!m_stepsLocks[i].predictors[j].preprocessLevels[k]) {
                        int length = m_stepsVect[i].predictors[j].preprocessLevels[k].size();
                        int row = asRandom(0, length - 1);
                        wxASSERT(m_stepsVect[i].predictors[j].preprocessLevels[k].size() > row);

                        SetPreprocessLevel(i, j, k, m_stepsVect[i].predictors[j].preprocessLevels[k][row]);
                    }

                    if (!m_stepsLocks[i].predictors[j].preprocessHours[k]) {
                        SetPreprocessHour(i, j, k,
                                          asRandom(m_stepsLowerLimit[i].predictors[j].preprocessHours[k],
                                                   m_stepsUpperLimit[i].predictors[j].preprocessHours[k],
                                                   m_stepsIteration[i].predictors[j].preprocessHours[k]));
                    }
                }
            } else {
                if (!m_stepsLocks[i].predictors[j].dataId) {
                    int length = m_stepsVect[i].predictors[j].dataId.size();
                    int row = asRandom(0, length - 1);
                    wxASSERT(m_stepsVect[i].predictors[j].dataId.size() > row);

                    SetPredictorDataId(i, j, m_stepsVect[i].predictors[j].dataId[row]);
                }

                if (!m_stepsLocks[i].predictors[j].level) {
                    int length = m_stepsVect[i].predictors[j].level.size();
                    int row = asRandom(0, length - 1);
                    wxASSERT(m_stepsVect[i].predictors[j].level.size() > row);

                    SetPredictorLevel(i, j, m_stepsVect[i].predictors[j].level[row]);
                }

                if (!m_stepsLocks[i].predictors[j].hours) {
                    SetPredictorHour(
                        i, j,
                        asRandom(m_stepsLowerLimit[i].predictors[j].hour, m_stepsUpperLimit[i].predictors[j].hour,
                                 m_stepsIteration[i].predictors[j].hour));
                }
            }

            if (!m_stepsLocks[i].predictors[j].xMin) {
                SetPredictorXmin(
                    i, j,
                    asRandom(m_stepsLowerLimit[i].predictors[j].xMin, m_stepsUpperLimit[i].predictors[j].xMin,
                             m_stepsIteration[i].predictors[j].xMin));
            }

            if (!m_stepsLocks[i].predictors[j].xPtsNb) {
                SetPredictorXptsnb(
                    i, j,
                    asRandom(m_stepsLowerLimit[i].predictors[j].xPtsNb, m_stepsUpperLimit[i].predictors[j].xPtsNb,
                             m_stepsIteration[i].predictors[j].xPtsNb));
            }

            if (!m_stepsLocks[i].predictors[j].yMin) {
                SetPredictorYmin(
                    i, j,
                    asRandom(m_stepsLowerLimit[i].predictors[j].yMin, m_stepsUpperLimit[i].predictors[j].yMin,
                             m_stepsIteration[i].predictors[j].yMin));
            }

            if (!m_stepsLocks[i].predictors[j].yPtsNb) {
                SetPredictorYptsnb(
                    i, j,
                    asRandom(m_stepsLowerLimit[i].predictors[j].yPtsNb, m_stepsUpperLimit[i].predictors[j].yPtsNb,
                             m_stepsIteration[i].predictors[j].yPtsNb));
            }

            if (!m_stepsLocks[i].predictors[j].weight) {
                SetPredictorWeight(
                    i, j,
                    asRandom(m_stepsLowerLimit[i].predictors[j].weight, m_stepsUpperLimit[i].predictors[j].weight,
                             m_stepsIteration[i].predictors[j].weight));
            }

            if (!m_stepsLocks[i].predictors[j].criteria) {
                int length = m_stepsVect[i].predictors[j].criteria.size();
                int row = asRandom(0, length - 1);
                wxASSERT(m_stepsVect[i].predictors[j].criteria.size() > row);

                SetPredictorCriteria(i, j, m_stepsVect[i].predictors[j].criteria[row]);
            }

            // Fix the criteria if S1
            if (NeedsPreprocessing(i, j)) {
                FixCriteriaIfGradientsPreprocessed(i, j);
            }
        }
    }

    FixWeights();
    FixCoordinates();
    CheckRange();
    FixAnalogsNb();
}

void asParametersOptimization::CheckRange() {
    // Check that the actual parameters values are within ranges
    if (!m_timeArrayAnalogsIntervalDaysLocks) {
        m_analogsIntervalDays = wxMax(wxMin(m_analogsIntervalDays, m_timeArrayAnalogsIntervalDaysUpperLimit),
                                      m_timeArrayAnalogsIntervalDaysLowerLimit);
    }
    wxASSERT(m_analogsIntervalDays > 0);

    for (int i = 0; i < GetStepsNb(); i++) {
        if (!m_stepsLocks[i].analogsNumber) {
            SetAnalogsNumber(i, wxMax(wxMin(GetAnalogsNumber(i), m_stepsUpperLimit[i].analogsNumber),
                                      m_stepsLowerLimit[i].analogsNumber));
        }

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (!GetPredictorGridType(i, j).IsSameAs("Regular", false))
                throw runtime_error(asStrF(_("asParametersOptimization::CheckRange is not ready to use on "
                                             "unregular grids (PredictorGridType = %s)"),
                                           GetPredictorGridType(i, j)));

            if (NeedsPreprocessing(i, j)) {
                int preprocessSize = GetPreprocessSize(i, j);
                for (int k = 0; k < preprocessSize; k++) {
                    if (!m_stepsLocks[i].predictors[j].preprocessHours[k]) {
                        SetPreprocessHour(i, j, k,
                                          wxMax(wxMin(GetPreprocessHour(i, j, k),
                                                      m_stepsUpperLimit[i].predictors[j].preprocessHours[k]),
                                                m_stepsLowerLimit[i].predictors[j].preprocessHours[k]));
                    }
                    SetPredictorHour(i, j, 0);
                }
            } else {
                if (!m_stepsLocks[i].predictors[j].hours) {
                    SetPredictorHour(i, j,
                                     wxMax(wxMin(GetPredictorHour(i, j), m_stepsUpperLimit[i].predictors[j].hour),
                                           m_stepsLowerLimit[i].predictors[j].hour));
                }
            }

            // Check ranges
            if (!m_stepsLocks[i].predictors[j].xMin) {
                SetPredictorXmin(i, j,
                                 wxMax(wxMin(GetPredictorXmin(i, j), m_stepsUpperLimit[i].predictors[j].xMin),
                                       m_stepsLowerLimit[i].predictors[j].xMin));
            }
            if (!m_stepsLocks[i].predictors[j].xPtsNb) {
                SetPredictorXptsnb(i, j,
                                   wxMax(wxMin(GetPredictorXptsnb(i, j), m_stepsUpperLimit[i].predictors[j].xPtsNb),
                                         m_stepsLowerLimit[i].predictors[j].xPtsNb));
            }

            if (!m_stepsLocks[i].predictors[j].yMin) {
                SetPredictorYmin(i, j,
                                 wxMax(wxMin(GetPredictorYmin(i, j), m_stepsUpperLimit[i].predictors[j].yMin),
                                       m_stepsLowerLimit[i].predictors[j].yMin));
            }
            if (!m_stepsLocks[i].predictors[j].yPtsNb) {
                SetPredictorYptsnb(i, j,
                                   wxMax(wxMin(GetPredictorYptsnb(i, j), m_stepsUpperLimit[i].predictors[j].yPtsNb),
                                         m_stepsLowerLimit[i].predictors[j].yPtsNb));
            }
            if (!m_stepsLocks[i].predictors[j].weight) {
                SetPredictorWeight(i, j,
                                   wxMax(wxMin(GetPredictorWeight(i, j), m_stepsUpperLimit[i].predictors[j].weight),
                                         m_stepsLowerLimit[i].predictors[j].weight));
            }

            if (!m_stepsLocks[i].predictors[j].xMin || !m_stepsLocks[i].predictors[j].xPtsNb) {
                if (GetPredictorXmin(i, j) + (GetPredictorXptsnb(i, j) - 1) * GetPredictorXstep(i, j) >
                    m_stepsUpperLimit[i].predictors[j].xMin +
                        (m_stepsUpperLimit[i].predictors[j].xPtsNb - 1) * GetPredictorXstep(i, j)) {
                    if (!m_stepsLocks[i].predictors[j].xPtsNb) {
                        SetPredictorXptsnb(i, j,
                                           (m_stepsUpperLimit[i].predictors[j].xMin - GetPredictorXmin(i, j)) /
                                                   GetPredictorXstep(i, j) +
                                               m_stepsUpperLimit[i].predictors[j].xPtsNb);  // Correct, no need of +1
                    } else {
                        SetPredictorXmin(i, j,
                                         m_stepsUpperLimit[i].predictors[j].xMin -
                                             GetPredictorXptsnb(i, j) * GetPredictorXstep(i, j));
                    }
                }
            }

            if (!m_stepsLocks[i].predictors[j].yMin || !m_stepsLocks[i].predictors[j].yPtsNb) {
                if (GetPredictorYmin(i, j) + (GetPredictorYptsnb(i, j) - 1) * GetPredictorYstep(i, j) >
                    m_stepsUpperLimit[i].predictors[j].yMin +
                        (m_stepsUpperLimit[i].predictors[j].yPtsNb - 1) * GetPredictorYstep(i, j)) {
                    if (!m_stepsLocks[i].predictors[j].yPtsNb) {
                        SetPredictorYptsnb(i, j,
                                           (m_stepsUpperLimit[i].predictors[j].yMin - GetPredictorYmin(i, j)) /
                                                   GetPredictorYstep(i, j) +
                                               m_stepsUpperLimit[i].predictors[j].yPtsNb);  // Correct, no need of +1
                    } else {
                        SetPredictorYmin(i, j,
                                         m_stepsUpperLimit[i].predictors[j].yMin -
                                             GetPredictorYptsnb(i, j) * GetPredictorYstep(i, j));
                    }
                }
            }
        }
    }

    FixHours();
    FixWeights();
    FixCoordinates();
    FixAnalogsNb();
}

bool asParametersOptimization::IsInRange() {
    // Check that the actual parameters values are within ranges
    if (!m_timeArrayAnalogsIntervalDaysLocks) {
        if (m_analogsIntervalDays > m_timeArrayAnalogsIntervalDaysUpperLimit) return false;
        if (m_analogsIntervalDays < m_timeArrayAnalogsIntervalDaysLowerLimit) return false;
    }

    for (int i = 0; i < GetStepsNb(); i++) {
        if (!m_stepsLocks[i].analogsNumber) {
            if (GetAnalogsNumber(i) > m_stepsUpperLimit[i].analogsNumber) return false;
            if (GetAnalogsNumber(i) < m_stepsLowerLimit[i].analogsNumber) return false;
        }

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (!m_stepsLocks[i].predictors[j].preprocessHours[k]) {
                        if (GetPreprocessHour(i, j, k) < m_stepsLowerLimit[i].predictors[j].preprocessHours[k])
                            return false;
                        if (GetPreprocessHour(i, j, k) < m_stepsLowerLimit[i].predictors[j].preprocessHours[k])
                            return false;
                    }
                }
            } else {
                if (!m_stepsLocks[i].predictors[j].hours) {
                    if (GetPredictorHour(i, j) < m_stepsLowerLimit[i].predictors[j].hour) return false;
                    if (GetPredictorHour(i, j) < m_stepsLowerLimit[i].predictors[j].hour) return false;
                }
            }

            if (!GetPredictorGridType(i, j).IsSameAs("Regular", false))
                throw runtime_error(asStrF(_("asParametersOptimization::CheckRange is not ready to use on "
                                             "unregular grids (PredictorGridType = %s)"),
                                           GetPredictorGridType(i, j)));

            // Check ranges
            if (!m_stepsLocks[i].predictors[j].xMin) {
                if (GetPredictorXmin(i, j) > m_stepsUpperLimit[i].predictors[j].xMin) return false;
                if (GetPredictorXmin(i, j) < m_stepsLowerLimit[i].predictors[j].xMin) return false;
            }
            if (!m_stepsLocks[i].predictors[j].xPtsNb) {
                if (GetPredictorXptsnb(i, j) < m_stepsLowerLimit[i].predictors[j].xPtsNb) return false;
                if (GetPredictorXptsnb(i, j) < m_stepsLowerLimit[i].predictors[j].xPtsNb) return false;
            }
            if (!m_stepsLocks[i].predictors[j].yMin) {
                if (GetPredictorYmin(i, j) < m_stepsLowerLimit[i].predictors[j].yMin) return false;
                if (GetPredictorYmin(i, j) < m_stepsLowerLimit[i].predictors[j].yMin) return false;
            }
            if (!m_stepsLocks[i].predictors[j].yPtsNb) {
                if (GetPredictorYptsnb(i, j) < m_stepsLowerLimit[i].predictors[j].yPtsNb) return false;
                if (GetPredictorYptsnb(i, j) < m_stepsLowerLimit[i].predictors[j].yPtsNb) return false;
            }
            if (!m_stepsLocks[i].predictors[j].weight) {
                if (GetPredictorWeight(i, j) < m_stepsLowerLimit[i].predictors[j].weight) return false;
                if (GetPredictorWeight(i, j) < m_stepsLowerLimit[i].predictors[j].weight) return false;
            }
            if (!m_stepsLocks[i].predictors[j].xMin || !m_stepsLocks[i].predictors[j].xPtsNb ||
                !m_stepsLocks[i].predictors[j].yMin || !m_stepsLocks[i].predictors[j].yPtsNb) {
                if (GetPredictorXmin(i, j) + GetPredictorXptsnb(i, j) * GetPredictorXstep(i, j) >
                    m_stepsUpperLimit[i].predictors[j].xMin +
                        m_stepsLowerLimit[i].predictors[j].xPtsNb * GetPredictorXstep(i, j))
                    return false;
                if (GetPredictorYmin(i, j) + GetPredictorYptsnb(i, j) * GetPredictorYstep(i, j) >
                    m_stepsUpperLimit[i].predictors[j].yMin +
                        m_stepsLowerLimit[i].predictors[j].yPtsNb * GetPredictorYstep(i, j))
                    return false;
            }
        }
    }

    return true;
}

bool asParametersOptimization::FixTimeLimits() {
    double minHour = 200.0, maxHour = -50.0;
    for (int i = 0; i < GetStepsNb(); i++) {
        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    minHour = wxMin(GetPreprocessHoursLowerLimit(i, j, k), minHour);
                    maxHour = wxMax(GetPreprocessHoursUpperLimit(i, j, k), maxHour);
                }
            } else {
                minHour = wxMin(GetPredictorHoursLowerLimit(i, j), minHour);
                maxHour = wxMax(GetPredictorHoursUpperLimit(i, j), maxHour);
            }
        }
    }

    m_timeMinHours = minHour;
    m_timeMaxHours = maxHour;

    return true;
}

void asParametersOptimization::FixHours() {
    for (int i = 0; i < GetStepsNb(); i++) {
        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (!m_stepsLocks[i].predictors[j].preprocessHours[k]) {
                        if (m_stepsIteration[i].predictors[j].preprocessHours[k] != 0) {
                            float ratio = (float)GetPreprocessHour(i, j, k) /
                                          (float)m_stepsIteration[i].predictors[j].preprocessHours[k];
                            ratio = asRound(ratio);
                            SetPreprocessHour(i, j, k, ratio * m_stepsIteration[i].predictors[j].preprocessHours[k]);
                        }
                    }
                }
            } else {
                if (!m_stepsLocks[i].predictors[j].hours) {
                    if (m_stepsIteration[i].predictors[j].hour != 0) {
                        float ratio = (float)GetPredictorHour(i, j) / (float)m_stepsIteration[i].predictors[j].hour;
                        ratio = asRound(ratio);
                        SetPredictorHour(i, j, ratio * m_stepsIteration[i].predictors[j].hour);
                    }
                }
            }
        }
    }
}

bool asParametersOptimization::FixWeights() {
    for (int i = 0; i < GetStepsNb(); i++) {
        // Sum the weights
        float totWeight = 0, totWeightLocked = 0;
        for (int j = 0; j < GetPredictorsNb(i); j++) {
            totWeight += GetPredictorWeight(i, j);

            if (IsPredictorWeightLocked(i, j)) {
                totWeightLocked += GetPredictorWeight(i, j);
            }
        }
        float totWeightManageable = totWeight - totWeightLocked;

        // Check total of the locked weights
        if (totWeightLocked > 1) {
            wxLogWarning(_("The sum of the locked weights of the analogy level number %d is higher than 1 (%f). They "
                           "were forced as unlocked."),
                         i + 1, totWeightLocked);
            totWeightManageable = totWeight;
            totWeightLocked = 0;
        }

        // Reset weights when sum is null
        if (totWeightManageable == 0) {
            for (int j = 0; j < GetPredictorsNb(i); j++) {
                if (!IsPredictorWeightLocked(i, j)) {
                    SetPredictorWeight(i, j, 1);
                    totWeightManageable += 1;
                }
            }
        }

        // For every weights but the last
        float newSum = 0;
        for (int j = 0; j < GetPredictorsNb(i) - 1; j++) {
            if (totWeightLocked > 1) {
                float precision = GetPredictorWeightIteration(i, j);
                float newWeight = GetPredictorWeight(i, j) / totWeightManageable;
                newWeight = wxMax(precision * asRound(newWeight * (1.0 / precision)),
                                  GetPredictorWeightLowerLimit(i, j));
                newSum += newWeight;

                SetPredictorWeight(i, j, newWeight);
            } else {
                if (!IsPredictorWeightLocked(i, j)) {
                    float precision = GetPredictorWeightIteration(i, j);
                    float newWeight = GetPredictorWeight(i, j) / totWeightManageable;
                    newWeight = wxMax(precision * asRound(newWeight * (1.0 / precision)),
                                      GetPredictorWeightLowerLimit(i, j));
                    newSum += newWeight;

                    SetPredictorWeight(i, j, newWeight);
                }
            }
        }

        // Last weight: difference to 0
        float lastWeight = wxMax(1.0f - newSum - totWeightLocked,
                                 GetPredictorWeightLowerLimit(i, GetPredictorsNb(i) - 1));
        SetPredictorWeight(i, GetPredictorsNb(i) - 1, lastWeight);
    }

    return true;
}

void asParametersOptimization::LockAll() {
    m_timeArrayAnalogsIntervalDaysLocks = true;

    for (int i = 0; i < GetStepsNb(); i++) {
        m_stepsLocks[i].analogsNumber = true;

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    m_stepsLocks[i].predictors[j].preprocessDataId[k] = true;
                    m_stepsLocks[i].predictors[j].preprocessLevels[k] = true;
                    m_stepsLocks[i].predictors[j].preprocessHours[k] = true;
                }
            } else {
                m_stepsLocks[i].predictors[j].dataId = true;
                m_stepsLocks[i].predictors[j].level = true;
                m_stepsLocks[i].predictors[j].hours = true;
            }

            m_stepsLocks[i].predictors[j].xMin = true;
            m_stepsLocks[i].predictors[j].xPtsNb = true;
            m_stepsLocks[i].predictors[j].yMin = true;
            m_stepsLocks[i].predictors[j].yPtsNb = true;
            m_stepsLocks[i].predictors[j].weight = true;
            m_stepsLocks[i].predictors[j].criteria = true;
        }
    }

    return;
}

void asParametersOptimization::Unlock(vi& indices) {
    int counter = 0;
    int length = indices.size();

    if (asFind(&indices[0], &indices[length - 1], counter) >= 0) {
        m_timeArrayAnalogsIntervalDaysLocks = false;
    }
    counter++;

    for (int i = 0; i < GetStepsNb(); i++) {
        if (asFind(&indices[0], &indices[length - 1], counter) >= 0) {
            m_stepsLocks[i].analogsNumber = false;
        }
        counter++;

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (asFind(&indices[0], &indices[length - 1], counter) >= 0) {
                        m_stepsLocks[i].predictors[j].preprocessDataId[k] = false;
                    }
                    counter++;
                    if (asFind(&indices[0], &indices[length - 1], counter) >= 0) {
                        m_stepsLocks[i].predictors[j].preprocessLevels[k] = false;
                    }
                    counter++;
                    if (asFind(&indices[0], &indices[length - 1], counter) >= 0) {
                        m_stepsLocks[i].predictors[j].preprocessHours[k] = false;
                    }
                    counter++;
                }
            } else {
                if (asFind(&indices[0], &indices[length - 1], counter) >= 0) {
                    m_stepsLocks[i].predictors[j].dataId = false;
                }
                counter++;
                if (asFind(&indices[0], &indices[length - 1], counter) >= 0) {
                    m_stepsLocks[i].predictors[j].level = false;
                }
                counter++;
                if (asFind(&indices[0], &indices[length - 1], counter) >= 0) {
                    m_stepsLocks[i].predictors[j].hours = false;
                }
                counter++;
            }

            if (asFind(&indices[0], &indices[length - 1], counter) >= 0) {
                m_stepsLocks[i].predictors[j].xMin = false;
            }
            counter++;
            if (asFind(&indices[0], &indices[length - 1], counter) >= 0) {
                m_stepsLocks[i].predictors[j].xPtsNb = false;
            }
            counter++;
            if (asFind(&indices[0], &indices[length - 1], counter) >= 0) {
                m_stepsLocks[i].predictors[j].yMin = false;
            }
            counter++;
            if (asFind(&indices[0], &indices[length - 1], counter) >= 0) {
                m_stepsLocks[i].predictors[j].yPtsNb = false;
            }
            counter++;
            if (asFind(&indices[0], &indices[length - 1], counter) >= 0) {
                m_stepsLocks[i].predictors[j].weight = false;
            }
            counter++;
            if (asFind(&indices[0], &indices[length - 1], counter) >= 0) {
                m_stepsLocks[i].predictors[j].criteria = false;
            }
            counter++;
        }
    }
}

int asParametersOptimization::GetVariablesNb() {
    int counter = 0;

    if (!m_timeArrayAnalogsIntervalDaysLocks) counter++;

    for (int i = 0; i < GetStepsNb(); i++) {
        if (!m_stepsLocks[i].analogsNumber) counter++;

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (!m_stepsLocks[i].predictors[j].preprocessDataId[k]) counter++;
                    if (!m_stepsLocks[i].predictors[j].preprocessLevels[k]) counter++;
                    if (!m_stepsLocks[i].predictors[j].preprocessHours[k]) counter++;
                }
            } else {
                if (!m_stepsLocks[i].predictors[j].dataId) counter++;
                if (!m_stepsLocks[i].predictors[j].level) counter++;
                if (!m_stepsLocks[i].predictors[j].hours) counter++;
            }

            if (!m_stepsLocks[i].predictors[j].xMin) counter++;
            if (!m_stepsLocks[i].predictors[j].xPtsNb) counter++;
            if (!m_stepsLocks[i].predictors[j].yMin) counter++;
            if (!m_stepsLocks[i].predictors[j].yPtsNb) counter++;
            if (!m_stepsLocks[i].predictors[j].weight) counter++;
            if (!m_stepsLocks[i].predictors[j].criteria) counter++;
        }
    }

    return counter;
}
