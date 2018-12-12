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

#include <asFileParametersOptimization.h>


asParametersOptimization::asParametersOptimization()
        : asParametersScoring(),
          m_variableParamsNb(0),
          m_timeArrayAnalogsIntervalDaysIteration(1),
          m_timeArrayAnalogsIntervalDaysUpperLimit(182),
          m_timeArrayAnalogsIntervalDaysLowerLimit(10),
          m_timeArrayAnalogsIntervalDaysLocks(false)
{

}

asParametersOptimization::~asParametersOptimization()
{
    //dtor
}

void asParametersOptimization::AddStep()
{
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

void asParametersOptimization::AddPredictorIteration(ParamsStep &step)
{
    ParamsPredictor predictor;

    predictor.xMin = 2.5;
    predictor.xPtsNb = 1;
    predictor.yMin = 2.5;
    predictor.yPtsNb = 1;
    predictor.timeHours = 6;
    predictor.weight = 0.01f;

    step.predictors.push_back(predictor);
}

void asParametersOptimization::AddPredictorUpperLimit(ParamsStep &step)
{
    ParamsPredictor predictor;

    predictor.xMin = 717.5;
    predictor.xPtsNb = 20;
    predictor.yMin = 87.5;
    predictor.yPtsNb = 16;
    predictor.timeHours = 36;
    predictor.weight = 1;

    step.predictors.push_back(predictor);
}

void asParametersOptimization::AddPredictorLowerLimit(ParamsStep &step)
{
    ParamsPredictor predictor;

    predictor.xMin = 0;
    predictor.xPtsNb = 1;
    predictor.yMin = 0;
    predictor.yPtsNb = 1;
    predictor.timeHours = 6;
    predictor.weight = 0;

    step.predictors.push_back(predictor);
}

void asParametersOptimization::AddPredictorLocks(ParamsStepBool &step)
{
    ParamsPredictorBool predictor;

    predictor.dataId = true;
    predictor.level = true;
    predictor.xMin = true;
    predictor.xPtsNb = true;
    predictor.yMin = true;
    predictor.yPtsNb = true;
    predictor.timeHours = true;
    predictor.weight = true;
    predictor.criteria = true;

    step.predictors.push_back(predictor);
}

bool asParametersOptimization::LoadFromFile(const wxString &filePath)
{
    wxLogVerbose(_("Loading parameters file."));

    if (filePath.IsEmpty()) {
        wxLogError(_("The given path to the parameters file is empty."));
        return false;
    }

    asFileParametersOptimization fileParams(filePath, asFile::ReadOnly);
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

        } else if (nodeProcess->GetName() == "evaluation") {
            if (!ParseScore(fileParams, nodeProcess))
                return false;

        } else {
            fileParams.UnknownNode(nodeProcess);
        }

        nodeProcess = nodeProcess->GetNext();
    }

    // Set properties
    if (!PreprocessingDataIdsOk())
        return false;
    if (!PreprocessingPropertiesOk())
        return false;
    SetSpatialWindowProperties();
    SetPreloadingProperties();

    // Check inputs and init parameters
    InitRandomValues();
    if (!InputsOK())
        return false;

    // Fixes
    FixTimeLimits();
    FixWeights();
    FixCoordinates();

    wxLogVerbose(_("Parameters file loaded."));

    return true;
}

bool asParametersOptimization::ParseDescription(asFileParametersOptimization &fileParams, const wxXmlNode *nodeProcess)
{
    wxXmlNode *nodeParam = nodeProcess->GetChildren();
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

bool asParametersOptimization::ParseTimeProperties(asFileParametersOptimization &fileParams,
                                                   const wxXmlNode *nodeProcess)
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
        } else if (nodeParamBlock->GetName() == "calibration_period") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "start_year") {
                    if (!SetCalibrationYearStart(fileParams.GetInt(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "end_year") {
                    if (!SetCalibrationYearEnd(fileParams.GetInt(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "start") {
                    if (!SetCalibrationStart(fileParams.GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "end") {
                    if (!SetCalibrationEnd(fileParams.GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "time_step") {
                    if (!SetTimeArrayTargetTimeStepHours(fileParams.GetDouble(nodeParam)))
                        return false;
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
        } else if (nodeParamBlock->GetName() == "validation_period") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            int yStart = 0, yEnd = 0;
            while (nodeParam) {
                if (nodeParam->GetName() == "years") {
                    if (!SetValidationYearsVector(fileParams.GetVectorInt(nodeParam)))
                        return false;
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
                if (!SetValidationYearsVector(vect))
                    return false;
            }
        } else if (nodeParamBlock->GetName() == "time_step") {
            if (!SetTimeArrayTargetTimeStepHours(fileParams.GetDouble(nodeParamBlock)))
                return false;
            if (!SetTimeArrayAnalogsTimeStepHours(fileParams.GetDouble(nodeParamBlock)))
                return false;
        } else if (nodeParamBlock->GetName() == "time_array_target") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "time_array") {
                    if (!SetTimeArrayTargetMode(fileParams.GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "predictand_serie_name") {
                    if (!SetTimeArrayTargetPredictandSerieName(fileParams.GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "predictand_min_threshold") {
                    if (!SetTimeArrayTargetPredictandMinThreshold(fileParams.GetFloat(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "predictand_max_threshold") {
                    if (!SetTimeArrayTargetPredictandMaxThreshold(fileParams.GetFloat(nodeParam)))
                        return false;
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
        } else if (nodeParamBlock->GetName() == "time_array_analogs") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "time_array") {
                    if (!SetTimeArrayAnalogsMode(fileParams.GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "interval_days") {
                    SetTimeArrayAnalogsIntervalDaysLock(fileParams.GetAttributeBool(nodeParam, "lock", true, false));
                    if (IsTimeArrayAnalogsIntervalDaysLocked()) {
                        if (!SetTimeArrayAnalogsIntervalDays(fileParams.GetInt(nodeParam)))
                            return false;
                        if (!SetTimeArrayAnalogsIntervalDaysLowerLimit(GetTimeArrayAnalogsIntervalDays()))
                            return false;
                        if (!SetTimeArrayAnalogsIntervalDaysUpperLimit(GetTimeArrayAnalogsIntervalDays()))
                            return false;
                        if (!SetTimeArrayAnalogsIntervalDaysIteration(1))
                            return false;
                    } else {
                        if (!SetTimeArrayAnalogsIntervalDaysLowerLimit(
                                fileParams.GetAttributeInt(nodeParam, "lowerlimit")))
                            return false;
                        if (!SetTimeArrayAnalogsIntervalDaysUpperLimit(
                                fileParams.GetAttributeInt(nodeParam, "upperlimit")))
                            return false;
                        if (!SetTimeArrayAnalogsIntervalDaysIteration(
                                fileParams.GetAttributeInt(nodeParam, "iteration")))
                            return false;
                    }
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

bool asParametersOptimization::ParseAnalogDatesParams(asFileParametersOptimization &fileParams, int iStep,
                                                      const wxXmlNode *nodeProcess)
{
    int iPtor = 0;
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "analogs_number") {
            SetAnalogsNumberLock(iStep, fileParams.GetAttributeBool(nodeParamBlock, "lock"));
            if (IsAnalogsNumberLocked(iStep)) {
                if (!SetAnalogsNumber(iStep, fileParams.GetInt(nodeParamBlock)))
                    return false;
                if (!SetAnalogsNumberLowerLimit(iStep, GetAnalogsNumber(iStep)))
                    return false;
                if (!SetAnalogsNumberUpperLimit(iStep, GetAnalogsNumber(iStep)))
                    return false;
                if (!SetAnalogsNumberIteration(iStep, 1))
                    return false;
            } else {
                if (!SetAnalogsNumberLowerLimit(iStep, fileParams.GetAttributeInt(nodeParamBlock, "lowerlimit")))
                    return false;
                if (!SetAnalogsNumberUpperLimit(iStep, fileParams.GetAttributeInt(nodeParamBlock, "upperlimit")))
                    return false;
                if (!SetAnalogsNumberIteration(iStep, fileParams.GetAttributeInt(nodeParamBlock, "iteration")))
                    return false;
            }
        } else if (nodeParamBlock->GetName() == "predictor") {
            AddPredictor(iStep);
            AddPredictorVect(m_stepsVect[iStep]);
            AddPredictorIteration(m_stepsIteration[iStep]);
            AddPredictorUpperLimit(m_stepsUpperLimit[iStep]);
            AddPredictorLowerLimit(m_stepsLowerLimit[iStep]);
            AddPredictorLocks(m_stepsLocks[iStep]);
            SetPreprocess(iStep, iPtor, false);
            SetPreload(iStep, iPtor, false);
            if (!ParsePredictors(fileParams, iStep, iPtor, nodeParamBlock))
                return false;
            iPtor++;
        } else {
            fileParams.UnknownNode(nodeParamBlock);
        }
        nodeParamBlock = nodeParamBlock->GetNext();
    }
    return true;
}

bool asParametersOptimization::ParsePredictors(asFileParametersOptimization &fileParams, int iStep, int iPtor,
                                               const wxXmlNode *nodeParamBlock)
{
    wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
    while (nodeParam) {
        if (nodeParam->GetName() == "preload") {
            SetPreload(iStep, iPtor, fileParams.GetBool(nodeParam));
        } else if (nodeParam->GetName() == "preprocessing") {
            SetPreprocess(iStep, iPtor, true);
            if (!ParsePreprocessedPredictors(fileParams, iStep, iPtor, nodeParam))
                return false;
        } else if (nodeParam->GetName() == "dataset_id") {
            if (!SetPredictorDatasetId(iStep, iPtor, fileParams.GetString(nodeParam)))
                return false;
        } else if (nodeParam->GetName() == "data_id") {
            if (!SetPredictorDataIdVector(iStep, iPtor, fileParams.GetVectorString(nodeParam)))
                return false;
            SetPredictorDataIdLock(iStep, iPtor, fileParams.GetAttributeBool(nodeParam, "lock", true, false));
            if (IsPredictorDataIdLocked(iStep, iPtor)) {
                if (!SetPredictorDataId(iStep, iPtor, fileParams.GetString(nodeParam)))
                    return false;
            }
        } else if (nodeParam->GetName() == "level") {
            if (!SetPredictorLevelVector(iStep, iPtor, fileParams.GetVectorFloat(nodeParam)))
                return false;
            SetPredictorLevelLock(iStep, iPtor, fileParams.GetAttributeBool(nodeParam, "lock", true, false));
            if (IsPredictorLevelLocked(iStep, iPtor)) {
                if (!SetPredictorLevel(iStep, iPtor, fileParams.GetFloat(nodeParam)))
                    return false;
            }
        } else if (nodeParam->GetName() == "time") {
            SetPredictorTimeHoursLock(iStep, iPtor, fileParams.GetAttributeBool(nodeParam, "lock", true, false));
            if (IsPredictorTimeHoursLocked(iStep, iPtor)) {
                if (!SetPredictorTimeHours(iStep, iPtor, fileParams.GetDouble(nodeParam)))
                    return false;
                vd vTimeHours;
                vTimeHours.push_back(GetPredictorTimeHours(iStep, iPtor));
                if (!SetPreloadTimeHours(iStep, iPtor, vTimeHours))
                    return false;
                if (!SetPredictorTimeHoursLowerLimit(iStep, iPtor, GetPredictorTimeHours(iStep, iPtor)))
                    return false;
                if (!SetPredictorTimeHoursUpperLimit(iStep, iPtor, GetPredictorTimeHours(iStep, iPtor)))
                    return false;
                if (!SetPredictorTimeHoursIteration(iStep, iPtor, 6))
                    return false;
            } else {
                if (!SetPredictorTimeHoursLowerLimit(iStep, iPtor,
                                                     fileParams.GetAttributeDouble(nodeParam, "lowerlimit")))
                    return false;
                if (!SetPredictorTimeHoursUpperLimit(iStep, iPtor,
                                                     fileParams.GetAttributeDouble(nodeParam, "upperlimit")))
                    return false;
                if (!SetPredictorTimeHoursIteration(iStep, iPtor,
                                                    fileParams.GetAttributeDouble(nodeParam, "iteration")))
                    return false;
                // Initialize to ensure correct array sizes
                if (!SetPredictorTimeHours(iStep, iPtor, GetPredictorTimeHoursLowerLimit(iStep, iPtor)))
                    return false;
            }
        } else if (nodeParam->GetName() == "members") {
            if (!SetPredictorMembersNb(iStep, iPtor, fileParams.GetInt(nodeParam)))
                return false;
        } else if (nodeParam->GetName() == "spatial_window") {
            if (!ParseSpatialWindow(fileParams, iStep, iPtor, nodeParam))
                return false;
        } else if (nodeParam->GetName() == "criteria") {
            if (!SetPredictorCriteriaVector(iStep, iPtor, fileParams.GetVectorString(nodeParam)))
                return false;
            SetPredictorCriteriaLock(iStep, iPtor, fileParams.GetAttributeBool(nodeParam, "lock", true, false));
            if (IsPredictorCriteriaLocked(iStep, iPtor)) {
                if (!SetPredictorCriteria(iStep, iPtor, fileParams.GetString(nodeParam)))
                    return false;
            }
        } else if (nodeParam->GetName() == "weight") {
            SetPredictorWeightLock(iStep, iPtor, fileParams.GetAttributeBool(nodeParam, "lock", true, false));
            if (IsPredictorWeightLocked(iStep, iPtor)) {
                if (!SetPredictorWeight(iStep, iPtor, fileParams.GetFloat(nodeParam)))
                    return false;
                if (!SetPredictorWeightLowerLimit(iStep, iPtor, GetPredictorWeight(iStep, iPtor)))
                    return false;
                if (!SetPredictorWeightUpperLimit(iStep, iPtor, GetPredictorWeight(iStep, iPtor)))
                    return false;
                if (!SetPredictorWeightIteration(iStep, iPtor, 1))
                    return false;
            } else {
                if (!SetPredictorWeightLowerLimit(iStep, iPtor, fileParams.GetAttributeFloat(nodeParam, "lowerlimit")))
                    return false;
                if (!SetPredictorWeightUpperLimit(iStep, iPtor, fileParams.GetAttributeFloat(nodeParam, "upperlimit")))
                    return false;
                if (!SetPredictorWeightIteration(iStep, iPtor, fileParams.GetAttributeFloat(nodeParam, "iteration")))
                    return false;
            }
        } else {
            fileParams.UnknownNode(nodeParam);
        }
        nodeParam = nodeParam->GetNext();
    }
    return true;
}

bool asParametersOptimization::ParseSpatialWindow(asFileParametersOptimization &fileParams, int iStep, int iPtor,
                                                  const wxXmlNode *nodeParam)
{
    wxXmlNode *nodeWindow = nodeParam->GetChildren();
    while (nodeWindow) {
        if (nodeWindow->GetName() == "grid_type") {
            if (!SetPredictorGridType(iStep, iPtor, fileParams.GetString(nodeWindow, "regular")))
                return false;
        } else if (nodeWindow->GetName() == "x_min") {
            SetPredictorXminLock(iStep, iPtor, fileParams.GetAttributeBool(nodeWindow, "lock"));
            if (IsPredictorXminLocked(iStep, iPtor)) {
                if (!SetPredictorXmin(iStep, iPtor, fileParams.GetDouble(nodeWindow)))
                    return false;
                if (!SetPredictorXminLowerLimit(iStep, iPtor, GetPredictorXmin(iStep, iPtor)))
                    return false;
                if (!SetPredictorXminUpperLimit(iStep, iPtor, GetPredictorXmin(iStep, iPtor)))
                    return false;
                if (!SetPredictorXminIteration(iStep, iPtor, 1))
                    return false;
            } else {
                if (!SetPredictorXminLowerLimit(iStep, iPtor, fileParams.GetAttributeDouble(nodeWindow, "lowerlimit")))
                    return false;
                if (!SetPredictorXminUpperLimit(iStep, iPtor, fileParams.GetAttributeDouble(nodeWindow, "upperlimit")))
                    return false;
                if (!SetPredictorXminIteration(iStep, iPtor, fileParams.GetAttributeDouble(nodeWindow, "iteration")))
                    return false;
            }
        } else if (nodeWindow->GetName() == "x_points_nb") {
            SetPredictorXptsnbLock(iStep, iPtor, fileParams.GetAttributeBool(nodeWindow, "lock"));
            if (IsPredictorXptsnbLocked(iStep, iPtor)) {
                if (!SetPredictorXptsnb(iStep, iPtor, fileParams.GetDouble(nodeWindow)))
                    return false;
                if (!SetPredictorXptsnbLowerLimit(iStep, iPtor, GetPredictorXptsnb(iStep, iPtor)))
                    return false;
                if (!SetPredictorXptsnbUpperLimit(iStep, iPtor, GetPredictorXptsnb(iStep, iPtor)))
                    return false;
                if (!SetPredictorXptsnbIteration(iStep, iPtor, 1))
                    return false;
            } else {
                if (!SetPredictorXptsnbLowerLimit(iStep, iPtor,
                                                  fileParams.GetAttributeDouble(nodeWindow, "lowerlimit")))
                    return false;
                if (!SetPredictorXptsnbUpperLimit(iStep, iPtor,
                                                  fileParams.GetAttributeDouble(nodeWindow, "upperlimit")))
                    return false;
                if (!SetPredictorXptsnbIteration(iStep, iPtor, fileParams.GetAttributeDouble(nodeWindow, "iteration")))
                    return false;
            }
        } else if (nodeWindow->GetName() == "x_step") {
            if (!SetPredictorXstep(iStep, iPtor, fileParams.GetDouble(nodeWindow)))
                return false;
        } else if (nodeWindow->GetName() == "y_min") {
            SetPredictorYminLock(iStep, iPtor, fileParams.GetAttributeBool(nodeWindow, "lock"));
            if (IsPredictorYminLocked(iStep, iPtor)) {
                if (!SetPredictorYmin(iStep, iPtor, fileParams.GetDouble(nodeWindow)))
                    return false;
                if (!SetPredictorYminLowerLimit(iStep, iPtor, GetPredictorYmin(iStep, iPtor)))
                    return false;
                if (!SetPredictorYminUpperLimit(iStep, iPtor, GetPredictorYmin(iStep, iPtor)))
                    return false;
                if (!SetPredictorYminIteration(iStep, iPtor, 1))
                    return false;
            } else {
                if (!SetPredictorYminLowerLimit(iStep, iPtor, fileParams.GetAttributeDouble(nodeWindow, "lowerlimit")))
                    return false;
                if (!SetPredictorYminUpperLimit(iStep, iPtor, fileParams.GetAttributeDouble(nodeWindow, "upperlimit")))
                    return false;
                if (!SetPredictorYminIteration(iStep, iPtor, fileParams.GetAttributeDouble(nodeWindow, "iteration")))
                    return false;
            }
        } else if (nodeWindow->GetName() == "y_points_nb") {
            SetPredictorYptsnbLock(iStep, iPtor, fileParams.GetAttributeBool(nodeWindow, "lock"));
            if (IsPredictorYptsnbLocked(iStep, iPtor)) {
                if (!SetPredictorYptsnb(iStep, iPtor, fileParams.GetDouble(nodeWindow)))
                    return false;
                if (!SetPredictorYptsnbLowerLimit(iStep, iPtor, GetPredictorYptsnb(iStep, iPtor)))
                    return false;
                if (!SetPredictorYptsnbUpperLimit(iStep, iPtor, GetPredictorYptsnb(iStep, iPtor)))
                    return false;
                if (!SetPredictorYptsnbIteration(iStep, iPtor, 1))
                    return false;
            } else {
                if (!SetPredictorYptsnbLowerLimit(iStep, iPtor,
                                                  fileParams.GetAttributeDouble(nodeWindow, "lowerlimit")))
                    return false;
                if (!SetPredictorYptsnbUpperLimit(iStep, iPtor,
                                                  fileParams.GetAttributeDouble(nodeWindow, "upperlimit")))
                    return false;
                if (!SetPredictorYptsnbIteration(iStep, iPtor, fileParams.GetAttributeDouble(nodeWindow, "iteration")))
                    return false;
            }
        } else if (nodeWindow->GetName() == "y_step") {
            if (!SetPredictorYstep(iStep, iPtor, fileParams.GetDouble(nodeWindow)))
                return false;
        } else {
            fileParams.UnknownNode(nodeWindow);
        }
        nodeWindow = nodeWindow->GetNext();
    }
    return true;
}

bool asParametersOptimization::ParsePreprocessedPredictors(asFileParametersOptimization &fileParams, int iStep,
                                                           int iPtor, const wxXmlNode *nodeParam)
{
    int iPre = 0;
    wxXmlNode *nodePreprocess = nodeParam->GetChildren();
    while (nodePreprocess) {
        if (nodePreprocess->GetName() == "preprocessing_method") {
            if (!SetPreprocessMethod(iStep, iPtor, fileParams.GetString(nodePreprocess)))
                return false;
        } else if (nodePreprocess->GetName() == "preprocessing_data") {
            if (!ParsePreprocessedPredictorDataset(fileParams, iStep, iPtor, iPre, nodePreprocess))
                return false;
            iPre++;
        } else {
            fileParams.UnknownNode(nodePreprocess);
        }
        nodePreprocess = nodePreprocess->GetNext();
    }
    return true;
}

bool asParametersOptimization::ParsePreprocessedPredictorDataset(asFileParametersOptimization &fileParams, int iStep,
                                                                 int iPtor, int iPre, const wxXmlNode *nodePreprocess)
{
    wxXmlNode *nodeParamPreprocess = nodePreprocess->GetChildren();
    while (nodeParamPreprocess) {
        if (nodeParamPreprocess->GetName() == "dataset_id") {
            if (!SetPreprocessDatasetId(iStep, iPtor, iPre, fileParams.GetString(nodeParamPreprocess)))
                return false;
        } else if (nodeParamPreprocess->GetName() == "data_id") {
            if (!SetPreprocessDataIdVector(iStep, iPtor, iPre, fileParams.GetVectorString(nodeParamPreprocess)))
                return false;
            SetPreprocessDataIdLock(iStep, iPtor, iPre,
                                    fileParams.GetAttributeBool(nodeParamPreprocess, "lock", true, false));
            if (IsPreprocessDataIdLocked(iStep, iPtor, iPre)) {
                if (!SetPreprocessDataId(iStep, iPtor, iPre, fileParams.GetString(nodeParamPreprocess)))
                    return false;
            } else {
                // Initialize to ensure correct array sizes
                if (!SetPreprocessDataId(iStep, iPtor, iPre, GetPreprocessDataIdVector(iStep, iPtor, iPre)[0]))
                    return false;
            }
        } else if (nodeParamPreprocess->GetName() == "level") {
            if (!SetPreprocessLevelVector(iStep, iPtor, iPre, fileParams.GetVectorFloat(nodeParamPreprocess)))
                return false;
            SetPreprocessLevelLock(iStep, iPtor, iPre,
                                   fileParams.GetAttributeBool(nodeParamPreprocess, "lock", true, false));
            if (IsPreprocessLevelLocked(iStep, iPtor, iPre)) {
                if (!SetPreprocessLevel(iStep, iPtor, iPre, fileParams.GetFloat(nodeParamPreprocess)))
                    return false;
            } else {
                // Initialize to ensure correct array sizes
                if (!SetPreprocessLevel(iStep, iPtor, iPre, GetPreprocessLevelVector(iStep, iPtor, iPre)[0]))
                    return false;
            }
        } else if (nodeParamPreprocess->GetName() == "time") {
            SetPreprocessTimeHoursLock(iStep, iPtor, iPre,
                                       fileParams.GetAttributeBool(nodeParamPreprocess, "lock", true, false));
            if (IsPreprocessTimeHoursLocked(iStep, iPtor, iPre)) {
                if (!SetPreprocessTimeHours(iStep, iPtor, iPre, fileParams.GetDouble(nodeParamPreprocess)))
                    return false;
                if (!SetPreprocessTimeHoursLowerLimit(iStep, iPtor, iPre, GetPreprocessTimeHours(iStep, iPtor, iPre)))
                    return false;
                if (!SetPreprocessTimeHoursUpperLimit(iStep, iPtor, iPre, GetPreprocessTimeHours(iStep, iPtor, iPre)))
                    return false;
                if (!SetPreprocessTimeHoursIteration(iStep, iPtor, iPre, 6))
                    return false;
            } else {
                if (!SetPreprocessTimeHoursLowerLimit(iStep, iPtor, iPre,
                                                      fileParams.GetAttributeDouble(nodeParamPreprocess, "lowerlimit")))
                    return false;
                if (!SetPreprocessTimeHoursUpperLimit(iStep, iPtor, iPre,
                                                      fileParams.GetAttributeDouble(nodeParamPreprocess, "upperlimit")))
                    return false;
                if (!SetPreprocessTimeHoursIteration(iStep, iPtor, iPre,
                                                     fileParams.GetAttributeDouble(nodeParamPreprocess, "iteration")))
                    return false;
                // Initialize to ensure correct array sizes
                if (!SetPreprocessTimeHours(iStep, iPtor, iPre, GetPreprocessTimeHoursLowerLimit(iStep, iPtor, iPre)))
                    return false;
            }
        } else if (nodeParamPreprocess->GetName() == "members") {
            if (!SetPreprocessMembersNb(iStep, iPtor, iPre, fileParams.GetInt(nodeParamPreprocess)))
                return false;
        } else {
            fileParams.UnknownNode(nodeParamPreprocess);
        }
        nodeParamPreprocess = nodeParamPreprocess->GetNext();
    }
    return true;
}

bool asParametersOptimization::ParseAnalogValuesParams(asFileParametersOptimization &fileParams,
                                                       const wxXmlNode *nodeProcess)
{
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "predictand") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "station_id") {
                    if (!SetPredictandStationIds(fileParams.GetStationIds(fileParams.GetString(nodeParam))))
                        return false;
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

bool asParametersOptimization::ParseScore(asFileParametersOptimization &fileParams, const wxXmlNode *nodeProcess)
{
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "score") {
            if (!SetScoreName(fileParams.GetString(nodeParamBlock)))
                return false;
        } else if (nodeParamBlock->GetName() == "threshold") {
            SetScoreThreshold(fileParams.GetFloat(nodeParamBlock));
        } else if (nodeParamBlock->GetName() == "quantile") {
            SetScoreQuantile(fileParams.GetFloat(nodeParamBlock));
        } else if (nodeParamBlock->GetName() == "time_array") {
            if (!SetScoreTimeArrayMode(fileParams.GetString(nodeParamBlock)))
                return false;
        } else if (nodeParamBlock->GetName() == "postprocessing") {
            wxLogError(_("The postptocessing is not yet fully implemented."));
        } else {
            fileParams.UnknownNode(nodeParamBlock);
        }
        nodeParamBlock = nodeParamBlock->GetNext();
    }
    return true;
}

bool asParametersOptimization::SetSpatialWindowProperties()
{
    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        for (int iPtor = 0; iPtor < GetPredictorsNb(iStep); iPtor++) {
            double xShift = std::fmod(GetPredictorXminLowerLimit(iStep, iPtor), GetPredictorXstep(iStep, iPtor));
            if (xShift < 0)
                xShift += GetPredictorXstep(iStep, iPtor);
            if (!SetPredictorXshift(iStep, iPtor, xShift))
                return false;

            double yShift = std::fmod(GetPredictorYminLowerLimit(iStep, iPtor), GetPredictorYstep(iStep, iPtor));
            if (yShift < 0)
                yShift += GetPredictorYstep(iStep, iPtor);
            if (!SetPredictorYshift(iStep, iPtor, yShift))
                return false;

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

bool asParametersOptimization::SetPreloadingProperties()
{
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
                    int xBasePtsNb = std::abs(
                            GetPredictorXminUpperLimit(iStep, iPtor) - GetPredictorXminLowerLimit(iStep, iPtor)) /
                                     GetPredictorXstep(iStep, iPtor);
                    SetPreloadXptsnb(iStep, iPtor,
                                     xBasePtsNb + GetPredictorXptsnbUpperLimit(iStep, iPtor)); // No need to add +1
                } else {
                    SetPreloadXptsnb(iStep, iPtor, GetPredictorXptsnb(iStep, iPtor));
                }

                if (!IsPredictorYptsnbLocked(iStep, iPtor)) {
                    int yBasePtsNb = std::abs(
                            GetPredictorYminUpperLimit(iStep, iPtor) - GetPredictorYminLowerLimit(iStep, iPtor)) /
                                     GetPredictorYstep(iStep, iPtor);
                    SetPreloadYptsnb(iStep, iPtor,
                                     yBasePtsNb + GetPredictorYptsnbUpperLimit(iStep, iPtor)); // No need to add +1
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
                    SetPredictorTimeHours(iStep, iPtor, GetPreprocessTimeHours(iStep, iPtor, 0));
                } else {
                    SetPredictorDatasetId(iStep, iPtor, "mix");
                    SetPredictorDataId(iStep, iPtor, "mix");
                    SetPredictorLevel(iStep, iPtor, 0);
                    SetPredictorTimeHours(iStep, iPtor, 0);
                }
            }

            // Set levels and time for preloading
            if (NeedsPreloading(iStep, iPtor) && !NeedsPreprocessing(iStep, iPtor)) {
                if (!SetPreloadDataIds(iStep, iPtor, GetPredictorDataIdVector(iStep, iPtor)))
                    return false;
                if (!SetPreloadLevels(iStep, iPtor, GetPredictorLevelVector(iStep, iPtor)))
                    return false;
                vd vTimeHours;
                for (double h = GetPredictorTimeHoursLowerLimit(iStep, iPtor);
                     h <= GetPredictorTimeHoursUpperLimit(iStep, iPtor); h += GetPredictorTimeHoursIteration(iStep,
                                                                                                             iPtor)) {
                    vTimeHours.push_back(h);
                }
                if (!SetPreloadTimeHours(iStep, iPtor, vTimeHours))
                    return false;
            } else if (NeedsPreloading(iStep, iPtor) && NeedsPreprocessing(iStep, iPtor)) {
                // Check the preprocessing method
                wxString method = GetPreprocessMethod(iStep, iPtor);
                vf preprocLevels;
                vd preprocTimeHours;

                // Different actions depending on the preprocessing method.
                if (method.IsSameAs("Gradients")) {
                    preprocLevels = GetPreprocessLevelVector(iStep, iPtor, 0);

                    for (double h = GetPreprocessTimeHoursLowerLimit(iStep, iPtor, 0);
                         h <= GetPreprocessTimeHoursUpperLimit(iStep, iPtor, 0); h += GetPreprocessTimeHoursIteration(
                            iStep, iPtor, 0)) {
                        preprocTimeHours.push_back(h);
                    }
                } else if (method.IsSameAs("HumidityFlux")) {
                    preprocLevels = GetPreprocessLevelVector(iStep, iPtor, 0);

                    for (double h = GetPreprocessTimeHoursLowerLimit(iStep, iPtor, 0);
                         h <= GetPreprocessTimeHoursUpperLimit(iStep, iPtor, 0); h += GetPreprocessTimeHoursIteration(
                            iStep, iPtor, 0)) {
                        preprocTimeHours.push_back(h);
                    }
                } else if (method.IsSameAs("Multiplication") || method.IsSameAs("Multiply") ||
                           method.IsSameAs("HumidityIndex")) {
                    preprocLevels = GetPreprocessLevelVector(iStep, iPtor, 0);

                    for (double h = GetPreprocessTimeHoursLowerLimit(iStep, iPtor, 0);
                         h <= GetPreprocessTimeHoursUpperLimit(iStep, iPtor, 0); h += GetPreprocessTimeHoursIteration(
                            iStep, iPtor, 0)) {
                        preprocTimeHours.push_back(h);
                    }
                } else if (method.IsSameAs("FormerHumidityIndex")) {
                    wxLogWarning(_("The %s preprocessing method is not handled in the optimizer."), method);
                    return false;
                } else {
                    wxLogWarning(_("The %s preprocessing method is not yet handled with the preload option."), method);
                }

                if (!SetPreloadLevels(iStep, iPtor, preprocLevels))
                    return false;
                if (!SetPreloadTimeHours(iStep, iPtor, preprocTimeHours))
                    return false;
            }
        }
    }

    return true;
}

void asParametersOptimization::InitRandomValues()
{
    if (!m_timeArrayAnalogsIntervalDaysLocks) {
        m_timeArrayAnalogsIntervalDays = asRandom(m_timeArrayAnalogsIntervalDaysLowerLimit,
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
                        wxASSERT(m_stepsVect[i].predictors[j].preprocessDataId[k].size() > (unsigned) row);

                        SetPreprocessDataId(i, j, k, m_stepsVect[i].predictors[j].preprocessDataId[k][row]);
                    }

                    if (!m_stepsLocks[i].predictors[j].preprocessLevels[k]) {
                        int length = m_stepsVect[i].predictors[j].preprocessLevels[k].size();
                        int row = asRandom(0, length - 1);
                        wxASSERT(m_stepsVect[i].predictors[j].preprocessLevels[k].size() > (unsigned) row);

                        SetPreprocessLevel(i, j, k, m_stepsVect[i].predictors[j].preprocessLevels[k][row]);
                    }

                    if (!m_stepsLocks[i].predictors[j].preprocessTimeHours[k]) {
                        SetPreprocessTimeHours(i, j, k, asRandom(
                                m_stepsLowerLimit[i].predictors[j].preprocessTimeHours[k],
                                m_stepsUpperLimit[i].predictors[j].preprocessTimeHours[k],
                                m_stepsIteration[i].predictors[j].preprocessTimeHours[k]));
                    }
                }
            } else {
                if (!m_stepsLocks[i].predictors[j].dataId) {
                    int length = m_stepsVect[i].predictors[j].dataId.size();
                    int row = asRandom(0, length - 1);
                    wxASSERT(m_stepsVect[i].predictors[j].dataId.size() > (unsigned) row);

                    SetPredictorDataId(i, j, m_stepsVect[i].predictors[j].dataId[row]);
                }

                if (!m_stepsLocks[i].predictors[j].level) {
                    int length = m_stepsVect[i].predictors[j].level.size();
                    int row = asRandom(0, length - 1);
                    wxASSERT(m_stepsVect[i].predictors[j].level.size() > (unsigned) row);

                    SetPredictorLevel(i, j, m_stepsVect[i].predictors[j].level[row]);
                }

                if (!m_stepsLocks[i].predictors[j].timeHours) {
                    SetPredictorTimeHours(i, j, asRandom(m_stepsLowerLimit[i].predictors[j].timeHours,
                                                                m_stepsUpperLimit[i].predictors[j].timeHours,
                                                                m_stepsIteration[i].predictors[j].timeHours));
                }

            }

            if (!m_stepsLocks[i].predictors[j].xMin) {
                SetPredictorXmin(i, j, asRandom(m_stepsLowerLimit[i].predictors[j].xMin,
                                                       m_stepsUpperLimit[i].predictors[j].xMin,
                                                       m_stepsIteration[i].predictors[j].xMin));
            }

            if (!m_stepsLocks[i].predictors[j].xPtsNb) {
                SetPredictorXptsnb(i, j, asRandom(m_stepsLowerLimit[i].predictors[j].xPtsNb,
                                                         m_stepsUpperLimit[i].predictors[j].xPtsNb,
                                                         m_stepsIteration[i].predictors[j].xPtsNb));
            }

            if (!m_stepsLocks[i].predictors[j].yMin) {
                SetPredictorYmin(i, j, asRandom(m_stepsLowerLimit[i].predictors[j].yMin,
                                                       m_stepsUpperLimit[i].predictors[j].yMin,
                                                       m_stepsIteration[i].predictors[j].yMin));
            }

            if (!m_stepsLocks[i].predictors[j].yPtsNb) {
                SetPredictorYptsnb(i, j, asRandom(m_stepsLowerLimit[i].predictors[j].yPtsNb,
                                                         m_stepsUpperLimit[i].predictors[j].yPtsNb,
                                                         m_stepsIteration[i].predictors[j].yPtsNb));
            }

            if (!m_stepsLocks[i].predictors[j].weight) {
                SetPredictorWeight(i, j, asRandom(m_stepsLowerLimit[i].predictors[j].weight,
                                                         m_stepsUpperLimit[i].predictors[j].weight,
                                                         m_stepsIteration[i].predictors[j].weight));
            }

            if (!m_stepsLocks[i].predictors[j].criteria) {
                int length = m_stepsVect[i].predictors[j].criteria.size();
                int row = asRandom(0, length - 1);
                wxASSERT(m_stepsVect[i].predictors[j].criteria.size() > (unsigned) row);

                SetPredictorCriteria(i, j, m_stepsVect[i].predictors[j].criteria[row]);
            }

            // Fix the criteria if S1
            if (NeedsPreprocessing(i, j) && GetPreprocessMethod(i, j).IsSameAs("Gradients") &&
                GetPredictorCriteria(i, j).IsSameAs("S1")) {
                SetPredictorCriteria(i, j, "S1grads");
            }

        }
    }

    FixWeights();
    FixCoordinates();
    CheckRange();
    FixAnalogsNb();
}

void asParametersOptimization::CheckRange()
{
    // Check that the actual parameters values are within ranges
    if (!m_timeArrayAnalogsIntervalDaysLocks) {
        m_timeArrayAnalogsIntervalDays = wxMax(
                wxMin(m_timeArrayAnalogsIntervalDays, m_timeArrayAnalogsIntervalDaysUpperLimit),
                m_timeArrayAnalogsIntervalDaysLowerLimit);
    }
    wxASSERT(m_timeArrayAnalogsIntervalDays > 0);

    for (int i = 0; i < GetStepsNb(); i++) {
        if (!m_stepsLocks[i].analogsNumber) {
            SetAnalogsNumber(i, wxMax(wxMin(GetAnalogsNumber(i), m_stepsUpperLimit[i].analogsNumber),
                                      m_stepsLowerLimit[i].analogsNumber));
        }

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (!GetPredictorGridType(i, j).IsSameAs("Regular", false))
                asThrowException(wxString::Format(
                        _("asParametersOptimization::CheckRange is not ready to use on unregular grids (PredictorGridType = %s)"),
                        GetPredictorGridType(i, j).c_str()));

            if (NeedsPreprocessing(i, j)) {
                int preprocessSize = GetPreprocessSize(i, j);
                for (int k = 0; k < preprocessSize; k++) {
                    if (!m_stepsLocks[i].predictors[j].preprocessTimeHours[k]) {
                        SetPreprocessTimeHours(i, j, k, wxMax(wxMin(GetPreprocessTimeHours(i, j, k),
                                                                    m_stepsUpperLimit[i].predictors[j].preprocessTimeHours[k]),
                                                              m_stepsLowerLimit[i].predictors[j].preprocessTimeHours[k]));
                    }
                    SetPredictorTimeHours(i, j, 0);
                }
            } else {
                if (!m_stepsLocks[i].predictors[j].timeHours) {
                    SetPredictorTimeHours(i, j, wxMax(wxMin(GetPredictorTimeHours(i, j),
                                                            m_stepsUpperLimit[i].predictors[j].timeHours),
                                                      m_stepsLowerLimit[i].predictors[j].timeHours));
                }
            }

            // Check ranges
            if (!m_stepsLocks[i].predictors[j].xMin) {
                SetPredictorXmin(i, j, wxMax(wxMin(GetPredictorXmin(i, j), m_stepsUpperLimit[i].predictors[j].xMin),
                                             m_stepsLowerLimit[i].predictors[j].xMin));
            }
            if (!m_stepsLocks[i].predictors[j].xPtsNb) {
                SetPredictorXptsnb(i, j,
                                   wxMax(wxMin(GetPredictorXptsnb(i, j), m_stepsUpperLimit[i].predictors[j].xPtsNb),
                                         m_stepsLowerLimit[i].predictors[j].xPtsNb));
            }

            if (!m_stepsLocks[i].predictors[j].yMin) {
                SetPredictorYmin(i, j, wxMax(wxMin(GetPredictorYmin(i, j), m_stepsUpperLimit[i].predictors[j].yMin),
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
                        SetPredictorXptsnb(i, j, (m_stepsUpperLimit[i].predictors[j].xMin - GetPredictorXmin(i, j)) /
                                                 GetPredictorXstep(i, j) +
                                                 m_stepsUpperLimit[i].predictors[j].xPtsNb); // Correct, no need of +1
                    } else {
                        SetPredictorXmin(i, j, m_stepsUpperLimit[i].predictors[j].xMin -
                                               GetPredictorXptsnb(i, j) * GetPredictorXstep(i, j));
                    }
                }
            }

            if (!m_stepsLocks[i].predictors[j].yMin || !m_stepsLocks[i].predictors[j].yPtsNb) {
                if (GetPredictorYmin(i, j) + (GetPredictorYptsnb(i, j) - 1) * GetPredictorYstep(i, j) >
                    m_stepsUpperLimit[i].predictors[j].yMin +
                    (m_stepsUpperLimit[i].predictors[j].yPtsNb - 1) * GetPredictorYstep(i, j)) {
                    if (!m_stepsLocks[i].predictors[j].yPtsNb) {
                        SetPredictorYptsnb(i, j, (m_stepsUpperLimit[i].predictors[j].yMin - GetPredictorYmin(i, j)) /
                                                 GetPredictorYstep(i, j) +
                                                 m_stepsUpperLimit[i].predictors[j].yPtsNb); // Correct, no need of +1
                    } else {
                        SetPredictorYmin(i, j, m_stepsUpperLimit[i].predictors[j].yMin -
                                               GetPredictorYptsnb(i, j) * GetPredictorYstep(i, j));
                    }
                }
            }
        }
    }

    FixTimeHours();
    FixWeights();
    FixCoordinates();
    FixAnalogsNb();
}

bool asParametersOptimization::IsInRange()
{
    // Check that the actual parameters values are within ranges
    if (!m_timeArrayAnalogsIntervalDaysLocks) {
        if (m_timeArrayAnalogsIntervalDays > m_timeArrayAnalogsIntervalDaysUpperLimit)
            return false;
        if (m_timeArrayAnalogsIntervalDays < m_timeArrayAnalogsIntervalDaysLowerLimit)
            return false;
    }

    for (int i = 0; i < GetStepsNb(); i++) {
        if (!m_stepsLocks[i].analogsNumber) {
            if (GetAnalogsNumber(i) > m_stepsUpperLimit[i].analogsNumber)
                return false;
            if (GetAnalogsNumber(i) < m_stepsLowerLimit[i].analogsNumber)
                return false;
        }

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (!m_stepsLocks[i].predictors[j].preprocessTimeHours[k]) {
                        if (GetPreprocessTimeHours(i, j, k) < m_stepsLowerLimit[i].predictors[j].preprocessTimeHours[k])
                            return false;
                        if (GetPreprocessTimeHours(i, j, k) < m_stepsLowerLimit[i].predictors[j].preprocessTimeHours[k])
                            return false;
                    }
                }
            } else {
                if (!m_stepsLocks[i].predictors[j].timeHours) {
                    if (GetPredictorTimeHours(i, j) < m_stepsLowerLimit[i].predictors[j].timeHours)
                        return false;
                    if (GetPredictorTimeHours(i, j) < m_stepsLowerLimit[i].predictors[j].timeHours)
                        return false;
                }
            }

            if (!GetPredictorGridType(i, j).IsSameAs("Regular", false))
                asThrowException(wxString::Format(
                        _("asParametersOptimization::CheckRange is not ready to use on unregular grids (PredictorGridType = %s)"),
                        GetPredictorGridType(i, j).c_str()));

            // Check ranges
            if (!m_stepsLocks[i].predictors[j].xMin) {
                if (GetPredictorXmin(i, j) > m_stepsUpperLimit[i].predictors[j].xMin)
                    return false;
                if (GetPredictorXmin(i, j) < m_stepsLowerLimit[i].predictors[j].xMin)
                    return false;
            }
            if (!m_stepsLocks[i].predictors[j].xPtsNb) {
                if (GetPredictorXptsnb(i, j) < m_stepsLowerLimit[i].predictors[j].xPtsNb)
                    return false;
                if (GetPredictorXptsnb(i, j) < m_stepsLowerLimit[i].predictors[j].xPtsNb)
                    return false;
            }
            if (!m_stepsLocks[i].predictors[j].yMin) {
                if (GetPredictorYmin(i, j) < m_stepsLowerLimit[i].predictors[j].yMin)
                    return false;
                if (GetPredictorYmin(i, j) < m_stepsLowerLimit[i].predictors[j].yMin)
                    return false;
            }
            if (!m_stepsLocks[i].predictors[j].yPtsNb) {
                if (GetPredictorYptsnb(i, j) < m_stepsLowerLimit[i].predictors[j].yPtsNb)
                    return false;
                if (GetPredictorYptsnb(i, j) < m_stepsLowerLimit[i].predictors[j].yPtsNb)
                    return false;
            }
            if (!m_stepsLocks[i].predictors[j].weight) {
                if (GetPredictorWeight(i, j) < m_stepsLowerLimit[i].predictors[j].weight)
                    return false;
                if (GetPredictorWeight(i, j) < m_stepsLowerLimit[i].predictors[j].weight)
                    return false;
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

bool asParametersOptimization::FixTimeLimits()
{
    double minHour = 200.0, maxHour = -50.0;
    for (int i = 0; i < GetStepsNb(); i++) {
        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    minHour = wxMin(GetPreprocessTimeHoursLowerLimit(i, j, k), minHour);
                    maxHour = wxMax(GetPreprocessTimeHoursUpperLimit(i, j, k), maxHour);
                }
            } else {
                minHour = wxMin(GetPredictorTimeHoursLowerLimit(i, j), minHour);
                maxHour = wxMax(GetPredictorTimeHoursUpperLimit(i, j), maxHour);
            }
        }
    }

    m_timeMinHours = minHour;
    m_timeMaxHours = maxHour;

    return true;
}

void asParametersOptimization::FixTimeHours()
{
    for (int i = 0; i < GetStepsNb(); i++) {
        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (!m_stepsLocks[i].predictors[j].preprocessTimeHours[k]) {
                        if (m_stepsIteration[i].predictors[j].preprocessTimeHours[k] != 0) {
                            float ratio = (float) GetPreprocessTimeHours(i, j, k) /
                                          (float) m_stepsIteration[i].predictors[j].preprocessTimeHours[k];
                            ratio = asRound(ratio);
                            SetPreprocessTimeHours(i, j, k,
                                                   ratio * m_stepsIteration[i].predictors[j].preprocessTimeHours[k]);
                        }
                    }
                }
            } else {
                if (!m_stepsLocks[i].predictors[j].timeHours) {
                    if (m_stepsIteration[i].predictors[j].timeHours != 0) {
                        float ratio = (float) GetPredictorTimeHours(i, j) /
                                      (float) m_stepsIteration[i].predictors[j].timeHours;
                        ratio = asRound(ratio);
                        SetPredictorTimeHours(i, j, ratio * m_stepsIteration[i].predictors[j].timeHours);
                    }
                }
            }
        }
    }
}

bool asParametersOptimization::FixWeights()
{
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
            wxLogWarning(_("The sum of the locked weights of the analogy level number %d is higher than 1 (%f). They were forced as unlocked."),
                         i + 1, totWeightLocked);
            totWeightManageable = totWeight;
            totWeightLocked = 0;
        }

        // For every weights but the last
        float newSum = 0;
        for (int j = 0; j < GetPredictorsNb(i) - 1; j++) {
            if (totWeightLocked > 1) {
                float precision = GetPredictorWeightIteration(i, j);
                float newWeight = GetPredictorWeight(i, j) / totWeightManageable;
                newWeight = precision * asRound(newWeight * (1.0 / precision));
                newSum += newWeight;

                SetPredictorWeight(i, j, newWeight);
            } else {
                if (!IsPredictorWeightLocked(i, j)) {
                    float precision = GetPredictorWeightIteration(i, j);
                    float newWeight = GetPredictorWeight(i, j) / totWeightManageable;
                    newWeight = precision * asRound(newWeight * (1.0 / precision));
                    newSum += newWeight;

                    SetPredictorWeight(i, j, newWeight);
                }
            }
        }

        // Last weight: difference to 0
        float lastWeight = 1.0f - newSum - totWeightLocked;
        SetPredictorWeight(i, GetPredictorsNb(i) - 1, lastWeight);
    }

    return true;
}

void asParametersOptimization::LockAll()
{
    m_timeArrayAnalogsIntervalDaysLocks = true;

    for (int i = 0; i < GetStepsNb(); i++) {
        m_stepsLocks[i].analogsNumber = true;

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    m_stepsLocks[i].predictors[j].preprocessDataId[k] = true;
                    m_stepsLocks[i].predictors[j].preprocessLevels[k] = true;
                    m_stepsLocks[i].predictors[j].preprocessTimeHours[k] = true;
                }
            } else {
                m_stepsLocks[i].predictors[j].dataId = true;
                m_stepsLocks[i].predictors[j].level = true;
                m_stepsLocks[i].predictors[j].timeHours = true;
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

void asParametersOptimization::Unlock(vi &indices)
{
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
                        m_stepsLocks[i].predictors[j].preprocessTimeHours[k] = false;
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
                    m_stepsLocks[i].predictors[j].timeHours = false;
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

int asParametersOptimization::GetVariablesNb()
{
    int counter = 0;

    if (!m_timeArrayAnalogsIntervalDaysLocks)
        counter++;

    for (int i = 0; i < GetStepsNb(); i++) {
        if (!m_stepsLocks[i].analogsNumber)
            counter++;

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (!m_stepsLocks[i].predictors[j].preprocessDataId[k])
                        counter++;
                    if (!m_stepsLocks[i].predictors[j].preprocessLevels[k])
                        counter++;
                    if (!m_stepsLocks[i].predictors[j].preprocessTimeHours[k])
                        counter++;
                }
            } else {
                if (!m_stepsLocks[i].predictors[j].dataId)
                    counter++;
                if (!m_stepsLocks[i].predictors[j].level)
                    counter++;
                if (!m_stepsLocks[i].predictors[j].timeHours)
                    counter++;
            }

            if (!m_stepsLocks[i].predictors[j].xMin)
                counter++;
            if (!m_stepsLocks[i].predictors[j].xPtsNb)
                counter++;
            if (!m_stepsLocks[i].predictors[j].yMin)
                counter++;
            if (!m_stepsLocks[i].predictors[j].yPtsNb)
                counter++;
            if (!m_stepsLocks[i].predictors[j].weight)
                counter++;
            if (!m_stepsLocks[i].predictors[j].criteria)
                counter++;
        }
    }

    return counter;
}
