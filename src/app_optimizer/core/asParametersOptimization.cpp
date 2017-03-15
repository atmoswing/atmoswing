#include "asParametersOptimization.h"

#include <asFileParametersOptimization.h>


asParametersOptimization::asParametersOptimization()
        : asParametersScoring()
{
    m_timeArrayAnalogsIntervalDaysIteration = 1;
    m_timeArrayAnalogsIntervalDaysUpperLimit = 182;
    m_timeArrayAnalogsIntervalDaysLowerLimit = 10;
    m_timeArrayAnalogsIntervalDaysLocks = false;
    m_variableParamsNb = 0;
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

    int i_step = 0;
    wxXmlNode *nodeProcess = fileParams.GetRoot()->GetChildren();
    while (nodeProcess) {

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

        } else if (nodeProcess->GetName() == "analog_forecast_score") {
            if(!ParseForecastScore(fileParams, nodeProcess))
                return false;

        } else if (nodeProcess->GetName() == "analog_forecast_score_final") {
            if(!ParseForecastScoreFinal(fileParams, nodeProcess))
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
            while (nodeParam) {
                if (nodeParam->GetName() == "years") {
                    if (!SetValidationYearsVector(fileParams.GetVectorInt(nodeParam)))
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

bool asParametersOptimization::ParseAnalogDatesParams(asFileParametersOptimization &fileParams, int i_step,
                                                      const wxXmlNode *nodeProcess)
{
    int i_ptor = 0;
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "analogs_number") {
            SetAnalogsNumberLock(i_step, fileParams.GetAttributeBool(nodeParamBlock, "lock"));
            if (IsAnalogsNumberLocked(i_step)) {
                if (!SetAnalogsNumber(i_step, fileParams.GetInt(nodeParamBlock)))
                    return false;
                if (!SetAnalogsNumberLowerLimit(i_step, GetAnalogsNumber(i_step)))
                    return false;
                if (!SetAnalogsNumberUpperLimit(i_step, GetAnalogsNumber(i_step)))
                    return false;
                if (!SetAnalogsNumberIteration(i_step, 1))
                    return false;
            } else {
                if (!SetAnalogsNumberLowerLimit(i_step, fileParams.GetAttributeInt(nodeParamBlock, "lowerlimit")))
                    return false;
                if (!SetAnalogsNumberUpperLimit(i_step, fileParams.GetAttributeInt(nodeParamBlock, "upperlimit")))
                    return false;
                if (!SetAnalogsNumberIteration(i_step, fileParams.GetAttributeInt(nodeParamBlock, "iteration")))
                    return false;
            }
        } else if (nodeParamBlock->GetName() == "predictor") {
            AddPredictor(i_step);
            AddPredictorVect(m_stepsVect[i_step]);
            AddPredictorIteration(m_stepsIteration[i_step]);
            AddPredictorUpperLimit(m_stepsUpperLimit[i_step]);
            AddPredictorLowerLimit(m_stepsLowerLimit[i_step]);
            AddPredictorLocks(m_stepsLocks[i_step]);
            SetPreprocess(i_step, i_ptor, false);
            SetPreload(i_step, i_ptor, false);
            if(!ParsePredictors(fileParams, i_step, i_ptor, nodeParamBlock))
                return false;
            i_ptor++;
        } else {
            fileParams.UnknownNode(nodeParamBlock);
        }
        nodeParamBlock = nodeParamBlock->GetNext();
    }
    return true;
}

bool asParametersOptimization::ParsePredictors(asFileParametersOptimization &fileParams, int i_step, int i_ptor,
                                               const wxXmlNode *nodeParamBlock)
{
    wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
    while (nodeParam) {
        if (nodeParam->GetName() == "preload") {
            SetPreload(i_step, i_ptor, fileParams.GetBool(nodeParam));
        } else if (nodeParam->GetName() == "preprocessing") {
            SetPreprocess(i_step, i_ptor, true);
            if (!ParsePreprocessedPredictors(fileParams, i_step, i_ptor, nodeParam))
                return false;
        } else if (nodeParam->GetName() == "dataset_id") {
            if (!SetPredictorDatasetId(i_step, i_ptor, fileParams.GetString(nodeParam)))
                return false;
        } else if (nodeParam->GetName() == "data_id") {
            if (!SetPredictorDataIdVector(i_step, i_ptor, fileParams.GetVectorString(nodeParam)))
                return false;
            SetPredictorDataIdLock(i_step, i_ptor, fileParams.GetAttributeBool(nodeParam, "lock", true, false));
            if (IsPredictorDataIdLocked(i_step, i_ptor)) {
                if (!SetPredictorDataId(i_step, i_ptor, fileParams.GetString(nodeParam)))
                    return false;
            }
        } else if (nodeParam->GetName() == "level") {
            if (!SetPredictorLevelVector(i_step, i_ptor, fileParams.GetVectorFloat(nodeParam)))
                return false;
            SetPredictorLevelLock(i_step, i_ptor, fileParams.GetAttributeBool(nodeParam, "lock", true, false));
            if (IsPredictorLevelLocked(i_step, i_ptor)) {
                if (!SetPredictorLevel(i_step, i_ptor, fileParams.GetFloat(nodeParam)))
                    return false;
            }
        } else if (nodeParam->GetName() == "time") {
            SetPredictorTimeHoursLock(i_step, i_ptor, fileParams.GetAttributeBool(nodeParam, "lock", true, false));
            if (IsPredictorTimeHoursLocked(i_step, i_ptor)) {
                if (!SetPredictorTimeHours(i_step, i_ptor, fileParams.GetDouble(nodeParam)))
                    return false;
                VectorDouble vTimeHours;
                vTimeHours.push_back(GetPredictorTimeHours(i_step, i_ptor));
                if (!SetPreloadTimeHours(i_step, i_ptor, vTimeHours))
                    return false;
                if (!SetPredictorTimeHoursLowerLimit(i_step, i_ptor, GetPredictorTimeHours(i_step, i_ptor)))
                    return false;
                if (!SetPredictorTimeHoursUpperLimit(i_step, i_ptor, GetPredictorTimeHours(i_step, i_ptor)))
                    return false;
                if (!SetPredictorTimeHoursIteration(i_step, i_ptor, 6))
                    return false;
            } else {
                if (!SetPredictorTimeHoursLowerLimit(i_step, i_ptor, fileParams.GetAttributeDouble(nodeParam, "lowerlimit")))
                    return false;
                if (!SetPredictorTimeHoursUpperLimit(i_step, i_ptor, fileParams.GetAttributeDouble(nodeParam, "upperlimit")))
                    return false;
                if (!SetPredictorTimeHoursIteration(i_step, i_ptor, fileParams.GetAttributeDouble(nodeParam, "iteration")))
                    return false;
                // Initialize to ensure correct array sizes
                if (!SetPredictorTimeHours(i_step, i_ptor, GetPredictorTimeHoursLowerLimit(i_step, i_ptor)))
                    return false;
            }
        } else if (nodeParam->GetName() == "spatial_window") {
            if(!ParseSpatialWindow(fileParams, i_step, i_ptor, nodeParam))
                return false;
        } else if (nodeParam->GetName() == "criteria") {
            if (!SetPredictorCriteriaVector(i_step, i_ptor, fileParams.GetVectorString(nodeParam)))
                return false;
            SetPredictorCriteriaLock(i_step, i_ptor, fileParams.GetAttributeBool(nodeParam, "lock", true, false));
            if (IsPredictorCriteriaLocked(i_step, i_ptor)) {
                if (!SetPredictorCriteria(i_step, i_ptor, fileParams.GetString(nodeParam)))
                    return false;
            }
        } else if (nodeParam->GetName() == "weight") {
            SetPredictorWeightLock(i_step, i_ptor, fileParams.GetAttributeBool(nodeParam, "lock", true, false));
            if (IsPredictorWeightLocked(i_step, i_ptor)) {
                if (!SetPredictorWeight(i_step, i_ptor, fileParams.GetFloat(nodeParam)))
                    return false;
                if (!SetPredictorWeightLowerLimit(i_step, i_ptor, GetPredictorWeight(i_step, i_ptor)))
                    return false;
                if (!SetPredictorWeightUpperLimit(i_step, i_ptor, GetPredictorWeight(i_step, i_ptor)))
                    return false;
                if (!SetPredictorWeightIteration(i_step, i_ptor, 1))
                    return false;
            } else {
                if (!SetPredictorWeightLowerLimit(i_step, i_ptor, fileParams.GetAttributeFloat(nodeParam, "lowerlimit")))
                    return false;
                if (!SetPredictorWeightUpperLimit(i_step, i_ptor, fileParams.GetAttributeFloat(nodeParam, "upperlimit")))
                    return false;
                if (!SetPredictorWeightIteration(i_step, i_ptor, fileParams.GetAttributeFloat(nodeParam, "iteration")))
                    return false;
            }
        } else {
            fileParams.UnknownNode(nodeParam);
        }
        nodeParam = nodeParam->GetNext();
    }
    return true;
}

bool asParametersOptimization::ParseSpatialWindow(asFileParametersOptimization &fileParams, int i_step, int i_ptor,
                                                  const wxXmlNode *nodeParam)
{
    wxXmlNode *nodeWindow = nodeParam->GetChildren();
    while (nodeWindow) {
        if (nodeWindow->GetName() == "grid_type") {
            if (!SetPredictorGridType(i_step, i_ptor, fileParams.GetString(nodeWindow, "regular")))
                return false;
        } else if (nodeWindow->GetName() == "x_min") {
            SetPredictorXminLock(i_step, i_ptor, fileParams.GetAttributeBool(nodeWindow, "lock"));
            if (IsPredictorXminLocked(i_step, i_ptor)) {
                if (!SetPredictorXmin(i_step, i_ptor, fileParams.GetDouble(nodeWindow)))
                    return false;
                if (!SetPredictorXminLowerLimit(i_step, i_ptor, GetPredictorXmin(i_step, i_ptor)))
                    return false;
                if (!SetPredictorXminUpperLimit(i_step, i_ptor, GetPredictorXmin(i_step, i_ptor)))
                    return false;
                if (!SetPredictorXminIteration(i_step, i_ptor, 1))
                    return false;
            } else {
                if (!SetPredictorXminLowerLimit(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "lowerlimit")))
                    return false;
                if (!SetPredictorXminUpperLimit(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "upperlimit")))
                    return false;
                if (!SetPredictorXminIteration(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "iteration")))
                    return false;
            }
        } else if (nodeWindow->GetName() == "x_points_nb") {
            SetPredictorXptsnbLock(i_step, i_ptor, fileParams.GetAttributeBool(nodeWindow, "lock"));
            if (IsPredictorXptsnbLocked(i_step, i_ptor)) {
                if (!SetPredictorXptsnb(i_step, i_ptor, fileParams.GetDouble(nodeWindow)))
                    return false;
                if (!SetPredictorXptsnbLowerLimit(i_step, i_ptor, GetPredictorXptsnb(i_step, i_ptor)))
                    return false;
                if (!SetPredictorXptsnbUpperLimit(i_step, i_ptor, GetPredictorXptsnb(i_step, i_ptor)))
                    return false;
                if (!SetPredictorXptsnbIteration(i_step, i_ptor, 1))
                    return false;
            } else {
                if (!SetPredictorXptsnbLowerLimit(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "lowerlimit")))
                    return false;
                if (!SetPredictorXptsnbUpperLimit(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "upperlimit")))
                    return false;
                if (!SetPredictorXptsnbIteration(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "iteration")))
                    return false;
            }
        } else if (nodeWindow->GetName() == "x_step") {
            if (!SetPredictorXstep(i_step, i_ptor, fileParams.GetDouble(nodeWindow)))
                return false;
        } else if (nodeWindow->GetName() == "y_min") {
            SetPredictorYminLock(i_step, i_ptor, fileParams.GetAttributeBool(nodeWindow, "lock"));
            if (IsPredictorYminLocked(i_step, i_ptor)) {
                if (!SetPredictorYmin(i_step, i_ptor, fileParams.GetDouble(nodeWindow)))
                    return false;
                if (!SetPredictorYminLowerLimit(i_step, i_ptor, GetPredictorYmin(i_step, i_ptor)))
                    return false;
                if (!SetPredictorYminUpperLimit(i_step, i_ptor, GetPredictorYmin(i_step, i_ptor)))
                    return false;
                if (!SetPredictorYminIteration(i_step, i_ptor, 1))
                    return false;
            } else {
                if (!SetPredictorYminLowerLimit(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "lowerlimit")))
                    return false;
                if (!SetPredictorYminUpperLimit(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "upperlimit")))
                    return false;
                if (!SetPredictorYminIteration(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "iteration")))
                    return false;
            }
        } else if (nodeWindow->GetName() == "y_points_nb") {
            SetPredictorYptsnbLock(i_step, i_ptor, fileParams.GetAttributeBool(nodeWindow, "lock"));
            if (IsPredictorYptsnbLocked(i_step, i_ptor)) {
                if (!SetPredictorYptsnb(i_step, i_ptor, fileParams.GetDouble(nodeWindow)))
                    return false;
                if (!SetPredictorYptsnbLowerLimit(i_step, i_ptor, GetPredictorYptsnb(i_step, i_ptor)))
                    return false;
                if (!SetPredictorYptsnbUpperLimit(i_step, i_ptor, GetPredictorYptsnb(i_step, i_ptor)))
                    return false;
                if (!SetPredictorYptsnbIteration(i_step, i_ptor, 1))
                    return false;
            } else {
                if (!SetPredictorYptsnbLowerLimit(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "lowerlimit")))
                    return false;
                if (!SetPredictorYptsnbUpperLimit(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "upperlimit")))
                    return false;
                if (!SetPredictorYptsnbIteration(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "iteration")))
                    return false;
            }
        } else if (nodeWindow->GetName() == "y_step") {
            if (!SetPredictorYstep(i_step, i_ptor, fileParams.GetDouble(nodeWindow)))
                return false;
        } else {
            fileParams.UnknownNode(nodeWindow);
        }
        nodeWindow = nodeWindow->GetNext();
    }
    return true;
}

bool asParametersOptimization::ParsePreprocessedPredictors(asFileParametersOptimization &fileParams, int i_step,
                                                           int i_ptor, const wxXmlNode *nodeParam)
{
    int i_dataset = 0;
    wxXmlNode *nodePreprocess = nodeParam->GetChildren();
    while (nodePreprocess) {
        if (nodePreprocess->GetName() == "preprocessing_method") {
            if (!SetPreprocessMethod(i_step, i_ptor, fileParams.GetString(nodePreprocess)))
                return false;
        } else if (nodePreprocess->GetName() == "preprocessing_data") {
            if(!ParsePreprocessedPredictorDataset(fileParams, i_step, i_ptor, i_dataset, nodePreprocess))
                return false;
            i_dataset++;
        } else {
            fileParams.UnknownNode(nodePreprocess);
        }
        nodePreprocess = nodePreprocess->GetNext();
    }
    return true;
}

bool asParametersOptimization::ParsePreprocessedPredictorDataset(asFileParametersOptimization &fileParams, int i_step,
                                                                 int i_ptor, int i_dataset,
                                                                 const wxXmlNode *nodePreprocess)
{
    wxXmlNode *nodeParamPreprocess = nodePreprocess->GetChildren();
    while (nodeParamPreprocess) {
        if (nodeParamPreprocess->GetName() == "dataset_id") {
            if (!SetPreprocessDatasetId(i_step, i_ptor, i_dataset, fileParams.GetString(nodeParamPreprocess)))
                return false;
        } else if (nodeParamPreprocess->GetName() == "data_id") {
            if (!SetPreprocessDataIdVector(i_step, i_ptor, i_dataset, fileParams.GetVectorString(nodeParamPreprocess)))
                return false;
            SetPreprocessDataIdLock(i_step, i_ptor, i_dataset,
                                    fileParams.GetAttributeBool(nodeParamPreprocess, "lock", true, false));
            if (IsPreprocessDataIdLocked(i_step, i_ptor, i_dataset)) {
                if (!SetPreprocessDataId(i_step, i_ptor, i_dataset, fileParams.GetString(nodeParamPreprocess)))
                    return false;
            } else {
                // Initialize to ensure correct array sizes
                if (!SetPreprocessDataId(i_step, i_ptor, i_dataset,
                                         GetPreprocessDataIdVector(i_step, i_ptor, i_dataset)[0]))
                    return false;
            }
        } else if (nodeParamPreprocess->GetName() == "level") {
            if (!SetPreprocessLevelVector(i_step, i_ptor, i_dataset, fileParams.GetVectorFloat(nodeParamPreprocess)))
                return false;
            SetPreprocessLevelLock(i_step, i_ptor, i_dataset,
                                   fileParams.GetAttributeBool(nodeParamPreprocess, "lock", true, false));
            if (IsPreprocessLevelLocked(i_step, i_ptor, i_dataset)) {
                if (!SetPreprocessLevel(i_step, i_ptor, i_dataset, fileParams.GetFloat(nodeParamPreprocess)))
                    return false;
            } else {
                // Initialize to ensure correct array sizes
                if (!SetPreprocessLevel(i_step, i_ptor, i_dataset,
                                        GetPreprocessLevelVector(i_step, i_ptor, i_dataset)[0]))
                    return false;
            }
        } else if (nodeParamPreprocess->GetName() == "time") {
            SetPreprocessTimeHoursLock(i_step, i_ptor, i_dataset,
                                       fileParams.GetAttributeBool(nodeParamPreprocess, "lock", true, false));
            if (IsPreprocessTimeHoursLocked(i_step, i_ptor, i_dataset)) {
                if (!SetPreprocessTimeHours(i_step, i_ptor, i_dataset, fileParams.GetDouble(nodeParamPreprocess)))
                    return false;
                if (!SetPreprocessTimeHoursLowerLimit(i_step, i_ptor, i_dataset,
                                                      GetPreprocessTimeHours(i_step, i_ptor, i_dataset)))
                    return false;
                if (!SetPreprocessTimeHoursUpperLimit(i_step, i_ptor, i_dataset,
                                                      GetPreprocessTimeHours(i_step, i_ptor, i_dataset)))
                    return false;
                if (!SetPreprocessTimeHoursIteration(i_step, i_ptor, i_dataset, 6))
                    return false;
            } else {
                if (!SetPreprocessTimeHoursLowerLimit(i_step, i_ptor, i_dataset,
                                                      fileParams.GetAttributeDouble(nodeParamPreprocess, "lowerlimit")))
                    return false;
                if (!SetPreprocessTimeHoursUpperLimit(i_step, i_ptor, i_dataset,
                                                      fileParams.GetAttributeDouble(nodeParamPreprocess, "upperlimit")))
                    return false;
                if (!SetPreprocessTimeHoursIteration(i_step, i_ptor, i_dataset,
                                                     fileParams.GetAttributeDouble(nodeParamPreprocess, "iteration")))
                    return false;
                // Initialize to ensure correct array sizes
                if (!SetPreprocessTimeHours(i_step, i_ptor, i_dataset,
                                            GetPreprocessTimeHoursLowerLimit(i_step, i_ptor, i_dataset)))
                    return false;
            }
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

bool
asParametersOptimization::ParseForecastScore(asFileParametersOptimization &fileParams, const wxXmlNode *nodeProcess)
{
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "score") {
            if (!SetForecastScoreName(fileParams.GetString(nodeParamBlock)))
                return false;
        } else if (nodeParamBlock->GetName() == "threshold") {
            SetForecastScoreThreshold(fileParams.GetFloat(nodeParamBlock));
        } else if (nodeParamBlock->GetName() == "quantile") {
            SetForecastScoreQuantile(fileParams.GetFloat(nodeParamBlock));
        } else if (nodeParamBlock->GetName() == "postprocessing") {
            wxLogError(_("The postptocessing is not yet fully implemented."));
        } else {
            fileParams.UnknownNode(nodeParamBlock);
        }
        nodeParamBlock = nodeParamBlock->GetNext();
    }
    return true;
}

bool asParametersOptimization::ParseForecastScoreFinal(asFileParametersOptimization &fileParams,
                                                       const wxXmlNode *nodeProcess)
{
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "time_array") {
            if (!SetForecastScoreTimeArrayMode(fileParams.GetString(nodeParamBlock)))
                return false;
        } else {
            fileParams.UnknownNode(nodeParamBlock);
        }
        nodeParamBlock = nodeParamBlock->GetNext();
    }
    return true;
}

bool asParametersOptimization::SetSpatialWindowProperties()
{
    for (int i_step = 0; i_step < GetStepsNb(); i_step++) {
        for (int i_ptor = 0; i_ptor < GetPredictorsNb(i_step); i_ptor++) {
            double Xshift = std::fmod(GetPredictorXminLowerLimit(i_step, i_ptor), GetPredictorXstep(i_step, i_ptor));
            if (Xshift < 0)
                Xshift += GetPredictorXstep(i_step, i_ptor);
            if (!SetPredictorXshift(i_step, i_ptor, Xshift))
                return false;

            double Yshift = std::fmod(GetPredictorYminLowerLimit(i_step, i_ptor), GetPredictorYstep(i_step, i_ptor));
            if (Yshift < 0)
                Yshift += GetPredictorYstep(i_step, i_ptor);
            if (!SetPredictorYshift(i_step, i_ptor, Yshift))
                return false;

            if (GetPredictorXptsnbLowerLimit(i_step, i_ptor) <= 1 || GetPredictorYptsnbLowerLimit(i_step, i_ptor) <= 1)
                SetPredictorFlatAllowed(i_step, i_ptor, asFLAT_ALLOWED);
            if (IsPredictorXptsnbLocked(i_step, i_ptor) && GetPredictorXptsnb(i_step, i_ptor) <= 1)
                SetPredictorFlatAllowed(i_step, i_ptor, asFLAT_ALLOWED);
            if (IsPredictorYptsnbLocked(i_step, i_ptor) && GetPredictorYptsnb(i_step, i_ptor) <= 1)
                SetPredictorFlatAllowed(i_step, i_ptor, asFLAT_ALLOWED);
        }
    }

    return true;
}

bool asParametersOptimization::SetPreloadingProperties()
{
    for (int i_step = 0; i_step < GetStepsNb(); i_step++) {
        for (int i_ptor = 0; i_ptor < GetPredictorsNb(i_step); i_ptor++) {
            // Set maximum extent
            if (NeedsPreloading(i_step, i_ptor)) {
                // Set maximum extent
                if (!IsPredictorXminLocked(i_step, i_ptor)) {
                    SetPreloadXmin(i_step, i_ptor, GetPredictorXminLowerLimit(i_step, i_ptor));
                } else {
                    SetPreloadXmin(i_step, i_ptor, GetPredictorXmin(i_step, i_ptor));
                }

                if (!IsPredictorYminLocked(i_step, i_ptor)) {
                    SetPreloadYmin(i_step, i_ptor, GetPredictorYminLowerLimit(i_step, i_ptor));
                } else {
                    SetPreloadYmin(i_step, i_ptor, GetPredictorYmin(i_step, i_ptor));
                }

                if (!IsPredictorXptsnbLocked(i_step, i_ptor)) {
                    int Xbaseptsnb = std::abs(
                            GetPredictorXminUpperLimit(i_step, i_ptor) - GetPredictorXminLowerLimit(i_step, i_ptor)) /
                                     GetPredictorXstep(i_step, i_ptor);
                    SetPreloadXptsnb(i_step, i_ptor,
                                     Xbaseptsnb + GetPredictorXptsnbUpperLimit(i_step, i_ptor)); // No need to add +1
                } else {
                    SetPreloadXptsnb(i_step, i_ptor, GetPredictorXptsnb(i_step, i_ptor));
                }

                if (!IsPredictorYptsnbLocked(i_step, i_ptor)) {
                    int Ybaseptsnb = std::abs(
                            GetPredictorYminUpperLimit(i_step, i_ptor) - GetPredictorYminLowerLimit(i_step, i_ptor)) /
                                     GetPredictorYstep(i_step, i_ptor);
                    SetPreloadYptsnb(i_step, i_ptor,
                                     Ybaseptsnb + GetPredictorYptsnbUpperLimit(i_step, i_ptor)); // No need to add +1
                } else {
                    SetPreloadYptsnb(i_step, i_ptor, GetPredictorYptsnb(i_step, i_ptor));
                }
            }

            // Change predictor properties when preprocessing
            if (NeedsPreprocessing(i_step, i_ptor)) {
                if (GetPreprocessSize(i_step, i_ptor) == 1) {
                    SetPredictorDatasetId(i_step, i_ptor, GetPreprocessDatasetId(i_step, i_ptor, 0));
                    SetPredictorDataId(i_step, i_ptor, GetPreprocessDataId(i_step, i_ptor, 0));
                    SetPredictorLevel(i_step, i_ptor, GetPreprocessLevel(i_step, i_ptor, 0));
                    SetPredictorTimeHours(i_step, i_ptor, GetPreprocessTimeHours(i_step, i_ptor, 0));
                } else {
                    SetPredictorDatasetId(i_step, i_ptor, "mix");
                    SetPredictorDataId(i_step, i_ptor, "mix");
                    SetPredictorLevel(i_step, i_ptor, 0);
                    SetPredictorTimeHours(i_step, i_ptor, 0);
                }
            }

            // Set levels and time for preloading
            if (NeedsPreloading(i_step, i_ptor) && !NeedsPreprocessing(i_step, i_ptor)) {
                if (!SetPreloadDataIds(i_step, i_ptor, GetPredictorDataIdVector(i_step, i_ptor)))
                    return false;
                if (!SetPreloadLevels(i_step, i_ptor, GetPredictorLevelVector(i_step, i_ptor)))
                    return false;
                VectorDouble vTimeHours;
                for (double h = GetPredictorTimeHoursLowerLimit(i_step, i_ptor);
                     h <= GetPredictorTimeHoursUpperLimit(i_step, i_ptor); h += GetPredictorTimeHoursIteration(i_step,
                                                                                                               i_ptor)) {
                    vTimeHours.push_back(h);
                }
                if (!SetPreloadTimeHours(i_step, i_ptor, vTimeHours))
                    return false;
            } else if (NeedsPreloading(i_step, i_ptor) && NeedsPreprocessing(i_step, i_ptor)) {
                // Check the preprocessing method
                wxString method = GetPreprocessMethod(i_step, i_ptor);
                VectorFloat preprocLevels;
                VectorDouble preprocTimeHours;

                // Different actions depending on the preprocessing method.
                if (method.IsSameAs("Gradients")) {
                    preprocLevels = GetPreprocessLevelVector(i_step, i_ptor, 0);

                    for (double h = GetPreprocessTimeHoursLowerLimit(i_step, i_ptor, 0);
                         h <= GetPreprocessTimeHoursUpperLimit(i_step, i_ptor, 0); h += GetPreprocessTimeHoursIteration(
                            i_step, i_ptor, 0)) {
                        preprocTimeHours.push_back(h);
                    }
                } else if (method.IsSameAs("HumidityFlux")) {
                    preprocLevels = GetPreprocessLevelVector(i_step, i_ptor, 0);

                    for (double h = GetPreprocessTimeHoursLowerLimit(i_step, i_ptor, 0);
                         h <= GetPreprocessTimeHoursUpperLimit(i_step, i_ptor, 0); h += GetPreprocessTimeHoursIteration(
                            i_step, i_ptor, 0)) {
                        preprocTimeHours.push_back(h);
                    }
                } else if (method.IsSameAs("Multiplication") || method.IsSameAs("Multiply") ||
                           method.IsSameAs("HumidityIndex")) {
                    preprocLevels = GetPreprocessLevelVector(i_step, i_ptor, 0);

                    for (double h = GetPreprocessTimeHoursLowerLimit(i_step, i_ptor, 0);
                         h <= GetPreprocessTimeHoursUpperLimit(i_step, i_ptor, 0); h += GetPreprocessTimeHoursIteration(
                            i_step, i_ptor, 0)) {
                        preprocTimeHours.push_back(h);
                    }
                } else if (method.IsSameAs("FormerHumidityIndex")) {
                    wxLogWarning(_("The %s preprocessing method is not handled in the optimizer."), method);
                    return false;
                } else {
                    wxLogWarning(_("The %s preprocessing method is not yet handled with the preload option."), method);
                }

                if (!SetPreloadLevels(i_step, i_ptor, preprocLevels))
                    return false;
                if (!SetPreloadTimeHours(i_step, i_ptor, preprocTimeHours))
                    return false;
            }
        }
    }

    return true;
}

void asParametersOptimization::InitRandomValues()
{
    if (!m_timeArrayAnalogsIntervalDaysLocks) {
        m_timeArrayAnalogsIntervalDays = asTools::Random(m_timeArrayAnalogsIntervalDaysLowerLimit,
                                                         m_timeArrayAnalogsIntervalDaysUpperLimit,
                                                         m_timeArrayAnalogsIntervalDaysIteration);
    }

    for (int i = 0; i < GetStepsNb(); i++) {
        if (!m_stepsLocks[i].analogsNumber) {
            SetAnalogsNumber(i, asTools::Random(m_stepsLowerLimit[i].analogsNumber, m_stepsUpperLimit[i].analogsNumber,
                                                m_stepsIteration[i].analogsNumber));
        }

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (!m_stepsLocks[i].predictors[j].preprocessDataId[k]) {
                        int length = m_stepsVect[i].predictors[j].preprocessDataId[k].size();
                        int row = asTools::Random(0, length - 1);
                        wxASSERT(m_stepsVect[i].predictors[j].preprocessDataId[k].size() > (unsigned) row);

                        SetPreprocessDataId(i, j, k, m_stepsVect[i].predictors[j].preprocessDataId[k][row]);
                    }

                    if (!m_stepsLocks[i].predictors[j].preprocessLevels[k]) {
                        int length = m_stepsVect[i].predictors[j].preprocessLevels[k].size();
                        int row = asTools::Random(0, length - 1);
                        wxASSERT(m_stepsVect[i].predictors[j].preprocessLevels[k].size() > (unsigned) row);

                        SetPreprocessLevel(i, j, k, m_stepsVect[i].predictors[j].preprocessLevels[k][row]);
                    }

                    if (!m_stepsLocks[i].predictors[j].preprocessTimeHours[k]) {
                        SetPreprocessTimeHours(i, j, k, asTools::Random(
                                m_stepsLowerLimit[i].predictors[j].preprocessTimeHours[k],
                                m_stepsUpperLimit[i].predictors[j].preprocessTimeHours[k],
                                m_stepsIteration[i].predictors[j].preprocessTimeHours[k]));
                    }
                }
            } else {
                if (!m_stepsLocks[i].predictors[j].dataId) {
                    int length = m_stepsVect[i].predictors[j].dataId.size();
                    int row = asTools::Random(0, length - 1);
                    wxASSERT(m_stepsVect[i].predictors[j].dataId.size() > (unsigned) row);

                    SetPredictorDataId(i, j, m_stepsVect[i].predictors[j].dataId[row]);
                }

                if (!m_stepsLocks[i].predictors[j].level) {
                    int length = m_stepsVect[i].predictors[j].level.size();
                    int row = asTools::Random(0, length - 1);
                    wxASSERT(m_stepsVect[i].predictors[j].level.size() > (unsigned) row);

                    SetPredictorLevel(i, j, m_stepsVect[i].predictors[j].level[row]);
                }

                if (!m_stepsLocks[i].predictors[j].timeHours) {
                    SetPredictorTimeHours(i, j, asTools::Random(m_stepsLowerLimit[i].predictors[j].timeHours,
                                                                m_stepsUpperLimit[i].predictors[j].timeHours,
                                                                m_stepsIteration[i].predictors[j].timeHours));
                }

            }

            if (!m_stepsLocks[i].predictors[j].xMin) {
                SetPredictorXmin(i, j, asTools::Random(m_stepsLowerLimit[i].predictors[j].xMin,
                                                       m_stepsUpperLimit[i].predictors[j].xMin,
                                                       m_stepsIteration[i].predictors[j].xMin));
            }

            if (!m_stepsLocks[i].predictors[j].xPtsNb) {
                SetPredictorXptsnb(i, j, asTools::Random(m_stepsLowerLimit[i].predictors[j].xPtsNb,
                                                         m_stepsUpperLimit[i].predictors[j].xPtsNb,
                                                         m_stepsIteration[i].predictors[j].xPtsNb));
            }

            if (!m_stepsLocks[i].predictors[j].yMin) {
                SetPredictorYmin(i, j, asTools::Random(m_stepsLowerLimit[i].predictors[j].yMin,
                                                       m_stepsUpperLimit[i].predictors[j].yMin,
                                                       m_stepsIteration[i].predictors[j].yMin));
            }

            if (!m_stepsLocks[i].predictors[j].yPtsNb) {
                SetPredictorYptsnb(i, j, asTools::Random(m_stepsLowerLimit[i].predictors[j].yPtsNb,
                                                         m_stepsUpperLimit[i].predictors[j].yPtsNb,
                                                         m_stepsIteration[i].predictors[j].yPtsNb));
            }

            if (!m_stepsLocks[i].predictors[j].weight) {
                SetPredictorWeight(i, j, asTools::Random(m_stepsLowerLimit[i].predictors[j].weight,
                                                         m_stepsUpperLimit[i].predictors[j].weight,
                                                         m_stepsIteration[i].predictors[j].weight));
            }

            if (!m_stepsLocks[i].predictors[j].criteria) {
                int length = m_stepsVect[i].predictors[j].criteria.size();
                int row = asTools::Random(0, length - 1);
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
                            ratio = asTools::Round(ratio);
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
                        ratio = asTools::Round(ratio);
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

        // Check total of the locked weights
        if (totWeightLocked > 1) {
            wxLogError(_("The sum of the locked weights of the analogy level number %d is higher than 1 (%f)."), i + 1,
                       totWeightLocked);
            return false;
        }
        float totWeightManageable = totWeight - totWeightLocked;

        // For every weights but the last
        float newSum = 0;
        for (int j = 0; j < GetPredictorsNb(i) - 1; j++) {
            if (!IsPredictorWeightLocked(i, j)) {
                float precision = GetPredictorWeightIteration(i, j);
                float newWeight = GetPredictorWeight(i, j) / totWeightManageable;
                newWeight = precision * asTools::Round(newWeight * (1.0 / precision));
                newSum += newWeight;

                SetPredictorWeight(i, j, newWeight);
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

// TODO (Pascal#1#): Can be optimized by looping on the given vector (sorted first) instead
void asParametersOptimization::Unlock(VectorInt &indices)
{
    int counter = 0;
    int length = indices.size();

    if (asTools::SortedArraySearch(&indices[0], &indices[length - 1], counter) >= 0) {
        m_timeArrayAnalogsIntervalDaysLocks = false;
    }
    counter++;

    for (int i = 0; i < GetStepsNb(); i++) {
        if (asTools::SortedArraySearch(&indices[0], &indices[length - 1], counter) >= 0) {
            m_stepsLocks[i].analogsNumber = false;
        }
        counter++;

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (asTools::SortedArraySearch(&indices[0], &indices[length - 1], counter) >= 0) {
                        m_stepsLocks[i].predictors[j].preprocessDataId[k] = false;
                    }
                    counter++;
                    if (asTools::SortedArraySearch(&indices[0], &indices[length - 1], counter) >= 0) {
                        m_stepsLocks[i].predictors[j].preprocessLevels[k] = false;
                    }
                    counter++;
                    if (asTools::SortedArraySearch(&indices[0], &indices[length - 1], counter) >= 0) {
                        m_stepsLocks[i].predictors[j].preprocessTimeHours[k] = false;
                    }
                    counter++;
                }
            } else {
                if (asTools::SortedArraySearch(&indices[0], &indices[length - 1], counter) >= 0) {
                    m_stepsLocks[i].predictors[j].dataId = false;
                }
                counter++;
                if (asTools::SortedArraySearch(&indices[0], &indices[length - 1], counter) >= 0) {
                    m_stepsLocks[i].predictors[j].level = false;
                }
                counter++;
                if (asTools::SortedArraySearch(&indices[0], &indices[length - 1], counter) >= 0) {
                    m_stepsLocks[i].predictors[j].timeHours = false;
                }
                counter++;
            }

            if (asTools::SortedArraySearch(&indices[0], &indices[length - 1], counter) >= 0) {
                m_stepsLocks[i].predictors[j].xMin = false;
            }
            counter++;
            if (asTools::SortedArraySearch(&indices[0], &indices[length - 1], counter) >= 0) {
                m_stepsLocks[i].predictors[j].xPtsNb = false;
            }
            counter++;
            if (asTools::SortedArraySearch(&indices[0], &indices[length - 1], counter) >= 0) {
                m_stepsLocks[i].predictors[j].yMin = false;
            }
            counter++;
            if (asTools::SortedArraySearch(&indices[0], &indices[length - 1], counter) >= 0) {
                m_stepsLocks[i].predictors[j].yPtsNb = false;
            }
            counter++;
            if (asTools::SortedArraySearch(&indices[0], &indices[length - 1], counter) >= 0) {
                m_stepsLocks[i].predictors[j].weight = false;
            }
            counter++;
            if (asTools::SortedArraySearch(&indices[0], &indices[length - 1], counter) >= 0) {
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
