#include "asParametersOptimization.h"

#include <asFileParametersOptimization.h>


asParametersOptimization::asParametersOptimization()
:
asParametersScoring()
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

    stepIteration.AnalogsNumber = 1;
    stepUpperLimit.AnalogsNumber = 1000;
    stepLowerLimit.AnalogsNumber = 5;
    stepLocks.AnalogsNumber = true;
    stepVect.AnalogsNumber.push_back(0);

    AddPredictorIteration(stepIteration);
    AddPredictorUpperLimit(stepUpperLimit);
    AddPredictorLowerLimit(stepLowerLimit);
    AddPredictorLocks(stepLocks);
    AddPredictorVect(stepVect);

    m_stepsIteration.push_back(stepIteration);
    m_stepsUpperLimit.push_back(stepUpperLimit);
    m_stepsLowerLimit.push_back(stepLowerLimit);
    m_stepsLocks.push_back(stepLocks);
    m_stepsVect.push_back(stepVect);

    // Set sizes
    SetSizes();

}

void asParametersOptimization::AddPredictorIteration(ParamsStep &step)
{
    ParamsPredictor predictor;

    predictor.Xmin = 2.5;
    predictor.Xptsnb = 1;
    predictor.Ymin = 2.5;
    predictor.Yptsnb = 1;
    predictor.TimeHours = 6;
    predictor.Weight = 0.01f;

    step.Predictors.push_back(predictor);
}

void asParametersOptimization::AddPredictorUpperLimit(ParamsStep &step)
{
    ParamsPredictor predictor;

    predictor.Xmin = 717.5;
    predictor.Xptsnb = 20;
    predictor.Ymin = 87.5;
    predictor.Yptsnb = 16;
    predictor.TimeHours = 36;
    predictor.Weight = 1;

    step.Predictors.push_back(predictor);
}

void asParametersOptimization::AddPredictorLowerLimit(ParamsStep &step)
{
    ParamsPredictor predictor;

    predictor.Xmin = 0;
    predictor.Xptsnb = 1;
    predictor.Ymin = 0;
    predictor.Yptsnb = 1;
    predictor.TimeHours = 6;
    predictor.Weight = 0;

    step.Predictors.push_back(predictor);
}

void asParametersOptimization::AddPredictorLocks(ParamsStepBool &step)
{
    ParamsPredictorBool predictor;

    predictor.DataId = true;
    predictor.Level = true;
    predictor.Xmin = true;
    predictor.Xptsnb = true;
    predictor.Ymin = true;
    predictor.Yptsnb = true;
    predictor.TimeHours = true;
    predictor.Weight = true;
    predictor.Criteria = true;

    step.Predictors.push_back(predictor);
}

bool asParametersOptimization::LoadFromFile(const wxString &filePath)
{
	asLogMessage(_("Loading parameters file."));

	if (filePath.IsEmpty())
	{
		asLogError(_("The given path to the parameters file is empty."));
		return false;
	}

	asFileParametersOptimization fileParams(filePath, asFile::ReadOnly);
	if (!fileParams.Open()) return false;

	if (!fileParams.CheckRootElement()) return false;

	int i_step = 0;
	wxXmlNode *nodeProcess = fileParams.GetRoot()->GetChildren();
	while (nodeProcess) {

		// Description
		if (nodeProcess->GetName() == "description") {
			wxXmlNode *nodeParam = nodeProcess->GetChildren();
			while (nodeParam) {
				if (nodeParam->GetName() == "method_id") {
					SetMethodId(fileParams.GetString(nodeParam));
				}
				else if (nodeParam->GetName() == "method_id_display") {
					SetMethodIdDisplay(fileParams.GetString(nodeParam));
				}
				else if (nodeParam->GetName() == "specific_tag") {
					SetSpecificTag(fileParams.GetString(nodeParam));
				}
				else if (nodeParam->GetName() == "specific_tag_display") {
					SetSpecificTagDisplay(fileParams.GetString(nodeParam));
				}
				else if (nodeParam->GetName() == "description") {
					SetDescription(fileParams.GetString(nodeParam));
				}
				else {
					fileParams.UnknownNode(nodeParam);
				}
				nodeParam = nodeParam->GetNext();
			}

			// Time properties
		}
		else if (nodeProcess->GetName() == "time_properties") {
			wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
			while (nodeParamBlock) {
				if (nodeParamBlock->GetName() == "archive_period") {
					wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
					while (nodeParam) {
						if (nodeParam->GetName() == "start_year") {
							if (!SetArchiveYearStart(fileParams.GetInt(nodeParam))) return false;
						}
						else if (nodeParam->GetName() == "end_year") {
							if (!SetArchiveYearEnd(fileParams.GetInt(nodeParam))) return false;
						}
						else if (nodeParam->GetName() == "start") {
							if (!SetArchiveStart(fileParams.GetString(nodeParam))) return false;
						}
						else if (nodeParam->GetName() == "end") {
							if (!SetArchiveEnd(fileParams.GetString(nodeParam))) return false;
						}
						else {
							fileParams.UnknownNode(nodeParam);
						}
						nodeParam = nodeParam->GetNext();
					}
				}
				else if (nodeParamBlock->GetName() == "calibration_period") {
					wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
					while (nodeParam) {
						if (nodeParam->GetName() == "start_year") {
							if (!SetCalibrationYearStart(fileParams.GetInt(nodeParam))) return false;
						}
						else if (nodeParam->GetName() == "end_year") {
							if (!SetCalibrationYearEnd(fileParams.GetInt(nodeParam))) return false;
						}
						else if (nodeParam->GetName() == "start") {
							if (!SetCalibrationStart(fileParams.GetString(nodeParam))) return false;
						}
						else if (nodeParam->GetName() == "end") {
							if (!SetCalibrationEnd(fileParams.GetString(nodeParam))) return false;
						}
						else {
							fileParams.UnknownNode(nodeParam);
						}
						nodeParam = nodeParam->GetNext();
					}
				}
				else if (nodeParamBlock->GetName() == "validation_period") {
					wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
					while (nodeParam) {
						if (nodeParam->GetName() == "years") {
							if (!SetValidationYearsVector(fileParams.GetVectorInt(nodeParam))) return false;
						}
						else {
							fileParams.UnknownNode(nodeParam);
						}
						nodeParam = nodeParam->GetNext();
					}
				}
				else if (nodeParamBlock->GetName() == "time_step") {
					if (!SetTimeArrayTargetTimeStepHours(fileParams.GetDouble(nodeParamBlock))) return false;
					if (!SetTimeArrayAnalogsTimeStepHours(fileParams.GetDouble(nodeParamBlock))) return false;
				}
				else if (nodeParamBlock->GetName() == "time_array_target") {
					wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
					while (nodeParam) {
						if (nodeParam->GetName() == "time_array") {
							if (!SetTimeArrayTargetMode(fileParams.GetString(nodeParam))) return false;
						}
						else if (nodeParam->GetName() == "predictand_serie_name") {
							if (!SetTimeArrayTargetPredictandSerieName(fileParams.GetString(nodeParam))) return false;
						}
						else if (nodeParam->GetName() == "predictand_min_threshold") {
							if (!SetTimeArrayTargetPredictandMinThreshold(fileParams.GetFloat(nodeParam))) return false;
						}
						else if (nodeParam->GetName() == "predictand_max_threshold") {
							if (!SetTimeArrayTargetPredictandMaxThreshold(fileParams.GetFloat(nodeParam))) return false;
						}
						else {
							fileParams.UnknownNode(nodeParam);
						}
						nodeParam = nodeParam->GetNext();
					}
				}
				else if (nodeParamBlock->GetName() == "time_array_analogs") {
					wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
					while (nodeParam) {
						if (nodeParam->GetName() == "time_array") {
							if (!SetTimeArrayAnalogsMode(fileParams.GetString(nodeParam))) return false;
						}
						else if (nodeParam->GetName() == "interval_days") {
							SetTimeArrayAnalogsIntervalDaysLock(fileParams.GetAttributeBool(nodeParam, "lock"));
							if (IsTimeArrayAnalogsIntervalDaysLocked()) {
								if (!SetTimeArrayAnalogsIntervalDays(fileParams.GetInt(nodeParam))) return false;
								if (!SetTimeArrayAnalogsIntervalDaysLowerLimit(GetTimeArrayAnalogsIntervalDays())) return false;
								if (!SetTimeArrayAnalogsIntervalDaysUpperLimit(GetTimeArrayAnalogsIntervalDays())) return false;
								if (!SetTimeArrayAnalogsIntervalDaysIteration(1)) return false;
							} else {
								if (!SetTimeArrayAnalogsIntervalDaysLowerLimit(fileParams.GetAttributeInt(nodeParam, "lowerlimit"))) return false;
								if (!SetTimeArrayAnalogsIntervalDaysUpperLimit(fileParams.GetAttributeInt(nodeParam, "upperlimit"))) return false;
								if (!SetTimeArrayAnalogsIntervalDaysIteration(fileParams.GetAttributeInt(nodeParam, "iteration"))) return false;
							}
						}
						else if (nodeParam->GetName() == "exclude_days") {
							if (!SetTimeArrayAnalogsExcludeDays(fileParams.GetInt(nodeParam))) return false;
						}
						else {
							fileParams.UnknownNode(nodeParam);
						}
						nodeParam = nodeParam->GetNext();
					}
				}
				else {
					fileParams.UnknownNode(nodeParamBlock);
				}
				nodeParamBlock = nodeParamBlock->GetNext();
			}

			// Analog dates
		}
		else if (nodeProcess->GetName() == "analog_dates") {
			AddStep();
			int i_ptor = 0;
			wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
			while (nodeParamBlock) {
				if (nodeParamBlock->GetName() == "analogs_number") {
					SetAnalogsNumberLock(i_step, fileParams.GetAttributeBool(nodeParamBlock, "lock"));
					if (IsAnalogsNumberLocked(i_step)) {
						if (!SetAnalogsNumber(i_step, fileParams.GetInt(nodeParamBlock))) return false;
						if (!SetAnalogsNumberLowerLimit(i_step, GetAnalogsNumber(i_step))) return false;
						if (!SetAnalogsNumberUpperLimit(i_step, GetAnalogsNumber(i_step))) return false;
						if (!SetAnalogsNumberUpperLimit(i_step, 1)) return false;
					}
					else {
						if (!SetAnalogsNumberLowerLimit(i_step, fileParams.GetAttributeInt(nodeParamBlock, "lowerlimit"))) return false;
						if (!SetAnalogsNumberUpperLimit(i_step, fileParams.GetAttributeInt(nodeParamBlock, "upperlimit"))) return false;
						if (!SetAnalogsNumberUpperLimit(i_step, fileParams.GetAttributeInt(nodeParamBlock, "iteration"))) return false;
					}
				}
				else if (nodeParamBlock->GetName() == "predictor") {
					AddPredictor(i_step);
					AddPredictorVect(m_stepsVect[i_step]);
					SetPreprocess(i_step, i_ptor, false);
					SetPreload(i_step, i_ptor, false);
					wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
					while (nodeParam) {
						if (nodeParam->GetName() == "preload") {
							SetPreload(i_step, i_ptor, fileParams.GetBool(nodeParam));
						}
						else if (nodeParam->GetName() == "preprocessing") {
							SetPreprocess(i_step, i_ptor, true);
							int i_dataset = 0;
							wxXmlNode *nodePreprocess = nodeParam->GetChildren();
							while (nodePreprocess) {
								if (nodePreprocess->GetName() == "preprocessing_method") {
									if (!SetPreprocessMethod(i_step, i_ptor, fileParams.GetString(nodePreprocess))) return false;
								}
								else if (nodePreprocess->GetName() == "preprocessing_data") {
									wxXmlNode *nodeParamPreprocess = nodePreprocess->GetChildren();
									while (nodeParamPreprocess) {
										if (nodeParamPreprocess->GetName() == "dataset_id") {
											if (!SetPreprocessDatasetId(i_step, i_ptor, i_dataset, fileParams.GetString(nodeParamPreprocess))) return false;
										}
										else if (nodeParamPreprocess->GetName() == "data_id") {
											if (!SetPreprocessDataIdVector(i_step, i_ptor, i_dataset, fileParams.GetVectorString(nodeParamPreprocess))) return false;
											SetPreprocessDataIdLock(i_step, i_ptor, i_dataset, fileParams.GetAttributeBool(nodeParamPreprocess, "lock"));
											if (IsPreprocessDataIdLocked(i_step, i_ptor, i_dataset))
											{
												if (!SetPreprocessDataId(i_step, i_ptor, i_dataset, fileParams.GetString(nodeParamPreprocess))) return false;
											}
											else {
												// Initialize to ensure correct array sizes
												if (!SetPreprocessDataId(i_step, i_ptor, i_dataset, GetPreprocessDataIdVector(i_step, i_ptor, i_dataset)[0])) return false;
											}
										}
										else if (nodeParamPreprocess->GetName() == "level") {
											if (!SetPreprocessLevelVector(i_step, i_ptor, i_dataset, fileParams.GetVectorFloat(nodeParamPreprocess))) return false;
											SetPreprocessLevelLock(i_step, i_ptor, i_dataset, fileParams.GetAttributeBool(nodeParamPreprocess, "lock"));
											if (IsPreprocessLevelLocked(i_step, i_ptor, i_dataset))
											{
												if (!SetPreprocessLevel(i_step, i_ptor, i_dataset, fileParams.GetFloat(nodeParamPreprocess))) return false;
											}
											else {
												// Initialize to ensure correct array sizes
												if (!SetPreprocessLevel(i_step, i_ptor, i_dataset, GetPreprocessLevelVector(i_step, i_ptor, i_dataset)[0])) return false;
											}
										}
										else if (nodeParamPreprocess->GetName() == "time") {
											SetPreprocessTimeHoursLock(i_step, i_ptor, i_dataset, fileParams.GetAttributeBool(nodeParamPreprocess, "lock"));
											if (IsPreprocessTimeHoursLocked(i_step, i_ptor, i_dataset)) {
												if (!SetPreprocessTimeHours(i_step, i_ptor, i_dataset, fileParams.GetDouble(nodeParamPreprocess))) return false;
												if (!SetPreprocessTimeHoursLowerLimit(i_step, i_ptor, i_dataset, GetPreprocessTimeHours(i_step, i_ptor, i_dataset))) return false;
												if (!SetPreprocessTimeHoursUpperLimit(i_step, i_ptor, i_dataset, GetPreprocessTimeHours(i_step, i_ptor, i_dataset))) return false;
												if (!SetPreprocessTimeHoursIteration(i_step, i_ptor, i_dataset, 6)) return false;
											}
											else {
												if (!SetPreprocessTimeHoursLowerLimit(i_step, i_ptor, i_dataset, fileParams.GetAttributeDouble(nodeParamPreprocess, "lowerlimit"))) return false;
												if (!SetPreprocessTimeHoursUpperLimit(i_step, i_ptor, i_dataset, fileParams.GetAttributeDouble(nodeParamPreprocess, "upperlimit"))) return false;
												if (!SetPreprocessTimeHoursIteration(i_step, i_ptor, i_dataset, fileParams.GetAttributeDouble(nodeParamPreprocess, "iteration"))) return false;
												// Initialize to ensure correct array sizes
												if (!SetPreprocessTimeHours(i_step, i_ptor, i_dataset, fileParams.GetVectorDouble(nodeParamPreprocess)[0])) return false;
											}
										}
										else {
											fileParams.UnknownNode(nodeParamPreprocess);
										}
										nodeParamPreprocess = nodeParamPreprocess->GetNext();
									}
									i_dataset++;
								}
								else {
									fileParams.UnknownNode(nodePreprocess);
								}
								nodePreprocess = nodePreprocess->GetNext();
							}
						}
						else if (nodeParam->GetName() == "dataset_id") {
							if (!SetPredictorDatasetId(i_step, i_ptor, fileParams.GetString(nodeParam))) return false;
						}
						else if (nodeParam->GetName() == "data_id") {
							if (!SetPredictorDataIdVector(i_step, i_ptor, fileParams.GetVectorString(nodeParam))) return false;
							SetPredictorDataIdLock(i_step, i_ptor, fileParams.GetAttributeBool(nodeParam, "lock"));
							if (IsPredictorDataIdLocked(i_step, i_ptor))
							{
								if (!SetPredictorDataId(i_step, i_ptor, fileParams.GetString(nodeParam))) return false;
							}
						}
						else if (nodeParam->GetName() == "level") {
							if (!SetPredictorLevelVector(i_step, i_ptor, fileParams.GetVectorFloat(nodeParam))) return false;
							SetPredictorLevelLock(i_step, i_ptor, fileParams.GetAttributeBool(nodeParam, "lock"));
							if (IsPredictorLevelLocked(i_step, i_ptor))
							{
								if (!SetPredictorLevel(i_step, i_ptor, fileParams.GetFloat(nodeParam))) return false;
							}
						}
						else if (nodeParam->GetName() == "time") {
							SetPredictorTimeHoursLock(i_step, i_ptor, fileParams.GetAttributeBool(nodeParam, "lock"));
							if (IsPredictorTimeHoursLocked(i_step, i_ptor)) {
								if (!SetPredictorTimeHours(i_step, i_ptor, fileParams.GetDouble(nodeParam))) return false;
								VectorDouble vTimeHours;
								vTimeHours.push_back(GetPredictorTimeHours(i_step, i_ptor));
								if (!SetPreloadTimeHours(i_step, i_ptor, vTimeHours)) return false;
								if (!SetPredictorTimeHoursLowerLimit(i_step, i_ptor, GetPredictorTimeHours(i_step, i_ptor))) return false;
								if (!SetPredictorTimeHoursUpperLimit(i_step, i_ptor, GetPredictorTimeHours(i_step, i_ptor))) return false;
								if (!SetPredictorTimeHoursIteration(i_step, i_ptor, 6)) return false;
							}
							else {
								if (!SetPredictorTimeHoursLowerLimit(i_step, i_ptor, fileParams.GetAttributeDouble(nodeParam, "lowerlimit"))) return false;
								if (!SetPredictorTimeHoursUpperLimit(i_step, i_ptor, fileParams.GetAttributeDouble(nodeParam, "upperlimit"))) return false;
								if (!SetPredictorTimeHoursIteration(i_step, i_ptor, fileParams.GetAttributeDouble(nodeParam, "iteration"))) return false;
							}
						}
						else if (nodeParam->GetName() == "spatial_window") {
							wxXmlNode *nodeWindow = nodeParam->GetChildren();
							while (nodeWindow) {
								if (nodeWindow->GetName() == "grid_type") {
									if (!SetPredictorGridType(i_step, i_ptor, fileParams.GetString(nodeWindow, "regular"))) return false;
								}
								else if (nodeWindow->GetName() == "x_min") {
									SetPredictorXminLock(i_step, i_ptor, fileParams.GetAttributeBool(nodeWindow, "lock"));
									if (IsPredictorXminLocked(i_step, i_ptor)) {
										if (!SetPredictorXmin(i_step, i_ptor, fileParams.GetDouble(nodeWindow))) return false;
										if (!SetPredictorXminLowerLimit(i_step, i_ptor, GetPredictorXmin(i_step, i_ptor))) return false;
										if (!SetPredictorXminUpperLimit(i_step, i_ptor, GetPredictorXmin(i_step, i_ptor))) return false;
										if (!SetPredictorXminIteration(i_step, i_ptor, 1)) return false;
									}
									else {
										if (!SetPredictorXminLowerLimit(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "lowerlimit"))) return false;
										if (!SetPredictorXminUpperLimit(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "upperlimit"))) return false;
										if (!SetPredictorXminIteration(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "iteration"))) return false;
									}
								}
								else if (nodeWindow->GetName() == "x_points_nb") {
									SetPredictorXptsnbLock(i_step, i_ptor, fileParams.GetAttributeBool(nodeWindow, "lock"));
									if (IsPredictorXptsnbLocked(i_step, i_ptor)) {
										if (!SetPredictorXptsnb(i_step, i_ptor, fileParams.GetDouble(nodeWindow))) return false;
										if (!SetPredictorXptsnbLowerLimit(i_step, i_ptor, GetPredictorXptsnb(i_step, i_ptor))) return false;
										if (!SetPredictorXptsnbUpperLimit(i_step, i_ptor, GetPredictorXptsnb(i_step, i_ptor))) return false;
										if (!SetPredictorXptsnbIteration(i_step, i_ptor, 1)) return false;
									}
									else {
										if (!SetPredictorXptsnbLowerLimit(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "lowerlimit"))) return false;
										if (!SetPredictorXptsnbUpperLimit(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "upperlimit"))) return false;
										if (!SetPredictorXptsnbIteration(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "iteration"))) return false;
									}
								}
								else if (nodeWindow->GetName() == "x_step") {
									if (!SetPredictorXstep(i_step, i_ptor, fileParams.GetDouble(nodeWindow))) return false;
								}
								else if (nodeWindow->GetName() == "y_min") {
									SetPredictorXminLock(i_step, i_ptor, fileParams.GetAttributeBool(nodeWindow, "lock"));
									if (IsPredictorXminLocked(i_step, i_ptor)) {
										if (!SetPredictorYmin(i_step, i_ptor, fileParams.GetDouble(nodeWindow))) return false;
										if (!SetPredictorYminLowerLimit(i_step, i_ptor, GetPredictorYmin(i_step, i_ptor))) return false;
										if (!SetPredictorYminUpperLimit(i_step, i_ptor, GetPredictorYmin(i_step, i_ptor))) return false;
										if (!SetPredictorYminIteration(i_step, i_ptor, 1)) return false;
									}
									else {
										if (!SetPredictorYminLowerLimit(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "lowerlimit"))) return false;
										if (!SetPredictorYminUpperLimit(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "upperlimit"))) return false;
										if (!SetPredictorYminIteration(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "iteration"))) return false;
									}
								}
								else if (nodeWindow->GetName() == "y_points_nb") {
									SetPredictorYptsnbLock(i_step, i_ptor, fileParams.GetAttributeBool(nodeWindow, "lock"));
									if (IsPredictorYptsnbLocked(i_step, i_ptor)) {
										if (!SetPredictorYptsnb(i_step, i_ptor, fileParams.GetDouble(nodeWindow))) return false;
										if (!SetPredictorYptsnbLowerLimit(i_step, i_ptor, GetPredictorYptsnb(i_step, i_ptor))) return false;
										if (!SetPredictorYptsnbUpperLimit(i_step, i_ptor, GetPredictorYptsnb(i_step, i_ptor))) return false;
										if (!SetPredictorYptsnbIteration(i_step, i_ptor, 1)) return false;
									}
									else {
										if (!SetPredictorYptsnbLowerLimit(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "lowerlimit"))) return false;
										if (!SetPredictorYptsnbUpperLimit(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "upperlimit"))) return false;
										if (!SetPredictorYptsnbIteration(i_step, i_ptor, fileParams.GetAttributeDouble(nodeWindow, "iteration"))) return false;
									}
								}
								else if (nodeWindow->GetName() == "y_step") {
									if (!SetPredictorYstep(i_step, i_ptor, fileParams.GetDouble(nodeWindow))) return false;
								}
								else {
									fileParams.UnknownNode(nodeWindow);
								}
								nodeWindow = nodeWindow->GetNext();
							}
						}
						else if (nodeParam->GetName() == "criteria") {
							if (!SetPredictorCriteriaVector(i_step, i_ptor, fileParams.GetVectorString(nodeParam))) return false;
							SetPredictorCriteriaLock(i_step, i_ptor, fileParams.GetAttributeBool(nodeParam, "lock"));
							if (IsPredictorCriteriaLocked(i_step, i_ptor))
							{
								if (!SetPredictorCriteria(i_step, i_ptor, fileParams.GetString(nodeParam))) return false;
							}
						}
						else if (nodeParam->GetName() == "weight") {
							SetPredictorWeightLock(i_step, i_ptor, fileParams.GetAttributeBool(nodeParam, "lock"));
							if (IsPredictorWeightLocked(i_step, i_ptor)) {
								if (!SetPredictorWeight(i_step, i_ptor, fileParams.GetFloat(nodeParam))) return false;
								if (!SetPredictorWeightLowerLimit(i_step, i_ptor, GetPredictorWeight(i_step, i_ptor))) return false;
								if (!SetPredictorWeightUpperLimit(i_step, i_ptor, GetPredictorWeight(i_step, i_ptor))) return false;
								if (!SetPredictorWeightIteration(i_step, i_ptor, 1)) return false;
							}
							else {
								if (!SetPredictorWeightLowerLimit(i_step, i_ptor, fileParams.GetAttributeFloat(nodeParam, "lowerlimit"))) return false;
								if (!SetPredictorWeightUpperLimit(i_step, i_ptor, fileParams.GetAttributeFloat(nodeParam, "upperlimit"))) return false;
								if (!SetPredictorWeightIteration(i_step, i_ptor, fileParams.GetAttributeFloat(nodeParam, "iteration"))) return false;
							}
						}
						else {
							fileParams.UnknownNode(nodeParam);
						}
						nodeParam = nodeParam->GetNext();
					}
					i_ptor++;
				}
				else {
					fileParams.UnknownNode(nodeParamBlock);
				}
				nodeParamBlock = nodeParamBlock->GetNext();
			}
			i_step++;

			// Analog values
		}
		else if (nodeProcess->GetName() == "analog_values") {
			wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
			while (nodeParamBlock) {
				if (nodeParamBlock->GetName() == "predictand") {
					wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
					while (nodeParam) {
						if (nodeParam->GetName() == "station_id") {
							//if(!SetPredictandStationIds(GetFileStationIds(fileParams.GetFirstElementAttributeValueText("PredictandStationId", "value")))) return false;
							if (!SetPredictandStationIdsVector(fileParams.GetStationIdsVector(nodeParam))) return false;
						}
						else {
							fileParams.UnknownNode(nodeParam);
						}
						nodeParam = nodeParam->GetNext();
					}
				}
				else {
					fileParams.UnknownNode(nodeParamBlock);
				}
				nodeParamBlock = nodeParamBlock->GetNext();
			}

			// Forecast scores
		}
		else if (nodeProcess->GetName() == "analog_forecast_score") {
			wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
			while (nodeParamBlock) {
				if (nodeParamBlock->GetName() == "score") {
					if (!SetForecastScoreName(fileParams.GetString(nodeParamBlock))) return false;
				}
				else if (nodeParamBlock->GetName() == "threshold") {
					SetForecastScoreThreshold(fileParams.GetFloat(nodeParamBlock));
				}
				else if (nodeParamBlock->GetName() == "quantile") {
					SetForecastScoreQuantile(fileParams.GetFloat(nodeParamBlock));
				}
				else if (nodeParamBlock->GetName() == "postprocessing") {
					asLogError(_("The postptocessing is not yet fully implemented."));
				}
				else {
					fileParams.UnknownNode(nodeParamBlock);
				}
				nodeParamBlock = nodeParamBlock->GetNext();
			}

			// Forecast score final
		}
		else if (nodeProcess->GetName() == "analog_forecast_score_final") {
			wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
			while (nodeParamBlock) {
				if (nodeParamBlock->GetName() == "time_array") {
					if (!SetForecastScoreTimeArrayMode(fileParams.GetString(nodeParamBlock))) return false;
				}
				else {
					fileParams.UnknownNode(nodeParamBlock);
				}
				nodeParamBlock = nodeParamBlock->GetNext();
			}

		}
		else {
			fileParams.UnknownNode(nodeProcess);
		}

		nodeProcess = nodeProcess->GetNext();
	}

	// Set properties
	if (!PreprocessingPropertiesOk()) return false;
	SetSpatialWindowProperties();
	SetPreloadingProperties();

	// Set sizes
	SetSizes();

	// Check inputs and init parameters
	if (!InputsOK()) return false;
	InitValues();

	// Fixes
	FixTimeLimits();
	FixWeights();
	FixCoordinates();

	asLogMessage(_("Parameters file loaded."));

	return true;
}

bool asParametersOptimization::SetSpatialWindowProperties()
{
	for (int i_step = 0; i_step<GetStepsNb(); i_step++)
	{
		for (int i_ptor = 0; i_ptor<GetPredictorsNb(i_step); i_ptor++)
		{
			double Xshift = std::fmod(GetPredictorXminLowerLimit(i_step, i_ptor), GetPredictorXstep(i_step, i_ptor));
			if (Xshift<0) Xshift += GetPredictorXstep(i_step, i_ptor);
			if (!SetPredictorXshift(i_step, i_ptor, Xshift)) return false;

			double Yshift = std::fmod(GetPredictorYminLowerLimit(i_step, i_ptor), GetPredictorYstep(i_step, i_ptor));
			if (Yshift<0) Yshift += GetPredictorYstep(i_step, i_ptor);
			if (!SetPredictorYshift(i_step, i_ptor, Yshift)) return false;

			if (GetPredictorXptsnbLowerLimit(i_step, i_ptor) <= 1 || GetPredictorYptsnbLowerLimit(i_step, i_ptor) <= 1) SetPredictorFlatAllowed(i_step, i_ptor, asFLAT_ALLOWED);
			if (IsPredictorXptsnbLocked(i_step, i_ptor) && GetPredictorXptsnb(i_step, i_ptor) <= 1) SetPredictorFlatAllowed(i_step, i_ptor, asFLAT_ALLOWED);
			if (IsPredictorYptsnbLocked(i_step, i_ptor) && GetPredictorYptsnb(i_step, i_ptor) <= 1) SetPredictorFlatAllowed(i_step, i_ptor, asFLAT_ALLOWED);
		}
	}

	return true;
}

bool asParametersOptimization::SetPreloadingProperties()
{
	for (int i_step = 0; i_step<GetStepsNb(); i_step++)
	{
		for (int i_ptor = 0; i_ptor<GetPredictorsNb(i_step); i_ptor++)
		{
			// Set maximum extent
			if (NeedsPreloading(i_step, i_ptor))
			{
				// Set maximum extent
				if (!IsPredictorXminLocked(i_step, i_ptor))
				{
					SetPreloadXmin(i_step, i_ptor, GetPredictorXminLowerLimit(i_step, i_ptor));
				}
				else
				{
					SetPreloadXmin(i_step, i_ptor, GetPredictorXmin(i_step, i_ptor));
				}

				if (!IsPredictorYminLocked(i_step, i_ptor))
				{
					SetPreloadYmin(i_step, i_ptor, GetPredictorYminLowerLimit(i_step, i_ptor));
				}
				else
				{
					SetPreloadYmin(i_step, i_ptor, GetPredictorYmin(i_step, i_ptor));
				}

				if (!IsPredictorXptsnbLocked(i_step, i_ptor))
				{
					int Xbaseptsnb = abs(GetPredictorXminUpperLimit(i_step, i_ptor) - GetPredictorXminLowerLimit(i_step, i_ptor)) / GetPredictorXstep(i_step, i_ptor);
					SetPreloadXptsnb(i_step, i_ptor, Xbaseptsnb + GetPredictorXptsnbUpperLimit(i_step, i_ptor)); // No need to add +1
				}
				else
				{
					SetPreloadXptsnb(i_step, i_ptor, GetPredictorXptsnb(i_step, i_ptor));
				}

				if (!IsPredictorYptsnbLocked(i_step, i_ptor))
				{
					int Ybaseptsnb = abs(GetPredictorYminUpperLimit(i_step, i_ptor) - GetPredictorYminLowerLimit(i_step, i_ptor)) / GetPredictorYstep(i_step, i_ptor);
					SetPreloadYptsnb(i_step, i_ptor, Ybaseptsnb + GetPredictorYptsnbUpperLimit(i_step, i_ptor)); // No need to add +1
				}
				else
				{
					SetPreloadYptsnb(i_step, i_ptor, GetPredictorYptsnb(i_step, i_ptor));
				}
			}

			// Change predictor properties when preprocessing 
			if (NeedsPreprocessing(i_step, i_ptor))
			{
				if (GetPreprocessSize(i_step, i_ptor) == 1)
				{
					SetPredictorDatasetId(i_step, i_ptor, GetPreprocessDatasetId(i_step, i_ptor, 0));
					SetPredictorDataId(i_step, i_ptor, GetPreprocessDataId(i_step, i_ptor, 0));
					SetPredictorLevel(i_step, i_ptor, GetPreprocessLevel(i_step, i_ptor, 0));
					SetPredictorTimeHours(i_step, i_ptor, GetPreprocessTimeHours(i_step, i_ptor, 0));
				}
				else
				{
					SetPredictorDatasetId(i_step, i_ptor, "mix");
					SetPredictorDataId(i_step, i_ptor, "mix");
					SetPredictorLevel(i_step, i_ptor, 0);
					SetPredictorTimeHours(i_step, i_ptor, 0);
				}
			}

			// Set levels and time for preloading
			if (NeedsPreloading(i_step, i_ptor) && !NeedsPreprocessing(i_step, i_ptor))
			{
				if (!SetPreloadLevels(i_step, i_ptor, GetPredictorLevelVector(i_step, i_ptor))) return false;
				VectorDouble vTimeHours;
				for (double h = GetPredictorTimeHoursLowerLimit(i_step, i_ptor);
					h <= GetPredictorTimeHoursUpperLimit(i_step, i_ptor);
					h += GetPredictorTimeHoursIteration(i_step, i_ptor))
				{
					vTimeHours.push_back(h);
				}
				if (!SetPreloadTimeHours(i_step, i_ptor, vTimeHours)) return false;
			}
			else if (NeedsPreloading(i_step, i_ptor) && NeedsPreprocessing(i_step, i_ptor))
			{
				// Check the preprocessing method
				wxString method = GetPreprocessMethod(i_step, i_ptor);
				VectorFloat preprocLevels;
				VectorDouble preprocTimeHours;

				// Different actions depending on the preprocessing method.
				if (method.IsSameAs("Gradients"))
				{
					preprocLevels = GetPreprocessLevelVector(i_step, i_ptor, 0);

					for (double h = GetPreprocessTimeHoursLowerLimit(i_step, i_ptor, 0);
						h <= GetPreprocessTimeHoursUpperLimit(i_step, i_ptor, 0);
						h += GetPreprocessTimeHoursIteration(i_step, i_ptor, 0))
					{
						preprocTimeHours.push_back(h);
					}
				}
				else if (method.IsSameAs("HumidityFlux"))
				{
					preprocLevels = GetPreprocessLevelVector(i_step, i_ptor, 0);

					for (double h = GetPreprocessTimeHoursLowerLimit(i_step, i_ptor, 0);
						h <= GetPreprocessTimeHoursUpperLimit(i_step, i_ptor, 0);
						h += GetPreprocessTimeHoursIteration(i_step, i_ptor, 0))
					{
						preprocTimeHours.push_back(h);
					}
				}
				else if (method.IsSameAs("Multiplication") || method.IsSameAs("Multiply"))
				{
					preprocLevels = GetPreprocessLevelVector(i_step, i_ptor, 0);

					for (double h = GetPreprocessTimeHoursLowerLimit(i_step, i_ptor, 0);
						h <= GetPreprocessTimeHoursUpperLimit(i_step, i_ptor, 0);
						h += GetPreprocessTimeHoursIteration(i_step, i_ptor, 0))
					{
						preprocTimeHours.push_back(h);
					}
				}
				else if (method.IsSameAs("FormerHumidityIndex"))
				{
					asLogWarning(wxString::Format(_("The %s preprocessing method is not handled in the optimizer."), method));
					return false;
				}
				else
				{
					asLogWarning(wxString::Format(_("The %s preprocessing method is not yet handled with the preload option."), method));
				}

				if (!SetPreloadLevels(i_step, i_ptor, preprocLevels)) return false;
				if (!SetPreloadTimeHours(i_step, i_ptor, preprocTimeHours)) return false;
			}
		}
	}

	return true;
}

void asParametersOptimization::InitRandomValues()
{
    if(!m_timeArrayAnalogsIntervalDaysLocks)
    {
        m_timeArrayAnalogsIntervalDays = asTools::Random(m_timeArrayAnalogsIntervalDaysLowerLimit, m_timeArrayAnalogsIntervalDaysUpperLimit, m_timeArrayAnalogsIntervalDaysIteration);
    }

    for (int i=0; i<GetStepsNb(); i++)
    {
        if(!m_stepsLocks[i].AnalogsNumber)
        {
            SetAnalogsNumber(i,asTools::Random(m_stepsLowerLimit[i].AnalogsNumber, m_stepsUpperLimit[i].AnalogsNumber, m_stepsIteration[i].AnalogsNumber));
        }

        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            if (NeedsPreprocessing(i,j))
            {
                for (int k=0; k<GetPreprocessSize(i,j); k++)
                {
                    if(!m_stepsLocks[i].Predictors[j].PreprocessDataId[k])
                    {
                        int length = m_stepsVect[i].Predictors[j].PreprocessDataId[k].size();
                        int row = asTools::Random(0,length-1);
                        wxASSERT(m_stepsVect[i].Predictors[j].PreprocessDataId[k].size()>(unsigned)row);

                        SetPreprocessDataId(i,j,k, m_stepsVect[i].Predictors[j].PreprocessDataId[k][row]);
                    }

                    if(!m_stepsLocks[i].Predictors[j].PreprocessLevels[k])
                    {
                        int length = m_stepsVect[i].Predictors[j].PreprocessLevels[k].size();
                        int row = asTools::Random(0,length-1);
                        wxASSERT(m_stepsVect[i].Predictors[j].PreprocessLevels[k].size()>(unsigned)row);

                        SetPreprocessLevel(i,j,k, m_stepsVect[i].Predictors[j].PreprocessLevels[k][row]);
                    }

                    if(!m_stepsLocks[i].Predictors[j].PreprocessTimeHours[k])
                    {
                        SetPreprocessTimeHours(i,j,k, asTools::Random(m_stepsLowerLimit[i].Predictors[j].PreprocessTimeHours[k], m_stepsUpperLimit[i].Predictors[j].PreprocessTimeHours[k], m_stepsIteration[i].Predictors[j].PreprocessTimeHours[k]));
                    }
                }
            }
            else
            {
                if(!m_stepsLocks[i].Predictors[j].DataId)
                {
                    int length = m_stepsVect[i].Predictors[j].DataId.size();
                    int row = asTools::Random(0,length-1);
                    wxASSERT(m_stepsVect[i].Predictors[j].DataId.size()>(unsigned)row);

                    SetPredictorDataId(i,j, m_stepsVect[i].Predictors[j].DataId[row]);
                }

                if(!m_stepsLocks[i].Predictors[j].Level)
                {
                    int length = m_stepsVect[i].Predictors[j].Level.size();
                    int row = asTools::Random(0,length-1);
                    wxASSERT(m_stepsVect[i].Predictors[j].Level.size()>(unsigned)row);

                    SetPredictorLevel(i,j, m_stepsVect[i].Predictors[j].Level[row]);
                }

                if(!m_stepsLocks[i].Predictors[j].TimeHours)
                {
                    SetPredictorTimeHours(i,j, asTools::Random(m_stepsLowerLimit[i].Predictors[j].TimeHours, m_stepsUpperLimit[i].Predictors[j].TimeHours, m_stepsIteration[i].Predictors[j].TimeHours));
                }

            }

            if(!m_stepsLocks[i].Predictors[j].Xmin)
            {
                SetPredictorXmin(i,j, asTools::Random(m_stepsLowerLimit[i].Predictors[j].Xmin, m_stepsUpperLimit[i].Predictors[j].Xmin, m_stepsIteration[i].Predictors[j].Xmin));
            }

            if(!m_stepsLocks[i].Predictors[j].Xptsnb)
            {
                SetPredictorXptsnb(i,j, asTools::Random(m_stepsLowerLimit[i].Predictors[j].Xptsnb, m_stepsUpperLimit[i].Predictors[j].Xptsnb, m_stepsIteration[i].Predictors[j].Xptsnb));
            }

            if(!m_stepsLocks[i].Predictors[j].Ymin)
            {
                SetPredictorYmin(i,j, asTools::Random(m_stepsLowerLimit[i].Predictors[j].Ymin, m_stepsUpperLimit[i].Predictors[j].Ymin, m_stepsIteration[i].Predictors[j].Ymin));
            }

            if(!m_stepsLocks[i].Predictors[j].Yptsnb)
            {
                SetPredictorYptsnb(i,j, asTools::Random(m_stepsLowerLimit[i].Predictors[j].Yptsnb, m_stepsUpperLimit[i].Predictors[j].Yptsnb, m_stepsIteration[i].Predictors[j].Yptsnb));
            }

            if(!m_stepsLocks[i].Predictors[j].Weight)
            {
                SetPredictorWeight(i,j, asTools::Random(m_stepsLowerLimit[i].Predictors[j].Weight, m_stepsUpperLimit[i].Predictors[j].Weight, m_stepsIteration[i].Predictors[j].Weight));
            }

            if(!m_stepsLocks[i].Predictors[j].Criteria)
            {
                int length = m_stepsVect[i].Predictors[j].Criteria.size();
                int row = asTools::Random(0,length-1);
                wxASSERT(m_stepsVect[i].Predictors[j].Criteria.size()>(unsigned)row);

                SetPredictorCriteria(i,j, m_stepsVect[i].Predictors[j].Criteria[row]);
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
    if(!m_timeArrayAnalogsIntervalDaysLocks)
    {
        m_timeArrayAnalogsIntervalDays = wxMax( wxMin(m_timeArrayAnalogsIntervalDays, m_timeArrayAnalogsIntervalDaysUpperLimit), m_timeArrayAnalogsIntervalDaysLowerLimit);
    }
    wxASSERT(m_timeArrayAnalogsIntervalDays>0);

    for (int i=0; i<GetStepsNb(); i++)
    {
        if(!m_stepsLocks[i].AnalogsNumber)
        {
            SetAnalogsNumber(i, wxMax( wxMin(GetAnalogsNumber(i), m_stepsUpperLimit[i].AnalogsNumber), m_stepsLowerLimit[i].AnalogsNumber));
        }

        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            if(!GetPredictorGridType(i,j).IsSameAs ("Regular", false)) asThrowException(wxString::Format(_("asParametersOptimization::CheckRange is not ready to use on unregular grids (PredictorGridType = %s)"), GetPredictorGridType(i,j).c_str()));

            if (NeedsPreprocessing(i, j))
            {
                int preprocessSize = GetPreprocessSize(i, j);
                for (int k=0; k<preprocessSize; k++)
                {
                    if(!m_stepsLocks[i].Predictors[j].PreprocessTimeHours[k])
                    {
                        SetPreprocessTimeHours(i, j, k, wxMax( wxMin(GetPreprocessTimeHours(i,j,k), m_stepsUpperLimit[i].Predictors[j].PreprocessTimeHours[k]), m_stepsLowerLimit[i].Predictors[j].PreprocessTimeHours[k]));
                    }
                    SetPredictorTimeHours(i, j, 0);
                }
            }
            else
            {
                if(!m_stepsLocks[i].Predictors[j].TimeHours)
                {
                    SetPredictorTimeHours(i, j, wxMax( wxMin(GetPredictorTimeHours(i,j), m_stepsUpperLimit[i].Predictors[j].TimeHours), m_stepsLowerLimit[i].Predictors[j].TimeHours));
                }
            }

            // Check ranges
            if(!m_stepsLocks[i].Predictors[j].Xmin)
            {
                SetPredictorXmin(i, j, wxMax( wxMin(GetPredictorXmin(i,j), m_stepsUpperLimit[i].Predictors[j].Xmin), m_stepsLowerLimit[i].Predictors[j].Xmin));
            }
            if(!m_stepsLocks[i].Predictors[j].Xptsnb)
            {
                SetPredictorXptsnb(i, j, wxMax( wxMin(GetPredictorXptsnb(i,j), m_stepsUpperLimit[i].Predictors[j].Xptsnb), m_stepsLowerLimit[i].Predictors[j].Xptsnb));
            }

            if(!m_stepsLocks[i].Predictors[j].Ymin)
            {
                SetPredictorYmin(i, j, wxMax( wxMin(GetPredictorYmin(i,j), m_stepsUpperLimit[i].Predictors[j].Ymin), m_stepsLowerLimit[i].Predictors[j].Ymin));
            }
            if(!m_stepsLocks[i].Predictors[j].Yptsnb)
            {
                SetPredictorYptsnb(i, j, wxMax( wxMin(GetPredictorYptsnb(i,j), m_stepsUpperLimit[i].Predictors[j].Yptsnb), m_stepsLowerLimit[i].Predictors[j].Yptsnb));
            }
            if(!m_stepsLocks[i].Predictors[j].Weight)
            {
                SetPredictorWeight(i, j, wxMax( wxMin(GetPredictorWeight(i,j), m_stepsUpperLimit[i].Predictors[j].Weight), m_stepsLowerLimit[i].Predictors[j].Weight));
            }

            if(!m_stepsLocks[i].Predictors[j].Xmin || !m_stepsLocks[i].Predictors[j].Xptsnb)
            {
                if(GetPredictorXmin(i,j)+(GetPredictorXptsnb(i,j)-1)*GetPredictorXstep(i,j) > m_stepsUpperLimit[i].Predictors[j].Xmin+(m_stepsUpperLimit[i].Predictors[j].Xptsnb-1)*GetPredictorXstep(i,j))
                {
                    if(!m_stepsLocks[i].Predictors[j].Xptsnb)
                    {
                        SetPredictorXptsnb(i, j, (m_stepsUpperLimit[i].Predictors[j].Xmin-GetPredictorXmin(i,j))/GetPredictorXstep(i,j)+m_stepsUpperLimit[i].Predictors[j].Xptsnb); // Correct, no need of +1
                    }
                    else
                    {
                        SetPredictorXmin(i, j, m_stepsUpperLimit[i].Predictors[j].Xmin-GetPredictorXptsnb(i,j)*GetPredictorXstep(i,j));
                    }
                }
            }

            if(!m_stepsLocks[i].Predictors[j].Ymin || !m_stepsLocks[i].Predictors[j].Yptsnb)
            {
                if(GetPredictorYmin(i,j)+(GetPredictorYptsnb(i,j)-1)*GetPredictorYstep(i,j) > m_stepsUpperLimit[i].Predictors[j].Ymin+(m_stepsUpperLimit[i].Predictors[j].Yptsnb-1)*GetPredictorYstep(i,j))
                {
                    if(!m_stepsLocks[i].Predictors[j].Yptsnb)
                    {
                        SetPredictorYptsnb(i, j, (m_stepsUpperLimit[i].Predictors[j].Ymin-GetPredictorYmin(i,j))/GetPredictorYstep(i,j)+m_stepsUpperLimit[i].Predictors[j].Yptsnb); // Correct, no need of +1
                    }
                    else
                    {
                        SetPredictorYmin(i, j, m_stepsUpperLimit[i].Predictors[j].Ymin-GetPredictorYptsnb(i,j)*GetPredictorYstep(i,j));
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
    if(!m_timeArrayAnalogsIntervalDaysLocks)
    {
        if (m_timeArrayAnalogsIntervalDays>m_timeArrayAnalogsIntervalDaysUpperLimit) return false;
        if (m_timeArrayAnalogsIntervalDays<m_timeArrayAnalogsIntervalDaysLowerLimit) return false;
    }

    for (int i=0; i<GetStepsNb(); i++)
    {
        if (!m_stepsLocks[i].AnalogsNumber)
        {
            if (GetAnalogsNumber(i)>m_stepsUpperLimit[i].AnalogsNumber) return false;
            if (GetAnalogsNumber(i)<m_stepsLowerLimit[i].AnalogsNumber) return false;
        }

        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            if (NeedsPreprocessing(i,j))
            {
                for (int k=0; k<GetPreprocessSize(i,j); k++)
                {
                    if (!m_stepsLocks[i].Predictors[j].PreprocessTimeHours[k])
                    {
                        if (GetPreprocessTimeHours(i,j,k)<m_stepsLowerLimit[i].Predictors[j].PreprocessTimeHours[k]) return false;
                        if (GetPreprocessTimeHours(i,j,k)<m_stepsLowerLimit[i].Predictors[j].PreprocessTimeHours[k]) return false;
                    }
                }
            }
            else
            {
                if (!m_stepsLocks[i].Predictors[j].TimeHours)
                {
                    if (GetPredictorTimeHours(i,j)<m_stepsLowerLimit[i].Predictors[j].TimeHours) return false;
                    if (GetPredictorTimeHours(i,j)<m_stepsLowerLimit[i].Predictors[j].TimeHours) return false;
                }
            }

            if(!GetPredictorGridType(i,j).IsSameAs ("Regular", false)) asThrowException(wxString::Format(_("asParametersOptimization::CheckRange is not ready to use on unregular grids (PredictorGridType = %s)"), GetPredictorGridType(i,j).c_str()));

            // Check ranges
            if (!m_stepsLocks[i].Predictors[j].Xmin)
            {
                if (GetPredictorXmin(i,j)>m_stepsUpperLimit[i].Predictors[j].Xmin) return false;
                if (GetPredictorXmin(i,j)<m_stepsLowerLimit[i].Predictors[j].Xmin) return false;
            }
            if (!m_stepsLocks[i].Predictors[j].Xptsnb)
            {
                if (GetPredictorXptsnb(i,j)<m_stepsLowerLimit[i].Predictors[j].Xptsnb) return false;
                if (GetPredictorXptsnb(i,j)<m_stepsLowerLimit[i].Predictors[j].Xptsnb) return false;
            }
            if (!m_stepsLocks[i].Predictors[j].Ymin)
            {
                if (GetPredictorYmin(i,j)<m_stepsLowerLimit[i].Predictors[j].Ymin) return false;
                if (GetPredictorYmin(i,j)<m_stepsLowerLimit[i].Predictors[j].Ymin) return false;
            }
            if (!m_stepsLocks[i].Predictors[j].Yptsnb)
            {
                if (GetPredictorYptsnb(i,j)<m_stepsLowerLimit[i].Predictors[j].Yptsnb) return false;
                if (GetPredictorYptsnb(i,j)<m_stepsLowerLimit[i].Predictors[j].Yptsnb) return false;
            }
            if (!m_stepsLocks[i].Predictors[j].Weight)
            {
                if (GetPredictorWeight(i,j)<m_stepsLowerLimit[i].Predictors[j].Weight) return false;
                if (GetPredictorWeight(i,j)<m_stepsLowerLimit[i].Predictors[j].Weight) return false;
            }
            if (!m_stepsLocks[i].Predictors[j].Xmin ||
                !m_stepsLocks[i].Predictors[j].Xptsnb ||
                !m_stepsLocks[i].Predictors[j].Ymin ||
                !m_stepsLocks[i].Predictors[j].Yptsnb)
            {
                if(GetPredictorXmin(i,j)+GetPredictorXptsnb(i,j)*GetPredictorXstep(i,j) > m_stepsUpperLimit[i].Predictors[j].Xmin+m_stepsLowerLimit[i].Predictors[j].Xptsnb*GetPredictorXstep(i,j)) return false;
                if(GetPredictorYmin(i,j)+GetPredictorYptsnb(i,j)*GetPredictorYstep(i,j) > m_stepsUpperLimit[i].Predictors[j].Ymin+m_stepsLowerLimit[i].Predictors[j].Yptsnb*GetPredictorYstep(i,j)) return false;
            }
        }
    }

    return true;
}

bool asParametersOptimization::FixTimeLimits()
{
    SetSizes();

    double minHour = 200.0, maxHour = -50.0;
    for(int i=0;i<GetStepsNb();i++)
    {
        for(int j=0;j<GetPredictorsNb(i);j++)
        {
            if (NeedsPreprocessing(i,j))
            {
                for(int k=0; k<GetPreprocessSize(i,j); k++)
                {
                    minHour = wxMin(GetPreprocessTimeHoursLowerLimit(i, j, k), minHour);
                    maxHour = wxMax(GetPreprocessTimeHoursUpperLimit(i, j, k), maxHour);
                }
            }
            else
            {
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
    for (int i=0; i<GetStepsNb(); i++)
    {
        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            if (NeedsPreprocessing(i,j))
            {
                for (int k=0; k<GetPreprocessSize(i,j); k++)
                {
                    if (!m_stepsLocks[i].Predictors[j].PreprocessTimeHours[k])
                    {
                        if (m_stepsIteration[i].Predictors[j].PreprocessTimeHours[k]!=0)
                        {
                            float ratio = (float)GetPreprocessTimeHours(i,j,k)/(float)m_stepsIteration[i].Predictors[j].PreprocessTimeHours[k];
                            ratio = asTools::Round(ratio);
                            SetPreprocessTimeHours(i, j, k, ratio*m_stepsIteration[i].Predictors[j].PreprocessTimeHours[k]);
                        }
                    }
                }
            }
            else
            {
                if (!m_stepsLocks[i].Predictors[j].TimeHours)
                {
                    if (m_stepsIteration[i].Predictors[j].TimeHours!=0)
                    {
                        float ratio = (float)GetPredictorTimeHours(i,j)/(float)m_stepsIteration[i].Predictors[j].TimeHours;
                        ratio = asTools::Round(ratio);
                        SetPredictorTimeHours(i, j, ratio*m_stepsIteration[i].Predictors[j].TimeHours);
                    }
                }
            }
        }
    }
}

bool asParametersOptimization::FixWeights()
{
    for (int i=0; i<GetStepsNb(); i++)
    {
        // Sum the weights
        float totWeight = 0, totWeightLocked = 0;
        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            totWeight += GetPredictorWeight(i, j);

            if(IsPredictorWeightLocked(i, j))
            {
                totWeightLocked += GetPredictorWeight(i, j);
            }
        }

        // Check total of the locked weights
        if (totWeightLocked>1)
        {
            asLogError(wxString::Format(_("The sum of the locked weights of the analogy level number %d is higher than 1 (%f)."), i+1, totWeightLocked));
            return false;
        }
        float totWeightManageable = totWeight - totWeightLocked;

        // For every weights but the last
        float newSum = 0;
        for (int j=0; j<GetPredictorsNb(i)-1; j++)
        {
            if(!IsPredictorWeightLocked(i, j))
            {
                float precision = GetPredictorWeightIteration(i, j);
                float newWeight = GetPredictorWeight(i, j)/totWeightManageable;
                newWeight = precision*asTools::Round(newWeight*(1.0/precision));
                newSum += newWeight;

                SetPredictorWeight(i, j, newWeight);
            }
        }

        // Last weight: difference to 0
        float lastWeight = 1.0f - newSum - totWeightLocked;
        SetPredictorWeight(i, GetPredictorsNb(i)-1, lastWeight);
    }

    return true;
}

void asParametersOptimization::LockAll()
{
    m_timeArrayAnalogsIntervalDaysLocks = true;

    for (int i=0; i<GetStepsNb(); i++)
    {
        m_stepsLocks[i].AnalogsNumber = true;

        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            if (NeedsPreprocessing(i,j))
            {
                for (int k=0; k<GetPreprocessSize(i,j); k++)
                {
                    m_stepsLocks[i].Predictors[j].PreprocessDataId[k] = true;
                    m_stepsLocks[i].Predictors[j].PreprocessLevels[k] = true;
                    m_stepsLocks[i].Predictors[j].PreprocessTimeHours[k] = true;
                }
            }
            else
            {
                m_stepsLocks[i].Predictors[j].DataId = true;
                m_stepsLocks[i].Predictors[j].Level = true;
                m_stepsLocks[i].Predictors[j].TimeHours = true;
            }

            m_stepsLocks[i].Predictors[j].Xmin = true;
            m_stepsLocks[i].Predictors[j].Xptsnb = true;
            m_stepsLocks[i].Predictors[j].Ymin = true;
            m_stepsLocks[i].Predictors[j].Yptsnb = true;
            m_stepsLocks[i].Predictors[j].Weight = true;
            m_stepsLocks[i].Predictors[j].Criteria = true;
        }
    }

    return;
}

// TODO (Pascal#1#): Can be optimized by looping on the given vector (sorted first) instead
void asParametersOptimization::Unlock(VectorInt &indices)
{
    int counter = 0;
    int length = indices.size();

    if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
    {
        m_timeArrayAnalogsIntervalDaysLocks = false;
    }
    counter++;

    for (int i=0; i<GetStepsNb(); i++)
    {
        if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
        {
            m_stepsLocks[i].AnalogsNumber = false;
        }
        counter++;

        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            if(NeedsPreprocessing(i,j))
            {
                for (int k=0; k<GetPreprocessSize(i,j); k++)
                {
                    if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
                    {
                        m_stepsLocks[i].Predictors[j].PreprocessDataId[k] = false;
                    }
                    counter++;
                    if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
                    {
                        m_stepsLocks[i].Predictors[j].PreprocessLevels[k] = false;
                    }
                    counter++;
                    if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
                    {
                        m_stepsLocks[i].Predictors[j].PreprocessTimeHours[k] = false;
                    }
                    counter++;
                }
            }
            else
            {
                if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
                {
                    m_stepsLocks[i].Predictors[j].DataId = false;
                }
                counter++;
                if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
                {
                    m_stepsLocks[i].Predictors[j].Level = false;
                }
                counter++;
                if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
                {
                    m_stepsLocks[i].Predictors[j].TimeHours = false;
                }
                counter++;
            }

            if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
            {
                m_stepsLocks[i].Predictors[j].Xmin = false;
            }
            counter++;
            if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
            {
                m_stepsLocks[i].Predictors[j].Xptsnb = false;
            }
            counter++;
            if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
            {
                m_stepsLocks[i].Predictors[j].Ymin = false;
            }
            counter++;
            if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
            {
                m_stepsLocks[i].Predictors[j].Yptsnb = false;
            }
            counter++;
            if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
            {
                m_stepsLocks[i].Predictors[j].Weight = false;
            }
            counter++;
            if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
            {
                m_stepsLocks[i].Predictors[j].Criteria = false;
            }
            counter++;
        }
    }
}

int asParametersOptimization::GetVariablesNb()
{
    int counter = 0;

    if(!m_timeArrayAnalogsIntervalDaysLocks) counter++;

    for (int i=0; i<GetStepsNb(); i++)
    {
        if(!m_stepsLocks[i].AnalogsNumber) counter++;

        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            if(NeedsPreprocessing(i,j))
            {
                for (int k=0; k<GetPreprocessSize(i,j); k++)
                {
                    if(!m_stepsLocks[i].Predictors[j].PreprocessDataId[k]) counter++;
                    if(!m_stepsLocks[i].Predictors[j].PreprocessLevels[k]) counter++;
                    if(!m_stepsLocks[i].Predictors[j].PreprocessTimeHours[k]) counter++;
                }
            }
            else
            {
                if(!m_stepsLocks[i].Predictors[j].DataId) counter++;
                if(!m_stepsLocks[i].Predictors[j].Level) counter++;
                if(!m_stepsLocks[i].Predictors[j].TimeHours) counter++;
            }

            if(!m_stepsLocks[i].Predictors[j].Xmin) counter++;
            if(!m_stepsLocks[i].Predictors[j].Xptsnb) counter++;
            if(!m_stepsLocks[i].Predictors[j].Ymin) counter++;
            if(!m_stepsLocks[i].Predictors[j].Yptsnb) counter++;
            if(!m_stepsLocks[i].Predictors[j].Weight) counter++;
            if(!m_stepsLocks[i].Predictors[j].Criteria) counter++;
        }
    }

    return counter;
}
