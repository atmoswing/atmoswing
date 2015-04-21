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
    // Load from file
    if(filePath.IsEmpty())
    {
        asLogError(_("The given path to the parameters file is empty."));
        return false;
    }

    asFileParametersOptimization fileParams(filePath, asFile::ReadOnly);
    if(!fileParams.Open()) return false;

    if(!fileParams.GoToRootElement()) return false;

    // Get general parameters
    if(!fileParams.GoToFirstNodeWithPath("General")) return false;
    if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.CheckDeprecatedChildNode("Period")) return false;
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Archive Period")) return false;
    wxString archiveStart = fileParams.GetFirstElementAttributeValueText("Start", "value");
    wxString archiveEnd = fileParams.GetFirstElementAttributeValueText("End", "value");
    if (!archiveStart.IsEmpty() && !archiveEnd.IsEmpty())
    {
        SetArchiveStart(archiveStart);
        SetArchiveEnd(archiveEnd);
    }
    else
    {
        if(!SetArchiveYearStart(fileParams.GetFirstElementAttributeValueInt("YearStart", "value"))) return false;
        if(!SetArchiveYearEnd(fileParams.GetFirstElementAttributeValueInt("YearEnd", "value"))) return false;
    }
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Calibration Period")) return false;
    wxString calibStart = fileParams.GetFirstElementAttributeValueText("Start", "value");
    wxString calibEnd = fileParams.GetFirstElementAttributeValueText("End", "value");
    if (!calibStart.IsEmpty() && !calibEnd.IsEmpty())
    {
        SetCalibrationStart(calibStart);
        SetCalibrationEnd(calibEnd);
    }
    else
    {
        if(!SetCalibrationYearStart(fileParams.GetFirstElementAttributeValueInt("YearStart", "value"))) return false;
        if(!SetCalibrationYearEnd(fileParams.GetFirstElementAttributeValueInt("YearEnd", "value"))) return false;
    }
    if(!fileParams.GoANodeBack()) return false;

    if(fileParams.GoToChildNodeWithAttributeValue("name", "Validation Period", asHIDE_WARNINGS))
    {
        if(!SetValidationYearsVector(GetFileParamIntVector(fileParams, "Years"))) return false;
        if(!fileParams.GoANodeBack()) return false;
    }

    if(fileParams.GoToChildNodeWithAttributeValue("name", "Time Properties", asHIDE_WARNINGS))
    {
        if(!SetTimeArrayTargetTimeStepHours(fileParams.GetFirstElementAttributeValueDouble("TimeStepHours", "value"))) return false;
        if(!SetTimeArrayAnalogsTimeStepHours(fileParams.GetFirstElementAttributeValueDouble("TimeStepHours", "value"))) return false;
        if(!fileParams.GoANodeBack()) return false;
    }
    else
    {
        if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array Target")) return false;
        if(!SetTimeArrayTargetTimeStepHours(fileParams.GetFirstElementAttributeValueDouble("TimeStepHours", "value"))) return false;
        if(!fileParams.GoANodeBack()) return false;
        if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array Analogs")) return false;
        if(!SetTimeArrayAnalogsTimeStepHours(fileParams.GetFirstElementAttributeValueDouble("TimeStepHours", "value"))) return false;
        if(!fileParams.GoANodeBack()) return false;
    }

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array Target")) return false;
    if(!SetTimeArrayTargetMode(fileParams.GetFirstElementAttributeValueText("TimeArrayMode", "value"))) return false;
    if(GetTimeArrayTargetMode().IsSameAs("PredictandThresholds"))
    {
        if(!SetTimeArrayTargetPredictandSerieName(fileParams.GetFirstElementAttributeValueText("PredictandSerieName", "value"))) return false;
        if(!SetTimeArrayTargetPredictandMinThreshold(fileParams.GetFirstElementAttributeValueFloat("PredictandMinThreshold", "value"))) return false;
        if(!SetTimeArrayTargetPredictandMaxThreshold(fileParams.GetFirstElementAttributeValueFloat("PredictandMaxThreshold", "value"))) return false;
    }
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array Analogs")) return false;
    if(!SetTimeArrayAnalogsMode(fileParams.GetFirstElementAttributeValueText("TimeArrayMode", "value"))) return false;
    if(!SetTimeArrayAnalogsExcludeDays(fileParams.GetFirstElementAttributeValueInt("ExcludeDays", "value"))) return false;

    SetTimeArrayAnalogsIntervalDaysLock(fileParams.GetFirstElementAttributeValueBool("IntervalDays", "lock", true));
    if(IsTimeArrayAnalogsIntervalDaysLocked())
    {
        if(!SetTimeArrayAnalogsIntervalDays(fileParams.GetFirstElementAttributeValueInt("IntervalDays", "value"))) return false;
        if(!SetTimeArrayAnalogsIntervalDaysLowerLimit(GetTimeArrayAnalogsIntervalDays())) return false;
        if(!SetTimeArrayAnalogsIntervalDaysUpperLimit(GetTimeArrayAnalogsIntervalDays())) return false;
        if(!SetTimeArrayAnalogsIntervalDaysIteration(1)) return false; // must be >0
    }
    else
    {
        if(!SetTimeArrayAnalogsIntervalDaysLowerLimit(fileParams.GetFirstElementAttributeValueInt("IntervalDays", "lowerlimit"))) return false;
        if(!SetTimeArrayAnalogsIntervalDaysUpperLimit(fileParams.GetFirstElementAttributeValueInt("IntervalDays", "upperlimit"))) return false;
        if(!SetTimeArrayAnalogsIntervalDaysIteration(fileParams.GetFirstElementAttributeValueInt("IntervalDays", "iteration"))) return false;
    }
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoANodeBack()) return false;

    // Get Analogs Dates processs
    int i_step = 0;
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Analogs Dates")) return false;

    while(true)
    {
        AddStep();

        if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
        if(!fileParams.GoANodeBack()) return false;

        if(!fileParams.GoToChildNodeWithAttributeValue("name", "Method Name")) return false;
        if(!SetMethodName(i_step, fileParams.GetFirstElementAttributeValueText("MethodName", "value"))) return false;
        if(!fileParams.GoANodeBack()) return false;

        if(!fileParams.GoToChildNodeWithAttributeValue("name", "Analogs Number")) return false;
        SetAnalogsNumberLock(i_step, fileParams.GetFirstElementAttributeValueBool("AnalogsNumber", "lock", true));
        if(IsAnalogsNumberLocked(i_step))
        {
            if(!SetAnalogsNumber(i_step, fileParams.GetFirstElementAttributeValueInt("AnalogsNumber", "value"))) return false;
            if(!SetAnalogsNumberLowerLimit(i_step, GetAnalogsNumber(i_step))) return false;
            if(!SetAnalogsNumberUpperLimit(i_step, GetAnalogsNumber(i_step))) return false;
            if(!SetAnalogsNumberIteration(i_step, 1)) return false; // must be >0
        }
        else
        {
            if(!SetAnalogsNumberLowerLimit(i_step, fileParams.GetFirstElementAttributeValueInt("AnalogsNumber", "lowerlimit"))) return false;
            if(!SetAnalogsNumberUpperLimit(i_step, fileParams.GetFirstElementAttributeValueInt("AnalogsNumber", "upperlimit"))) return false;
            if(!SetAnalogsNumberIteration(i_step, fileParams.GetFirstElementAttributeValueInt("AnalogsNumber", "iteration"))) return false;
        }
        if(!fileParams.GoANodeBack()) return false;

        // Get first data
        if(!fileParams.GoToFirstNodeWithPath("Data")) return false;
        bool dataOver = false;
        int i_ptor = 0;
        while(!dataOver)
        {
            wxString predictorNature = fileParams.GetThisElementAttributeValueText("name", "value");

            if(predictorNature.IsSameAs("Predictor", false))
            {
                if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
                if(!fileParams.GoANodeBack()) return false;

                if(fileParams.GoToChildNodeWithAttributeValue("name", "Preprocessing", asHIDE_WARNINGS))
                {
                    SetPreprocess(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Preprocess", "value"));
                    if(NeedsPreprocessing(i_step, i_ptor))
                    {
                        asLogError(_("Preprocessing option is not coherent."));
                        return false;
                    }
                    if(!fileParams.GoANodeBack()) return false;
                }

                if(fileParams.GoToChildNodeWithAttributeValue("name", "Preload", asHIDE_WARNINGS))
                {
                    SetPreload(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Preload", "value"));
                    if(!fileParams.GoANodeBack()) return false;
                }
                else
                {
                    SetPreload(i_step, i_ptor, false);
                }

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Data")) return false;
                if(!SetPredictorDatasetId(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("DatasetId", "value"))) return false;
                if(!SetPredictorDataIdVector(i_step, i_ptor, GetFileParamStringVector(fileParams, "DataId"))) return false;
                SetPredictorDataIdLock(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("DataId", "lock", true));
                if(IsPredictorDataIdLocked(i_step, i_ptor))
                {
                    if(!SetPredictorDataId(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("DataId", "value"))) return false;
                }
                if(!fileParams.GoANodeBack()) return false;

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Level")) return false;
                if(!SetPredictorLevelVector(i_step, i_ptor, GetFileParamFloatVector(fileParams, "Level"))) return false;
                SetPredictorLevelLock(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Level", "lock", true));
                if (NeedsPreloading(i_step, i_ptor))
                {
                    if(!SetPreloadLevels(i_step, i_ptor, GetFileParamFloatVector(fileParams, "Level"))) return false;
                }
                if(IsPredictorLevelLocked(i_step, i_ptor))
                {
                    if(!SetPredictorLevel(i_step, i_ptor, fileParams.GetFirstElementAttributeValueFloat("Level", "value"))) return false;
                }
                if(!fileParams.GoANodeBack()) return false;

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Frame")) return false;
                SetPredictorTimeHoursLock(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("TimeHours", "lock", true));
                if(IsPredictorTimeHoursLocked(i_step, i_ptor))
                {
                    if(!SetPredictorTimeHours(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("TimeHours", "value"))) return false;
                    VectorDouble vTimeHours;
                    vTimeHours.push_back(fileParams.GetFirstElementAttributeValueDouble("TimeHours", "value"));
                    if(!SetPreloadTimeHours(i_step, i_ptor, vTimeHours)) return false;
                    if(!SetPredictorTimeHoursLowerLimit(i_step, i_ptor, GetPredictorTimeHours(i_step, i_ptor))) return false;
                    if(!SetPredictorTimeHoursUpperLimit(i_step, i_ptor, GetPredictorTimeHours(i_step, i_ptor))) return false;
                    if(!SetPredictorTimeHoursIteration(i_step, i_ptor, 6)) return false; // must be >0
                }
                else
                {
                    if(!SetPredictorTimeHoursLowerLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("TimeHours", "lowerlimit"))) return false;
                    if(!SetPredictorTimeHoursUpperLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("TimeHours", "upperlimit"))) return false;
                    if(!SetPredictorTimeHoursIteration(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("TimeHours", "iteration"))) return false;
                    VectorDouble vTimeHours;
                    for (double h=GetPredictorTimeHoursLowerLimit(i_step, i_ptor);
                         h<=GetPredictorTimeHoursUpperLimit(i_step, i_ptor);
                         h+=GetPredictorTimeHoursIteration(i_step, i_ptor))
                    {
                        vTimeHours.push_back(h);
                    }
                    if(!SetPreloadTimeHours(i_step, i_ptor, vTimeHours)) return false;
                }
                if(!fileParams.GoANodeBack()) return false;

            }
            else if(predictorNature.IsSameAs("Predictor Preprocessed", false))
            {
                if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
                if(!fileParams.GoANodeBack()) return false;

                if(fileParams.GoToChildNodeWithAttributeValue("name", "Preload", asHIDE_WARNINGS))
                {
                    SetPreload(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Preload", "value"));
                    if(!fileParams.GoANodeBack()) return false;
                }
                else
                {
                    SetPreload(i_step, i_ptor, false);
                }

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Preprocessing")) return false;
                SetPreprocess(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Preprocess", "value"));
                if(!SetPreprocessMethod(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("PreprocessMethod", "value"))) return false;
                if(!NeedsPreprocessing(i_step, i_ptor))
                {
                    asLogError(_("Preprocessing option is not coherent."));
                    return false;
                }

                if(!fileParams.GoToFirstNodeWithPath("SubData")) return false;
                int i_dataset = 0;
                bool preprocessDataOver = false;
                while(!preprocessDataOver)
                {
                    if(!SetPreprocessDatasetId(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueText("PreprocessDatasetId", "value"))) return false;
                    if(!SetPreprocessDataIdVector(i_step, i_ptor, i_dataset, GetFileParamStringVector(fileParams, "PreprocessDataId"))) return false;
                    SetPreprocessDataIdLock(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueBool("PreprocessDataId", "lock", true));
                    if(IsPreprocessDataIdLocked(i_step, i_ptor, i_dataset))
                    {
                        if(!SetPreprocessDataId(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueText("PreprocessDataId", "value"))) return false;
                    }
                    else
                    {
                        // Initialize to ensure correct array sizes
                        if(!SetPreprocessDataId(i_step, i_ptor, i_dataset, GetFileParamStringVector(fileParams, "PreprocessDataId")[0])) return false;
                    }

                    if(!SetPreprocessLevelVector(i_step, i_ptor, i_dataset, GetFileParamFloatVector(fileParams, "PreprocessLevel"))) return false;
                    SetPreprocessLevelLock(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueBool("PreprocessLevel", "lock", true));
                    if(IsPreprocessLevelLocked(i_step, i_ptor, i_dataset))
                    {
                        if(!SetPreprocessLevel(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueFloat("PreprocessLevel", "value"))) return false;
                    }
                    else
                    {
                        // Initialize to ensure correct array sizes
                        if(!SetPreprocessLevel(i_step, i_ptor, i_dataset, GetFileParamFloatVector(fileParams, "PreprocessLevel")[0])) return false;
                    }

                    SetPreprocessTimeHoursLock(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueBool("PreprocessTimeHours", "lock", true));
                    if(IsPreprocessTimeHoursLocked(i_step, i_ptor, i_dataset))
                    {
                        if(!SetPreprocessTimeHours(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueDouble("PreprocessTimeHours", "value"))) return false;
                        if(!SetPreprocessTimeHoursLowerLimit(i_step, i_ptor, i_dataset, GetPreprocessTimeHours(i_step, i_ptor, i_dataset))) return false;
                        if(!SetPreprocessTimeHoursUpperLimit(i_step, i_ptor, i_dataset, GetPreprocessTimeHours(i_step, i_ptor, i_dataset))) return false;
                        if(!SetPreprocessTimeHoursIteration(i_step, i_ptor, i_dataset, 6)) return false; // must be >0
                    }
                    else
                    {
                        if(!SetPreprocessTimeHoursLowerLimit(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueDouble("PreprocessTimeHours", "lowerlimit"))) return false;
                        if(!SetPreprocessTimeHoursUpperLimit(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueDouble("PreprocessTimeHours", "upperlimit"))) return false;
                        if(!SetPreprocessTimeHoursIteration(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueDouble("PreprocessTimeHours", "iteration"))) return false;
                        // Initialize to ensure correct array sizes
                        if(!SetPreprocessTimeHours(i_step, i_ptor, i_dataset, GetPreprocessTimeHoursLowerLimit(i_step, i_ptor, i_dataset))) return false;
                    }

                    if(fileParams.GoToNextSameNode())
                    {
                        i_dataset++;
                    }
                    else
                    {
                        preprocessDataOver = true;
                    }
                }

                if(NeedsPreloading(i_step, i_ptor))
                {
                    // Check the preprocessing method
                    wxString method = GetPreprocessMethod(i_step, i_ptor);
                    VectorFloat preprocLevels;
                    VectorDouble preprocTimeHours;
                    int preprocSize = GetPreprocessSize(i_step, i_ptor);

                    // Check that the data ID is locked
                    for (int i_preproc=0; i_preproc<preprocSize; i_preproc++)
                    {
                        if(!IsPreprocessDataIdLocked(i_step, i_ptor, i_preproc))
                        {
                            asLogError(_("The preprocess DataId option unlocked is not compatible with the preload option."));
                            return false;
                        }
                    }

                    wxASSERT(GetPreprocessTimeHoursIteration(i_step, i_ptor, 0)>0);

                    // Different actions depending on the preprocessing method.
                    if (method.IsSameAs("Gradients"))
                    {
                        if (preprocSize!=1)
                        {
                            asLogError(wxString::Format(_("The size of the provided predictors (%d) does not match the requirements (1) in the preprocessing Gradients method."), preprocSize));
                            return false;
                        }
                        preprocLevels = GetPreprocessLevelVector(i_step, i_ptor, 0);

                        for (double h=GetPreprocessTimeHoursLowerLimit(i_step, i_ptor, 0);
                             h<=GetPreprocessTimeHoursUpperLimit(i_step, i_ptor, 0);
                             h+=GetPreprocessTimeHoursIteration(i_step, i_ptor, 0))
                        {
                            preprocTimeHours.push_back(h);
                        }
                    }
                    else if (method.IsSameAs("HumidityFlux"))
                    {
                        if (preprocSize!=4)
                        {
                            asLogError(wxString::Format(_("The size of the provided predictors (%d) does not match the requirements (4) in the preprocessing HumidityFlux method."), preprocSize));
                            return false;
                        }
                        preprocLevels = GetPreprocessLevelVector(i_step, i_ptor, 0);

                        for (double h=GetPreprocessTimeHoursLowerLimit(i_step, i_ptor, 0);
                             h<=GetPreprocessTimeHoursUpperLimit(i_step, i_ptor, 0);
                             h+=GetPreprocessTimeHoursIteration(i_step, i_ptor, 0))
                        {
                            preprocTimeHours.push_back(h);
                        }
                    }
                    else if (method.IsSameAs("Multiplication") || method.IsSameAs("Multiply") || method.IsSameAs("HumidityIndex"))
                    {
                        if (preprocSize!=2)
                        {
                            asLogError(wxString::Format(_("The size of the provided predictors (%d) does not match the requirements (2) in the preprocessing Multiply method."), preprocSize));
                            return false;
                        }
                        preprocLevels = GetPreprocessLevelVector(i_step, i_ptor, 0);

                        for (double h=GetPreprocessTimeHoursLowerLimit(i_step, i_ptor, 0);
                             h<=GetPreprocessTimeHoursUpperLimit(i_step, i_ptor, 0);
                             h+=GetPreprocessTimeHoursIteration(i_step, i_ptor, 0))
                        {
                            preprocTimeHours.push_back(h);
                        }
                    }
                    else
                    {
                        asLogWarning(wxString::Format(_("The %s preprocessing method is not yet handled with the preload option."), method.c_str()));

                        for (int i_preproc=0; i_preproc<preprocSize; i_preproc++)
                        {
                            if(!IsPreprocessLevelLocked(i_step, i_ptor, i_preproc))
                            {
                                asLogError(_("The preprocess Level option unlocked is not compatible with the preload option."));
                                return false;
                            }
                            if(!IsPreprocessTimeHoursLocked(i_step, i_ptor, i_preproc))
                            {
                                asLogError(_("The preprocess TimeHours option unlocked is not compatible with the preload option."));
                                return false;
                            }
                        }
                    }

                    if(!SetPreloadLevels(i_step, i_ptor, preprocLevels)) return false;
                    if(!SetPreloadTimeHours(i_step, i_ptor, preprocTimeHours)) return false;
                }

                // Set data for predictor
                SetPredictorDatasetId(i_step, i_ptor, "mix");
                SetPredictorDataId(i_step, i_ptor, "mix");
                SetPredictorLevel(i_step, i_ptor, 0);
                SetPredictorTimeHours(i_step, i_ptor, 0);

                if(!fileParams.GoANodeBack()) return false;
                if(!fileParams.GoANodeBack()) return false;
            }
            else
            {
                asThrowException(_("Preprocessing option not correctly defined in the parameters file."));
            }

            if(!fileParams.GoToChildNodeWithAttributeValue("name", "Area", asHIDE_WARNINGS))
            {
                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Area Moving")) return false;
            }
            if(!SetPredictorGridType(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("GridType", "value", "Regular"))) return false;
            SetPredictorXminLock(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Xmin", "lock", true));
            if(IsPredictorXminLocked(i_step, i_ptor))
            {
                if(!SetPredictorXmin(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Xmin", "value"))) return false;
                if(!SetPredictorXminLowerLimit(i_step, i_ptor, GetPredictorXmin(i_step, i_ptor))) return false;
                if(!SetPredictorXminUpperLimit(i_step, i_ptor, GetPredictorXmin(i_step, i_ptor))) return false;
                if(!SetPredictorXminIteration(i_step, i_ptor, 1)) return false; // must be >0
            }
            else
            {
                if(!SetPredictorXminLowerLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Xmin", "lowerlimit"))) return false;
                if(!SetPredictorXminUpperLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Xmin", "upperlimit"))) return false;
                if(!SetPredictorXminIteration(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Xmin", "iteration"))) return false;
            }

            SetPredictorXptsnbLock(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Xptsnb", "lock", true));
            if(IsPredictorXptsnbLocked(i_step, i_ptor))
            {
                if(!SetPredictorXptsnb(i_step, i_ptor, fileParams.GetFirstElementAttributeValueInt("Xptsnb", "value"))) return false;
                if(!SetPredictorXptsnbLowerLimit(i_step, i_ptor, GetPredictorXptsnb(i_step, i_ptor))) return false;
                if(!SetPredictorXptsnbUpperLimit(i_step, i_ptor, GetPredictorXptsnb(i_step, i_ptor))) return false;
                if(!SetPredictorXptsnbIteration(i_step, i_ptor, 1)) return false; // must be >0
            }
            else
            {
                if(!SetPredictorXptsnbLowerLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueInt("Xptsnb", "lowerlimit"))) return false;
                if (GetPredictorXptsnbLowerLimit(i_step, i_ptor)==0)
                {
                    SetPredictorXptsnbLowerLimit(i_step, i_ptor, 1);
                }
                if(!SetPredictorXptsnbUpperLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueInt("Xptsnb", "upperlimit"))) return false;
                if(!SetPredictorXptsnbIteration(i_step, i_ptor, fileParams.GetFirstElementAttributeValueInt("Xptsnb", "iteration"))) return false;
            }

            if(!SetPredictorXstep(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Xstep", "value"))) return false;
            double Ushift = fmod(GetPredictorXminLowerLimit(i_step, i_ptor), GetPredictorXstep(i_step, i_ptor));
            if (Ushift<0) Ushift += GetPredictorXstep(i_step, i_ptor);
            if(!SetPredictorUshift(i_step, i_ptor, Ushift)) return false;

            SetPredictorYminLock(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Ymin", "lock", true));
            if(IsPredictorYminLocked(i_step, i_ptor))
            {
                if(!SetPredictorYmin(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Ymin", "value"))) return false;
                if(!SetPredictorYminLowerLimit(i_step, i_ptor, GetPredictorYmin(i_step, i_ptor))) return false;
                if(!SetPredictorYminUpperLimit(i_step, i_ptor, GetPredictorYmin(i_step, i_ptor))) return false;
                if(!SetPredictorYminIteration(i_step, i_ptor, 1)) return false; // must be >0
            }
            else
            {
                if(!SetPredictorYminLowerLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Ymin", "lowerlimit"))) return false;
                if(!SetPredictorYminUpperLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Ymin", "upperlimit"))) return false;
                if(!SetPredictorYminIteration(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Ymin", "iteration"))) return false;
            }

            SetPredictorYptsnbLock(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Yptsnb", "lock", true));
            if(IsPredictorYptsnbLocked(i_step, i_ptor))
            {
                if(!SetPredictorYptsnb(i_step, i_ptor, fileParams.GetFirstElementAttributeValueInt("Yptsnb", "value"))) return false;
                if(!SetPredictorYptsnbLowerLimit(i_step, i_ptor, GetPredictorYptsnb(i_step, i_ptor))) return false;
                if(!SetPredictorYptsnbUpperLimit(i_step, i_ptor, GetPredictorYptsnb(i_step, i_ptor))) return false;
                if(!SetPredictorYptsnbIteration(i_step, i_ptor, 1)) return false; // must be >0
            }
            else
            {
                if(!SetPredictorYptsnbLowerLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueInt("Yptsnb", "lowerlimit"))) return false;
                if (GetPredictorYptsnbLowerLimit(i_step, i_ptor)==0) SetPredictorYptsnbLowerLimit(i_step, i_ptor, 1);
                if(!SetPredictorYptsnbUpperLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueInt("Yptsnb", "upperlimit"))) return false;
                if(!SetPredictorYptsnbIteration(i_step, i_ptor, fileParams.GetFirstElementAttributeValueInt("Yptsnb", "iteration"))) return false;
            }

            if(!SetPredictorYstep(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Ystep", "value"))) return false;
            double Vshift = fmod(GetPredictorYminLowerLimit(i_step, i_ptor), GetPredictorYstep(i_step, i_ptor));
            if (Vshift<0) Vshift += GetPredictorYstep(i_step, i_ptor);
            if(!SetPredictorVshift(i_step, i_ptor, Vshift)) return false;

            if (GetPredictorXptsnbLowerLimit(i_step, i_ptor)<=1 || GetPredictorYptsnbLowerLimit(i_step, i_ptor)<=1) SetPredictorFlatAllowed(i_step, i_ptor, asFLAT_ALLOWED);
            if (IsPredictorXptsnbLocked(i_step, i_ptor) && GetPredictorXptsnb(i_step, i_ptor)<=1) SetPredictorFlatAllowed(i_step, i_ptor, asFLAT_ALLOWED);
            if (IsPredictorYptsnbLocked(i_step, i_ptor) && GetPredictorYptsnb(i_step, i_ptor)<=1) SetPredictorFlatAllowed(i_step, i_ptor, asFLAT_ALLOWED);
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
                    int Ubaseptsnb = abs(GetPredictorXminUpperLimit(i_step, i_ptor)-GetPredictorXminLowerLimit(i_step, i_ptor))/GetPredictorXstep(i_step, i_ptor);
                    SetPreloadXptsnb(i_step, i_ptor, Ubaseptsnb+GetPredictorXptsnbUpperLimit(i_step, i_ptor)); // No need to add +1
                }
                else
                {
                    SetPreloadXptsnb(i_step, i_ptor, GetPredictorXptsnb(i_step, i_ptor));
                }

                if (!IsPredictorYptsnbLocked(i_step, i_ptor))
                {
                    int Vbaseptsnb = abs(GetPredictorYminUpperLimit(i_step, i_ptor)-GetPredictorYminLowerLimit(i_step, i_ptor))/GetPredictorYstep(i_step, i_ptor);
                    SetPreloadYptsnb(i_step, i_ptor, Vbaseptsnb+GetPredictorYptsnbUpperLimit(i_step, i_ptor)); // No need to add +1
                }
                else
                {
                    SetPreloadYptsnb(i_step, i_ptor, GetPredictorYptsnb(i_step, i_ptor));
                }
            }
            if(!fileParams.GoANodeBack()) return false;

            if(!fileParams.GoToChildNodeWithAttributeValue("name", "Criteria")) return false;
            if(!SetPredictorCriteriaVector(i_step, i_ptor, GetFileParamStringVector(fileParams, "Criteria"))) return false;
            SetPredictorCriteriaLock(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Criteria", "lock", true));
            if(IsPredictorCriteriaLocked(i_step, i_ptor))
            {
                if(!SetPredictorCriteria(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("Criteria", "value"))) return false;
            }
            if(!fileParams.GoANodeBack()) return false;

            // Fix the criteria if preprocessed
            if (NeedsPreprocessing(i_step, i_ptor))
            {
                if(GetPreprocessMethod(i_step, i_ptor).IsSameAs("Gradients") && GetPredictorCriteria(i_step, i_ptor).IsSameAs("S1"))
                {
                    SetPredictorCriteria(i_step, i_ptor, "S1grads");
                }
            }

            if(!fileParams.GoToChildNodeWithAttributeValue("name", "Weight")) return false;
            SetPredictorWeightLock(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Weight", "lock", true));
            if(IsPredictorWeightLocked(i_step, i_ptor))
            {
                if(!SetPredictorWeight(i_step, i_ptor, fileParams.GetFirstElementAttributeValueFloat("Weight", "value"))) return false;
                if(!SetPredictorWeightLowerLimit(i_step, i_ptor, GetPredictorWeight(i_step, i_ptor))) return false;
                if(!SetPredictorWeightUpperLimit(i_step, i_ptor, GetPredictorWeight(i_step, i_ptor))) return false;
                if(!SetPredictorWeightIteration(i_step, i_ptor, 0.1f)) return false; // must be >0
            }
            else
            {
                if(!SetPredictorWeightLowerLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueFloat("Weight", "lowerlimit"))) return false;
                if(!SetPredictorWeightUpperLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueFloat("Weight", "upperlimit"))) return false;
                if(!SetPredictorWeightIteration(i_step, i_ptor, fileParams.GetFirstElementAttributeValueFloat("Weight", "iteration"))) return false;
            }
            if(!fileParams.GoANodeBack()) return false;

            if(fileParams.GoToNextSameNode())
            {
                i_ptor++;
                AddPredictor(i_step);
                AddPredictorIteration(m_stepsIteration[i_step]);
                AddPredictorUpperLimit(m_stepsUpperLimit[i_step]);
                AddPredictorLowerLimit(m_stepsLowerLimit[i_step]);
                AddPredictorLocks(m_stepsLocks[i_step]);
                AddPredictorVect(m_stepsVect[i_step]);
            }
            else
            {
                dataOver = true;
            }
        }
        if(!fileParams.GoANodeBack()) return false;

        // Find the next analogs date block
        if (!fileParams.GoToNextSameNodeWithAttributeValue("name", "Analogs Dates", asHIDE_WARNINGS)) break;

        i_step++;
    }
    if(!fileParams.GoANodeBack()) return false;

    // Get first Analogs Dates process
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Analogs Values")) return false;
    if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Predictand")) return false;
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Database")) return false;
    if(!SetPredictandStationIds(GetFileStationIds(fileParams.GetFirstElementAttributeValueText("PredictandStationId", "value")))) return false;
    if(!SetPredictandTimeHours(fileParams.GetFirstElementAttributeValueDouble("PredictandTimeHours", "value", 0.0))) return false;
    if(!fileParams.GoANodeBack()) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoANodeBack()) return false;

    // Get Analogs Forecast Scores process
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Analogs Forecast Scores")) return false;
    if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Method")) return false;
    wxString forecastScore = fileParams.GetFirstElementAttributeValueText("Name", "value");
    if (forecastScore.IsSameAs("RankHistogram", false) || forecastScore.IsSameAs("RankHistogramReliability", false))
    {
        asLogError(_("The rank histogram can only be processed in the 'all scores' evalution method."));
        return false;
    }
    if(!SetForecastScoreName(forecastScore)) return false;
    SetForecastScoreThreshold(fileParams.GetFirstElementAttributeValueFloat("Threshold", "value", NaNFloat));
    SetForecastScorePercentile(fileParams.GetFirstElementAttributeValueFloat("Percentile", "value", NaNFloat));
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.CheckDeprecatedChildNode("Analogs Number")) return false;

    if(fileParams.GoToChildNodeWithAttributeValue("name", "Postprocessing", asHIDE_WARNINGS))
    {
        SetForecastScorePostprocess(1);
        SetForecastScorePostprocessMethod(fileParams.GetFirstElementAttributeValueText("Method", "value"));
        SetForecastScorePostprocessDupliExp(fileParams.GetFirstElementAttributeValueFloat("DuplicationExponent", "value"));
        if(!fileParams.GoANodeBack()) return false;
    }
    else
    {
        SetForecastScorePostprocessMethod(wxEmptyString);
        SetForecastScorePostprocessDupliExp(NaNFloat);
    }

    if(!fileParams.GoANodeBack()) return false;

    // Get Analogs Forecast Score Final process
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Analogs Forecast Score Final")) return false;
    if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array")) return false;
    if(!SetForecastScoreTimeArrayMode(fileParams.GetFirstElementAttributeValueText("Mode", "value"))) return false;
    SetForecastScoreTimeArrayDate(fileParams.GetFirstElementAttributeValueDouble("Date", "value"));
    SetForecastScoreTimeArrayIntervalDays(fileParams.GetFirstElementAttributeValueDouble("IntervalDays", "value"));
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.CheckDeprecatedChildNode("Validation")) return false;

    // Set sizes
    SetSizes();

    // Fixes
    FixTimeLimits();

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
