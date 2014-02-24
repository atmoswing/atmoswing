#include "asParametersOptimization.h"

#include <asFileParametersOptimization.h>


asParametersOptimization::asParametersOptimization()
:
asParametersScoring()
{
    m_ForecastScoreIteration.AnalogsNumber = 1;
    m_ForecastScoreUpperLimit.AnalogsNumber = 100;
    m_ForecastScoreLowerLimit.AnalogsNumber = 10;
    m_ForecastScoreLocks.AnalogsNumber = false;
    m_TimeArrayAnalogsIntervalDaysIteration = 1;
    m_TimeArrayAnalogsIntervalDaysUpperLimit = 182;
    m_TimeArrayAnalogsIntervalDaysLowerLimit = 10;
    m_TimeArrayAnalogsIntervalDaysLocks = false;
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

    m_StepsIteration.push_back(stepIteration);
    m_StepsUpperLimit.push_back(stepUpperLimit);
    m_StepsLowerLimit.push_back(stepLowerLimit);
    m_StepsLocks.push_back(stepLocks);
    m_StepsVect.push_back(stepVect);

    // Set sizes
    SetSizes();

}

void asParametersOptimization::AddPredictorIteration(ParamsStep &step)
{
    ParamsPredictor predictor;

    predictor.Umin = 2.5;
    predictor.Uptsnb = 1;
    predictor.Vmin = 2.5;
    predictor.Vptsnb = 1;
    predictor.TimeHours = 6;
    predictor.Weight = 0.01f;

    step.Predictors.push_back(predictor);
}

void asParametersOptimization::AddPredictorUpperLimit(ParamsStep &step)
{
    ParamsPredictor predictor;

    predictor.Umin = 717.5;
    predictor.Uptsnb = 20;
    predictor.Vmin = 87.5;
    predictor.Vptsnb = 16;
    predictor.TimeHours = 36;
    predictor.Weight = 1;

    step.Predictors.push_back(predictor);
}

void asParametersOptimization::AddPredictorLowerLimit(ParamsStep &step)
{
    ParamsPredictor predictor;

    predictor.Umin = 0;
    predictor.Uptsnb = 1;
    predictor.Vmin = 0;
    predictor.Vptsnb = 1;
    predictor.TimeHours = 6;
    predictor.Weight = 0;

    step.Predictors.push_back(predictor);
}

void asParametersOptimization::AddPredictorLocks(ParamsStepBool &step)
{
    ParamsPredictorBool predictor;

    predictor.DataId = false;
    predictor.Level = false;
    predictor.Umin = false;
    predictor.Uptsnb = false;
    predictor.Vmin = false;
    predictor.Vptsnb = false;
    predictor.TimeHours = true;
    predictor.Weight = false;
    predictor.Criteria = false;

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
    if(!SetArchiveYearStart(fileParams.GetFirstElementAttributeValueInt("YearStart", "value"))) return false;
    if(!SetArchiveYearEnd(fileParams.GetFirstElementAttributeValueInt("YearEnd", "value"))) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Calibration Period")) return false;
    if(!SetCalibrationYearStart(fileParams.GetFirstElementAttributeValueInt("YearStart", "value"))) return false;
    if(!SetCalibrationYearEnd(fileParams.GetFirstElementAttributeValueInt("YearEnd", "value"))) return false;
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

    if(!SetTimeArrayAnalogsIntervalDaysLowerLimit(fileParams.GetFirstElementAttributeValueInt("IntervalDays", "lowerlimit"))) return false;
    if(!SetTimeArrayAnalogsIntervalDaysUpperLimit(fileParams.GetFirstElementAttributeValueInt("IntervalDays", "upperlimit"))) return false;
    if(!SetTimeArrayAnalogsIntervalDaysIteration(fileParams.GetFirstElementAttributeValueInt("IntervalDays", "iteration"))) return false;
    SetTimeArrayAnalogsIntervalDaysLock(fileParams.GetFirstElementAttributeValueBool("IntervalDays", "lock", true));
    if(IsTimeArrayAnalogsIntervalDaysLocked())
    {
        if(!SetTimeArrayAnalogsIntervalDays(fileParams.GetFirstElementAttributeValueInt("IntervalDays", "value"))) return false;
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
        if(!SetAnalogsNumberLowerLimit(i_step, fileParams.GetFirstElementAttributeValueInt("AnalogsNumber", "lowerlimit"))) return false;
        if(!SetAnalogsNumberUpperLimit(i_step, fileParams.GetFirstElementAttributeValueInt("AnalogsNumber", "upperlimit"))) return false;
        if(!SetAnalogsNumberIteration(i_step, fileParams.GetFirstElementAttributeValueInt("AnalogsNumber", "iteration"))) return false;
        SetAnalogsNumberLock(i_step, fileParams.GetFirstElementAttributeValueBool("AnalogsNumber", "lock", true));
        if(IsAnalogsNumberLocked(i_step))
        {
            if(!SetAnalogsNumber(i_step, fileParams.GetFirstElementAttributeValueInt("AnalogsNumber", "value"))) return false;
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
                if(!SetPredictorTimeHoursLowerLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("TimeHours", "lowerlimit"))) return false;
                if(!SetPredictorTimeHoursUpperLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("TimeHours", "upperlimit"))) return false;
                if(!SetPredictorTimeHoursIteration(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("TimeHours", "iteration"))) return false;
                SetPredictorTimeHoursLock(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("TimeHours", "lock", true));
                if(IsPredictorTimeHoursLocked(i_step, i_ptor))
                {
                    if(!SetPredictorTimeHours(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("TimeHours", "value"))) return false;
                    VectorDouble vTimeHours;
                    vTimeHours.push_back(fileParams.GetFirstElementAttributeValueDouble("TimeHours", "value"));
                    if(!SetPreloadTimeHours(i_step, i_ptor, vTimeHours)) return false;
                }
                else
                {
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

                    if(!SetPreprocessTimeHoursLowerLimit(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueDouble("PreprocessTimeHours", "lowerlimit"))) return false;
                    if(!SetPreprocessTimeHoursUpperLimit(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueDouble("PreprocessTimeHours", "upperlimit"))) return false;
                    if(!SetPreprocessTimeHoursIteration(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueDouble("PreprocessTimeHours", "iteration"))) return false;
                    SetPreprocessTimeHoursLock(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueBool("PreprocessTimeHours", "lock", true));
                    if(IsPreprocessTimeHoursLocked(i_step, i_ptor, i_dataset))
                    {
                        if(!SetPreprocessTimeHours(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueDouble("PreprocessTimeHours", "value"))) return false;
                    }
                    else
                    {
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

                    // Different actions depending on the preprocessing method.
                    if (method.IsSameAs("Gradients"))
                    {
                        if (preprocSize!=1)
                        {
                            asLogError(wxString::Format(_("The size of the provided predictors (%d) does not match the requirements (1) in the preprocessing Gradients method."), preprocSize));
                            return false;
                        }
                        preprocLevels = GetPreprocessLevelVector(i_step, i_ptor, 0);
                        preprocTimeHours = GetPreprocessTimeHoursVector(i_step, i_ptor, 0);
                    }
                    else if (method.IsSameAs("HumidityFlux"))
                    {
                        if (preprocSize!=4)
                        {
                            asLogError(wxString::Format(_("The size of the provided predictors (%d) does not match the requirements (4) in the preprocessing HumidityFlux method."), preprocSize));
                            return false;
                        }
                        preprocLevels = GetPreprocessLevelVector(i_step, i_ptor, 0);
                        preprocTimeHours = GetPreprocessTimeHoursVector(i_step, i_ptor, 0);
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

                    // Fix the criteria if S1
                    if (method.IsSameAs("Gradients"))
                    {
                        if (GetPredictorCriteria(i_step, i_ptor).IsSameAs("S1"))
                        {
                            SetPredictorCriteria(i_step, i_ptor, "S1grads");
                        }
                    }
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

            if(!fileParams.GoToChildNodeWithAttributeValue("name", "Area")) return false;
            if(!SetPredictorGridType(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("GridType", "value", "Regular"))) return false;
            if(!SetPredictorUminLowerLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Umin", "lowerlimit"))) return false;
            if(!SetPredictorUminUpperLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Umin", "upperlimit"))) return false;
            if(!SetPredictorUminIteration(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Umin", "iteration"))) return false;
            SetPredictorUminLock(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Umin", "lock", true));
            if(IsPredictorUminLocked(i_step, i_ptor))
            {
                if(!SetPredictorUmin(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Umin", "value"))) return false;
                if(!SetPredictorUminLowerLimit(i_step, i_ptor, GetPredictorUmin(i_step, i_ptor))) return false;
                if(!SetPredictorUminUpperLimit(i_step, i_ptor, GetPredictorUmin(i_step, i_ptor))) return false;
            }

            if(!SetPredictorUptsnbLowerLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueInt("Uptsnb", "lowerlimit"))) return false;
            if (GetPredictorUptsnbLowerLimit(i_step, i_ptor)==0)
            {
                SetPredictorUptsnbLowerLimit(i_step, i_ptor, 1);
            }
            if(!SetPredictorUptsnbUpperLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueInt("Uptsnb", "upperlimit"))) return false;
            if(!SetPredictorUptsnbIteration(i_step, i_ptor, fileParams.GetFirstElementAttributeValueInt("Uptsnb", "iteration"))) return false;
            SetPredictorUptsnbLock(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Uptsnb", "lock", true));
            if(IsPredictorUptsnbLocked(i_step, i_ptor))
            {
                if(!SetPredictorUptsnb(i_step, i_ptor, fileParams.GetFirstElementAttributeValueInt("Uptsnb", "value"))) return false;
                if(!SetPredictorUptsnbLowerLimit(i_step, i_ptor, GetPredictorUptsnb(i_step, i_ptor))) return false;
                if(!SetPredictorUptsnbUpperLimit(i_step, i_ptor, GetPredictorUptsnb(i_step, i_ptor))) return false;
            }
            if(!SetPredictorUstep(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Ustep", "value"))) return false;
            double Ushift = fmod(GetPredictorUminLowerLimit(i_step, i_ptor), GetPredictorUstep(i_step, i_ptor));
            if (Ushift<0) Ushift += GetPredictorUstep(i_step, i_ptor);
            if(!SetPredictorUshift(i_step, i_ptor, Ushift)) return false;

            if(!SetPredictorVminLowerLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Vmin", "lowerlimit"))) return false;
            if(!SetPredictorVminUpperLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Vmin", "upperlimit"))) return false;
            if(!SetPredictorVminIteration(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Vmin", "iteration"))) return false;
            SetPredictorVminLock(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Vmin", "lock", true));
            if(IsPredictorVminLocked(i_step, i_ptor))
            {
                if(!SetPredictorVmin(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Vmin", "value"))) return false;
                if(!SetPredictorVminLowerLimit(i_step, i_ptor, GetPredictorVmin(i_step, i_ptor))) return false;
                if(!SetPredictorVminUpperLimit(i_step, i_ptor, GetPredictorVmin(i_step, i_ptor))) return false;
            }

            if(!SetPredictorVptsnbLowerLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueInt("Vptsnb", "lowerlimit"))) return false;
            if (GetPredictorVptsnbLowerLimit(i_step, i_ptor)==0) SetPredictorVptsnbLowerLimit(i_step, i_ptor, 1);
            if(!SetPredictorVptsnbUpperLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueInt("Vptsnb", "upperlimit"))) return false;
            if(!SetPredictorVptsnbIteration(i_step, i_ptor, fileParams.GetFirstElementAttributeValueInt("Vptsnb", "iteration"))) return false;
            SetPredictorVptsnbLock(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Vptsnb", "lock", true));
            if(IsPredictorVptsnbLocked(i_step, i_ptor))
            {
                if(!SetPredictorVptsnb(i_step, i_ptor, fileParams.GetFirstElementAttributeValueInt("Vptsnb", "value"))) return false;
                if(!SetPredictorVptsnbLowerLimit(i_step, i_ptor, GetPredictorVptsnb(i_step, i_ptor))) return false;
                if(!SetPredictorVptsnbUpperLimit(i_step, i_ptor, GetPredictorVptsnb(i_step, i_ptor))) return false;
            }

            if(!SetPredictorVstep(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Vstep", "value"))) return false;
            double Vshift = fmod(GetPredictorVminLowerLimit(i_step, i_ptor), GetPredictorVstep(i_step, i_ptor));
            if (Vshift<0) Vshift += GetPredictorVstep(i_step, i_ptor);
            if(!SetPredictorVshift(i_step, i_ptor, Vshift)) return false;

            if (GetPredictorUptsnbLowerLimit(i_step, i_ptor)<=1 || GetPredictorVptsnbLowerLimit(i_step, i_ptor)<=1) SetPredictorFlatAllowed(i_step, i_ptor, asFLAT_ALLOWED);
            if (IsPredictorUptsnbLocked(i_step, i_ptor) && GetPredictorUptsnb(i_step, i_ptor)<=1) SetPredictorFlatAllowed(i_step, i_ptor, asFLAT_ALLOWED);
            if (IsPredictorVptsnbLocked(i_step, i_ptor) && GetPredictorVptsnb(i_step, i_ptor)<=1) SetPredictorFlatAllowed(i_step, i_ptor, asFLAT_ALLOWED);
            if (NeedsPreloading(i_step, i_ptor))
            {
                // Set maximum extent
                if (!IsPredictorUminLocked(i_step, i_ptor))
                {
                    SetPreloadUmin(i_step, i_ptor, GetPredictorUminLowerLimit(i_step, i_ptor));
                }
                else
                {
                    SetPreloadUmin(i_step, i_ptor, GetPredictorUmin(i_step, i_ptor));
                }

                if (!IsPredictorVminLocked(i_step, i_ptor))
                {
                    SetPreloadVmin(i_step, i_ptor, GetPredictorVminLowerLimit(i_step, i_ptor));
                }
                else
                {
                    SetPreloadVmin(i_step, i_ptor, GetPredictorVmin(i_step, i_ptor));
                }

                if (!IsPredictorUptsnbLocked(i_step, i_ptor))
                {
                    int Ubaseptsnb = abs(GetPredictorUminUpperLimit(i_step, i_ptor)-GetPredictorUminLowerLimit(i_step, i_ptor))/GetPredictorUstep(i_step, i_ptor);
                    SetPreloadUptsnb(i_step, i_ptor, Ubaseptsnb+GetPredictorUptsnbUpperLimit(i_step, i_ptor));
                }
                else
                {
                    SetPreloadUptsnb(i_step, i_ptor, GetPredictorUptsnb(i_step, i_ptor));
                }

                if (!IsPredictorVptsnbLocked(i_step, i_ptor))
                {
                    int Vbaseptsnb = abs(GetPredictorVminUpperLimit(i_step, i_ptor)-GetPredictorVminLowerLimit(i_step, i_ptor))/GetPredictorVstep(i_step, i_ptor);
                    SetPreloadVptsnb(i_step, i_ptor, Vbaseptsnb+GetPredictorVptsnbUpperLimit(i_step, i_ptor));
                }
                else
                {
                    SetPreloadVptsnb(i_step, i_ptor, GetPredictorVptsnb(i_step, i_ptor));
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

            if(!fileParams.GoToChildNodeWithAttributeValue("name", "Weight")) return false;
            if(!SetPredictorWeightLowerLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueFloat("Weight", "lowerlimit"))) return false;
            if(!SetPredictorWeightUpperLimit(i_step, i_ptor, fileParams.GetFirstElementAttributeValueFloat("Weight", "upperlimit"))) return false;
            if(!SetPredictorWeightIteration(i_step, i_ptor, fileParams.GetFirstElementAttributeValueFloat("Weight", "iteration"))) return false;
            SetPredictorWeightLock(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Weight", "lock", true));
            if(IsPredictorWeightLocked(i_step, i_ptor))
            {
                if(!SetPredictorWeight(i_step, i_ptor, fileParams.GetFirstElementAttributeValueFloat("Weight", "value"))) return false;
            }
            if(!fileParams.GoANodeBack()) return false;

            if(fileParams.GoToNextSameNode())
            {
                i_ptor++;
                AddPredictor(i_step);
                AddPredictorIteration(m_StepsIteration[i_step]);
                AddPredictorUpperLimit(m_StepsUpperLimit[i_step]);
                AddPredictorLowerLimit(m_StepsLowerLimit[i_step]);
                AddPredictorLocks(m_StepsLocks[i_step]);
                AddPredictorVect(m_StepsVect[i_step]);
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
    if(!SetPredictandStationId(fileParams.GetFirstElementAttributeValueInt("PredictandStationId", "value", 0))) return false;
    if(!SetPredictandTimeHours(fileParams.GetFirstElementAttributeValueDouble("PredictandTimeHours", "value", 0.0))) return false;
    if(!fileParams.GoANodeBack()) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoANodeBack()) return false;

    // Get Analogs Forecast Scores process
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Analogs Forecast Scores")) return false;
    if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Method")) return false;
    if(!SetForecastScoreName(fileParams.GetFirstElementAttributeValueText("Name", "value"))) return false;
    SetForecastScoreThreshold(fileParams.GetFirstElementAttributeValueFloat("Threshold", "value", NaNFloat));
    SetForecastScorePercentile(fileParams.GetFirstElementAttributeValueFloat("Percentile", "value", NaNFloat));
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Analogs Number")) return false;
    if(!SetForecastScoreAnalogsNumberLowerLimit(fileParams.GetFirstElementAttributeValueInt("AnalogsNumber", "lowerlimit"))) return false;
    if(!SetForecastScoreAnalogsNumberUpperLimit(fileParams.GetFirstElementAttributeValueInt("AnalogsNumber", "upperlimit"))) return false;
    if(!SetForecastScoreAnalogsNumberIteration(fileParams.GetFirstElementAttributeValueInt("AnalogsNumber", "iteration"))) return false;
    SetForecastScoreAnalogsNumberLock(fileParams.GetFirstElementAttributeValueBool("AnalogsNumber", "lock", true));
    if(IsForecastScoreAnalogsNumberLocked())
    {
        if(!SetForecastScoreAnalogsNumber(fileParams.GetFirstElementAttributeValueInt("AnalogsNumber", "value"))) return false;
    }
    if(!fileParams.GoANodeBack()) return false;

    if(fileParams.GoToChildNodeWithAttributeValue("name", "Postprocessing"))
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

    m_TimeMinHours = minHour;
    m_TimeMaxHours = maxHour;

    return true;
}

void asParametersOptimization::InitRandomValues()
{
    if(!m_TimeArrayAnalogsIntervalDaysLocks)
    {
        m_TimeArrayAnalogsIntervalDays = asTools::Random(m_TimeArrayAnalogsIntervalDaysLowerLimit, m_TimeArrayAnalogsIntervalDaysUpperLimit, m_TimeArrayAnalogsIntervalDaysIteration);
    }

    for (int i=0; i<GetStepsNb(); i++)
    {
        if(!m_StepsLocks[i].AnalogsNumber)
        {
            SetAnalogsNumber(i,asTools::Random(m_StepsLowerLimit[i].AnalogsNumber, m_StepsUpperLimit[i].AnalogsNumber, m_StepsIteration[i].AnalogsNumber));
        }

        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            if (NeedsPreprocessing(i,j))
            {
                for (int k=0; k<GetPreprocessSize(i,j); k++)
                {
                    if(!m_StepsLocks[i].Predictors[j].PreprocessDataId[k])
                    {
                        int length = m_StepsVect[i].Predictors[j].PreprocessDataId[k].size();
                        int row = asTools::Random(0,length-1);
                        wxASSERT(m_StepsVect[i].Predictors[j].PreprocessDataId[k].size()>(unsigned)row);

                        SetPreprocessDataId(i,j,k, m_StepsVect[i].Predictors[j].PreprocessDataId[k][row]);
                    }

                    if(!m_StepsLocks[i].Predictors[j].PreprocessLevels[k])
                    {
                        int length = m_StepsVect[i].Predictors[j].PreprocessLevels[k].size();
                        int row = asTools::Random(0,length-1);
                        wxASSERT(m_StepsVect[i].Predictors[j].PreprocessLevels[k].size()>(unsigned)row);

                        SetPreprocessLevel(i,j,k, m_StepsVect[i].Predictors[j].PreprocessLevels[k][row]);
                    }

                    if(!m_StepsLocks[i].Predictors[j].PreprocessTimeHours[k])
                    {
                        SetPreprocessTimeHours(i,j,k, asTools::Random(m_StepsLowerLimit[i].Predictors[j].PreprocessTimeHours[k], m_StepsUpperLimit[i].Predictors[j].PreprocessTimeHours[k], m_StepsIteration[i].Predictors[j].PreprocessTimeHours[k]));
                    }
                }
            }
            else
            {
                if(!m_StepsLocks[i].Predictors[j].DataId)
                {
                    int length = m_StepsVect[i].Predictors[j].DataId.size();
                    int row = asTools::Random(0,length-1);
                    wxASSERT(m_StepsVect[i].Predictors[j].DataId.size()>(unsigned)row);

                    SetPredictorDataId(i,j, m_StepsVect[i].Predictors[j].DataId[row]);
                }

                if(!m_StepsLocks[i].Predictors[j].Level)
                {
                    int length = m_StepsVect[i].Predictors[j].Level.size();
                    int row = asTools::Random(0,length-1);
                    wxASSERT(m_StepsVect[i].Predictors[j].Level.size()>(unsigned)row);

                    SetPredictorLevel(i,j, m_StepsVect[i].Predictors[j].Level[row]);
                }

                if(!m_StepsLocks[i].Predictors[j].TimeHours)
                {
                    SetPredictorTimeHours(i,j, asTools::Random(m_StepsLowerLimit[i].Predictors[j].TimeHours, m_StepsUpperLimit[i].Predictors[j].TimeHours, m_StepsIteration[i].Predictors[j].TimeHours));
                }

            }

            if(!m_StepsLocks[i].Predictors[j].Umin)
            {
                SetPredictorUmin(i,j, asTools::Random(m_StepsLowerLimit[i].Predictors[j].Umin, m_StepsUpperLimit[i].Predictors[j].Umin, m_StepsIteration[i].Predictors[j].Umin));
            }

            if(!m_StepsLocks[i].Predictors[j].Uptsnb)
            {
                SetPredictorUptsnb(i,j, asTools::Random(m_StepsLowerLimit[i].Predictors[j].Uptsnb, m_StepsUpperLimit[i].Predictors[j].Uptsnb, m_StepsIteration[i].Predictors[j].Uptsnb));
            }

            if(!m_StepsLocks[i].Predictors[j].Vmin)
            {
                SetPredictorVmin(i,j, asTools::Random(m_StepsLowerLimit[i].Predictors[j].Vmin, m_StepsUpperLimit[i].Predictors[j].Vmin, m_StepsIteration[i].Predictors[j].Vmin));
            }

            if(!m_StepsLocks[i].Predictors[j].Vptsnb)
            {
                SetPredictorVptsnb(i,j, asTools::Random(m_StepsLowerLimit[i].Predictors[j].Vptsnb, m_StepsUpperLimit[i].Predictors[j].Vptsnb, m_StepsIteration[i].Predictors[j].Vptsnb));
            }

            if(!m_StepsLocks[i].Predictors[j].Weight)
            {
                SetPredictorWeight(i,j, asTools::Random(m_StepsLowerLimit[i].Predictors[j].Weight, m_StepsUpperLimit[i].Predictors[j].Weight, m_StepsIteration[i].Predictors[j].Weight));
            }

            if(!m_StepsLocks[i].Predictors[j].Criteria)
            {
                int length = m_StepsVect[i].Predictors[j].Criteria.size();
                int row = asTools::Random(0,length-1);
                wxASSERT(m_StepsVect[i].Predictors[j].Criteria.size()>(unsigned)row);

                SetPredictorCriteria(i,j, m_StepsVect[i].Predictors[j].Criteria[row]);
            }

        }
    }

    if(!m_ForecastScoreLocks.AnalogsNumber)
    {
        SetForecastScoreAnalogsNumber(asTools::Random(m_ForecastScoreLowerLimit.AnalogsNumber, m_ForecastScoreUpperLimit.AnalogsNumber, m_ForecastScoreIteration.AnalogsNumber));
    }

    FixWeights();
    FixCoordinates();
    CheckRange();
    FixAnalogsNb();
}

void asParametersOptimization::CheckRange()
{
    // Check that the actual parameters values are within ranges
    if(!m_TimeArrayAnalogsIntervalDaysLocks)
    {
        m_TimeArrayAnalogsIntervalDays = wxMax( wxMin(m_TimeArrayAnalogsIntervalDays, m_TimeArrayAnalogsIntervalDaysUpperLimit), m_TimeArrayAnalogsIntervalDaysLowerLimit);
    }
    wxASSERT(m_TimeArrayAnalogsIntervalDays>0);

    for (int i=0; i<GetStepsNb(); i++)
    {
        if(!m_StepsLocks[i].AnalogsNumber)
        {
            SetAnalogsNumber(i, wxMax( wxMin(GetAnalogsNumber(i), m_StepsUpperLimit[i].AnalogsNumber), m_StepsLowerLimit[i].AnalogsNumber));
        }

        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            if(!GetPredictorGridType(i,j).IsSameAs ("Regular", false)) asThrowException(wxString::Format(_("asParametersOptimization::CheckRange is not ready to use on unregular grids (PredictorGridType = %s)"), GetPredictorGridType(i,j).c_str()));

            if (NeedsPreprocessing(i, j))
            {
                int preprocessSize = GetPreprocessSize(i, j);
                for (int k=0; k<preprocessSize; k++)
                {
                    if(!m_StepsLocks[i].Predictors[j].PreprocessTimeHours[k])
                    {
                        SetPreprocessTimeHours(i, j, k, wxMax( wxMin(GetPreprocessTimeHours(i,j,k), m_StepsUpperLimit[i].Predictors[j].PreprocessTimeHours[k]), m_StepsLowerLimit[i].Predictors[j].PreprocessTimeHours[k]));
                    }
                    SetPredictorTimeHours(i, j, 0);
                }
            }
            else
            {
                if(!m_StepsLocks[i].Predictors[j].TimeHours)
                {
                    SetPredictorTimeHours(i, j, wxMax( wxMin(GetPredictorTimeHours(i,j), m_StepsUpperLimit[i].Predictors[j].TimeHours), m_StepsLowerLimit[i].Predictors[j].TimeHours));
                }
            }

            // Check ranges
            if(!m_StepsLocks[i].Predictors[j].Umin)
            {
                SetPredictorUmin(i, j, wxMax( wxMin(GetPredictorUmin(i,j), m_StepsUpperLimit[i].Predictors[j].Umin), m_StepsLowerLimit[i].Predictors[j].Umin));
            }
            if(!m_StepsLocks[i].Predictors[j].Uptsnb)
            {
                SetPredictorUptsnb(i, j, wxMax( wxMin(GetPredictorUptsnb(i,j), m_StepsUpperLimit[i].Predictors[j].Uptsnb), m_StepsLowerLimit[i].Predictors[j].Uptsnb));
            }

            if(!m_StepsLocks[i].Predictors[j].Vmin)
            {
                SetPredictorVmin(i, j, wxMax( wxMin(GetPredictorVmin(i,j), m_StepsUpperLimit[i].Predictors[j].Vmin), m_StepsLowerLimit[i].Predictors[j].Vmin));
            }
            if(!m_StepsLocks[i].Predictors[j].Vptsnb)
            {
                SetPredictorVptsnb(i, j, wxMax( wxMin(GetPredictorVptsnb(i,j), m_StepsUpperLimit[i].Predictors[j].Vptsnb), m_StepsLowerLimit[i].Predictors[j].Vptsnb));
            }
            if(!m_StepsLocks[i].Predictors[j].Weight)
            {
                SetPredictorWeight(i, j, wxMax( wxMin(GetPredictorWeight(i,j), m_StepsUpperLimit[i].Predictors[j].Weight), m_StepsLowerLimit[i].Predictors[j].Weight));
            }

            if(!m_StepsLocks[i].Predictors[j].Umin || !m_StepsLocks[i].Predictors[j].Uptsnb)
            {
                if(GetPredictorUmin(i,j)+GetPredictorUptsnb(i,j)*GetPredictorUstep(i,j) > m_StepsUpperLimit[i].Predictors[j].Umin+m_StepsLowerLimit[i].Predictors[j].Uptsnb*GetPredictorUstep(i,j))
                {
                    if(!m_StepsLocks[i].Predictors[j].Uptsnb)
                    {
                        SetPredictorUptsnb(i, j, (m_StepsUpperLimit[i].Predictors[j].Umin-GetPredictorUmin(i,j))/GetPredictorUstep(i,j)+m_StepsLowerLimit[i].Predictors[j].Uptsnb);
                    }
                    else
                    {
                        SetPredictorUmin(i, j, m_StepsUpperLimit[i].Predictors[j].Umin-GetPredictorUptsnb(i,j)*GetPredictorUstep(i,j));
                    }
                }
            }

            if(!m_StepsLocks[i].Predictors[j].Vmin || !m_StepsLocks[i].Predictors[j].Vptsnb)
            {
                if(GetPredictorVmin(i,j)+GetPredictorVptsnb(i,j)*GetPredictorVstep(i,j) > m_StepsUpperLimit[i].Predictors[j].Vmin+m_StepsLowerLimit[i].Predictors[j].Vptsnb*GetPredictorVstep(i,j))
                {
                    if(!m_StepsLocks[i].Predictors[j].Vptsnb)
                    {
                        SetPredictorVptsnb(i, j, (m_StepsUpperLimit[i].Predictors[j].Vmin-GetPredictorVmin(i,j))/GetPredictorVstep(i,j)+m_StepsLowerLimit[i].Predictors[j].Vptsnb);
                    }
                    else
                    {
                        SetPredictorVmin(i, j, m_StepsUpperLimit[i].Predictors[j].Vmin-GetPredictorVptsnb(i,j)*GetPredictorVstep(i,j));
                    }
                }
            }
        }
    }

    if(!m_ForecastScoreLocks.AnalogsNumber)
    {
        SetForecastScoreAnalogsNumber(wxMax( wxMin(GetForecastScoreAnalogsNumber(), m_ForecastScoreUpperLimit.AnalogsNumber), m_ForecastScoreLowerLimit.AnalogsNumber));
    }

    FixTimeHours();
    FixWeights();
    FixCoordinates();
    FixAnalogsNb();
}

bool asParametersOptimization::IsInRange()
{
    // Check that the actual parameters values are within ranges
    if(!m_TimeArrayAnalogsIntervalDaysLocks)
    {
        if (m_TimeArrayAnalogsIntervalDays>m_TimeArrayAnalogsIntervalDaysUpperLimit) return false;
        if (m_TimeArrayAnalogsIntervalDays<m_TimeArrayAnalogsIntervalDaysLowerLimit) return false;
    }

    for (int i=0; i<GetStepsNb(); i++)
    {
        if (!m_StepsLocks[i].AnalogsNumber)
        {
            if (GetAnalogsNumber(i)>m_StepsUpperLimit[i].AnalogsNumber) return false;
            if (GetAnalogsNumber(i)<m_StepsLowerLimit[i].AnalogsNumber) return false;
        }

        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            if (NeedsPreprocessing(i,j))
            {
                for (int k=0; k<GetPreprocessSize(i,j); k++)
                {
                    if (!m_StepsLocks[i].Predictors[j].PreprocessTimeHours[k])
                    {
                        if (GetPreprocessTimeHours(i,j,k)<m_StepsLowerLimit[i].Predictors[j].PreprocessTimeHours[k]) return false;
                        if (GetPreprocessTimeHours(i,j,k)<m_StepsLowerLimit[i].Predictors[j].PreprocessTimeHours[k]) return false;
                    }
                }
            }
            else
            {
                if (!m_StepsLocks[i].Predictors[j].TimeHours)
                {
                    if (GetPredictorTimeHours(i,j)<m_StepsLowerLimit[i].Predictors[j].TimeHours) return false;
                    if (GetPredictorTimeHours(i,j)<m_StepsLowerLimit[i].Predictors[j].TimeHours) return false;
                }
            }

            if(!GetPredictorGridType(i,j).IsSameAs ("Regular", false)) asThrowException(wxString::Format(_("asParametersOptimization::CheckRange is not ready to use on unregular grids (PredictorGridType = %s)"), GetPredictorGridType(i,j).c_str()));

            // Check ranges
            if (!m_StepsLocks[i].Predictors[j].Umin)
            {
                if (GetPredictorUmin(i,j)>m_StepsUpperLimit[i].Predictors[j].Umin) return false;
                if (GetPredictorUmin(i,j)<m_StepsLowerLimit[i].Predictors[j].Umin) return false;
            }
            if (!m_StepsLocks[i].Predictors[j].Uptsnb)
            {
                if (GetPredictorUptsnb(i,j)<m_StepsLowerLimit[i].Predictors[j].Uptsnb) return false;
                if (GetPredictorUptsnb(i,j)<m_StepsLowerLimit[i].Predictors[j].Uptsnb) return false;
            }
            if (!m_StepsLocks[i].Predictors[j].Vmin)
            {
                if (GetPredictorVmin(i,j)<m_StepsLowerLimit[i].Predictors[j].Vmin) return false;
                if (GetPredictorVmin(i,j)<m_StepsLowerLimit[i].Predictors[j].Vmin) return false;
            }
            if (!m_StepsLocks[i].Predictors[j].Vptsnb)
            {
                if (GetPredictorVptsnb(i,j)<m_StepsLowerLimit[i].Predictors[j].Vptsnb) return false;
                if (GetPredictorVptsnb(i,j)<m_StepsLowerLimit[i].Predictors[j].Vptsnb) return false;
            }
            if (!m_StepsLocks[i].Predictors[j].Weight)
            {
                if (GetPredictorWeight(i,j)<m_StepsLowerLimit[i].Predictors[j].Weight) return false;
                if (GetPredictorWeight(i,j)<m_StepsLowerLimit[i].Predictors[j].Weight) return false;
            }
            if (!m_StepsLocks[i].Predictors[j].Umin |
                !m_StepsLocks[i].Predictors[j].Uptsnb |
                !m_StepsLocks[i].Predictors[j].Vmin |
                !m_StepsLocks[i].Predictors[j].Vptsnb)
            {
                if(GetPredictorUmin(i,j)+GetPredictorUptsnb(i,j)*GetPredictorUstep(i,j) > m_StepsUpperLimit[i].Predictors[j].Umin+m_StepsLowerLimit[i].Predictors[j].Uptsnb*GetPredictorUstep(i,j)) return false;
                if(GetPredictorVmin(i,j)+GetPredictorVptsnb(i,j)*GetPredictorVstep(i,j) > m_StepsUpperLimit[i].Predictors[j].Vmin+m_StepsLowerLimit[i].Predictors[j].Vptsnb*GetPredictorVstep(i,j)) return false;
            }
        }
    }

    if (!m_ForecastScoreLocks.AnalogsNumber)
    {
        if (GetForecastScoreAnalogsNumber()>m_ForecastScoreUpperLimit.AnalogsNumber) return false;
        if (GetForecastScoreAnalogsNumber()<m_ForecastScoreLowerLimit.AnalogsNumber) return false;
    }

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
                    if (m_StepsIteration[i].Predictors[j].PreprocessTimeHours[k]!=0)
                    {
                        float ratio = (float)GetPreprocessTimeHours(i,j,k)/(float)m_StepsIteration[i].Predictors[j].PreprocessTimeHours[k];
                        ratio = asTools::Round(ratio);
                        SetPreprocessTimeHours(i, j, k, ratio*m_StepsIteration[i].Predictors[j].PreprocessTimeHours[k]);
                    }
                }
            }
            else
            {
                if (m_StepsIteration[i].Predictors[j].TimeHours!=0)
                {
                    float ratio = (float)GetPredictorTimeHours(i,j)/(float)m_StepsIteration[i].Predictors[j].TimeHours;
                    ratio = asTools::Round(ratio);
                    SetPredictorTimeHours(i, j, ratio*m_StepsIteration[i].Predictors[j].TimeHours);
                }
            }
        }
    }
}

void asParametersOptimization::LockAll()
{
    m_TimeArrayAnalogsIntervalDaysLocks = true;

    for (int i=0; i<GetStepsNb(); i++)
    {
        m_StepsLocks[i].AnalogsNumber = true;

        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            if (NeedsPreprocessing(i,j))
            {
                for (int k=0; k<GetPreprocessSize(i,j); k++)
                {
                    m_StepsLocks[i].Predictors[j].PreprocessDataId[k] = true;
                    m_StepsLocks[i].Predictors[j].PreprocessLevels[k] = true;
                    m_StepsLocks[i].Predictors[j].PreprocessTimeHours[k] = true;
                }
            }
            else
            {
                m_StepsLocks[i].Predictors[j].DataId = true;
                m_StepsLocks[i].Predictors[j].Level = true;
                m_StepsLocks[i].Predictors[j].TimeHours = true;
            }

            m_StepsLocks[i].Predictors[j].Umin = true;
            m_StepsLocks[i].Predictors[j].Uptsnb = true;
            m_StepsLocks[i].Predictors[j].Vmin = true;
            m_StepsLocks[i].Predictors[j].Vptsnb = true;
            m_StepsLocks[i].Predictors[j].Weight = true;
            m_StepsLocks[i].Predictors[j].Criteria = true;
        }
    }

    m_ForecastScoreLocks.AnalogsNumber = true;

    return;
}

// TODO (Pascal#1#): Can be optimized by looping on the given vector (sorted first) instead
void asParametersOptimization::Unlock(VectorInt &indices)
{
    int counter = 0;
    int length = indices.size();

    if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
    {
        m_TimeArrayAnalogsIntervalDaysLocks = false;
    }
    counter++;

    for (int i=0; i<GetStepsNb(); i++)
    {
        if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
        {
            m_StepsLocks[i].AnalogsNumber = false;
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
                        m_StepsLocks[i].Predictors[j].PreprocessDataId[k] = false;
                    }
                    counter++;
                    if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
                    {
                        m_StepsLocks[i].Predictors[j].PreprocessLevels[k] = false;
                    }
                    counter++;
                    if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
                    {
                        m_StepsLocks[i].Predictors[j].PreprocessTimeHours[k] = false;
                    }
                    counter++;
                }
            }
            else
            {
                if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
                {
                    m_StepsLocks[i].Predictors[j].DataId = false;
                }
                counter++;
                if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
                {
                    m_StepsLocks[i].Predictors[j].Level = false;
                }
                counter++;
                if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
                {
                    m_StepsLocks[i].Predictors[j].TimeHours = false;
                }
                counter++;
            }

            if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
            {
                m_StepsLocks[i].Predictors[j].Umin = false;
            }
            counter++;
            if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
            {
                m_StepsLocks[i].Predictors[j].Uptsnb = false;
            }
            counter++;
            if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
            {
                m_StepsLocks[i].Predictors[j].Vmin = false;
            }
            counter++;
            if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
            {
                m_StepsLocks[i].Predictors[j].Vptsnb = false;
            }
            counter++;
            if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
            {
                m_StepsLocks[i].Predictors[j].Weight = false;
            }
            counter++;
            if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
            {
                m_StepsLocks[i].Predictors[j].Criteria = false;
            }
            counter++;
        }
    }

    if(asTools::SortedArraySearch(&indices[0], &indices[length-1], counter)>=0)
    {
        m_ForecastScoreLocks.AnalogsNumber = false;
    }
    counter++;
}

int asParametersOptimization::GetVariablesNb()
{
    int counter = 0;

    if(!m_TimeArrayAnalogsIntervalDaysLocks) counter++;

    for (int i=0; i<GetStepsNb(); i++)
    {
        if(!m_StepsLocks[i].AnalogsNumber) counter++;

        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            if(NeedsPreprocessing(i,j))
            {
                for (int k=0; k<GetPreprocessSize(i,j); k++)
                {
                    if(!m_StepsLocks[i].Predictors[j].PreprocessDataId[k]) counter++;
                    if(!m_StepsLocks[i].Predictors[j].PreprocessLevels[k]) counter++;
                    if(!m_StepsLocks[i].Predictors[j].PreprocessTimeHours[k]) counter++;
                }
            }
            else
            {
                if(!m_StepsLocks[i].Predictors[j].DataId) counter++;
                if(!m_StepsLocks[i].Predictors[j].Level) counter++;
                if(!m_StepsLocks[i].Predictors[j].TimeHours) counter++;
            }

            if(!m_StepsLocks[i].Predictors[j].Umin) counter++;
            if(!m_StepsLocks[i].Predictors[j].Uptsnb) counter++;
            if(!m_StepsLocks[i].Predictors[j].Vmin) counter++;
            if(!m_StepsLocks[i].Predictors[j].Vptsnb) counter++;
            if(!m_StepsLocks[i].Predictors[j].Weight) counter++;
            if(!m_StepsLocks[i].Predictors[j].Criteria) counter++;
        }
    }

    if(!m_ForecastScoreLocks.AnalogsNumber) counter++;

    return counter;
}

bool asParametersOptimization::IsCloseTo(asParametersOptimization &otherParam)
{
    bool isclose = true;

    if(abs(m_TimeArrayAnalogsIntervalDays-otherParam.GetTimeArrayAnalogsIntervalDays())>=m_TimeArrayAnalogsIntervalDaysIteration) isclose = false;

    // Loop over every parameter to find one that is not close
    for (int i_step=0; i_step<GetStepsNb(); i_step++)
    {
        if(abs(GetAnalogsNumber(i_step)-otherParam.GetAnalogsNumber(i_step))>=m_StepsIteration[i_step].AnalogsNumber) isclose = false;

        for (int i_ptor=0; i_ptor<GetPredictorsNb(i_step); i_ptor++)
        {
            if (NeedsPreprocessing(i_step, i_ptor))
            {
                for (int i_pre=0; i_pre<GetPreprocessSize(i_step, i_ptor); i_pre++)
                {
                    if(abs(GetPreprocessTimeHours(i_step, i_ptor, i_pre)-otherParam.GetPreprocessTimeHours(i_step, i_ptor, i_pre))>=m_StepsIteration[i_step].Predictors[i_ptor].PreprocessTimeHours[i_pre]) isclose = false;
                }
            }
            else
            {
                if(abs(GetPredictorTimeHours(i_step, i_ptor)-otherParam.GetPredictorTimeHours(i_step, i_ptor))>=m_StepsIteration[i_step].Predictors[i_ptor].TimeHours) isclose = false;
            }
            if(abs(GetPredictorUmin(i_step, i_ptor)-otherParam.GetPredictorUmin(i_step, i_ptor))>=m_StepsIteration[i_step].Predictors[i_ptor].Umin) isclose = false;
            if(abs(GetPredictorUptsnb(i_step, i_ptor)-otherParam.GetPredictorUptsnb(i_step, i_ptor))>=m_StepsIteration[i_step].Predictors[i_ptor].Uptsnb) isclose = false;
            if(abs(GetPredictorVmin(i_step, i_ptor)-otherParam.GetPredictorVmin(i_step, i_ptor))>=m_StepsIteration[i_step].Predictors[i_ptor].Vmin) isclose = false;
            if(abs(GetPredictorVptsnb(i_step, i_ptor)-otherParam.GetPredictorVptsnb(i_step, i_ptor))>=m_StepsIteration[i_step].Predictors[i_ptor].Vptsnb) isclose = false;
            if(abs(GetPredictorWeight(i_step, i_ptor)-otherParam.GetPredictorWeight(i_step, i_ptor))>=m_StepsIteration[i_step].Predictors[i_ptor].Weight) isclose = false;
        }
    }
    if(abs(GetForecastScoreAnalogsNumber()-otherParam.GetForecastScoreAnalogsNumber())>=m_ForecastScoreIteration.AnalogsNumber) isclose = false;

    return isclose;
}
