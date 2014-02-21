#ifndef ASPARAMETERSOPTIMIZATION_H
#define ASPARAMETERSOPTIMIZATION_H

#include "asIncludes.h"
#include <asParametersScoring.h>

class asFileParametersOptimization;


class asParametersOptimization : public asParametersScoring
{
public:

    asParametersOptimization();
    virtual ~asParametersOptimization();

    void AddStep();
    void AddPredictorIteration(ParamsStep &step);
    void AddPredictorUpperLimit(ParamsStep &step);
    void AddPredictorLowerLimit(ParamsStep &step);
    void AddPredictorLocks(ParamsStepBool &step);
    void AddPredictorRandomInits(ParamsStepBool &step);

    void InitRandomValues();

    bool LoadFromFile(const wxString &filePath);

    void CheckRange();

    bool IsInRange();

    void FixTimeHours();

    void LockAll();

    void Unlock(VectorInt &indices);

    // May vary
    int GetVariablesNb();

    // Does not change after importation from file.
    int GetVariableParamsNb()
    {
        return m_VariableParamsNb;
    }

    bool IsCloseTo(asParametersOptimization &otherParam);

    int GetTimeArrayAnalogsIntervalDaysIteration()
    {
        return m_TimeArrayAnalogsIntervalDaysIteration;
    }

    void SetTimeArrayAnalogsIntervalDaysIteration(int val)
    {
        m_TimeArrayAnalogsIntervalDaysIteration = val;
    }

    int GetAnalogsNumberIteration(int i_step)
    {
        return m_StepsIteration[i_step].AnalogsNumber;
    }

    void SetAnalogsNumberIteration(int i_step, int val)
    {
        m_StepsIteration[i_step].AnalogsNumber = val;
    }

    double GetPreprocessTimeHoursIteration(int i_step, int i_predictor, int i_dataset)
    {
        wxASSERT(m_StepsLowerLimit[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>(unsigned)i_dataset);
        return m_StepsIteration[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset];
    }

    void SetPreprocessTimeHoursIteration(int i_step, int i_predictor, int i_dataset, double val)
    {
        if(m_StepsIteration[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>=(unsigned)(i_dataset+1))
        {
            m_StepsIteration[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset] = val;
        }
        else
        {
            wxASSERT(m_StepsIteration[i_step].Predictors[i_predictor].PreprocessTimeHours.size()==(unsigned)i_dataset);
            m_StepsIteration[i_step].Predictors[i_predictor].PreprocessTimeHours.push_back(val);
        }
    }

    double GetPredictorTimeHoursIteration(int i_step, int i_predictor)
    {
        return m_StepsIteration[i_step].Predictors[i_predictor].TimeHours;
    }

    void SetPredictorTimeHoursIteration(int i_step, int i_predictor, double val)
    {
        m_StepsIteration[i_step].Predictors[i_predictor].TimeHours = val;
    }

    double GetPredictorUminIteration(int i_step, int i_predictor)
    {
        return m_StepsIteration[i_step].Predictors[i_predictor].Umin;
    }

    void SetPredictorUminIteration(int i_step, int i_predictor, double val)
    {
        m_StepsIteration[i_step].Predictors[i_predictor].Umin = val;
    }

    int GetPredictorUptsnbIteration(int i_step, int i_predictor)
    {
        return m_StepsIteration[i_step].Predictors[i_predictor].Uptsnb;
    }

    void SetPredictorUptsnbIteration(int i_step, int i_predictor, int val)
    {
        m_StepsIteration[i_step].Predictors[i_predictor].Uptsnb = val;
    }

    double GetPredictorVminIteration(int i_step, int i_predictor)
    {
        return m_StepsIteration[i_step].Predictors[i_predictor].Vmin;
    }

    void SetPredictorVminIteration(int i_step, int i_predictor, double val)
    {
        m_StepsIteration[i_step].Predictors[i_predictor].Vmin = val;
    }

    int GetPredictorVptsnbIteration(int i_step, int i_predictor)
    {
        return m_StepsIteration[i_step].Predictors[i_predictor].Vptsnb;
    }

    void SetPredictorVptsnbIteration(int i_step, int i_predictor, int val)
    {
        m_StepsIteration[i_step].Predictors[i_predictor].Vptsnb = val;
    }

    float GetPredictorWeightIteration(int i_step, int i_predictor)
    {
        return m_StepsIteration[i_step].Predictors[i_predictor].Weight;
    }

    void SetPredictorWeightIteration(int i_step, int i_predictor, float val)
    {
        m_StepsIteration[i_step].Predictors[i_predictor].Weight = val;
    }

    int GetForecastScoreAnalogsNumberIteration()
    {
        return m_ForecastScoreIteration.AnalogsNumber;
    }

    void SetForecastScoreAnalogsNumberIteration(int val)
    {
        m_ForecastScoreIteration.AnalogsNumber = val;
    }

    int GetTimeArrayAnalogsIntervalDaysUpperLimit()
    {
        return m_TimeArrayAnalogsIntervalDaysUpperLimit;
    }

    void SetTimeArrayAnalogsIntervalDaysUpperLimit(int val)
    {
        m_TimeArrayAnalogsIntervalDaysUpperLimit = val;
    }

    int GetAnalogsNumberUpperLimit(int i_step)
    {
        return m_StepsUpperLimit[i_step].AnalogsNumber;
    }

    void SetAnalogsNumberUpperLimit(int i_step, int val)
    {
        m_StepsUpperLimit[i_step].AnalogsNumber = val;
    }

    double GetPreprocessTimeHoursUpperLimit(int i_step, int i_predictor, int i_dataset)
    {
        wxASSERT(m_StepsUpperLimit[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>(unsigned)i_dataset);
        return m_StepsUpperLimit[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset];
    }

    void SetPreprocessTimeHoursUpperLimit(int i_step, int i_predictor, int i_dataset, double val)
    {
        if(m_StepsUpperLimit[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>=(unsigned)(i_dataset+1))
        {
            m_StepsUpperLimit[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset] = val;
        }
        else
        {
            wxASSERT(m_StepsUpperLimit[i_step].Predictors[i_predictor].PreprocessTimeHours.size()==(unsigned)i_dataset);
            m_StepsUpperLimit[i_step].Predictors[i_predictor].PreprocessTimeHours.push_back(val);
        }
    }

    double GetPredictorTimeHoursUpperLimit(int i_step, int i_predictor)
    {
        return m_StepsUpperLimit[i_step].Predictors[i_predictor].TimeHours;
    }

    void SetPredictorTimeHoursUpperLimit(int i_step, int i_predictor, double val)
    {
        m_StepsUpperLimit[i_step].Predictors[i_predictor].TimeHours = val;
    }

    double GetPredictorUminUpperLimit(int i_step, int i_predictor)
    {
        return m_StepsUpperLimit[i_step].Predictors[i_predictor].Umin;
    }

    void SetPredictorUminUpperLimit(int i_step, int i_predictor, double val)
    {
        m_StepsUpperLimit[i_step].Predictors[i_predictor].Umin = val;
    }

    int GetPredictorUptsnbUpperLimit(int i_step, int i_predictor)
    {
        return m_StepsUpperLimit[i_step].Predictors[i_predictor].Uptsnb;
    }

    void SetPredictorUptsnbUpperLimit(int i_step, int i_predictor, int val)
    {
        m_StepsUpperLimit[i_step].Predictors[i_predictor].Uptsnb = val;
    }

    double GetPredictorVminUpperLimit(int i_step, int i_predictor)
    {
        return m_StepsUpperLimit[i_step].Predictors[i_predictor].Vmin;
    }

    void SetPredictorVminUpperLimit(int i_step, int i_predictor, double val)
    {
        m_StepsUpperLimit[i_step].Predictors[i_predictor].Vmin = val;
    }

    int GetPredictorVptsnbUpperLimit(int i_step, int i_predictor)
    {
        return m_StepsUpperLimit[i_step].Predictors[i_predictor].Vptsnb;
    }

    void SetPredictorVptsnbUpperLimit(int i_step, int i_predictor, int val)
    {
        m_StepsUpperLimit[i_step].Predictors[i_predictor].Vptsnb = val;
    }

    float GetPredictorWeightUpperLimit(int i_step, int i_predictor)
    {
        return m_StepsUpperLimit[i_step].Predictors[i_predictor].Weight;
    }

    void SetPredictorWeightUpperLimit(int i_step, int i_predictor, float val)
    {
        m_StepsUpperLimit[i_step].Predictors[i_predictor].Weight = val;
    }

    int GetForecastScoreAnalogsNumberUpperLimit()
    {
        return m_ForecastScoreUpperLimit.AnalogsNumber;
    }

    void SetForecastScoreAnalogsNumberUpperLimit(int val)
    {
        m_ForecastScoreUpperLimit.AnalogsNumber = val;
    }

    int GetTimeArrayAnalogsIntervalDaysLowerLimit()
    {
        return m_TimeArrayAnalogsIntervalDaysLowerLimit;
    }

    void SetTimeArrayAnalogsIntervalDaysLowerLimit(int val)
    {
        m_TimeArrayAnalogsIntervalDaysLowerLimit = val;
    }

    int GetAnalogsNumberLowerLimit(int i_step)
    {
        return m_StepsLowerLimit[i_step].AnalogsNumber;
    }

    void SetAnalogsNumberLowerLimit(int i_step, int val)
    {
        m_StepsLowerLimit[i_step].AnalogsNumber = val;
    }

    double GetPreprocessTimeHoursLowerLimit(int i_step, int i_predictor, int i_dataset)
    {
        wxASSERT(m_StepsLowerLimit[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>(unsigned)i_dataset);
        return m_StepsLowerLimit[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset];
    }

    void SetPreprocessTimeHoursLowerLimit(int i_step, int i_predictor, int i_dataset, double val)
    {
        if(m_StepsLowerLimit[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>=(unsigned)(i_dataset+1))
        {
            m_StepsLowerLimit[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset] = val;
        }
        else
        {
            wxASSERT(m_StepsLowerLimit[i_step].Predictors[i_predictor].PreprocessTimeHours.size()==(unsigned)i_dataset);
            m_StepsLowerLimit[i_step].Predictors[i_predictor].PreprocessTimeHours.push_back(val);
        }
    }

    double GetPredictorTimeHoursLowerLimit(int i_step, int i_predictor)
    {
        return m_StepsLowerLimit[i_step].Predictors[i_predictor].TimeHours;
    }

    void SetPredictorTimeHoursLowerLimit(int i_step, int i_predictor, double val)
    {
        m_StepsLowerLimit[i_step].Predictors[i_predictor].TimeHours = val;
    }

    double GetPredictorUminLowerLimit(int i_step, int i_predictor)
    {
        return m_StepsLowerLimit[i_step].Predictors[i_predictor].Umin;
    }

    void SetPredictorUminLowerLimit(int i_step, int i_predictor, double val)
    {
        m_StepsLowerLimit[i_step].Predictors[i_predictor].Umin = val;
    }

    int GetPredictorUptsnbLowerLimit(int i_step, int i_predictor)
    {
        return m_StepsLowerLimit[i_step].Predictors[i_predictor].Uptsnb;
    }

    void SetPredictorUptsnbLowerLimit(int i_step, int i_predictor, int val)
    {
        m_StepsLowerLimit[i_step].Predictors[i_predictor].Uptsnb = val;
    }

    double GetPredictorVminLowerLimit(int i_step, int i_predictor)
    {
        return m_StepsLowerLimit[i_step].Predictors[i_predictor].Vmin;
    }

    void SetPredictorVminLowerLimit(int i_step, int i_predictor, double val)
    {
        m_StepsLowerLimit[i_step].Predictors[i_predictor].Vmin = val;
    }

    int GetPredictorVptsnbLowerLimit(int i_step, int i_predictor)
    {
        return m_StepsLowerLimit[i_step].Predictors[i_predictor].Vptsnb;
    }

    void SetPredictorVptsnbLowerLimit(int i_step, int i_predictor, double val)
    {
        m_StepsLowerLimit[i_step].Predictors[i_predictor].Vptsnb = val;
    }

    float GetPredictorWeightLowerLimit(int i_step, int i_predictor)
    {
        return m_StepsLowerLimit[i_step].Predictors[i_predictor].Weight;
    }

    void SetPredictorWeightLowerLimit(int i_step, int i_predictor, float val)
    {
        m_StepsLowerLimit[i_step].Predictors[i_predictor].Weight = val;
    }

    int GetForecastScoreAnalogsNumberLowerLimit()
    {
        return m_ForecastScoreLowerLimit.AnalogsNumber;
    }

    void SetForecastScoreAnalogsNumberLowerLimit(int val)
    {
        m_ForecastScoreLowerLimit.AnalogsNumber = val;
    }

    bool IsAnalogsNumberLocked(int i_step)
    {
        return m_StepsLocks[i_step].AnalogsNumber;
    }

    void SetAnalogsNumberLock(int i_step, bool val)
    {
        m_StepsLocks[i_step].AnalogsNumber = val;
    }

    bool IsTimeArrayAnalogsIntervalDaysLocked()
    {
        return m_TimeArrayAnalogsIntervalDaysLocks;
    }

    void SetTimeArrayAnalogsIntervalDaysLock(bool val)
    {
        m_TimeArrayAnalogsIntervalDaysLocks = val;
    }

    bool IsPreprocessDataIdLocked(int i_step, int i_predictor, int i_preprocess)
    {
        wxASSERT(m_StepsLocks[i_step].Predictors[i_predictor].PreprocessDataId.size()>(unsigned)i_preprocess);
        return m_StepsLocks[i_step].Predictors[i_predictor].PreprocessDataId[i_preprocess];
    }

    void SetPreprocessDataIdLock(int i_step, int i_predictor, int i_preprocess, bool val)
    {
        if(m_StepsLocks[i_step].Predictors[i_predictor].PreprocessDataId.size()>(unsigned)(i_preprocess))
        {
            m_StepsLocks[i_step].Predictors[i_predictor].PreprocessDataId[i_preprocess] = val;
        }
        else
        {
            wxASSERT(m_StepsLocks[i_step].Predictors[i_predictor].PreprocessDataId.size()==(unsigned)i_preprocess);
            m_StepsLocks[i_step].Predictors[i_predictor].PreprocessDataId.push_back(val);
        }
    }

    bool IsPreprocessLevelLocked(int i_step, int i_predictor, int i_preprocess)
    {
        wxASSERT(m_StepsLocks[i_step].Predictors[i_predictor].PreprocessLevels.size()>(unsigned)i_preprocess);
        return m_StepsLocks[i_step].Predictors[i_predictor].PreprocessLevels[i_preprocess];
    }

    void SetPreprocessLevelLock(int i_step, int i_predictor, int i_preprocess, bool val)
    {
        if(m_StepsLocks[i_step].Predictors[i_predictor].PreprocessLevels.size()>(unsigned)(i_preprocess))
        {
            m_StepsLocks[i_step].Predictors[i_predictor].PreprocessLevels[i_preprocess] = val;
        }
        else
        {
            wxASSERT(m_StepsLocks[i_step].Predictors[i_predictor].PreprocessLevels.size()==(unsigned)i_preprocess);
            m_StepsLocks[i_step].Predictors[i_predictor].PreprocessLevels.push_back(val);
        }
    }

    bool IsPreprocessTimeHoursLocked(int i_step, int i_predictor, int i_preprocess)
    {
        wxASSERT(m_StepsLocks[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>(unsigned)i_preprocess);
        return m_StepsLocks[i_step].Predictors[i_predictor].PreprocessTimeHours[i_preprocess];
    }

    void SetPreprocessTimeHoursLock(int i_step, int i_predictor, int i_preprocess, bool val)
    {
        if(m_StepsLocks[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>(unsigned)(i_preprocess))
        {
            m_StepsLocks[i_step].Predictors[i_predictor].PreprocessTimeHours[i_preprocess] = val;
        }
        else
        {
            wxASSERT(m_StepsLocks[i_step].Predictors[i_predictor].PreprocessTimeHours.size()==(unsigned)i_preprocess);
            m_StepsLocks[i_step].Predictors[i_predictor].PreprocessTimeHours.push_back(val);
        }
    }

    bool IsPredictorTimeHoursLocked(int i_step, int i_predictor)
    {
        return m_StepsLocks[i_step].Predictors[i_predictor].TimeHours;
    }

    void SetPredictorTimeHoursLock(int i_step, int i_predictor, bool val)
    {
        m_StepsLocks[i_step].Predictors[i_predictor].TimeHours = val;
    }

    bool IsPredictorDataIdLocked(int i_step, int i_predictor)
    {
        return m_StepsLocks[i_step].Predictors[i_predictor].DataId;
    }

    void SetPredictorDataIdLock(int i_step, int i_predictor, bool val)
    {
        m_StepsLocks[i_step].Predictors[i_predictor].DataId = val;
    }

    bool IsPredictorLevelLocked(int i_step, int i_predictor)
    {
        return m_StepsLocks[i_step].Predictors[i_predictor].Level;
    }

    void SetPredictorLevelLock(int i_step, int i_predictor, bool val)
    {
        m_StepsLocks[i_step].Predictors[i_predictor].Level = val;
    }

    bool IsPredictorUminLocked(int i_step, int i_predictor)
    {
        return m_StepsLocks[i_step].Predictors[i_predictor].Umin;
    }

    void SetPredictorUminLock(int i_step, int i_predictor, bool val)
    {
        m_StepsLocks[i_step].Predictors[i_predictor].Umin = val;
    }

    bool IsPredictorUptsnbLocked(int i_step, int i_predictor)
    {
        return m_StepsLocks[i_step].Predictors[i_predictor].Uptsnb;
    }

    void SetPredictorUptsnbLock(int i_step, int i_predictor, bool val)
    {
        m_StepsLocks[i_step].Predictors[i_predictor].Uptsnb = val;
    }

    bool IsPredictorVminLocked(int i_step, int i_predictor)
    {
        return m_StepsLocks[i_step].Predictors[i_predictor].Vmin;
    }

    void SetPredictorVminLock(int i_step, int i_predictor, bool val)
    {
        m_StepsLocks[i_step].Predictors[i_predictor].Vmin = val;
    }

    bool IsPredictorVptsnbLocked(int i_step, int i_predictor)
    {
        return m_StepsLocks[i_step].Predictors[i_predictor].Vptsnb;
    }

    void SetPredictorVptsnbLock(int i_step, int i_predictor, bool val)
    {
        m_StepsLocks[i_step].Predictors[i_predictor].Vptsnb = val;
    }

    bool IsPredictorWeightLocked(int i_step, int i_predictor)
    {
        return m_StepsLocks[i_step].Predictors[i_predictor].Weight;
    }

    void SetPredictorWeightLock(int i_step, int i_predictor, bool val)
    {
        m_StepsLocks[i_step].Predictors[i_predictor].Weight = val;
    }

    bool IsPredictorCriteriaLocked(int i_step, int i_predictor)
    {
        return m_StepsLocks[i_step].Predictors[i_predictor].Criteria;
    }

    void SetPredictorCriteriaLock(int i_step, int i_predictor, bool val)
    {
        m_StepsLocks[i_step].Predictors[i_predictor].Criteria = val;
    }

    bool IsForecastScoreAnalogsNumberLocked()
    {
        return m_ForecastScoreLocks.AnalogsNumber;
    }

    void SetForecastScoreAnalogsNumberLock(bool val)
    {
        m_ForecastScoreLocks.AnalogsNumber = val;
    }

    bool IsTimeArrayAnalogsIntervalDaysRandomInit()
    {
        return m_TimeArrayAnalogsIntervalDaysRandomInit;
    }

    void SetTimeArrayAnalogsIntervalDaysRandomInit(bool val)
    {
        m_TimeArrayAnalogsIntervalDaysRandomInit = val;
    }

    bool IsAnalogsNumberRandomInit(int i_step)
    {
        return m_StepsRandomInits[i_step].AnalogsNumber;
    }

    void SetAnalogsNumberRandomInit(int i_step, bool val)
    {
        m_StepsRandomInits[i_step].AnalogsNumber = val;
    }

    bool IsPreprocessDataIdRandomInit(int i_step, int i_predictor, int i_preprocess)
    {
        wxASSERT(m_StepsRandomInits[i_step].Predictors[i_predictor].PreprocessDataId.size()>(unsigned)i_preprocess);
        return m_StepsRandomInits[i_step].Predictors[i_predictor].PreprocessDataId[i_preprocess];
    }

    void SetPreprocessDataIdRandomInit(int i_step, int i_predictor, int i_preprocess, bool val)
    {
        if(m_StepsRandomInits[i_step].Predictors[i_predictor].PreprocessDataId.size()>(unsigned)(i_preprocess))
        {
            m_StepsRandomInits[i_step].Predictors[i_predictor].PreprocessDataId[i_preprocess] = val;
        }
        else
        {
            wxASSERT(m_StepsRandomInits[i_step].Predictors[i_predictor].PreprocessDataId.size()==(unsigned)i_preprocess);
            m_StepsRandomInits[i_step].Predictors[i_predictor].PreprocessDataId.push_back(val);
        }
    }

    bool IsPreprocessLevelRandomInit(int i_step, int i_predictor, int i_preprocess)
    {
        wxASSERT(m_StepsRandomInits[i_step].Predictors[i_predictor].PreprocessLevels.size()>(unsigned)i_preprocess);
        return m_StepsRandomInits[i_step].Predictors[i_predictor].PreprocessLevels[i_preprocess];
    }

    void SetPreprocessLevelRandomInit(int i_step, int i_predictor, int i_preprocess, bool val)
    {
        if(m_StepsRandomInits[i_step].Predictors[i_predictor].PreprocessLevels.size()>(unsigned)(i_preprocess))
        {
            m_StepsRandomInits[i_step].Predictors[i_predictor].PreprocessLevels[i_preprocess] = val;
        }
        else
        {
            wxASSERT(m_StepsRandomInits[i_step].Predictors[i_predictor].PreprocessLevels.size()==(unsigned)i_preprocess);
            m_StepsRandomInits[i_step].Predictors[i_predictor].PreprocessLevels.push_back(val);
        }
    }

    bool IsPreprocessTimeHoursRandomInit(int i_step, int i_predictor, int i_preprocess)
    {
        wxASSERT(m_StepsRandomInits[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>(unsigned)i_preprocess);
        return m_StepsRandomInits[i_step].Predictors[i_predictor].PreprocessTimeHours[i_preprocess];
    }

    void SetPreprocessTimeHoursRandomInit(int i_step, int i_predictor, int i_preprocess, bool val)
    {
        if(m_StepsRandomInits[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>(unsigned)(i_preprocess))
        {
            m_StepsRandomInits[i_step].Predictors[i_predictor].PreprocessTimeHours[i_preprocess] = val;
        }
        else
        {
            wxASSERT(m_StepsRandomInits[i_step].Predictors[i_predictor].PreprocessTimeHours.size()==(unsigned)i_preprocess);
            m_StepsRandomInits[i_step].Predictors[i_predictor].PreprocessTimeHours.push_back(val);
        }
    }

    bool IsPredictorDataIdRandomInit(int i_step, int i_predictor)
    {
        return m_StepsRandomInits[i_step].Predictors[i_predictor].DataId;
    }

    void SetPredictorDataIdRandomInit(int i_step, int i_predictor, bool val)
    {
        m_StepsRandomInits[i_step].Predictors[i_predictor].DataId = val;
    }

    bool IsPredictorLevelRandomInit(int i_step, int i_predictor)
    {
        return m_StepsRandomInits[i_step].Predictors[i_predictor].Level;
    }

    void SetPredictorLevelRandomInit(int i_step, int i_predictor, bool val)
    {
        m_StepsRandomInits[i_step].Predictors[i_predictor].Level = val;
    }

    bool IsPredictorUminRandomInit(int i_step, int i_predictor)
    {
        return m_StepsRandomInits[i_step].Predictors[i_predictor].Umin;
    }

    void SetPredictorUminRandomInit(int i_step, int i_predictor, bool val)
    {
        m_StepsRandomInits[i_step].Predictors[i_predictor].Umin = val;
    }

    bool IsPredictorUptsnbRandomInit(int i_step, int i_predictor)
    {
        return m_StepsRandomInits[i_step].Predictors[i_predictor].Uptsnb;
    }

    void SetPredictorUptsnbRandomInit(int i_step, int i_predictor, bool val)
    {
        m_StepsRandomInits[i_step].Predictors[i_predictor].Uptsnb = val;
    }

    bool IsPredictorVminRandomInit(int i_step, int i_predictor)
    {
        return m_StepsRandomInits[i_step].Predictors[i_predictor].Vmin;
    }

    void SetPredictorVminRandomInit(int i_step, int i_predictor, bool val)
    {
        m_StepsRandomInits[i_step].Predictors[i_predictor].Vmin = val;
    }

    bool IsPredictorVptsnbRandomInit(int i_step, int i_predictor)
    {
        return m_StepsRandomInits[i_step].Predictors[i_predictor].Vptsnb;
    }

    void SetPredictorVptsnbRandomInit(int i_step, int i_predictor, bool val)
    {
        m_StepsRandomInits[i_step].Predictors[i_predictor].Vptsnb = val;
    }

    bool IsPredictorTimeHoursRandomInit(int i_step, int i_predictor)
    {
        return m_StepsRandomInits[i_step].Predictors[i_predictor].TimeHours;
    }

    void SetPredictorTimeHoursRandomInit(int i_step, int i_predictor, bool val)
    {
        m_StepsRandomInits[i_step].Predictors[i_predictor].TimeHours = val;
    }

    bool IsPredictorWeightRandomInit(int i_step, int i_predictor)
    {
        return m_StepsRandomInits[i_step].Predictors[i_predictor].Weight;
    }

    void SetPredictorWeightRandomInit(int i_step, int i_predictor, bool val)
    {
        m_StepsRandomInits[i_step].Predictors[i_predictor].Weight = val;
    }

    bool IsPredictorCriteriaRandomInit(int i_step, int i_predictor)
    {
        return m_StepsRandomInits[i_step].Predictors[i_predictor].Criteria;
    }

    void SetPredictorCriteriaRandomInit(int i_step, int i_predictor, bool val)
    {
        m_StepsRandomInits[i_step].Predictors[i_predictor].Criteria = val;
    }

    bool IsForecastScoreAnalogsNumberRandomInit()
    {
        return m_ForecastScoreRandomInits.AnalogsNumber;
    }

    void SetForecastScoreAnalogsNumberRandomInit(bool val)
    {
        m_ForecastScoreRandomInits.AnalogsNumber = val;
    }

    bool IncrementAnalogsNumber(int i_step)
    {
        if (GetAnalogsNumber(i_step)+m_StepsIteration[i_step].AnalogsNumber <= m_StepsUpperLimit[i_step].AnalogsNumber)
        {
            SetAnalogsNumber(i_step, GetAnalogsNumber(i_step) + m_StepsIteration[i_step].AnalogsNumber);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool IncrementTimeArrayAnalogsIntervalDays()
    {
        if (m_TimeArrayAnalogsIntervalDays+m_TimeArrayAnalogsIntervalDaysIteration <= m_TimeArrayAnalogsIntervalDaysUpperLimit)
        {
            m_TimeArrayAnalogsIntervalDays += m_TimeArrayAnalogsIntervalDaysIteration;
            return true;
        }
        else
        {
            return false;
        }
    }

    bool IncrementPredictorUmin(int i_step, int i_predictor)
    {
        if (GetPredictorUmin(i_step,i_predictor)+m_StepsIteration[i_step].Predictors[i_predictor].Umin <= m_StepsUpperLimit[i_step].Predictors[i_predictor].Umin)
        {
            SetPredictorUmin(i_step,i_predictor, GetPredictorUmin(i_step,i_predictor) + m_StepsIteration[i_step].Predictors[i_predictor].Umin);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool IncrementPredictorUptsnb(int i_step, int i_predictor)
    {
        if (GetPredictorUptsnb(i_step,i_predictor)+m_StepsIteration[i_step].Predictors[i_predictor].Uptsnb <= m_StepsUpperLimit[i_step].Predictors[i_predictor].Uptsnb)
        {
            SetPredictorUptsnb(i_step,i_predictor, GetPredictorUptsnb(i_step,i_predictor) + m_StepsIteration[i_step].Predictors[i_predictor].Uptsnb);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool IncrementPredictorVmin(int i_step, int i_predictor)
    {
        if (GetPredictorVmin(i_step,i_predictor)+m_StepsIteration[i_step].Predictors[i_predictor].Vmin <= m_StepsUpperLimit[i_step].Predictors[i_predictor].Vmin)
        {
            SetPredictorVmin(i_step,i_predictor, GetPredictorVmin(i_step,i_predictor) + m_StepsIteration[i_step].Predictors[i_predictor].Vmin);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool IncrementPredictorVptsnb(int i_step, int i_predictor)
    {
        if (GetPredictorVptsnb(i_step,i_predictor)+m_StepsIteration[i_step].Predictors[i_predictor].Vptsnb <= m_StepsUpperLimit[i_step].Predictors[i_predictor].Vptsnb)
        {
            SetPredictorVptsnb(i_step,i_predictor, GetPredictorVptsnb(i_step,i_predictor) + m_StepsIteration[i_step].Predictors[i_predictor].Vptsnb);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool IncrementPredictorTimeHours(int i_step, int i_predictor)
    {
        if (GetPredictorTimeHours(i_step,i_predictor)+m_StepsIteration[i_step].Predictors[i_predictor].TimeHours <= m_StepsUpperLimit[i_step].Predictors[i_predictor].TimeHours)
        {
            SetPredictorTimeHours(i_step,i_predictor, GetPredictorTimeHours(i_step,i_predictor) + m_StepsIteration[i_step].Predictors[i_predictor].TimeHours);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool IncrementPredictorWeight(int i_step, int i_predictor)
    {
        if (GetPredictorWeight(i_step,i_predictor)+m_StepsIteration[i_step].Predictors[i_predictor].Weight <= m_StepsUpperLimit[i_step].Predictors[i_predictor].Weight)
        {
            SetPredictorWeight(i_step,i_predictor, GetPredictorWeight(i_step,i_predictor) + m_StepsIteration[i_step].Predictors[i_predictor].Weight);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool IncrementForecastScoreAnalogsNumber()
    {
        if (GetForecastScoreAnalogsNumber()+m_ForecastScoreIteration.AnalogsNumber <= m_ForecastScoreUpperLimit.AnalogsNumber)
        {
            SetForecastScoreAnalogsNumber(GetForecastScoreAnalogsNumber() + m_ForecastScoreIteration.AnalogsNumber);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool DecrementAnalogsNumber(int i_step)
    {
        if (GetAnalogsNumber(i_step)-m_StepsIteration[i_step].AnalogsNumber >= m_StepsLowerLimit[i_step].AnalogsNumber)
        {
            SetAnalogsNumber(i_step, GetAnalogsNumber(i_step) - m_StepsIteration[i_step].AnalogsNumber);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool DecrementTimeArrayAnalogsIntervalDays(int i_step)
    {
        if (m_TimeArrayAnalogsIntervalDays-m_TimeArrayAnalogsIntervalDaysIteration >= m_TimeArrayAnalogsIntervalDaysLowerLimit)
        {
            m_TimeArrayAnalogsIntervalDays -= m_TimeArrayAnalogsIntervalDaysIteration;
            return true;
        }
        else
        {
            return false;
        }
    }

    bool DecrementPredictorUmin(int i_step, int i_predictor)
    {
        if (GetPredictorUmin(i_step,i_predictor)-m_StepsIteration[i_step].Predictors[i_predictor].Umin >= m_StepsLowerLimit[i_step].Predictors[i_predictor].Umin)
        {
            SetPredictorUmin(i_step,i_predictor, GetPredictorUmin(i_step,i_predictor) - m_StepsIteration[i_step].Predictors[i_predictor].Umin);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool DecrementPredictorUptsnb(int i_step, int i_predictor)
    {
        if (GetPredictorUptsnb(i_step,i_predictor)-m_StepsIteration[i_step].Predictors[i_predictor].Uptsnb >= m_StepsLowerLimit[i_step].Predictors[i_predictor].Uptsnb)
        {
            SetPredictorUptsnb(i_step,i_predictor, GetPredictorUptsnb(i_step,i_predictor) - m_StepsIteration[i_step].Predictors[i_predictor].Uptsnb);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool DecrementPredictorVmin(int i_step, int i_predictor)
    {
        if (GetPredictorVmin(i_step,i_predictor)-m_StepsIteration[i_step].Predictors[i_predictor].Vmin >= m_StepsLowerLimit[i_step].Predictors[i_predictor].Vmin)
        {
            SetPredictorVmin(i_step,i_predictor, GetPredictorVmin(i_step,i_predictor) - m_StepsIteration[i_step].Predictors[i_predictor].Vmin);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool DecrementPredictorVptsnb(int i_step, int i_predictor)
    {
        if (GetPredictorVptsnb(i_step,i_predictor)-m_StepsIteration[i_step].Predictors[i_predictor].Vptsnb >= m_StepsLowerLimit[i_step].Predictors[i_predictor].Vptsnb)
        {
            SetPredictorVptsnb(i_step,i_predictor, GetPredictorVptsnb(i_step,i_predictor) - m_StepsIteration[i_step].Predictors[i_predictor].Vptsnb);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool DecrementPredictorTimeHours(int i_step, int i_predictor)
    {
        if (GetPredictorTimeHours(i_step,i_predictor)-m_StepsIteration[i_step].Predictors[i_predictor].TimeHours >= m_StepsLowerLimit[i_step].Predictors[i_predictor].TimeHours)
        {
            SetPredictorTimeHours(i_step,i_predictor, GetPredictorTimeHours(i_step,i_predictor) - m_StepsIteration[i_step].Predictors[i_predictor].TimeHours);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool DecrementPredictorWeight(int i_step, int i_predictor)
    {
        if (GetPredictorWeight(i_step,i_predictor)-m_StepsIteration[i_step].Predictors[i_predictor].Weight >= m_StepsLowerLimit[i_step].Predictors[i_predictor].Weight)
        {
            SetPredictorWeight(i_step,i_predictor, GetPredictorWeight(i_step,i_predictor) - m_StepsIteration[i_step].Predictors[i_predictor].Weight);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool DecrementForecastScoreAnalogsNumber()
    {
        if (GetForecastScoreAnalogsNumber()-m_ForecastScoreIteration.AnalogsNumber >= m_ForecastScoreLowerLimit.AnalogsNumber)
        {
            SetForecastScoreAnalogsNumber(GetForecastScoreAnalogsNumber() - m_ForecastScoreIteration.AnalogsNumber);
            return true;
        }
        else
        {
            return false;
        }
    }

    /* Vector elements */

    VectorString GetPreprocessDataIdVector(int i_step, int i_predictor, int i_dataset)
    {
        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessDataId.size()>=(unsigned)(i_dataset+1))
        {
            return m_StepsVect[i_step].Predictors[i_predictor].PreprocessDataId[i_dataset];
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessDataId in the parameters object."));
            VectorString empty;
            return empty;
        }
    }

    void SetPreprocessDataIdVector(int i_step, int i_predictor, int i_dataset, VectorString val)
    {
        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessDataId.size()>=(unsigned)(i_dataset+1))
        {
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessDataId[i_dataset].clear();
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessDataId[i_dataset] = val;
        }
        else
        {
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessDataId.push_back(val);
        }
    }

    VectorFloat GetPreprocessLevelVector(int i_step, int i_predictor, int i_dataset)
    {
        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels.size()>=(unsigned)(i_dataset+1))
        {
            return m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset];
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessLevels in the parameters object."));
            VectorFloat empty;
            return empty;
        }
    }

    void SetPreprocessLevelVector(int i_step, int i_predictor, int i_dataset, VectorFloat val)
    {
        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels.size()>=(unsigned)(i_dataset+1))
        {
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset].clear();
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset] = val;
        }
        else
        {
            wxASSERT(m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels.size()==(unsigned)i_dataset);
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels.push_back(val);
        }
    }

    VectorDouble GetPreprocessTimeHoursVector(int i_step, int i_predictor, int i_dataset)
    {
        wxASSERT(m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>(unsigned)i_dataset);

        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>=(unsigned)(i_dataset+1))
        {
            return m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset];
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessTimeHours (vect in optimization) in the parameters object."));
            VectorDouble empty;
            return empty;
        }
    }

    VectorString GetPredictorDataIdVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].DataId;
    }

    void SetPredictorDataIdVector(int i_step, int i_predictor, VectorString val)
    {
        m_StepsVect[i_step].Predictors[i_predictor].DataId = val;
    }

    VectorFloat GetPredictorLevelVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Level;
    }

    void SetPredictorLevelVector(int i_step, int i_predictor, VectorFloat val)
    {
        m_StepsVect[i_step].Predictors[i_predictor].Level = val;
    }

    VectorString GetPredictorCriteriaVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Criteria;
    }

    void SetPredictorCriteriaVector(int i_step, int i_predictor, VectorString val)
    {
        m_StepsVect[i_step].Predictors[i_predictor].Criteria = val;
    }


protected:
    int m_VariableParamsNb;
    int m_TimeArrayAnalogsIntervalDaysIteration;
    int m_TimeArrayAnalogsIntervalDaysUpperLimit;
    int m_TimeArrayAnalogsIntervalDaysLowerLimit;
    bool m_TimeArrayAnalogsIntervalDaysLocks;
    bool m_TimeArrayAnalogsIntervalDaysRandomInit;
    VectorParamsStep m_StepsIteration;
    VectorParamsStep m_StepsUpperLimit;
    VectorParamsStep m_StepsLowerLimit;
    VectorParamsStepBool m_StepsLocks;
    VectorParamsStepBool m_StepsRandomInits;
    ParamsForecastScore m_ForecastScoreIteration;
    ParamsForecastScore m_ForecastScoreUpperLimit;
    ParamsForecastScore m_ForecastScoreLowerLimit;
    ParamsForecastScoreBool m_ForecastScoreLocks;
    ParamsForecastScoreBool m_ForecastScoreRandomInits;
    VectorParamsStepVect m_StepsVect;

private:


};

#endif // ASPARAMETERSOPTIMIZATION_H
