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

    void InitRandomValues();

    bool LoadFromFile(const wxString &filePath);

    void CheckRange();

    bool IsInRange();

    bool FixTimeLimits();

    void FixTimeHours();

    bool FixWeights();

    void LockAll();

    void Unlock(VectorInt &indices);

    // May vary
    int GetVariablesNb();

    // Does not change after importation from file.
    int GetVariableParamsNb()
    {
        return m_variableParamsNb;
    }

    int GetTimeArrayAnalogsIntervalDaysIteration()
    {
        return m_timeArrayAnalogsIntervalDaysIteration;
    }

    bool SetTimeArrayAnalogsIntervalDaysIteration(int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the 'days interval' is null"));
            return false;
        }
        m_timeArrayAnalogsIntervalDaysIteration = val;
        return true;
    }

    int GetAnalogsNumberIteration(int i_step)
    {
        return m_stepsIteration[i_step].AnalogsNumber;
    }

    bool SetAnalogsNumberIteration(int i_step, int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the analogs number is null"));
            return false;
        }
        m_stepsIteration[i_step].AnalogsNumber = val;
        return true;
    }

    double GetPreprocessTimeHoursIteration(int i_step, int i_predictor, int i_dataset)
    {
        wxASSERT(m_stepsLowerLimit[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>(unsigned)i_dataset);
        return m_stepsIteration[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset];
    }

    bool SetPreprocessTimeHoursIteration(int i_step, int i_predictor, int i_dataset, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided iteration value for the preprocess time frame is null"));
            return false;
        }

        if(m_stepsIteration[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>=(unsigned)(i_dataset+1))
        {
            m_stepsIteration[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset] = val;
        }
        else
        {
            wxASSERT(m_stepsIteration[i_step].Predictors[i_predictor].PreprocessTimeHours.size()==(unsigned)i_dataset);
            m_stepsIteration[i_step].Predictors[i_predictor].PreprocessTimeHours.push_back(val);
        }
        return true;
    }

    double GetPredictorTimeHoursIteration(int i_step, int i_predictor)
    {
        return m_stepsIteration[i_step].Predictors[i_predictor].TimeHours;
    }

    bool SetPredictorTimeHoursIteration(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the time frame is null"));
            return false;
        }
        m_stepsIteration[i_step].Predictors[i_predictor].TimeHours = val;
        return true;
    }

    double GetPredictorXminIteration(int i_step, int i_predictor)
    {
        return m_stepsIteration[i_step].Predictors[i_predictor].Xmin;
    }

    bool SetPredictorXminIteration(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for Xmin is null"));
            return false;
        }
        m_stepsIteration[i_step].Predictors[i_predictor].Xmin = val;
        return true;
    }

    int GetPredictorXptsnbIteration(int i_step, int i_predictor)
    {
        return m_stepsIteration[i_step].Predictors[i_predictor].Xptsnb;
    }

    bool SetPredictorXptsnbIteration(int i_step, int i_predictor, int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for Xptsnb is null"));
            return false;
        }
        m_stepsIteration[i_step].Predictors[i_predictor].Xptsnb = val;
        return true;
    }

    double GetPredictorYminIteration(int i_step, int i_predictor)
    {
        return m_stepsIteration[i_step].Predictors[i_predictor].Ymin;
    }

    bool SetPredictorYminIteration(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for Ymin is null"));
            return false;
        }
        m_stepsIteration[i_step].Predictors[i_predictor].Ymin = val;
        return true;
    }

    int GetPredictorYptsnbIteration(int i_step, int i_predictor)
    {
        return m_stepsIteration[i_step].Predictors[i_predictor].Yptsnb;
    }

    bool SetPredictorYptsnbIteration(int i_step, int i_predictor, int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for Yptsnb is null"));
            return false;
        }
        m_stepsIteration[i_step].Predictors[i_predictor].Yptsnb = val;
        return true;
    }

    float GetPredictorWeightIteration(int i_step, int i_predictor)
    {
        return m_stepsIteration[i_step].Predictors[i_predictor].Weight;
    }

    bool SetPredictorWeightIteration(int i_step, int i_predictor, float val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the predictor weight is null"));
            return false;
        }
        m_stepsIteration[i_step].Predictors[i_predictor].Weight = val;
        return true;
    }

    int GetTimeArrayAnalogsIntervalDaysUpperLimit()
    {
        return m_timeArrayAnalogsIntervalDaysUpperLimit;
    }

    bool SetTimeArrayAnalogsIntervalDaysUpperLimit(int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the 'days interval' is null"));
            return false;
        }
        m_timeArrayAnalogsIntervalDaysUpperLimit = val;
        return true;
    }

    int GetAnalogsNumberUpperLimit(int i_step)
    {
        return m_stepsUpperLimit[i_step].AnalogsNumber;
    }

    bool SetAnalogsNumberUpperLimit(int i_step, int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the analogs number is null"));
            return false;
        }
        m_stepsUpperLimit[i_step].AnalogsNumber = val;
        return true;
    }

    double GetPreprocessTimeHoursUpperLimit(int i_step, int i_predictor, int i_dataset)
    {
        wxASSERT(m_stepsUpperLimit[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>(unsigned)i_dataset);
        return m_stepsUpperLimit[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset];
    }

    bool SetPreprocessTimeHoursUpperLimit(int i_step, int i_predictor, int i_dataset, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided upper value value for the preprocess time frame is null"));
            return false;
        }

        if(m_stepsUpperLimit[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>=(unsigned)(i_dataset+1))
        {
            m_stepsUpperLimit[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset] = val;
        }
        else
        {
            wxASSERT(m_stepsUpperLimit[i_step].Predictors[i_predictor].PreprocessTimeHours.size()==(unsigned)i_dataset);
            m_stepsUpperLimit[i_step].Predictors[i_predictor].PreprocessTimeHours.push_back(val);
        }
        return true;
    }

    double GetPredictorTimeHoursUpperLimit(int i_step, int i_predictor)
    {
        return m_stepsUpperLimit[i_step].Predictors[i_predictor].TimeHours;
    }

    bool SetPredictorTimeHoursUpperLimit(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the time frame is null"));
            return false;
        }
        m_stepsUpperLimit[i_step].Predictors[i_predictor].TimeHours = val;
        return true;
    }

    double GetPredictorXminUpperLimit(int i_step, int i_predictor)
    {
        return m_stepsUpperLimit[i_step].Predictors[i_predictor].Xmin;
    }

    bool SetPredictorXminUpperLimit(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for Xmin is null"));
            return false;
        }
        m_stepsUpperLimit[i_step].Predictors[i_predictor].Xmin = val;
        return true;
    }

    int GetPredictorXptsnbUpperLimit(int i_step, int i_predictor)
    {
        return m_stepsUpperLimit[i_step].Predictors[i_predictor].Xptsnb;
    }

    bool SetPredictorXptsnbUpperLimit(int i_step, int i_predictor, int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for Xptsnb is null"));
            return false;
        }
        m_stepsUpperLimit[i_step].Predictors[i_predictor].Xptsnb = val;
        return true;
    }

    double GetPredictorYminUpperLimit(int i_step, int i_predictor)
    {
        return m_stepsUpperLimit[i_step].Predictors[i_predictor].Ymin;
    }

    bool SetPredictorYminUpperLimit(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for Ymin is null"));
            return false;
        }
        m_stepsUpperLimit[i_step].Predictors[i_predictor].Ymin = val;
        return true;
    }

    int GetPredictorYptsnbUpperLimit(int i_step, int i_predictor)
    {
        return m_stepsUpperLimit[i_step].Predictors[i_predictor].Yptsnb;
    }

    bool SetPredictorYptsnbUpperLimit(int i_step, int i_predictor, int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for Yptsnb is null"));
            return false;
        }
        m_stepsUpperLimit[i_step].Predictors[i_predictor].Yptsnb = val;
        return true;
    }

    float GetPredictorWeightUpperLimit(int i_step, int i_predictor)
    {
        return m_stepsUpperLimit[i_step].Predictors[i_predictor].Weight;
    }

    bool SetPredictorWeightUpperLimit(int i_step, int i_predictor, float val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the predictor weight is null"));
            return false;
        }
        m_stepsUpperLimit[i_step].Predictors[i_predictor].Weight = val;
        return true;
    }

    int GetTimeArrayAnalogsIntervalDaysLowerLimit()
    {
        return m_timeArrayAnalogsIntervalDaysLowerLimit;
    }

    bool SetTimeArrayAnalogsIntervalDaysLowerLimit(int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the 'days interval' is null"));
            return false;
        }
        m_timeArrayAnalogsIntervalDaysLowerLimit = val;
        return true;
    }

    int GetAnalogsNumberLowerLimit(int i_step)
    {
        return m_stepsLowerLimit[i_step].AnalogsNumber;
    }

    bool SetAnalogsNumberLowerLimit(int i_step, int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the analogs number is null"));
            return false;
        }
        m_stepsLowerLimit[i_step].AnalogsNumber = val;
        return true;
    }

    double GetPreprocessTimeHoursLowerLimit(int i_step, int i_predictor, int i_dataset)
    {
        wxASSERT(m_stepsLowerLimit[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>(unsigned)i_dataset);
        return m_stepsLowerLimit[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset];
    }

    bool SetPreprocessTimeHoursLowerLimit(int i_step, int i_predictor, int i_dataset, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided lower value value for the preprocess time frame is null"));
            return false;
        }

        if(m_stepsLowerLimit[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>=(unsigned)(i_dataset+1))
        {
            m_stepsLowerLimit[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset] = val;
        }
        else
        {
            wxASSERT(m_stepsLowerLimit[i_step].Predictors[i_predictor].PreprocessTimeHours.size()==(unsigned)i_dataset);
            m_stepsLowerLimit[i_step].Predictors[i_predictor].PreprocessTimeHours.push_back(val);
        }
        return true;
    }

    double GetPredictorTimeHoursLowerLimit(int i_step, int i_predictor)
    {
        return m_stepsLowerLimit[i_step].Predictors[i_predictor].TimeHours;
    }

    bool SetPredictorTimeHoursLowerLimit(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the time frame is null"));
            return false;
        }
        m_stepsLowerLimit[i_step].Predictors[i_predictor].TimeHours = val;
        return true;
    }

    double GetPredictorXminLowerLimit(int i_step, int i_predictor)
    {
        return m_stepsLowerLimit[i_step].Predictors[i_predictor].Xmin;
    }

    bool SetPredictorXminLowerLimit(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for Xmin is null"));
            return false;
        }
        m_stepsLowerLimit[i_step].Predictors[i_predictor].Xmin = val;
        return true;
    }

    int GetPredictorXptsnbLowerLimit(int i_step, int i_predictor)
    {
        return m_stepsLowerLimit[i_step].Predictors[i_predictor].Xptsnb;
    }

    bool SetPredictorXptsnbLowerLimit(int i_step, int i_predictor, int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for Xptsnb is null"));
            return false;
        }
        m_stepsLowerLimit[i_step].Predictors[i_predictor].Xptsnb = val;
        return true;
    }

    double GetPredictorYminLowerLimit(int i_step, int i_predictor)
    {
        return m_stepsLowerLimit[i_step].Predictors[i_predictor].Ymin;
    }

    bool SetPredictorYminLowerLimit(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for Ymin is null"));
            return false;
        }
        m_stepsLowerLimit[i_step].Predictors[i_predictor].Ymin = val;
        return true;
    }

    int GetPredictorYptsnbLowerLimit(int i_step, int i_predictor)
    {
        return m_stepsLowerLimit[i_step].Predictors[i_predictor].Yptsnb;
    }

    bool SetPredictorYptsnbLowerLimit(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for Yptsnb is null"));
            return false;
        }
        m_stepsLowerLimit[i_step].Predictors[i_predictor].Yptsnb = val;
        return true;
    }

    float GetPredictorWeightLowerLimit(int i_step, int i_predictor)
    {
        return m_stepsLowerLimit[i_step].Predictors[i_predictor].Weight;
    }

    bool SetPredictorWeightLowerLimit(int i_step, int i_predictor, float val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the predictor weight is null"));
            return false;
        }
        m_stepsLowerLimit[i_step].Predictors[i_predictor].Weight = val;
        return true;
    }

    bool IsAnalogsNumberLocked(int i_step)
    {
        return m_stepsLocks[i_step].AnalogsNumber;
    }

    void SetAnalogsNumberLock(int i_step, bool val)
    {
        m_stepsLocks[i_step].AnalogsNumber = val;
    }

    bool IsTimeArrayAnalogsIntervalDaysLocked()
    {
        return m_timeArrayAnalogsIntervalDaysLocks;
    }

    void SetTimeArrayAnalogsIntervalDaysLock(bool val)
    {
        m_timeArrayAnalogsIntervalDaysLocks = val;
    }

    bool IsPreprocessDataIdLocked(int i_step, int i_predictor, int i_preprocess)
    {
        wxASSERT(m_stepsLocks[i_step].Predictors[i_predictor].PreprocessDataId.size()>(unsigned)i_preprocess);
        return m_stepsLocks[i_step].Predictors[i_predictor].PreprocessDataId[i_preprocess];
    }

    void SetPreprocessDataIdLock(int i_step, int i_predictor, int i_preprocess, bool val)
    {
        if(m_stepsLocks[i_step].Predictors[i_predictor].PreprocessDataId.size()>(unsigned)(i_preprocess))
        {
            m_stepsLocks[i_step].Predictors[i_predictor].PreprocessDataId[i_preprocess] = val;
        }
        else
        {
            wxASSERT(m_stepsLocks[i_step].Predictors[i_predictor].PreprocessDataId.size()==(unsigned)i_preprocess);
            m_stepsLocks[i_step].Predictors[i_predictor].PreprocessDataId.push_back(val);
        }
    }

    bool IsPreprocessLevelLocked(int i_step, int i_predictor, int i_preprocess)
    {
        wxASSERT(m_stepsLocks[i_step].Predictors[i_predictor].PreprocessLevels.size()>(unsigned)i_preprocess);
        return m_stepsLocks[i_step].Predictors[i_predictor].PreprocessLevels[i_preprocess];
    }

    void SetPreprocessLevelLock(int i_step, int i_predictor, int i_preprocess, bool val)
    {
        if(m_stepsLocks[i_step].Predictors[i_predictor].PreprocessLevels.size()>(unsigned)(i_preprocess))
        {
            m_stepsLocks[i_step].Predictors[i_predictor].PreprocessLevels[i_preprocess] = val;
        }
        else
        {
            wxASSERT(m_stepsLocks[i_step].Predictors[i_predictor].PreprocessLevels.size()==(unsigned)i_preprocess);
            m_stepsLocks[i_step].Predictors[i_predictor].PreprocessLevels.push_back(val);
        }
    }

    bool IsPreprocessTimeHoursLocked(int i_step, int i_predictor, int i_preprocess)
    {
        wxASSERT(m_stepsLocks[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>(unsigned)i_preprocess);
        return m_stepsLocks[i_step].Predictors[i_predictor].PreprocessTimeHours[i_preprocess];
    }

    void SetPreprocessTimeHoursLock(int i_step, int i_predictor, int i_preprocess, bool val)
    {
        if(m_stepsLocks[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>(unsigned)(i_preprocess))
        {
            m_stepsLocks[i_step].Predictors[i_predictor].PreprocessTimeHours[i_preprocess] = val;
        }
        else
        {
            wxASSERT(m_stepsLocks[i_step].Predictors[i_predictor].PreprocessTimeHours.size()==(unsigned)i_preprocess);
            m_stepsLocks[i_step].Predictors[i_predictor].PreprocessTimeHours.push_back(val);
        }
    }

    bool IsPredictorTimeHoursLocked(int i_step, int i_predictor)
    {
        return m_stepsLocks[i_step].Predictors[i_predictor].TimeHours;
    }

    void SetPredictorTimeHoursLock(int i_step, int i_predictor, bool val)
    {
        m_stepsLocks[i_step].Predictors[i_predictor].TimeHours = val;
    }

    bool IsPredictorDataIdLocked(int i_step, int i_predictor)
    {
        return m_stepsLocks[i_step].Predictors[i_predictor].DataId;
    }

    void SetPredictorDataIdLock(int i_step, int i_predictor, bool val)
    {
        m_stepsLocks[i_step].Predictors[i_predictor].DataId = val;
    }

    bool IsPredictorLevelLocked(int i_step, int i_predictor)
    {
        return m_stepsLocks[i_step].Predictors[i_predictor].Level;
    }

    void SetPredictorLevelLock(int i_step, int i_predictor, bool val)
    {
        m_stepsLocks[i_step].Predictors[i_predictor].Level = val;
    }

    bool IsPredictorXminLocked(int i_step, int i_predictor)
    {
        return m_stepsLocks[i_step].Predictors[i_predictor].Xmin;
    }

    void SetPredictorXminLock(int i_step, int i_predictor, bool val)
    {
        m_stepsLocks[i_step].Predictors[i_predictor].Xmin = val;
    }

    bool IsPredictorXptsnbLocked(int i_step, int i_predictor)
    {
        return m_stepsLocks[i_step].Predictors[i_predictor].Xptsnb;
    }

    void SetPredictorXptsnbLock(int i_step, int i_predictor, bool val)
    {
        m_stepsLocks[i_step].Predictors[i_predictor].Xptsnb = val;
    }

    bool IsPredictorYminLocked(int i_step, int i_predictor)
    {
        return m_stepsLocks[i_step].Predictors[i_predictor].Ymin;
    }

    void SetPredictorYminLock(int i_step, int i_predictor, bool val)
    {
        m_stepsLocks[i_step].Predictors[i_predictor].Ymin = val;
    }

    bool IsPredictorYptsnbLocked(int i_step, int i_predictor)
    {
        return m_stepsLocks[i_step].Predictors[i_predictor].Yptsnb;
    }

    void SetPredictorYptsnbLock(int i_step, int i_predictor, bool val)
    {
        m_stepsLocks[i_step].Predictors[i_predictor].Yptsnb = val;
    }

    bool IsPredictorWeightLocked(int i_step, int i_predictor)
    {
        return m_stepsLocks[i_step].Predictors[i_predictor].Weight;
    }

    void SetPredictorWeightLock(int i_step, int i_predictor, bool val)
    {
        m_stepsLocks[i_step].Predictors[i_predictor].Weight = val;
    }

    bool IsPredictorCriteriaLocked(int i_step, int i_predictor)
    {
        return m_stepsLocks[i_step].Predictors[i_predictor].Criteria;
    }

    void SetPredictorCriteriaLock(int i_step, int i_predictor, bool val)
    {
        m_stepsLocks[i_step].Predictors[i_predictor].Criteria = val;
    }

    bool IncrementAnalogsNumber(int i_step)
    {
        if (GetAnalogsNumber(i_step)+m_stepsIteration[i_step].AnalogsNumber <= m_stepsUpperLimit[i_step].AnalogsNumber)
        {
            SetAnalogsNumber(i_step, GetAnalogsNumber(i_step) + m_stepsIteration[i_step].AnalogsNumber);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool IncrementTimeArrayAnalogsIntervalDays()
    {
        if (m_timeArrayAnalogsIntervalDays+m_timeArrayAnalogsIntervalDaysIteration <= m_timeArrayAnalogsIntervalDaysUpperLimit)
        {
            m_timeArrayAnalogsIntervalDays += m_timeArrayAnalogsIntervalDaysIteration;
            return true;
        }
        else
        {
            return false;
        }
    }

    bool IncrementPredictorXmin(int i_step, int i_predictor)
    {
        if (GetPredictorXmin(i_step,i_predictor)+m_stepsIteration[i_step].Predictors[i_predictor].Xmin <= m_stepsUpperLimit[i_step].Predictors[i_predictor].Xmin)
        {
            SetPredictorXmin(i_step,i_predictor, GetPredictorXmin(i_step,i_predictor) + m_stepsIteration[i_step].Predictors[i_predictor].Xmin);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool IncrementPredictorXptsnb(int i_step, int i_predictor)
    {
        if (GetPredictorXptsnb(i_step,i_predictor)+m_stepsIteration[i_step].Predictors[i_predictor].Xptsnb <= m_stepsUpperLimit[i_step].Predictors[i_predictor].Xptsnb)
        {
            SetPredictorXptsnb(i_step,i_predictor, GetPredictorXptsnb(i_step,i_predictor) + m_stepsIteration[i_step].Predictors[i_predictor].Xptsnb);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool IncrementPredictorYmin(int i_step, int i_predictor)
    {
        if (GetPredictorYmin(i_step,i_predictor)+m_stepsIteration[i_step].Predictors[i_predictor].Ymin <= m_stepsUpperLimit[i_step].Predictors[i_predictor].Ymin)
        {
            SetPredictorYmin(i_step,i_predictor, GetPredictorYmin(i_step,i_predictor) + m_stepsIteration[i_step].Predictors[i_predictor].Ymin);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool IncrementPredictorYptsnb(int i_step, int i_predictor)
    {
        if (GetPredictorYptsnb(i_step,i_predictor)+m_stepsIteration[i_step].Predictors[i_predictor].Yptsnb <= m_stepsUpperLimit[i_step].Predictors[i_predictor].Yptsnb)
        {
            SetPredictorYptsnb(i_step,i_predictor, GetPredictorYptsnb(i_step,i_predictor) + m_stepsIteration[i_step].Predictors[i_predictor].Yptsnb);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool IncrementPredictorTimeHours(int i_step, int i_predictor)
    {
        if (GetPredictorTimeHours(i_step,i_predictor)+m_stepsIteration[i_step].Predictors[i_predictor].TimeHours <= m_stepsUpperLimit[i_step].Predictors[i_predictor].TimeHours)
        {
            SetPredictorTimeHours(i_step,i_predictor, GetPredictorTimeHours(i_step,i_predictor) + m_stepsIteration[i_step].Predictors[i_predictor].TimeHours);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool IncrementPredictorWeight(int i_step, int i_predictor)
    {
        if (GetPredictorWeight(i_step,i_predictor)+m_stepsIteration[i_step].Predictors[i_predictor].Weight <= m_stepsUpperLimit[i_step].Predictors[i_predictor].Weight)
        {
            SetPredictorWeight(i_step,i_predictor, GetPredictorWeight(i_step,i_predictor) + m_stepsIteration[i_step].Predictors[i_predictor].Weight);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool DecrementAnalogsNumber(int i_step)
    {
        if (GetAnalogsNumber(i_step)-m_stepsIteration[i_step].AnalogsNumber >= m_stepsLowerLimit[i_step].AnalogsNumber)
        {
            SetAnalogsNumber(i_step, GetAnalogsNumber(i_step) - m_stepsIteration[i_step].AnalogsNumber);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool DecrementTimeArrayAnalogsIntervalDays(int i_step)
    {
        if (m_timeArrayAnalogsIntervalDays-m_timeArrayAnalogsIntervalDaysIteration >= m_timeArrayAnalogsIntervalDaysLowerLimit)
        {
            m_timeArrayAnalogsIntervalDays -= m_timeArrayAnalogsIntervalDaysIteration;
            return true;
        }
        else
        {
            return false;
        }
    }

    bool DecrementPredictorXmin(int i_step, int i_predictor)
    {
        if (GetPredictorXmin(i_step,i_predictor)-m_stepsIteration[i_step].Predictors[i_predictor].Xmin >= m_stepsLowerLimit[i_step].Predictors[i_predictor].Xmin)
        {
            SetPredictorXmin(i_step,i_predictor, GetPredictorXmin(i_step,i_predictor) - m_stepsIteration[i_step].Predictors[i_predictor].Xmin);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool DecrementPredictorXptsnb(int i_step, int i_predictor)
    {
        if (GetPredictorXptsnb(i_step,i_predictor)-m_stepsIteration[i_step].Predictors[i_predictor].Xptsnb >= m_stepsLowerLimit[i_step].Predictors[i_predictor].Xptsnb)
        {
            SetPredictorXptsnb(i_step,i_predictor, GetPredictorXptsnb(i_step,i_predictor) - m_stepsIteration[i_step].Predictors[i_predictor].Xptsnb);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool DecrementPredictorYmin(int i_step, int i_predictor)
    {
        if (GetPredictorYmin(i_step,i_predictor)-m_stepsIteration[i_step].Predictors[i_predictor].Ymin >= m_stepsLowerLimit[i_step].Predictors[i_predictor].Ymin)
        {
            SetPredictorYmin(i_step,i_predictor, GetPredictorYmin(i_step,i_predictor) - m_stepsIteration[i_step].Predictors[i_predictor].Ymin);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool DecrementPredictorYptsnb(int i_step, int i_predictor)
    {
        if (GetPredictorYptsnb(i_step,i_predictor)-m_stepsIteration[i_step].Predictors[i_predictor].Yptsnb >= m_stepsLowerLimit[i_step].Predictors[i_predictor].Yptsnb)
        {
            SetPredictorYptsnb(i_step,i_predictor, GetPredictorYptsnb(i_step,i_predictor) - m_stepsIteration[i_step].Predictors[i_predictor].Yptsnb);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool DecrementPredictorTimeHours(int i_step, int i_predictor)
    {
        if (GetPredictorTimeHours(i_step,i_predictor)-m_stepsIteration[i_step].Predictors[i_predictor].TimeHours >= m_stepsLowerLimit[i_step].Predictors[i_predictor].TimeHours)
        {
            SetPredictorTimeHours(i_step,i_predictor, GetPredictorTimeHours(i_step,i_predictor) - m_stepsIteration[i_step].Predictors[i_predictor].TimeHours);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool DecrementPredictorWeight(int i_step, int i_predictor)
    {
        if (GetPredictorWeight(i_step,i_predictor)-m_stepsIteration[i_step].Predictors[i_predictor].Weight >= m_stepsLowerLimit[i_step].Predictors[i_predictor].Weight)
        {
            SetPredictorWeight(i_step,i_predictor, GetPredictorWeight(i_step,i_predictor) - m_stepsIteration[i_step].Predictors[i_predictor].Weight);
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
        if(m_stepsVect[i_step].Predictors[i_predictor].PreprocessDataId.size()>=(unsigned)(i_dataset+1))
        {
            return m_stepsVect[i_step].Predictors[i_predictor].PreprocessDataId[i_dataset];
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessDataId in the parameters object."));
            VectorString empty;
            return empty;
        }
    }

    bool SetPreprocessDataIdVector(int i_step, int i_predictor, int i_dataset, VectorString val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided preprocess data ID vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
            {
                if (val[i].IsEmpty())
                {
                    asLogError(_("There are NaN values in the provided preprocess data ID vector."));
                    return false;
                }
            }
        }

        if(m_stepsVect[i_step].Predictors[i_predictor].PreprocessDataId.size()>=(unsigned)(i_dataset+1))
        {
            m_stepsVect[i_step].Predictors[i_predictor].PreprocessDataId[i_dataset].clear();
            m_stepsVect[i_step].Predictors[i_predictor].PreprocessDataId[i_dataset] = val;
        }
        else
        {
            m_stepsVect[i_step].Predictors[i_predictor].PreprocessDataId.push_back(val);
        }

        return true;
    }

    VectorFloat GetPreprocessLevelVector(int i_step, int i_predictor, int i_dataset)
    {
        if(m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels.size()>=(unsigned)(i_dataset+1))
        {
            return m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset];
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessLevels in the parameters object."));
            VectorFloat empty;
            return empty;
        }
    }

    bool SetPreprocessLevelVector(int i_step, int i_predictor, int i_dataset, VectorFloat val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided preprocess levels vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
            {
                if (asTools::IsNaN(val[i]))
                {
                    asLogError(_("There are NaN values in the provided preprocess levels vector."));
                    return false;
                }
            }
        }

        if(m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels.size()>=(unsigned)(i_dataset+1))
        {
            m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset].clear();
            m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset] = val;
        }
        else
        {
            m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels.push_back(val);
        }

        return true;
    }

    VectorDouble GetPreprocessTimeHoursVector(int i_step, int i_predictor, int i_dataset)
    {
        wxASSERT(m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>(unsigned)i_dataset);

        if(m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>=(unsigned)(i_dataset+1))
        {
            return m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset];
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
        return m_stepsVect[i_step].Predictors[i_predictor].DataId;
    }

    bool SetPredictorDataIdVector(int i_step, int i_predictor, VectorString val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided data ID vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
            {
                if (val[i].IsEmpty())
                {
                    asLogError(_("There are NaN values in the provided data ID vector."));
                    return false;
                }
            }
        }
        m_stepsVect[i_step].Predictors[i_predictor].DataId = val;
        return true;
    }

    VectorFloat GetPredictorLevelVector(int i_step, int i_predictor)
    {
        return m_stepsVect[i_step].Predictors[i_predictor].Level;
    }

    bool SetPredictorLevelVector(int i_step, int i_predictor, VectorFloat val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided predictor levels vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
            {
                if (asTools::IsNaN(val[i]))
                {
                    asLogError(_("There are NaN values in the provided predictor levels vector."));
                    return false;
                }
            }
        }
        m_stepsVect[i_step].Predictors[i_predictor].Level = val;
        return true;
    }

    VectorString GetPredictorCriteriaVector(int i_step, int i_predictor)
    {
        return m_stepsVect[i_step].Predictors[i_predictor].Criteria;
    }

    bool SetPredictorCriteriaVector(int i_step, int i_predictor, VectorString val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided predictor criteria vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
            {
                if (val[i].IsEmpty())
                {
                    asLogError(_("There are NaN values in the provided predictor criteria vector."));
                    return false;
                }
            }
        }
        m_stepsVect[i_step].Predictors[i_predictor].Criteria = val;
        return true;
    }


protected:
    int m_variableParamsNb;
    int m_timeArrayAnalogsIntervalDaysIteration;
    int m_timeArrayAnalogsIntervalDaysUpperLimit;
    int m_timeArrayAnalogsIntervalDaysLowerLimit;
    bool m_timeArrayAnalogsIntervalDaysLocks;
    VectorParamsStep m_stepsIteration;
    VectorParamsStep m_stepsUpperLimit;
    VectorParamsStep m_stepsLowerLimit;
    VectorParamsStepBool m_stepsLocks;
    ParamsForecastScore m_forecastScoreIteration;
    ParamsForecastScore m_forecastScoreUpperLimit;
    ParamsForecastScore m_forecastScoreLowerLimit;
    VectorParamsStepVect m_stepsVect;

private:


};

#endif // ASPARAMETERSOPTIMIZATION_H
