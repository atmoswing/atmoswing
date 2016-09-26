#ifndef ASPARAMETERSOPTIMIZATION_H
#define ASPARAMETERSOPTIMIZATION_H

#include "asIncludes.h"
#include <asParametersScoring.h>
#include <asParameters.h>

class asFileParametersOptimization;


class asParametersOptimization
        : public asParametersScoring
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

    bool SetSpatialWindowProperties();

    bool SetPreloadingProperties();

    bool LoadFromFile(const wxString &filePath);

    void CheckRange();

    bool IsInRange();

    bool FixTimeLimits();

    void FixTimeHours();

    bool FixWeights();

    void LockAll();

    void Unlock(VectorInt &indices);

    int GetPreprocessDataIdVectorSize(int i_step, int i_ptor, int i_preproc)
    {
        return GetPreprocessDataIdVector(i_step, i_ptor, i_preproc).size();
    }

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
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for the 'days interval' is null"));
            return false;
        }
        m_timeArrayAnalogsIntervalDaysIteration = val;
        return true;
    }

    int GetAnalogsNumberIteration(int i_step)
    {
        return m_stepsIteration[i_step].analogsNumber;
    }

    bool SetAnalogsNumberIteration(int i_step, int val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for the analogs number is null"));
            return false;
        }
        m_stepsIteration[i_step].analogsNumber = val;
        return true;
    }

    double GetPreprocessTimeHoursIteration(int i_step, int i_predictor, int i_dataset)
    {
        wxASSERT(m_stepsLowerLimit[i_step].predictors[i_predictor].preprocessTimeHours.size() > (unsigned) i_dataset);
        return m_stepsIteration[i_step].predictors[i_predictor].preprocessTimeHours[i_dataset];
    }

    bool SetPreprocessTimeHoursIteration(int i_step, int i_predictor, int i_dataset, double val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided iteration value for the preprocess time frame is null"));
            return false;
        }

        if (m_stepsIteration[i_step].predictors[i_predictor].preprocessTimeHours.size() >= (unsigned) (i_dataset + 1)) {
            m_stepsIteration[i_step].predictors[i_predictor].preprocessTimeHours[i_dataset] = val;
        } else {
            wxASSERT(m_stepsIteration[i_step].predictors[i_predictor].preprocessTimeHours.size() ==
                     (unsigned) i_dataset);
            m_stepsIteration[i_step].predictors[i_predictor].preprocessTimeHours.push_back(val);
        }
        return true;
    }

    double GetPredictorTimeHoursIteration(int i_step, int i_predictor)
    {
        return m_stepsIteration[i_step].predictors[i_predictor].timeHours;
    }

    bool SetPredictorTimeHoursIteration(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for the time frame is null"));
            return false;
        }
        m_stepsIteration[i_step].predictors[i_predictor].timeHours = val;
        return true;
    }

    double GetPredictorXminIteration(int i_step, int i_predictor)
    {
        return m_stepsIteration[i_step].predictors[i_predictor].xMin;
    }

    bool SetPredictorXminIteration(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for Xmin is null"));
            return false;
        }
        m_stepsIteration[i_step].predictors[i_predictor].xMin = val;
        return true;
    }

    int GetPredictorXptsnbIteration(int i_step, int i_predictor)
    {
        return m_stepsIteration[i_step].predictors[i_predictor].xPtsNb;
    }

    bool SetPredictorXptsnbIteration(int i_step, int i_predictor, int val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for Xptsnb is null"));
            return false;
        }
        m_stepsIteration[i_step].predictors[i_predictor].xPtsNb = val;
        return true;
    }

    double GetPredictorYminIteration(int i_step, int i_predictor)
    {
        return m_stepsIteration[i_step].predictors[i_predictor].yMin;
    }

    bool SetPredictorYminIteration(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for Ymin is null"));
            return false;
        }
        m_stepsIteration[i_step].predictors[i_predictor].yMin = val;
        return true;
    }

    int GetPredictorYptsnbIteration(int i_step, int i_predictor)
    {
        return m_stepsIteration[i_step].predictors[i_predictor].yPtsNb;
    }

    bool SetPredictorYptsnbIteration(int i_step, int i_predictor, int val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for Yptsnb is null"));
            return false;
        }
        m_stepsIteration[i_step].predictors[i_predictor].yPtsNb = val;
        return true;
    }

    float GetPredictorWeightIteration(int i_step, int i_predictor)
    {
        return m_stepsIteration[i_step].predictors[i_predictor].weight;
    }

    bool SetPredictorWeightIteration(int i_step, int i_predictor, float val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for the predictor weight is null"));
            return false;
        }
        m_stepsIteration[i_step].predictors[i_predictor].weight = val;
        return true;
    }

    int GetTimeArrayAnalogsIntervalDaysUpperLimit()
    {
        return m_timeArrayAnalogsIntervalDaysUpperLimit;
    }

    bool SetTimeArrayAnalogsIntervalDaysUpperLimit(int val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for the 'days interval' is null"));
            return false;
        }
        m_timeArrayAnalogsIntervalDaysUpperLimit = val;
        return true;
    }

    int GetAnalogsNumberUpperLimit(int i_step)
    {
        return m_stepsUpperLimit[i_step].analogsNumber;
    }

    bool SetAnalogsNumberUpperLimit(int i_step, int val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for the analogs number is null"));
            return false;
        }
        m_stepsUpperLimit[i_step].analogsNumber = val;
        return true;
    }

    double GetPreprocessTimeHoursUpperLimit(int i_step, int i_predictor, int i_dataset)
    {
        wxASSERT(m_stepsUpperLimit[i_step].predictors[i_predictor].preprocessTimeHours.size() > (unsigned) i_dataset);
        return m_stepsUpperLimit[i_step].predictors[i_predictor].preprocessTimeHours[i_dataset];
    }

    bool SetPreprocessTimeHoursUpperLimit(int i_step, int i_predictor, int i_dataset, double val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided upper value value for the preprocess time frame is null"));
            return false;
        }

        if (m_stepsUpperLimit[i_step].predictors[i_predictor].preprocessTimeHours.size() >=
            (unsigned) (i_dataset + 1)) {
            m_stepsUpperLimit[i_step].predictors[i_predictor].preprocessTimeHours[i_dataset] = val;
        } else {
            wxASSERT(m_stepsUpperLimit[i_step].predictors[i_predictor].preprocessTimeHours.size() ==
                     (unsigned) i_dataset);
            m_stepsUpperLimit[i_step].predictors[i_predictor].preprocessTimeHours.push_back(val);
        }
        return true;
    }

    double GetPredictorTimeHoursUpperLimit(int i_step, int i_predictor)
    {
        return m_stepsUpperLimit[i_step].predictors[i_predictor].timeHours;
    }

    bool SetPredictorTimeHoursUpperLimit(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for the time frame is null"));
            return false;
        }
        m_stepsUpperLimit[i_step].predictors[i_predictor].timeHours = val;
        return true;
    }

    double GetPredictorXminUpperLimit(int i_step, int i_predictor)
    {
        return m_stepsUpperLimit[i_step].predictors[i_predictor].xMin;
    }

    bool SetPredictorXminUpperLimit(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for Xmin is null"));
            return false;
        }
        m_stepsUpperLimit[i_step].predictors[i_predictor].xMin = val;
        return true;
    }

    int GetPredictorXptsnbUpperLimit(int i_step, int i_predictor)
    {
        return m_stepsUpperLimit[i_step].predictors[i_predictor].xPtsNb;
    }

    bool SetPredictorXptsnbUpperLimit(int i_step, int i_predictor, int val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for Xptsnb is null"));
            return false;
        }
        m_stepsUpperLimit[i_step].predictors[i_predictor].xPtsNb = val;
        return true;
    }

    double GetPredictorYminUpperLimit(int i_step, int i_predictor)
    {
        return m_stepsUpperLimit[i_step].predictors[i_predictor].yMin;
    }

    bool SetPredictorYminUpperLimit(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for Ymin is null"));
            return false;
        }
        m_stepsUpperLimit[i_step].predictors[i_predictor].yMin = val;
        return true;
    }

    int GetPredictorYptsnbUpperLimit(int i_step, int i_predictor)
    {
        return m_stepsUpperLimit[i_step].predictors[i_predictor].yPtsNb;
    }

    bool SetPredictorYptsnbUpperLimit(int i_step, int i_predictor, int val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for Yptsnb is null"));
            return false;
        }
        m_stepsUpperLimit[i_step].predictors[i_predictor].yPtsNb = val;
        return true;
    }

    float GetPredictorWeightUpperLimit(int i_step, int i_predictor)
    {
        return m_stepsUpperLimit[i_step].predictors[i_predictor].weight;
    }

    bool SetPredictorWeightUpperLimit(int i_step, int i_predictor, float val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for the predictor weight is null"));
            return false;
        }
        m_stepsUpperLimit[i_step].predictors[i_predictor].weight = val;
        return true;
    }

    int GetTimeArrayAnalogsIntervalDaysLowerLimit()
    {
        return m_timeArrayAnalogsIntervalDaysLowerLimit;
    }

    bool SetTimeArrayAnalogsIntervalDaysLowerLimit(int val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for the 'days interval' is null"));
            return false;
        }
        m_timeArrayAnalogsIntervalDaysLowerLimit = val;
        return true;
    }

    int GetAnalogsNumberLowerLimit(int i_step)
    {
        return m_stepsLowerLimit[i_step].analogsNumber;
    }

    bool SetAnalogsNumberLowerLimit(int i_step, int val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for the analogs number is null"));
            return false;
        }
        m_stepsLowerLimit[i_step].analogsNumber = val;
        return true;
    }

    double GetPreprocessTimeHoursLowerLimit(int i_step, int i_predictor, int i_dataset)
    {
        wxASSERT(m_stepsLowerLimit[i_step].predictors[i_predictor].preprocessTimeHours.size() > (unsigned) i_dataset);
        return m_stepsLowerLimit[i_step].predictors[i_predictor].preprocessTimeHours[i_dataset];
    }

    bool SetPreprocessTimeHoursLowerLimit(int i_step, int i_predictor, int i_dataset, double val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided lower value value for the preprocess time frame is null"));
            return false;
        }

        if (m_stepsLowerLimit[i_step].predictors[i_predictor].preprocessTimeHours.size() >=
            (unsigned) (i_dataset + 1)) {
            m_stepsLowerLimit[i_step].predictors[i_predictor].preprocessTimeHours[i_dataset] = val;
        } else {
            wxASSERT(m_stepsLowerLimit[i_step].predictors[i_predictor].preprocessTimeHours.size() ==
                     (unsigned) i_dataset);
            m_stepsLowerLimit[i_step].predictors[i_predictor].preprocessTimeHours.push_back(val);
        }
        return true;
    }

    double GetPredictorTimeHoursLowerLimit(int i_step, int i_predictor)
    {
        return m_stepsLowerLimit[i_step].predictors[i_predictor].timeHours;
    }

    bool SetPredictorTimeHoursLowerLimit(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for the time frame is null"));
            return false;
        }
        m_stepsLowerLimit[i_step].predictors[i_predictor].timeHours = val;
        return true;
    }

    double GetPredictorXminLowerLimit(int i_step, int i_predictor)
    {
        return m_stepsLowerLimit[i_step].predictors[i_predictor].xMin;
    }

    bool SetPredictorXminLowerLimit(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for Xmin is null"));
            return false;
        }
        m_stepsLowerLimit[i_step].predictors[i_predictor].xMin = val;
        return true;
    }

    int GetPredictorXptsnbLowerLimit(int i_step, int i_predictor)
    {
        return m_stepsLowerLimit[i_step].predictors[i_predictor].xPtsNb;
    }

    bool SetPredictorXptsnbLowerLimit(int i_step, int i_predictor, int val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for Xptsnb is null"));
            return false;
        }
        m_stepsLowerLimit[i_step].predictors[i_predictor].xPtsNb = val;
        return true;
    }

    double GetPredictorYminLowerLimit(int i_step, int i_predictor)
    {
        return m_stepsLowerLimit[i_step].predictors[i_predictor].yMin;
    }

    bool SetPredictorYminLowerLimit(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for Ymin is null"));
            return false;
        }
        m_stepsLowerLimit[i_step].predictors[i_predictor].yMin = val;
        return true;
    }

    int GetPredictorYptsnbLowerLimit(int i_step, int i_predictor)
    {
        return m_stepsLowerLimit[i_step].predictors[i_predictor].yPtsNb;
    }

    bool SetPredictorYptsnbLowerLimit(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for Yptsnb is null"));
            return false;
        }
        m_stepsLowerLimit[i_step].predictors[i_predictor].yPtsNb = (int) val;
        return true;
    }

    float GetPredictorWeightLowerLimit(int i_step, int i_predictor)
    {
        return m_stepsLowerLimit[i_step].predictors[i_predictor].weight;
    }

    bool SetPredictorWeightLowerLimit(int i_step, int i_predictor, float val)
    {
        if (asTools::IsNaN(val)) {
            asLogError(_("The provided value for the predictor weight is null"));
            return false;
        }
        m_stepsLowerLimit[i_step].predictors[i_predictor].weight = val;
        return true;
    }

    bool IsAnalogsNumberLocked(int i_step)
    {
        return m_stepsLocks[i_step].analogsNumber;
    }

    void SetAnalogsNumberLock(int i_step, bool val)
    {
        m_stepsLocks[i_step].analogsNumber = val;
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
        wxASSERT(m_stepsLocks[i_step].predictors[i_predictor].preprocessDataId.size() > (unsigned) i_preprocess);
        return m_stepsLocks[i_step].predictors[i_predictor].preprocessDataId[i_preprocess];
    }

    void SetPreprocessDataIdLock(int i_step, int i_predictor, int i_preprocess, bool val)
    {
        if (m_stepsLocks[i_step].predictors[i_predictor].preprocessDataId.size() > (unsigned) (i_preprocess)) {
            m_stepsLocks[i_step].predictors[i_predictor].preprocessDataId[i_preprocess] = val;
        } else {
            wxASSERT(m_stepsLocks[i_step].predictors[i_predictor].preprocessDataId.size() == (unsigned) i_preprocess);
            m_stepsLocks[i_step].predictors[i_predictor].preprocessDataId.push_back(val);
        }
    }

    bool IsPreprocessLevelLocked(int i_step, int i_predictor, int i_preprocess)
    {
        wxASSERT(m_stepsLocks[i_step].predictors[i_predictor].preprocessLevels.size() > (unsigned) i_preprocess);
        return m_stepsLocks[i_step].predictors[i_predictor].preprocessLevels[i_preprocess];
    }

    void SetPreprocessLevelLock(int i_step, int i_predictor, int i_preprocess, bool val)
    {
        if (m_stepsLocks[i_step].predictors[i_predictor].preprocessLevels.size() > (unsigned) (i_preprocess)) {
            m_stepsLocks[i_step].predictors[i_predictor].preprocessLevels[i_preprocess] = val;
        } else {
            wxASSERT(m_stepsLocks[i_step].predictors[i_predictor].preprocessLevels.size() == (unsigned) i_preprocess);
            m_stepsLocks[i_step].predictors[i_predictor].preprocessLevels.push_back(val);
        }
    }

    bool IsPreprocessTimeHoursLocked(int i_step, int i_predictor, int i_preprocess)
    {
        wxASSERT(m_stepsLocks[i_step].predictors[i_predictor].preprocessTimeHours.size() > (unsigned) i_preprocess);
        return m_stepsLocks[i_step].predictors[i_predictor].preprocessTimeHours[i_preprocess];
    }

    void SetPreprocessTimeHoursLock(int i_step, int i_predictor, int i_preprocess, bool val)
    {
        if (m_stepsLocks[i_step].predictors[i_predictor].preprocessTimeHours.size() > (unsigned) (i_preprocess)) {
            m_stepsLocks[i_step].predictors[i_predictor].preprocessTimeHours[i_preprocess] = val;
        } else {
            wxASSERT(
                    m_stepsLocks[i_step].predictors[i_predictor].preprocessTimeHours.size() == (unsigned) i_preprocess);
            m_stepsLocks[i_step].predictors[i_predictor].preprocessTimeHours.push_back(val);
        }
    }

    bool IsPredictorTimeHoursLocked(int i_step, int i_predictor)
    {
        return m_stepsLocks[i_step].predictors[i_predictor].timeHours;
    }

    void SetPredictorTimeHoursLock(int i_step, int i_predictor, bool val)
    {
        m_stepsLocks[i_step].predictors[i_predictor].timeHours = val;
    }

    bool IsPredictorDataIdLocked(int i_step, int i_predictor)
    {
        return m_stepsLocks[i_step].predictors[i_predictor].dataId;
    }

    void SetPredictorDataIdLock(int i_step, int i_predictor, bool val)
    {
        m_stepsLocks[i_step].predictors[i_predictor].dataId = val;
    }

    bool IsPredictorLevelLocked(int i_step, int i_predictor)
    {
        return m_stepsLocks[i_step].predictors[i_predictor].level;
    }

    void SetPredictorLevelLock(int i_step, int i_predictor, bool val)
    {
        m_stepsLocks[i_step].predictors[i_predictor].level = val;
    }

    bool IsPredictorXminLocked(int i_step, int i_predictor)
    {
        return m_stepsLocks[i_step].predictors[i_predictor].xMin;
    }

    void SetPredictorXminLock(int i_step, int i_predictor, bool val)
    {
        m_stepsLocks[i_step].predictors[i_predictor].xMin = val;
    }

    bool IsPredictorXptsnbLocked(int i_step, int i_predictor)
    {
        return m_stepsLocks[i_step].predictors[i_predictor].xPtsNb;
    }

    void SetPredictorXptsnbLock(int i_step, int i_predictor, bool val)
    {
        m_stepsLocks[i_step].predictors[i_predictor].xPtsNb = val;
    }

    bool IsPredictorYminLocked(int i_step, int i_predictor)
    {
        return m_stepsLocks[i_step].predictors[i_predictor].yMin;
    }

    void SetPredictorYminLock(int i_step, int i_predictor, bool val)
    {
        m_stepsLocks[i_step].predictors[i_predictor].yMin = val;
    }

    bool IsPredictorYptsnbLocked(int i_step, int i_predictor)
    {
        return m_stepsLocks[i_step].predictors[i_predictor].yPtsNb;
    }

    void SetPredictorYptsnbLock(int i_step, int i_predictor, bool val)
    {
        m_stepsLocks[i_step].predictors[i_predictor].yPtsNb = val;
    }

    bool IsPredictorWeightLocked(int i_step, int i_predictor)
    {
        return m_stepsLocks[i_step].predictors[i_predictor].weight;
    }

    void SetPredictorWeightLock(int i_step, int i_predictor, bool val)
    {
        m_stepsLocks[i_step].predictors[i_predictor].weight = val;
    }

    bool IsPredictorCriteriaLocked(int i_step, int i_predictor)
    {
        return m_stepsLocks[i_step].predictors[i_predictor].criteria;
    }

    void SetPredictorCriteriaLock(int i_step, int i_predictor, bool val)
    {
        m_stepsLocks[i_step].predictors[i_predictor].criteria = val;
    }

    bool IncrementAnalogsNumber(int i_step)
    {
        if (GetAnalogsNumber(i_step) + m_stepsIteration[i_step].analogsNumber <=
            m_stepsUpperLimit[i_step].analogsNumber) {
            SetAnalogsNumber(i_step, GetAnalogsNumber(i_step) + m_stepsIteration[i_step].analogsNumber);
            return true;
        } else {
            return false;
        }
    }

    bool IncrementTimeArrayAnalogsIntervalDays()
    {
        if (m_timeArrayAnalogsIntervalDays + m_timeArrayAnalogsIntervalDaysIteration <=
            m_timeArrayAnalogsIntervalDaysUpperLimit) {
            m_timeArrayAnalogsIntervalDays += m_timeArrayAnalogsIntervalDaysIteration;
            return true;
        } else {
            return false;
        }
    }

    bool IncrementPredictorXmin(int i_step, int i_predictor)
    {
        if (GetPredictorXmin(i_step, i_predictor) + m_stepsIteration[i_step].predictors[i_predictor].xMin <=
            m_stepsUpperLimit[i_step].predictors[i_predictor].xMin) {
            SetPredictorXmin(i_step, i_predictor, GetPredictorXmin(i_step, i_predictor) +
                                                  m_stepsIteration[i_step].predictors[i_predictor].xMin);
            return true;
        } else {
            return false;
        }
    }

    bool IncrementPredictorXptsnb(int i_step, int i_predictor)
    {
        if (GetPredictorXptsnb(i_step, i_predictor) + m_stepsIteration[i_step].predictors[i_predictor].xPtsNb <=
            m_stepsUpperLimit[i_step].predictors[i_predictor].xPtsNb) {
            SetPredictorXptsnb(i_step, i_predictor, GetPredictorXptsnb(i_step, i_predictor) +
                                                    m_stepsIteration[i_step].predictors[i_predictor].xPtsNb);
            return true;
        } else {
            return false;
        }
    }

    bool IncrementPredictorYmin(int i_step, int i_predictor)
    {
        if (GetPredictorYmin(i_step, i_predictor) + m_stepsIteration[i_step].predictors[i_predictor].yMin <=
            m_stepsUpperLimit[i_step].predictors[i_predictor].yMin) {
            SetPredictorYmin(i_step, i_predictor, GetPredictorYmin(i_step, i_predictor) +
                                                  m_stepsIteration[i_step].predictors[i_predictor].yMin);
            return true;
        } else {
            return false;
        }
    }

    bool IncrementPredictorYptsnb(int i_step, int i_predictor)
    {
        if (GetPredictorYptsnb(i_step, i_predictor) + m_stepsIteration[i_step].predictors[i_predictor].yPtsNb <=
            m_stepsUpperLimit[i_step].predictors[i_predictor].yPtsNb) {
            SetPredictorYptsnb(i_step, i_predictor, GetPredictorYptsnb(i_step, i_predictor) +
                                                    m_stepsIteration[i_step].predictors[i_predictor].yPtsNb);
            return true;
        } else {
            return false;
        }
    }

    bool IncrementPredictorTimeHours(int i_step, int i_predictor)
    {
        if (GetPredictorTimeHours(i_step, i_predictor) + m_stepsIteration[i_step].predictors[i_predictor].timeHours <=
            m_stepsUpperLimit[i_step].predictors[i_predictor].timeHours) {
            SetPredictorTimeHours(i_step, i_predictor, GetPredictorTimeHours(i_step, i_predictor) +
                                                       m_stepsIteration[i_step].predictors[i_predictor].timeHours);
            return true;
        } else {
            return false;
        }
    }

    bool IncrementPredictorWeight(int i_step, int i_predictor)
    {
        if (GetPredictorWeight(i_step, i_predictor) + m_stepsIteration[i_step].predictors[i_predictor].weight <=
            m_stepsUpperLimit[i_step].predictors[i_predictor].weight) {
            SetPredictorWeight(i_step, i_predictor, GetPredictorWeight(i_step, i_predictor) +
                                                    m_stepsIteration[i_step].predictors[i_predictor].weight);
            return true;
        } else {
            return false;
        }
    }

    bool DecrementAnalogsNumber(int i_step)
    {
        if (GetAnalogsNumber(i_step) - m_stepsIteration[i_step].analogsNumber >=
            m_stepsLowerLimit[i_step].analogsNumber) {
            SetAnalogsNumber(i_step, GetAnalogsNumber(i_step) - m_stepsIteration[i_step].analogsNumber);
            return true;
        } else {
            return false;
        }
    }

    bool DecrementTimeArrayAnalogsIntervalDays(int i_step)
    {
        if (m_timeArrayAnalogsIntervalDays - m_timeArrayAnalogsIntervalDaysIteration >=
            m_timeArrayAnalogsIntervalDaysLowerLimit) {
            m_timeArrayAnalogsIntervalDays -= m_timeArrayAnalogsIntervalDaysIteration;
            return true;
        } else {
            return false;
        }
    }

    bool DecrementPredictorXmin(int i_step, int i_predictor)
    {
        if (GetPredictorXmin(i_step, i_predictor) - m_stepsIteration[i_step].predictors[i_predictor].xMin >=
            m_stepsLowerLimit[i_step].predictors[i_predictor].xMin) {
            SetPredictorXmin(i_step, i_predictor, GetPredictorXmin(i_step, i_predictor) -
                                                  m_stepsIteration[i_step].predictors[i_predictor].xMin);
            return true;
        } else {
            return false;
        }
    }

    bool DecrementPredictorXptsnb(int i_step, int i_predictor)
    {
        if (GetPredictorXptsnb(i_step, i_predictor) - m_stepsIteration[i_step].predictors[i_predictor].xPtsNb >=
            m_stepsLowerLimit[i_step].predictors[i_predictor].xPtsNb) {
            SetPredictorXptsnb(i_step, i_predictor, GetPredictorXptsnb(i_step, i_predictor) -
                                                    m_stepsIteration[i_step].predictors[i_predictor].xPtsNb);
            return true;
        } else {
            return false;
        }
    }

    bool DecrementPredictorYmin(int i_step, int i_predictor)
    {
        if (GetPredictorYmin(i_step, i_predictor) - m_stepsIteration[i_step].predictors[i_predictor].yMin >=
            m_stepsLowerLimit[i_step].predictors[i_predictor].yMin) {
            SetPredictorYmin(i_step, i_predictor, GetPredictorYmin(i_step, i_predictor) -
                                                  m_stepsIteration[i_step].predictors[i_predictor].yMin);
            return true;
        } else {
            return false;
        }
    }

    bool DecrementPredictorYptsnb(int i_step, int i_predictor)
    {
        if (GetPredictorYptsnb(i_step, i_predictor) - m_stepsIteration[i_step].predictors[i_predictor].yPtsNb >=
            m_stepsLowerLimit[i_step].predictors[i_predictor].yPtsNb) {
            SetPredictorYptsnb(i_step, i_predictor, GetPredictorYptsnb(i_step, i_predictor) -
                                                    m_stepsIteration[i_step].predictors[i_predictor].yPtsNb);
            return true;
        } else {
            return false;
        }
    }

    bool DecrementPredictorTimeHours(int i_step, int i_predictor)
    {
        if (GetPredictorTimeHours(i_step, i_predictor) - m_stepsIteration[i_step].predictors[i_predictor].timeHours >=
            m_stepsLowerLimit[i_step].predictors[i_predictor].timeHours) {
            SetPredictorTimeHours(i_step, i_predictor, GetPredictorTimeHours(i_step, i_predictor) -
                                                       m_stepsIteration[i_step].predictors[i_predictor].timeHours);
            return true;
        } else {
            return false;
        }
    }

    bool DecrementPredictorWeight(int i_step, int i_predictor)
    {
        if (GetPredictorWeight(i_step, i_predictor) - m_stepsIteration[i_step].predictors[i_predictor].weight >=
            m_stepsLowerLimit[i_step].predictors[i_predictor].weight) {
            SetPredictorWeight(i_step, i_predictor, GetPredictorWeight(i_step, i_predictor) -
                                                    m_stepsIteration[i_step].predictors[i_predictor].weight);
            return true;
        } else {
            return false;
        }
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

private:

    bool ParseDescription(asFileParametersOptimization &fileParams, const wxXmlNode *nodeProcess);

    bool ParseTimeProperties(asFileParametersOptimization &fileParams, const wxXmlNode *nodeProcess);

    bool ParseAnalogDatesParams(asFileParametersOptimization &fileParams, int i_step, const wxXmlNode *nodeProcess);

    bool ParsePredictors(asFileParametersOptimization &fileParams, int i_step, int i_ptor,
                         const wxXmlNode *nodeParamBlock);

    bool ParsePreprocessedPredictors(asFileParametersOptimization &fileParams, int i_step, int i_ptor,
                                     const wxXmlNode *nodeParam);

    bool ParsePreprocessedPredictorDataset(asFileParametersOptimization &fileParams, int i_step, int i_ptor, int i_dataset,
                                           const wxXmlNode *nodePreprocess);

    bool ParseSpatialWindow(asFileParametersOptimization &fileParams, int i_step, int i_ptor, const wxXmlNode *nodeParam);

    bool ParseAnalogValuesParams(asFileParametersOptimization &fileParams, const wxXmlNode *nodeProcess);

    bool ParseForecastScore(asFileParametersOptimization &fileParams, const wxXmlNode *nodeProcess);

    bool ParseForecastScoreFinal(asFileParametersOptimization &fileParams, const wxXmlNode *nodeProcess);

};

#endif // ASPARAMETERSOPTIMIZATION_H
