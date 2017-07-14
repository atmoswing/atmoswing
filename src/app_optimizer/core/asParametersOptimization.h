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

    void Unlock(vi &indices);

    int GetPreprocessDataIdVectorSize(int iStep, int iPtor, int iPre) const
    {
        return GetPreprocessDataIdVector(iStep, iPtor, iPre).size();
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
            wxLogError(_("The provided value for the 'days interval' is null"));
            return false;
        }
        m_timeArrayAnalogsIntervalDaysIteration = val;
        return true;
    }

    int GetAnalogsNumberIteration(int iStep)
    {
        return m_stepsIteration[iStep].analogsNumber;
    }

    bool SetAnalogsNumberIteration(int iStep, int val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for the analogs number is null"));
            return false;
        }
        m_stepsIteration[iStep].analogsNumber = val;
        return true;
    }

    double GetPreprocessTimeHoursIteration(int iStep, int iPtor, int iPre)
    {
        wxASSERT(m_stepsLowerLimit[iStep].predictors[iPtor].preprocessTimeHours.size() > (unsigned) iPre);
        return m_stepsIteration[iStep].predictors[iPtor].preprocessTimeHours[iPre];
    }

    bool SetPreprocessTimeHoursIteration(int iStep, int iPtor, int iPre, double val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided iteration value for the preprocess time frame is null"));
            return false;
        }

        if (m_stepsIteration[iStep].predictors[iPtor].preprocessTimeHours.size() >= (unsigned) (iPre + 1)) {
            m_stepsIteration[iStep].predictors[iPtor].preprocessTimeHours[iPre] = val;
        } else {
            wxASSERT(m_stepsIteration[iStep].predictors[iPtor].preprocessTimeHours.size() == (unsigned) iPre);
            m_stepsIteration[iStep].predictors[iPtor].preprocessTimeHours.push_back(val);
        }
        return true;
    }

    double GetPredictorTimeHoursIteration(int iStep, int iPtor)
    {
        return m_stepsIteration[iStep].predictors[iPtor].timeHours;
    }

    bool SetPredictorTimeHoursIteration(int iStep, int iPtor, double val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for the time frame is null"));
            return false;
        }
        m_stepsIteration[iStep].predictors[iPtor].timeHours = val;
        return true;
    }

    double GetPredictorXminIteration(int iStep, int iPtor)
    {
        return m_stepsIteration[iStep].predictors[iPtor].xMin;
    }

    bool SetPredictorXminIteration(int iStep, int iPtor, double val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for Xmin is null"));
            return false;
        }
        m_stepsIteration[iStep].predictors[iPtor].xMin = val;
        return true;
    }

    int GetPredictorXptsnbIteration(int iStep, int iPtor)
    {
        return m_stepsIteration[iStep].predictors[iPtor].xPtsNb;
    }

    bool SetPredictorXptsnbIteration(int iStep, int iPtor, int val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for Xptsnb is null"));
            return false;
        }
        m_stepsIteration[iStep].predictors[iPtor].xPtsNb = val;
        return true;
    }

    double GetPredictorYminIteration(int iStep, int iPtor)
    {
        return m_stepsIteration[iStep].predictors[iPtor].yMin;
    }

    bool SetPredictorYminIteration(int iStep, int iPtor, double val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for Ymin is null"));
            return false;
        }
        m_stepsIteration[iStep].predictors[iPtor].yMin = val;
        return true;
    }

    int GetPredictorYptsnbIteration(int iStep, int iPtor)
    {
        return m_stepsIteration[iStep].predictors[iPtor].yPtsNb;
    }

    bool SetPredictorYptsnbIteration(int iStep, int iPtor, int val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for Yptsnb is null"));
            return false;
        }
        m_stepsIteration[iStep].predictors[iPtor].yPtsNb = val;
        return true;
    }

    float GetPredictorWeightIteration(int iStep, int iPtor)
    {
        return m_stepsIteration[iStep].predictors[iPtor].weight;
    }

    bool SetPredictorWeightIteration(int iStep, int iPtor, float val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for the predictor weight is null"));
            return false;
        }
        m_stepsIteration[iStep].predictors[iPtor].weight = val;
        return true;
    }

    int GetTimeArrayAnalogsIntervalDaysUpperLimit()
    {
        return m_timeArrayAnalogsIntervalDaysUpperLimit;
    }

    bool SetTimeArrayAnalogsIntervalDaysUpperLimit(int val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for the 'days interval' is null"));
            return false;
        }
        m_timeArrayAnalogsIntervalDaysUpperLimit = val;
        return true;
    }

    int GetAnalogsNumberUpperLimit(int iStep)
    {
        return m_stepsUpperLimit[iStep].analogsNumber;
    }

    bool SetAnalogsNumberUpperLimit(int iStep, int val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for the analogs number is null"));
            return false;
        }
        m_stepsUpperLimit[iStep].analogsNumber = val;
        return true;
    }

    double GetPreprocessTimeHoursUpperLimit(int iStep, int iPtor, int iPre)
    {
        wxASSERT(m_stepsUpperLimit[iStep].predictors[iPtor].preprocessTimeHours.size() > (unsigned) iPre);
        return m_stepsUpperLimit[iStep].predictors[iPtor].preprocessTimeHours[iPre];
    }

    bool SetPreprocessTimeHoursUpperLimit(int iStep, int iPtor, int iPre, double val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided upper value value for the preprocess time frame is null"));
            return false;
        }

        if (m_stepsUpperLimit[iStep].predictors[iPtor].preprocessTimeHours.size() >= (unsigned) (iPre + 1)) {
            m_stepsUpperLimit[iStep].predictors[iPtor].preprocessTimeHours[iPre] = val;
        } else {
            wxASSERT(m_stepsUpperLimit[iStep].predictors[iPtor].preprocessTimeHours.size() == (unsigned) iPre);
            m_stepsUpperLimit[iStep].predictors[iPtor].preprocessTimeHours.push_back(val);
        }
        return true;
    }

    double GetPredictorTimeHoursUpperLimit(int iStep, int iPtor)
    {
        return m_stepsUpperLimit[iStep].predictors[iPtor].timeHours;
    }

    bool SetPredictorTimeHoursUpperLimit(int iStep, int iPtor, double val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for the time frame is null"));
            return false;
        }
        m_stepsUpperLimit[iStep].predictors[iPtor].timeHours = val;
        return true;
    }

    double GetPredictorXminUpperLimit(int iStep, int iPtor)
    {
        return m_stepsUpperLimit[iStep].predictors[iPtor].xMin;
    }

    bool SetPredictorXminUpperLimit(int iStep, int iPtor, double val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for Xmin is null"));
            return false;
        }
        m_stepsUpperLimit[iStep].predictors[iPtor].xMin = val;
        return true;
    }

    int GetPredictorXptsnbUpperLimit(int iStep, int iPtor)
    {
        return m_stepsUpperLimit[iStep].predictors[iPtor].xPtsNb;
    }

    bool SetPredictorXptsnbUpperLimit(int iStep, int iPtor, int val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for Xptsnb is null"));
            return false;
        }
        m_stepsUpperLimit[iStep].predictors[iPtor].xPtsNb = val;
        return true;
    }

    double GetPredictorYminUpperLimit(int iStep, int iPtor)
    {
        return m_stepsUpperLimit[iStep].predictors[iPtor].yMin;
    }

    bool SetPredictorYminUpperLimit(int iStep, int iPtor, double val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for Ymin is null"));
            return false;
        }
        m_stepsUpperLimit[iStep].predictors[iPtor].yMin = val;
        return true;
    }

    int GetPredictorYptsnbUpperLimit(int iStep, int iPtor)
    {
        return m_stepsUpperLimit[iStep].predictors[iPtor].yPtsNb;
    }

    bool SetPredictorYptsnbUpperLimit(int iStep, int iPtor, int val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for Yptsnb is null"));
            return false;
        }
        m_stepsUpperLimit[iStep].predictors[iPtor].yPtsNb = val;
        return true;
    }

    float GetPredictorWeightUpperLimit(int iStep, int iPtor)
    {
        return m_stepsUpperLimit[iStep].predictors[iPtor].weight;
    }

    bool SetPredictorWeightUpperLimit(int iStep, int iPtor, float val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for the predictor weight is null"));
            return false;
        }
        m_stepsUpperLimit[iStep].predictors[iPtor].weight = val;
        return true;
    }

    int GetTimeArrayAnalogsIntervalDaysLowerLimit()
    {
        return m_timeArrayAnalogsIntervalDaysLowerLimit;
    }

    bool SetTimeArrayAnalogsIntervalDaysLowerLimit(int val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for the 'days interval' is null"));
            return false;
        }
        m_timeArrayAnalogsIntervalDaysLowerLimit = val;
        return true;
    }

    int GetAnalogsNumberLowerLimit(int iStep)
    {
        return m_stepsLowerLimit[iStep].analogsNumber;
    }

    bool SetAnalogsNumberLowerLimit(int iStep, int val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for the analogs number is null"));
            return false;
        }
        m_stepsLowerLimit[iStep].analogsNumber = val;
        return true;
    }

    double GetPreprocessTimeHoursLowerLimit(int iStep, int iPtor, int iPre)
    {
        wxASSERT(m_stepsLowerLimit[iStep].predictors[iPtor].preprocessTimeHours.size() > (unsigned) iPre);
        return m_stepsLowerLimit[iStep].predictors[iPtor].preprocessTimeHours[iPre];
    }

    bool SetPreprocessTimeHoursLowerLimit(int iStep, int iPtor, int iPre, double val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided lower value value for the preprocess time frame is null"));
            return false;
        }

        if (m_stepsLowerLimit[iStep].predictors[iPtor].preprocessTimeHours.size() >= (unsigned) (iPre + 1)) {
            m_stepsLowerLimit[iStep].predictors[iPtor].preprocessTimeHours[iPre] = val;
        } else {
            wxASSERT(m_stepsLowerLimit[iStep].predictors[iPtor].preprocessTimeHours.size() == (unsigned) iPre);
            m_stepsLowerLimit[iStep].predictors[iPtor].preprocessTimeHours.push_back(val);
        }
        return true;
    }

    double GetPredictorTimeHoursLowerLimit(int iStep, int iPtor)
    {
        return m_stepsLowerLimit[iStep].predictors[iPtor].timeHours;
    }

    bool SetPredictorTimeHoursLowerLimit(int iStep, int iPtor, double val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for the time frame is null"));
            return false;
        }
        m_stepsLowerLimit[iStep].predictors[iPtor].timeHours = val;
        return true;
    }

    double GetPredictorXminLowerLimit(int iStep, int iPtor)
    {
        return m_stepsLowerLimit[iStep].predictors[iPtor].xMin;
    }

    bool SetPredictorXminLowerLimit(int iStep, int iPtor, double val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for Xmin is null"));
            return false;
        }
        m_stepsLowerLimit[iStep].predictors[iPtor].xMin = val;
        return true;
    }

    int GetPredictorXptsnbLowerLimit(int iStep, int iPtor)
    {
        return m_stepsLowerLimit[iStep].predictors[iPtor].xPtsNb;
    }

    bool SetPredictorXptsnbLowerLimit(int iStep, int iPtor, int val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for Xptsnb is null"));
            return false;
        }
        m_stepsLowerLimit[iStep].predictors[iPtor].xPtsNb = val;
        return true;
    }

    double GetPredictorYminLowerLimit(int iStep, int iPtor)
    {
        return m_stepsLowerLimit[iStep].predictors[iPtor].yMin;
    }

    bool SetPredictorYminLowerLimit(int iStep, int iPtor, double val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for Ymin is null"));
            return false;
        }
        m_stepsLowerLimit[iStep].predictors[iPtor].yMin = val;
        return true;
    }

    int GetPredictorYptsnbLowerLimit(int iStep, int iPtor)
    {
        return m_stepsLowerLimit[iStep].predictors[iPtor].yPtsNb;
    }

    bool SetPredictorYptsnbLowerLimit(int iStep, int iPtor, double val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for Yptsnb is null"));
            return false;
        }
        m_stepsLowerLimit[iStep].predictors[iPtor].yPtsNb = (int) val;
        return true;
    }

    float GetPredictorWeightLowerLimit(int iStep, int iPtor)
    {
        return m_stepsLowerLimit[iStep].predictors[iPtor].weight;
    }

    bool SetPredictorWeightLowerLimit(int iStep, int iPtor, float val)
    {
        if (asTools::IsNaN(val)) {
            wxLogError(_("The provided value for the predictor weight is null"));
            return false;
        }
        m_stepsLowerLimit[iStep].predictors[iPtor].weight = val;
        return true;
    }

    bool IsAnalogsNumberLocked(int iStep)
    {
        return m_stepsLocks[iStep].analogsNumber;
    }

    void SetAnalogsNumberLock(int iStep, bool val)
    {
        m_stepsLocks[iStep].analogsNumber = val;
    }

    bool IsTimeArrayAnalogsIntervalDaysLocked()
    {
        return m_timeArrayAnalogsIntervalDaysLocks;
    }

    void SetTimeArrayAnalogsIntervalDaysLock(bool val)
    {
        m_timeArrayAnalogsIntervalDaysLocks = val;
    }

    bool IsPreprocessDataIdLocked(int iStep, int iPtor, int iPreess)
    {
        wxASSERT(m_stepsLocks[iStep].predictors[iPtor].preprocessDataId.size() > (unsigned) iPreess);
        return m_stepsLocks[iStep].predictors[iPtor].preprocessDataId[iPreess];
    }

    void SetPreprocessDataIdLock(int iStep, int iPtor, int iPreess, bool val)
    {
        if (m_stepsLocks[iStep].predictors[iPtor].preprocessDataId.size() > (unsigned) (iPreess)) {
            m_stepsLocks[iStep].predictors[iPtor].preprocessDataId[iPreess] = val;
        } else {
            wxASSERT(m_stepsLocks[iStep].predictors[iPtor].preprocessDataId.size() == (unsigned) iPreess);
            m_stepsLocks[iStep].predictors[iPtor].preprocessDataId.push_back(val);
        }
    }

    bool IsPreprocessLevelLocked(int iStep, int iPtor, int iPreess)
    {
        wxASSERT(m_stepsLocks[iStep].predictors[iPtor].preprocessLevels.size() > (unsigned) iPreess);
        return m_stepsLocks[iStep].predictors[iPtor].preprocessLevels[iPreess];
    }

    void SetPreprocessLevelLock(int iStep, int iPtor, int iPreess, bool val)
    {
        if (m_stepsLocks[iStep].predictors[iPtor].preprocessLevels.size() > (unsigned) (iPreess)) {
            m_stepsLocks[iStep].predictors[iPtor].preprocessLevels[iPreess] = val;
        } else {
            wxASSERT(m_stepsLocks[iStep].predictors[iPtor].preprocessLevels.size() == (unsigned) iPreess);
            m_stepsLocks[iStep].predictors[iPtor].preprocessLevels.push_back(val);
        }
    }

    bool IsPreprocessTimeHoursLocked(int iStep, int iPtor, int iPreess)
    {
        wxASSERT(m_stepsLocks[iStep].predictors[iPtor].preprocessTimeHours.size() > (unsigned) iPreess);
        return m_stepsLocks[iStep].predictors[iPtor].preprocessTimeHours[iPreess];
    }

    void SetPreprocessTimeHoursLock(int iStep, int iPtor, int iPreess, bool val)
    {
        if (m_stepsLocks[iStep].predictors[iPtor].preprocessTimeHours.size() > (unsigned) (iPreess)) {
            m_stepsLocks[iStep].predictors[iPtor].preprocessTimeHours[iPreess] = val;
        } else {
            wxASSERT(m_stepsLocks[iStep].predictors[iPtor].preprocessTimeHours.size() == (unsigned) iPreess);
            m_stepsLocks[iStep].predictors[iPtor].preprocessTimeHours.push_back(val);
        }
    }

    bool IsPredictorTimeHoursLocked(int iStep, int iPtor)
    {
        return m_stepsLocks[iStep].predictors[iPtor].timeHours;
    }

    void SetPredictorTimeHoursLock(int iStep, int iPtor, bool val)
    {
        m_stepsLocks[iStep].predictors[iPtor].timeHours = val;
    }

    bool IsPredictorDataIdLocked(int iStep, int iPtor)
    {
        return m_stepsLocks[iStep].predictors[iPtor].dataId;
    }

    void SetPredictorDataIdLock(int iStep, int iPtor, bool val)
    {
        m_stepsLocks[iStep].predictors[iPtor].dataId = val;
    }

    bool IsPredictorLevelLocked(int iStep, int iPtor)
    {
        return m_stepsLocks[iStep].predictors[iPtor].level;
    }

    void SetPredictorLevelLock(int iStep, int iPtor, bool val)
    {
        m_stepsLocks[iStep].predictors[iPtor].level = val;
    }

    bool IsPredictorXminLocked(int iStep, int iPtor)
    {
        return m_stepsLocks[iStep].predictors[iPtor].xMin;
    }

    void SetPredictorXminLock(int iStep, int iPtor, bool val)
    {
        m_stepsLocks[iStep].predictors[iPtor].xMin = val;
    }

    bool IsPredictorXptsnbLocked(int iStep, int iPtor)
    {
        return m_stepsLocks[iStep].predictors[iPtor].xPtsNb;
    }

    void SetPredictorXptsnbLock(int iStep, int iPtor, bool val)
    {
        m_stepsLocks[iStep].predictors[iPtor].xPtsNb = val;
    }

    bool IsPredictorYminLocked(int iStep, int iPtor)
    {
        return m_stepsLocks[iStep].predictors[iPtor].yMin;
    }

    void SetPredictorYminLock(int iStep, int iPtor, bool val)
    {
        m_stepsLocks[iStep].predictors[iPtor].yMin = val;
    }

    bool IsPredictorYptsnbLocked(int iStep, int iPtor)
    {
        return m_stepsLocks[iStep].predictors[iPtor].yPtsNb;
    }

    void SetPredictorYptsnbLock(int iStep, int iPtor, bool val)
    {
        m_stepsLocks[iStep].predictors[iPtor].yPtsNb = val;
    }

    bool IsPredictorWeightLocked(int iStep, int iPtor)
    {
        return m_stepsLocks[iStep].predictors[iPtor].weight;
    }

    void SetPredictorWeightLock(int iStep, int iPtor, bool val)
    {
        m_stepsLocks[iStep].predictors[iPtor].weight = val;
    }

    bool IsPredictorCriteriaLocked(int iStep, int iPtor)
    {
        return m_stepsLocks[iStep].predictors[iPtor].criteria;
    }

    void SetPredictorCriteriaLock(int iStep, int iPtor, bool val)
    {
        m_stepsLocks[iStep].predictors[iPtor].criteria = val;
    }

    bool IncrementAnalogsNumber(int iStep)
    {
        if (GetAnalogsNumber(iStep) + m_stepsIteration[iStep].analogsNumber <= m_stepsUpperLimit[iStep].analogsNumber) {
            SetAnalogsNumber(iStep, GetAnalogsNumber(iStep) + m_stepsIteration[iStep].analogsNumber);
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

    bool IncrementPredictorXmin(int iStep, int iPtor)
    {
        if (GetPredictorXmin(iStep, iPtor) + m_stepsIteration[iStep].predictors[iPtor].xMin <=
            m_stepsUpperLimit[iStep].predictors[iPtor].xMin) {
            SetPredictorXmin(iStep, iPtor,
                             GetPredictorXmin(iStep, iPtor) + m_stepsIteration[iStep].predictors[iPtor].xMin);
            return true;
        } else {
            return false;
        }
    }

    bool IncrementPredictorXptsnb(int iStep, int iPtor)
    {
        if (GetPredictorXptsnb(iStep, iPtor) + m_stepsIteration[iStep].predictors[iPtor].xPtsNb <=
            m_stepsUpperLimit[iStep].predictors[iPtor].xPtsNb) {
            SetPredictorXptsnb(iStep, iPtor,
                               GetPredictorXptsnb(iStep, iPtor) + m_stepsIteration[iStep].predictors[iPtor].xPtsNb);
            return true;
        } else {
            return false;
        }
    }

    bool IncrementPredictorYmin(int iStep, int iPtor)
    {
        if (GetPredictorYmin(iStep, iPtor) + m_stepsIteration[iStep].predictors[iPtor].yMin <=
            m_stepsUpperLimit[iStep].predictors[iPtor].yMin) {
            SetPredictorYmin(iStep, iPtor,
                             GetPredictorYmin(iStep, iPtor) + m_stepsIteration[iStep].predictors[iPtor].yMin);
            return true;
        } else {
            return false;
        }
    }

    bool IncrementPredictorYptsnb(int iStep, int iPtor)
    {
        if (GetPredictorYptsnb(iStep, iPtor) + m_stepsIteration[iStep].predictors[iPtor].yPtsNb <=
            m_stepsUpperLimit[iStep].predictors[iPtor].yPtsNb) {
            SetPredictorYptsnb(iStep, iPtor,
                               GetPredictorYptsnb(iStep, iPtor) + m_stepsIteration[iStep].predictors[iPtor].yPtsNb);
            return true;
        } else {
            return false;
        }
    }

    bool IncrementPredictorTimeHours(int iStep, int iPtor)
    {
        if (GetPredictorTimeHours(iStep, iPtor) + m_stepsIteration[iStep].predictors[iPtor].timeHours <=
            m_stepsUpperLimit[iStep].predictors[iPtor].timeHours) {
            SetPredictorTimeHours(iStep, iPtor, GetPredictorTimeHours(iStep, iPtor) +
                                                m_stepsIteration[iStep].predictors[iPtor].timeHours);
            return true;
        } else {
            return false;
        }
    }

    bool IncrementPredictorWeight(int iStep, int iPtor)
    {
        if (GetPredictorWeight(iStep, iPtor) + m_stepsIteration[iStep].predictors[iPtor].weight <=
            m_stepsUpperLimit[iStep].predictors[iPtor].weight) {
            SetPredictorWeight(iStep, iPtor,
                               GetPredictorWeight(iStep, iPtor) + m_stepsIteration[iStep].predictors[iPtor].weight);
            return true;
        } else {
            return false;
        }
    }

    bool DecrementAnalogsNumber(int iStep)
    {
        if (GetAnalogsNumber(iStep) - m_stepsIteration[iStep].analogsNumber >= m_stepsLowerLimit[iStep].analogsNumber) {
            SetAnalogsNumber(iStep, GetAnalogsNumber(iStep) - m_stepsIteration[iStep].analogsNumber);
            return true;
        } else {
            return false;
        }
    }

    bool DecrementTimeArrayAnalogsIntervalDays(int iStep)
    {
        if (m_timeArrayAnalogsIntervalDays - m_timeArrayAnalogsIntervalDaysIteration >=
            m_timeArrayAnalogsIntervalDaysLowerLimit) {
            m_timeArrayAnalogsIntervalDays -= m_timeArrayAnalogsIntervalDaysIteration;
            return true;
        } else {
            return false;
        }
    }

    bool DecrementPredictorXmin(int iStep, int iPtor)
    {
        if (GetPredictorXmin(iStep, iPtor) - m_stepsIteration[iStep].predictors[iPtor].xMin >=
            m_stepsLowerLimit[iStep].predictors[iPtor].xMin) {
            SetPredictorXmin(iStep, iPtor,
                             GetPredictorXmin(iStep, iPtor) - m_stepsIteration[iStep].predictors[iPtor].xMin);
            return true;
        } else {
            return false;
        }
    }

    bool DecrementPredictorXptsnb(int iStep, int iPtor)
    {
        if (GetPredictorXptsnb(iStep, iPtor) - m_stepsIteration[iStep].predictors[iPtor].xPtsNb >=
            m_stepsLowerLimit[iStep].predictors[iPtor].xPtsNb) {
            SetPredictorXptsnb(iStep, iPtor,
                               GetPredictorXptsnb(iStep, iPtor) - m_stepsIteration[iStep].predictors[iPtor].xPtsNb);
            return true;
        } else {
            return false;
        }
    }

    bool DecrementPredictorYmin(int iStep, int iPtor)
    {
        if (GetPredictorYmin(iStep, iPtor) - m_stepsIteration[iStep].predictors[iPtor].yMin >=
            m_stepsLowerLimit[iStep].predictors[iPtor].yMin) {
            SetPredictorYmin(iStep, iPtor,
                             GetPredictorYmin(iStep, iPtor) - m_stepsIteration[iStep].predictors[iPtor].yMin);
            return true;
        } else {
            return false;
        }
    }

    bool DecrementPredictorYptsnb(int iStep, int iPtor)
    {
        if (GetPredictorYptsnb(iStep, iPtor) - m_stepsIteration[iStep].predictors[iPtor].yPtsNb >=
            m_stepsLowerLimit[iStep].predictors[iPtor].yPtsNb) {
            SetPredictorYptsnb(iStep, iPtor,
                               GetPredictorYptsnb(iStep, iPtor) - m_stepsIteration[iStep].predictors[iPtor].yPtsNb);
            return true;
        } else {
            return false;
        }
    }

    bool DecrementPredictorTimeHours(int iStep, int iPtor)
    {
        if (GetPredictorTimeHours(iStep, iPtor) - m_stepsIteration[iStep].predictors[iPtor].timeHours >=
            m_stepsLowerLimit[iStep].predictors[iPtor].timeHours) {
            SetPredictorTimeHours(iStep, iPtor, GetPredictorTimeHours(iStep, iPtor) -
                                                m_stepsIteration[iStep].predictors[iPtor].timeHours);
            return true;
        } else {
            return false;
        }
    }

    bool DecrementPredictorWeight(int iStep, int iPtor)
    {
        if (GetPredictorWeight(iStep, iPtor) - m_stepsIteration[iStep].predictors[iPtor].weight >=
            m_stepsLowerLimit[iStep].predictors[iPtor].weight) {
            SetPredictorWeight(iStep, iPtor,
                               GetPredictorWeight(iStep, iPtor) - m_stepsIteration[iStep].predictors[iPtor].weight);
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

    bool ParseAnalogDatesParams(asFileParametersOptimization &fileParams, int iStep, const wxXmlNode *nodeProcess);

    bool ParsePredictors(asFileParametersOptimization &fileParams, int iStep, int iPtor,
                         const wxXmlNode *nodeParamBlock);

    bool ParsePreprocessedPredictors(asFileParametersOptimization &fileParams, int iStep, int iPtor,
                                     const wxXmlNode *nodeParam);

    bool ParsePreprocessedPredictorDataset(asFileParametersOptimization &fileParams, int iStep, int iPtor, int iPre,
                                           const wxXmlNode *nodePreprocess);

    bool ParseSpatialWindow(asFileParametersOptimization &fileParams, int iStep, int iPtor, const wxXmlNode *nodeParam);

    bool ParseAnalogValuesParams(asFileParametersOptimization &fileParams, const wxXmlNode *nodeProcess);

    bool ParseForecastScore(asFileParametersOptimization &fileParams, const wxXmlNode *nodeProcess);

    bool ParseForecastScoreFinal(asFileParametersOptimization &fileParams, const wxXmlNode *nodeProcess);

};

#endif // ASPARAMETERSOPTIMIZATION_H
