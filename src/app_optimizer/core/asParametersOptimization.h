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

#ifndef AS_PARAMETERS_OPTIMIZATION_H
#define AS_PARAMETERS_OPTIMIZATION_H

#include "asIncludes.h"
#include "asParameters.h"
#include "asParametersScoring.h"

class asFileParametersOptimization;

class asParametersOptimization : public asParametersScoring {
  public:
    asParametersOptimization();

    virtual ~asParametersOptimization();

    void AddStep();

    void AddPredictorIteration(ParamsStep& step);

    void AddPredictorUpperLimit(ParamsStep& step);

    void AddPredictorLowerLimit(ParamsStep& step);

    void AddPredictorLocks(ParamsStepBool& step);

    void InitRandomValues();

    bool SetSpatialWindowProperties();

    bool SetPreloadingProperties();

    bool LoadFromFile(const wxString& filePath);

    void CheckRange();

    bool IsInRange();

    bool FixTimeLimits();

    void FixHours();

    bool FixWeights();

    void LockAll();

    void Unlock(vi& indices);

    int GetPreprocessDataIdVectorSize(int iStep, int iPtor, int iPre) const {
        return GetPreprocessDataIdVector(iStep, iPtor, iPre).size();
    }

    // May vary
    int GetVariablesNb();

    // Does not change after importation from file.
    int GetVariableParamsNb() {
        return m_variableParamsNb;
    }

    int GetTimeArrayAnalogsIntervalDaysIteration() {
        return m_timeArrayAnalogsIntervalDaysIteration;
    }

    void SetTimeArrayAnalogsIntervalDaysIteration(int val) {
        wxASSERT(!asIsNaN(val));
        m_timeArrayAnalogsIntervalDaysIteration = val;
    }

    int GetAnalogsNumberIteration(int iStep) {
        return m_stepsIteration[iStep].analogsNumber;
    }

    void SetAnalogsNumberIteration(int iStep, int val) {
        wxASSERT(!asIsNaN(val));
        m_stepsIteration[iStep].analogsNumber = val;
    }

    double GetPreprocessHoursIteration(int iStep, int iPtor, int iPre) {
        wxASSERT(m_stepsLowerLimit[iStep].predictors[iPtor].preprocessHours.size() > iPre);
        return m_stepsIteration[iStep].predictors[iPtor].preprocessHours[iPre];
    }

    void SetPreprocessHoursIteration(int iStep, int iPtor, int iPre, double val) {
        wxASSERT(!asIsNaN(val));
        if (m_stepsIteration[iStep].predictors[iPtor].preprocessHours.size() >= iPre + 1) {
            m_stepsIteration[iStep].predictors[iPtor].preprocessHours[iPre] = val;
        } else {
            wxASSERT(m_stepsIteration[iStep].predictors[iPtor].preprocessHours.size() == iPre);
            m_stepsIteration[iStep].predictors[iPtor].preprocessHours.push_back(val);
        }
    }

    double GetPredictorHoursIteration(int iStep, int iPtor) {
        return m_stepsIteration[iStep].predictors[iPtor].hour;
    }

    void SetPredictorHoursIteration(int iStep, int iPtor, double val) {
        wxASSERT(!asIsNaN(val));
        m_stepsIteration[iStep].predictors[iPtor].hour = val;
    }

    double GetPredictorXminIteration(int iStep, int iPtor) {
        return m_stepsIteration[iStep].predictors[iPtor].xMin;
    }

    void SetPredictorXminIteration(int iStep, int iPtor, double val) {
        wxASSERT(!asIsNaN(val));
        m_stepsIteration[iStep].predictors[iPtor].xMin = val;
    }

    int GetPredictorXptsnbIteration(int iStep, int iPtor) {
        return m_stepsIteration[iStep].predictors[iPtor].xPtsNb;
    }

    void SetPredictorXptsnbIteration(int iStep, int iPtor, int val) {
        wxASSERT(!asIsNaN(val));
        m_stepsIteration[iStep].predictors[iPtor].xPtsNb = val;
    }

    double GetPredictorYminIteration(int iStep, int iPtor) {
        return m_stepsIteration[iStep].predictors[iPtor].yMin;
    }

    void SetPredictorYminIteration(int iStep, int iPtor, double val) {
        wxASSERT(!asIsNaN(val));
        m_stepsIteration[iStep].predictors[iPtor].yMin = val;
    }

    int GetPredictorYptsnbIteration(int iStep, int iPtor) {
        return m_stepsIteration[iStep].predictors[iPtor].yPtsNb;
    }

    void SetPredictorYptsnbIteration(int iStep, int iPtor, int val) {
        wxASSERT(!asIsNaN(val));
        m_stepsIteration[iStep].predictors[iPtor].yPtsNb = val;
    }

    float GetPredictorWeightIteration(int iStep, int iPtor) {
        return m_stepsIteration[iStep].predictors[iPtor].weight;
    }

    void SetPredictorWeightIteration(int iStep, int iPtor, float val) {
        wxASSERT(!asIsNaN(val));
        m_stepsIteration[iStep].predictors[iPtor].weight = val;
    }

    int GetTimeArrayAnalogsIntervalDaysUpperLimit() {
        return m_timeArrayAnalogsIntervalDaysUpperLimit;
    }

    void SetTimeArrayAnalogsIntervalDaysUpperLimit(int val) {
        wxASSERT(!asIsNaN(val));
        m_timeArrayAnalogsIntervalDaysUpperLimit = val;
    }

    int GetAnalogsNumberUpperLimit(int iStep) {
        return m_stepsUpperLimit[iStep].analogsNumber;
    }

    void SetAnalogsNumberUpperLimit(int iStep, int val) {
        wxASSERT(!asIsNaN(val));
        m_stepsUpperLimit[iStep].analogsNumber = val;
    }

    double GetPreprocessHoursUpperLimit(int iStep, int iPtor, int iPre) {
        wxASSERT(m_stepsUpperLimit[iStep].predictors[iPtor].preprocessHours.size() > iPre);
        return m_stepsUpperLimit[iStep].predictors[iPtor].preprocessHours[iPre];
    }

    void SetPreprocessHoursUpperLimit(int iStep, int iPtor, int iPre, double val) {
        wxASSERT(!asIsNaN(val));
        if (m_stepsUpperLimit[iStep].predictors[iPtor].preprocessHours.size() >= iPre + 1) {
            m_stepsUpperLimit[iStep].predictors[iPtor].preprocessHours[iPre] = val;
        } else {
            wxASSERT(m_stepsUpperLimit[iStep].predictors[iPtor].preprocessHours.size() == iPre);
            m_stepsUpperLimit[iStep].predictors[iPtor].preprocessHours.push_back(val);
        }
    }

    double GetPredictorHoursUpperLimit(int iStep, int iPtor) {
        return m_stepsUpperLimit[iStep].predictors[iPtor].hour;
    }

    void SetPredictorHoursUpperLimit(int iStep, int iPtor, double val) {
        wxASSERT(!asIsNaN(val));
        m_stepsUpperLimit[iStep].predictors[iPtor].hour = val;
    }

    double GetPredictorXminUpperLimit(int iStep, int iPtor) {
        return m_stepsUpperLimit[iStep].predictors[iPtor].xMin;
    }

    void SetPredictorXminUpperLimit(int iStep, int iPtor, double val) {
        wxASSERT(!asIsNaN(val));
        m_stepsUpperLimit[iStep].predictors[iPtor].xMin = val;
    }

    int GetPredictorXptsnbUpperLimit(int iStep, int iPtor) {
        return m_stepsUpperLimit[iStep].predictors[iPtor].xPtsNb;
    }

    void SetPredictorXptsnbUpperLimit(int iStep, int iPtor, int val) {
        wxASSERT(!asIsNaN(val));
        m_stepsUpperLimit[iStep].predictors[iPtor].xPtsNb = val;
    }

    double GetPredictorYminUpperLimit(int iStep, int iPtor) {
        return m_stepsUpperLimit[iStep].predictors[iPtor].yMin;
    }

    void SetPredictorYminUpperLimit(int iStep, int iPtor, double val) {
        wxASSERT(!asIsNaN(val));
        m_stepsUpperLimit[iStep].predictors[iPtor].yMin = val;
    }

    int GetPredictorYptsnbUpperLimit(int iStep, int iPtor) {
        return m_stepsUpperLimit[iStep].predictors[iPtor].yPtsNb;
    }

    void SetPredictorYptsnbUpperLimit(int iStep, int iPtor, int val) {
        wxASSERT(!asIsNaN(val));
        m_stepsUpperLimit[iStep].predictors[iPtor].yPtsNb = val;
    }

    float GetPredictorWeightUpperLimit(int iStep, int iPtor) {
        return m_stepsUpperLimit[iStep].predictors[iPtor].weight;
    }

    void SetPredictorWeightUpperLimit(int iStep, int iPtor, float val) {
        wxASSERT(!asIsNaN(val));
        m_stepsUpperLimit[iStep].predictors[iPtor].weight = val;
    }

    int GetTimeArrayAnalogsIntervalDaysLowerLimit() {
        return m_timeArrayAnalogsIntervalDaysLowerLimit;
    }

    void SetTimeArrayAnalogsIntervalDaysLowerLimit(int val) {
        wxASSERT(!asIsNaN(val));
        m_timeArrayAnalogsIntervalDaysLowerLimit = val;
    }

    int GetAnalogsNumberLowerLimit(int iStep) {
        return m_stepsLowerLimit[iStep].analogsNumber;
    }

    void SetAnalogsNumberLowerLimit(int iStep, int val) {
        wxASSERT(!asIsNaN(val));
        m_stepsLowerLimit[iStep].analogsNumber = val;
    }

    double GetPreprocessHoursLowerLimit(int iStep, int iPtor, int iPre) {
        wxASSERT(m_stepsLowerLimit[iStep].predictors[iPtor].preprocessHours.size() > iPre);
        return m_stepsLowerLimit[iStep].predictors[iPtor].preprocessHours[iPre];
    }

    void SetPreprocessHoursLowerLimit(int iStep, int iPtor, int iPre, double val) {
        wxASSERT(!asIsNaN(val));
        if (m_stepsLowerLimit[iStep].predictors[iPtor].preprocessHours.size() >= iPre + 1) {
            m_stepsLowerLimit[iStep].predictors[iPtor].preprocessHours[iPre] = val;
        } else {
            wxASSERT(m_stepsLowerLimit[iStep].predictors[iPtor].preprocessHours.size() == iPre);
            m_stepsLowerLimit[iStep].predictors[iPtor].preprocessHours.push_back(val);
        }
    }

    double GetPredictorHoursLowerLimit(int iStep, int iPtor) {
        return m_stepsLowerLimit[iStep].predictors[iPtor].hour;
    }

    void SetPredictorHoursLowerLimit(int iStep, int iPtor, double val) {
        wxASSERT(!asIsNaN(val));
        m_stepsLowerLimit[iStep].predictors[iPtor].hour = val;
    }

    double GetPredictorXminLowerLimit(int iStep, int iPtor) {
        return m_stepsLowerLimit[iStep].predictors[iPtor].xMin;
    }

    void SetPredictorXminLowerLimit(int iStep, int iPtor, double val) {
        wxASSERT(!asIsNaN(val));
        m_stepsLowerLimit[iStep].predictors[iPtor].xMin = val;
    }

    int GetPredictorXptsnbLowerLimit(int iStep, int iPtor) {
        return m_stepsLowerLimit[iStep].predictors[iPtor].xPtsNb;
    }

    void SetPredictorXptsnbLowerLimit(int iStep, int iPtor, int val) {
        wxASSERT(!asIsNaN(val));
        m_stepsLowerLimit[iStep].predictors[iPtor].xPtsNb = val;
    }

    double GetPredictorYminLowerLimit(int iStep, int iPtor) {
        return m_stepsLowerLimit[iStep].predictors[iPtor].yMin;
    }

    void SetPredictorYminLowerLimit(int iStep, int iPtor, double val) {
        wxASSERT(!asIsNaN(val));
        m_stepsLowerLimit[iStep].predictors[iPtor].yMin = val;
    }

    int GetPredictorYptsnbLowerLimit(int iStep, int iPtor) {
        return m_stepsLowerLimit[iStep].predictors[iPtor].yPtsNb;
    }

    void SetPredictorYptsnbLowerLimit(int iStep, int iPtor, double val) {
        wxASSERT(!asIsNaN(val));
        m_stepsLowerLimit[iStep].predictors[iPtor].yPtsNb = (int)val;
    }

    float GetPredictorWeightLowerLimit(int iStep, int iPtor) {
        return m_stepsLowerLimit[iStep].predictors[iPtor].weight;
    }

    void SetPredictorWeightLowerLimit(int iStep, int iPtor, float val) {
        wxASSERT(!asIsNaN(val));
        m_stepsLowerLimit[iStep].predictors[iPtor].weight = val;
    }

    bool IsAnalogsNumberLocked(int iStep) {
        return m_stepsLocks[iStep].analogsNumber;
    }

    void SetAnalogsNumberLock(int iStep, bool val) {
        m_stepsLocks[iStep].analogsNumber = val;
    }

    bool IsTimeArrayAnalogsIntervalDaysLocked() {
        return m_timeArrayAnalogsIntervalDaysLocks;
    }

    void SetTimeArrayAnalogsIntervalDaysLock(bool val) {
        m_timeArrayAnalogsIntervalDaysLocks = val;
    }

    bool IsPreprocessDataIdLocked(int iStep, int iPtor, int iPreess) {
        wxASSERT(m_stepsLocks[iStep].predictors[iPtor].preprocessDataId.size() > iPreess);
        return m_stepsLocks[iStep].predictors[iPtor].preprocessDataId[iPreess];
    }

    void SetPreprocessDataIdLock(int iStep, int iPtor, int iPreess, bool val) {
        if (m_stepsLocks[iStep].predictors[iPtor].preprocessDataId.size() > iPreess) {
            m_stepsLocks[iStep].predictors[iPtor].preprocessDataId[iPreess] = val;
        } else {
            wxASSERT(m_stepsLocks[iStep].predictors[iPtor].preprocessDataId.size() == iPreess);
            m_stepsLocks[iStep].predictors[iPtor].preprocessDataId.push_back(val);
        }
    }

    bool IsPreprocessLevelLocked(int iStep, int iPtor, int iPreess) {
        wxASSERT(m_stepsLocks[iStep].predictors[iPtor].preprocessLevels.size() > iPreess);
        return m_stepsLocks[iStep].predictors[iPtor].preprocessLevels[iPreess];
    }

    void SetPreprocessLevelLock(int iStep, int iPtor, int iPreess, bool val) {
        if (m_stepsLocks[iStep].predictors[iPtor].preprocessLevels.size() > iPreess) {
            m_stepsLocks[iStep].predictors[iPtor].preprocessLevels[iPreess] = val;
        } else {
            wxASSERT(m_stepsLocks[iStep].predictors[iPtor].preprocessLevels.size() == iPreess);
            m_stepsLocks[iStep].predictors[iPtor].preprocessLevels.push_back(val);
        }
    }

    bool IsPreprocessHourLocked(int iStep, int iPtor, int iPreess) {
        wxASSERT(m_stepsLocks[iStep].predictors[iPtor].preprocessHours.size() > iPreess);
        return m_stepsLocks[iStep].predictors[iPtor].preprocessHours[iPreess];
    }

    void SetPreprocessHourLock(int iStep, int iPtor, int iPreess, bool val) {
        if (m_stepsLocks[iStep].predictors[iPtor].preprocessHours.size() > iPreess) {
            m_stepsLocks[iStep].predictors[iPtor].preprocessHours[iPreess] = val;
        } else {
            wxASSERT(m_stepsLocks[iStep].predictors[iPtor].preprocessHours.size() == iPreess);
            m_stepsLocks[iStep].predictors[iPtor].preprocessHours.push_back(val);
        }
    }

    bool IsPredictorHourLocked(int iStep, int iPtor) {
        return m_stepsLocks[iStep].predictors[iPtor].hours;
    }

    void SetPredictorHourLock(int iStep, int iPtor, bool val) {
        m_stepsLocks[iStep].predictors[iPtor].hours = val;
    }

    bool IsPredictorDataIdLocked(int iStep, int iPtor) {
        return m_stepsLocks[iStep].predictors[iPtor].dataId;
    }

    void SetPredictorDataIdLock(int iStep, int iPtor, bool val) {
        m_stepsLocks[iStep].predictors[iPtor].dataId = val;
    }

    bool IsPredictorLevelLocked(int iStep, int iPtor) {
        return m_stepsLocks[iStep].predictors[iPtor].level;
    }

    void SetPredictorLevelLock(int iStep, int iPtor, bool val) {
        m_stepsLocks[iStep].predictors[iPtor].level = val;
    }

    bool IsPredictorXminLocked(int iStep, int iPtor) {
        return m_stepsLocks[iStep].predictors[iPtor].xMin;
    }

    void SetPredictorXminLock(int iStep, int iPtor, bool val) {
        m_stepsLocks[iStep].predictors[iPtor].xMin = val;
    }

    bool IsPredictorXptsnbLocked(int iStep, int iPtor) {
        return m_stepsLocks[iStep].predictors[iPtor].xPtsNb;
    }

    void SetPredictorXptsnbLock(int iStep, int iPtor, bool val) {
        m_stepsLocks[iStep].predictors[iPtor].xPtsNb = val;
    }

    bool IsPredictorYminLocked(int iStep, int iPtor) {
        return m_stepsLocks[iStep].predictors[iPtor].yMin;
    }

    void SetPredictorYminLock(int iStep, int iPtor, bool val) {
        m_stepsLocks[iStep].predictors[iPtor].yMin = val;
    }

    bool IsPredictorYptsnbLocked(int iStep, int iPtor) {
        return m_stepsLocks[iStep].predictors[iPtor].yPtsNb;
    }

    void SetPredictorYptsnbLock(int iStep, int iPtor, bool val) {
        m_stepsLocks[iStep].predictors[iPtor].yPtsNb = val;
    }

    bool IsPredictorWeightLocked(int iStep, int iPtor) {
        return m_stepsLocks[iStep].predictors[iPtor].weight;
    }

    void SetPredictorWeightLock(int iStep, int iPtor, bool val) {
        m_stepsLocks[iStep].predictors[iPtor].weight = val;
    }

    bool IsPredictorCriteriaLocked(int iStep, int iPtor) {
        return m_stepsLocks[iStep].predictors[iPtor].criteria;
    }

    void SetPredictorCriteriaLock(int iStep, int iPtor, bool val) {
        m_stepsLocks[iStep].predictors[iPtor].criteria = val;
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

  private:
    bool ParseDescription(asFileParametersOptimization& fileParams, const wxXmlNode* nodeProcess);

    bool ParseTimeProperties(asFileParametersOptimization& fileParams, const wxXmlNode* nodeProcess);

    bool ParseAnalogDatesParams(asFileParametersOptimization& fileParams, int iStep, const wxXmlNode* nodeProcess);

    bool ParsePredictors(asFileParametersOptimization& fileParams, int iStep, int iPtor,
                         const wxXmlNode* nodeParamBlock);

    bool ParsePreprocessedPredictors(asFileParametersOptimization& fileParams, int iStep, int iPtor,
                                     const wxXmlNode* nodeParam);

    bool ParsePreprocessedPredictorDataset(asFileParametersOptimization& fileParams, int iStep, int iPtor, int iPre,
                                           const wxXmlNode* nodePreprocess);

    bool ParseSpatialWindow(asFileParametersOptimization& fileParams, int iStep, int iPtor, const wxXmlNode* nodeParam);

    bool ParseAnalogValuesParams(asFileParametersOptimization& fileParams, const wxXmlNode* nodeProcess);

    bool ParseScore(asFileParametersOptimization& fileParams, const wxXmlNode* nodeProcess);
};

#endif
