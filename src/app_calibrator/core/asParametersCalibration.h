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
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 * Portions Copyright 2013-2014 Pascal Horton, Terr@num.
 */
 
#ifndef ASPARAMETERSCALIBRATION_H
#define ASPARAMETERSCALIBRATION_H

#include "asIncludes.h"
#include <asParametersScoring.h>

class asFileParametersCalibration;


class asParametersCalibration : public asParametersScoring
{
public:

    asParametersCalibration();
    virtual ~asParametersCalibration();

    void AddStep();

    bool LoadFromFile(const wxString &filePath);
    
    bool SetSpatialWindowProperties();

    bool SetPreloadingProperties();

    bool InputsOK();

    bool FixTimeLimits();

    void InitValues();

    VVectorInt GetPredictandStationsIdsVector()
    {
        return m_PredictandStationsIdsVect;
    }

    bool SetPredictandStationsIdsVector(VVectorInt val);

    VectorInt GetTimeArrayAnalogsIntervalDaysVector()
    {
        return m_TimeArrayAnalogsIntervalDaysVect;
    }

    bool SetTimeArrayAnalogsIntervalDaysVector(VectorInt val);

    VectorInt GetAnalogsNumberVector(int i_step)
    {
        return m_StepsVect[i_step].AnalogsNumber;
    }

    bool SetAnalogsNumberVector(int i_step, VectorInt val);

    VectorString GetPreprocessDataIdVector(int i_step, int i_predictor, int i_dataset);

    bool SetPreprocessDataIdVector(int i_step, int i_predictor, int i_dataset, VectorString val);

    VectorFloat GetPreprocessLevelVector(int i_step, int i_predictor, int i_dataset);

    bool SetPreprocessLevelVector(int i_step, int i_predictor, int i_dataset, VectorFloat val);

    VectorDouble GetPreprocessTimeHoursVector(int i_step, int i_predictor, int i_dataset);

    bool SetPreprocessTimeHoursVector(int i_step, int i_predictor, int i_dataset, VectorDouble val);

    VectorString GetPredictorDataIdVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].DataId;
    }

    bool SetPredictorDataIdVector(int i_step, int i_predictor, VectorString val);

    VectorFloat GetPredictorLevelVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Level;
    }

    bool SetPredictorLevelVector(int i_step, int i_predictor, VectorFloat val);

    VectorDouble GetPredictorXminVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Xmin;
    }

    bool SetPredictorXminVector(int i_step, int i_predictor, VectorDouble val);

    VectorInt GetPredictorXptsnbVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Xptsnb;
    }

    bool SetPredictorXptsnbVector(int i_step, int i_predictor, VectorInt val);

    VectorDouble GetPredictorYminVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Ymin;
    }

    bool SetPredictorYminVector(int i_step, int i_predictor, VectorDouble val);

    VectorInt GetPredictorYptsnbVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Yptsnb;
    }

    bool SetPredictorYptsnbVector(int i_step, int i_predictor, VectorInt val);

    VectorDouble GetPredictorTimeHoursVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].TimeHours;
    }

    bool SetPredictorTimeHoursVector(int i_step, int i_predictor, VectorDouble val);

    VectorString GetPredictorCriteriaVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Criteria;
    }

    bool SetPredictorCriteriaVector(int i_step, int i_predictor, VectorString val);

    VectorFloat GetPredictorWeightVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Weight;
    }

    bool SetPredictorWeightVector(int i_step, int i_predictor, VectorFloat val);

    VectorString GetForecastScoreNameVector()
    {
        return m_ForecastScoreVect.Name;
    }

    bool SetForecastScoreNameVector(VectorString val);

    VectorString GetForecastScoreTimeArrayModeVector()
    {
        return m_ForecastScoreVect.TimeArrayMode;
    }

    bool SetForecastScoreTimeArrayModeVector(VectorString val);

    VectorDouble GetForecastScoreTimeArrayDateVector()
    {
        return m_ForecastScoreVect.TimeArrayDate;
    }

    bool SetForecastScoreTimeArrayDateVector(VectorDouble val);

    VectorInt GetForecastScoreTimeArrayIntervalDaysVector()
    {
        return m_ForecastScoreVect.TimeArrayIntervalDays;
    }

    bool SetForecastScoreTimeArrayIntervalDaysVector(VectorInt val);

    VectorFloat GetForecastScorePostprocessDupliExpVector()
    {
        return m_ForecastScoreVect.PostprocessDupliExp;
    }

    bool SetForecastScorePostprocessDupliExpVector(VectorFloat val);

    int GetTimeArrayAnalogsIntervalDaysLowerLimit();

    int GetAnalogsNumberLowerLimit(int i_step);

    float GetPreprocessLevelLowerLimit(int i_step, int i_predictor, int i_dataset);

    double GetPreprocessTimeHoursLowerLimit(int i_step, int i_predictor, int i_dataset);

    float GetPredictorLevelLowerLimit(int i_step, int i_predictor);

    double GetPredictorXminLowerLimit(int i_step, int i_predictor);

    int GetPredictorXptsnbLowerLimit(int i_step, int i_predictor);

    double GetPredictorYminLowerLimit(int i_step, int i_predictor);

    int GetPredictorYptsnbLowerLimit(int i_step, int i_predictor);

    double GetPredictorTimeHoursLowerLimit(int i_step, int i_predictor);

    float GetPredictorWeightLowerLimit(int i_step, int i_predictor);

    double GetForecastScoreTimeArrayDateLowerLimit();

    int GetForecastScoreTimeArrayIntervalDaysLowerLimit();

    float GetForecastScorePostprocessDupliExpLowerLimit();

    int GetTimeArrayAnalogsIntervalDaysUpperLimit();

    int GetAnalogsNumberUpperLimit(int i_step);

    float GetPreprocessLevelUpperLimit(int i_step, int i_predictor, int i_dataset);

    double GetPreprocessTimeHoursUpperLimit(int i_step, int i_predictor, int i_dataset);

    float GetPredictorLevelUpperLimit(int i_step, int i_predictor);

    double GetPredictorXminUpperLimit(int i_step, int i_predictor);

    int GetPredictorXptsnbUpperLimit(int i_step, int i_predictor);

    double GetPredictorYminUpperLimit(int i_step, int i_predictor);

    int GetPredictorYptsnbUpperLimit(int i_step, int i_predictor);

    double GetPredictorTimeHoursUpperLimit(int i_step, int i_predictor);

    float GetPredictorWeightUpperLimit(int i_step, int i_predictor);

    double GetForecastScoreTimeArrayDateUpperLimit();

    int GetForecastScoreTimeArrayIntervalDaysUpperLimit();

    float GetForecastScorePostprocessDupliExpUpperLimit();

    int GetTimeArrayAnalogsIntervalDaysIteration();

    int GetAnalogsNumberIteration(int i_step);

    double GetPreprocessTimeHoursIteration(int i_step, int i_predictor, int i_dataset);

    double GetPredictorXminIteration(int i_step, int i_predictor);

    int GetPredictorXptsnbIteration(int i_step, int i_predictor);

    double GetPredictorYminIteration(int i_step, int i_predictor);

    int GetPredictorYptsnbIteration(int i_step, int i_predictor);

    double GetPredictorTimeHoursIteration(int i_step, int i_predictor);

    float GetPredictorWeightIteration(int i_step, int i_predictor);

    double GetForecastScoreTimeArrayDateIteration();

    int GetForecastScoreTimeArrayIntervalDaysIteration();

    float GetForecastScorePostprocessDupliExpIteration();

protected:

private:
    VVectorInt m_PredictandStationsIdsVect;
    VectorInt m_TimeArrayAnalogsIntervalDaysVect;
    VectorParamsStepVect m_StepsVect;
    ParamsForecastScoreVect m_ForecastScoreVect;
};

#endif // ASPARAMETERSCALIBRATION_H
