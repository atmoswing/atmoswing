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
 * Portions Copyright 2013-2014 Pascal Horton, Terranum.
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
    
	virtual bool SetSpatialWindowProperties();

    virtual bool SetPreloadingProperties();

    bool InputsOK();

    bool FixTimeLimits();

    void InitValues();

	int GetPreprocessDataIdVectorSize(int i_step, int i_ptor, int i_prep)
	{
		return (int) GetPreprocessDataIdVector(i_step, i_ptor, i_prep).size();
	}

    VVectorInt GetPredictandStationIdsVector()
    {
        return m_predictandStationIdsVect;
    }

    bool SetPredictandStationIdsVector(VVectorInt val);

    VectorInt GetTimeArrayAnalogsIntervalDaysVector()
    {
        return m_timeArrayAnalogsIntervalDaysVect;
    }

    bool SetTimeArrayAnalogsIntervalDaysVector(VectorInt val);

    VectorString GetForecastScoreNameVector()
    {
        return m_forecastScoreVect.Name;
    }

    bool SetForecastScoreNameVector(VectorString val);

    VectorString GetForecastScoreTimeArrayModeVector()
    {
        return m_forecastScoreVect.TimeArrayMode;
    }

    bool SetForecastScoreTimeArrayModeVector(VectorString val);

    VectorDouble GetForecastScoreTimeArrayDateVector()
    {
        return m_forecastScoreVect.TimeArrayDate;
    }

    bool SetForecastScoreTimeArrayDateVector(VectorDouble val);

    VectorInt GetForecastScoreTimeArrayIntervalDaysVector()
    {
        return m_forecastScoreVect.TimeArrayIntervalDays;
    }

    bool SetForecastScoreTimeArrayIntervalDaysVector(VectorInt val);

    VectorFloat GetForecastScorePostprocessDupliExpVector()
    {
        return m_forecastScoreVect.PostprocessDupliExp;
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
    VVectorInt m_predictandStationIdsVect;
    VectorInt m_timeArrayAnalogsIntervalDaysVect;
    ParamsForecastScoreVect m_forecastScoreVect;
};

#endif // ASPARAMETERSCALIBRATION_H
