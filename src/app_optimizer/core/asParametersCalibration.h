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


class asParametersCalibration
        : public asParametersScoring
{
public:

    asParametersCalibration();

    virtual ~asParametersCalibration();

    void AddStep();

    bool LoadFromFile(const wxString &filePath);

    virtual bool SetSpatialWindowProperties();

    virtual bool SetPreloadingProperties();

    bool InputsOK() const;

    bool FixTimeLimits();

    void InitValues();

    int GetPreprocessDataIdVectorSize(int i_step, int i_ptor, int i_prep) const
    {
        return (int) GetPreprocessDataIdVector(i_step, i_ptor, i_prep).size();
    }

    VVectorInt GetPredictandStationIdsVector() const
    {
        return m_predictandStationIdsVect;
    }

    bool SetPredictandStationIdsVector(VVectorInt val);

    VectorInt GetTimeArrayAnalogsIntervalDaysVector() const
    {
        return m_timeArrayAnalogsIntervalDaysVect;
    }

    bool SetTimeArrayAnalogsIntervalDaysVector(VectorInt val);

    VectorString GetForecastScoreNameVector() const
    {
        return m_forecastScoreVect.name;
    }

    bool SetForecastScoreNameVector(VectorString val);

    VectorString GetForecastScoreTimeArrayModeVector() const
    {
        return m_forecastScoreVect.timeArrayMode;
    }

    bool SetForecastScoreTimeArrayModeVector(VectorString val);

    VectorDouble GetForecastScoreTimeArrayDateVector() const
    {
        return m_forecastScoreVect.timeArrayDate;
    }

    VectorInt GetForecastScoreTimeArrayIntervalDaysVector() const
    {
        return m_forecastScoreVect.timeArrayIntervalDays;
    }

    VectorFloat GetForecastScorePostprocessDupliExpVector() const
    {
        return m_forecastScoreVect.postprocessDupliExp;
    }

    double GetPreprocessTimeHoursLowerLimit(int i_step, int i_predictor, int i_dataset) const;

    double GetPredictorXminLowerLimit(int i_step, int i_predictor) const;

    int GetPredictorXptsnbLowerLimit(int i_step, int i_predictor) const;

    double GetPredictorYminLowerLimit(int i_step, int i_predictor) const;

    int GetPredictorYptsnbLowerLimit(int i_step, int i_predictor) const;

    double GetPredictorTimeHoursLowerLimit(int i_step, int i_predictor) const;

    double GetPreprocessTimeHoursUpperLimit(int i_step, int i_predictor, int i_dataset) const;

    double GetPredictorXminUpperLimit(int i_step, int i_predictor) const;

    int GetPredictorXptsnbUpperLimit(int i_step, int i_predictor) const;

    double GetPredictorYminUpperLimit(int i_step, int i_predictor) const;

    int GetPredictorYptsnbUpperLimit(int i_step, int i_predictor) const;

    double GetPredictorTimeHoursUpperLimit(int i_step, int i_predictor) const;

    double GetPredictorXminIteration(int i_step, int i_predictor) const;

    int GetPredictorXptsnbIteration(int i_step, int i_predictor) const;

    double GetPredictorYminIteration(int i_step, int i_predictor) const;

    int GetPredictorYptsnbIteration(int i_step, int i_predictor) const;

protected:

private:
    VVectorInt m_predictandStationIdsVect;
    VectorInt m_timeArrayAnalogsIntervalDaysVect;
    ParamsForecastScoreVect m_forecastScoreVect;
};

#endif // ASPARAMETERSCALIBRATION_H
