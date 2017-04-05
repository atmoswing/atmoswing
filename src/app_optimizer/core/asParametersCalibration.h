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

    int GetPreprocessDataIdVectorSize(int iStep, int iPtor, int iPre) const
    {
        return (int) GetPreprocessDataIdVector(iStep, iPtor, iPre).size();
    }

    vvi GetPredictandStationIdsVector() const
    {
        return m_predictandStationIdsVect;
    }

    bool SetPredictandStationIdsVector(vvi val);

    vi GetTimeArrayAnalogsIntervalDaysVector() const
    {
        return m_timeArrayAnalogsIntervalDaysVect;
    }

    bool SetTimeArrayAnalogsIntervalDaysVector(vi val);

    vwxs GetForecastScoreNameVector() const
    {
        return m_forecastScoreVect.name;
    }

    bool SetForecastScoreNameVector(vwxs val);

    vwxs GetForecastScoreTimeArrayModeVector() const
    {
        return m_forecastScoreVect.timeArrayMode;
    }

    bool SetForecastScoreTimeArrayModeVector(vwxs val);

    vd GetForecastScoreTimeArrayDateVector() const
    {
        return m_forecastScoreVect.timeArrayDate;
    }

    vi GetForecastScoreTimeArrayIntervalDaysVector() const
    {
        return m_forecastScoreVect.timeArrayIntervalDays;
    }

    vf GetForecastScorePostprocessDupliExpVector() const
    {
        return m_forecastScoreVect.postprocessDupliExp;
    }

    double GetPreprocessTimeHoursLowerLimit(int iStep, int iPtor, int iPre) const;

    double GetPredictorXminLowerLimit(int iStep, int iPtor) const;

    int GetPredictorXptsnbLowerLimit(int iStep, int iPtor) const;

    double GetPredictorYminLowerLimit(int iStep, int iPtor) const;

    int GetPredictorYptsnbLowerLimit(int iStep, int iPtor) const;

    double GetPredictorTimeHoursLowerLimit(int iStep, int iPtor) const;

    double GetPreprocessTimeHoursUpperLimit(int iStep, int iPtor, int iPre) const;

    double GetPredictorXminUpperLimit(int iStep, int iPtor) const;

    int GetPredictorXptsnbUpperLimit(int iStep, int iPtor) const;

    double GetPredictorYminUpperLimit(int iStep, int iPtor) const;

    int GetPredictorYptsnbUpperLimit(int iStep, int iPtor) const;

    double GetPredictorTimeHoursUpperLimit(int iStep, int iPtor) const;

    double GetPredictorXminIteration(int iStep, int iPtor) const;

    int GetPredictorXptsnbIteration(int iStep, int iPtor) const;

    double GetPredictorYminIteration(int iStep, int iPtor) const;

    int GetPredictorYptsnbIteration(int iStep, int iPtor) const;

protected:
    vd GetVectorXmin(asFileParametersCalibration &fileParams, wxXmlNode *node, int iStep, int iPtor);

    vd GetVectorYmin(asFileParametersCalibration &fileParams, wxXmlNode *node, int iStep, int iPtor);

private:
    vvi m_predictandStationIdsVect;
    vi m_timeArrayAnalogsIntervalDaysVect;
    ParamsForecastScoreVect m_forecastScoreVect;

    void GetAllPreprocessTimesAndLevels(int iStep, int iPtor, vf &preprocLevels, vd &preprocTimeHours) const;

    bool ParseDescription(asFileParametersCalibration &fileParams, const wxXmlNode *nodeProcess);

    bool ParseTimeProperties(asFileParametersCalibration &fileParams, const wxXmlNode *nodeProcess);

    bool ParseAnalogDatesParams(asFileParametersCalibration &fileParams, int iStep, const wxXmlNode *nodeProcess);

    bool ParsePreprocessedPredictors(asFileParametersCalibration &fileParams, int iStep, int iPtor,
                                     const wxXmlNode *nodeParam);

    bool ParseAnalogValuesParams(asFileParametersCalibration &fileParams, const wxXmlNode *nodeProcess);

    bool ParseForecastScore(asFileParametersCalibration &fileParams, const wxXmlNode *nodeProcess);

    bool ParseForecastScoreFinal(asFileParametersCalibration &fileParams, const wxXmlNode *nodeProcess);
};

#endif // ASPARAMETERSCALIBRATION_H
