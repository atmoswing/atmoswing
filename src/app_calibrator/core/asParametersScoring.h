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
 * Portions Copyright 2013 Pascal Horton, Terr@num.
 */
 
#ifndef ASPARAMETERSSCORING_H
#define ASPARAMETERSSCORING_H

#include "asIncludes.h"
#include <asParameters.h>

class asFileParameters;

class asParametersScoring : public asParameters
{
public:

    typedef struct
    {
        wxString Name;
        int AnalogsNumber;
        wxString TimeArrayMode;
        double TimeArrayDate;
        int TimeArrayIntervalDays;
        bool Postprocess;
        float PostprocessDupliExp;
        wxString PostprocessMethod;
        float Threshold;
        float Percentile;
    } ParamsForecastScore;

    /** Vectors */
    typedef struct
    {
        VVectorString PreprocessDataId;
        VVectorFloat PreprocessLevels;
        VVectorDouble PreprocessDTimeHours;
        VVectorDouble PreprocessDTimeDays;
        VVectorDouble PreprocessTimeHour;
        VectorString DataId;
        VectorFloat Level;
        VectorDouble Umin;
        VectorInt Uptsnb;
        VectorDouble Vmin;
        VectorInt Vptsnb;
        VectorDouble DTimeHours;
        VectorString Criteria;
        VectorFloat Weight;
    } ParamsPredictorVect;

    typedef std::vector < ParamsPredictorVect > VectorParamsPredictorsVect;

    typedef struct
    {
        VectorInt AnalogsNumber;
        VectorParamsPredictorsVect Predictors;
    } ParamsStepVect;

    typedef std::vector < ParamsStepVect > VectorParamsStepVect;

    typedef struct
    {
        VectorString Name;
        VectorInt AnalogsNumber;
        VectorString TimeArrayMode;
        VectorDouble TimeArrayDate;
        VectorInt TimeArrayIntervalDays;
        VectorFloat PostprocessDupliExp;
    } ParamsForecastScoreVect;

    /** Booleans */
    typedef struct
    {
        VectorBool PreprocessDataId;
        VectorBool PreprocessLevels;
        VectorBool PreprocessDTimeHours;
        bool DataId;
        bool Level;
        bool Umin;
        bool Uptsnb;
        bool Vmin;
        bool Vptsnb;
        bool DTimeHours;
        bool Weight;
        bool Criteria;
    } ParamsPredictorBool;

    typedef std::vector < ParamsPredictorBool > VectorParamsPredictorsBool;

    typedef struct
    {
        bool AnalogsNumber;
        VectorParamsPredictorsBool Predictors;
    } ParamsStepBool;

    typedef std::vector < ParamsStepBool > VectorParamsStepBool;

    typedef struct
    {
        bool AnalogsNumber;
    } ParamsForecastScoreBool;


    /** Default constructor */
    asParametersScoring();

    /** Default destructor */
    virtual ~asParametersScoring();

    void AddPredictorVect(ParamsStepVect &step);

    VectorInt GetFileParamIntVector(asFileParameters &fileParams, const wxString &tag);

    VectorFloat GetFileParamFloatVector(asFileParameters &fileParams, const wxString &tag);

    VectorDouble GetFileParamDoubleVector(asFileParameters &fileParams, const wxString &tag);

    VectorString GetFileParamStringVector(asFileParameters &fileParams, const wxString &tag);

    bool FixAnalogsNb();

    wxString Print();

    int GetCalibrationYearStart()
    {
        return m_CalibrationYearStart;
    }

    void SetCalibrationYearStart(int val)
    {
        m_CalibrationYearStart = val;
    }

    int GetCalibrationYearEnd()
    {
        return m_CalibrationYearEnd;
    }

    void SetCalibrationYearEnd(int val)
    {
        m_CalibrationYearEnd = val;
    }

    VectorInt GetValidationYearsVector()
    {
        return m_ValidationYears;
    }

    void SetValidationYearsVector(VectorInt val)
    {
        m_ValidationYears = val;
    }

    bool HasValidationPeriod()
    {
        if (m_ValidationYears.size()>0) return true;
        else return false;

        return false;
    }

    wxString GetForecastScoreName()
    {
        return m_ForecastScore.Name;
    }

    void SetForecastScoreName(const wxString& val)
    {
        m_ForecastScore.Name = val;
    }

    float GetForecastScoreThreshold()
    {
        return m_ForecastScore.Threshold;
    }

    void SetForecastScoreThreshold(float val)
    {
        m_ForecastScore.Threshold = val;
    }

    float GetForecastScorePercentile()
    {
        return m_ForecastScore.Percentile;
    }

    void SetForecastScorePercentile(float val)
    {
        m_ForecastScore.Percentile = val;
    }

    int GetForecastScoreAnalogsNumber()
    {
        return m_ForecastScore.AnalogsNumber;
    }

    void SetForecastScoreAnalogsNumber(int val)
    {
        m_ForecastScore.AnalogsNumber = val;
    }

    wxString GetForecastScoreTimeArrayMode()
    {
        return m_ForecastScore.TimeArrayMode;
    }

    void SetForecastScoreTimeArrayMode(const wxString& val)
    {
        m_ForecastScore.TimeArrayMode = val;
    }

    double GetForecastScoreTimeArrayDate()
    {
        return m_ForecastScore.TimeArrayDate;
    }

    void SetForecastScoreTimeArrayDate(double val)
    {
        m_ForecastScore.TimeArrayDate = val;
    }

    int GetForecastScoreTimeArrayIntervalDays()
    {
        return m_ForecastScore.TimeArrayIntervalDays;
    }

    void SetForecastScoreTimeArrayIntervalDays(int val)
    {
        m_ForecastScore.TimeArrayIntervalDays = val;
    }

    bool ForecastScoreNeedsPostprocessing()
    {
        return m_ForecastScore.Postprocess;
    }

    void SetForecastScorePostprocess(bool val)
    {
        m_ForecastScore.Postprocess = val;
    }

    wxString GetForecastScorePostprocessMethod()
    {
        return m_ForecastScore.PostprocessMethod;
    }

    void SetForecastScorePostprocessMethod(const wxString& val)
    {
        m_ForecastScore.PostprocessMethod = val;
    }

    float GetForecastScorePostprocessDupliExp()
    {
        return m_ForecastScore.PostprocessDupliExp;
    }

    void SetForecastScorePostprocessDupliExp(float val)
    {
        m_ForecastScore.PostprocessDupliExp = val;
    }

protected:
    int m_CalibrationYearStart;
    int m_CalibrationYearEnd;
    VectorInt m_ValidationYears;

private:
    ParamsForecastScore m_ForecastScore;
};

#endif // ASPARAMETERSSCORING_H
