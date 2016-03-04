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
        wxString TimeArrayMode;
        double TimeArrayDate;
        int TimeArrayIntervalDays;
        bool Postprocess;
        float PostprocessDupliExp;
        wxString PostprocessMethod;
        float Threshold;
        float Quantile;
    } ParamsForecastScore;

    /** Vectors */
    typedef struct
    {
        VVectorString PreprocessDataId;
        VVectorFloat PreprocessLevels;
        VVectorDouble PreprocessTimeHours;
        VectorString DataId;
        VectorFloat Level;
        VectorDouble Xmin;
        VectorInt Xptsnb;
        VectorDouble Ymin;
        VectorInt Yptsnb;
        VectorDouble TimeHours;
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
        VectorBool PreprocessTimeHours;
        bool DataId;
        bool Level;
        bool Xmin;
        bool Xptsnb;
        bool Ymin;
        bool Yptsnb;
        bool TimeHours;
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


    /** Default constructor */
    asParametersScoring();

    /** Default destructor */
    virtual ~asParametersScoring();

    void AddPredictorVect(ParamsStepVect &step);

    bool GenerateSimpleParametersFile(const wxString &filePath);

    wxString GetPredictandStationIdsVectorString(VVectorInt &predictandStationIdsVect);
    
    wxString Print();

    bool GetValuesFromString(wxString stringVals); // We copy the string as we'll modify it.

    bool SetCalibrationYearStart(int val)
    {
        m_calibrationStart = asTime::GetMJD(val, 1, 1);
        return true;
    }

    bool SetCalibrationYearEnd(int val)
    {
        m_calibrationEnd = asTime::GetMJD(val, 12, 31);
        return true;
    }

    double GetCalibrationStart()
    {
        return m_calibrationStart;
    }

    bool SetCalibrationStart(double val)
    {
        m_calibrationStart = val;
        return true;
    }

    bool SetCalibrationStart(wxString val)
    {
        m_calibrationStart = asTime::GetTimeFromString(val);
        return true;
    }
    
    double GetCalibrationEnd()
    {
        return m_calibrationEnd;
    }
    
    bool SetCalibrationEnd(double val)
    {
        m_calibrationEnd = val;
        return true;
    }
    
    bool SetCalibrationEnd(wxString val)
    {
        m_calibrationEnd = asTime::GetTimeFromString(val);
        return true;
    }

    VectorInt GetValidationYearsVector()
    {
        return m_validationYears;
    }

    bool SetValidationYearsVector(VectorInt val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided validation years vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<(int)val.size(); i++)
            {
                if (asTools::IsNaN(val[i]))
                {
                    asLogError(_("There are NaN values in the provided validation years vector."));
                    return false;
                }
            }
        }
        m_validationYears = val;
        return true;
    }

    bool HasValidationPeriod()
    {
        if (m_validationYears.size()>0) return true;
        else return false;
    }

    int GetValidationYearStart()
    {
        if (!HasValidationPeriod()) {
            return NaNInt;
        }

        int minVal = m_validationYears[0];
        for (int i=0; i<(int)m_validationYears.size(); i++)
        {
            minVal = wxMin(minVal, m_validationYears[i]);
        }

        return minVal;
    }

    int GetValidationYearEnd()
    {
        if (!HasValidationPeriod()) {
            return NaNInt;
        }

        int maxVal = m_validationYears[0];
        for (int i=0; i<(int)m_validationYears.size(); i++)
        {
            maxVal = wxMax(maxVal, m_validationYears[i]);
        }

        return maxVal;
    }

    wxString GetForecastScoreName()
    {
        return m_forecastScore.Name;
    }

    bool SetForecastScoreName(const wxString& val)
    {
        if (val.IsEmpty())
        {
            asLogError(_("The provided forecast score is null"));
            return false;
        }
        m_forecastScore.Name = val;
        return true;
    }

    float GetForecastScoreThreshold()
    {
        return m_forecastScore.Threshold;
    }

    void SetForecastScoreThreshold(float val)
    {
        m_forecastScore.Threshold = val;
    }

    float GetForecastScoreQuantile()
    {
        return m_forecastScore.Quantile;
    }

    void SetForecastScoreQuantile(float val)
    {
        m_forecastScore.Quantile = val;
    }

    int GetForecastScoreAnalogsNumber()
    {
        return GetAnalogsNumber(GetStepsNb()-1);
    }

    wxString GetForecastScoreTimeArrayMode()
    {
        return m_forecastScore.TimeArrayMode;
    }

    bool SetForecastScoreTimeArrayMode(const wxString& val)
    {
        if (val.IsEmpty())
        {
            asLogError(_("The provided time array mode for the forecast score is null"));
            return false;
        }
        m_forecastScore.TimeArrayMode = val;
        return true;
    }

    double GetForecastScoreTimeArrayDate()
    {
        return m_forecastScore.TimeArrayDate;
    }

    void SetForecastScoreTimeArrayDate(double val)
    {
        m_forecastScore.TimeArrayDate = val;
    }

    int GetForecastScoreTimeArrayIntervalDays()
    {
        return m_forecastScore.TimeArrayIntervalDays;
    }

    void SetForecastScoreTimeArrayIntervalDays(int val)
    {
        m_forecastScore.TimeArrayIntervalDays = val;
    }

    bool ForecastScoreNeedsPostprocessing()
    {
        return m_forecastScore.Postprocess;
    }

    void SetForecastScorePostprocess(bool val)
    {
        m_forecastScore.Postprocess = val;
    }

    wxString GetForecastScorePostprocessMethod()
    {
        return m_forecastScore.PostprocessMethod;
    }

    bool SetForecastScorePostprocessMethod(const wxString& val)
    {
        if (val.IsEmpty() && ForecastScoreNeedsPostprocessing())
        {
            asLogError(_("The provided value for the postprocessing method is null"));
            return false;
        }
        m_forecastScore.PostprocessMethod = val;
        return true;
    }

    float GetForecastScorePostprocessDupliExp()
    {
        return m_forecastScore.PostprocessDupliExp;
    }

    void SetForecastScorePostprocessDupliExp(float val)
    {
        m_forecastScore.PostprocessDupliExp = val;
    }

protected:
    double m_calibrationStart;
    double m_calibrationEnd;
    VectorInt m_validationYears;

private:
    ParamsForecastScore m_forecastScore;
};

#endif // ASPARAMETERSSCORING_H
