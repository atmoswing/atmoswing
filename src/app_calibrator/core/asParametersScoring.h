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
        VVectorDouble PreprocessTimeHours;
        VectorString DataId;
        VectorFloat Level;
        VectorDouble Umin;
        VectorInt Uptsnb;
        VectorDouble Vmin;
        VectorInt Vptsnb;
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
        VectorBool PreprocessTimeHours;
        bool DataId;
        bool Level;
        bool Umin;
        bool Uptsnb;
        bool Vmin;
        bool Vptsnb;
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

    typedef struct
    {
        bool AnalogsNumber;
    } ParamsForecastScoreBool;


    /** Default constructor */
    asParametersScoring();

    /** Default destructor */
    virtual ~asParametersScoring();

    void AddPredictorVect(ParamsStepVect &step);

    bool GenerateSimpleParametersFile(const wxString &filePath);

    wxString GetPredictandStationIdsVectorString(VVectorInt &predictandStationIdsVect);

    VectorInt GetFileParamIntVector(asFileParameters &fileParams, const wxString &tag);

    VectorFloat GetFileParamFloatVector(asFileParameters &fileParams, const wxString &tag);

    VectorDouble GetFileParamDoubleVector(asFileParameters &fileParams, const wxString &tag);

    VectorString GetFileParamStringVector(asFileParameters &fileParams, const wxString &tag);

    VVectorInt GetFileStationIdsVector(asFileParameters &fileParams);

    bool FixAnalogsNb();

    wxString Print();

    bool GetValuesFromString(wxString stringVals); // We copy the string as we'll modify it.

    bool SetCalibrationYearStart(int val)
    {
        m_CalibrationStart = asTime::GetMJD(val, 1, 1);
        return true;
    }

    bool SetCalibrationYearEnd(int val)
    {
        m_CalibrationEnd = asTime::GetMJD(val, 12, 31);
        return true;
    }

    double GetCalibrationStart()
    {
        return m_CalibrationStart;
    }

    bool SetCalibrationStart(double val)
    {
        m_CalibrationStart = val;
        return true;
    }

    bool SetCalibrationStart(wxString val)
    {
        m_CalibrationStart = asTime::GetTimeFromString(val);
        return true;
    }
    
    double GetCalibrationEnd()
    {
        return m_CalibrationEnd;
    }
    
    bool SetCalibrationEnd(double val)
    {
        m_CalibrationEnd = val;
        return true;
    }
    
    bool SetCalibrationEnd(wxString val)
    {
        m_CalibrationEnd = asTime::GetTimeFromString(val);
        return true;
    }

    VectorInt GetValidationYearsVector()
    {
        return m_ValidationYears;
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
            for (int i=0; i<val.size(); i++)
            {
                if (asTools::IsNaN(val[i]))
                {
                    asLogError(_("There are NaN values in the provided validation years vector."));
                    return false;
                }
            }
        }
        m_ValidationYears = val;
        return true;
    }

    bool HasValidationPeriod()
    {
        if (m_ValidationYears.size()>0) return true;
        else return false;
    }

    int GetValidationYearStart()
    {
        if (!HasValidationPeriod()) {
            return NaNInt;
        }

        int minVal = m_ValidationYears[0];
        for (int i=0; i<m_ValidationYears.size(); i++)
        {
            minVal = wxMin(minVal, m_ValidationYears[i]);
        }

        return minVal;
    }

    int GetValidationYearEnd()
    {
        if (!HasValidationPeriod()) {
            return NaNInt;
        }

        int maxVal = m_ValidationYears[0];
        for (int i=0; i<m_ValidationYears.size(); i++)
        {
            maxVal = wxMax(maxVal, m_ValidationYears[i]);
        }

        return maxVal;
    }

    wxString GetForecastScoreName()
    {
        return m_ForecastScore.Name;
    }

    bool SetForecastScoreName(const wxString& val)
    {
        if (val.IsEmpty())
        {
            asLogError(_("The provided forecast score is null"));
            return false;
        }
        m_ForecastScore.Name = val;
        return true;
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

    bool SetForecastScoreAnalogsNumber(int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the final analogs number is null"));
            return false;
        }
        m_ForecastScore.AnalogsNumber = val;
        return true;
    }

    wxString GetForecastScoreTimeArrayMode()
    {
        return m_ForecastScore.TimeArrayMode;
    }

    bool SetForecastScoreTimeArrayMode(const wxString& val)
    {
        if (val.IsEmpty())
        {
            asLogError(_("The provided time array mode for the forecast score is null"));
            return false;
        }
        m_ForecastScore.TimeArrayMode = val;
        return true;
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

    bool SetForecastScorePostprocessMethod(const wxString& val)
    {
        if (val.IsEmpty() && ForecastScoreNeedsPostprocessing())
        {
            asLogError(_("The provided value for the postprocessing method is null"));
            return false;
        }
        m_ForecastScore.PostprocessMethod = val;
        return true;
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
    double m_CalibrationStart;
    double m_CalibrationEnd;
    VectorInt m_ValidationYears;

private:
    ParamsForecastScore m_ForecastScore;
};

#endif // ASPARAMETERSSCORING_H
