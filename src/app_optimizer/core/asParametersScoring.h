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

#ifndef ASPARAMETERSSCORING_H
#define ASPARAMETERSSCORING_H

#include "asIncludes.h"
#include <asParameters.h>

class asFileParameters;

class asParametersScoring
        : public asParameters
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

    typedef std::vector<ParamsPredictorVect> VectorParamsPredictorsVect;

    typedef struct
    {
        VectorInt AnalogsNumber;
        VectorParamsPredictorsVect Predictors;
    } ParamsStepVect;

    typedef std::vector<ParamsStepVect> VectorParamsStepVect;

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

    typedef std::vector<ParamsPredictorBool> VectorParamsPredictorsBool;

    typedef struct
    {
        bool AnalogsNumber;
        VectorParamsPredictorsBool Predictors;
    } ParamsStepBool;

    typedef std::vector<ParamsStepBool> VectorParamsStepBool;


    asParametersScoring();

    virtual ~asParametersScoring();

    void AddPredictorVect(ParamsStepVect &step);

    bool GenerateSimpleParametersFile(const wxString &filePath);

    bool PreprocessingPropertiesOk();

    wxString GetPredictandStationIdsVectorString(VVectorInt &predictandStationIdsVect);

    wxString Print();

    virtual int GetPreprocessDataIdVectorSize(int i_step, int i_ptor, int i_preproc)
    {
        return 1;
    }

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

    bool SetCalibrationStart(wxString val)
    {
        m_calibrationStart = asTime::GetTimeFromString(val);
        return true;
    }

    double GetCalibrationEnd()
    {
        return m_calibrationEnd;
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
        if (val.size() < 1) {
            asLogError(_("The provided validation years vector is empty."));
            return false;
        } else {
            for (int i = 0; i < (int) val.size(); i++) {
                if (asTools::IsNaN(val[i])) {
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
        return m_validationYears.size() > 0;
    }

    wxString GetForecastScoreName()
    {
        return m_forecastScore.Name;
    }

    bool SetForecastScoreName(const wxString &val)
    {
        if (val.IsEmpty()) {
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
        return GetAnalogsNumber(GetStepsNb() - 1);
    }

    wxString GetForecastScoreTimeArrayMode()
    {
        return m_forecastScore.TimeArrayMode;
    }

    bool SetForecastScoreTimeArrayMode(const wxString &val)
    {
        if (val.IsEmpty()) {
            asLogError(_("The provided time array mode for the forecast score is null"));
            return false;
        }
        m_forecastScore.TimeArrayMode = val;
        return true;
    }

    bool ForecastScoreNeedsPostprocessing()
    {
        return m_forecastScore.Postprocess;
    }

    /* Vector elements */

    VectorInt GetAnalogsNumberVector(int i_step)
    {
        return m_stepsVect[i_step].AnalogsNumber;
    }

    bool SetAnalogsNumberVector(int i_step, VectorInt val)
    {
        if (val.size() < 1) {
            asLogError(_("The provided analogs number vector is empty."));
            return false;
        } else {
            for (int i = 0; i < (int) val.size(); i++) {
                if (asTools::IsNaN(val[i])) {
                    asLogError(_("There are NaN values in the provided analogs number vector."));
                    return false;
                }
            }
        }
        m_stepsVect[i_step].AnalogsNumber = val;
        return true;
    }

    bool SetPreprocessTimeHoursVector(int i_step, int i_predictor, int i_dataset, VectorDouble val)
    {
        if (val.size() < 1) {
            asLogError(_("The provided preprocess time (hours) vector is empty."));
            return false;
        } else {
            for (int i = 0; i < (int) val.size(); i++) {
                if (asTools::IsNaN(val[i])) {
                    asLogError(_("There are NaN values in the provided preprocess time (hours) vector."));
                    return false;
                }
            }
        }

        if (m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours.size() >= (unsigned) (i_dataset + 1)) {
            m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset].clear();
            m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset] = val;
        } else {
            m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours.push_back(val);
        }

        return true;
    }

    VectorDouble GetPredictorXminVector(int i_step, int i_predictor)
    {
        return m_stepsVect[i_step].Predictors[i_predictor].Xmin;
    }

    bool SetPredictorXminVector(int i_step, int i_predictor, VectorDouble val)
    {
        if (val.size() < 1) {
            asLogError(_("The provided Xmin vector is empty."));
            return false;
        } else {
            for (int i = 0; i < (int) val.size(); i++) {
                if (asTools::IsNaN(val[i])) {
                    asLogError(_("There are NaN values in the provided Xmin vector."));
                    return false;
                }
            }
        }
        m_stepsVect[i_step].Predictors[i_predictor].Xmin = val;
        return true;
    }

    VectorInt GetPredictorXptsnbVector(int i_step, int i_predictor)
    {
        return m_stepsVect[i_step].Predictors[i_predictor].Xptsnb;
    }

    bool SetPredictorXptsnbVector(int i_step, int i_predictor, VectorInt val)
    {
        if (val.size() < 1) {
            asLogError(_("The provided Xptsnb vector is empty."));
            return false;
        } else {
            for (int i = 0; i < (int) val.size(); i++) {
                if (asTools::IsNaN(val[i])) {
                    asLogError(_("There are NaN values in the provided Xptsnb vector."));
                    return false;
                }
            }
        }
        m_stepsVect[i_step].Predictors[i_predictor].Xptsnb = val;
        return true;
    }

    VectorDouble GetPredictorYminVector(int i_step, int i_predictor)
    {
        return m_stepsVect[i_step].Predictors[i_predictor].Ymin;
    }

    bool SetPredictorYminVector(int i_step, int i_predictor, VectorDouble val)
    {
        if (val.size() < 1) {
            asLogError(_("The provided Ymin vector is empty."));
            return false;
        } else {
            for (int i = 0; i < (int) val.size(); i++) {
                if (asTools::IsNaN(val[i])) {
                    asLogError(_("There are NaN values in the provided Ymin vector."));
                    return false;
                }
            }
        }
        m_stepsVect[i_step].Predictors[i_predictor].Ymin = val;
        return true;
    }

    VectorInt GetPredictorYptsnbVector(int i_step, int i_predictor)
    {
        return m_stepsVect[i_step].Predictors[i_predictor].Yptsnb;
    }

    bool SetPredictorYptsnbVector(int i_step, int i_predictor, VectorInt val)
    {
        if (val.size() < 1) {
            asLogError(_("The provided Yptsnb vector is empty."));
            return false;
        } else {
            for (int i = 0; i < (int) val.size(); i++) {
                if (asTools::IsNaN(val[i])) {
                    asLogError(_("There are NaN values in the provided Yptsnb vector."));
                    return false;
                }
            }
        }
        m_stepsVect[i_step].Predictors[i_predictor].Yptsnb = val;
        return true;
    }

    VectorDouble GetPredictorTimeHoursVector(int i_step, int i_predictor)
    {
        return m_stepsVect[i_step].Predictors[i_predictor].TimeHours;
    }

    bool SetPredictorTimeHoursVector(int i_step, int i_predictor, VectorDouble val)
    {
        if (val.size() < 1) {
            asLogError(_("The provided predictor time (hours) vector is empty."));
            return false;
        } else {
            for (int i = 0; i < (int) val.size(); i++) {
                if (asTools::IsNaN(val[i])) {
                    asLogError(_("There are NaN values in the provided predictor time (hours) vector."));
                    return false;
                }
            }
        }
        m_stepsVect[i_step].Predictors[i_predictor].TimeHours = val;
        return true;
    }

    VectorFloat GetPredictorWeightVector(int i_step, int i_predictor)
    {
        return m_stepsVect[i_step].Predictors[i_predictor].Weight;
    }

    bool SetPredictorWeightVector(int i_step, int i_predictor, VectorFloat val)
    {
        if (val.size() < 1) {
            asLogError(_("The provided predictor weights vector is empty."));
            return false;
        } else {
            for (int i = 0; i < (int) val.size(); i++) {
                if (asTools::IsNaN(val[i])) {
                    asLogError(_("There are NaN values in the provided predictor weights vector."));
                    return false;
                }
            }
        }
        m_stepsVect[i_step].Predictors[i_predictor].Weight = val;
        return true;
    }

    VectorString GetPreprocessDataIdVector(int i_step, int i_predictor, int i_dataset)
    {
        if (m_stepsVect[i_step].Predictors[i_predictor].PreprocessDataId.size() >= (unsigned) (i_dataset + 1)) {
            return m_stepsVect[i_step].Predictors[i_predictor].PreprocessDataId[i_dataset];
        } else {
            asLogError(_("Trying to access to an element outside of PreprocessDataId in the parameters object."));
            VectorString empty;
            return empty;
        }
    }

    bool SetPreprocessDataIdVector(int i_step, int i_predictor, int i_dataset, VectorString val)
    {
        if (val.size() < 1) {
            asLogError(_("The provided preprocess data ID vector is empty."));
            return false;
        } else {
            for (int i = 0; i < val.size(); i++) {
                if (val[i].IsEmpty()) {
                    asLogError(_("There are NaN values in the provided preprocess data ID vector."));
                    return false;
                }
            }
        }

        if (m_stepsVect[i_step].Predictors[i_predictor].PreprocessDataId.size() >= (unsigned) (i_dataset + 1)) {
            m_stepsVect[i_step].Predictors[i_predictor].PreprocessDataId[i_dataset].clear();
            m_stepsVect[i_step].Predictors[i_predictor].PreprocessDataId[i_dataset] = val;
        } else {
            m_stepsVect[i_step].Predictors[i_predictor].PreprocessDataId.push_back(val);
        }

        return true;
    }

    VectorFloat GetPreprocessLevelVector(int i_step, int i_predictor, int i_dataset)
    {
        if (m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels.size() >= (unsigned) (i_dataset + 1)) {
            return m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset];
        } else {
            asLogError(_("Trying to access to an element outside of PreprocessLevels in the parameters object."));
            VectorFloat empty;
            return empty;
        }
    }

    bool SetPreprocessLevelVector(int i_step, int i_predictor, int i_dataset, VectorFloat val)
    {
        if (val.size() < 1) {
            asLogError(_("The provided preprocess levels vector is empty."));
            return false;
        } else {
            for (int i = 0; i < val.size(); i++) {
                if (asTools::IsNaN(val[i])) {
                    asLogError(_("There are NaN values in the provided preprocess levels vector."));
                    return false;
                }
            }
        }

        if (m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels.size() >= (unsigned) (i_dataset + 1)) {
            m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset].clear();
            m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset] = val;
        } else {
            m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels.push_back(val);
        }

        return true;
    }

    VectorDouble GetPreprocessTimeHoursVector(int i_step, int i_predictor, int i_dataset)
    {
        wxASSERT(m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours.size() > (unsigned) i_dataset);

        if (m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours.size() >= (unsigned) (i_dataset + 1)) {
            return m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset];
        } else {
            asLogError(
                    _("Trying to access to an element outside of PreprocessTimeHours (vect) in the parameters object."));
            VectorDouble empty;
            return empty;
        }
    }

    VectorString GetPredictorDataIdVector(int i_step, int i_predictor)
    {
        return m_stepsVect[i_step].Predictors[i_predictor].DataId;
    }

    bool SetPredictorDataIdVector(int i_step, int i_predictor, VectorString val)
    {
        if (val.size() < 1) {
            asLogError(_("The provided data ID vector is empty."));
            return false;
        } else {
            for (int i = 0; i < val.size(); i++) {
                if (val[i].IsEmpty()) {
                    asLogError(_("There are NaN values in the provided data ID vector."));
                    return false;
                }
            }
        }
        m_stepsVect[i_step].Predictors[i_predictor].DataId = val;
        return true;
    }

    VectorFloat GetPredictorLevelVector(int i_step, int i_predictor)
    {
        return m_stepsVect[i_step].Predictors[i_predictor].Level;
    }

    bool SetPredictorLevelVector(int i_step, int i_predictor, VectorFloat val)
    {
        if (val.size() < 1) {
            asLogError(_("The provided predictor levels vector is empty."));
            return false;
        } else {
            for (int i = 0; i < val.size(); i++) {
                if (asTools::IsNaN(val[i])) {
                    asLogError(_("There are NaN values in the provided predictor levels vector."));
                    return false;
                }
            }
        }
        m_stepsVect[i_step].Predictors[i_predictor].Level = val;
        return true;
    }

    VectorString GetPredictorCriteriaVector(int i_step, int i_predictor)
    {
        return m_stepsVect[i_step].Predictors[i_predictor].Criteria;
    }

    bool SetPredictorCriteriaVector(int i_step, int i_predictor, VectorString val)
    {
        if (val.size() < 1) {
            asLogError(_("The provided predictor criteria vector is empty."));
            return false;
        } else {
            for (int i = 0; i < val.size(); i++) {
                if (val[i].IsEmpty()) {
                    asLogError(_("There are NaN values in the provided predictor criteria vector."));
                    return false;
                }
            }
        }
        m_stepsVect[i_step].Predictors[i_predictor].Criteria = val;
        return true;
    }

protected:
    double m_calibrationStart;
    double m_calibrationEnd;
    VectorInt m_validationYears;
    VectorParamsStepVect m_stepsVect;

private:
    ParamsForecastScore m_forecastScore;
};

#endif // ASPARAMETERSSCORING_H
