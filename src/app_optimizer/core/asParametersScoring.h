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
        wxString name;
        wxString timeArrayMode;
        double timeArrayDate;
        int timeArrayIntervalDays;
        bool postprocess;
        float postprocessDupliExp;
        wxString postprocessMethod;
        float threshold;
        float quantile;
    } ParamsScore;

    /** Vectors */
    typedef struct
    {
        vvwxs preprocessDataId;
        vvf preprocessLevels;
        vvd preprocessTimeHours;
        vwxs dataId;
        vf level;
        vd xMin;
        vi xPtsNb;
        vd yMin;
        vi yPtsNb;
        vd timeHours;
        vwxs criteria;
        vf weight;
    } ParamsPredictorVect;

    typedef std::vector<ParamsPredictorVect> VectorParamsPredictorsVect;

    typedef struct
    {
        vi analogsNumber;
        VectorParamsPredictorsVect predictors;
    } ParamsStepVect;

    typedef std::vector<ParamsStepVect> VectorParamsStepVect;

    typedef struct
    {
        vwxs name;
        vwxs timeArrayMode;
        vd timeArrayDate;
        vi timeArrayIntervalDays;
        vf postprocessDupliExp;
    } ParamsScoreVect;

    /** Booleans */
    typedef struct
    {
        vb preprocessDataId;
        vb preprocessLevels;
        vb preprocessTimeHours;
        bool dataId;
        bool level;
        bool xMin;
        bool xPtsNb;
        bool yMin;
        bool yPtsNb;
        bool timeHours;
        bool weight;
        bool criteria;
    } ParamsPredictorBool;

    typedef std::vector<ParamsPredictorBool> VectorParamsPredictorsBool;

    typedef struct
    {
        bool analogsNumber;
        VectorParamsPredictorsBool predictors;
    } ParamsStepBool;

    typedef std::vector<ParamsStepBool> VectorParamsStepBool;

    asParametersScoring();

    ~asParametersScoring() override;

    void AddPredictorVect(ParamsStepVect &step);

    bool GenerateSimpleParametersFile(const wxString &filePath) const;

    bool PreprocessingDataIdsOk();

    wxString GetPredictandStationIdsVectorString(vvi &predictandStationIdsVect) const;

    wxString Print() const override;

    virtual int GetPreprocessDataIdVectorSize(int iStep, int iPtor, int iPre) const
    {
        return 1;
    }

    bool GetValuesFromString(wxString stringVals) override; // We copy the string as we'll modify it.

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

    double GetCalibrationStart() const
    {
        return m_calibrationStart;
    }

    bool SetCalibrationStart(const wxString &val)
    {
        m_calibrationStart = asTime::GetTimeFromString(val);
        return true;
    }

    double GetCalibrationEnd() const
    {
        return m_calibrationEnd;
    }

    bool SetCalibrationEnd(const wxString &val)
    {
        m_calibrationEnd = asTime::GetTimeFromString(val);
        return true;
    }

    vi GetValidationYearsVector() const
    {
        return m_validationYears;
    }

    bool SetValidationYearsVector(vi val)
    {
        if (val.empty()) {
            wxLogError(_("The provided validation years vector is empty."));
            return false;
        } else {
            for (int y : val) {
                if (asIsNaN(y)) {
                    wxLogError(_("There are NaN values in the provided validation years vector."));
                    return false;
                }
            }
        }
        m_validationYears = val;
        return true;
    }

    bool HasValidationPeriod() const
    {
        return !m_validationYears.empty();
    }

    wxString GetScoreName() const
    {
        return m_score.name;
    }

    bool SetScoreName(const wxString &val)
    {
        if (val.IsEmpty()) {
            wxLogError(_("The provided score is null"));
            return false;
        }
        m_score.name = val;
        return true;
    }

    float GetScoreThreshold() const
    {
        return m_score.threshold;
    }

    void SetScoreThreshold(float val)
    {
        m_score.threshold = val;
    }

    float GetScoreQuantile() const
    {
        return m_score.quantile;
    }

    void SetScoreQuantile(float val)
    {
        m_score.quantile = val;
    }

    int GetScoreAnalogsNumber() const
    {
        return GetAnalogsNumber(GetStepsNb() - 1);
    }

    wxString GetScoreTimeArrayMode() const
    {
        return m_score.timeArrayMode;
    }

    bool SetScoreTimeArrayMode(const wxString &val)
    {
        if (val.IsEmpty()) {
            wxLogError(_("The provided time array mode for the score is null"));
            return false;
        }
        m_score.timeArrayMode = val;
        return true;
    }

    bool ScoreNeedsPostprocessing() const
    {
        return m_score.postprocess;
    }

    /* Vector elements */

    vi GetAnalogsNumberVector(int iStep) const
    {
        return m_stepsVect[iStep].analogsNumber;
    }

    bool SetAnalogsNumberVector(int iStep, vi val)
    {
        if (val.empty()) {
            wxLogError(_("The provided analogs number vector is empty."));
            return false;
        } else {
            for (int n : val) {
                if (asIsNaN(n)) {
                    wxLogError(_("There are NaN values in the provided analogs number vector."));
                    return false;
                }
            }
        }
        m_stepsVect[iStep].analogsNumber = val;
        return true;
    }

    bool SetPreprocessTimeHoursVector(int iStep, int iPtor, int iPre, vd val)
    {
        if (val.empty()) {
            wxLogError(_("The provided preprocess time (hours) vector is empty."));
            return false;
        } else {
            for (double v : val) {
                if (asIsNaN(v)) {
                    wxLogError(_("There are NaN values in the provided preprocess time (hours) vector."));
                    return false;
                }
            }
        }

        if (m_stepsVect[iStep].predictors[iPtor].preprocessTimeHours.size() >= (unsigned) (iPre + 1)) {
            m_stepsVect[iStep].predictors[iPtor].preprocessTimeHours[iPre].clear();
            m_stepsVect[iStep].predictors[iPtor].preprocessTimeHours[iPre] = val;
        } else {
            m_stepsVect[iStep].predictors[iPtor].preprocessTimeHours.push_back(val);
        }

        return true;
    }

    vd GetPredictorXminVector(int iStep, int iPtor) const
    {
        return m_stepsVect[iStep].predictors[iPtor].xMin;
    }

    bool SetPredictorXminVector(int iStep, int iPtor, vd val)
    {
        if (val.empty()) {
            wxLogError(_("The provided xMin vector is empty."));
            return false;
        } else {
            for (double v : val) {
                if (asIsNaN(v)) {
                    wxLogError(_("There are NaN values in the provided xMin vector."));
                    return false;
                }
            }
        }
        m_stepsVect[iStep].predictors[iPtor].xMin = val;
        return true;
    }

    vi GetPredictorXptsnbVector(int iStep, int iPtor) const
    {
        return m_stepsVect[iStep].predictors[iPtor].xPtsNb;
    }

    bool SetPredictorXptsnbVector(int iStep, int iPtor, vi val)
    {
        if (val.empty()) {
            wxLogError(_("The provided xPtsNb vector is empty."));
            return false;
        } else {
            for (int v : val) {
                if (asIsNaN(v)) {
                    wxLogError(_("There are NaN values in the provided xPtsNb vector."));
                    return false;
                }
            }
        }
        m_stepsVect[iStep].predictors[iPtor].xPtsNb = val;
        return true;
    }

    vd GetPredictorYminVector(int iStep, int iPtor) const
    {
        return m_stepsVect[iStep].predictors[iPtor].yMin;
    }

    bool SetPredictorYminVector(int iStep, int iPtor, vd val)
    {
        if (val.empty()) {
            wxLogError(_("The provided yMin vector is empty."));
            return false;
        } else {
            for (double v : val) {
                if (asIsNaN(v)) {
                    wxLogError(_("There are NaN values in the provided yMin vector."));
                    return false;
                }
            }
        }
        m_stepsVect[iStep].predictors[iPtor].yMin = val;
        return true;
    }

    vi GetPredictorYptsnbVector(int iStep, int iPtor) const
    {
        return m_stepsVect[iStep].predictors[iPtor].yPtsNb;
    }

    bool SetPredictorYptsnbVector(int iStep, int iPtor, vi val)
    {
        if (val.empty()) {
            wxLogError(_("The provided yPtsNb vector is empty."));
            return false;
        } else {
            for (int v : val) {
                if (asIsNaN(v)) {
                    wxLogError(_("There are NaN values in the provided yPtsNb vector."));
                    return false;
                }
            }
        }
        m_stepsVect[iStep].predictors[iPtor].yPtsNb = val;
        return true;
    }

    vd GetPredictorTimeHoursVector(int iStep, int iPtor) const
    {
        return m_stepsVect[iStep].predictors[iPtor].timeHours;
    }

    bool SetPredictorTimeHoursVector(int iStep, int iPtor, vd val)
    {
        if (val.empty()) {
            wxLogError(_("The provided predictor time (hours) vector is empty."));
            return false;
        } else {
            for (double v : val) {
                if (asIsNaN(v)) {
                    wxLogError(_("There are NaN values in the provided predictor time (hours) vector."));
                    return false;
                }
            }
        }
        m_stepsVect[iStep].predictors[iPtor].timeHours = val;
        return true;
    }

    vf GetPredictorWeightVector(int iStep, int iPtor) const
    {
        return m_stepsVect[iStep].predictors[iPtor].weight;
    }

    bool SetPredictorWeightVector(int iStep, int iPtor, vf val)
    {
        if (val.empty()) {
            wxLogError(_("The provided predictor weights vector is empty."));
            return false;
        } else {
            for (float v : val) {
                if (asIsNaN(v)) {
                    wxLogError(_("There are NaN values in the provided predictor weights vector."));
                    return false;
                }
            }
        }
        m_stepsVect[iStep].predictors[iPtor].weight = val;
        return true;
    }

    vwxs GetPreprocessDataIdVector(int iStep, int iPtor, int iPre) const
    {
        if (m_stepsVect[iStep].predictors[iPtor].preprocessDataId.size() >= (unsigned) (iPre + 1)) {
            return m_stepsVect[iStep].predictors[iPtor].preprocessDataId[iPre];
        } else {
            wxLogError(_("Trying to access to an element outside of preprocessDataId in the parameters object."));
            vwxs empty;
            return empty;
        }
    }

    bool SetPreprocessDataIdVector(int iStep, int iPtor, int iPre, vwxs val)
    {
        if (val.empty()) {
            wxLogError(_("The provided preprocess data ID vector is empty."));
            return false;
        } else {
            for (auto &v : val) {
                if (v.IsEmpty()) {
                    wxLogError(_("There are NaN values in the provided preprocess data ID vector."));
                    return false;
                }
            }
        }

        if (m_stepsVect[iStep].predictors[iPtor].preprocessDataId.size() >= (unsigned) (iPre + 1)) {
            m_stepsVect[iStep].predictors[iPtor].preprocessDataId[iPre].clear();
            m_stepsVect[iStep].predictors[iPtor].preprocessDataId[iPre] = val;
        } else {
            m_stepsVect[iStep].predictors[iPtor].preprocessDataId.push_back(val);
        }

        return true;
    }

    vf GetPreprocessLevelVector(int iStep, int iPtor, int iPre) const
    {
        if (m_stepsVect[iStep].predictors[iPtor].preprocessLevels.size() >= (unsigned) (iPre + 1)) {
            return m_stepsVect[iStep].predictors[iPtor].preprocessLevels[iPre];
        } else {
            wxLogError(_("Trying to access to an element outside of preprocessLevels in the parameters object."));
            vf empty;
            return empty;
        }
    }

    bool SetPreprocessLevelVector(int iStep, int iPtor, int iPre, vf val)
    {
        if (val.empty()) {
            wxLogError(_("The provided preprocess levels vector is empty."));
            return false;
        } else {
            for (float v : val) {
                if (asIsNaN(v)) {
                    wxLogError(_("There are NaN values in the provided preprocess levels vector."));
                    return false;
                }
            }
        }

        if (m_stepsVect[iStep].predictors[iPtor].preprocessLevels.size() >= (unsigned) (iPre + 1)) {
            m_stepsVect[iStep].predictors[iPtor].preprocessLevels[iPre].clear();
            m_stepsVect[iStep].predictors[iPtor].preprocessLevels[iPre] = val;
        } else {
            m_stepsVect[iStep].predictors[iPtor].preprocessLevels.push_back(val);
        }

        return true;
    }

    vd GetPreprocessTimeHoursVector(int iStep, int iPtor, int iPre) const
    {
        wxASSERT(m_stepsVect[iStep].predictors[iPtor].preprocessTimeHours.size() > (unsigned) iPre);

        if (m_stepsVect[iStep].predictors[iPtor].preprocessTimeHours.size() >= (unsigned) (iPre + 1)) {
            return m_stepsVect[iStep].predictors[iPtor].preprocessTimeHours[iPre];
        } else {
            wxLogError(_("Trying to access to an element outside of preprocessTimeHours (vect) in the parameters object."));
            vd empty;
            return empty;
        }
    }

    vwxs GetPredictorDataIdVector(int iStep, int iPtor) const
    {
        return m_stepsVect[iStep].predictors[iPtor].dataId;
    }

    int GetPredictorDataIdNb(int iStep, int iPtor) const override
    {
        return (int)m_stepsVect[iStep].predictors[iPtor].dataId.size();
    }

    bool SetPredictorDataIdVector(int iStep, int iPtor, vwxs val)
    {
        if (val.empty()) {
            wxLogError(_("The provided data ID vector is empty."));
            return false;
        } else {
            for (auto &v : val) {
                if (v.IsEmpty()) {
                    wxLogError(_("There are NaN values in the provided data ID vector."));
                    return false;
                }
            }
        }
        m_stepsVect[iStep].predictors[iPtor].dataId = val;
        return true;
    }

    vf GetPredictorLevelVector(int iStep, int iPtor) const
    {
        return m_stepsVect[iStep].predictors[iPtor].level;
    }

    bool SetPredictorLevelVector(int iStep, int iPtor, vf val)
    {
        if (val.empty()) {
            wxLogError(_("The provided predictor levels vector is empty."));
            return false;
        } else {
            for (float v : val) {
                if (asIsNaN(v)) {
                    wxLogError(_("There are NaN values in the provided predictor levels vector."));
                    return false;
                }
            }
        }
        m_stepsVect[iStep].predictors[iPtor].level = val;
        return true;
    }

    vwxs GetPredictorCriteriaVector(int iStep, int iPtor) const
    {
        return m_stepsVect[iStep].predictors[iPtor].criteria;
    }

    bool SetPredictorCriteriaVector(int iStep, int iPtor, vwxs val)
    {
        if (val.empty()) {
            wxLogError(_("The provided predictor criteria vector is empty."));
            return false;
        } else {
            for (auto &v : val) {
                if (v.IsEmpty()) {
                    wxLogError(_("There are NaN values in the provided predictor criteria vector."));
                    return false;
                }
            }
        }
        m_stepsVect[iStep].predictors[iPtor].criteria = val;
        return true;
    }

protected:
    double m_calibrationStart;
    double m_calibrationEnd;
    vi m_validationYears;
    VectorParamsStepVect m_stepsVect;

private:
    ParamsScore m_score;
};

#endif // ASPARAMETERSSCORING_H
