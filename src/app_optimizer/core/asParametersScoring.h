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

#ifndef AS_PARAMETERS_SCORING_H
#define AS_PARAMETERS_SCORING_H

#include "asIncludes.h"
#include "asParameters.h"

class asFileParameters;

class asParametersScoring : public asParameters {
  public:
    typedef struct ParamsScore {
        wxString name;
        wxString timeArrayMode;
        double timeArrayDate = 0;
        int timeArrayIntervalDays = 0;
        bool postprocess = false;
        float postprocessDupliExp = 0;
        wxString postprocessMethod;
        bool onMean = false;
        float threshold = NAN;
        float quantile = NAN;
    } ParamsScore;

    /** Vectors */
    typedef struct ParamsPredictorVect {
        vvwxs preprocessDataId;
        vvf preprocessLevels;
        vvd preprocessHours;
        vwxs dataId;
        vf level;
        vd xMin;
        vi xPtsNb;
        vd yMin;
        vi yPtsNb;
        vd hours;
        vwxs criteria;
        vf weight;
    } ParamsPredictorVect;

    typedef vector<ParamsPredictorVect> VectorParamsPredictorsVect;

    typedef struct ParamsStepVect {
        vi analogsNumber;
        VectorParamsPredictorsVect predictors;
    } ParamsStepVect;

    typedef vector<ParamsStepVect> VectorParamsStepVect;

    typedef struct ParamsScoreVect {
        vwxs name;
        vwxs timeArrayMode;
        vd timeArrayDate;
        vi timeArrayIntervalDays;
        vf postprocessDupliExp;
    } ParamsScoreVect;

    /** Booleans */
    typedef struct ParamsPredictorBool {
        vb preprocessDataId;
        vb preprocessLevels;
        vb preprocessHours;
        bool dataId = true;
        bool level = true;
        bool xMin = true;
        bool xPtsNb = true;
        bool yMin = true;
        bool yPtsNb = true;
        bool hours = true;
        bool weight = true;
        bool criteria = true;
    } ParamsPredictorBool;

    typedef vector<ParamsPredictorBool> VectorParamsPredictorsBool;

    typedef struct ParamsStepBool {
        bool analogsNumber = false;
        VectorParamsPredictorsBool predictors;
    } ParamsStepBool;

    typedef vector<ParamsStepBool> VectorParamsStepBool;

    asParametersScoring();

    ~asParametersScoring() override;

    void AddPredictorVect(ParamsStepVect& step);

    bool GenerateSimpleParametersFile(const wxString& filePath) const;

    bool PreprocessingDataIdsOk();

    wxString GetPredictandStationIdsVectorString(vvi& predictandStationIdsVect) const;

    wxString Print() const override;

    virtual int GetPreprocessDataIdVectorSize(int iStep, int iPtor, int iPre) const {
        return 1;
    }

    bool GetValuesFromString(wxString stringVals) override;  // We copy the string as we'll modify it.

    void SetCalibrationYearStart(int val) {
        m_calibrationStart = asTime::GetMJD(val, 1, 1);
    }

    void SetCalibrationYearEnd(int val) {
        m_calibrationEnd = asTime::GetMJD(val, 12, 31);
    }

    double GetCalibrationStart() const {
        return m_calibrationStart;
    }

    void SetCalibrationStart(const wxString& val) {
        m_calibrationStart = asTime::GetTimeFromString(val);
    }

    double GetCalibrationEnd() const {
        return m_calibrationEnd;
    }

    void SetCalibrationEnd(const wxString& val) {
        m_calibrationEnd = asTime::GetTimeFromString(val);
    }

    vi GetValidationYearsVector() const {
        return m_validationYears;
    }

    void SetValidationYearsVector(vi val) {
        wxASSERT(!val.empty());
        m_validationYears = val;
    }

    bool HasValidationPeriod() const {
        return !m_validationYears.empty();
    }

    ParamsScore GetScore() const {
        return m_score;
    }

    wxString GetScoreName() const {
        return m_score.name;
    }

    void SetScoreName(const wxString& val) {
        wxASSERT(!val.IsEmpty());
        m_score.name = val;
    }

    float GetScoreThreshold() const {
        return m_score.threshold;
    }

    void SetScoreThreshold(float val) {
        m_score.threshold = val;
    }

    bool GetOnMean() const {
        return m_score.onMean;
    }

    void SetOnMean(bool val) {
        m_score.onMean = val;
    }

    float GetScoreQuantile() const {
        return m_score.quantile;
    }

    void SetScoreQuantile(float val) {
        m_score.quantile = val;
    }

    int GetScoreAnalogsNumber() const {
        return GetAnalogsNumber(GetStepsNb() - 1);
    }

    wxString GetScoreTimeArrayMode() const {
        return m_score.timeArrayMode;
    }

    void SetScoreTimeArrayMode(const wxString& val) {
        wxASSERT(!val.IsEmpty());
        m_score.timeArrayMode = val;
    }

    bool ScoreNeedsPostprocessing() const {
        return m_score.postprocess;
    }

    /* Vector elements */

    vi GetAnalogsNumberVector(int iStep) const {
        return m_stepsVect[iStep].analogsNumber;
    }

    void SetAnalogsNumberVector(int iStep, vi val) {
        wxASSERT(!val.empty());
        m_stepsVect[iStep].analogsNumber = val;
    }

    void SetPreprocessHourVector(int iStep, int iPtor, int iPre, vd val) {
        wxASSERT(!val.empty());
        if (m_stepsVect[iStep].predictors[iPtor].preprocessHours.size() >= iPre + 1) {
            m_stepsVect[iStep].predictors[iPtor].preprocessHours[iPre].clear();
            m_stepsVect[iStep].predictors[iPtor].preprocessHours[iPre] = val;
        } else {
            m_stepsVect[iStep].predictors[iPtor].preprocessHours.push_back(val);
        }
    }

    vd GetPredictorXminVector(int iStep, int iPtor) const {
        return m_stepsVect[iStep].predictors[iPtor].xMin;
    }

    void SetPredictorXminVector(int iStep, int iPtor, vd val) {
        wxASSERT(!val.empty());
        m_stepsVect[iStep].predictors[iPtor].xMin = val;
    }

    vi GetPredictorXptsnbVector(int iStep, int iPtor) const {
        return m_stepsVect[iStep].predictors[iPtor].xPtsNb;
    }

    void SetPredictorXptsnbVector(int iStep, int iPtor, vi val) {
        wxASSERT(!val.empty());
        m_stepsVect[iStep].predictors[iPtor].xPtsNb = val;
    }

    vd GetPredictorYminVector(int iStep, int iPtor) const {
        return m_stepsVect[iStep].predictors[iPtor].yMin;
    }

    void SetPredictorYminVector(int iStep, int iPtor, vd val) {
        wxASSERT(!val.empty());
        m_stepsVect[iStep].predictors[iPtor].yMin = val;
    }

    vi GetPredictorYptsnbVector(int iStep, int iPtor) const {
        return m_stepsVect[iStep].predictors[iPtor].yPtsNb;
    }

    void SetPredictorYptsnbVector(int iStep, int iPtor, vi val) {
        wxASSERT(!val.empty());
        m_stepsVect[iStep].predictors[iPtor].yPtsNb = val;
    }

    vd GetPredictorHourVector(int iStep, int iPtor) const {
        return m_stepsVect[iStep].predictors[iPtor].hours;
    }

    void SetPredictorHoursVector(int iStep, int iPtor, vd val) {
        wxASSERT(!val.empty());
        m_stepsVect[iStep].predictors[iPtor].hours = val;
    }

    vf GetPredictorWeightVector(int iStep, int iPtor) const {
        return m_stepsVect[iStep].predictors[iPtor].weight;
    }

    void SetPredictorWeightVector(int iStep, int iPtor, vf val) {
        wxASSERT(!val.empty());
        m_stepsVect[iStep].predictors[iPtor].weight = val;
    }

    vwxs GetPreprocessDataIdVector(int iStep, int iPtor, int iPre) const {
        if (m_stepsVect[iStep].predictors[iPtor].preprocessDataId.size() >= iPre + 1) {
            return m_stepsVect[iStep].predictors[iPtor].preprocessDataId[iPre];
        } else {
            wxLogError(_("Trying to access to an element outside of preprocessDataId in the parameters object."));
            vwxs empty;
            return empty;
        }
    }

    void SetPreprocessDataIdVector(int iStep, int iPtor, int iPre, vwxs val) {
        wxASSERT(!val.empty());
        if (m_stepsVect[iStep].predictors[iPtor].preprocessDataId.size() >= iPre + 1) {
            m_stepsVect[iStep].predictors[iPtor].preprocessDataId[iPre].clear();
            m_stepsVect[iStep].predictors[iPtor].preprocessDataId[iPre] = val;
        } else {
            m_stepsVect[iStep].predictors[iPtor].preprocessDataId.push_back(val);
        }
    }

    vf GetPreprocessLevelVector(int iStep, int iPtor, int iPre) const {
        if (m_stepsVect[iStep].predictors[iPtor].preprocessLevels.size() >= iPre + 1) {
            return m_stepsVect[iStep].predictors[iPtor].preprocessLevels[iPre];
        } else {
            wxLogError(_("Trying to access to an element outside of preprocessLevels in the parameters object."));
            vf empty;
            return empty;
        }
    }

    void SetPreprocessLevelVector(int iStep, int iPtor, int iPre, vf val) {
        wxASSERT(!val.empty());
        if (m_stepsVect[iStep].predictors[iPtor].preprocessLevels.size() >= iPre + 1) {
            m_stepsVect[iStep].predictors[iPtor].preprocessLevels[iPre].clear();
            m_stepsVect[iStep].predictors[iPtor].preprocessLevels[iPre] = val;
        } else {
            m_stepsVect[iStep].predictors[iPtor].preprocessLevels.push_back(val);
        }
    }

    vd GetPreprocessHourVector(int iStep, int iPtor, int iPre) const {
        wxASSERT(m_stepsVect[iStep].predictors[iPtor].preprocessHours.size() > iPre);

        if (m_stepsVect[iStep].predictors[iPtor].preprocessHours.size() >= iPre + 1) {
            return m_stepsVect[iStep].predictors[iPtor].preprocessHours[iPre];
        } else {
            wxLogError(_("Trying to access to an element outside of preprocessHours (vect) in the parameters object."));
            vd empty;
            return empty;
        }
    }

    vwxs GetPredictorDataIdVector(int iStep, int iPtor) const {
        return m_stepsVect[iStep].predictors[iPtor].dataId;
    }

    int GetPredictorDataIdNb(int iStep, int iPtor) const override {
        return (int)m_stepsVect[iStep].predictors[iPtor].dataId.size();
    }

    void SetPredictorDataIdVector(int iStep, int iPtor, vwxs val) {
        wxASSERT(!val.empty());
        m_stepsVect[iStep].predictors[iPtor].dataId = val;
    }

    vf GetPredictorLevelVector(int iStep, int iPtor) const {
        return m_stepsVect[iStep].predictors[iPtor].level;
    }

    void SetPredictorLevelVector(int iStep, int iPtor, vf val) {
        wxASSERT(!val.empty());
        m_stepsVect[iStep].predictors[iPtor].level = val;
    }

    vwxs GetPredictorCriteriaVector(int iStep, int iPtor) const {
        return m_stepsVect[iStep].predictors[iPtor].criteria;
    }

    void SetPredictorCriteriaVector(int iStep, int iPtor, vwxs val) {
        wxASSERT(!val.empty());
        m_stepsVect[iStep].predictors[iPtor].criteria = val;
    }

  protected:
    double m_calibrationStart;
    double m_calibrationEnd;
    vi m_validationYears;
    VectorParamsStepVect m_stepsVect;

  private:
    ParamsScore m_score;
};

#endif
