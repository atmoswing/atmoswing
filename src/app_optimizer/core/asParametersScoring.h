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

    wxString GetPredictandStationIdsVectorString(const vvi& predictandStationIdsVect) const;

    wxString Print() const override;

    virtual int GetPreprocessDataIdVectorSize(int iStep, int iPtor, int iPre) const {
        return 1;
    }

    bool GetValuesFromString(wxString stringVals) override;  // We copy the string as we'll modify it.

    void SetCalibrationYearStart(int val) {
        _calibrationStart = asTime::GetMJD(val, 1, 1);
    }

    void SetCalibrationYearEnd(int val) {
        _calibrationEnd = asTime::GetMJD(val, 12, 31);
    }

    double GetCalibrationStart() const {
        return _calibrationStart;
    }

    void SetCalibrationStart(const wxString& val) {
        _calibrationStart = asTime::GetTimeFromString(val);
    }

    double GetCalibrationEnd() const {
        return _calibrationEnd;
    }

    void SetCalibrationEnd(const wxString& val) {
        _calibrationEnd = asTime::GetTimeFromString(val);
    }

    vi GetValidationYearsVector() const {
        return _validationYears;
    }

    void SetValidationYearsVector(const vi& val) {
        wxASSERT(!val.empty());
        _validationYears = val;
    }

    bool HasValidationPeriod() const {
        return !_validationYears.empty();
    }

    ParamsScore GetScore() const {
        return _score;
    }

    wxString GetScoreName() const {
        return _score.name;
    }

    void SetScoreName(const wxString& val) {
        wxASSERT(!val.IsEmpty());
        _score.name = val;
    }

    float GetScoreThreshold() const {
        return _score.threshold;
    }

    void SetScoreThreshold(float val) {
        _score.threshold = val;
    }

    bool GetOnMean() const {
        return _score.onMean;
    }

    void SetOnMean(bool val) {
        _score.onMean = val;
    }

    float GetScoreQuantile() const {
        return _score.quantile;
    }

    void SetScoreQuantile(float val) {
        _score.quantile = val;
    }

    int GetScoreAnalogsNumber() const {
        return GetAnalogsNumber(GetStepsNb() - 1);
    }

    wxString GetScoreTimeArrayMode() const {
        return _score.timeArrayMode;
    }

    void SetScoreTimeArrayMode(const wxString& val) {
        wxASSERT(!val.IsEmpty());
        _score.timeArrayMode = val;
    }

    bool ScoreNeedsPostprocessing() const {
        return _score.postprocess;
    }

    /* Vector elements */

    vi GetAnalogsNumberVector(int iStep) const {
        return _stepsVect[iStep].analogsNumber;
    }

    void SetAnalogsNumberVector(int iStep, const vi& val) {
        wxASSERT(!val.empty());
        _stepsVect[iStep].analogsNumber = val;
    }

    void SetPreprocessHourVector(int iStep, int iPtor, int iPre, const vd& val) {
        wxASSERT(!val.empty());
        if (_stepsVect[iStep].predictors[iPtor].preprocessHours.size() >= iPre + 1) {
            _stepsVect[iStep].predictors[iPtor].preprocessHours[iPre].clear();
            _stepsVect[iStep].predictors[iPtor].preprocessHours[iPre] = val;
        } else {
            _stepsVect[iStep].predictors[iPtor].preprocessHours.push_back(val);
        }
    }

    vd GetPredictorXminVector(int iStep, int iPtor) const {
        return _stepsVect[iStep].predictors[iPtor].xMin;
    }

    void SetPredictorXminVector(int iStep, int iPtor, const vd& val) {
        wxASSERT(!val.empty());
        _stepsVect[iStep].predictors[iPtor].xMin = val;
    }

    vi GetPredictorXptsnbVector(int iStep, int iPtor) const {
        return _stepsVect[iStep].predictors[iPtor].xPtsNb;
    }

    void SetPredictorXptsnbVector(int iStep, int iPtor, const vi& val) {
        wxASSERT(!val.empty());
        _stepsVect[iStep].predictors[iPtor].xPtsNb = val;
    }

    vd GetPredictorYminVector(int iStep, int iPtor) const {
        return _stepsVect[iStep].predictors[iPtor].yMin;
    }

    void SetPredictorYminVector(int iStep, int iPtor, const vd& val) {
        wxASSERT(!val.empty());
        _stepsVect[iStep].predictors[iPtor].yMin = val;
    }

    vi GetPredictorYptsnbVector(int iStep, int iPtor) const {
        return _stepsVect[iStep].predictors[iPtor].yPtsNb;
    }

    void SetPredictorYptsnbVector(int iStep, int iPtor, const vi& val) {
        wxASSERT(!val.empty());
        _stepsVect[iStep].predictors[iPtor].yPtsNb = val;
    }

    vd GetPredictorHourVector(int iStep, int iPtor) const {
        return _stepsVect[iStep].predictors[iPtor].hours;
    }

    void SetPredictorHoursVector(int iStep, int iPtor, const vd& val) {
        wxASSERT(!val.empty());
        _stepsVect[iStep].predictors[iPtor].hours = val;
    }

    vf GetPredictorWeightVector(int iStep, int iPtor) const {
        return _stepsVect[iStep].predictors[iPtor].weight;
    }

    void SetPredictorWeightVector(int iStep, int iPtor, const vf& val) {
        wxASSERT(!val.empty());
        _stepsVect[iStep].predictors[iPtor].weight = val;
    }

    vwxs GetPreprocessDataIdVector(int iStep, int iPtor, int iPre) const {
        if (_stepsVect[iStep].predictors[iPtor].preprocessDataId.size() >= iPre + 1) {
            return _stepsVect[iStep].predictors[iPtor].preprocessDataId[iPre];
        } else {
            wxLogError(_("Trying to access to an element outside of preprocessDataId in the parameters object."));
            vwxs empty;
            return empty;
        }
    }

    void SetPreprocessDataIdVector(int iStep, int iPtor, int iPre, const vwxs& val) {
        wxASSERT(!val.empty());
        if (_stepsVect[iStep].predictors[iPtor].preprocessDataId.size() >= iPre + 1) {
            _stepsVect[iStep].predictors[iPtor].preprocessDataId[iPre].clear();
            _stepsVect[iStep].predictors[iPtor].preprocessDataId[iPre] = val;
        } else {
            _stepsVect[iStep].predictors[iPtor].preprocessDataId.push_back(val);
        }
    }

    vf GetPreprocessLevelVector(int iStep, int iPtor, int iPre) const {
        if (_stepsVect[iStep].predictors[iPtor].preprocessLevels.size() >= iPre + 1) {
            return _stepsVect[iStep].predictors[iPtor].preprocessLevels[iPre];
        } else {
            wxLogError(_("Trying to access to an element outside of preprocessLevels in the parameters object."));
            vf empty;
            return empty;
        }
    }

    void SetPreprocessLevelVector(int iStep, int iPtor, int iPre, const vf& val) {
        wxASSERT(!val.empty());
        if (_stepsVect[iStep].predictors[iPtor].preprocessLevels.size() >= iPre + 1) {
            _stepsVect[iStep].predictors[iPtor].preprocessLevels[iPre].clear();
            _stepsVect[iStep].predictors[iPtor].preprocessLevels[iPre] = val;
        } else {
            _stepsVect[iStep].predictors[iPtor].preprocessLevels.push_back(val);
        }
    }

    vd GetPreprocessHourVector(int iStep, int iPtor, int iPre) const {
        wxASSERT(_stepsVect[iStep].predictors[iPtor].preprocessHours.size() > iPre);

        if (_stepsVect[iStep].predictors[iPtor].preprocessHours.size() >= iPre + 1) {
            return _stepsVect[iStep].predictors[iPtor].preprocessHours[iPre];
        } else {
            wxLogError(_("Trying to access to an element outside of preprocessHours (vect) in the parameters object."));
            vd empty;
            return empty;
        }
    }

    vwxs GetPredictorDataIdVector(int iStep, int iPtor) const {
        return _stepsVect[iStep].predictors[iPtor].dataId;
    }

    int GetPredictorDataIdNb(int iStep, int iPtor) const override {
        return (int)_stepsVect[iStep].predictors[iPtor].dataId.size();
    }

    void SetPredictorDataIdVector(int iStep, int iPtor, const vwxs& val) {
        wxASSERT(!val.empty());
        _stepsVect[iStep].predictors[iPtor].dataId = val;
    }

    vf GetPredictorLevelVector(int iStep, int iPtor) const {
        return _stepsVect[iStep].predictors[iPtor].level;
    }

    void SetPredictorLevelVector(int iStep, int iPtor, const vf& val) {
        wxASSERT(!val.empty());
        _stepsVect[iStep].predictors[iPtor].level = val;
    }

    vwxs GetPredictorCriteriaVector(int iStep, int iPtor) const {
        return _stepsVect[iStep].predictors[iPtor].criteria;
    }

    void SetPredictorCriteriaVector(int iStep, int iPtor, const vwxs& val) {
        wxASSERT(!val.empty());
        _stepsVect[iStep].predictors[iPtor].criteria = val;
    }

  protected:
    double _calibrationStart;
    double _calibrationEnd;
    vi _validationYears;
    VectorParamsStepVect _stepsVect;

  private:
    ParamsScore _score;
};

#endif
