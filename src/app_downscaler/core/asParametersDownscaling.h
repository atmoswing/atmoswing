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
 * Portions Copyright 2017 Pascal Horton, University of Bern.
 */

#ifndef AS_PARAMETERS_DOWNSCALING_H
#define AS_PARAMETERS_DOWNSCALING_H

#include "asParameters.h"
#include "asTime.h"

class asFileParametersDownscaling;

class asParametersDownscaling : public asParameters {
  public:
    struct ParamsPredictorProj {
        wxString datasetId;
        wxString dataId;
        int membersNb = 0;
        vwxs preprocessDatasetIds;
        vwxs preprocessDataIds;
        int preprocessMembersNb = 0;
    };

    using VectorParamsPredictorsProj = vector<ParamsPredictorProj>;

    struct ParamsStepProj {
        VectorParamsPredictorsProj predictors;
    };

    using VectorParamsStepProj = vector<ParamsStepProj>;

    asParametersDownscaling();

    virtual ~asParametersDownscaling();

    void AddStep();

    void AddPredictorProj(ParamsStepProj& step);

    bool LoadFromFile(const wxString& filePath);

    bool InputsOK() const;

    bool FixTimeLimits();

    void InitValues();

    void SetModel(const wxString& model) {
        _model = model;
    }

    wxString GetModel() const {
        return _model;
    }

    void SetScenario(const wxString& scenario) {
        _scenario = scenario;
    }

    wxString GetScenario() const {
        return _scenario;
    }

    vvi GetPredictandStationIdsVector() const {
        return _predictandStationIdsVect;
    }

    bool SetPredictandStationIdsVector(vvi val);

    wxString GetPredictorProjDatasetId(int iStep, int iPtor) const {
        return _stepsProj[iStep].predictors[iPtor].datasetId;
    }

    void SetPredictorProjDatasetId(int iStep, int iPtor, const wxString& val);

    wxString GetPredictorProjDataId(int iStep, int iPtor) const {
        return _stepsProj[iStep].predictors[iPtor].dataId;
    }

    void SetPredictorProjDataId(int iStep, int iPtor, const wxString& val);

    wxString GetPreprocessProjDatasetId(int iStep, int iPtor, int iPre) const;

    void SetPreprocessProjDatasetId(int iStep, int iPtor, int iPre, const wxString& val);

    wxString GetPreprocessProjDataId(int iStep, int iPtor, int iPre) const;

    void SetPreprocessProjDataId(int iStep, int iPtor, int iPre, const wxString& val);

    void SetPredictorProjMembersNb(int iStep, int iPtor, int val) {
        _stepsProj[iStep].predictors[iPtor].membersNb = val;
    }

    int GetPredictorProjMembersNb(int iStep, int iPtor) const {
        return _stepsProj[iStep].predictors[iPtor].membersNb;
    }

    void SetPreprocessProjMembersNb(int iStep, int iPtor, int /*iPre*/, int val) {
        _stepsProj[iStep].predictors[iPtor].preprocessMembersNb = val;
    }

    int GetPreprocessProjMembersNb(int iStep, int iPtor, int /*iPre*/) const {
        return _stepsProj[iStep].predictors[iPtor].preprocessMembersNb;
    }

    void SetDownscalingYearStart(int val) {
        _downscalingStart = asTime::GetMJD(val, 1, 1);
    }

    void SetDownscalingYearEnd(int val) {
        _downscalingEnd = asTime::GetMJD(val, 12, 31);
    }

    double GetDownscalingStart() const {
        return _downscalingStart;
    }

    void SetDownscalingStart(wxString val) {
        _downscalingStart = asTime::GetTimeFromString(val);
    }

    double GetDownscalingEnd() const {
        return _downscalingEnd;
    }

    void SetDownscalingEnd(wxString val) {
        _downscalingEnd = asTime::GetTimeFromString(val);
    }

  protected:
    double _downscalingStart;
    double _downscalingEnd;

  private:
    wxString _model;
    wxString _scenario;
    vvi _predictandStationIdsVect;
    VectorParamsStepProj _stepsProj;

    bool ParseDescription(asFileParametersDownscaling& fileParams, const wxXmlNode* nodeProcess);

    bool ParseTimeProperties(asFileParametersDownscaling& fileParams, const wxXmlNode* nodeProcess);

    bool ParseAnalogDatesParams(asFileParametersDownscaling& fileParams, int iStep, const wxXmlNode* nodeProcess);

    bool ParsePreprocessedPredictors(asFileParametersDownscaling& fileParams, int iStep, int iPtor,
                                     const wxXmlNode* nodeParam);

    bool ParseAnalogValuesParams(asFileParametersDownscaling& fileParams, const wxXmlNode* nodeProcess);
};

#endif
