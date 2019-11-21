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

#include <asParameters.h>

#include "asIncludes.h"

class asFileParametersDownscaling;

class asParametersDownscaling : public asParameters {
 public:
  typedef struct {
    wxString datasetId;
    wxString dataId;
    int membersNb;
    vwxs preprocessDatasetIds;
    vwxs preprocessDataIds;
    int preprocessMembersNb;
  } ParamsPredictorProj;

  typedef std::vector<ParamsPredictorProj> VectorParamsPredictorsProj;

  typedef struct {
    VectorParamsPredictorsProj predictors;
  } ParamsStepProj;

  typedef std::vector<ParamsStepProj> VectorParamsStepProj;

  asParametersDownscaling();

  virtual ~asParametersDownscaling();

  void AddStep();

  void AddPredictorProj(ParamsStepProj &step);

  bool LoadFromFile(const wxString &filePath);

  bool InputsOK() const;

  bool FixTimeLimits();

  void InitValues();

  void SetModel(const wxString &model) {
    m_model = model;
  }

  wxString GetModel() const {
    return m_model;
  }

  void SetScenario(const wxString &scenario) {
    m_scenario = scenario;
  }

  wxString GetScenario() const {
    return m_scenario;
  }

  vvi GetPredictandStationIdsVector() const {
    return m_predictandStationIdsVect;
  }

  bool SetPredictandStationIdsVector(vvi val);

  wxString GetPredictorProjDatasetId(int iStep, int iPtor) const {
    return m_stepsProj[iStep].predictors[iPtor].datasetId;
  }

  bool SetPredictorProjDatasetId(int iStep, int iPtor, const wxString &val);

  wxString GetPredictorProjDataId(int iStep, int iPtor) const {
    return m_stepsProj[iStep].predictors[iPtor].dataId;
  }

  bool SetPredictorProjDataId(int iStep, int iPtor, const wxString &val);

  wxString GetPreprocessProjDatasetId(int iStep, int iPtor, int iPre) const;

  bool SetPreprocessProjDatasetId(int iStep, int iPtor, int iPre, const wxString &val);

  wxString GetPreprocessProjDataId(int iStep, int iPtor, int iPre) const;

  bool SetPreprocessProjDataId(int iStep, int iPtor, int iPre, const wxString &val);

  void SetPredictorProjMembersNb(int iStep, int iPtor, int val) {
    m_stepsProj[iStep].predictors[iPtor].membersNb = val;
  }

  int GetPredictorProjMembersNb(int iStep, int iPtor) const {
    return m_stepsProj[iStep].predictors[iPtor].membersNb;
  }

  void SetPreprocessProjMembersNb(int iStep, int iPtor, int iPre, int val) {
    m_stepsProj[iStep].predictors[iPtor].preprocessMembersNb = val;
  }

  int GetPreprocessProjMembersNb(int iStep, int iPtor, int iPre) const {
    return m_stepsProj[iStep].predictors[iPtor].preprocessMembersNb;
  }

  bool SetDownscalingYearStart(int val) {
    m_downscalingStart = asTime::GetMJD(val, 1, 1);
    return true;
  }

  bool SetDownscalingYearEnd(int val) {
    m_downscalingEnd = asTime::GetMJD(val, 12, 31);
    return true;
  }

  double GetDownscalingStart() const {
    return m_downscalingStart;
  }

  bool SetDownscalingStart(wxString val) {
    m_downscalingStart = asTime::GetTimeFromString(val);
    return true;
  }

  double GetDownscalingEnd() const {
    return m_downscalingEnd;
  }

  bool SetDownscalingEnd(wxString val) {
    m_downscalingEnd = asTime::GetTimeFromString(val);
    return true;
  }

 protected:
  double m_downscalingStart;
  double m_downscalingEnd;

 private:
  wxString m_model;
  wxString m_scenario;
  vvi m_predictandStationIdsVect;
  VectorParamsStepProj m_stepsProj;

  bool ParseDescription(asFileParametersDownscaling &fileParams, const wxXmlNode *nodeProcess);

  bool ParseTimeProperties(asFileParametersDownscaling &fileParams, const wxXmlNode *nodeProcess);

  bool ParseAnalogDatesParams(asFileParametersDownscaling &fileParams, int iStep, const wxXmlNode *nodeProcess);

  bool ParsePreprocessedPredictors(asFileParametersDownscaling &fileParams, int iStep, int iPtor,
                                   const wxXmlNode *nodeParam);

  bool ParseAnalogValuesParams(asFileParametersDownscaling &fileParams, const wxXmlNode *nodeProcess);
};

#endif
