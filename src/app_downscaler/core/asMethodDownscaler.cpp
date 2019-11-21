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

#include "asMethodDownscaler.h"

#include <asAreaCompGrid.h>
#include <asCriteria.h>
#include <asIncludes.h>
#include <asParametersDownscaling.h>
#include <asPreprocessor.h>
#include <asProcessor.h>
#include <asResultsDates.h>
#include <asResultsValues.h>
#include <asTimeArray.h>

#include "asPredictorProj.h"

#ifndef UNIT_TESTING

#endif

asMethodDownscaler::asMethodDownscaler() : asMethodStandard() {
  // Seeds the random generator
  asInitRandom();
}

asMethodDownscaler::~asMethodDownscaler() {
  DeletePreloadedArchiveData();
}

bool asMethodDownscaler::Manager() {
  // Set unresponsive to speedup
  g_responsive = false;

  // Seeds the random generator
  asInitRandom();

  // Load parameters
  asParametersDownscaling params;
  if (!params.LoadFromFile(m_paramsFilePath)) {
    return false;
  }
  if (!m_predictandStationIds.empty()) {
    vvi idsVect;
    idsVect.push_back(m_predictandStationIds);
    params.SetPredictandStationIdsVector(idsVect);
  }
  params.InitValues();

  // Load the Predictand DB
  if (!LoadPredictandDB(m_predictandDBFilePath)) {
    return false;
  }

  // Watch
  wxStopWatch sw;

  // Downscale
  if (Downscale(params)) {
    // Display processing time
    wxLogMessage(_("The whole processing took %.3f min to execute"), float(sw.Time()) / 60000.0f);
#if wxUSE_GUI
    wxLogStatus(_("Downscaling over."));
#endif
  } else {
    wxLogError(_("The series could not be downscaled"));
  }

  // Delete preloaded data and cleanup
  DeletePreloadedArchiveData();

  return true;
}

void asMethodDownscaler::ClearAll() {
  m_parameters.clear();
}

double asMethodDownscaler::GetTimeStartDownscaling(asParametersDownscaling *params) const {
  return params->GetDownscalingStart() + params->GetTimeShiftDays();
}

double asMethodDownscaler::GetTimeEndDownscaling(asParametersDownscaling *params) const {
  return params->GetDownscalingEnd() - params->GetTimeSpanDays();
}

double asMethodDownscaler::GetEffectiveArchiveDataStart(asParameters *params) const {
  return GetTimeStartArchive(params);
}

double asMethodDownscaler::GetEffectiveArchiveDataEnd(asParameters *params) const {
  return GetTimeEndArchive(params);
}

bool asMethodDownscaler::GetAnalogsDates(asResultsDates &results, asParametersDownscaling *params, int iStep,
                                         bool &containsNaNs) {
  // Initialize the result object
  results.SetCurrentStep(iStep);
  results.Init(params);

  // Archive date array
  asTimeArray timeArrayArchive(GetTimeStartArchive(params), GetTimeEndArchive(params),
                               params->GetAnalogsTimeStepHours(), asTimeArray::Simple);
  if (!timeArrayArchive.Init()) {
    wxLogError(_("The time array mode for the archive dates is not correctly defined."));
    return false;
  }

  // Check on the archive length
  if (timeArrayArchive.GetSize() < 100) {
    wxLogError(_("The time array is not consistent in asMethodDownscaler::GetAnalogsDates: size=%d."),
               timeArrayArchive.GetSize());
    return false;
  }

  // Target date array
  asTimeArray timeArrayTarget(GetTimeStartDownscaling(params), GetTimeEndDownscaling(params),
                              params->GetTargetTimeStepHours(), params->GetTimeArrayTargetMode());
  if (!timeArrayTarget.Init()) {
    wxLogError(_("The time array mode for the target dates is not correctly defined."));
    return false;
  }

  // Load the archive data
  std::vector<asPredictor *> predictorsArch;
  if (!LoadArchiveData(predictorsArch, params, iStep, GetTimeStartArchive(params), GetTimeEndArchive(params))) {
    wxLogError(_("Failed loading predictor data."));
    Cleanup(predictorsArch);
    return false;
  }

  // Load the scenario data
  std::vector<asPredictor *> predictorsProj;
  if (!LoadProjectionData(predictorsProj, params, iStep, GetTimeStartDownscaling(params),
                          GetTimeEndDownscaling(params))) {
    wxLogError(_("Failed loading predictor data."));
    Cleanup(predictorsArch);
    Cleanup(predictorsProj);
    return false;
  }

  // Create the criterion
  std::vector<asCriteria *> criteria;
  for (int iPtor = 0; iPtor < params->GetPredictorsNb(iStep); iPtor++) {
    // Instantiate a score object
    asCriteria *criterion = asCriteria::GetInstance(params->GetPredictorCriteria(iStep, iPtor));
    criteria.push_back(criterion);
  }

  // Check time sizes
#ifdef _DEBUG
  int prevTimeSize = 0;

  for (int i = 0; i < predictorsProj.size(); i++) {
    if (i > 0) {
      wxASSERT(predictorsProj[i]->GetTimeSize() == prevTimeSize);
    }
    prevTimeSize = predictorsProj[i]->GetTimeSize();
  }
  for (int i = 0; i < predictorsArch.size(); i++) {
    if (i > 0) {
      wxASSERT(predictorsArch[i]->GetTimeSize() == prevTimeSize);
    }
    prevTimeSize = predictorsArch[i]->GetTimeSize();
  }
#endif  // _DEBUG

  // Inline the data when possible
  for (int iPtor = 0; iPtor < predictorsArch.size(); iPtor++) {
    if (criteria[iPtor]->CanUseInline()) {
      predictorsArch[iPtor]->Inline();
    }
  }

  // Send data and criteria to processor
  wxLogVerbose(_("Start processing the comparison."));

  if (!asProcessor::GetAnalogsDates(predictorsArch, predictorsProj, timeArrayArchive, timeArrayArchive, timeArrayTarget,
                                    timeArrayTarget, criteria, params, iStep, results, containsNaNs)) {
    wxLogError(_("Failed processing the analogs dates."));
    Cleanup(predictorsArch);
    Cleanup(predictorsProj);
    Cleanup(criteria);
    return false;
  }
  wxLogVerbose(_("The processing is over."));

  Cleanup(predictorsArch);
  Cleanup(predictorsProj);
  Cleanup(criteria);

  return true;
}

bool asMethodDownscaler::GetAnalogsSubDates(asResultsDates &results, asParametersDownscaling *params,
                                            asResultsDates &anaDates, int iStep, bool &containsNaNs) {
  // Initialize the result object
  results.SetCurrentStep(iStep);
  results.Init(params);

  // Date array object instantiation for the processor
  wxLogVerbose(_("Creating a date arrays for the processor."));
  double timeStart = params->GetArchiveStart();
  double timeEnd = params->GetArchiveEnd() - params->GetTimeSpanDays();
  asTimeArray timeArrayArchive(timeStart, timeEnd, params->GetAnalogsTimeStepHours(), asTimeArray::Simple);
  timeArrayArchive.Init();
  wxLogVerbose(_("Date arrays created."));

  // Load the predictor data
  std::vector<asPredictor *> predictors;
  if (!LoadArchiveData(predictors, params, iStep, timeStart, timeEnd)) {
    wxLogError(_("Failed loading predictor data."));
    Cleanup(predictors);
    return false;
  }

  // Create the score objects
  std::vector<asCriteria *> criteria;
  for (int iPtor = 0; iPtor < params->GetPredictorsNb(iStep); iPtor++) {
    wxLogVerbose(_("Creating a criterion object."));
    asCriteria *criterion = asCriteria::GetInstance(params->GetPredictorCriteria(iStep, iPtor));
    criteria.push_back(criterion);
    wxLogVerbose(_("Criterion object created."));
  }

  // Inline the data when possible
  for (int iPtor = 0; iPtor < predictors.size(); iPtor++) {
    if (criteria[iPtor]->CanUseInline()) {
      predictors[iPtor]->Inline();
    }
  }

  // Send data and criteria to processor
  wxLogVerbose(_("Start processing the comparison."));
  if (!asProcessor::GetAnalogsSubDates(predictors, predictors, timeArrayArchive, timeArrayArchive, anaDates, criteria,
                                       params, iStep, results, containsNaNs)) {
    wxLogError(_("Failed processing the analogs dates."));
    Cleanup(predictors);
    Cleanup(criteria);
    return false;
  }
  wxLogVerbose(_("The processing is over."));

  Cleanup(predictors);
  Cleanup(criteria);

  return true;
}

bool asMethodDownscaler::GetAnalogsValues(asResultsValues &results, asParametersDownscaling *params,
                                          asResultsDates &anaDates, int iStep) {
  // Initialize the result object
  results.SetCurrentStep(iStep);
  results.Init(params);

  // Set the predictand values to the corresponding analog dates
  wxASSERT(m_predictandDB);
  wxLogVerbose(_("Start setting the predictand values to the corresponding analog dates."));
  if (!asProcessor::GetAnalogsValues(*m_predictandDB, anaDates, params, results)) {
    wxLogError(_("Failed setting the predictand values to the corresponding analog dates."));
    return false;
  }
  wxLogVerbose(_("Predictand association over."));

  return true;
}

bool asMethodDownscaler::SaveDetails(asParametersDownscaling *params) {
  asResultsDates anaDatesPrevious;
  asResultsDates anaDates;
  asResultsValues anaValues;

  // Process every step one after the other
  int stepsNb = params->GetStepsNb();
  for (int iStep = 0; iStep < stepsNb; iStep++) {
    bool containsNaNs = false;
    if (iStep == 0) {
      if (!GetAnalogsDates(anaDates, params, iStep, containsNaNs)) return false;
    } else {
      anaDatesPrevious = anaDates;
      if (!GetAnalogsSubDates(anaDates, params, anaDatesPrevious, iStep, containsNaNs)) return false;
    }
    if (containsNaNs) {
      wxLogError(_("The dates selection contains NaNs"));
      return false;
    }
  }
  if (!GetAnalogsValues(anaValues, params, anaDates, stepsNb - 1)) return false;

  anaDates.SetSubFolder("downscaling");
  anaDates.Save();
  anaValues.SetSubFolder("downscaling");
  anaValues.Save();

  return true;
}

bool asMethodDownscaler::LoadProjectionData(std::vector<asPredictor *> &predictors, asParametersDownscaling *params,
                                            int iStep, double timeStartData, double timeEndData) {
  try {
    // Loop through every predictor
    for (int iPtor = 0; iPtor < params->GetPredictorsNb(iStep); iPtor++) {
      wxLogVerbose(_("Loading model scenario."));

      if (!params->NeedsPreprocessing(iStep, iPtor)) {
        if (!ExtractProjectionDataWithoutPreprocessing(predictors, params, iStep, iPtor, timeStartData, timeEndData)) {
          return false;
        }
      } else {
        if (!ExtractProjectionDataWithPreprocessing(predictors, params, iStep, iPtor, timeStartData, timeEndData)) {
          return false;
        }
      }

      wxLogVerbose(_("Projection data loaded."));
    }
  } catch (std::bad_alloc &ba) {
    wxString msg(ba.what(), wxConvUTF8);
    wxLogError(_("Bad allocation during scenario data loading: %s"), msg);
    return false;
  } catch (std::exception &e) {
    wxString msg(e.what(), wxConvUTF8);
    wxLogError(_("Exception during scenario data loading: %s"), msg);
    return false;
  }

  return true;
}

bool asMethodDownscaler::ExtractProjectionDataWithoutPreprocessing(std::vector<asPredictor *> &predictors,
                                                                   asParametersDownscaling *params, int iStep,
                                                                   int iPtor, double timeStartData,
                                                                   double timeEndData) {
  // Date array object instantiation for the data loading.
  double ptorStart = timeStartData + params->GetPredictorTimeAsDays(iStep, iPtor);
  double ptorEnd = timeEndData + params->GetPredictorTimeAsDays(iStep, iPtor);
  asTimeArray timeArray(ptorStart, ptorEnd, params->GetAnalogsTimeStepHours(), params->GetTimeArrayAnalogsMode());
  timeArray.Init();

  // Loading the datasets information
  asPredictorProj *predictor = asPredictorProj::GetInstance(
      params->GetPredictorProjDatasetId(iStep, iPtor), params->GetModel(), params->GetScenario(),
      params->GetPredictorProjDataId(iStep, iPtor), m_predictorProjectionDataDir);
  if (!predictor) {
    return false;
  }

  // Set standardize option
  if (params->GetStandardize(iStep, iPtor)) {
    predictor->SetStandardize(true);
  }

  // Select the number of members for ensemble data.
  if (predictor->IsEnsemble()) {
    predictor->SelectMembers(params->GetPredictorProjMembersNb(iStep, iPtor));
  }

  // Area object instantiation
  asAreaCompGrid *area = asAreaCompGrid::GetInstance(params, iStep, iPtor);
  wxASSERT(area);

  // Data loading
  if (!predictor->Load(area, timeArray, params->GetPredictorLevel(iStep, iPtor))) {
    wxLogError(_("The data could not be loaded."));
    wxDELETE(area);
    wxDELETE(predictor);
    return false;
  }
  wxDELETE(area);
  predictors.push_back(predictor);

  return true;
}

bool asMethodDownscaler::ExtractProjectionDataWithPreprocessing(std::vector<asPredictor *> &predictors,
                                                                asParametersDownscaling *params, int iStep, int iPtor,
                                                                double timeStartData, double timeEndData) {
  std::vector<asPredictorProj *> predictorsPreprocess;

  int preprocessSize = params->GetPreprocessSize(iStep, iPtor);

  wxLogVerbose(_("Preprocessing data (%d predictor(s)) while loading."), preprocessSize);

  for (int iPre = 0; iPre < preprocessSize; iPre++) {
    // Date array object instantiation for the data loading.
    double ptorStart = timeStartData + params->GetPreprocessTimeAsDays(iStep, iPtor, iPre);
    double ptorEnd = timeEndData + params->GetPreprocessTimeAsDays(iStep, iPtor, iPre);
    asTimeArray timeArray(ptorStart, ptorEnd, params->GetAnalogsTimeStepHours(), params->GetTimeArrayAnalogsMode());
    timeArray.Init();

    // Loading the dataset information
    asPredictorProj *predictorPreprocess = asPredictorProj::GetInstance(
        params->GetPreprocessProjDatasetId(iStep, iPtor, iPre), params->GetModel(), params->GetScenario(),
        params->GetPreprocessProjDataId(iStep, iPtor, iPre), m_predictorProjectionDataDir);
    if (!predictorPreprocess) {
      Cleanup(predictorsPreprocess);
      return false;
    }

    // Select the number of members for ensemble data.
    if (predictorPreprocess->IsEnsemble()) {
      predictorPreprocess->SelectMembers(params->GetPreprocessMembersNb(iStep, iPtor, iPre));
    }

    // Area object instantiation
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(params, iStep, iPtor);
    wxASSERT(area);

    // Data loading
    if (!predictorPreprocess->Load(area, timeArray, params->GetPredictorLevel(iStep, iPtor))) {
      wxLogError(_("The data could not be loaded."));
      wxDELETE(area);
      wxDELETE(predictorPreprocess);
      Cleanup(predictorsPreprocess);
      return false;
    }
    wxDELETE(area);
    predictorsPreprocess.push_back(predictorPreprocess);
  }

  // Fix the criteria if S1
  params->FixCriteriaIfGradientsPreprocessed(iStep, iPtor);

  auto *predictor = new asPredictorProj(*predictorsPreprocess[0]);
  if (!Preprocess(predictorsPreprocess, params->GetPreprocessMethod(iStep, iPtor), predictor)) {
    wxLogError(_("Data preprocessing failed."));
    Cleanup(predictorsPreprocess);
    wxDELETE(predictor);
    return false;
  }

  // Standardize
  if (params->GetStandardize(iStep, iPtor)) {
    predictor->StandardizeData();
  }

  Cleanup(predictorsPreprocess);
  predictors.push_back(predictor);

  return true;
}

bool asMethodDownscaler::Preprocess(std::vector<asPredictorProj *> predictors, const wxString &method,
                                    asPredictor *result) {
  std::vector<asPredictor *> ptorsPredictors(predictors.begin(), predictors.end());

  return asPreprocessor::Preprocess(ptorsPredictors, method, result);
}

void asMethodDownscaler::Cleanup(std::vector<asPredictorProj *> predictors) {
  if (!predictors.empty()) {
    for (auto &predictor : predictors) {
      wxDELETE(predictor);
    }
    predictors.resize(0);
  }
}

void asMethodDownscaler::Cleanup(std::vector<asPredictor *> predictors) {
  if (!predictors.empty()) {
    for (auto &predictor : predictors) {
      wxDELETE(predictor);
    }
    predictors.resize(0);
  }
}

void asMethodDownscaler::Cleanup(std::vector<asCriteria *> criteria) {
  if (!criteria.empty()) {
    for (auto &i : criteria) {
      wxDELETE(i);
    }
    criteria.resize(0);
  }
}