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
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#include "asMethodCalibratorEvaluateAllScores.h"

#include "asTotalScore.h"
#include "asTotalScoreRankHistogramReliability.h"

asMethodCalibratorEvaluateAllScores::asMethodCalibratorEvaluateAllScores() : asMethodCalibrator() {}

asMethodCalibratorEvaluateAllScores::~asMethodCalibratorEvaluateAllScores() {}

bool asMethodCalibratorEvaluateAllScores::Calibrate(asParametersCalibration &params) {
  // Check that we really handle a EvaluateAllScores case
  bool checkSizes = true;
  wxString errorField = wxEmptyString;
  if (params.GetTimeArrayAnalogsIntervalDaysVector().size() > 1) {
    checkSizes = false;
    errorField.Append("IntervalDays, ");
  }
  if (params.GetScoreNameVector().size() > 1) {
    checkSizes = false;
    errorField.Append("ScoreName, ");
  }
  if (params.GetScoreTimeArrayModeVector().size() > 1) {
    checkSizes = false;
    errorField.Append("ScoreTimeArrayMode, ");
  }
  if (params.GetScoreTimeArrayDateVector().size() > 1) {
    checkSizes = false;
    errorField.Append("ScoreTimeArrayDate ,");
  }
  if (params.GetScoreTimeArrayIntervalDaysVector().size() > 1) {
    checkSizes = false;
    errorField.Append("ScoreTimeArrayIntervalDays, ");
  }
  if (params.GetScorePostprocessDupliExpVector().size() > 1) {
    checkSizes = false;
    errorField.Append("ScorePostprocessDupliExp, ");
  }

  for (int iStep = 0; iStep < params.GetStepsNb(); iStep++) {
    if (params.GetAnalogsNumberVector(iStep).size() > 1) {
      checkSizes = false;
      errorField.Append(wxString::Format("analogsNumber (step %d), ", iStep));
    }
    for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
      if (params.NeedsPreprocessing(iStep, iPtor)) {
        for (int iPre = 0; iPre < params.GetPreprocessSize(iStep, iPtor); iPre++) {
          if (params.GetPreprocessDataIdVector(iStep, iPtor, iPre).size() > 1) {
            checkSizes = false;
            errorField.Append(
                wxString::Format("preprocessDataId (step %d, predictor %d, preprocess %d), ", iStep, iPtor, iPre));
          }
          if (params.GetPreprocessLevelVector(iStep, iPtor, iPre).size() > 1) {
            checkSizes = false;
            errorField.Append(
                wxString::Format("PreprocessLevel (step %d, predictor %d, preprocess %d), ", iStep, iPtor, iPre));
          }
          if (params.GetPreprocessHourVector(iStep, iPtor, iPre).size() > 1) {
            checkSizes = false;
            errorField.Append(
                wxString::Format("preprocessHours (step %d, predictor %d, preprocess %d), ", iStep, iPtor, iPre));
          }
        }
      }

      // Do the other ones anyway
      if (params.GetPredictorDataIdVector(iStep, iPtor).size() > 1) {
        checkSizes = false;
        errorField.Append(wxString::Format("PredictorDataId (step %d, predictor %d), ", iStep, iPtor));
      }
      if (params.GetPredictorLevelVector(iStep, iPtor).size() > 1) {
        checkSizes = false;
        errorField.Append(wxString::Format("PredictorLevel (step %d, predictor %d), ", iStep, iPtor));
      }
      if (params.GetPredictorHourVector(iStep, iPtor).size() > 1) {
        checkSizes = false;
        errorField.Append(wxString::Format("PredictorHours (step %d, predictor %d), ", iStep, iPtor));
      }
      if (params.GetPredictorXminVector(iStep, iPtor).size() > 1) {
        checkSizes = false;
        errorField.Append(wxString::Format("PredictorXmin (step %d, predictor %d), ", iStep, iPtor));
      }
      if (params.GetPredictorXptsnbVector(iStep, iPtor).size() > 1) {
        checkSizes = false;
        errorField.Append(wxString::Format("PredictorXptsnb (step %d, predictor %d), ", iStep, iPtor));
      }
      if (params.GetPredictorYminVector(iStep, iPtor).size() > 1) {
        checkSizes = false;
        errorField.Append(wxString::Format("PredictorYmin (step %d, predictor %d), ", iStep, iPtor));
      }
      if (params.GetPredictorYptsnbVector(iStep, iPtor).size() > 1) {
        checkSizes = false;
        errorField.Append(wxString::Format("PredictorYptsnb (step %d, predictor %d), ", iStep, iPtor));
      }
      if (params.GetPredictorCriteriaVector(iStep, iPtor).size() > 1) {
        checkSizes = false;
        errorField.Append(wxString::Format("PredictorCriteria (step %d, predictor %d), ", iStep, iPtor));
      }
      if (params.GetPredictorWeightVector(iStep, iPtor).size() > 1) {
        checkSizes = false;
        errorField.Append(wxString::Format("PredictorWeight (step %d, predictor %d), ", iStep, iPtor));
      }

      if (params.NeedsPreprocessing(iStep, iPtor)) {
        for (int iPre = 0; iPre < params.GetPreprocessSize(iStep, iPtor); iPre++) {
          if (params.GetPreprocessLevelVector(iStep, iPtor, iPre).size() > 1) {
            checkSizes = false;
            errorField.Append(wxString::Format("PreprocessLevel (step %d, predictor %d), ", iStep, iPtor));
          }
          if (params.GetPreprocessHourVector(iStep, iPtor, iPre).size() > 1) {
            checkSizes = false;
            errorField.Append(wxString::Format("PreprocessHoursV (step %d, predictor %d), ", iStep, iPtor));
          }
        }
      }
    }
  }

  if (!checkSizes) {
    errorField = errorField.Remove(errorField.Length() - 3, 2);  // Removes the last coma
    wxString errorMessage =
        _("The following parameters are not compatible with the EvaluateAllScores assessment: ") + errorField;
    wxLogError(errorMessage);
    return false;
  }

  // TODO: Set this as an option
  bool processContingencyScores = false;
  bool processContinuousScores = true;
  bool processRankHistogramScores = true;

  // Extract the stations IDs
  vvi stationsId = params.GetPredictandStationIdsVector();

  for (int iStat = 0; iStat < stationsId.size(); iStat++) {
    ClearAll();

    vi stationId = stationsId[iStat];
    wxLogMessage(_("Processing station %s"), GetPredictandStationIdsList(stationId));

    // Create result objects to save the parameters sets
    asResultsParametersArray results;
    results.Init(wxString::Format(_("station_%s_evaluation"), GetPredictandStationIdsList(stationId)));

    // Set the next station ID
    params.SetPredictandStationIds(stationId);

    // Get the number of steps
    int stepsNb = params.GetStepsNb();

    // Reset the score of the climatology
    m_scoreClimatology.clear();

    /*
     * On the calibration period
     */

    // Create results objects
    asResultsDates anaDates;
    asResultsDates anaDatesPrevious;
    asResultsValues anaValues;
    asResultsScores anaScores;
    asResultsTotalScore anaScoreFinal;

    // Process every step one after the other
    for (int iStep = 0; iStep < stepsNb; iStep++) {
      bool containsNaNs = false;
      if (iStep == 0) {
        if (!GetAnalogsDates(anaDates, &params, iStep, containsNaNs)) return false;
      } else {
        anaDatesPrevious = anaDates;
        if (!GetAnalogsSubDates(anaDates, &params, anaDatesPrevious, iStep, containsNaNs)) return false;
      }
      if (containsNaNs) {
        wxLogError(_("The dates selection contains NaNs"));
        return false;
      }
    }
    m_parameters.push_back(params);
    wxASSERT(m_parameters.size() == 1);

    if (!GetAnalogsValues(anaValues, &params, anaDates, stepsNb - 1)) return false;

    /*
     * On the validation period
     */

    asResultsDates anaDatesValid;
    asResultsDates anaDatesPreviousValid;
    asResultsValues anaValuesValid;
    asResultsScores anaScoresValid;
    asResultsTotalScore anaScoreFinalValid;

    // Get validation data
    if (params.HasValidationPeriod())  // Validate
    {
      m_validationMode = true;

      // Process every step one after the other
      for (int iStep = 0; iStep < stepsNb; iStep++) {
        bool containsNaNs = false;
        if (iStep == 0) {
          if (!GetAnalogsDates(anaDatesValid, &params, iStep, containsNaNs)) return false;
        } else {
          anaDatesPreviousValid = anaDatesValid;
          if (!GetAnalogsSubDates(anaDatesValid, &params, anaDatesPreviousValid, iStep, containsNaNs)) return false;
        }
        if (containsNaNs) {
          wxLogError(_("The dates selection contains NaNs"));
          return false;
        }
      }

      if (!GetAnalogsValues(anaValuesValid, &params, anaDatesValid, stepsNb - 1)) return false;

      m_validationMode = false;
    }

    /*
     * Scores based on the contingency table
     */

    if (processContingencyScores) {
      vwxs scoresContingency;
      scoresContingency.push_back("PC");    // PC - Proportion correct
      scoresContingency.push_back("TS");    // TS - Threat score
      scoresContingency.push_back("BIAS");  // BIAS - Bias
      scoresContingency.push_back("FARA");  // FARA - False alarm ratio
      scoresContingency.push_back("H");     // H - Hit rate
      scoresContingency.push_back("F");     // F - False alarm rate
      scoresContingency.push_back("HSS");   // HSS - Heidke skill score
      scoresContingency.push_back("PSS");   // PSS - Pierce skill score
      scoresContingency.push_back("GSS");   // GSS - Gilbert skill score

      vf thresholds;
      thresholds.push_back(0.0001f);
      thresholds.push_back(0.5f);  // 1/2 of P10 (if data are normalized)
      thresholds.push_back(1);     // P10 (if data are normalized)

      vf quantiles;
      quantiles.push_back(0.2f);
      quantiles.push_back(0.6f);
      quantiles.push_back(0.9f);

      for (int iScore = 0; iScore < scoresContingency.size(); iScore++) {
        wxLogMessage(_("Processing %s"), scoresContingency[iScore]);
        for (int iThres = 0; iThres < thresholds.size(); iThres++) {
          for (int iPc = 0; iPc < quantiles.size(); iPc++) {
            params.SetScoreName(scoresContingency[iScore]);
            params.SetScoreQuantile(quantiles[iPc]);
            params.SetScoreThreshold(thresholds[iThres]);
            if (!GetAnalogsScores(anaScores, &params, anaValues, stepsNb - 1)) return false;
            if (!GetAnalogsTotalScore(anaScoreFinal, &params, anaScores, stepsNb - 1)) return false;
            if (!GetAnalogsScores(anaScoresValid, &params, anaValuesValid, stepsNb - 1)) return false;
            if (!GetAnalogsTotalScore(anaScoreFinalValid, &params, anaScoresValid, stepsNb - 1)) return false;
            results.Add(params, anaScoreFinal.GetScore(), anaScoreFinalValid.GetScore());
            m_scoreClimatology.clear();
          }
        }
      }

      vwxs scoresQuantile;
      scoresQuantile.push_back("MAE");    // MAE - Mean absolute error
      scoresQuantile.push_back("RMSE");   // RMSE - Root mean squared error
      scoresQuantile.push_back("SEEPS");  // SEEPS - Stable equitable error in probability space

      for (int iScore = 0; iScore < scoresQuantile.size(); iScore++) {
        wxLogMessage(_("Processing %s"), scoresQuantile[iScore]);
        for (int iPc = 0; iPc < quantiles.size(); iPc++) {
          params.SetScoreName(scoresQuantile[iScore]);
          params.SetScoreQuantile(quantiles[iPc]);
          if (!GetAnalogsScores(anaScores, &params, anaValues, stepsNb - 1)) return false;
          if (!GetAnalogsTotalScore(anaScoreFinal, &params, anaScores, stepsNb - 1)) return false;
          if (!GetAnalogsScores(anaScoresValid, &params, anaValuesValid, stepsNb - 1)) return false;
          if (!GetAnalogsTotalScore(anaScoreFinalValid, &params, anaScoresValid, stepsNb - 1)) return false;
          results.Add(params, anaScoreFinal.GetScore(), anaScoreFinalValid.GetScore());
          m_scoreClimatology.clear();
        }
      }

      vwxs scoresThreshold;
      scoresThreshold.push_back("BS");   // BS - Brier score
      scoresThreshold.push_back("BSS");  // BSS - Brier skill score

      for (int iScore = 0; iScore < scoresThreshold.size(); iScore++) {
        wxLogMessage(_("Processing %s"), scoresThreshold[iScore]);
        for (int iThres = 0; iThres < thresholds.size(); iThres++) {
          params.SetScoreName(scoresThreshold[iScore]);
          params.SetScoreThreshold(thresholds[iThres]);
          if (!GetAnalogsScores(anaScores, &params, anaValues, stepsNb - 1)) return false;
          if (!GetAnalogsTotalScore(anaScoreFinal, &params, anaScores, stepsNb - 1)) return false;
          if (!GetAnalogsScores(anaScoresValid, &params, anaValuesValid, stepsNb - 1)) return false;
          if (!GetAnalogsTotalScore(anaScoreFinalValid, &params, anaScoresValid, stepsNb - 1)) return false;
          results.Add(params, anaScoreFinal.GetScore(), anaScoreFinalValid.GetScore());
          m_scoreClimatology.clear();
        }
      }
    }

    /*
     * Continuous scores
     */

    if (processContinuousScores) {
      vwxs scoresContinuous;
      scoresContinuous.push_back("DF0");   // DF0 - absolute difference of the frequency of null precipitations
      scoresContinuous.push_back("CRPS");  // CRPSAR - approximation with the rectangle method
      scoresContinuous.push_back(
          "CRPSS");  // CRPSS - CRPS skill score using the approximation with the rectangle method
      scoresContinuous.push_back(
          "CRPSaccuracyAR");  // CRPS accuracy, approximation with the rectangle method (Bontron, 2004)
      scoresContinuous.push_back(
          "CRPSsharpnessAR");  // CRPS sharpness, approximation with the rectangle method (Bontron, 2004)
      scoresContinuous.push_back("CRPSreliability");  // reliability of the CRPS (Hersbach, 2000)
      scoresContinuous.push_back("CRPSpotential");    // CRPS potential (Hersbach, 2000)

      for (int iScore = 0; iScore < scoresContinuous.size(); iScore++) {
        wxLogMessage(_("Processing %s"), scoresContinuous[iScore]);
        params.SetScoreName(scoresContinuous[iScore]);
        if (!GetAnalogsScores(anaScores, &params, anaValues, stepsNb - 1)) return false;
        if (!GetAnalogsTotalScore(anaScoreFinal, &params, anaScores, stepsNb - 1)) return false;
        if (!GetAnalogsScores(anaScoresValid, &params, anaValuesValid, stepsNb - 1)) return false;
        if (!GetAnalogsTotalScore(anaScoreFinalValid, &params, anaScoresValid, stepsNb - 1)) return false;
        results.Add(params, anaScoreFinal.GetScore(), anaScoreFinalValid.GetScore());
        m_scoreClimatology.clear();
      }
    }

    /*
     * The Verification Rank Histogram (Talagrand Diagram)
     */

    if (processRankHistogramScores) {
      wxLogMessage(_("Processing the Verification Rank Histogram"));

      int boostrapNb = 10000;
      params.SetScoreName("RankHistogram");
      m_parameters[0] = params;

      va1f histoCalib;
      va1f histoValid;

      for (int iBoot = 0; iBoot < boostrapNb; iBoot++) {
        if (!GetAnalogsScores(anaScores, &params, anaValues, stepsNb - 1)) return false;
        if (!GetAnalogsTotalScore(anaScoreFinal, &params, anaScores, stepsNb - 1)) return false;
        if (!GetAnalogsScores(anaScoresValid, &params, anaValuesValid, stepsNb - 1)) return false;
        if (!GetAnalogsTotalScore(anaScoreFinalValid, &params, anaScoresValid, stepsNb - 1)) return false;

        // Store every assessment
        histoCalib.push_back(anaScoreFinal.GetScoreArray());
        histoValid.push_back(anaScoreFinalValid.GetScoreArray());
      }

      // Average all histograms assessments
      a1f averageHistoCalib = a1f::Zero(histoCalib[0].size());
      a1f averageHistoValid = a1f::Zero(histoValid[0].size());
      for (int iBoot = 0; iBoot < boostrapNb; iBoot++) {
        averageHistoCalib += histoCalib[iBoot];
        averageHistoValid += histoValid[iBoot];
      }
      averageHistoCalib = averageHistoCalib / boostrapNb;
      averageHistoValid = averageHistoValid / boostrapNb;

      results.Add(params, averageHistoCalib, averageHistoValid);
      m_scoreClimatology.clear();

      // Reliability of the Verification Rank Histogram (Talagrand Diagram)
      params.SetScoreName("RankHistogramReliability");
      int scoresSize = anaScores.GetScores().size();
      int scoresSizeValid = anaScoresValid.GetScores().size();

      asTotalScoreRankHistogramReliability rankHistogramReliability("Total");
      rankHistogramReliability.SetRanksNb(params.GetScoreAnalogsNumber() + 1);
      float resultCalib = rankHistogramReliability.AssessOnBootstrap(averageHistoCalib, scoresSize);
      float resultValid = rankHistogramReliability.AssessOnBootstrap(averageHistoValid, scoresSizeValid);

      results.Add(params, resultCalib, resultValid);
      m_scoreClimatology.clear();
    }

    if (!results.Print()) return false;
  }

  return true;
}
