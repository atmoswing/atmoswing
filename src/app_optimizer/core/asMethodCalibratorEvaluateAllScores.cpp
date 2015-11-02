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
 * Portions Copyright 2013 Pascal Horton, Terr@num.
 */

#include "asMethodCalibratorEvaluateAllScores.h"
#include "asForecastScoreFinal.h"
#include "asForecastScoreFinalRankHistogramReliability.h"

asMethodCalibratorEvaluateAllScores::asMethodCalibratorEvaluateAllScores()
:
asMethodCalibrator()
{

}

asMethodCalibratorEvaluateAllScores::~asMethodCalibratorEvaluateAllScores()
{

}

bool asMethodCalibratorEvaluateAllScores::Calibrate(asParametersCalibration &params)
{
    // Check that we really handle a EvaluateAllScores case
    bool checkSizes = true;
    wxString errorField = wxEmptyString;
    if( params.GetTimeArrayAnalogsIntervalDaysVector().size()>1 )
    {
        checkSizes = false;
        errorField.Append("IntervalDays, ");
    }
    if( params.GetForecastScoreNameVector().size()>1 )
    {
        checkSizes = false;
        errorField.Append("ForecastScoreName, ");
    }
    if( params.GetForecastScoreTimeArrayModeVector().size()>1 )
    {
        checkSizes = false;
        errorField.Append("ForecastScoreTimeArrayMode, ");
    }
    if( params.GetForecastScoreTimeArrayDateVector().size()>1 )
    {
        checkSizes = false;
        errorField.Append("ForecastScoreTimeArrayDate ,");
    }
    if( params.GetForecastScoreTimeArrayIntervalDaysVector().size()>1 )
    {
        checkSizes = false;
        errorField.Append("ForecastScoreTimeArrayIntervalDays, ");
    }
    if( params.GetForecastScorePostprocessDupliExpVector().size()>1 )
    {
        checkSizes = false;
        errorField.Append("ForecastScorePostprocessDupliExp, ");
    }

    for (int i_step=0; i_step<params.GetStepsNb(); i_step++)
    {
        if( params.GetAnalogsNumberVector(i_step).size()>1 )
        {
            checkSizes = false;
            errorField.Append(wxString::Format("AnalogsNumber (step %d), ", i_step));
        }
        for (int i_predictor=0; i_predictor<params.GetPredictorsNb(i_step); i_predictor++)
        {
            if (params.NeedsPreprocessing(i_step, i_predictor))
            {
                for (int i_pre=0; i_pre<params.GetPreprocessSize(i_step, i_predictor); i_pre++)
                {
                    if( params.GetPreprocessDataIdVector(i_step, i_predictor, i_pre).size()>1 )
                    {
                        checkSizes = false;
                        errorField.Append(wxString::Format("PreprocessDataId (step %d, predictor %d, preprocess %d), ", i_step, i_predictor, i_pre));
                    }
                    if( params.GetPreprocessLevelVector(i_step, i_predictor, i_pre).size()>1 )
                    {
                        checkSizes = false;
                        errorField.Append(wxString::Format("PreprocessLevel (step %d, predictor %d, preprocess %d), ", i_step, i_predictor, i_pre));
                    }
                    if( params.GetPreprocessTimeHoursVector(i_step, i_predictor, i_pre).size()>1 )
                    {
                        checkSizes = false;
                        errorField.Append(wxString::Format("PreprocessTimeHours (step %d, predictor %d, preprocess %d), ", i_step, i_predictor, i_pre));
                    }
                }
            }

            // Do the other ones anyway
            if( params.GetPredictorDataIdVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorDataId (step %d, predictor %d), ", i_step, i_predictor));
            }
            if( params.GetPredictorLevelVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorLevel (step %d, predictor %d), ", i_step, i_predictor));
            }
            if( params.GetPredictorTimeHoursVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorTimeHours (step %d, predictor %d), ", i_step, i_predictor));
            }
            if( params.GetPredictorXminVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorXmin (step %d, predictor %d), ", i_step, i_predictor));
            }
            if( params.GetPredictorXptsnbVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorXptsnb (step %d, predictor %d), ", i_step, i_predictor));
            }
            if( params.GetPredictorYminVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorYmin (step %d, predictor %d), ", i_step, i_predictor));
            }
            if( params.GetPredictorYptsnbVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorYptsnb (step %d, predictor %d), ", i_step, i_predictor));
            }
            if( params.GetPredictorCriteriaVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorCriteria (step %d, predictor %d), ", i_step, i_predictor));
            }
            if( params.GetPredictorWeightVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorWeight (step %d, predictor %d), ", i_step, i_predictor));
            }

            if(params.NeedsPreprocessing(i_step, i_predictor))
            {
                for (int i_dataset=0; i_dataset<params.GetPreprocessSize(i_step, i_predictor); i_dataset++)
                {
                    if( params.GetPreprocessLevelVector(i_step, i_predictor, i_dataset).size()>1 )
                    {
                        checkSizes = false;
                        errorField.Append(wxString::Format("PreprocessLevel (step %d, predictor %d), ", i_step, i_predictor));
                    }
                    if( params.GetPreprocessTimeHoursVector(i_step, i_predictor, i_dataset).size()>1 )
                    {
                        checkSizes = false;
                        errorField.Append(wxString::Format("PreprocessTimeHoursV (step %d, predictor %d), ", i_step, i_predictor));
                    }
                }
            }
        }
    }

    if(!checkSizes)
    {
        errorField = errorField.Remove(errorField.Length()-3, 2); // Removes the last coma
        wxString errorMessage = _("The following parameters are not compatible with the EvaluateAllScores assessment: ") + errorField;
        asLogError(errorMessage);
        return false;
    }

    // TODO: set this as an option
    bool processContingencyScores = false;
    bool processContinuousScores = true;
    bool processRankHistogramScores = true;

    // Extract the stations IDs
    VVectorInt stationsId = params.GetPredictandStationIdsVector();

    for (unsigned int i_stat=0; i_stat<stationsId.size(); i_stat++)
    {
        ClearAll();

        VectorInt stationId = stationsId[i_stat];
        asLogMessageImportant(wxString::Format(_("Processing station %s"), GetPredictandStationIdsList(stationId)));

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
        asResultsAnalogsDates anaDates;
        asResultsAnalogsDates anaDatesPrevious;
        asResultsAnalogsValues anaValues;
        asResultsAnalogsForecastScores anaScores;
        asResultsAnalogsForecastScoreFinal anaScoreFinal;

        // Process every step one after the other
        for (int i_step=0; i_step<stepsNb; i_step++)
        {
            bool containsNaNs = false;
            if (i_step==0)
            {
                if(!GetAnalogsDates(anaDates, params, i_step, containsNaNs)) return false;
            }
            else
            {
                anaDatesPrevious = anaDates;
                if(!GetAnalogsSubDates(anaDates, params, anaDatesPrevious, i_step, containsNaNs)) return false;
            }
            if (containsNaNs)
            {
                asLogError(_("The dates selection contains NaNs"));
                return false;
            }
        }
        m_parameters.push_back(params);
        wxASSERT(m_parameters.size()==1);

        if(!GetAnalogsValues(anaValues, params, anaDates, stepsNb-1)) return false;
        
        /* 
         * On the validation period 
         */

        asResultsAnalogsDates anaDatesValid;
        asResultsAnalogsDates anaDatesPreviousValid;
        asResultsAnalogsValues anaValuesValid;
        asResultsAnalogsForecastScores anaScoresValid;
        asResultsAnalogsForecastScoreFinal anaScoreFinalValid;

        // Get validation data
        if (params.HasValidationPeriod()) // Validate
        {
            m_validationMode = true;

            // Process every step one after the other
            for (int i_step=0; i_step<stepsNb; i_step++)
            {
                bool containsNaNs = false;
                if (i_step==0)
                {
                    if(!GetAnalogsDates(anaDatesValid, params, i_step, containsNaNs)) return false;
                }
                else
                {
                    anaDatesPreviousValid = anaDatesValid;
                    if(!GetAnalogsSubDates(anaDatesValid, params, anaDatesPreviousValid, i_step, containsNaNs)) return false;
                }
                if (containsNaNs)
                {
                    asLogError(_("The dates selection contains NaNs"));
                    return false;
                }
            }

            if(!GetAnalogsValues(anaValuesValid, params, anaDatesValid, stepsNb-1)) return false;

            m_validationMode = false;
        }

        /* 
         * Scores based on the contingency table 
         */

        if (processContingencyScores)
        {
            VectorString scoresContingency;
            scoresContingency.push_back("PC"); // PC - Proportion correct
            scoresContingency.push_back("TS"); // TS - Threat score
            scoresContingency.push_back("BIAS"); // BIAS - Bias
            scoresContingency.push_back("FARA"); // FARA - False alarm ratio
            scoresContingency.push_back("H"); // H - Hit rate
            scoresContingency.push_back("F"); // F - False alarm rate
            scoresContingency.push_back("HSS"); // HSS - Heidke skill score
            scoresContingency.push_back("PSS"); // PSS - Pierce skill score
            scoresContingency.push_back("GSS"); // GSS - Gilbert skill score

            VectorFloat thresholds;
            thresholds.push_back(0.0001f);
            thresholds.push_back(0.5f); // 1/2 of P10 (if data are normalized)
            thresholds.push_back(1);  // P10 (if data are normalized)

            VectorFloat quantiles;
            quantiles.push_back(0.2f);
            quantiles.push_back(0.6f);
            quantiles.push_back(0.9f);

            for (unsigned int i_score=0;i_score<scoresContingency.size();i_score++)
            {
                asLogMessageImportant(wxString::Format(_("Processing %s"), scoresContingency[i_score]));
                for (unsigned int i_thres=0;i_thres<thresholds.size();i_thres++)
                {
                    for (unsigned int i_pc=0;i_pc<quantiles.size();i_pc++)
                    {
                        params.SetForecastScoreName(scoresContingency[i_score]);
                        params.SetForecastScoreQuantile(quantiles[i_pc]);
                        params.SetForecastScoreThreshold(thresholds[i_thres]);
                        if(!GetAnalogsForecastScores(anaScores, params, anaValues, stepsNb-1)) return false;
                        if(!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, stepsNb-1)) return false;
                        if(!GetAnalogsForecastScores(anaScoresValid, params, anaValuesValid, stepsNb-1)) return false;
                        if(!GetAnalogsForecastScoreFinal(anaScoreFinalValid, params, anaScoresValid, stepsNb-1)) return false;
                        results.Add(params,anaScoreFinal.GetForecastScore(), anaScoreFinalValid.GetForecastScore());
                        m_scoreClimatology.clear();
                    }
                }
            }

            VectorString scoresQuantile;
            scoresQuantile.push_back("MAE"); // MAE - Mean absolute error
            scoresQuantile.push_back("RMSE"); // RMSE - Root mean squared error
            scoresQuantile.push_back("SEEPS"); // SEEPS - Stable equitable error in probability space

            for (unsigned int i_score=0;i_score<scoresQuantile.size();i_score++)
            {
                asLogMessageImportant(wxString::Format(_("Processing %s"), scoresQuantile[i_score]));
                for (unsigned int i_pc=0;i_pc<quantiles.size();i_pc++)
                {
                    params.SetForecastScoreName(scoresQuantile[i_score]);
                    params.SetForecastScoreQuantile(quantiles[i_pc]);
                    if(!GetAnalogsForecastScores(anaScores, params, anaValues, stepsNb-1)) return false;
                    if(!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, stepsNb-1)) return false;
                    if(!GetAnalogsForecastScores(anaScoresValid, params, anaValuesValid, stepsNb-1)) return false;
                    if(!GetAnalogsForecastScoreFinal(anaScoreFinalValid, params, anaScoresValid, stepsNb-1)) return false;
                    results.Add(params,anaScoreFinal.GetForecastScore(), anaScoreFinalValid.GetForecastScore());
                    m_scoreClimatology.clear();
                }
            }

            VectorString scoresThreshold;
            scoresThreshold.push_back("BS"); // BS - Brier score
            scoresThreshold.push_back("BSS"); // BSS - Brier skill score

            for (unsigned int i_score=0;i_score<scoresThreshold.size();i_score++)
            {
                asLogMessageImportant(wxString::Format(_("Processing %s"), scoresThreshold[i_score]));
                for (unsigned int i_thres=0;i_thres<thresholds.size();i_thres++)
                {
                    params.SetForecastScoreName(scoresThreshold[i_score]);
                    params.SetForecastScoreThreshold(thresholds[i_thres]);
                    if(!GetAnalogsForecastScores(anaScores, params, anaValues, stepsNb-1)) return false;
                    if(!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, stepsNb-1)) return false;
                    if(!GetAnalogsForecastScores(anaScoresValid, params, anaValuesValid, stepsNb-1)) return false;
                    if(!GetAnalogsForecastScoreFinal(anaScoreFinalValid, params, anaScoresValid, stepsNb-1)) return false;
                    results.Add(params,anaScoreFinal.GetForecastScore(), anaScoreFinalValid.GetForecastScore());
                    m_scoreClimatology.clear();
                }
            }
        }

        /* 
         * Continuous scores 
         */
        
        if (processContinuousScores)
        {
            VectorString scoresContinuous;
            scoresContinuous.push_back("DF0"); // DF0 - absolute difference of the frequency of null precipitations
            scoresContinuous.push_back("CRPS"); // CRPSAR - approximation with the rectangle method
            scoresContinuous.push_back("CRPSS"); // CRPSS - CRPS skill score using the approximation with the rectangle method
            scoresContinuous.push_back("CRPSaccuracyAR"); // CRPS accuracy, approximation with the rectangle method (Bontron, 2004)
            scoresContinuous.push_back("CRPSsharpnessAR"); // CRPS sharpness, approximation with the rectangle method (Bontron, 2004)
            scoresContinuous.push_back("CRPSreliability"); // reliability of the CRPS (Hersbach, 2000)
            scoresContinuous.push_back("CRPSpotential"); // CRPS potential (Hersbach, 2000)

            for (unsigned int i_score=0;i_score<scoresContinuous.size();i_score++)
            {
                asLogMessageImportant(wxString::Format(_("Processing %s"), scoresContinuous[i_score]));
                params.SetForecastScoreName(scoresContinuous[i_score]);
                if(!GetAnalogsForecastScores(anaScores, params, anaValues, stepsNb-1)) return false;
                if(!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, stepsNb-1)) return false;
                if(!GetAnalogsForecastScores(anaScoresValid, params, anaValuesValid, stepsNb-1)) return false;
                if(!GetAnalogsForecastScoreFinal(anaScoreFinalValid, params, anaScoresValid, stepsNb-1)) return false;
                results.Add(params,anaScoreFinal.GetForecastScore(), anaScoreFinalValid.GetForecastScore());
                m_scoreClimatology.clear();
            }
        }
        
        /* 
         * The Verification Rank Histogram (Talagrand Diagram) 
         */

        if (processRankHistogramScores)
        {
            asLogMessageImportant(_("Processing the Verification Rank Histogram"));

            int boostrapNb = 10000;
            params.SetForecastScoreName("RankHistogram");
            m_parameters[0]=params;

            std::vector < Array1DFloat > histoCalib;
            std::vector < Array1DFloat > histoValid;

            for (int i_boot=0; i_boot<boostrapNb; i_boot++)
            {
                if(!GetAnalogsForecastScores(anaScores, params, anaValues, stepsNb-1)) return false;
                if(!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, stepsNb-1)) return false;
                if(!GetAnalogsForecastScores(anaScoresValid, params, anaValuesValid, stepsNb-1)) return false;
                if(!GetAnalogsForecastScoreFinal(anaScoreFinalValid, params, anaScoresValid, stepsNb-1)) return false;

                // Store every assessment
                histoCalib.push_back(anaScoreFinal.GetForecastScoreArray());
                histoValid.push_back(anaScoreFinalValid.GetForecastScoreArray());
            }
        
            // Average all histograms assessments
            Array1DFloat averageHistoCalib = Array1DFloat::Zero(histoCalib[0].size());
            Array1DFloat averageHistoValid = Array1DFloat::Zero(histoValid[0].size());
            for (int i_boot=0; i_boot<boostrapNb; i_boot++)
            {
                averageHistoCalib += histoCalib[i_boot];
                averageHistoValid += histoValid[i_boot];
            }
            averageHistoCalib = averageHistoCalib/boostrapNb;
            averageHistoValid = averageHistoValid/boostrapNb;

            results.Add(params, averageHistoCalib, averageHistoValid);
            m_scoreClimatology.clear();

            // Reliability of the Verification Rank Histogram (Talagrand Diagram)
            params.SetForecastScoreName("RankHistogramReliability");
            int forecastScoresSize = anaScores.GetForecastScores().size();
            int forecastScoresSizeValid = anaScoresValid.GetForecastScores().size();

            asForecastScoreFinalRankHistogramReliability rankHistogramReliability(asForecastScoreFinal::Total);
            rankHistogramReliability.SetRanksNb(params.GetForecastScoreAnalogsNumber()+1);
            float resultCalib = rankHistogramReliability.AssessOnBootstrap(averageHistoCalib, forecastScoresSize);
            float resultValid = rankHistogramReliability.AssessOnBootstrap(averageHistoValid, forecastScoresSizeValid);

            results.Add(params, resultCalib, resultValid);
            m_scoreClimatology.clear();
        }

        if(!results.Print()) return false;

    }

    return true;
}
