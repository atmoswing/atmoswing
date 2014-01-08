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
    if( params.GetForecastScoreAnalogsNumberVector().size()>1 )
    {
        checkSizes = false;
        errorField.Append("ForecastScoreAnalogsNumber, ");
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
                    if( params.GetPreprocessDTimeHoursVector(i_step, i_predictor, i_pre).size()>1 )
                    {
                        checkSizes = false;
                        errorField.Append(wxString::Format("PreprocessDTimeHours (step %d, predictor %d, preprocess %d), ", i_step, i_predictor, i_pre));
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
            if( params.GetPredictorDTimeHoursVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorDTimeHours (step %d, predictor %d), ", i_step, i_predictor));
            }
            if( params.GetPredictorUminVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorUmin (step %d, predictor %d), ", i_step, i_predictor));
            }
            if( params.GetPredictorUptsnbVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorUptsnb (step %d, predictor %d), ", i_step, i_predictor));
            }
            if( params.GetPredictorVminVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorVmin (step %d, predictor %d), ", i_step, i_predictor));
            }
            if( params.GetPredictorVptsnbVector(i_step, i_predictor).size()>1 )
            {
                checkSizes = false;
                errorField.Append(wxString::Format("PredictorVptsnb (step %d, predictor %d), ", i_step, i_predictor));
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
                    if( params.GetPreprocessDTimeHoursVector(i_step, i_predictor, i_dataset).size()>1 )
                    {
                        checkSizes = false;
                        errorField.Append(wxString::Format("PreprocessDTimeHoursV (step %d, predictor %d), ", i_step, i_predictor));
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

    // Extract the stations IDs
    VectorInt stationsId = params.GetPredictandStationsIdVector();

    // Create a analogsdate object to save previous analogs dates selection.
    asResultsAnalogsDates anaDatesPrevious;

    for (unsigned int i_stat=0; i_stat<stationsId.size(); i_stat++)
    {
        ClearAll();

        int stationId = stationsId[i_stat];

        // Reset the score of the climatology
        m_ScoreClimatology = 0;

        // Create results objects
        asResultsAnalogsDates anaDates;
        asResultsAnalogsValues anaValues;
        asResultsAnalogsForecastScores anaScores;
        asResultsAnalogsForecastScoreFinal anaScoreFinal;

        // Create result objects to save the parameters sets
        asResultsParametersArray results;
        results.Init(wxString::Format(_("station_%d_evaluation"), stationId));

        // Set the next station ID
        params.SetPredictandStationId(stationId);

        // Process every step one after the other
        int stepsNb = params.GetStepsNb();
        for (int i_step=0; i_step<stepsNb; i_step++)
        {
            bool containsNaNs = false;
            if (i_step==0)
            {
                if(!GetAnalogsDates(anaDates, params, i_step, containsNaNs)) return false;
            }
            else
            {
                if(!GetAnalogsSubDates(anaDates, params, anaDatesPrevious, i_step, containsNaNs)) return false;
            }
            if (containsNaNs)
            {
                asLogError(_("The dates selection contains NaNs"));
                return false;
            }

            // Keep the analogs dates of the best parameters set
            anaDatesPrevious = anaDates;
        }
        m_Parameters.push_back(params);
        wxASSERT(m_Parameters.size()==1);

        if(!GetAnalogsValues(anaValues, params, anaDates, stepsNb-1)) return false;

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
        thresholds.push_back(0.333333333f); // 1/3 of P10 (if data are normalized)
        thresholds.push_back(0.666666667f); // 2/3 of P10 (if data are normalized)
        thresholds.push_back(1);  // P10 (if data are normalized)

        VectorFloat percentiles;
        percentiles.push_back(0.2f);
        percentiles.push_back(0.6f);
        percentiles.push_back(0.9f);

		for (unsigned int i_score=0;i_score<scoresContingency.size();i_score++)
        {
            asLogMessageImportant(wxString::Format(_("Processing %s"), scoresContingency[i_score]));
            for (unsigned int i_thres=0;i_thres<thresholds.size();i_thres++)
            {
                for (unsigned int i_pc=0;i_pc<percentiles.size();i_pc++)
                {
                    params.SetForecastScoreName(scoresContingency[i_score]);
                    params.SetForecastScorePercentile(percentiles[i_pc]);
                    params.SetForecastScoreThreshold(thresholds[i_thres]);
                    if(!GetAnalogsForecastScores(anaScores, params, anaValues, stepsNb-1)) return false;
                    if(!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, stepsNb-1)) return false;
                    m_Parameters[0]=params;
                    Validate();
                    results.Add(params,anaScoreFinal.GetForecastScore(), m_ScoreValid);
                    m_ScoreClimatology=0;
                }
            }
            if(!results.Print()) return false;
        }

        VectorString scoresPercentile;
        scoresPercentile.push_back("MAE"); // MAE - Mean absolute error
        scoresPercentile.push_back("RMSE"); // RMSE - Root mean squared error
        scoresPercentile.push_back("SEEPS"); // SEEPS - Stable equitable error in probability space

        for (unsigned int i_score=0;i_score<scoresPercentile.size();i_score++)
        {
            asLogMessageImportant(wxString::Format(_("Processing %s"), scoresPercentile[i_score]));
            for (unsigned int i_pc=0;i_pc<percentiles.size();i_pc++)
            {
                params.SetForecastScoreName(scoresPercentile[i_score]);
                params.SetForecastScorePercentile(percentiles[i_pc]);
                if(!GetAnalogsForecastScores(anaScores, params, anaValues, stepsNb-1)) return false;
                if(!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, stepsNb-1)) return false;
                m_Parameters[0]=params;
                Validate();
                results.Add(params,anaScoreFinal.GetForecastScore(), m_ScoreValid);
                m_ScoreClimatology=0;
            }
            if(!results.Print()) return false;
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
                m_Parameters[0]=params;
                Validate();
                results.Add(params,anaScoreFinal.GetForecastScore(), m_ScoreValid);
                m_ScoreClimatology=0;
            }
            if(!results.Print()) return false;
        }

        VectorString scoresContinuous;
        scoresContinuous.push_back("CRPS"); // CRPSAR - approximation with the rectangle method
        scoresContinuous.push_back("CRPSS"); // CRPSS - CRPS skill score using the approximation with the rectangle method
        scoresContinuous.push_back("DF0"); // DF0 - absolute difference of the frequency of null precipitations

        for (unsigned int i_score=0;i_score<scoresContinuous.size();i_score++)
        {
            asLogMessageImportant(wxString::Format(_("Processing %s"), scoresContinuous[i_score]));
            params.SetForecastScoreName(scoresContinuous[i_score]);
            if(!GetAnalogsForecastScores(anaScores, params, anaValues, stepsNb-1)) return false;
            if(!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, stepsNb-1)) return false;
            m_Parameters[0]=params;
            Validate();
            results.Add(params,anaScoreFinal.GetForecastScore(), m_ScoreValid);
            m_ScoreClimatology=0;
        }
        if(!results.Print()) return false;

    }

    // Delete preloaded data
    DeletePreloadedData();

    return true;
}
