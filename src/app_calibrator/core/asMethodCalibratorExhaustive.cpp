#include "asMethodCalibratorExhaustive.h"

asMethodCalibratorExhaustive::asMethodCalibratorExhaustive()
:
asMethodCalibrator()
{

}

asMethodCalibratorExhaustive::~asMethodCalibratorExhaustive()
{

}

bool asMethodCalibratorExhaustive::Calibrate(asParametersCalibration &params)
{




    wxFAIL_MSG(_("ProcessExhaustive is not ready yet to be used. Needs correction before using."));
// FIXME (Pascal#1#): Actually doesn't manage multiple steps -> add asResultsAnalogsDates anaDatesPrevious
// FIXME (Pascal#1#): Set the stations loop in the beginning to be able to calibrate the next steps and to save a resulting file per station







    // Tested parameters object
    asResultsParametersArray results_tested;
    results_tested.Init("exhaustive_tested_parameters");

    // Create results objects
    asResultsAnalogsDates anaDates;
    asResultsAnalogsValues anaValues;
    asResultsAnalogsForecastScores anaScores;
    asResultsAnalogsForecastScoreFinal anaScoreFinal;

    asLogState(_("Calibration: going through every parameters combination."));

    VectorInt vIntervalDays = params.GetTimeArrayAnalogsIntervalDaysVector();

    for(unsigned int i_intervaldays=0; i_intervaldays<vIntervalDays.size(); i_intervaldays++)
    {
        params.SetTimeArrayAnalogsIntervalDays(vIntervalDays[i_intervaldays]);

        for(int i_step=0; i_step<params.GetStepsNb(); i_step++)
        {
            for(int i_ptor=0; i_ptor<params.GetPredictorsNb(i_step); i_ptor++)
            {
                if(params.NeedsPreprocessing(i_step, i_ptor))
                {
                    for(int i_dataset=0; i_dataset<params.GetPreprocessSize(i_step, i_ptor); i_dataset++)
                    {
                        VectorDouble vPreprocessDTimeHours = params.GetPreprocessDTimeHoursVector(i_step, i_ptor, i_dataset);

                        for(unsigned int i_preprocessdtime=0; i_preprocessdtime<vPreprocessDTimeHours.size(); i_preprocessdtime++)
                        {
                            params.SetPreprocessDTimeHours(i_step, i_ptor, i_dataset, vPreprocessDTimeHours[i_preprocessdtime]);
                            params.SetPredictorDTimeHours(i_step, i_ptor, vPreprocessDTimeHours[i_preprocessdtime]);

                            VectorFloat vPreprocessLevels = params.GetPreprocessLevelVector(i_step, i_ptor, i_dataset);

                            for(unsigned int i_preproceslevel=0; i_preproceslevel<vPreprocessLevels.size(); i_preproceslevel++)
                            {
                                params.SetPreprocessLevel(i_step, i_ptor, i_dataset, vPreprocessLevels[i_preproceslevel]);
                                params.SetPredictorLevel(i_step, i_ptor, vPreprocessLevels[i_preproceslevel]);

                                VectorInt vPredictorUptsnb = params.GetPredictorUptsnbVector(i_step, i_ptor);

                                for(unsigned int i_uptsnb=0; i_uptsnb<vPredictorUptsnb.size(); i_uptsnb++)
                                {
                                    int uptsnb = vPredictorUptsnb[i_uptsnb];

                                    VectorInt vPredictorVptsnb = params.GetPredictorVptsnbVector(i_step, i_ptor);

                                    for(unsigned int i_vptsnb=0; i_vptsnb<vPredictorVptsnb.size(); i_vptsnb++)
                                    {
                                        int vptsnb = vPredictorVptsnb[i_vptsnb];

                                        VectorDouble vPredictorUmin = params.GetPredictorUminVector(i_step, i_ptor);

                                        for(unsigned int i_umin=0; i_umin<vPredictorUmin.size(); i_umin++)
                                        {
                                            double umin = vPredictorUmin[i_umin];

                                            VectorDouble vPredictorVmin = params.GetPredictorVminVector(i_step, i_ptor);

                                            for(unsigned int i_vmin=0; i_vmin<vPredictorVmin.size(); i_vmin++)
                                            {
                                                double vmin = vPredictorVmin[i_vmin];

                                                params.SetPredictorUmin(i_step, i_ptor, umin);
                                                params.SetPredictorUptsnb(i_step, i_ptor, uptsnb);
                                                params.SetPredictorVmin(i_step, i_ptor, vmin);
                                                params.SetPredictorVptsnb(i_step, i_ptor, vptsnb);

                                                asLogMessage(wxString::Format(_("Processing area: umin=%.2f, uptsnb=%.2f, vmin=%.2f, vptsnb=%.2f."), umin, uptsnb, vmin, vptsnb));

                                                VectorString vPredictorCriteria = params.GetPredictorCriteriaVector(i_step, i_ptor);

                                                for(unsigned int i_criteria=0; i_criteria<vPredictorCriteria.size(); i_criteria++)
                                                {
                                                    params.SetPredictorCriteria(i_step, i_ptor, vPredictorCriteria[i_criteria]);

                                                    VectorFloat vPredictorWeight = params.GetPredictorWeightVector(i_step, i_ptor);

                                                    for(unsigned int i_weight=0; i_weight<vPredictorWeight.size(); i_weight++)
                                                    {
                                                        params.SetPredictorWeight(i_step, i_ptor, vPredictorWeight[i_weight]);

                                                        VectorInt vAnalogsNb = params.GetAnalogsNumberVector(i_step);

                                                        for(unsigned int i_anb=0; i_anb<vAnalogsNb.size(); i_anb++)
                                                        {
                                                            params.SetAnalogsNumber(i_step, vAnalogsNb[i_anb]);

                                                            VectorInt vStationId = params.GetPredictandStationsIdVector();

                                                            for(unsigned int i_statid=0; i_statid<vStationId.size(); i_statid++)
                                                            {
                                                                params.SetPredictandStationId(vStationId[i_statid]);

                                                                // Reset the score of the climatology
                                                                m_ScoreClimatology = 0;

                                                                // Analogs dates
                                                                asLogMessage(_("Processing analogs dates."));
                                                                params.FixTimeShift();
                                                                params.FixAnalogsNb();
                                                                params.FixCoordinates();
                                                                params.FixWeights();

                                                                bool containsNaNs = false;
                                                                if(!GetAnalogsDates(anaDates, params, i_step, containsNaNs)) return false;
                                                                if (containsNaNs)
                                                                {
                                                                    asLogError(_("The dates selection contains NaNs"));
                                                                    return false;
                                                                }

                                                                // Analogs values
                                                                asLogMessage(_("Processing analogs values."));
                                                                if(!GetAnalogsValues(anaValues, params, anaDates, i_step)) return false;

                                                                VectorString vForecastScoreName = params.GetForecastScoreNameVector();

                                                                for(unsigned int i_fscorename=0; i_fscorename<vForecastScoreName.size(); i_fscorename++)
                                                                {
                                                                    params.SetForecastScoreName(vForecastScoreName[i_fscorename]);

                                                                    VectorInt vForecastScoreAnalogsNumber = params.GetForecastScoreAnalogsNumberVector();

                                                                    for(unsigned int i_fscoreanb=0; i_fscoreanb<vForecastScoreAnalogsNumber.size(); i_fscoreanb++)
                                                                    {
                                                                        params.SetForecastScoreAnalogsNumber(vForecastScoreAnalogsNumber[i_fscoreanb]);

                                                                        // Forecast scores
                                                                        asLogMessage(_("Processing forecast scores."));
                                                                        params.FixAnalogsNb();

                                                                        if(!GetAnalogsForecastScores(anaScores, params, anaValues, i_step)) return false;

                                                                        VectorString vForecastScoreTimeArrayMode = params.GetForecastScoreTimeArrayModeVector();

                                                                        for(unsigned int i_fscoretamode=0; i_fscoretamode<vForecastScoreTimeArrayMode.size(); i_fscoretamode++)
                                                                        {
                                                                            params.SetForecastScoreTimeArrayMode(vForecastScoreTimeArrayMode[i_fscoretamode]);

                                                                            VectorDouble vForecastScoreTimeArrayDate = params.GetForecastScoreTimeArrayDateVector();

                                                                            for(unsigned int i_fscoretadate=0; i_fscoretadate<vForecastScoreTimeArrayDate.size(); i_fscoretadate++)
                                                                            {
                                                                                params.SetForecastScoreTimeArrayDate(vForecastScoreTimeArrayDate[i_fscoretadate]);

                                                                                VectorInt vForecastScoreTimeArrayIntervalDays = params.GetForecastScoreTimeArrayIntervalDaysVector();

                                                                                for(unsigned int i_fscoretaintdays=0; i_fscoretaintdays<vForecastScoreTimeArrayIntervalDays.size(); i_fscoretaintdays++)
                                                                                {
                                                                                    params.SetForecastScoreTimeArrayIntervalDays(vForecastScoreTimeArrayIntervalDays[i_fscoretaintdays]);

                                                                                    VectorFloat vForecastScorePostprocessDupliExp = params.GetForecastScorePostprocessDupliExpVector();

                                                                                    for(unsigned int i_fscorepostdupliexp=0; i_fscorepostdupliexp<vForecastScorePostprocessDupliExp.size(); i_fscorepostdupliexp++)
                                                                                    {
                                                                                        params.SetForecastScorePostprocessDupliExp(vForecastScorePostprocessDupliExp[i_fscorepostdupliexp]);

                                                                                        // Final forecast score
                                                                                        if(!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, i_step)) return false;

                                                                                        // Validate
                                                                                        Validate();

                                                                                        // Saving results
                                                                                        results_tested.Add(params, anaScoreFinal.GetForecastScore(), m_ScoreValid);
                                                                                        if(!results_tested.Print()) return false;
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    VectorDouble vPredictorDTimeHours = params.GetPredictorDTimeHoursVector(i_step, i_ptor);

                    for(unsigned int i_predictordtime=0; i_predictordtime<vPredictorDTimeHours.size(); i_predictordtime++)
                    {
                        params.SetPredictorDTimeHours(i_step, i_ptor, vPredictorDTimeHours[i_predictordtime]);

                        VectorFloat vPredictorLevels = params.GetPredictorLevelVector(i_step, i_ptor);

                        for(unsigned int i_predictorlevel=0; i_predictorlevel<vPredictorLevels.size(); i_predictorlevel++)
                        {
                            params.SetPredictorLevel(i_step, i_ptor, vPredictorLevels[i_predictorlevel]);

                            VectorInt vPredictorUptsnb = params.GetPredictorUptsnbVector(i_step, i_ptor);

                            for(unsigned int i_uptsnb=0; i_uptsnb<vPredictorUptsnb.size(); i_uptsnb++)
                            {
                                int uptsnb = vPredictorUptsnb[i_uptsnb];

                                VectorInt vPredictorVptsnb = params.GetPredictorVptsnbVector(i_step, i_ptor);

                                for(unsigned int i_vptsnb=0; i_vptsnb<vPredictorVptsnb.size(); i_vptsnb++)
                                {
                                    int vptsnb = vPredictorVptsnb[i_vptsnb];

                                    VectorDouble vPredictorUmin = params.GetPredictorUminVector(i_step, i_ptor);

                                    for(unsigned int i_umin=0; i_umin<vPredictorUmin.size(); i_umin++)
                                    {
                                        double umin = vPredictorUmin[i_umin];

                                        VectorDouble vPredictorVmin = params.GetPredictorVminVector(i_step, i_ptor);

                                        for(unsigned int i_vmin=0; i_vmin<vPredictorVmin.size(); i_vmin++)
                                        {
                                            double vmin = vPredictorVmin[i_vmin];

                                            params.SetPredictorUmin(i_step, i_ptor, umin);
                                            params.SetPredictorUptsnb(i_step, i_ptor, uptsnb);
                                            params.SetPredictorVmin(i_step, i_ptor, vmin);
                                            params.SetPredictorVptsnb(i_step, i_ptor, vptsnb);

                                            asLogMessage(wxString::Format(_("Processing area: umin=%.2f, uptsnb=%.2f, vmin=%.2f, vptsnb=%.2f."), umin, uptsnb, vmin, vptsnb));

                                            VectorString vPredictorCriteria = params.GetPredictorCriteriaVector(i_step, i_ptor);

                                            for(unsigned int i_criteria=0; i_criteria<vPredictorCriteria.size(); i_criteria++)
                                            {
                                                params.SetPredictorCriteria(i_step, i_ptor, vPredictorCriteria[i_criteria]);

                                                VectorFloat vPredictorWeight = params.GetPredictorWeightVector(i_step, i_ptor);

                                                for(unsigned int i_weight=0; i_weight<vPredictorWeight.size(); i_weight++)
                                                {
                                                    params.SetPredictorWeight(i_step, i_ptor, vPredictorWeight[i_weight]);

                                                    VectorInt vAnalogsNb = params.GetAnalogsNumberVector(i_step);

                                                    for(unsigned int i_anb=0; i_anb<vAnalogsNb.size(); i_anb++)
                                                    {
                                                        params.SetAnalogsNumber(i_step, vAnalogsNb[i_anb]);

                                                        VectorInt vStationId = params.GetPredictandStationsIdVector();

                                                        for(unsigned int i_statid=0; i_statid<vStationId.size(); i_statid++)
                                                        {
                                                            params.SetPredictandStationId(vStationId[i_statid]);

                                                            // Reset the score of the climatology
                                                            m_ScoreClimatology = 0;

                                                            // Analogs dates
                                                            asLogMessage(_("Processing analogs dates."));
                                                            params.FixTimeShift();
                                                            params.FixAnalogsNb();
                                                            params.FixCoordinates();
                                                            params.FixWeights();

                                                            bool containsNaNs = false;
                                                            if(!GetAnalogsDates(anaDates, params, i_step, containsNaNs)) return false;
                                                            if (containsNaNs)
                                                            {
                                                                asLogError(_("The dates selection contains NaNs"));
                                                                return false;
                                                            }

                                                            // Analogs values
                                                            asLogMessage(_("Processing analogs values."));
                                                            if(!GetAnalogsValues(anaValues, params, anaDates, i_step)) return false;

                                                            VectorString vForecastScoreName = params.GetForecastScoreNameVector();

                                                            for(unsigned int i_fscorename=0; i_fscorename<vForecastScoreName.size(); i_fscorename++)
                                                            {
                                                                params.SetForecastScoreName(vForecastScoreName[i_fscorename]);

                                                                VectorInt vForecastScoreAnalogsNumber = params.GetForecastScoreAnalogsNumberVector();

                                                                for(unsigned int i_fscoreanb=0; i_fscoreanb<vForecastScoreAnalogsNumber.size(); i_fscoreanb++)
                                                                {
                                                                    params.SetForecastScoreAnalogsNumber(vForecastScoreAnalogsNumber[i_fscoreanb]);

                                                                    // Forecast scores
                                                                    asLogMessage(_("Processing forecast scores."));
                                                                    params.FixAnalogsNb();

                                                                    if(!GetAnalogsForecastScores(anaScores, params, anaValues, i_step)) return false;

                                                                    VectorString vForecastScoreTimeArrayMode = params.GetForecastScoreTimeArrayModeVector();

                                                                    for(unsigned int i_fscoretamode=0; i_fscoretamode<vForecastScoreTimeArrayMode.size(); i_fscoretamode++)
                                                                    {
                                                                        params.SetForecastScoreTimeArrayMode(vForecastScoreTimeArrayMode[i_fscoretamode]);

                                                                        VectorDouble vForecastScoreTimeArrayDate = params.GetForecastScoreTimeArrayDateVector();

                                                                        for(unsigned int i_fscoretadate=0; i_fscoretadate<vForecastScoreTimeArrayDate.size(); i_fscoretadate++)
                                                                        {
                                                                            params.SetForecastScoreTimeArrayDate(vForecastScoreTimeArrayDate[i_fscoretadate]);

                                                                            VectorInt vForecastScoreTimeArrayIntervalDays = params.GetForecastScoreTimeArrayIntervalDaysVector();

                                                                            for(unsigned int i_fscoretaintdays=0; i_fscoretaintdays<vForecastScoreTimeArrayIntervalDays.size(); i_fscoretaintdays++)
                                                                            {
                                                                                params.SetForecastScoreTimeArrayIntervalDays(vForecastScoreTimeArrayIntervalDays[i_fscoretaintdays]);

                                                                                VectorFloat vForecastScorePostprocessDupliExp = params.GetForecastScorePostprocessDupliExpVector();

                                                                                for(unsigned int i_fscorepostdupliexp=0; i_fscorepostdupliexp<vForecastScorePostprocessDupliExp.size(); i_fscorepostdupliexp++)
                                                                                {
                                                                                    params.SetForecastScorePostprocessDupliExp(vForecastScorePostprocessDupliExp[i_fscorepostdupliexp]);

                                                                                    // Final forecast score
                                                                                    if(!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, i_step)) return false;

                                                                                    // Validate
                                                                                    Validate();

                                                                                    // Saving results
                                                                                    results_tested.Add(params, anaScoreFinal.GetForecastScore(), m_ScoreValid);
                                                                                    if(!results_tested.Print()) return false;
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Delete preloaded data
    DeletePreloadedData();

    return true;
}
