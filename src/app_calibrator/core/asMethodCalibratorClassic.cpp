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

#include "asMethodCalibratorClassic.h"

asMethodCalibratorClassic::asMethodCalibratorClassic()
:
asMethodCalibrator()
{

}

asMethodCalibratorClassic::~asMethodCalibratorClassic()
{

}

bool asMethodCalibratorClassic::Calibrate(asParametersCalibration &params)
{
    // Copy of the original parameters set.
    m_OriginalParams = params;

    // Extract the stations IDs
    VVectorInt stationsId = params.GetPredictandStationsIdsVector();

    // Create result object to save the final parameters sets
    asResultsParametersArray results_all;
    results_all.Init(_("all_station_best_parameters"));

    // Create a analogsdate object to save previous analogs dates selection.
    asResultsAnalogsDates anaDatesPrevious;

    for (unsigned int i_stat=0; i_stat<stationsId.size(); i_stat++)
    {
        VectorInt stationId = stationsId[i_stat];

        // Reset the score of the climatology
        m_ScoreClimatology.clear();

        // Create results objects
        asResultsAnalogsDates anaDates;
        asResultsAnalogsValues anaValues;
        asResultsAnalogsForecastScores anaScores;
        asResultsAnalogsForecastScoreFinal anaScoreFinal;

        // Clear previous results
        ClearAll();

        // Create result objects to save the parameters sets
        asResultsParametersArray results_tested;
        results_tested.Init(wxString::Format(_("station_%s_tested_parameters"), GetPredictandStationIdsList(stationId).c_str()));
        asResultsParametersArray results_best;
        results_best.Init(wxString::Format(_("station_%s_best_parameters"), GetPredictandStationIdsList(stationId).c_str()));
        wxString resultsXmlFilePath = wxFileConfig::Get()->Read("/Paths/CalibrationResultsDir", asConfig::GetDefaultUserWorkingDir());
        wxString time = asTime::GetStringTime(asTime::NowMJD(asLOCAL), concentrate);
        resultsXmlFilePath.Append(wxString::Format("/Calibration/%s_station_%s_best_parameters.xml", time.c_str(), GetPredictandStationIdsList(stationId).c_str()));

        // Create a complete relevance map
        asLogState(_("Calibration: creating the complete relevance map for a given predictor."));

        // Get a copy of the original parameters
        params = m_OriginalParams;

        // Set the next station ID
        params.SetPredictandStationIds(stationId);

        // Process every step one after the other
        int stepsNb = params.GetStepsNb();
        for (int i_step=0; i_step<stepsNb; i_step++)
        {
            // Restore previous best parameters
            if (i_step>0)
            {
                params = m_Parameters[0];
            }

            // Clear previous results
            ClearAll();

            // Set the same weight to every predictors
            int ptorsNb = params.GetPredictorsNb(i_step);
            float weight = (float)1/(float)(ptorsNb);
            for (int i_ptor=0; i_ptor<ptorsNb; i_ptor++)
            {
                params.SetPredictorWeight(i_step, i_ptor, weight);
            }

            // Get spatial boundaries
            double predictorUminLowerLimit = params.GetPredictorUminLowerLimit(i_step, 0);
            double predictorUminUpperLimit = params.GetPredictorUminUpperLimit(i_step, 0);
            double predictorUminIteration = params.GetPredictorUminIteration(i_step, 0);
            int predictorUptsnbIteration = params.GetPredictorUptsnbIteration(i_step, 0);
            int predictorUptsnbLowerLimit = params.GetPredictorUptsnbLowerLimit(i_step, 0);
            int predictorUptsnbUpperLimit = params.GetPredictorUptsnbUpperLimit(i_step, 0);
            double predictorVminLowerLimit = params.GetPredictorVminLowerLimit(i_step, 0);
            double predictorVminUpperLimit = params.GetPredictorVminUpperLimit(i_step, 0);
            double predictorVminIteration = params.GetPredictorVminIteration(i_step, 0);
            int predictorVptsnbIteration = params.GetPredictorVptsnbIteration(i_step, 0);
            int predictorVptsnbLowerLimit = params.GetPredictorVptsnbLowerLimit(i_step, 0);
            int predictorVptsnbUpperLimit = params.GetPredictorVptsnbUpperLimit(i_step, 0);

            for (int i_ptor=0; i_ptor<ptorsNb; i_ptor++)
            {
                predictorUminLowerLimit = wxMax(predictorUminLowerLimit, params.GetPredictorUminLowerLimit(i_step, i_ptor));
                predictorUminUpperLimit = wxMin(predictorUminUpperLimit, params.GetPredictorUminUpperLimit(i_step, i_ptor));
                predictorUminIteration = wxMin(predictorUminIteration, params.GetPredictorUminIteration(i_step, i_ptor));
                predictorUptsnbIteration = wxMin(predictorUptsnbIteration, params.GetPredictorUptsnbIteration(i_step, i_ptor));
                predictorUptsnbLowerLimit = wxMax(predictorUptsnbLowerLimit, params.GetPredictorUptsnbLowerLimit(i_step, i_ptor));
                predictorUptsnbUpperLimit = wxMin(predictorUptsnbUpperLimit, params.GetPredictorUptsnbUpperLimit(i_step, i_ptor));
                predictorVminLowerLimit = wxMax(predictorVminLowerLimit, params.GetPredictorVminLowerLimit(i_step, i_ptor));
                predictorVminUpperLimit = wxMin(predictorVminUpperLimit, params.GetPredictorVminUpperLimit(i_step, i_ptor));
                predictorVminIteration = wxMin(predictorVminIteration, params.GetPredictorVminIteration(i_step, i_ptor));
                predictorVptsnbIteration = wxMin(predictorVptsnbIteration, params.GetPredictorVptsnbIteration(i_step, i_ptor));
                predictorVptsnbLowerLimit = wxMax(predictorVptsnbLowerLimit, params.GetPredictorVptsnbLowerLimit(i_step, i_ptor));
                predictorVptsnbUpperLimit = wxMax(predictorVptsnbUpperLimit, params.GetPredictorVptsnbUpperLimit(i_step, i_ptor));
            }

            if (predictorUminIteration==0) predictorUminIteration = 1;
            if (predictorVminIteration==0) predictorVminIteration = 1;

            // Set the minimal size
            for (int i_ptor=0; i_ptor<ptorsNb; i_ptor++)
            {
                if (params.GetPredictorFlatAllowed(i_step, i_ptor))
                {
                    params.SetPredictorUptsnb(i_step, i_ptor, 1);
                    params.SetPredictorVptsnb(i_step, i_ptor, 1);
                }
                else
                {
                    params.SetPredictorUptsnb(i_step, i_ptor, predictorUptsnbIteration+1);
                    params.SetPredictorVptsnb(i_step, i_ptor, predictorVptsnbIteration+1);
                }
            }

            // Set the same analogs number at the step level and at the score level.
            int initalAnalogsNb = 0;
            VectorInt initalAnalogsNbVect = params.GetAnalogsNumberVector(i_step);
            if(initalAnalogsNbVect.size()>1)
            {
                int indexAnb = floor(initalAnalogsNbVect.size()/2.0);
                initalAnalogsNb = initalAnalogsNbVect[indexAnb]; // Take the median
            }
            else
            {
                initalAnalogsNb = initalAnalogsNbVect[0];
            }
            for (int i=i_step; i<stepsNb; i++) // For the current step and the next ones
            {
                params.SetAnalogsNumber(i, initalAnalogsNb);
            }
            params.FixAnalogsNb();

            // Build map to explore
            ClearTemp();

            for (int i_u=0; i_u<=((predictorUminUpperLimit-predictorUminLowerLimit)/(predictorUminIteration)); i_u++)
            {
                for (int i_v=0; i_v<=((predictorVminUpperLimit-predictorVminLowerLimit)/(predictorVminIteration)); i_v++)
                {
                    double u = predictorUminLowerLimit+(predictorUminIteration)*i_u;
                    double v = predictorVminLowerLimit+(predictorVminIteration)*i_v;

                    for (int i_ptor=0; i_ptor<ptorsNb; i_ptor++)
                    {
                        params.SetPredictorUmin(i_step, i_ptor, u);
                        params.SetPredictorVmin(i_step, i_ptor, v);

                        // Fixes and checks
                        params.FixWeights();
                        params.FixCoordinates();
                    }

                    m_ParametersTemp.push_back(params);
                }
            }

            // Process the relevance map
            asLogState(wxString::Format(_("Calibration: processing the relevance map for all the predictors of step %d (station %s)."), i_step, GetPredictandStationIdsList(stationId).c_str()));
            for (unsigned int i_param=0; i_param<m_ParametersTemp.size(); i_param++)
            {
                bool containsNaNs = false;
                if (i_step==0)
                {
                    if(!GetAnalogsDates(anaDates, m_ParametersTemp[i_param], i_step, containsNaNs)) return false;
                }
                else
                {
                    if(!GetAnalogsSubDates(anaDates, m_ParametersTemp[i_param], anaDatesPrevious, i_step, containsNaNs)) return false;
                }
                if (containsNaNs)
                {
                    asLogError(_("The dates selection contains NaNs"));
                    return false;
                }
                if(!GetAnalogsValues(anaValues, m_ParametersTemp[i_param], anaDates, i_step)) return false;
                if(!GetAnalogsForecastScores(anaScores, m_ParametersTemp[i_param], anaValues, i_step)) return false;
                if(!GetAnalogsForecastScoreFinal(anaScoreFinal, m_ParametersTemp[i_param], anaScores, i_step)) return false;

                // Store the result
                m_ScoresCalibTemp.push_back(anaScoreFinal.GetForecastScore());
                results_tested.Add(m_ParametersTemp[i_param],anaScoreFinal.GetForecastScore());
            }

            // Keep the best parameter set
            wxASSERT(m_ParametersTemp.size()>0);
            PushBackBestTemp();
            ClearTemp();

            // Resize domain
            asLogState(wxString::Format(_("Calibration: resize the spatial domain for every predictor (station %s)."), GetPredictandStationIdsList(stationId).c_str()));

            bool isover = false;
            while (!isover)
            {
                double utmp, vtmp;
                int uptsnbtmp, vptsnbtmp;
                isover = true;

                ClearTemp();

                for (int i_resizing=0; i_resizing<4; i_resizing++)
                {
                    // Consider the best point in previous iteration
                    params = m_Parameters[0];

                    for (int i_ptor=0; i_ptor<ptorsNb; i_ptor++)
                    {
                        switch (i_resizing)
                        {
                            case 0:
                                // Enlarge top
                                vptsnbtmp = params.GetPredictorVptsnb(i_step, i_ptor)+predictorVptsnbIteration;
                                vptsnbtmp = wxMax(wxMin(vptsnbtmp, predictorVptsnbUpperLimit), predictorVptsnbLowerLimit);
                                params.SetPredictorVptsnb(i_step, i_ptor, vptsnbtmp);
                                break;
                            case 1:
                                // Enlarge right
                                uptsnbtmp = params.GetPredictorUptsnb(i_step, i_ptor)+predictorUptsnbIteration;
                                uptsnbtmp = wxMax(wxMin(uptsnbtmp, predictorUptsnbUpperLimit), predictorUptsnbLowerLimit);
                                params.SetPredictorUptsnb(i_step, i_ptor, uptsnbtmp);
                                break;
                            case 2:
                                // Enlarge bottom
                                vptsnbtmp = params.GetPredictorVptsnb(i_step, i_ptor)+predictorVptsnbIteration;
                                vptsnbtmp = wxMax(wxMin(vptsnbtmp, predictorVptsnbUpperLimit), predictorVptsnbLowerLimit);
                                params.SetPredictorVptsnb(i_step, i_ptor, vptsnbtmp);
                                vtmp = params.GetPredictorVmin(i_step, i_ptor)-predictorVminIteration;
                                vtmp = wxMax(wxMin(vtmp, predictorVminUpperLimit), predictorVminLowerLimit);
                                params.SetPredictorVmin(i_step, i_ptor, vtmp);
                                break;
                            case 3:
                                // Enlarge left
                                uptsnbtmp = params.GetPredictorUptsnb(i_step, i_ptor)+predictorUptsnbIteration;
                                uptsnbtmp = wxMax(wxMin(uptsnbtmp, predictorUptsnbUpperLimit), predictorUptsnbLowerLimit);
                                params.SetPredictorUptsnb(i_step, i_ptor, uptsnbtmp);
                                utmp = params.GetPredictorUmin(i_step, i_ptor)-predictorUminIteration;
                                utmp = wxMax(wxMin(utmp, predictorUminUpperLimit), predictorUminLowerLimit);
                                params.SetPredictorUmin(i_step, i_ptor, utmp);
                                break;
                            default:
                                asLogError(_("Resizing not correctly defined."));
                        }
                    }

                    // Fixes and checks
                    params.FixWeights();
                    params.FixCoordinates();

                    // Assess parameters
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
                    if(!GetAnalogsValues(anaValues, params, anaDates, i_step)) return false;
                    if(!GetAnalogsForecastScores(anaScores, params, anaValues, i_step)) return false;
                    if(!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, i_step)) return false;
                    results_tested.Add(params,anaScoreFinal.GetForecastScore());

                    // If better, store it and try again to resize.
                    if (PushBackInTempIfBetter(params, anaScoreFinal))
                    {
                        isover = false;
                    }
                }

                // Apply the resizing that provides the best improvement
                if (m_ParametersTemp.size()>0)
                {
                    KeepBestTemp();
                }
            }

            // Consider the best point in previous iteration
            params = m_Parameters[0];

            // Keep the analogs dates of the best parameters set
            bool containsNaNs = false;
            if (i_step==0)
            {
                if(!GetAnalogsDates(anaDatesPrevious, params, i_step, containsNaNs)) return false;
            }
            else if (i_step<stepsNb)
            {
                asResultsAnalogsDates anaDatesPreviousNew;
                if(!GetAnalogsSubDates(anaDatesPreviousNew, params, anaDatesPrevious, i_step, containsNaNs)) return false;
                anaDatesPrevious = anaDatesPreviousNew;
            }
            if (containsNaNs)
            {
                asLogError(_("The dates selection contains NaNs"));
                return false;
            }
        }

        // Finally calibrate the number of analogs for every step
        asLogState(_("Calibration: find the analogs number for every step."));
        ClearTemp();
        asResultsAnalogsDates tempDates;
        if(!SubProcessAnalogsNumber(params, tempDates)) return false;

        // Extract intermediate results from temporary vectors
        for (unsigned int i_res=0; i_res<m_ParametersTemp.size(); i_res++)
        {
            results_tested.Add(m_ParametersTemp[i_res],m_ScoresCalibTemp[i_res]);
        }
        results_tested.Print();

        // Keep the best parameter set
        wxASSERT(m_ParametersTemp.size()>0);
        KeepBestTemp();
        ClearTemp();

        // Validate
        Validate();

        // Keep the best parameters set
        SetBestParameters(results_best);
        if(!results_best.Print()) return false;
        results_all.Add(m_Parameters[0],m_ScoresCalib[0],m_ScoreValid);
        if(!results_all.Print()) return false;
        if(!results_all.Print()) return false;
        if(!m_Parameters[0].GenerateSimpleParametersFile(resultsXmlFilePath)) return false;
    }

    return true;
}
