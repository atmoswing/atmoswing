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
 
#include "asMethodCalibratorClassicPlus.h"

asMethodCalibratorClassicPlus::asMethodCalibratorClassicPlus()
:
asMethodCalibrator()
{

}

asMethodCalibratorClassicPlus::~asMethodCalibratorClassicPlus()
{

}

bool asMethodCalibratorClassicPlus::Calibrate(asParametersCalibration &params)
{
    // Copy of the original parameters set.
    m_OriginalParams = params;

    // Get preferences
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    int stepsLatPertinenceMap = 0;
    pConfig->Read("/Calibration/ClassicPlus/StepsLatPertinenceMap", &stepsLatPertinenceMap, 2);
    if(stepsLatPertinenceMap<1) stepsLatPertinenceMap = 1;
    int stepsLonPertinenceMap = 0;
    pConfig->Read("/Calibration/ClassicPlus/StepsLonPertinenceMap", &stepsLonPertinenceMap, 2);
    if(stepsLonPertinenceMap<1) stepsLonPertinenceMap = 1;
    int resizingIterations = 1;
    pConfig->Read("/Calibration/ClassicPlus/ResizingIterations", &resizingIterations, 1);
    if(resizingIterations<1) resizingIterations = 1;
    bool proceedSequentially = true;
    pConfig->Read("/Calibration/ClassicPlus/ProceedSequentially", &proceedSequentially, true);
    ThreadsManager().CritSectionConfig().Leave();

    // Extract the stations IDs
    VVectorInt stationsId = params.GetPredictandStationsIdsVector();

    // Preload data
    try
    {
        if (!PreloadData(params))
        {
            asLogError(_("Could not preload the data."));
            return false;
        }
    }
    catch(bad_alloc& ba)
    {
        wxString msg(ba.what(), wxConvUTF8);
        asLogError(wxString::Format(_("Bad allocation caught in the data preloading: %s"), msg.c_str()));
        DeletePreloadedData();
        return false;
    }
    catch (exception& e)
    {
        wxString msg(e.what(), wxConvUTF8);
        asLogError(wxString::Format(_("Exception in the data preloading: %s"), msg.c_str()));
        DeletePreloadedData();
        return false;
    }

    // Create result object to save the final parameters sets
    asResultsParametersArray results_all;
    results_all.Init(_("all_station_best_parameters"));

    // Create a analogsdate object to save previous analogs dates selection.
    asResultsAnalogsDates anaDatesPrevious;
    asResultsAnalogsDates anaDatesPreviousSubRuns;

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
        results_tested.Init(wxString::Format(_("station_%d_tested_parameters"), stationId));
        asResultsParametersArray results_best;
        results_best.Init(wxString::Format(_("station_%d_best_parameters"), stationId));
        wxString resultsXmlFilePath = wxFileConfig::Get()->Read("/StandardPaths/CalibrationResultsDir", asConfig::GetDefaultUserWorkingDir());
        wxString time = asTime::GetStringTime(asTime::NowMJD(asLOCAL), concentrate);
        resultsXmlFilePath.Append(wxString::Format("/Calibration/%s_station_%d_best_parameters.xml", time.c_str(), stationId));

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

            // For the current step
            params.SetAnalogsNumber(i_step, initalAnalogsNb);
            // And the next ones
            if (proceedSequentially)
            {
                for (int i=i_step; i<stepsNb; i++)
                {
                    params.SetAnalogsNumber(i, initalAnalogsNb);
                }
                params.SetForecastScoreAnalogsNumber(initalAnalogsNb);
            }
            params.FixAnalogsNb();

            if (predictorUminIteration!=0 || predictorUptsnbIteration!=0 || predictorVminIteration!=0 || predictorVptsnbIteration!=0)
            {
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
                        params.SetPredictorUptsnb(i_step, i_ptor, predictorUptsnbLowerLimit);
                        params.SetPredictorVptsnb(i_step, i_ptor, predictorVptsnbLowerLimit);
                    }
                }

                wxStopWatch swMap;

                // Build map to explore
                ClearTemp();

                for (int i_u=0; i_u<=((predictorUminUpperLimit-predictorUminLowerLimit)/(predictorUminIteration*stepsLonPertinenceMap)); i_u++)
                {
                    for (int i_v=0; i_v<=((predictorVminUpperLimit-predictorVminLowerLimit)/(predictorVminIteration*stepsLatPertinenceMap)); i_v++)
                    {
                        double u = predictorUminLowerLimit+(predictorUminIteration*stepsLonPertinenceMap)*i_u;
                        double v = predictorVminLowerLimit+(predictorVminIteration*stepsLatPertinenceMap)*i_v;
// TODO (phorton#5#): This is not exact for unregular grids... Improve this approach !
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
                asLogState(wxString::Format(_("Calibration: processing the relevance map for all the predictors of step %d (station %d)."), i_step, stationId));
                for (unsigned int i_param=0; i_param<m_ParametersTemp.size(); i_param++)
                {
                    if (proceedSequentially)
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
                            m_ScoresCalibTemp.push_back(NaNFloat);
                            continue;
                        }
                        if(!GetAnalogsValues(anaValues, m_ParametersTemp[i_param], anaDates, i_step)) return false;
                        if(!GetAnalogsForecastScores(anaScores, m_ParametersTemp[i_param], anaValues, i_step)) return false;
                        if(!GetAnalogsForecastScoreFinal(anaScoreFinal, m_ParametersTemp[i_param], anaScores, i_step)) return false;
                    }
                    else
                    {
                        bool continueLoop = true;
                        anaDatesPreviousSubRuns = anaDatesPrevious;
                        for (int sub_step=i_step; sub_step<stepsNb; sub_step++)
                        {
                            asLogMessage(wxString::Format(_("Process sub-level %d"), sub_step));
                            bool containsNaNs = false;
                            if (sub_step==0)
                            {
                                if(!GetAnalogsDates(anaDates, m_ParametersTemp[i_param], sub_step, containsNaNs)) return false;
                            }
                            else
                            {
                                if(!GetAnalogsSubDates(anaDates, m_ParametersTemp[i_param], anaDatesPreviousSubRuns, sub_step, containsNaNs)) return false;
                            }
                            if (containsNaNs)
                            {
                                continueLoop = false;
                                m_ScoresCalibTemp.push_back(NaNFloat);
                                continue;
                            }
                            anaDatesPreviousSubRuns = anaDates;
                        }
                        if (continueLoop)
                        {
                            if(!GetAnalogsValues(anaValues, m_ParametersTemp[i_param], anaDates, stepsNb-1)) return false;
                            if(!GetAnalogsForecastScores(anaScores, m_ParametersTemp[i_param], anaValues, stepsNb-1)) return false;
                            if(!GetAnalogsForecastScoreFinal(anaScoreFinal, m_ParametersTemp[i_param], anaScores, stepsNb-1)) return false;
                        }
                    }

                    // Store the result
                    m_ScoresCalibTemp.push_back(anaScoreFinal.GetForecastScore());
                    results_tested.Add(m_ParametersTemp[i_param],anaScoreFinal.GetForecastScore());
                }

                asLogMessageImportant(wxString::Format(_("Time to process the relevance map: %ldms"), swMap.Time()));

                // Keep the best parameter set
                wxASSERT(m_ParametersTemp.size()>0);
                RemoveNaNsInTemp();
                PushBackBestTemp();
                wxASSERT(m_Parameters.size()==1);
                ClearTemp();

                asLogMessageImportant(wxString::Format(_("Best point on relevance map: %.2f lat, %.2f lon"), m_Parameters[m_Parameters.size()-1].GetPredictorVmin(i_step, 0), m_Parameters[m_Parameters.size()-1].GetPredictorUmin(i_step, 0)));

                // Resize domain
                asLogState(wxString::Format(_("Calibration: resize the spatial domain for every predictor (station %d)."), stationId));

                wxStopWatch swEnlarge;

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
                        if (proceedSequentially)
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
                                isover = false;
                                continue;
                            }
                            if(!GetAnalogsValues(anaValues, params, anaDates, i_step)) return false;
                            if(!GetAnalogsForecastScores(anaScores, params, anaValues, i_step)) return false;
                            if(!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, i_step)) return false;
                        }
                        else
                        {
                            bool continueLoop = true;
                            anaDatesPreviousSubRuns = anaDatesPrevious;
                            for (int sub_step=i_step; sub_step<stepsNb; sub_step++)
                            {
                                asLogMessage(wxString::Format(_("Process sub-level %d"), sub_step));
                                bool containsNaNs = false;
                                if (sub_step==0)
                                {
                                    if(!GetAnalogsDates(anaDates, params, sub_step, containsNaNs)) return false;
                                }
                                else
                                {
                                    if(!GetAnalogsSubDates(anaDates, params, anaDatesPreviousSubRuns, sub_step, containsNaNs)) return false;
                                }
                                if (containsNaNs)
                                {
                                    continueLoop = false;
                                    isover = false;
                                    continue;
                                }
                                anaDatesPreviousSubRuns = anaDates;
                            }
                            if (continueLoop)
                            {
                                if(!GetAnalogsValues(anaValues, params, anaDates, stepsNb-1)) return false;
                                if(!GetAnalogsForecastScores(anaScores, params, anaValues, stepsNb-1)) return false;
                                if(!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, stepsNb-1)) return false;
                            }
                        }

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

                asLogMessageImportant(wxString::Format(_("Time to process the first resizing procedure: %ldms"), swEnlarge.Time()));

                // Resize domain
                asLogState(wxString::Format(_("Calibration: reshape again the spatial domain for every predictor (station %d)."), stationId));

                // Try other moves. No while loop but reinitialize the for loops
                double utmp, vtmp;
                int uptsnbtmp, vptsnbtmp;

                ClearTemp();

                wxStopWatch swResize;

                for (int multipleFactor=1; multipleFactor<=resizingIterations; multipleFactor++)
                {
                    for (int i_resizing=0; i_resizing<22; i_resizing++)
                    {
                        // Consider the best point in previous iteration
                        params = m_Parameters[0];

                        for (int i_ptor=0; i_ptor<ptorsNb; i_ptor++)
                        {
                            switch (i_resizing)
                            {
                                case 0:
                                    // Enlarge all
                                    vptsnbtmp = params.GetPredictorVptsnb(i_step, i_ptor)+2*multipleFactor*predictorVptsnbIteration;
                                    vptsnbtmp = wxMax(wxMin(vptsnbtmp, predictorVptsnbUpperLimit), predictorVptsnbLowerLimit);
                                    uptsnbtmp = params.GetPredictorUptsnb(i_step, i_ptor)+2*multipleFactor*predictorUptsnbIteration;
                                    uptsnbtmp = wxMax(wxMin(uptsnbtmp, predictorUptsnbUpperLimit), predictorUptsnbLowerLimit);
                                    params.SetPredictorVptsnb(i_step, i_ptor, vptsnbtmp);
                                    params.SetPredictorUptsnb(i_step, i_ptor, uptsnbtmp);
                                    vtmp = params.GetPredictorVmin(i_step, i_ptor)-multipleFactor*predictorVminIteration;
                                    vtmp = wxMax(wxMin(vtmp, predictorVminUpperLimit), predictorVminLowerLimit);
                                    utmp = params.GetPredictorUmin(i_step, i_ptor)-multipleFactor*predictorUminIteration;
                                    utmp = wxMax(wxMin(utmp, predictorUminUpperLimit), predictorUminLowerLimit);
                                    params.SetPredictorVmin(i_step, i_ptor, vtmp);
                                    params.SetPredictorUmin(i_step, i_ptor, utmp);
                                    break;
                                case 1:
                                    // Reduce all
                                    vptsnbtmp = params.GetPredictorVptsnb(i_step, i_ptor)-2*multipleFactor*predictorVptsnbIteration;
                                    vptsnbtmp = wxMax(wxMin(vptsnbtmp, predictorVptsnbUpperLimit), predictorVptsnbLowerLimit);
                                    uptsnbtmp = params.GetPredictorUptsnb(i_step, i_ptor)-2*multipleFactor*predictorUptsnbIteration;
                                    uptsnbtmp = wxMax(wxMin(uptsnbtmp, predictorUptsnbUpperLimit), predictorUptsnbLowerLimit);
                                    params.SetPredictorVptsnb(i_step, i_ptor, vptsnbtmp);
                                    params.SetPredictorUptsnb(i_step, i_ptor, uptsnbtmp);
                                    vtmp = params.GetPredictorVmin(i_step, i_ptor)+multipleFactor*predictorVminIteration;
                                    vtmp = wxMax(wxMin(vtmp, predictorVminUpperLimit), predictorVminLowerLimit);
                                    utmp = params.GetPredictorUmin(i_step, i_ptor)+multipleFactor*predictorUminIteration;
                                    utmp = wxMax(wxMin(utmp, predictorUminUpperLimit), predictorUminLowerLimit);
                                    params.SetPredictorVmin(i_step, i_ptor, vtmp);
                                    params.SetPredictorUmin(i_step, i_ptor, utmp);
                                    break;
                                case 2:
                                    // Reduce top
                                    vptsnbtmp = params.GetPredictorVptsnb(i_step, i_ptor)-multipleFactor*predictorVptsnbIteration;
                                    vptsnbtmp = wxMax(wxMin(vptsnbtmp, predictorVptsnbUpperLimit), predictorVptsnbLowerLimit);
                                    params.SetPredictorVptsnb(i_step, i_ptor, vptsnbtmp);
                                    break;
                                case 3:
                                    // Reduce right
                                    uptsnbtmp = params.GetPredictorUptsnb(i_step, i_ptor)-multipleFactor*predictorUptsnbIteration;
                                    uptsnbtmp = wxMax(wxMin(uptsnbtmp, predictorUptsnbUpperLimit), predictorUptsnbLowerLimit);
                                    params.SetPredictorUptsnb(i_step, i_ptor, uptsnbtmp);
                                    break;
                                case 4:
                                    // Reduce bottom
                                    vptsnbtmp = params.GetPredictorVptsnb(i_step, i_ptor)-multipleFactor*predictorVptsnbIteration;
                                    vptsnbtmp = wxMax(wxMin(vptsnbtmp, predictorVptsnbUpperLimit), predictorVptsnbLowerLimit);
                                    params.SetPredictorVptsnb(i_step, i_ptor, vptsnbtmp);
                                    vtmp = params.GetPredictorVmin(i_step, i_ptor)+multipleFactor*predictorVminIteration;
                                    vtmp = wxMax(wxMin(vtmp, predictorVminUpperLimit), predictorVminLowerLimit);
                                    params.SetPredictorVmin(i_step, i_ptor, vtmp);
                                    break;
                                case 5:
                                    // Reduce left
                                    uptsnbtmp = params.GetPredictorUptsnb(i_step, i_ptor)-multipleFactor*predictorUptsnbIteration;
                                    uptsnbtmp = wxMax(wxMin(uptsnbtmp, predictorUptsnbUpperLimit), predictorUptsnbLowerLimit);
                                    params.SetPredictorUptsnb(i_step, i_ptor, uptsnbtmp);
                                    utmp = params.GetPredictorUmin(i_step, i_ptor)+multipleFactor*predictorUminIteration;
                                    utmp = wxMax(wxMin(utmp, predictorUminUpperLimit), predictorUminLowerLimit);
                                    params.SetPredictorUmin(i_step, i_ptor, utmp);
                                    break;
                                case 6:
                                    // Reduce top & bottom
                                    vptsnbtmp = params.GetPredictorVptsnb(i_step, i_ptor)-2*multipleFactor*predictorVptsnbIteration;
                                    vptsnbtmp = wxMax(wxMin(vptsnbtmp, predictorVptsnbUpperLimit), predictorVptsnbLowerLimit);
                                    params.SetPredictorVptsnb(i_step, i_ptor, vptsnbtmp);
                                    vtmp = params.GetPredictorVmin(i_step, i_ptor)+multipleFactor*predictorVminIteration;
                                    vtmp = wxMax(wxMin(vtmp, predictorVminUpperLimit), predictorVminLowerLimit);
                                    params.SetPredictorVmin(i_step, i_ptor, vtmp);
                                    break;
                                case 7:
                                    // Reduce right & left
                                    uptsnbtmp = params.GetPredictorUptsnb(i_step, i_ptor)-2*multipleFactor*predictorUptsnbIteration;
                                    uptsnbtmp = wxMax(wxMin(uptsnbtmp, predictorUptsnbUpperLimit), predictorUptsnbLowerLimit);
                                    params.SetPredictorUptsnb(i_step, i_ptor, uptsnbtmp);
                                    utmp = params.GetPredictorUmin(i_step, i_ptor)+multipleFactor*predictorUminIteration;
                                    utmp = wxMax(wxMin(utmp, predictorUminUpperLimit), predictorUminLowerLimit);
                                    params.SetPredictorUmin(i_step, i_ptor, utmp);
                                    break;
                                case 8:
                                    // Enlarge top
                                    vptsnbtmp = params.GetPredictorVptsnb(i_step, i_ptor)+multipleFactor*predictorVptsnbIteration;
                                    vptsnbtmp = wxMax(wxMin(vptsnbtmp, predictorVptsnbUpperLimit), predictorVptsnbLowerLimit);
                                    params.SetPredictorVptsnb(i_step, i_ptor, vptsnbtmp);
                                    break;
                                case 9:
                                    // Enlarge right
                                    uptsnbtmp = params.GetPredictorUptsnb(i_step, i_ptor)+multipleFactor*predictorUptsnbIteration;
                                    uptsnbtmp = wxMax(wxMin(uptsnbtmp, predictorUptsnbUpperLimit), predictorUptsnbLowerLimit);
                                    params.SetPredictorUptsnb(i_step, i_ptor, uptsnbtmp);
                                    break;
                                case 10:
                                    // Enlarge bottom
                                    vptsnbtmp = params.GetPredictorVptsnb(i_step, i_ptor)+multipleFactor*predictorVptsnbIteration;
                                    vptsnbtmp = wxMax(wxMin(vptsnbtmp, predictorVptsnbUpperLimit), predictorVptsnbLowerLimit);
                                    params.SetPredictorVptsnb(i_step, i_ptor, vptsnbtmp);
                                    vtmp = params.GetPredictorVmin(i_step, i_ptor)-multipleFactor*predictorVminIteration;
                                    vtmp = wxMax(wxMin(vtmp, predictorVminUpperLimit), predictorVminLowerLimit);
                                    params.SetPredictorVmin(i_step, i_ptor, vtmp);
                                    break;
                                case 11:
                                    // Enlarge left
                                    uptsnbtmp = params.GetPredictorUptsnb(i_step, i_ptor)+multipleFactor*predictorUptsnbIteration;
                                    uptsnbtmp = wxMax(wxMin(uptsnbtmp, predictorUptsnbUpperLimit), predictorUptsnbLowerLimit);
                                    params.SetPredictorUptsnb(i_step, i_ptor, uptsnbtmp);
                                    utmp = params.GetPredictorUmin(i_step, i_ptor)-multipleFactor*predictorUminIteration;
                                    utmp = wxMax(wxMin(utmp, predictorUminUpperLimit), predictorUminLowerLimit);
                                    params.SetPredictorUmin(i_step, i_ptor, utmp);
                                    break;
                                case 12:
                                    // Enlarge top & bottom
                                    vptsnbtmp = params.GetPredictorVptsnb(i_step, i_ptor)+2*multipleFactor*predictorVptsnbIteration;
                                    vptsnbtmp = wxMax(wxMin(vptsnbtmp, predictorVptsnbUpperLimit), predictorVptsnbLowerLimit);
                                    params.SetPredictorVptsnb(i_step, i_ptor, vptsnbtmp);
                                    vtmp = params.GetPredictorVmin(i_step, i_ptor)-multipleFactor*predictorVminIteration;
                                    vtmp = wxMax(wxMin(vtmp, predictorVminUpperLimit), predictorVminLowerLimit);
                                    params.SetPredictorVmin(i_step, i_ptor, vtmp);
                                    break;
                                case 13:
                                    // Enlarge right & left
                                    uptsnbtmp = params.GetPredictorUptsnb(i_step, i_ptor)+2*multipleFactor*predictorUptsnbIteration;
                                    uptsnbtmp = wxMax(wxMin(uptsnbtmp, predictorUptsnbUpperLimit), predictorUptsnbLowerLimit);
                                    params.SetPredictorUptsnb(i_step, i_ptor, uptsnbtmp);
                                    utmp = params.GetPredictorUmin(i_step, i_ptor)-multipleFactor*predictorUminIteration;
                                    utmp = wxMax(wxMin(utmp, predictorUminUpperLimit), predictorUminLowerLimit);
                                    params.SetPredictorUmin(i_step, i_ptor, utmp);
                                    break;
                                case 14:
                                    // Move top
                                    vtmp = params.GetPredictorVmin(i_step, i_ptor)+multipleFactor*predictorVminIteration;
                                    vtmp = wxMax(wxMin(vtmp, predictorVminUpperLimit), predictorVminLowerLimit);
                                    params.SetPredictorVmin(i_step, i_ptor, vtmp);
                                    break;
                                case 15:
                                    // Move right
                                    utmp = params.GetPredictorUmin(i_step, i_ptor)+multipleFactor*predictorUminIteration;
                                    utmp = wxMax(wxMin(utmp, predictorUminUpperLimit), predictorUminLowerLimit);
                                    params.SetPredictorUmin(i_step, i_ptor, utmp);
                                    break;
                                case 16:
                                    // Move bottom
                                    vtmp = params.GetPredictorVmin(i_step, i_ptor)-multipleFactor*predictorVminIteration;
                                    vtmp = wxMax(wxMin(vtmp, predictorVminUpperLimit), predictorVminLowerLimit);
                                    params.SetPredictorVmin(i_step, i_ptor, vtmp);
                                    break;
                                case 17:
                                    // Move left
                                    utmp = params.GetPredictorUmin(i_step, i_ptor)-multipleFactor*predictorUminIteration;
                                    utmp = wxMax(wxMin(utmp, predictorUminUpperLimit), predictorUminLowerLimit);
                                    params.SetPredictorUmin(i_step, i_ptor, utmp);
                                    break;
                                case 18:
                                    // Move top-left
                                    vtmp = params.GetPredictorVmin(i_step, i_ptor)+multipleFactor*predictorVminIteration;
                                    vtmp = wxMax(wxMin(vtmp, predictorVminUpperLimit), predictorVminLowerLimit);
                                    params.SetPredictorVmin(i_step, i_ptor, vtmp);
                                    utmp = params.GetPredictorUmin(i_step, i_ptor)-multipleFactor*predictorUminIteration;
                                    utmp = wxMax(wxMin(utmp, predictorUminUpperLimit), predictorUminLowerLimit);
                                    params.SetPredictorUmin(i_step, i_ptor, utmp);
                                    break;
                                case 19:
                                    // Move top-right
                                    vtmp = params.GetPredictorVmin(i_step, i_ptor)+multipleFactor*predictorVminIteration;
                                    vtmp = wxMax(wxMin(vtmp, predictorVminUpperLimit), predictorVminLowerLimit);
                                    params.SetPredictorVmin(i_step, i_ptor, vtmp);
                                    utmp = params.GetPredictorUmin(i_step, i_ptor)+multipleFactor*predictorUminIteration;
                                    utmp = wxMax(wxMin(utmp, predictorUminUpperLimit), predictorUminLowerLimit);
                                    params.SetPredictorUmin(i_step, i_ptor, utmp);
                                    break;
                                case 20:
                                    // Move bottom-left
                                    vtmp = params.GetPredictorVmin(i_step, i_ptor)-multipleFactor*predictorVminIteration;
                                    vtmp = wxMax(wxMin(vtmp, predictorVminUpperLimit), predictorVminLowerLimit);
                                    params.SetPredictorVmin(i_step, i_ptor, vtmp);
                                    utmp = params.GetPredictorUmin(i_step, i_ptor)-multipleFactor*predictorUminIteration;
                                    utmp = wxMax(wxMin(utmp, predictorUminUpperLimit), predictorUminLowerLimit);
                                    params.SetPredictorUmin(i_step, i_ptor, utmp);
                                    break;
                                case 21:
                                    // Move bottom-right
                                    vtmp = params.GetPredictorVmin(i_step, i_ptor)-multipleFactor*predictorVminIteration;
                                    vtmp = wxMax(wxMin(vtmp, predictorVminUpperLimit), predictorVminLowerLimit);
                                    params.SetPredictorVmin(i_step, i_ptor, vtmp);
                                    utmp = params.GetPredictorUmin(i_step, i_ptor)+multipleFactor*predictorUminIteration;
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
                        if (proceedSequentially)
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
                                continue;
                            }
                            if(!GetAnalogsValues(anaValues, params, anaDates, i_step)) return false;
                            if(!GetAnalogsForecastScores(anaScores, params, anaValues, i_step)) return false;
                            if(!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, i_step)) return false;
                        }
                        else
                        {
                            bool continueLoop = true;
                            anaDatesPreviousSubRuns = anaDatesPrevious;
                            for (int sub_step=i_step; sub_step<stepsNb; sub_step++)
                            {
                                asLogMessage(wxString::Format(_("Process sub-level %d"), sub_step));
                                bool containsNaNs = false;
                                if (sub_step==0)
                                {
                                    if(!GetAnalogsDates(anaDates, params, sub_step, containsNaNs)) return false;
                                }
                                else
                                {
                                    if(!GetAnalogsSubDates(anaDates, params, anaDatesPreviousSubRuns, sub_step, containsNaNs)) return false;
                                }
                                if (containsNaNs)
                                {
                                    continueLoop = false;
                                    continue;
                                }
                                anaDatesPreviousSubRuns = anaDates;
                            }
                            if(continueLoop)
                            {
                                if(!GetAnalogsValues(anaValues, params, anaDates, stepsNb-1)) return false;
                                if(!GetAnalogsForecastScores(anaScores, params, anaValues, stepsNb-1)) return false;
                                if(!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, stepsNb-1)) return false;
                            }
                        }

                        results_tested.Add(params,anaScoreFinal.GetForecastScore());

                        // If better, keep it and start again
                        if (KeepIfBetter(params, anaScoreFinal))
                        {
                            asLogMessageImportant(wxString::Format("Improved spatial window size and position (move %d, factor %d)", i_resizing, multipleFactor));
                            i_resizing = 0;
                            multipleFactor = 1;
                        }
                    }
                }

                asLogMessageImportant(wxString::Format(_("Time to process the second resizing procedure: %ldms"), swResize.Time()));

                // Consider the best point in previous iteration
                params = m_Parameters[0];
            }
            else
            {
                // Fixes and checks
                params.FixWeights();
                params.FixCoordinates();
                m_Parameters.push_back(params);
            }

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
                asLogError(_("The final dates selection contains NaNs"));

                double tmpVmin = m_Parameters[m_Parameters.size()-1].GetPredictorVmin(i_step, 0);
                double tmpUmin = m_Parameters[m_Parameters.size()-1].GetPredictorUmin(i_step, 0);
                int tmpVptsnb = m_Parameters[m_Parameters.size()-1].GetPredictorVptsnb(i_step, 0);
                int tmpUptsnb = m_Parameters[m_Parameters.size()-1].GetPredictorUptsnb(i_step, 0);
                asLogMessageImportant(wxString::Format(_("Area: Vmin = %.2f, Vptsnb = %d, Umin = %.2f, Uptsnb = %d"), tmpVmin, tmpVptsnb, tmpUmin, tmpUptsnb ));


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
        wxASSERT(m_Parameters.size()>0);
        wxASSERT(m_ParametersTemp.size()>0);
        wxASSERT(m_ScoresCalibTemp.size()>0);
        KeepBestTemp();
        ClearTemp();

        // Validate
        Validate();

        // Keep the best parameters set
        SetBestParameters(results_best);
        if(!results_best.Print()) return false;
        results_all.Add(m_Parameters[0],m_ScoresCalib[0],m_ScoreValid);
        if(!results_all.Print()) return false;
        if(!m_Parameters[0].GenerateSimpleParametersFile(resultsXmlFilePath)) return false;
    }

    return true;
}
