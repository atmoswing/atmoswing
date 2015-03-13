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
    m_originalParams = params;

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
    VVectorInt stationsId = params.GetPredictandStationIdsVector();

    // Preload data
    try
    {
        if (!PreloadData(params))
        {
            asLogError(_("Could not preload the data."));
            return false;
        }
    }
	catch (std::bad_alloc& ba)
    {
        wxString msg(ba.what(), wxConvUTF8);
        asLogError(wxString::Format(_("Bad allocation caught in the data preloading: %s"), msg.c_str()));
        DeletePreloadedData();
        return false;
    }
	catch (std::exception& e)
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
        m_scoreClimatology.clear();

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
        params = m_originalParams;

        // Set the next station ID
        params.SetPredictandStationIds(stationId);

        // Process every step one after the other
        int stepsNb = params.GetStepsNb();
        for (int i_step=0; i_step<stepsNb; i_step++)
        {
            // Restore previous best parameters
            if (i_step>0)
            {
                params = m_parameters[0];
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
            double predictorXminLowerLimit = params.GetPredictorXminLowerLimit(i_step, 0);
            double predictorXminUpperLimit = params.GetPredictorXminUpperLimit(i_step, 0);
            double predictorXminIteration = params.GetPredictorXminIteration(i_step, 0);
            int predictorXptsnbIteration = params.GetPredictorXptsnbIteration(i_step, 0);
            int predictorXptsnbLowerLimit = params.GetPredictorXptsnbLowerLimit(i_step, 0);
            int predictorXptsnbUpperLimit = params.GetPredictorXptsnbUpperLimit(i_step, 0);
            double predictorYminLowerLimit = params.GetPredictorYminLowerLimit(i_step, 0);
            double predictorYminUpperLimit = params.GetPredictorYminUpperLimit(i_step, 0);
            double predictorYminIteration = params.GetPredictorYminIteration(i_step, 0);
            int predictorYptsnbIteration = params.GetPredictorYptsnbIteration(i_step, 0);
            int predictorYptsnbLowerLimit = params.GetPredictorYptsnbLowerLimit(i_step, 0);
            int predictorYptsnbUpperLimit = params.GetPredictorYptsnbUpperLimit(i_step, 0);

            for (int i_ptor=0; i_ptor<ptorsNb; i_ptor++)
            {
                predictorXminLowerLimit = wxMax(predictorXminLowerLimit, params.GetPredictorXminLowerLimit(i_step, i_ptor));
                predictorXminUpperLimit = wxMin(predictorXminUpperLimit, params.GetPredictorXminUpperLimit(i_step, i_ptor));
                predictorXminIteration = wxMin(predictorXminIteration, params.GetPredictorXminIteration(i_step, i_ptor));
                predictorXptsnbIteration = wxMin(predictorXptsnbIteration, params.GetPredictorXptsnbIteration(i_step, i_ptor));
                predictorXptsnbLowerLimit = wxMax(predictorXptsnbLowerLimit, params.GetPredictorXptsnbLowerLimit(i_step, i_ptor));
                predictorXptsnbUpperLimit = wxMin(predictorXptsnbUpperLimit, params.GetPredictorXptsnbUpperLimit(i_step, i_ptor));
                predictorYminLowerLimit = wxMax(predictorYminLowerLimit, params.GetPredictorYminLowerLimit(i_step, i_ptor));
                predictorYminUpperLimit = wxMin(predictorYminUpperLimit, params.GetPredictorYminUpperLimit(i_step, i_ptor));
                predictorYminIteration = wxMin(predictorYminIteration, params.GetPredictorYminIteration(i_step, i_ptor));
                predictorYptsnbIteration = wxMin(predictorYptsnbIteration, params.GetPredictorYptsnbIteration(i_step, i_ptor));
                predictorYptsnbLowerLimit = wxMax(predictorYptsnbLowerLimit, params.GetPredictorYptsnbLowerLimit(i_step, i_ptor));
                predictorYptsnbUpperLimit = wxMax(predictorYptsnbUpperLimit, params.GetPredictorYptsnbUpperLimit(i_step, i_ptor));
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
            }
            params.FixAnalogsNb();

            if (predictorXminIteration!=0 || predictorXptsnbIteration!=0 || predictorYminIteration!=0 || predictorYptsnbIteration!=0)
            {
                // Set the minimal size
                for (int i_ptor=0; i_ptor<ptorsNb; i_ptor++)
                {
                    if (params.GetPredictorFlatAllowed(i_step, i_ptor))
                    {
                        params.SetPredictorXptsnb(i_step, i_ptor, 1);
                        params.SetPredictorYptsnb(i_step, i_ptor, 1);
                    }
                    else
                    {
                        params.SetPredictorXptsnb(i_step, i_ptor, predictorXptsnbLowerLimit);
                        params.SetPredictorYptsnb(i_step, i_ptor, predictorYptsnbLowerLimit);
                    }
                }

                wxStopWatch swMap;

                // Build map to explore
                ClearTemp();

                for (int i_x=0; i_x<=((predictorXminUpperLimit-predictorXminLowerLimit)/(predictorXminIteration*stepsLonPertinenceMap)); i_x++)
                {
                    for (int i_y=0; i_y<=((predictorYminUpperLimit-predictorYminLowerLimit)/(predictorYminIteration*stepsLatPertinenceMap)); i_y++)
                    {
                        double x = predictorXminLowerLimit+(predictorXminIteration*stepsLonPertinenceMap)*i_x;
                        double y = predictorYminLowerLimit+(predictorYminIteration*stepsLatPertinenceMap)*i_y;
// TODO (phorton#5#): This is not exact for unregular grids... Improve this approach !
                        for (int i_ptor=0; i_ptor<ptorsNb; i_ptor++)
                        {
                            params.SetPredictorXmin(i_step, i_ptor, x);
                            params.SetPredictorYmin(i_step, i_ptor, y);

                            // Fixes and checks
                            params.FixWeights();
                            params.FixCoordinates();
                        }
                        m_parametersTemp.push_back(params);
                    }
                }

                // Process the relevance map
                asLogState(wxString::Format(_("Calibration: processing the relevance map for all the predictors of step %d (station %s)."), i_step, GetPredictandStationIdsList(stationId).c_str()));
                for (unsigned int i_param=0; i_param<m_parametersTemp.size(); i_param++)
                {
                    if (proceedSequentially)
                    {
                        bool containsNaNs = false;
                        if (i_step==0)
                        {
                            if(!GetAnalogsDates(anaDates, m_parametersTemp[i_param], i_step, containsNaNs)) return false;
                        }
                        else
                        {
                            if(!GetAnalogsSubDates(anaDates, m_parametersTemp[i_param], anaDatesPrevious, i_step, containsNaNs)) return false;
                        }
                        if (containsNaNs)
                        {
                            m_scoresCalibTemp.push_back(NaNFloat);
                            continue;
                        }
                        if(!GetAnalogsValues(anaValues, m_parametersTemp[i_param], anaDates, i_step)) return false;
                        if(!GetAnalogsForecastScores(anaScores, m_parametersTemp[i_param], anaValues, i_step)) return false;
                        if(!GetAnalogsForecastScoreFinal(anaScoreFinal, m_parametersTemp[i_param], anaScores, i_step)) return false;
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
                                if(!GetAnalogsDates(anaDates, m_parametersTemp[i_param], sub_step, containsNaNs)) return false;
                            }
                            else
                            {
                                if(!GetAnalogsSubDates(anaDates, m_parametersTemp[i_param], anaDatesPreviousSubRuns, sub_step, containsNaNs)) return false;
                            }
                            if (containsNaNs)
                            {
                                continueLoop = false;
                                m_scoresCalibTemp.push_back(NaNFloat);
                                continue;
                            }
                            anaDatesPreviousSubRuns = anaDates;
                        }
                        if (continueLoop)
                        {
                            if(!GetAnalogsValues(anaValues, m_parametersTemp[i_param], anaDates, stepsNb-1)) return false;
                            if(!GetAnalogsForecastScores(anaScores, m_parametersTemp[i_param], anaValues, stepsNb-1)) return false;
                            if(!GetAnalogsForecastScoreFinal(anaScoreFinal, m_parametersTemp[i_param], anaScores, stepsNb-1)) return false;
                        }
                    }

                    // Store the result
                    m_scoresCalibTemp.push_back(anaScoreFinal.GetForecastScore());
                    results_tested.Add(m_parametersTemp[i_param],anaScoreFinal.GetForecastScore());
                }

                asLogMessageImportant(wxString::Format(_("Time to process the relevance map: %ldms"), swMap.Time()));

                // Keep the best parameter set
                wxASSERT(m_parametersTemp.size()>0);
                RemoveNaNsInTemp();
                PushBackBestTemp();
                wxASSERT(m_parameters.size()==1);
                ClearTemp();

                asLogMessageImportant(wxString::Format(_("Best point on relevance map: %.2f lat, %.2f lon"), m_parameters[m_parameters.size()-1].GetPredictorYmin(i_step, 0), m_parameters[m_parameters.size()-1].GetPredictorXmin(i_step, 0)));

                // Resize domain
                asLogState(wxString::Format(_("Calibration: resize the spatial domain for every predictor (station %s)."), GetPredictandStationIdsList(stationId).c_str()));

                wxStopWatch swEnlarge;

                bool isover = false;
                while (!isover)
                {
                    double xtmp, ytmp;
                    int xptsnbtmp, yptsnbtmp;
                    isover = true;

                    ClearTemp();

                    for (int i_resizing=0; i_resizing<4; i_resizing++)
                    {
                        // Consider the best point in previous iteration
                        params = m_parameters[0];

                        for (int i_ptor=0; i_ptor<ptorsNb; i_ptor++)
                        {
                            switch (i_resizing)
                            {
                                case 0:
                                    // Enlarge top
                                    yptsnbtmp = params.GetPredictorYptsnb(i_step, i_ptor)+predictorYptsnbIteration;
                                    yptsnbtmp = wxMax(wxMin(yptsnbtmp, predictorYptsnbUpperLimit), predictorYptsnbLowerLimit);
                                    params.SetPredictorYptsnb(i_step, i_ptor, yptsnbtmp);
                                    break;
                                case 1:
                                    // Enlarge right
                                    xptsnbtmp = params.GetPredictorXptsnb(i_step, i_ptor)+predictorXptsnbIteration;
                                    xptsnbtmp = wxMax(wxMin(xptsnbtmp, predictorXptsnbUpperLimit), predictorXptsnbLowerLimit);
                                    params.SetPredictorXptsnb(i_step, i_ptor, xptsnbtmp);
                                    break;
                                case 2:
                                    // Enlarge bottom
                                    yptsnbtmp = params.GetPredictorYptsnb(i_step, i_ptor)+predictorYptsnbIteration;
                                    yptsnbtmp = wxMax(wxMin(yptsnbtmp, predictorYptsnbUpperLimit), predictorYptsnbLowerLimit);
                                    params.SetPredictorYptsnb(i_step, i_ptor, yptsnbtmp);
                                    ytmp = params.GetPredictorYmin(i_step, i_ptor)-predictorYminIteration;
                                    ytmp = wxMax(wxMin(ytmp, predictorYminUpperLimit), predictorYminLowerLimit);
                                    params.SetPredictorYmin(i_step, i_ptor, ytmp);
                                    break;
                                case 3:
                                    // Enlarge left
                                    xptsnbtmp = params.GetPredictorXptsnb(i_step, i_ptor)+predictorXptsnbIteration;
                                    xptsnbtmp = wxMax(wxMin(xptsnbtmp, predictorXptsnbUpperLimit), predictorXptsnbLowerLimit);
                                    params.SetPredictorXptsnb(i_step, i_ptor, xptsnbtmp);
                                    xtmp = params.GetPredictorXmin(i_step, i_ptor)-predictorXminIteration;
                                    xtmp = wxMax(wxMin(xtmp, predictorXminUpperLimit), predictorXminLowerLimit);
                                    params.SetPredictorXmin(i_step, i_ptor, xtmp);
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
                    if (m_parametersTemp.size()>0)
                    {
                        KeepBestTemp();
                    }
                }

                asLogMessageImportant(wxString::Format(_("Time to process the first resizing procedure: %ldms"), swEnlarge.Time()));

                // Resize domain
                asLogState(wxString::Format(_("Calibration: reshape again the spatial domain for every predictor (station %s)."), GetPredictandStationIdsList(stationId).c_str()));

                // Try other moves. No while loop but reinitialize the for loops
                double xtmp, ytmp;
                int xptsnbtmp, yptsnbtmp;

                ClearTemp();

                wxStopWatch swResize;

                for (int multipleFactor=1; multipleFactor<=resizingIterations; multipleFactor++)
                {
                    for (int i_resizing=0; i_resizing<22; i_resizing++)
                    {
                        // Consider the best point in previous iteration
                        params = m_parameters[0];

                        for (int i_ptor=0; i_ptor<ptorsNb; i_ptor++)
                        {
                            switch (i_resizing)
                            {
                                case 0:
                                    // Enlarge all
                                    yptsnbtmp = params.GetPredictorYptsnb(i_step, i_ptor)+2*multipleFactor*predictorYptsnbIteration;
                                    yptsnbtmp = wxMax(wxMin(yptsnbtmp, predictorYptsnbUpperLimit), predictorYptsnbLowerLimit);
                                    xptsnbtmp = params.GetPredictorXptsnb(i_step, i_ptor)+2*multipleFactor*predictorXptsnbIteration;
                                    xptsnbtmp = wxMax(wxMin(xptsnbtmp, predictorXptsnbUpperLimit), predictorXptsnbLowerLimit);
                                    params.SetPredictorYptsnb(i_step, i_ptor, yptsnbtmp);
                                    params.SetPredictorXptsnb(i_step, i_ptor, xptsnbtmp);
                                    ytmp = params.GetPredictorYmin(i_step, i_ptor)-multipleFactor*predictorYminIteration;
                                    ytmp = wxMax(wxMin(ytmp, predictorYminUpperLimit), predictorYminLowerLimit);
                                    xtmp = params.GetPredictorXmin(i_step, i_ptor)-multipleFactor*predictorXminIteration;
                                    xtmp = wxMax(wxMin(ytmp, predictorXminUpperLimit), predictorXminLowerLimit);
                                    params.SetPredictorYmin(i_step, i_ptor, ytmp);
                                    params.SetPredictorXmin(i_step, i_ptor, xtmp);
                                    break;
                                case 1:
                                    // Reduce all
                                    yptsnbtmp = params.GetPredictorYptsnb(i_step, i_ptor)-2*multipleFactor*predictorYptsnbIteration;
                                    yptsnbtmp = wxMax(wxMin(yptsnbtmp, predictorYptsnbUpperLimit), predictorYptsnbLowerLimit);
                                    xptsnbtmp = params.GetPredictorXptsnb(i_step, i_ptor)-2*multipleFactor*predictorXptsnbIteration;
                                    xptsnbtmp = wxMax(wxMin(xptsnbtmp, predictorXptsnbUpperLimit), predictorXptsnbLowerLimit);
                                    params.SetPredictorYptsnb(i_step, i_ptor, yptsnbtmp);
                                    params.SetPredictorXptsnb(i_step, i_ptor, xptsnbtmp);
                                    ytmp = params.GetPredictorYmin(i_step, i_ptor)+multipleFactor*predictorYminIteration;
                                    ytmp = wxMax(wxMin(ytmp, predictorYminUpperLimit), predictorYminLowerLimit);
                                    xtmp = params.GetPredictorXmin(i_step, i_ptor)+multipleFactor*predictorXminIteration;
                                    xtmp = wxMax(wxMin(xtmp, predictorXminUpperLimit), predictorXminLowerLimit);
                                    params.SetPredictorYmin(i_step, i_ptor, ytmp);
                                    params.SetPredictorXmin(i_step, i_ptor, xtmp);
                                    break;
                                case 2:
                                    // Reduce top
                                    yptsnbtmp = params.GetPredictorYptsnb(i_step, i_ptor)-multipleFactor*predictorYptsnbIteration;
                                    yptsnbtmp = wxMax(wxMin(yptsnbtmp, predictorYptsnbUpperLimit), predictorYptsnbLowerLimit);
                                    params.SetPredictorYptsnb(i_step, i_ptor, yptsnbtmp);
                                    break;
                                case 3:
                                    // Reduce right
                                    xptsnbtmp = params.GetPredictorXptsnb(i_step, i_ptor)-multipleFactor*predictorXptsnbIteration;
                                    xptsnbtmp = wxMax(wxMin(xptsnbtmp, predictorXptsnbUpperLimit), predictorXptsnbLowerLimit);
                                    params.SetPredictorXptsnb(i_step, i_ptor, xptsnbtmp);
                                    break;
                                case 4:
                                    // Reduce bottom
                                    yptsnbtmp = params.GetPredictorYptsnb(i_step, i_ptor)-multipleFactor*predictorYptsnbIteration;
                                    yptsnbtmp = wxMax(wxMin(yptsnbtmp, predictorYptsnbUpperLimit), predictorYptsnbLowerLimit);
                                    params.SetPredictorYptsnb(i_step, i_ptor, yptsnbtmp);
                                    ytmp = params.GetPredictorYmin(i_step, i_ptor)+multipleFactor*predictorYminIteration;
                                    ytmp = wxMax(wxMin(ytmp, predictorYminUpperLimit), predictorYminLowerLimit);
                                    params.SetPredictorYmin(i_step, i_ptor, ytmp);
                                    break;
                                case 5:
                                    // Reduce left
                                    xptsnbtmp = params.GetPredictorXptsnb(i_step, i_ptor)-multipleFactor*predictorXptsnbIteration;
                                    xptsnbtmp = wxMax(wxMin(xptsnbtmp, predictorXptsnbUpperLimit), predictorXptsnbLowerLimit);
                                    params.SetPredictorXptsnb(i_step, i_ptor, xptsnbtmp);
                                    xtmp = params.GetPredictorXmin(i_step, i_ptor)+multipleFactor*predictorXminIteration;
                                    xtmp = wxMax(wxMin(xtmp, predictorXminUpperLimit), predictorXminLowerLimit);
                                    params.SetPredictorXmin(i_step, i_ptor, xtmp);
                                    break;
                                case 6:
                                    // Reduce top & bottom
                                    yptsnbtmp = params.GetPredictorYptsnb(i_step, i_ptor)-2*multipleFactor*predictorYptsnbIteration;
                                    yptsnbtmp = wxMax(wxMin(yptsnbtmp, predictorYptsnbUpperLimit), predictorYptsnbLowerLimit);
                                    params.SetPredictorYptsnb(i_step, i_ptor, yptsnbtmp);
                                    ytmp = params.GetPredictorYmin(i_step, i_ptor)+multipleFactor*predictorYminIteration;
                                    ytmp = wxMax(wxMin(ytmp, predictorYminUpperLimit), predictorYminLowerLimit);
                                    params.SetPredictorYmin(i_step, i_ptor, ytmp);
                                    break;
                                case 7:
                                    // Reduce right & left
                                    xptsnbtmp = params.GetPredictorXptsnb(i_step, i_ptor)-2*multipleFactor*predictorXptsnbIteration;
                                    xptsnbtmp = wxMax(wxMin(xptsnbtmp, predictorXptsnbUpperLimit), predictorXptsnbLowerLimit);
                                    params.SetPredictorXptsnb(i_step, i_ptor, xptsnbtmp);
                                    xtmp = params.GetPredictorXmin(i_step, i_ptor)+multipleFactor*predictorXminIteration;
                                    xtmp = wxMax(wxMin(xtmp, predictorXminUpperLimit), predictorXminLowerLimit);
                                    params.SetPredictorXmin(i_step, i_ptor, xtmp);
                                    break;
                                case 8:
                                    // Enlarge top
                                    yptsnbtmp = params.GetPredictorYptsnb(i_step, i_ptor)+multipleFactor*predictorYptsnbIteration;
                                    yptsnbtmp = wxMax(wxMin(yptsnbtmp, predictorYptsnbUpperLimit), predictorYptsnbLowerLimit);
                                    params.SetPredictorYptsnb(i_step, i_ptor, yptsnbtmp);
                                    break;
                                case 9:
                                    // Enlarge right
                                    xptsnbtmp = params.GetPredictorXptsnb(i_step, i_ptor)+multipleFactor*predictorXptsnbIteration;
                                    xptsnbtmp = wxMax(wxMin(xptsnbtmp, predictorXptsnbUpperLimit), predictorXptsnbLowerLimit);
                                    params.SetPredictorXptsnb(i_step, i_ptor, xptsnbtmp);
                                    break;
                                case 10:
                                    // Enlarge bottom
                                    yptsnbtmp = params.GetPredictorYptsnb(i_step, i_ptor)+multipleFactor*predictorYptsnbIteration;
                                    yptsnbtmp = wxMax(wxMin(yptsnbtmp, predictorYptsnbUpperLimit), predictorYptsnbLowerLimit);
                                    params.SetPredictorYptsnb(i_step, i_ptor, yptsnbtmp);
                                    ytmp = params.GetPredictorYmin(i_step, i_ptor)-multipleFactor*predictorYminIteration;
                                    ytmp = wxMax(wxMin(ytmp, predictorYminUpperLimit), predictorYminLowerLimit);
                                    params.SetPredictorYmin(i_step, i_ptor, ytmp);
                                    break;
                                case 11:
                                    // Enlarge left
                                    xptsnbtmp = params.GetPredictorXptsnb(i_step, i_ptor)+multipleFactor*predictorXptsnbIteration;
                                    xptsnbtmp = wxMax(wxMin(xptsnbtmp, predictorXptsnbUpperLimit), predictorXptsnbLowerLimit);
                                    params.SetPredictorXptsnb(i_step, i_ptor, xptsnbtmp);
                                    xtmp = params.GetPredictorXmin(i_step, i_ptor)-multipleFactor*predictorXminIteration;
                                    xtmp = wxMax(wxMin(xtmp, predictorXminUpperLimit), predictorXminLowerLimit);
                                    params.SetPredictorXmin(i_step, i_ptor, xtmp);
                                    break;
                                case 12:
                                    // Enlarge top & bottom
                                    yptsnbtmp = params.GetPredictorYptsnb(i_step, i_ptor)+2*multipleFactor*predictorYptsnbIteration;
                                    yptsnbtmp = wxMax(wxMin(yptsnbtmp, predictorYptsnbUpperLimit), predictorYptsnbLowerLimit);
                                    params.SetPredictorYptsnb(i_step, i_ptor, yptsnbtmp);
                                    ytmp = params.GetPredictorYmin(i_step, i_ptor)-multipleFactor*predictorYminIteration;
                                    ytmp = wxMax(wxMin(ytmp, predictorYminUpperLimit), predictorYminLowerLimit);
                                    params.SetPredictorYmin(i_step, i_ptor, ytmp);
                                    break;
                                case 13:
                                    // Enlarge right & left
                                    xptsnbtmp = params.GetPredictorXptsnb(i_step, i_ptor)+2*multipleFactor*predictorXptsnbIteration;
                                    xptsnbtmp = wxMax(wxMin(xptsnbtmp, predictorXptsnbUpperLimit), predictorXptsnbLowerLimit);
                                    params.SetPredictorXptsnb(i_step, i_ptor, xptsnbtmp);
                                    xtmp = params.GetPredictorXmin(i_step, i_ptor)-multipleFactor*predictorXminIteration;
                                    xtmp = wxMax(wxMin(xtmp, predictorXminUpperLimit), predictorXminLowerLimit);
                                    params.SetPredictorXmin(i_step, i_ptor, xtmp);
                                    break;
                                case 14:
                                    // Move top
                                    ytmp = params.GetPredictorYmin(i_step, i_ptor)+multipleFactor*predictorYminIteration;
                                    ytmp = wxMax(wxMin(ytmp, predictorYminUpperLimit), predictorYminLowerLimit);
                                    params.SetPredictorYmin(i_step, i_ptor, ytmp);
                                    break;
                                case 15:
                                    // Move right
                                    xtmp = params.GetPredictorXmin(i_step, i_ptor)+multipleFactor*predictorXminIteration;
                                    xtmp = wxMax(wxMin(xtmp, predictorXminUpperLimit), predictorXminLowerLimit);
                                    params.SetPredictorXmin(i_step, i_ptor, xtmp);
                                    break;
                                case 16:
                                    // Move bottom
                                    ytmp = params.GetPredictorYmin(i_step, i_ptor)-multipleFactor*predictorYminIteration;
                                    ytmp = wxMax(wxMin(ytmp, predictorYminUpperLimit), predictorYminLowerLimit);
                                    params.SetPredictorYmin(i_step, i_ptor, ytmp);
                                    break;
                                case 17:
                                    // Move left
                                    xtmp = params.GetPredictorXmin(i_step, i_ptor)-multipleFactor*predictorXminIteration;
                                    xtmp = wxMax(wxMin(xtmp, predictorXminUpperLimit), predictorXminLowerLimit);
                                    params.SetPredictorXmin(i_step, i_ptor, xtmp);
                                    break;
                                case 18:
                                    // Move top-left
                                    ytmp = params.GetPredictorYmin(i_step, i_ptor)+multipleFactor*predictorYminIteration;
                                    ytmp = wxMax(wxMin(ytmp, predictorYminUpperLimit), predictorYminLowerLimit);
                                    params.SetPredictorYmin(i_step, i_ptor, ytmp);
                                    xtmp = params.GetPredictorXmin(i_step, i_ptor)-multipleFactor*predictorXminIteration;
                                    xtmp = wxMax(wxMin(xtmp, predictorXminUpperLimit), predictorXminLowerLimit);
                                    params.SetPredictorXmin(i_step, i_ptor, xtmp);
                                    break;
                                case 19:
                                    // Move top-right
                                    ytmp = params.GetPredictorYmin(i_step, i_ptor)+multipleFactor*predictorYminIteration;
                                    ytmp = wxMax(wxMin(ytmp, predictorYminUpperLimit), predictorYminLowerLimit);
                                    params.SetPredictorYmin(i_step, i_ptor, ytmp);
                                    xtmp = params.GetPredictorXmin(i_step, i_ptor)+multipleFactor*predictorXminIteration;
                                    xtmp = wxMax(wxMin(xtmp, predictorXminUpperLimit), predictorXminLowerLimit);
                                    params.SetPredictorXmin(i_step, i_ptor, xtmp);
                                    break;
                                case 20:
                                    // Move bottom-left
                                    ytmp = params.GetPredictorYmin(i_step, i_ptor)-multipleFactor*predictorYminIteration;
                                    ytmp = wxMax(wxMin(ytmp, predictorYminUpperLimit), predictorYminLowerLimit);
                                    params.SetPredictorYmin(i_step, i_ptor, ytmp);
                                    xtmp = params.GetPredictorXmin(i_step, i_ptor)-multipleFactor*predictorXminIteration;
                                    xtmp = wxMax(wxMin(xtmp, predictorXminUpperLimit), predictorXminLowerLimit);
                                    params.SetPredictorXmin(i_step, i_ptor, xtmp);
                                    break;
                                case 21:
                                    // Move bottom-right
                                    ytmp = params.GetPredictorYmin(i_step, i_ptor)-multipleFactor*predictorYminIteration;
                                    ytmp = wxMax(wxMin(ytmp, predictorYminUpperLimit), predictorYminLowerLimit);
                                    params.SetPredictorYmin(i_step, i_ptor, ytmp);
                                    xtmp = params.GetPredictorXmin(i_step, i_ptor)+multipleFactor*predictorXminIteration;
                                    xtmp = wxMax(wxMin(xtmp, predictorXminUpperLimit), predictorXminLowerLimit);
                                    params.SetPredictorXmin(i_step, i_ptor, xtmp);
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
                params = m_parameters[0];
            }
            else
            {
                // Fixes and checks
                params.FixWeights();
                params.FixCoordinates();
                m_parameters.push_back(params);
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

                double tmpYmin = m_parameters[m_parameters.size()-1].GetPredictorYmin(i_step, 0);
                double tmpXmin = m_parameters[m_parameters.size()-1].GetPredictorXmin(i_step, 0);
                int tmpYptsnb = m_parameters[m_parameters.size()-1].GetPredictorYptsnb(i_step, 0);
                int tmpXptsnb = m_parameters[m_parameters.size()-1].GetPredictorXptsnb(i_step, 0);
                asLogMessageImportant(wxString::Format(_("Area: Ymin = %.2f, Yptsnb = %d, Xmin = %.2f, Xptsnb = %d"), tmpYmin, tmpYptsnb, tmpXmin, tmpXptsnb ));


                return false;
            }
        }

        // Finally calibrate the number of analogs for every step
        asLogState(_("Calibration: find the analogs number for every step."));
        ClearTemp();
        asResultsAnalogsDates tempDates;
        if(!SubProcessAnalogsNumber(params, tempDates)) return false;

        // Extract intermediate results from temporary vectors
        for (unsigned int i_res=0; i_res<m_parametersTemp.size(); i_res++)
        {
            results_tested.Add(m_parametersTemp[i_res],m_scoresCalibTemp[i_res]);
        }
        results_tested.Print();

        // Keep the best parameter set
        wxASSERT(m_parameters.size()>0);
        wxASSERT(m_parametersTemp.size()>0);
        wxASSERT(m_scoresCalibTemp.size()>0);
        KeepBestTemp();
        ClearTemp();

        // Validate
        Validate();

        // Keep the best parameters set
        SetBestParameters(results_best);
        if(!results_best.Print()) return false;
        results_all.Add(m_parameters[0],m_scoresCalib[0],m_scoreValid);
        if(!results_all.Print()) return false;
        if(!m_parameters[0].GenerateSimpleParametersFile(resultsXmlFilePath)) return false;
    }

    return true;
}
