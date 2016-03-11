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

#include "asMethodCalibrator.h"

#ifndef UNIT_TESTING
    #include "AtmoswingAppOptimizer.h"
#endif

asMethodCalibrator::asMethodCalibrator()
    :
    asMethodStandard()
{
    m_predictorDataDir = wxEmptyString;
    m_paramsFilePath = wxEmptyString;
    m_preloaded = false;
    m_validationMode = false;
    m_scoreValid = NaNFloat;

    // Seeds the random generator
    asTools::InitRandom();
}

asMethodCalibrator::~asMethodCalibrator()
{
    DeletePreloadedData();
}

bool asMethodCalibrator::Manager()
{
    // Set unresponsive to speedup
    g_responsive = false;

    // Reset the score of the climatology
    m_scoreClimatology.clear();

    // Seeds the random generator
    asTools::InitRandom();

    // Load parameters
    asParametersCalibration params;
    if(!params.LoadFromFile(m_paramsFilePath)) return false;
    params.InitValues();
    m_originalParams = params;

    // Load the Predictand DB
    asLogMessage(_("Loading the Predictand DB."));
    if(!LoadPredictandDB(m_predictandDBFilePath)) return false;
    asLogMessage(_("Predictand DB loaded."));

    // Get a forecast score object to extract the score order
    asForecastScore* forecastScore = asForecastScore::GetInstance(params.GetForecastScoreName());
    Order scoreOrder = forecastScore->GetOrder();
    SetScoreOrder(scoreOrder);
    wxDELETE(forecastScore);

    // Watch
    wxStopWatch sw;

    // Calibrate
    if(Calibrate(params))
    {
        // Display processing time
        asLogMessageImportant(wxString::Format(_("The whole processing took %ldms to execute"), sw.Time()));
        asLogState(_("Calibration over."));
    }
    else
    {
        asLogError(_("The parameters could not be calibrated"));
    }

    // Delete preloaded data and cleanup
    DeletePreloadedData();

    return true;
}

void asMethodCalibrator::ClearAll()
{
    m_parametersTemp.clear();
    m_scoresCalibTemp.clear();
    m_parameters.clear();
    m_scoresCalib.clear();
    m_scoreValid = NaNFloat;
}

void asMethodCalibrator::ClearTemp()
{
    m_parametersTemp.clear();
    m_scoresCalibTemp.clear();
}

void asMethodCalibrator::ClearScores()
{
    for (unsigned int i=0; i<m_scoresCalib.size(); i++)
    {
        m_scoresCalib[i] = NaNFloat;
    }
    m_scoreValid = NaNFloat;
}

void asMethodCalibrator::PushBackBestTemp()
{
    SortScoresAndParametersTemp();
    PushBackFirstTemp();
}

void asMethodCalibrator::RemoveNaNsInTemp()
{
    wxASSERT(m_parametersTemp.size()==m_scoresCalibTemp.size());

    std::vector <asParametersCalibration> copyParametersTemp;
    VectorFloat copyScoresCalibTemp;

    for (unsigned int i=0; i<m_scoresCalibTemp.size(); i++)
    {
        if (!asTools::IsNaN(m_scoresCalibTemp[i]))
        {
            copyScoresCalibTemp.push_back(m_scoresCalibTemp[i]);
            copyParametersTemp.push_back(m_parametersTemp[i]);
        }
    }

    m_scoresCalibTemp = copyScoresCalibTemp;
    m_parametersTemp = copyParametersTemp;

    wxASSERT(m_parametersTemp.size()==m_scoresCalibTemp.size());
    wxASSERT(m_parametersTemp.size()>0);
}

void asMethodCalibrator::KeepBestTemp()
{
    SortScoresAndParametersTemp();
    KeepFirstTemp();
}

void asMethodCalibrator::PushBackFirstTemp()
{
    m_parameters.push_back(m_parametersTemp[0]);
    m_scoresCalib.push_back(m_scoresCalibTemp[0]);
}

void asMethodCalibrator::KeepFirstTemp()
{
    wxASSERT(m_parameters.size()>0);
    wxASSERT(m_parametersTemp.size()>0);
    wxASSERT(m_scoresCalibTemp.size()>0);
    m_parameters[0] = m_parametersTemp[0];
    if (m_scoresCalib.size()==0)
    {
        m_scoresCalib.push_back(m_scoresCalibTemp[0]);
    }
    else
    {
        m_scoresCalib[0] = m_scoresCalibTemp[0];
    }

}

void asMethodCalibrator::SortScoresAndParameters()
{
    wxASSERT(m_scoresCalib.size()==m_parameters.size());
    wxASSERT(m_scoresCalib.size()>=1);
    wxASSERT(m_parameters.size()>=1);

    if (m_parameters.size()==1) return;

    // Sort according to the score
    Array1DFloat vIndices = Array1DFloat::LinSpaced(Eigen::Sequential,m_scoresCalib.size(),
                            0,m_scoresCalib.size()-1);
    asTools::SortArrays(&m_scoresCalib[0], &m_scoresCalib[m_scoresCalib.size()-1],
                        &vIndices[0], &vIndices[m_scoresCalib.size()-1],
                        m_scoreOrder);

    // Sort the parameters sets as the scores
    std::vector <asParametersCalibration> copyParameters;
    for (unsigned int i=0; i<m_scoresCalib.size(); i++)
    {
        copyParameters.push_back(m_parameters[i]);
    }
    for (unsigned int i=0; i<m_scoresCalib.size(); i++)
    {
        int index = vIndices(i);
        m_parameters[i] = copyParameters[index];
    }
}

void asMethodCalibrator::SortScoresAndParametersTemp()
{
    wxASSERT(m_scoresCalibTemp.size()==m_parametersTemp.size());
    wxASSERT(m_scoresCalibTemp.size()>0);
    wxASSERT(m_parametersTemp.size()>0);

    if (m_parametersTemp.size()==1) return;

    // Sort according to the score
    Array1DFloat vIndices = Array1DFloat::LinSpaced(Eigen::Sequential,m_scoresCalibTemp.size(),
                            0,m_scoresCalibTemp.size()-1);
    asTools::SortArrays(&m_scoresCalibTemp[0], &m_scoresCalibTemp[m_scoresCalibTemp.size()-1],
                        &vIndices[0], &vIndices[m_scoresCalibTemp.size()-1],
                        m_scoreOrder);

    // Sort the parameters sets as the scores
    std::vector <asParametersCalibration> copyParameters;
    for (unsigned int i=0; i<m_scoresCalibTemp.size(); i++)
    {
        copyParameters.push_back(m_parametersTemp[i]);
    }
    for (unsigned int i=0; i<m_scoresCalibTemp.size(); i++)
    {
        int index = vIndices(i);
        m_parametersTemp[i] = copyParameters[index];
    }
}

bool asMethodCalibrator::PushBackInTempIfBetter(asParametersCalibration &params, asResultsAnalogsForecastScoreFinal &scoreFinal)
{
    float thisScore = scoreFinal.GetForecastScore();

    switch (m_scoreOrder)
    {
    case Asc:
        if (thisScore<m_scoresCalib[0])
        {
            m_parametersTemp.push_back(params);
            m_scoresCalibTemp.push_back(thisScore);
            return true;
        }
        break;

    case Desc:
        if (thisScore>m_scoresCalib[0])
        {
            m_parametersTemp.push_back(params);
            m_scoresCalibTemp.push_back(thisScore);
            return true;
        }
        break;

    default:
        asThrowException(_("The score order is not correcty defined."));
    }

    return false;
}

bool asMethodCalibrator::KeepIfBetter(asParametersCalibration &params, asResultsAnalogsForecastScoreFinal &scoreFinal)
{
    float thisScore = scoreFinal.GetForecastScore();

    switch (m_scoreOrder)
    {
    case Asc:
        if (thisScore<m_scoresCalib[0])
        {
            wxASSERT(m_parameters.size()>0);
            wxASSERT(m_scoresCalib.size()>0);
            m_parameters[0] = params;
            m_scoresCalib[0] = thisScore;
            return true;
        }
        break;

    case Desc:
        if (thisScore>m_scoresCalib[0])
        {
            wxASSERT(m_parameters.size()>0);
            wxASSERT(m_scoresCalib.size()>0);
            m_parameters[0] = params;
            m_scoresCalib[0] = thisScore;
            return true;
        }
        break;

    default:
        asThrowException(_("The score order is not correcty defined."));
    }

    return false;
}

bool asMethodCalibrator::SetSelectedParameters(asResultsParametersArray &results)
{
    // Extract selected parameters & best parameters
    for (unsigned int i=0; i<m_parameters.size(); i++)
    {
        results.Add(m_parameters[i],m_scoresCalib[i],m_scoreValid);
    }

    return true;
}

bool asMethodCalibrator::SetBestParameters(asResultsParametersArray &results)
{
    wxASSERT(m_parameters.size()>0);
    wxASSERT(m_scoresCalib.size()>0);

    // Extract selected parameters & best parameters
    float bestscore = m_scoresCalib[0];
    int bestscorerow = 0;

    for (unsigned int i=0; i<m_parameters.size(); i++)
    {
        if(m_scoreOrder==Asc)
        {
            if(m_scoresCalib[i]<bestscore)
            {
                bestscore = m_scoresCalib[i];
                bestscorerow = i;
            }
        }
        else
        {
            if(m_scoresCalib[i]>bestscore)
            {
                bestscore = m_scoresCalib[i];
                bestscorerow = i;
            }
        }
    }

    if (bestscorerow!=0)
    {
        // Re-validate
        Validate(bestscorerow);
    }

    results.Add(m_parameters[bestscorerow],m_scoresCalib[bestscorerow],m_scoreValid);

    return true;
}

wxString asMethodCalibrator::GetPredictandStationIdsList(VectorInt &stationIds)
{
    wxString id;

    if (stationIds.size()==1)
    {
        id << stationIds[0];
    }
    else
    {
        for (int i=0; i<(int)stationIds.size(); i++)
        {
            id << stationIds[i];

            if (i<(int)stationIds.size()-1)
            {
                id << ",";
            }
        }
    }

    return id;
}

bool asMethodCalibrator::PreloadData(asParametersScoring &params)
{
    // Load data once.
    if(!m_preloaded)
    {
        // Set preload to true here, so cleanup is made in case of exceptions.
        m_preloaded = true;

        // Archive date array
        double timeStartArchive = params.GetArchiveStart();
        double timeEndArchive = params.GetArchiveEnd();
        timeStartArchive += std::abs(params.GetTimeShiftDays()); // To avoid having dates before the start of the archive
        timeEndArchive = wxMin(timeEndArchive, timeEndArchive-params.GetTimeSpanDays()); // Adjust so the predictors search won't overtake the array

        // Target date array
        double timeStartCalibration = params.GetCalibrationStart();
        double timeEndCalibration = params.GetCalibrationEnd();
        timeStartCalibration += std::abs(params.GetTimeShiftDays()); // To avoid having dates before the start of the archive
        timeEndCalibration = wxMin(timeEndCalibration, timeEndCalibration-params.GetTimeSpanDays()); // Adjust so the predictors search won't overtake the array

        double timeStartData = wxMin(timeStartCalibration, timeStartArchive); // Always Jan 1st
        double timeEndData = wxMax(timeEndCalibration, timeEndArchive);

        // Resize container
        if (m_preloadedArchive.size()==0)
        {
            m_preloadedArchive.resize(params.GetStepsNb());
            m_preloadedArchivePointerCopy.resize(params.GetStepsNb());
            for (int tmp_step=0; tmp_step<params.GetStepsNb(); tmp_step++)
            {
                m_preloadedArchive[tmp_step].resize(params.GetPredictorsNb(tmp_step));
                m_preloadedArchivePointerCopy[tmp_step].resize(params.GetPredictorsNb(tmp_step));

                for (int tmp_ptor=0; tmp_ptor<params.GetPredictorsNb(tmp_step); tmp_ptor++)
                {
                    m_preloadedArchivePointerCopy[tmp_step][tmp_ptor] = false;
                    m_preloadedArchive[tmp_step][tmp_ptor].resize(1);
                    m_preloadedArchive[tmp_step][tmp_ptor][0].resize(1);
                    m_preloadedArchive[tmp_step][tmp_ptor][0][0]=NULL;
                }
            }
        }

        for (int tmp_step=0; tmp_step<params.GetStepsNb(); tmp_step++)
        {
            for (int tmp_ptor=0; tmp_ptor<params.GetPredictorsNb(tmp_step); tmp_ptor++)
            {
                if(params.NeedsPreloading(tmp_step, tmp_ptor))
                {
                    // Try to share pointers
                    if(tmp_ptor>0)
                    {
                        int prev = 0;
                        bool share = false;

                        for (prev=0;prev<tmp_ptor;prev++)
                        {
                            share = true;

                            if(!params.NeedsPreprocessing(tmp_step, tmp_ptor))
                            {
                                if(!params.GetPredictorDatasetId(tmp_step, tmp_ptor).IsSameAs(params.GetPredictorDatasetId(tmp_step, prev), false)) share = false;
                                if(!params.GetPredictorDataId(tmp_step, tmp_ptor).IsSameAs(params.GetPredictorDataId(tmp_step, prev), false)) share = false;
                                if(!params.GetPredictorGridType(tmp_step, tmp_ptor).IsSameAs(params.GetPredictorGridType(tmp_step, prev), false)) share = false;
                            }
                            else
                            {
                                if(!params.GetPreprocessMethod(tmp_step, tmp_ptor).IsSameAs(params.GetPreprocessMethod(tmp_step, prev), false)) share = false;
                                if(params.GetPreprocessSize(tmp_step, tmp_ptor)!=params.GetPreprocessSize(tmp_step, prev))
                                {
                                    share = false;
                                }
                                else
                                {
                                    int preprocessSize = params.GetPreprocessSize(tmp_step, tmp_ptor);

                                    for (int tmp_prepro=0; tmp_prepro<preprocessSize; tmp_prepro++)
                                    {
                                        if(!params.GetPreprocessDatasetId(tmp_step, tmp_ptor, tmp_prepro).IsSameAs(params.GetPreprocessDatasetId(tmp_step, prev, tmp_prepro), false)) share = false;
                                        if(!params.GetPreprocessDataId(tmp_step, tmp_ptor, tmp_prepro).IsSameAs(params.GetPreprocessDataId(tmp_step, prev, tmp_prepro), false)) share = false;
                                    }
                                }
                            }

                            if(params.GetPreloadXmin(tmp_step, tmp_ptor)!=params.GetPreloadXmin(tmp_step, prev)) share = false;
                            if(params.GetPreloadXptsnb(tmp_step, tmp_ptor)!=params.GetPreloadXptsnb(tmp_step, prev)) share = false;
                            if(params.GetPredictorXstep(tmp_step, tmp_ptor)!=params.GetPredictorXstep(tmp_step, prev)) share = false;
                            if(params.GetPreloadYmin(tmp_step, tmp_ptor)!=params.GetPreloadYmin(tmp_step, prev)) share = false;
                            if(params.GetPreloadYptsnb(tmp_step, tmp_ptor)!=params.GetPreloadYptsnb(tmp_step, prev)) share = false;
                            if(params.GetPredictorYstep(tmp_step, tmp_ptor)!=params.GetPredictorYstep(tmp_step, prev)) share = false;
                            if(params.GetPredictorFlatAllowed(tmp_step, tmp_ptor)!=params.GetPredictorFlatAllowed(tmp_step, prev)) share = false;

                            VectorFloat levels1 = params.GetPreloadLevels(tmp_step, tmp_ptor);
                            VectorFloat levels2 = params.GetPreloadLevels(tmp_step, prev);
                            if(levels1.size()!=levels2.size())
                            {
                                share = false;
                            }
                            else
                            {
                                for (unsigned int i=0;i<levels1.size();i++)
                                {
                                    if(levels1[i]!=levels2[i]) share = false;
                                }
                            }

                            VectorDouble hours1 = params.GetPreloadTimeHours(tmp_step, tmp_ptor);
                            VectorDouble hours2 = params.GetPreloadTimeHours(tmp_step, prev);
                            if(hours1.size()!=hours2.size())
                            {
                                share = false;
                            }
                            else
                            {
                                for (unsigned int i=0;i<hours1.size();i++)
                                {
                                    if(hours1[i]!=hours2[i]) share = false;
                                }
                            }

                            if (share) break;
                        }

                        if (share)
                        {
                            asLogMessage(_("Share data pointer"));

                            VectorFloat preloadLevels = params.GetPreloadLevels(tmp_step, tmp_ptor);
                            VectorDouble preloadTimeHours = params.GetPreloadTimeHours(tmp_step, tmp_ptor);
                            wxASSERT(preloadLevels.size()>0);
                            wxASSERT(preloadTimeHours.size()>0);

                            m_preloadedArchivePointerCopy[tmp_step][tmp_ptor] = true;
                            m_preloadedArchive[tmp_step][tmp_ptor].resize(preloadLevels.size());

                            wxASSERT(m_preloadedArchive[tmp_step].size()>(unsigned)prev);
                            wxASSERT(m_preloadedArchive[tmp_step][prev].size()==preloadLevels.size());

                            // Load data for every level and every hour
                            for (unsigned int tmp_level=0; tmp_level<preloadLevels.size(); tmp_level++)
                            {
                                m_preloadedArchive[tmp_step][tmp_ptor][tmp_level].resize(preloadTimeHours.size());
                                wxASSERT(m_preloadedArchive[tmp_step][prev][tmp_level].size()==preloadTimeHours.size());

                                for (unsigned int tmp_hour=0; tmp_hour<preloadTimeHours.size(); tmp_hour++)
                                {
                                    // Copy pointer
                                    m_preloadedArchive[tmp_step][tmp_ptor][tmp_level][tmp_hour]=m_preloadedArchive[tmp_step][prev][tmp_level][tmp_hour];
                                }
                            }

                            params.SetPreloadYptsnb(tmp_step, tmp_ptor, params.GetPreloadYptsnb(tmp_step, prev));

                            continue;
                        }
                    }

                    if(!params.NeedsPreprocessing(tmp_step, tmp_ptor))
                    {
                        asLogMessage(wxString::Format(_("Preloading data for predictor %d of step %d."), tmp_ptor, tmp_step));

                        VectorFloat preloadLevels = params.GetPreloadLevels(tmp_step, tmp_ptor);
                        VectorDouble preloadTimeHours = params.GetPreloadTimeHours(tmp_step, tmp_ptor);
                        wxASSERT(preloadLevels.size()>0);
                        wxASSERT(preloadTimeHours.size()>0);

                        // Resize container and set null pointers
                        m_preloadedArchive[tmp_step][tmp_ptor].resize(preloadLevels.size());
                        for (unsigned int tmp_level=0; tmp_level<preloadLevels.size(); tmp_level++)
                        {
                            m_preloadedArchive[tmp_step][tmp_ptor][tmp_level].resize(preloadTimeHours.size());
                            for (unsigned int tmp_hour=0; tmp_hour<preloadTimeHours.size(); tmp_hour++)
                            {
                                m_preloadedArchive[tmp_step][tmp_ptor][tmp_level][tmp_hour] = NULL;
                            }
                        }

                        // Load data for every level and every hour
                        for (unsigned int tmp_level=0; tmp_level<preloadLevels.size(); tmp_level++)
                        {
                            for (unsigned int tmp_hour=0; tmp_hour<preloadTimeHours.size(); tmp_hour++)
                            {
                                // Loading the datasets information
                                asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance(params.GetPredictorDatasetId(tmp_step, tmp_ptor), params.GetPredictorDataId(tmp_step, tmp_ptor), m_predictorDataDir);
                                if(!predictor) {
                                    return false;
                                }

                                // Date array object instantiation for the data loading. The array has the same length than timeArrayArchive, and the predictor dates are aligned with the target dates, but the dates are not the same.
                                double ptorStart = timeStartData-double(params.GetTimeShiftDays())+preloadTimeHours[tmp_hour]/24.0;

                                // For debugging:
                                // wxLogMessage("%f - %f + %f = %f", timeStartData, double(params.GetTimeShiftDays()), preloadTimeHours[tmp_hour]/24.0, ptorStart);
                                // wxLogMessage("ptorStart = %s", asTime::GetStringTime(ptorStart));
                                // wxLogMessage("timeStartData = %s", asTime::GetStringTime(timeStartData));
                                // wxLogMessage("params.GetTimeShiftDays() = %f", double(params.GetTimeShiftDays()));
                                // wxLogMessage("preloadTimeHours[tmp_hour]/24.0 = %f", preloadTimeHours[tmp_hour]/24.0);

                                double ptorEnd = timeEndData-double(params.GetTimeShiftDays())+preloadTimeHours[tmp_hour]/24.0;

                                asTimeArray timeArray(ptorStart, ptorEnd,
                                                      params.GetTimeArrayAnalogsTimeStepHours(),
                                                      asTimeArray::Simple);
                                timeArray.Init();

                                asGeo geo;
                                double Ymax = params.GetPreloadYmin(tmp_step, tmp_ptor)+params.GetPredictorYstep(tmp_step, tmp_ptor)*(double)(params.GetPreloadYptsnb(tmp_step, tmp_ptor)-1);
                                if (Ymax > geo.GetAxisYmax())
                                {
                                    double diff = Ymax-geo.GetAxisYmax();
                                    int removePts = asTools::Round(diff/params.GetPredictorYstep(tmp_step, tmp_ptor));
                                    params.SetPreloadYptsnb(tmp_step, tmp_ptor, params.GetPreloadYptsnb(tmp_step, tmp_ptor)-removePts);
                                    asLogMessage(wxString::Format(_("Adapt Y axis extent according to the maximum allowed (from %.3f to %.3f)."), Ymax, Ymax-diff));
                                    asLogMessage(wxString::Format(_("Remove %d points (%.3f-%.3f)/%.3f."), removePts, Ymax, geo.GetAxisYmax(), params.GetPredictorYstep(tmp_step, tmp_ptor)));
                                }

                                wxASSERT(params.GetPreloadXptsnb(tmp_step, tmp_ptor)>0);
                                wxASSERT(params.GetPreloadYptsnb(tmp_step, tmp_ptor)>0);

                                // Area object instantiation
                                asGeoAreaCompositeGrid* area = asGeoAreaCompositeGrid::GetInstance(params.GetPredictorGridType(tmp_step, tmp_ptor),
                                    params.GetPreloadXmin(tmp_step, tmp_ptor),
                                    params.GetPreloadXptsnb(tmp_step, tmp_ptor),
                                    params.GetPredictorXstep(tmp_step, tmp_ptor),
                                    params.GetPreloadYmin(tmp_step, tmp_ptor),
                                    params.GetPreloadYptsnb(tmp_step, tmp_ptor),
                                    params.GetPredictorYstep(tmp_step, tmp_ptor),
                                    preloadLevels[tmp_level],
                                    asNONE,
                                    params.GetPredictorFlatAllowed(tmp_step, tmp_ptor));
                                wxASSERT(area);

                                // Check the starting dates coherence
                                if (predictor->GetOriginalProviderStart()>ptorStart)
                                {
                                    asLogError(wxString::Format(_("The first year defined in the parameters (%s) is prior to the start date of the data (%s) (in asMethodCalibrator::PreloadData)."),
                                                                asTime::GetStringTime(ptorStart), asTime::GetStringTime(predictor->GetOriginalProviderStart())));
                                    wxDELETE(area);
                                    wxDELETE(predictor);
                                    return false;
                                }

                                // Data loading
                                asLogMessage(wxString::Format(_("Loading %s data for level %d, %d h."),
                                                              params.GetPredictorDataId(tmp_step, tmp_ptor),
                                                              (int)preloadLevels[tmp_level],
                                                              (int)preloadTimeHours[tmp_hour]));
                                try
                                {
                                    if(!predictor->Load(area, timeArray))
                                    {
                                        asLogError(_("The data could not be loaded."));
                                        wxDELETE(area);
                                        wxDELETE(predictor);
                                        return false;
                                    }
                                }
								catch (std::bad_alloc& ba)
                                {
                                    wxString msg(ba.what(), wxConvUTF8);
                                    asLogError(wxString::Format(_("Bad allocation in the data preloading: %s"), msg));
                                    wxDELETE(area);
                                    wxDELETE(predictor);
                                    return false;
                                }
								catch (std::exception& e)
                                {
                                    wxString msg(e.what(), wxConvUTF8);
                                    asLogError(wxString::Format(_("Exception in the data preloading: %s"), msg));
                                    wxDELETE(area);
                                    wxDELETE(predictor);
                                    return false;
                                }
                                asLogMessage(_("Data loaded."));
                                wxDELETE(area);

                                m_preloadedArchive[tmp_step][tmp_ptor][tmp_level][tmp_hour]=predictor;
                            }
                        }
                    }
                    else // preloading with preprocessing
                    {
                        asLogMessage(wxString::Format(_("Preloading data for predictor preprocessed %d of step %d."), tmp_ptor, tmp_step));

                        // Check the preprocessing method
                        wxString method = params.GetPreprocessMethod(tmp_step, tmp_ptor);

                        // Get the number of sub predictors
                        int preprocessSize = params.GetPreprocessSize(tmp_step, tmp_ptor);

                        // Levels and time arrays
                        VectorFloat preloadLevels = params.GetPreloadLevels(tmp_step, tmp_ptor);
                        VectorDouble preloadTimeHours = params.GetPreloadTimeHours(tmp_step, tmp_ptor);

                        // Check on which variable to loop
                        unsigned int preloadLevelsSize = preloadLevels.size();
                        unsigned int preloadTimeHoursSize = preloadTimeHours.size();
                        bool loopOnLevels = true;
                        bool loopOnTimeHours = true;


                        if(preloadLevelsSize==0)
                        {
                            loopOnLevels = false;
                            preloadLevelsSize = 1;
                        }

                        if(preloadTimeHoursSize==0)
                        {
                            loopOnTimeHours = false;
                            preloadTimeHoursSize = 1;
                        }

                        // Resize container and set null pointers
                        m_preloadedArchive[tmp_step][tmp_ptor].resize(preloadLevelsSize);
                        for (unsigned int tmp_level=0; tmp_level<preloadLevelsSize; tmp_level++)
                        {
                            m_preloadedArchive[tmp_step][tmp_ptor][tmp_level].resize(preloadTimeHoursSize);
                            for (unsigned int tmp_hour=0; tmp_hour<preloadTimeHoursSize; tmp_hour++)
                            {
                                m_preloadedArchive[tmp_step][tmp_ptor][tmp_level][tmp_hour] = NULL;
                            }
                        }

                        asLogMessage(wxString::Format(_("Preprocessing data (%d predictor(s)) while loading."),
                                                      preprocessSize));

                        // Load data for every level and every hour
                        for (unsigned int tmp_level=0; tmp_level<preloadLevelsSize; tmp_level++)
                        {
                            for (unsigned int tmp_hour=0; tmp_hour<preloadTimeHoursSize; tmp_hour++)
                            {
                                std::vector < asDataPredictorArchive* > predictorsPreprocess;

                                for (int tmp_prepro=0; tmp_prepro<preprocessSize; tmp_prepro++)
                                {
                                    asLogMessage(wxString::Format(_("Preloading data for predictor %d (preprocess %d) of step %d."), tmp_ptor, tmp_prepro, tmp_step));

                                    // Get level
                                    float level;
                                    if (loopOnLevels)
                                    {
                                        level = preloadLevels[tmp_level];
                                    }
                                    else
                                    {
                                        level = params.GetPreprocessLevel(tmp_step, tmp_ptor, tmp_prepro);
                                    }

                                    // Get time
                                    double timeHours;
                                    if (loopOnTimeHours)
                                    {
                                        timeHours = preloadTimeHours[tmp_hour];
                                    }
                                    else
                                    {
                                        timeHours = params.GetPreprocessTimeHours(tmp_step, tmp_ptor, tmp_prepro);
                                    }

                                    // Correct according to the method
                                    if (method.IsSameAs("Gradients"))
                                    {
                                        // Nothing to change
                                    }
                                    else if (method.IsSameAs("HumidityFlux"))
                                    {
                                        if(tmp_prepro==2) level = 0; // pr_wtr
                                    }
                                    else if (method.IsSameAs("FormerHumidityIndex"))
                                    {
                                        if(tmp_prepro==2) level = 0; // pr_wtr
                                        if(tmp_prepro==3) level = 0; // pr_wtr
                                        if(tmp_prepro==0) timeHours = preloadTimeHours[0];
                                        if(tmp_prepro==1) timeHours = preloadTimeHours[1];
                                        if(tmp_prepro==2) timeHours = preloadTimeHours[0];
                                        if(tmp_prepro==3) timeHours = preloadTimeHours[1];
                                    }

                                    // Date array object instantiation for the data loading. The array has the same length than timeArrayArchive, and the predictor dates are aligned with the target dates, but the dates are not the same.
                                    double ptorStart = timeStartData-double(params.GetTimeShiftDays())+timeHours/24.0;
                                    double ptorEnd = timeEndData-double(params.GetTimeShiftDays())+timeHours/24.0;
                                    asTimeArray timeArray(ptorStart, ptorEnd,
                                                          params.GetTimeArrayAnalogsTimeStepHours(),
                                                          asTimeArray::Simple);
                                    timeArray.Init();

                                    // Loading the datasets information
                                    asDataPredictorArchive* predictorPreprocess = asDataPredictorArchive::GetInstance(params.GetPreprocessDatasetId(tmp_step, tmp_ptor, tmp_prepro),
                                                                                                                      params.GetPreprocessDataId(tmp_step, tmp_ptor, tmp_prepro),
                                                                                                                      m_predictorDataDir);
                                    if(!predictorPreprocess)
                                    {
                                        Cleanup(predictorsPreprocess);
                                        return false;
                                    }

                                    asGeo geo;
                                    double Ymax = params.GetPreloadYmin(tmp_step, tmp_ptor)+params.GetPredictorYstep(tmp_step, tmp_ptor)*(double)(params.GetPreloadYptsnb(tmp_step, tmp_ptor)-1);
                                    if (Ymax > geo.GetAxisYmax())
                                    {
                                        double diff = Ymax-geo.GetAxisYmax();
                                        int removePts = asTools::Round(diff/params.GetPredictorYstep(tmp_step, tmp_ptor));
                                        params.SetPreloadYptsnb(tmp_step, tmp_ptor, params.GetPreloadYptsnb(tmp_step, tmp_ptor)-removePts);
                                        asLogMessage(wxString::Format(_("Adapt Y axis extent according to the maximum allowed (from %.2f to %.2f)."), Ymax, Ymax-diff));
                                    }

                                    // Area object instantiation
                                    asGeoAreaCompositeGrid* area = asGeoAreaCompositeGrid::GetInstance(params.GetPredictorGridType(tmp_step, tmp_ptor),
                                        params.GetPreloadXmin(tmp_step, tmp_ptor),
                                        params.GetPreloadXptsnb(tmp_step, tmp_ptor),
                                        params.GetPredictorXstep(tmp_step, tmp_ptor),
                                        params.GetPreloadYmin(tmp_step, tmp_ptor),
                                        params.GetPreloadYptsnb(tmp_step, tmp_ptor),
                                        params.GetPredictorYstep(tmp_step, tmp_ptor),
                                        level,
                                        asNONE,
                                        params.GetPredictorFlatAllowed(tmp_step, tmp_ptor));
                                    wxASSERT(area);

                                    // Check the starting dates coherence
                                    if (predictorPreprocess->GetOriginalProviderStart()>ptorStart)
                                    {
                                        asLogError(wxString::Format(_("The first year defined in the parameters (%s) is prior to the start date of the data (%s) (in asMethodCalibrator::PreloadData, preprocessing)."),
                                                                    asTime::GetStringTime(ptorStart), asTime::GetStringTime(predictorPreprocess->GetOriginalProviderStart())));
                                        wxDELETE(area);
                                        wxDELETE(predictorPreprocess);
                                        Cleanup(predictorsPreprocess);
                                        return false;
                                    }

                                    // Data loading
                                    asLogMessage(wxString::Format(_("Loading %s data for level %d, %d h."),
                                                                  params.GetPreprocessDataId(tmp_step, tmp_ptor, tmp_prepro),
                                                                  (int)level,
                                                                  (int)timeHours));
                                    if(!predictorPreprocess->Load(area, timeArray))
                                    {
                                        asLogError(_("The data could not be loaded."));
                                        wxDELETE(area);
                                        wxDELETE(predictorPreprocess);
                                        Cleanup(predictorsPreprocess);
                                        return false;
                                    }
                                    wxDELETE(area);
                                    predictorsPreprocess.push_back(predictorPreprocess);
                                }

                                asLogMessage(_("Preprocessing data."));
                                asDataPredictorArchive* predictor = new asDataPredictorArchive(*predictorsPreprocess[0]);
                                try
                                {
                                    if(!asPreprocessor::Preprocess(predictorsPreprocess, params.GetPreprocessMethod(tmp_step, tmp_ptor), predictor))
                                    {
                                        asLogError(_("Data preprocessing failed."));
                                        wxDELETE(predictor);
                                        Cleanup(predictorsPreprocess);
                                        return false;
                                    }
                                    m_preloadedArchive[tmp_step][tmp_ptor][tmp_level][tmp_hour]=predictor;
                                }
								catch (std::bad_alloc& ba)
                                {
                                    m_preloadedArchive[tmp_step][tmp_ptor][tmp_level][tmp_hour]=NULL;
                                    wxString msg(ba.what(), wxConvUTF8);
                                    asLogError(wxString::Format(_("Bad allocation caught in the data preprocessing: %s"), msg));
                                    wxDELETE(predictor);
                                    Cleanup(predictorsPreprocess);
                                    return false;
                                }
								catch (std::exception& e)
                                {
                                    m_preloadedArchive[tmp_step][tmp_ptor][tmp_level][tmp_hour]=NULL;
                                    wxString msg(e.what(), wxConvUTF8);
                                    asLogError(wxString::Format(_("Exception in the data preprocessing: %s"), msg));
                                    wxDELETE(predictor);
                                    Cleanup(predictorsPreprocess);
                                    return false;
                                }
                                Cleanup(predictorsPreprocess);
                                asLogMessage(_("Preprocessing over."));
                            }
                        }

                        // Fix the criteria if S1
                        if (method.IsSameAs("Gradients") && params.GetPredictorCriteria(tmp_step, tmp_ptor).IsSameAs("S1"))
                        {
                            params.SetPredictorCriteria(tmp_step, tmp_ptor, "S1grads");
                        }
                    }
                }
                else // no preloading
                {
                    VectorFloat preloadLevels = params.GetPreloadLevels(tmp_step, tmp_ptor);
                    VectorDouble preloadTimeHours = params.GetPreloadTimeHours(tmp_step, tmp_ptor);

                    int preloadLevelsSize = wxMax(preloadLevels.size(),1);
                    int preloadTimeHoursSize = wxMax(preloadTimeHours.size(),1);

                    m_preloadedArchive[tmp_step][tmp_ptor].resize(preloadLevelsSize);

                    // Load data for every level and every hour
                    for (int tmp_level=0; tmp_level<preloadLevelsSize; tmp_level++)
                    {
                        m_preloadedArchive[tmp_step][tmp_ptor][tmp_level].resize(preloadTimeHoursSize);

                        for (int tmp_hour=0; tmp_hour<preloadTimeHoursSize; tmp_hour++)
                        {
                            m_preloadedArchive[tmp_step][tmp_ptor][tmp_level][tmp_hour]=NULL;
                        }
                    }
                }
            }
        }
    }

    return true;
}

bool asMethodCalibrator::LoadData(std::vector < asDataPredictor* > &predictors, asParametersScoring &params, int i_step, double timeStartData, double timeEndData)
{
    // Loop through every predictor
    for(int i_ptor=0; i_ptor<params.GetPredictorsNb(i_step); i_ptor++)
    {
        if (!PreloadData(params))
        {
            asLogError(_("Could not preload the data."));
            return false;
        }

        if(params.NeedsPreloading(i_step, i_ptor))
        {
            asLogMessage(_("Using preloaded data."));

            bool doPreprocessGradients = false;

            // Get preload arrays
            VectorFloat preloadLevels = params.GetPreloadLevels(i_step, i_ptor);
            VectorDouble preloadTimeHours = params.GetPreloadTimeHours(i_step, i_ptor);
            float level;
            double time;
            int i_level = 0, i_hour = 0;

            if(!params.NeedsPreprocessing(i_step, i_ptor))
            {
                wxASSERT(preloadLevels.size()>0);
                wxASSERT(preloadTimeHours.size()>0);

                level = params.GetPredictorLevel(i_step, i_ptor);
                time = params.GetPredictorTimeHours(i_step, i_ptor);

                // Get level and hour indices
                i_level = asTools::SortedArraySearch(&preloadLevels[0], &preloadLevels[preloadLevels.size()-1], level);
                i_hour = asTools::SortedArraySearch(&preloadTimeHours[0], &preloadTimeHours[preloadTimeHours.size()-1], time);

                // Force gradients proprocessing anyway.
                if (params.GetPredictorCriteria(i_step, i_ptor).IsSameAs("S1"))
                {
                    doPreprocessGradients = true;
                    params.SetPredictorCriteria(i_step, i_ptor, "S1grads");
                }
                else if (params.GetPredictorCriteria(i_step, i_ptor).IsSameAs("S1grads"))
                {
                    doPreprocessGradients = true;
                }
            }
            else
            {
                // Correct according to the method
                if (params.GetPreprocessMethod(i_step, i_ptor).IsSameAs("Gradients"))
                {
                    level = params.GetPreprocessLevel(i_step, i_ptor, 0);
                    time = params.GetPreprocessTimeHours(i_step, i_ptor, 0);
                }
                else if (params.GetPreprocessMethod(i_step, i_ptor).IsSameAs("HumidityFlux"))
                {
                    level = params.GetPreprocessLevel(i_step, i_ptor, 0);
                    time = params.GetPreprocessTimeHours(i_step, i_ptor, 0);
                }
                else if (params.GetPreprocessMethod(i_step, i_ptor).IsSameAs("FormerHumidityIndex"))
                {
                    level = params.GetPreprocessLevel(i_step, i_ptor, 0);
                    time = params.GetPreprocessTimeHours(i_step, i_ptor, 0);
                }
                else
                {
                    level = params.GetPreprocessLevel(i_step, i_ptor, 0);
                    time = params.GetPreprocessTimeHours(i_step, i_ptor, 0);
                }

                // Get level and hour indices
                if (preloadLevels.size()>0)
                {
                    i_level = asTools::SortedArraySearch(&preloadLevels[0], &preloadLevels[preloadLevels.size()-1], level);
                }
                if (preloadTimeHours.size()>0)
                {
                    i_hour = asTools::SortedArraySearch(&preloadTimeHours[0], &preloadTimeHours[preloadTimeHours.size()-1], time);
                }
            }

            // Check indices
            if (i_level==asNOT_FOUND || i_level==asOUT_OF_RANGE)
            {
                asLogError(wxString::Format(_("The level (%f) could not be found in the preloaded data."), level));
                return false;
            }
            if (i_hour==asNOT_FOUND || i_hour==asOUT_OF_RANGE)
            {
                asLogError(wxString::Format(_("The hour (%d) could not be found in the preloaded data."), (int)time));
                return false;
            }

            // Get data on the desired domain
            wxASSERT_MSG((unsigned)i_step<m_preloadedArchive.size(), wxString::Format("i_step=%d, m_preloadedArchive.size()=%d", i_step, (int)m_preloadedArchive.size()));
            wxASSERT_MSG((unsigned)i_ptor<m_preloadedArchive[i_step].size(), wxString::Format("i_ptor=%d, m_preloadedArchive[i_step].size()=%d", i_ptor, (int)m_preloadedArchive[i_step].size()));
            wxASSERT_MSG((unsigned)i_level<m_preloadedArchive[i_step][i_ptor].size(), wxString::Format("i_level=%d, m_preloadedArchive[i_step][i_ptor].size()=%d", i_level, (int)m_preloadedArchive[i_step][i_ptor].size()));
            wxASSERT_MSG((unsigned)i_hour<m_preloadedArchive[i_step][i_ptor][i_level].size(), wxString::Format("i_hour=%d, m_preloadedArchive[i_step][i_ptor][i_level].size()=%d", i_hour, (int)m_preloadedArchive[i_step][i_ptor][i_level].size()));
            ThreadsManager().CritSectionPreloadedData().Enter();
            if (m_preloadedArchive[i_step][i_ptor][i_level][i_hour]==NULL)
            {
                asLogError(_("The pointer to preloaded data is null."));
                return false;
            }
            // Copy the data
            wxASSERT(m_preloadedArchive[i_step][i_ptor][i_level][i_hour]);
            asDataPredictorArchive* desiredPredictor = new asDataPredictorArchive(*m_preloadedArchive[i_step][i_ptor][i_level][i_hour]);
            ThreadsManager().CritSectionPreloadedData().Leave();

            // Area object instantiation
            asGeoAreaCompositeGrid* desiredArea = asGeoAreaCompositeGrid::GetInstance(params.GetPredictorGridType(i_step, i_ptor),
                params.GetPredictorXmin(i_step, i_ptor),
                params.GetPredictorXptsnb(i_step, i_ptor),
                params.GetPredictorXstep(i_step, i_ptor),
                params.GetPredictorYmin(i_step, i_ptor),
                params.GetPredictorYptsnb(i_step, i_ptor),
                params.GetPredictorYstep(i_step, i_ptor),
                params.GetPredictorLevel(i_step, i_ptor),
                asNONE,
                params.GetPredictorFlatAllowed(i_step, i_ptor));

            wxASSERT(desiredArea);

            if(!desiredPredictor->ClipToArea(desiredArea))
            {
                asLogError(_("The data could not be extracted."));
                wxDELETE(desiredArea);
                wxDELETE(desiredPredictor);
                return false;
            }
            wxDELETE(desiredArea);

            if (doPreprocessGradients)
            {
                std::vector < asDataPredictorArchive* > predictorsPreprocess;
                predictorsPreprocess.push_back(desiredPredictor);

                asDataPredictorArchive* newPredictor = new asDataPredictorArchive(*predictorsPreprocess[0]);
                if(!asPreprocessor::Preprocess(predictorsPreprocess, "Gradients", newPredictor))
                {
                   asLogError(_("Data preprocessing failed."));
                   Cleanup(predictorsPreprocess);
                   return false;
                }

                Cleanup(predictorsPreprocess);

                wxASSERT(newPredictor->GetTimeSize()>0);
                predictors.push_back(newPredictor);
                continue;
            }

            wxASSERT(desiredPredictor->GetTimeSize()>0);
            predictors.push_back(desiredPredictor);
        }
        else
        {
            asLogMessage(_("Loading data."));

            if(!params.NeedsPreprocessing(i_step, i_ptor))
            {
                // Date array object instantiation for the data loading. The array has the same length than timeArrayArchive, and the predictor dates are aligned with the target dates, but the dates are not the same.
                double ptorStart = timeStartData-params.GetTimeShiftDays()
                                   +params.GetPredictorTimeHours(i_step, i_ptor)/24.0;
                double ptorEnd = timeEndData-params.GetTimeShiftDays()
                                 +params.GetPredictorTimeHours(i_step, i_ptor)/24.0;
                asTimeArray timeArray(ptorStart, ptorEnd,
                                      params.GetTimeArrayAnalogsTimeStepHours(),
                                      asTimeArray::Simple);
                timeArray.Init();

                // Loading the datasets information
                asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance(params.GetPredictorDatasetId(i_step, i_ptor),
                                                                                        params.GetPredictorDataId(i_step, i_ptor),
                                                                                        m_predictorDataDir);
                if(!predictor)
                {
                    return false;
                }

                // Area object instantiation
                asGeoAreaCompositeGrid* area = asGeoAreaCompositeGrid::GetInstance(params.GetPredictorGridType(i_step, i_ptor),
                    params.GetPredictorXmin(i_step, i_ptor),
                    params.GetPredictorXptsnb(i_step, i_ptor),
                    params.GetPredictorXstep(i_step, i_ptor),
                    params.GetPredictorYmin(i_step, i_ptor),
                    params.GetPredictorYptsnb(i_step, i_ptor),
                    params.GetPredictorYstep(i_step, i_ptor),
                    params.GetPredictorLevel(i_step, i_ptor),
                    asNONE,
                    params.GetPredictorFlatAllowed(i_step, i_ptor));
                wxASSERT(area);

                // Check the starting dates coherence
                if (predictor->GetOriginalProviderStart()>ptorStart)
                {
                    asLogError(wxString::Format(_("The first year defined in the parameters (%s) is prior to the start date of the data (%s) (in asMethodCalibrator::GetAnalogsDates, no preprocessing)."),
                                                asTime::GetStringTime(ptorStart), asTime::GetStringTime(predictor->GetOriginalProviderStart())));
                    wxDELETE(area);
                    wxDELETE(predictor);
                    return false;
                }

                // Data loading
                if(!predictor->Load(area, timeArray))
                {
                    asLogError(_("The data could not be loaded."));
                    wxDELETE(area);
                    wxDELETE(predictor);
                    return false;
                }
                wxDELETE(area);
                predictors.push_back(predictor);
            }
            else
            {
                std::vector < asDataPredictorArchive* > predictorsPreprocess;

                int preprocessSize = params.GetPreprocessSize(i_step, i_ptor);

                asLogMessage(wxString::Format(_("Preprocessing data (%d predictor(s)) while loading."),
                                              preprocessSize));

                for (int i_prepro=0; i_prepro<preprocessSize; i_prepro++)
                {
                    // Date array object instantiation for the data loading. The array has the same length than timeArrayArchive, and the predictor dates are aligned with the target dates, but the dates are not the same.
                    double ptorStart = timeStartData-double(params.GetTimeShiftDays())+params.GetPreprocessTimeHours(i_step, i_ptor, i_prepro)/24.0;
                    double ptorEnd = timeEndData-double(params.GetTimeShiftDays())+params.GetPreprocessTimeHours(i_step, i_ptor, i_prepro)/24.0;
                    asTimeArray timeArray(ptorStart, ptorEnd,
                                          params.GetTimeArrayAnalogsTimeStepHours(),
                                          asTimeArray::Simple);
                    timeArray.Init();

                    // Loading the datasets information
                    asDataPredictorArchive* predictorPreprocess = asDataPredictorArchive::GetInstance(params.GetPreprocessDatasetId(i_step, i_ptor, i_prepro),
                                                                                                             params.GetPreprocessDataId(i_step, i_ptor, i_prepro),
                                                                                                             m_predictorDataDir);
                    if(!predictorPreprocess)
                    {
                        Cleanup(predictorsPreprocess);
                        return false;
                    }

                    // Area object instantiation
                    asGeoAreaCompositeGrid* area = asGeoAreaCompositeGrid::GetInstance(params.GetPredictorGridType(i_step, i_ptor),
                        params.GetPredictorXmin(i_step, i_ptor),
                        params.GetPredictorXptsnb(i_step, i_ptor),
                        params.GetPredictorXstep(i_step, i_ptor),
                        params.GetPredictorYmin(i_step, i_ptor),
                        params.GetPredictorYptsnb(i_step, i_ptor),
                        params.GetPredictorYstep(i_step, i_ptor),
                        params.GetPreprocessLevel(i_step, i_ptor, i_prepro),
                        asNONE,
                        params.GetPredictorFlatAllowed(i_step, i_ptor));
                    wxASSERT(area);

                    // Check the starting dates coherence
                    if (predictorPreprocess->GetOriginalProviderStart()>ptorStart)
                    {
                        asLogError(wxString::Format(_("The first year defined in the parameters (%s) is prior to the start date of the data (%s) (in asMethodCalibrator::GetAnalogsDates, preprocessing)."),
                                                    asTime::GetStringTime(ptorStart), asTime::GetStringTime(predictorPreprocess->GetOriginalProviderStart())));
                        wxDELETE(area);
                        wxDELETE(predictorPreprocess);
                        Cleanup(predictorsPreprocess);
                        return false;
                    }

                    // Data loading
                    if(!predictorPreprocess->Load(area, timeArray))
                    {
                        asLogError(_("The data could not be loaded."));
                        wxDELETE(area);
                        wxDELETE(predictorPreprocess);
                        Cleanup(predictorsPreprocess);
                        return false;
                    }
                    wxDELETE(area);
                    predictorsPreprocess.push_back(predictorPreprocess);
                }

                // Fix the criteria if S1
                if (params.GetPreprocessMethod(i_step, i_ptor).IsSameAs("Gradients") && params.GetPredictorCriteria(i_step, i_ptor).IsSameAs("S1"))
                {
                    params.SetPredictorCriteria(i_step, i_ptor, "S1grads");
                }

                asDataPredictorArchive* predictor = new asDataPredictorArchive(*predictorsPreprocess[0]);
                if(!asPreprocessor::Preprocess(predictorsPreprocess, params.GetPreprocessMethod(i_step, i_ptor), predictor))
                {
                   asLogError(_("Data preprocessing failed."));
                   Cleanup(predictorsPreprocess);
                   return false;
                }

                Cleanup(predictorsPreprocess);
                predictors.push_back(predictor);
            }

            asLogMessage(_("Data loaded"));
        }
    }

    return true;
}

VArray1DFloat asMethodCalibrator::GetClimatologyData(asParametersScoring &params)
{
	VectorInt stationIds = params.GetPredictandStationIds();

	// Get start and end dates
	Array1DDouble predictandTime = m_predictandDB->GetTime();
	float predictandTimeDays = params.GetPredictandTimeHours()/24.0;
	double timeStart, timeEnd;
	timeStart = wxMax(predictandTime[0],params.GetArchiveStart());
	timeStart = floor(timeStart)+predictandTimeDays;
	timeEnd = wxMin(predictandTime[predictandTime.size()-1],params.GetArchiveEnd());
	timeEnd = floor(timeEnd)+predictandTimeDays;

	// Check if data are effectively available for this period
	int indexPredictandTimeStart = asTools::SortedArraySearchCeil(&predictandTime[0],&predictandTime[predictandTime.size()-1],timeStart);
	int indexPredictandTimeEnd = asTools::SortedArraySearchFloor(&predictandTime[0],&predictandTime[predictandTime.size()-1],timeEnd);

	for (int i_st=0; i_st<(int)stationIds.size(); i_st++)
	{
		Array1DFloat predictandDataNorm = m_predictandDB->GetDataNormalizedStation(stationIds[i_st]);

		while (asTools::IsNaN(predictandDataNorm(indexPredictandTimeStart)))
		{
			indexPredictandTimeStart++;
		}
		while (asTools::IsNaN(predictandDataNorm(indexPredictandTimeEnd)))
		{
			indexPredictandTimeEnd--;
		}
	}

	timeStart = predictandTime[indexPredictandTimeStart];
	timeStart = floor(timeStart)+predictandTimeDays;
	timeEnd = predictandTime[indexPredictandTimeEnd];
	timeEnd = floor(timeEnd)+predictandTimeDays;
	indexPredictandTimeStart = asTools::SortedArraySearchCeil(&predictandTime[0],&predictandTime[predictandTime.size()-1],timeStart);
	indexPredictandTimeEnd = asTools::SortedArraySearchFloor(&predictandTime[0],&predictandTime[predictandTime.size()-1],timeEnd);

	// Get index step
	double predictandTimeStep = predictandTime[1]-predictandTime[0];
	double targetTimeStep = params.GetTimeArrayTargetTimeStepHours()/24.0;
	int indexStep = targetTimeStep/predictandTimeStep;

	// Get vector length
	int dataLength = (indexPredictandTimeEnd-indexPredictandTimeStart)/indexStep+1;

	// Process the climatology score
	VArray1DFloat climatologyData(stationIds.size(), Array1DFloat(dataLength));
	for (int i_st=0; i_st<(int)stationIds.size(); i_st++)
	{
		Array1DFloat predictandDataNorm = m_predictandDB->GetDataNormalizedStation(stationIds[i_st]);

		// Set data
		int counter = 0;
		for (int i=indexPredictandTimeStart; i<=indexPredictandTimeEnd; i+=indexStep)
		{
			climatologyData[i_st][counter] = predictandDataNorm[i];
			counter++;
		}
		wxASSERT(dataLength==counter);
	}

	return climatologyData;
}

void asMethodCalibrator::Cleanup(std::vector < asDataPredictorArchive* > predictorsPreprocess)
{
    if (predictorsPreprocess.size()>0)
    {
        for (unsigned int i=0; i<predictorsPreprocess.size(); i++)
        {
            wxDELETE(predictorsPreprocess[i]);
        }
        predictorsPreprocess.resize(0);
    }
}

void asMethodCalibrator::Cleanup(std::vector < asDataPredictor* > predictors)
{
    if (predictors.size()>0)
    {
        for (unsigned int i=0; i<predictors.size(); i++)
        {
            wxDELETE(predictors[i]);
        }
        predictors.resize(0);
    }
}

void asMethodCalibrator::Cleanup(std::vector < asPredictorCriteria* > criteria)
{
    if (criteria.size()>0)
    {
        for (unsigned int i=0; i<criteria.size(); i++)
        {
            wxDELETE(criteria[i]);
        }
        criteria.resize(0);
    }
}

void asMethodCalibrator::DeletePreloadedData()
{
    if (!m_preloaded) return;

    for (unsigned int i=0; i<m_preloadedArchive.size(); i++)
    {
        for (unsigned int j=0; j<m_preloadedArchive[i].size(); j++)
        {
            if(!m_preloadedArchivePointerCopy[i][j])
            {
                for (unsigned int k=0; k<m_preloadedArchive[i][j].size(); k++)
                {
                    for (unsigned int l=0; l<m_preloadedArchive[i][j][k].size(); l++)
                    {
                        wxDELETE(m_preloadedArchive[i][j][k][l]);
                    }
                }
            }
        }
    }

    m_preloaded = false;
}

bool asMethodCalibrator::GetAnalogsDates(asResultsAnalogsDates &results, asParametersScoring &params, int i_step, bool &containsNaNs)
{
    // Get the linear algebra method
    ThreadsManager().CritSectionConfig().Enter();
    int linAlgebraMethod = (int)(wxFileConfig::Get()->Read("/Processing/LinAlgebra", (long)asLIN_ALGEBRA_NOVAR));
    ThreadsManager().CritSectionConfig().Leave();

    // Initialize the result object
    results.SetCurrentStep(i_step);
    results.Init(params);

    // If result file already exists, load it
    if (results.Load())
    {
        return true;
    }

    // Archive date array
    double timeStartArchive = params.GetArchiveStart();
    double timeEndArchive = params.GetArchiveEnd();
    timeStartArchive += std::abs(params.GetTimeShiftDays()); // To avoid having dates before the start of the archive
    timeEndArchive = wxMin(timeEndArchive, timeEndArchive-params.GetTimeSpanDays()); // Adjust so the predictors search won't overtake the array
    asTimeArray timeArrayArchive(timeStartArchive, timeEndArchive,
                                 params.GetTimeArrayAnalogsTimeStepHours(),
                                 asTimeArray::Simple);
    if(params.HasValidationPeriod()) // remove validation years
    {
        timeArrayArchive.SetForbiddenYears(params.GetValidationYearsVector());
    }
    timeArrayArchive.Init();

    // Target date array
    double timeStartCalibration = params.GetCalibrationStart();
    double timeEndCalibration = params.GetCalibrationEnd();
    timeStartCalibration += std::abs(params.GetTimeShiftDays()); // To avoid having dates before the start of the archive
    timeEndCalibration = wxMin(timeEndCalibration, timeEndCalibration-params.GetTimeSpanDays()); // Adjust so the predictors search won't overtake the array
    asTimeArray timeArrayTarget(timeStartCalibration, timeEndCalibration,
                                params.GetTimeArrayTargetTimeStepHours(),
                                params.GetTimeArrayTargetMode());

    if(!m_validationMode)
    {
        if(params.HasValidationPeriod()) // remove validation years
        {
            timeArrayTarget.SetForbiddenYears(params.GetValidationYearsVector());
        }
    }
    
    if(params.GetTimeArrayTargetMode().CmpNoCase("predictand_thresholds")==0 
        || params.GetTimeArrayTargetMode().CmpNoCase("PredictandThresholds")==0)
    {
        VectorInt stations = params.GetPredictandStationIds();
        if (stations.size()>1)
        {
            asLogError(_("You cannot use predictand thresholds with the multivariate approach."));
            return false;
        }

        if(!timeArrayTarget.Init(*m_predictandDB, params.GetTimeArrayTargetPredictandSerieName(), stations[0], params.GetTimeArrayTargetPredictandMinThreshold(), params.GetTimeArrayTargetPredictandMaxThreshold()))
        {
            asLogError(_("The time array mode for the target dates is not correctly defined."));
            return false;
        }
    }
    else
    {
        if(!timeArrayTarget.Init())
        {
            asLogError(_("The time array mode for the target dates is not correctly defined."));
            return false;
        }
    }

    // If in validation mode, only keep validation years
    if(m_validationMode)
    {
        timeArrayTarget.KeepOnlyYears(params.GetValidationYearsVector());
    }

    // Data date array
    double timeStartData = wxMin(timeStartCalibration, timeStartArchive); // Always Jan 1st
    double timeEndData = wxMax(timeEndCalibration, timeEndArchive);
    asTimeArray timeArrayData(timeStartData, timeEndData,
                                 params.GetTimeArrayAnalogsTimeStepHours(),
                                 asTimeArray::Simple);
    timeArrayData.Init();

    // Check on the archive length
    if(timeArrayArchive.GetSize()<100)
    {
        asLogError(wxString::Format(_("The time array is not consistent in asMethodCalibrator::GetAnalogsDates: size=%d."),timeArrayArchive.GetSize()));
        return false;
    }
    asLogMessage(_("Date arrays created."));
/*
    // Calculate needed memory
    wxLongLong neededMem = 0;
    for(int i_ptor=0; i_ptor<params.GetPredictorsNb(i_step); i_ptor++)
    {
        neededMem += (params.GetPredictorXptsnb(i_step, i_ptor))
                     * (params.GetPredictorYptsnb(i_step, i_ptor));
    }
    neededMem *= timeArrayArchive.GetSize(); // time dimension
    neededMem *= 4; // to bytes (for floats)
    double neededMemMb = neededMem.ToDouble();
    neededMemMb /= 1048576.0; // to Mb

    // Get available memory
    wxMemorySize freeMemSize = wxGetFreeMemory();
    wxLongLong freeMem = freeMemSize;
    double freeMemMb = freeMem.ToDouble();
    freeMemMb /= 1048576.0; // To Mb

    if(freeMemSize==-1)
    {
        asLogMessage(wxString::Format(_("Needed memory for data: %.2f Mb (cannot evaluate available memory)"),
                                      neededMemMb));
    }
    else
    {
        asLogMessage(wxString::Format(_("Needed memory for data: %.2f Mb (%.2f Mb available)"),
                                      neededMemMb, freeMemMb));
        if(neededMemMb>freeMemMb)
        {
            asLogError(_("Data cannot fit into available memory."));
            return false;
        }
    }
*/
    // Load the predictor data
    std::vector < asDataPredictor* > predictors;
    if(!LoadData(predictors, params, i_step, timeStartData, timeEndData))
    {
        asLogError(_("Failed loading predictor data."));
        Cleanup(predictors);
        return false;
    }

    // Create the score objects
    std::vector < asPredictorCriteria* > criteria;
    for(int i_ptor=0; i_ptor<params.GetPredictorsNb(i_step); i_ptor++)
    {
        // Instantiate a score object
        asPredictorCriteria* criterion = asPredictorCriteria::GetInstance(params.GetPredictorCriteria(i_step, i_ptor),
                                                                          linAlgebraMethod);
        criteria.push_back(criterion);
    }

    // Check time sizes
    #ifdef _DEBUG
        int prevTimeSize = 0;

        for (unsigned int i=0; i<predictors.size(); i++)
        {
            if (i>0)
            {
                wxASSERT(predictors[i]->GetTimeSize()==prevTimeSize);
            }
            prevTimeSize = predictors[i]->GetTimeSize();
        }
    #endif // _DEBUG

    // Inline the data when possible
    for (int i_ptor=0; i_ptor<(int)predictors.size(); i_ptor++)
    {
        if (criteria[i_ptor]->CanUseInline())
        {
            predictors[i_ptor]->Inline();
        }
    }

    // Send data and criteria to processor
    asLogMessage(_("Start processing the comparison."));

    if(!asProcessor::GetAnalogsDates(predictors, predictors,
                                     timeArrayData, timeArrayArchive, timeArrayData, timeArrayTarget,
                                     criteria, params, i_step, results, containsNaNs))
    {
        asLogError(_("Failed processing the analogs dates."));
        Cleanup(predictors);
        Cleanup(criteria);
        return false;
    }
    asLogMessage(_("The processing is over."));

    // Saving intermediate results
    results.Save();

    Cleanup(predictors);
    Cleanup(criteria);

    return true;
}

bool asMethodCalibrator::GetAnalogsSubDates(asResultsAnalogsDates &results, asParametersScoring &params, asResultsAnalogsDates &anaDates, int i_step, bool &containsNaNs)
{
    // Get the linear algebra method
    ThreadsManager().CritSectionConfig().Enter();
    int linAlgebraMethod = (int)(wxFileConfig::Get()->Read("/Processing/LinAlgebra", (long)asLIN_ALGEBRA_NOVAR));
    ThreadsManager().CritSectionConfig().Leave();

    // Initialize the result object
    results.SetCurrentStep(i_step);
    results.Init(params);

    // If result file already exists, load it
    if (results.Load()) return true;

    // Date array object instantiation for the processor
    asLogMessage(_("Creating a date arrays for the processor."));
    double timeStart = params.GetArchiveStart();
    double timeEnd = params.GetArchiveEnd();
    timeEnd = wxMin(timeEnd, timeEnd-params.GetTimeSpanDays()); // Adjust so the predictors search won't overtake the array
    asTimeArray timeArrayArchive(timeStart, timeEnd, params.GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
    timeArrayArchive.Init();
    asLogMessage(_("Date arrays created."));

    // Load the predictor data
    std::vector < asDataPredictor* > predictors;
    if(!LoadData(predictors, params, i_step, timeStart, timeEnd))
    {
        asLogError(_("Failed loading predictor data."));
        Cleanup(predictors);
        return false;
    }

    // Create the score objects
    std::vector < asPredictorCriteria* > criteria;
    for(int i_ptor=0; i_ptor<params.GetPredictorsNb(i_step); i_ptor++)
    {
        asLogMessage(_("Creating a criterion object."));
        asPredictorCriteria* criterion = asPredictorCriteria::GetInstance(params.GetPredictorCriteria(i_step, i_ptor), linAlgebraMethod);
        criteria.push_back(criterion);
        asLogMessage(_("Criterion object created."));

    }

    // Inline the data when possible
    for (int i_ptor=0; i_ptor<(int)predictors.size(); i_ptor++)
    {
        if (criteria[i_ptor]->CanUseInline())
        {
            predictors[i_ptor]->Inline();
        }
    }

    // Send data and criteria to processor
    asLogMessage(_("Start processing the comparison."));
    if(!asProcessor::GetAnalogsSubDates(predictors, predictors, timeArrayArchive, timeArrayArchive, anaDates, criteria, params, i_step, results, containsNaNs))
    {
        asLogError(_("Failed processing the analogs dates."));
        Cleanup(predictors);
        Cleanup(criteria);
        return false;
    }
    asLogMessage(_("The processing is over."));

    // Saving intermediate results
    results.Save();

    Cleanup(predictors);
    Cleanup(criteria);

    return true;
}

bool asMethodCalibrator::GetAnalogsValues(asResultsAnalogsValues &results, asParametersScoring &params, asResultsAnalogsDates &anaDates, int i_step)
{
    // Initialize the result object
    results.SetCurrentStep(i_step);
    results.Init(params);

    // If result file already exists, load it
    if (results.Load()) return true;

    // Set the predictands values to the corresponding analog dates
    wxASSERT(m_predictandDB);
    asLogMessage(_("Start setting the predictand values to the corresponding analog dates."));
    if(!asProcessor::GetAnalogsValues(*m_predictandDB, anaDates, params, results))
    {
        asLogError(_("Failed setting the predictand values to the corresponding analog dates."));
        return false;
    }
    asLogMessage(_("Predictands association over."));

    // Saving intermediate results
    results.Save();

    return true;
}

bool asMethodCalibrator::GetAnalogsForecastScores(asResultsAnalogsForecastScores &results, asParametersScoring &params, asResultsAnalogsValues &anaValues, int i_step)
{
    // Initialize the result object
    results.SetCurrentStep(i_step);
    results.Init(params);

    // If result file already exists, load it
    if (results.Load()) return true;

    // Instantiate a forecast score object
    asLogMessage(_("Instantiating a forecast score object"));
    asForecastScore* forecastScore = asForecastScore::GetInstance(params.GetForecastScoreName());
    forecastScore->SetQuantile(params.GetForecastScoreQuantile());
    forecastScore->SetThreshold(params.GetForecastScoreThreshold());

    if (forecastScore->UsesClimatology() && m_scoreClimatology.size()==0)
    {
        asLogMessage(_("Processing the score of the climatology."));

        VArray1DFloat climatologyData = GetClimatologyData(params);
        VectorInt stationIds = params.GetPredictandStationIds();
        m_scoreClimatology.resize(stationIds.size());

        for (int i_st=0; i_st<(int)stationIds.size(); i_st++)
        {
            forecastScore->ProcessScoreClimatology(anaValues.GetTargetValues()[i_st], climatologyData[i_st]);
            m_scoreClimatology[i_st] = forecastScore->GetScoreClimatology();
        }
    }

    // Pass data and score to processor
    asLogMessage(_("Start processing the forecast score."));

    if(!asProcessorForecastScore::GetAnalogsForecastScores(anaValues, forecastScore, params, results, m_scoreClimatology))
    {
        asLogError(_("Failed processing the forecast score."));
        wxDELETE(forecastScore);
        return false;
    }

    // Saving intermediate results
    results.Save();

    wxDELETE(forecastScore);

    return true;
}

bool asMethodCalibrator::GetAnalogsForecastScoreFinal(asResultsAnalogsForecastScoreFinal &results, asParametersScoring &params, asResultsAnalogsForecastScores &anaScores, int i_step)
{
    // Initialize the result object
    results.SetCurrentStep(i_step);
    results.Init(params);

    // If result file already exists, load it
    if (results.Load()) return true;

    // Date array object instantiation for the final score
    asLogMessage(_("Creating a date array for the final score."));
    double timeStart = params.GetCalibrationStart();
    double timeEnd = params.GetCalibrationEnd()+1;
    while (timeEnd>params.GetCalibrationEnd()+0.999)
    {
        timeEnd -= params.GetTimeArrayTargetTimeStepHours()/24.0;
    }
    asTimeArray timeArray(timeStart, timeEnd, params.GetTimeArrayTargetTimeStepHours(), params.GetForecastScoreTimeArrayMode());

// TODO (phorton#1#): Fix me: add every options for the Init function (generic version)
//    timeArray.Init(params.GetForecastScoreTimeArrayDate(), params.GetForecastScoreTimeArrayIntervalDays());
    timeArray.Init();
    asLogMessage(_("Date array created."));

    // Pass data and score to processor
    asLogMessage(_("Start processing the final score."));
    if(!asProcessorForecastScore::GetAnalogsForecastScoreFinal(anaScores, timeArray, params, results))
    {
        asLogError(_("Failed to process the final score."));
        return false;
    }
    asLogMessage(_("Processing over."));

    // Saving intermediate results
    results.Save();

    return true;
}

bool asMethodCalibrator::SubProcessAnalogsNumber(asParametersCalibration &params, asResultsAnalogsDates &anaDatesPrevious, int i_step )
{
    VectorInt analogsNbVect = params.GetAnalogsNumberVector(i_step);

    // Cannot be superior to previous analogs nb
    int rowEnd = analogsNbVect.size()-1;
    if (i_step>0)
    {
        int prevAnalogsNb = params.GetAnalogsNumber(i_step-1);
        if (prevAnalogsNb<analogsNbVect[analogsNbVect.size()-1])
        {
            rowEnd = asTools::SortedArraySearchFloor(&analogsNbVect[0],
                     &analogsNbVect[analogsNbVect.size()-1],
                     prevAnalogsNb);
        }
    }

    asResultsAnalogsDates anaDates;
    asResultsAnalogsValues anaValues;

    // If at the end of the chain
    if (i_step==params.GetStepsNb()-1)
    {
        // Set the maximum and let play with the analogs nb on the forecast score (faster)
        params.SetAnalogsNumber(i_step, analogsNbVect[rowEnd]);

        // Process first the dates and the values
        bool containsNaNs = false;
        if (i_step==0)
        {
            if(!GetAnalogsDates(anaDates, params, i_step, containsNaNs)) return false;
        }
        else
        {
            if(!GetAnalogsSubDates(anaDates, params, anaDatesPrevious, i_step, containsNaNs))
                return false;
        }
        if (containsNaNs)
        {
            asLogError(_("The dates selection contains NaNs"));
            return false;
        }
        if(!GetAnalogsValues(anaValues, params, anaDates, i_step))
            return false;

        asResultsAnalogsForecastScores anaScores;
        asResultsAnalogsForecastScoreFinal anaScoreFinal;
        asResultsAnalogsDates anaDatesTmp(anaDates);
        Array2DFloat dates = anaDates.GetAnalogsDates();

        for (int i_anb=0; i_anb<=rowEnd; i_anb++)
        {
            params.SetAnalogsNumber(i_step, analogsNbVect[i_anb]);

            // Fixes and checks
            params.FixAnalogsNb();
            
            // Extract analogs dates from former results
            Array2DFloat subDates = dates.leftCols(params.GetAnalogsNumber(i_step));
            anaDatesTmp.SetAnalogsDates(subDates);

            if(!GetAnalogsForecastScores(anaScores, params, anaValues, i_step))
                return false;
            if(!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, i_step))
                return false;

            m_parametersTemp.push_back(params);
            m_scoresCalibTemp.push_back(anaScoreFinal.GetForecastScore());
        }

    }
    else
    {
        int nextStep = i_step+1;

        for (int i_anb=0; i_anb<=rowEnd; i_anb++)
        {
            params.SetAnalogsNumber(i_step, analogsNbVect[i_anb]);

            // Fixes and checks
            params.FixAnalogsNb();

            // Process the dates and the values
            bool containsNaNs = false;
            if (i_step==0)
            {
                if(!GetAnalogsDates(anaDates, params, i_step, containsNaNs))
                    return false;
            }
            else
            {
                if(!GetAnalogsSubDates(anaDates, params, anaDatesPrevious, i_step, containsNaNs))
                    return false;
            }
            if (containsNaNs)
            {
                asLogError(_("The dates selection contains NaNs"));
                return false;
            }

            // Continue
            if(!SubProcessAnalogsNumber(params, anaDates, nextStep ))
                return false;
        }
    }

    return true;
}

bool asMethodCalibrator::Validate(const int bestscorerow)
{
    bool skipValidation = false;
    wxFileConfig::Get()->Read("/Optimizer/SkipValidation", &skipValidation, false);

    if (skipValidation)
    {
        return true;
    }

    if (m_parameters.size()==0)
    {
        asLogError("The parameters array is empty in the validation procedure.");
        return false;
    }
    else if (m_parameters.size()<unsigned(bestscorerow+1))
    {
        asLogError("Trying to access parameters outside the array in the validation procedure.");
        return false;
    }

    if (!m_parameters[bestscorerow].HasValidationPeriod())
    {
        asLogWarning("The parameters have no validation period !");
        return false;
    }

    m_validationMode = true;

    asResultsAnalogsDates anaDatesPrevious;
    asResultsAnalogsDates anaDates;
    asResultsAnalogsValues anaValues;
    asResultsAnalogsForecastScores anaScores;
    asResultsAnalogsForecastScoreFinal anaScoreFinal;

    // Process every step one after the other
    int stepsNb = m_parameters[bestscorerow].GetStepsNb();
    for (int i_step=0; i_step<stepsNb; i_step++)
    {
        bool containsNaNs = false;
        if (i_step==0)
        {
            if(!GetAnalogsDates(anaDates, m_parameters[bestscorerow], i_step, containsNaNs)) return false;
        }
        else
        {
            anaDatesPrevious = anaDates;
            if(!GetAnalogsSubDates(anaDates, m_parameters[bestscorerow], anaDatesPrevious, i_step, containsNaNs)) return false;
        }
        if (containsNaNs)
        {
            asLogError(_("The dates selection contains NaNs"));
            return false;
        }
    }
    if(!GetAnalogsValues(anaValues, m_parameters[bestscorerow], anaDates, stepsNb-1)) return false;
    if(!GetAnalogsForecastScores(anaScores, m_parameters[bestscorerow], anaValues, stepsNb-1)) return false;
    if(!GetAnalogsForecastScoreFinal(anaScoreFinal, m_parameters[bestscorerow], anaScores, stepsNb-1)) return false;

    m_scoreValid = anaScoreFinal.GetForecastScore();

    m_validationMode = false;

    return true;
}