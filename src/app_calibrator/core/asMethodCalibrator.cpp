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

#include "asMethodCalibrator.h"

#ifndef UNIT_TESTING
    #include "AtmoswingAppCalibrator.h"
#endif

asMethodCalibrator::asMethodCalibrator()
    :
    asMethodStandard()
{
    m_PredictorDataDir = wxEmptyString;
    m_ParamsFilePath = wxEmptyString;
    m_ScoreClimatology = 0;
    m_Preloaded = false;
    m_ValidationMode = false;
    m_ScoreValid = NaNFloat;

    // Seeds the random generator
    asTools::InitRandom();
}

asMethodCalibrator::~asMethodCalibrator()
{
    DeletePreloadedData();
    Cleanup();
}

bool asMethodCalibrator::Manager()
{
    // Set unresponsive to speedup
    g_Responsive = false;

    // Reset the score of the climatology
    m_ScoreClimatology = 0;

    // Seeds the random generator
    asTools::InitRandom();

    // Load parameters
    asParametersCalibration params;
    if(!params.LoadFromFile(m_ParamsFilePath)) return false;
    params.InitValues();
    m_OriginalParams = params;

    // Load the Predictand DB
    asLogMessage(_("Loading the Predictand DB."));
    if(!LoadPredictandDB(m_PredictandDBFilePath)) return false;
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
    Cleanup();

    return true;
}

void asMethodCalibrator::ClearAll()
{
    m_ParametersTemp.clear();
    m_ScoresCalibTemp.clear();
    m_Parameters.clear();
    m_ScoresCalib.clear();
    m_ScoreValid = NaNFloat;
}

void asMethodCalibrator::ClearTemp()
{
    m_ParametersTemp.clear();
    m_ScoresCalibTemp.clear();
}

void asMethodCalibrator::ClearScores()
{
    for (unsigned int i=0; i<m_ScoresCalib.size(); i++)
    {
        m_ScoresCalib[i] = NaNFloat;
    }
    m_ScoreValid = NaNFloat;
}

void asMethodCalibrator::PushBackBestTemp()
{
    SortScoresAndParametersTemp();
    PushBackFirstTemp();
}

void asMethodCalibrator::RemoveNaNsInTemp()
{
    wxASSERT(m_ParametersTemp.size()==m_ScoresCalibTemp.size());

    std::vector <asParametersCalibration> CopyParametersTemp;
    VectorFloat CopyScoresCalibTemp;

    for (unsigned int i=0; i<m_ScoresCalibTemp.size(); i++)
    {
        if (!asTools::IsNaN(m_ScoresCalibTemp[i]))
        {
            CopyScoresCalibTemp.push_back(m_ScoresCalibTemp[i]);
            CopyParametersTemp.push_back(m_ParametersTemp[i]);
        }
    }

    m_ScoresCalibTemp = CopyScoresCalibTemp;
    m_ParametersTemp = CopyParametersTemp;

    wxASSERT(m_ParametersTemp.size()==m_ScoresCalibTemp.size());
    wxASSERT(m_ParametersTemp.size()>0);
}

void asMethodCalibrator::KeepBestTemp()
{
    SortScoresAndParametersTemp();
    KeepFirstTemp();
}

void asMethodCalibrator::PushBackFirstTemp()
{
    m_Parameters.push_back(m_ParametersTemp[0]);
    m_ScoresCalib.push_back(m_ScoresCalibTemp[0]);
}

void asMethodCalibrator::KeepFirstTemp()
{
    wxASSERT(m_Parameters.size()>0);
    wxASSERT(m_ParametersTemp.size()>0);
    wxASSERT(m_ScoresCalibTemp.size()>0);
    m_Parameters[0] = m_ParametersTemp[0];
    if (m_ScoresCalib.size()==0)
    {
        m_ScoresCalib.push_back(m_ScoresCalibTemp[0]);
    }
    else
    {
        m_ScoresCalib[0] = m_ScoresCalibTemp[0];
    }

}

void asMethodCalibrator::SortScoresAndParameters()
{
    wxASSERT(m_ScoresCalib.size()==m_Parameters.size());
    wxASSERT(m_ScoresCalib.size()>=1);
    wxASSERT(m_Parameters.size()>=1);

    if (m_Parameters.size()==1) return;

    // Sort according to the score
    Array1DFloat vIndices = Array1DFloat::LinSpaced(Eigen::Sequential,m_ScoresCalib.size(),
                            0,m_ScoresCalib.size()-1);
    asTools::SortArrays(&m_ScoresCalib[0], &m_ScoresCalib[m_ScoresCalib.size()-1],
                        &vIndices[0], &vIndices[m_ScoresCalib.size()-1],
                        m_ScoreOrder);

    // Sort the parameters sets as the scores
    std::vector <asParametersCalibration> copyParameters;
    for (unsigned int i=0; i<m_ScoresCalib.size(); i++)
    {
        copyParameters.push_back(m_Parameters[i]);
    }
    for (unsigned int i=0; i<m_ScoresCalib.size(); i++)
    {
        int index = vIndices(i);
        m_Parameters[i] = copyParameters[index];
    }
}

void asMethodCalibrator::SortScoresAndParametersTemp()
{
    wxASSERT(m_ScoresCalibTemp.size()==m_ParametersTemp.size());
    wxASSERT(m_ScoresCalibTemp.size()>0);
    wxASSERT(m_ParametersTemp.size()>0);

    if (m_ParametersTemp.size()==1) return;

    // Sort according to the score
    Array1DFloat vIndices = Array1DFloat::LinSpaced(Eigen::Sequential,m_ScoresCalibTemp.size(),
                            0,m_ScoresCalibTemp.size()-1);
    asTools::SortArrays(&m_ScoresCalibTemp[0], &m_ScoresCalibTemp[m_ScoresCalibTemp.size()-1],
                        &vIndices[0], &vIndices[m_ScoresCalibTemp.size()-1],
                        m_ScoreOrder);

    // Sort the parameters sets as the scores
    std::vector <asParametersCalibration> copyParameters;
    for (unsigned int i=0; i<m_ScoresCalibTemp.size(); i++)
    {
        copyParameters.push_back(m_ParametersTemp[i]);
    }
    for (unsigned int i=0; i<m_ScoresCalibTemp.size(); i++)
    {
        int index = vIndices(i);
        m_ParametersTemp[i] = copyParameters[index];
    }
}

bool asMethodCalibrator::PushBackInTempIfBetter(asParametersCalibration &params, asResultsAnalogsForecastScoreFinal &scoreFinal)
{
    float thisScore = scoreFinal.GetForecastScore();

    switch (m_ScoreOrder)
    {
    case Asc:
        if (thisScore<m_ScoresCalib[0])
        {
            m_ParametersTemp.push_back(params);
            m_ScoresCalibTemp.push_back(thisScore);
            return true;
        }
        break;

    case Desc:
        if (thisScore>m_ScoresCalib[0])
        {
            m_ParametersTemp.push_back(params);
            m_ScoresCalibTemp.push_back(thisScore);
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

    switch (m_ScoreOrder)
    {
    case Asc:
        if (thisScore<m_ScoresCalib[0])
        {
            wxASSERT(m_Parameters.size()>0);
            wxASSERT(m_ScoresCalib.size()>0);
            m_Parameters[0] = params;
            m_ScoresCalib[0] = thisScore;
            return true;
        }
        break;

    case Desc:
        if (thisScore>m_ScoresCalib[0])
        {
            wxASSERT(m_Parameters.size()>0);
            wxASSERT(m_ScoresCalib.size()>0);
            m_Parameters[0] = params;
            m_ScoresCalib[0] = thisScore;
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
    for (unsigned int i=0; i<m_Parameters.size(); i++)
    {
        results.Add(m_Parameters[i],m_ScoresCalib[i],m_ScoreValid);
    }

    return true;
}

bool asMethodCalibrator::SetBestParameters(asResultsParametersArray &results)
{
    wxASSERT(m_Parameters.size()>0);
    wxASSERT(m_ScoresCalib.size()>0);

    // Extract selected parameters & best parameters
    float bestscore = m_ScoresCalib[0];
    int bestscorerow = 0;

    for (unsigned int i=0; i<m_Parameters.size(); i++)
    {
        if(m_ScoreOrder==Asc)
        {
            if(m_ScoresCalib[i]<bestscore)
            {
                bestscore = m_ScoresCalib[i];
                bestscorerow = i;
            }
        }
        else
        {
            if(m_ScoresCalib[i]>bestscore)
            {
                bestscore = m_ScoresCalib[i];
                bestscorerow = i;
            }
        }
    }

    if (bestscorerow!=0)
    {
        // Re-validate
        Validate(bestscorerow);
    }

    results.Add(m_Parameters[bestscorerow],m_ScoresCalib[bestscorerow],m_ScoreValid);

    return true;
}

bool asMethodCalibrator::PreloadData(asParametersScoring &params)
{
    // Load data once.
    if(!m_Preloaded)
    {
        // Set preload to true here, so cleanup is made in case of exceptions.
        m_Preloaded = true;

        // Get first year
        int yearStart = wxMin(params.GetArchiveYearStart(), params.GetCalibrationYearStart());
        int validationStart = params.GetValidationYearStart();
        if (!asTools::IsNaN(validationStart)) {
            yearStart = wxMin(yearStart, validationStart);
        }

        // Get last year
        int yearEnd = wxMax(params.GetArchiveYearEnd(), params.GetCalibrationYearEnd());
        int validationEnd = params.GetValidationYearEnd();
        if (!asTools::IsNaN(validationEnd)) {
            yearEnd = wxMax(yearEnd, validationEnd);
        }

        // Get dates
        double timeStartData = asTime::GetMJD(yearStart,1,1); // Always Jan 1st
        double timeEndData = asTime::GetMJD(yearEnd,12,31); // Always Dec 31
        asTimeArray timeArrayData(timeStartData, timeEndData,
                                     params.GetTimeArrayAnalogsTimeStepHours(),
                                     asTimeArray::Simple);
        timeArrayData.Init();

        // Resize container
        if (m_PreloadedArchive.size()==0)
        {
            m_PreloadedArchive.resize(params.GetStepsNb());
            m_PreloadedArchivePointerCopy.resize(params.GetStepsNb());
            for (int tmp_step=0; tmp_step<params.GetStepsNb(); tmp_step++)
            {
                m_PreloadedArchive[tmp_step].resize(params.GetPredictorsNb(tmp_step));
                m_PreloadedArchivePointerCopy[tmp_step].resize(params.GetPredictorsNb(tmp_step));

                for (int tmp_ptor=0; tmp_ptor<params.GetPredictorsNb(tmp_step); tmp_ptor++)
                {
                    m_PreloadedArchivePointerCopy[tmp_step][tmp_ptor] = false;
                    m_PreloadedArchive[tmp_step][tmp_ptor].resize(1);
                    m_PreloadedArchive[tmp_step][tmp_ptor][0].resize(1);
                    m_PreloadedArchive[tmp_step][tmp_ptor][0][0]=NULL;
                }
            }
        }

        for (int tmp_step=0; tmp_step<params.GetStepsNb(); tmp_step++)
        {
            for (int tmp_ptor=0; tmp_ptor<params.GetPredictorsNb(tmp_step); tmp_ptor++)
            {
                if(params.NeedsPreloading(tmp_step, tmp_ptor))
                {
                    if(!params.NeedsPreprocessing(tmp_step, tmp_ptor))
                    {
                        asLogMessage(wxString::Format(_("Preloading data for predictor %d of step %d."), tmp_ptor, tmp_step));

                        // Try to share pointers
                        if(tmp_ptor>0)
                        {
                            int prev = 0;
                            bool share = false;

                            for (prev=0;prev<tmp_ptor;prev++)
                            {
                                share = true;

                                if(!params.GetPredictorDatasetId(tmp_step, tmp_ptor).IsSameAs(params.GetPredictorDatasetId(tmp_step, prev), false)) share = false;
                                if(!params.GetPredictorDataId(tmp_step, tmp_ptor).IsSameAs(params.GetPredictorDataId(tmp_step, prev))) share = false;
                                if(!params.GetPredictorGridType(tmp_step, tmp_ptor).IsSameAs(params.GetPredictorGridType(tmp_step, prev))) share = false;
                                if(params.GetPreloadUmin(tmp_step, tmp_ptor)!=params.GetPreloadUmin(tmp_step, prev)) share = false;
                                if(params.GetPreloadUptsnb(tmp_step, tmp_ptor)!=params.GetPreloadUptsnb(tmp_step, prev)) share = false;
                                if(params.GetPredictorUstep(tmp_step, tmp_ptor)!=params.GetPredictorUstep(tmp_step, prev)) share = false;
                                if(params.GetPreloadVmin(tmp_step, tmp_ptor)!=params.GetPreloadVmin(tmp_step, prev)) share = false;
                                if(params.GetPreloadVptsnb(tmp_step, tmp_ptor)!=params.GetPreloadVptsnb(tmp_step, prev)) share = false;
                                if(params.GetPredictorVstep(tmp_step, tmp_ptor)!=params.GetPredictorVstep(tmp_step, prev)) share = false;
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

                                m_PreloadedArchivePointerCopy[tmp_step][tmp_ptor] = true;
                                m_PreloadedArchive[tmp_step][tmp_ptor].resize(preloadLevels.size());

                                wxASSERT(m_PreloadedArchive[tmp_step].size()>(unsigned)prev);
                                wxASSERT(m_PreloadedArchive[tmp_step][prev].size()==preloadLevels.size());

                                // Load data for every level and every hour
                                for (unsigned int tmp_level=0; tmp_level<preloadLevels.size(); tmp_level++)
                                {
                                    m_PreloadedArchive[tmp_step][tmp_ptor][tmp_level].resize(preloadTimeHours.size());
                                    wxASSERT(m_PreloadedArchive[tmp_step][prev][tmp_level].size()==preloadTimeHours.size());

                                    for (unsigned int tmp_hour=0; tmp_hour<preloadTimeHours.size(); tmp_hour++)
                                    {
                                        // Copy pointer
                                        m_PreloadedArchive[tmp_step][tmp_ptor][tmp_level][tmp_hour]=m_PreloadedArchive[tmp_step][prev][tmp_level][tmp_hour];
                                    }
                                }

                                params.SetPreloadVptsnb(tmp_step, tmp_ptor, params.GetPreloadVptsnb(tmp_step, prev));

                                continue;
                            }
                        }

                        // Loading the datasets information
                        asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance(params.GetPredictorDatasetId(tmp_step, tmp_ptor), params.GetPredictorDataId(tmp_step, tmp_ptor), m_PredictorDataDir);
                        if(!predictor) {
                            return false;
                        }

                        VectorFloat preloadLevels = params.GetPreloadLevels(tmp_step, tmp_ptor);
                        VectorDouble preloadTimeHours = params.GetPreloadTimeHours(tmp_step, tmp_ptor);
                        wxASSERT(preloadLevels.size()>0);
                        wxASSERT(preloadTimeHours.size()>0);

                        // Resize container and set null pointers
                        m_PreloadedArchive[tmp_step][tmp_ptor].resize(preloadLevels.size());
                        for (unsigned int tmp_level=0; tmp_level<preloadLevels.size(); tmp_level++)
                        {
                            m_PreloadedArchive[tmp_step][tmp_ptor][tmp_level].resize(preloadTimeHours.size());
                            for (unsigned int tmp_hour=0; tmp_hour<preloadTimeHours.size(); tmp_hour++)
                            {
                                m_PreloadedArchive[tmp_step][tmp_ptor][tmp_level][tmp_hour] = NULL;
                            }
                        }

                        // Load data for every level and every hour
                        for (unsigned int tmp_level=0; tmp_level<preloadLevels.size(); tmp_level++)
                        {
                            for (unsigned int tmp_hour=0; tmp_hour<preloadTimeHours.size(); tmp_hour++)
                            {
                                asGeo geo(predictor->GetCoordSys());
                                double Vmax = params.GetPreloadVmin(tmp_step, tmp_ptor)+params.GetPredictorVstep(tmp_step, tmp_ptor)*(double)(params.GetPreloadVptsnb(tmp_step, tmp_ptor)-1);
                                if (Vmax > geo.GetAxisVmax())
                                {
                                    double diff = Vmax-geo.GetAxisVmax();
                                    int removePts = asTools::Round(diff/params.GetPredictorVstep(tmp_step, tmp_ptor));
                                    params.SetPreloadVptsnb(tmp_step, tmp_ptor, params.GetPreloadVptsnb(tmp_step, tmp_ptor)-removePts);
                                    asLogMessage(wxString::Format(_("Adapt V axis extent according to the maximum allowed (from %.3f to %.3f)."), Vmax, Vmax-diff));
                                    asLogMessage(wxString::Format(_("Remove %d points (%.3f-%.3f)/%.3f."), removePts, Vmax, geo.GetAxisVmax(), params.GetPredictorVstep(tmp_step, tmp_ptor)));
                                }

                                wxASSERT(params.GetPreloadUptsnb(tmp_step, tmp_ptor)>0);
                                wxASSERT(params.GetPreloadVptsnb(tmp_step, tmp_ptor)>0);

                                // Area object instantiation
                                asGeoAreaCompositeGrid* area = asGeoAreaCompositeGrid::GetInstance(predictor->GetCoordSys(),
                                                               params.GetPredictorGridType(tmp_step, tmp_ptor),
                                                               params.GetPreloadUmin(tmp_step, tmp_ptor),
                                                               params.GetPreloadUptsnb(tmp_step, tmp_ptor),
                                                               params.GetPredictorUstep(tmp_step, tmp_ptor),
                                                               params.GetPreloadVmin(tmp_step, tmp_ptor),
                                                               params.GetPreloadVptsnb(tmp_step, tmp_ptor),
                                                               params.GetPredictorVstep(tmp_step, tmp_ptor),
                                                               preloadLevels[tmp_level],
                                                               asNONE,
                                                               params.GetPredictorFlatAllowed(tmp_step, tmp_ptor));
                                wxASSERT(area);

                                // Check the starting dates coherence
                                if (predictor->GetOriginalProviderStart()>timeStartData)
                                {
                                    asLogError(wxString::Format(_("The first year defined in the parameters (%s) is prior to the start date of the data (%s) (in asMethodCalibrator::PreloadData)."),
                                                                asTime::GetStringTime(timeStartData), asTime::GetStringTime(predictor->GetOriginalProviderStart())));
                                    wxDELETE(area);
                                    wxDELETE(predictor);
                                    return false;
                                }

                                // Data loading
                                asLogMessage(wxString::Format(_("Loading %s data for level %d, %d h."), params.GetPredictorDataId(tmp_step, tmp_ptor).c_str(), (int)preloadLevels[tmp_level], (int)preloadTimeHours[tmp_hour]));
                                try
                                {
                                    if(!predictor->Load(area, timeArrayData))
                                    {
                                        asLogError(_("The data could not be loaded."));
                                        wxDELETE(area);
                                        wxDELETE(predictor);
                                        return false;
                                    }
                                }
                                catch(bad_alloc& ba)
                                {
                                    wxString msg(ba.what(), wxConvUTF8);
                                    asLogError(wxString::Format(_("Bad allocation in the data preloading: %s"), msg.c_str()));
                                    wxDELETE(area);
                                    wxDELETE(predictor);
                                    return false;
                                }
                                catch (exception& e)
                                {
                                    wxString msg(e.what(), wxConvUTF8);
                                    asLogError(wxString::Format(_("Exception in the data preloading: %s"), msg.c_str()));
                                    wxDELETE(area);
                                    wxDELETE(predictor);
                                    return false;
                                }
                                asLogMessage(_("Data loaded."));
                                wxDELETE(area);

                                m_PreloadedArchive[tmp_step][tmp_ptor][tmp_level][tmp_hour]=predictor;
                            }
                        }
                    }
                    else
                    {
                        asLogMessage(wxString::Format(_("Preloading data for predictor preprocessed %d of step %d."), tmp_ptor, tmp_step));

                        int preprocessSize = params.GetPreprocessSize(tmp_step, tmp_ptor);

                        asLogMessage(wxString::Format(_("Preprocessing data (%d predictor(s)) while loading."),
                                                      preprocessSize));

                        for (int tmp_prepro=0; tmp_prepro<preprocessSize; tmp_prepro++)
                        {
                            asLogMessage(wxString::Format(_("Preloading data for predictor %d (preprocess %d) of step %d."), tmp_ptor, tmp_prepro, tmp_step));

                            // Loading the datasets information
                            asDataPredictorArchive* predictorPreprocess = asDataPredictorArchive::GetInstance(params.GetPreprocessDatasetId(tmp_step, tmp_ptor, tmp_prepro),
                                                                                                              params.GetPreprocessDataId(tmp_step, tmp_ptor, tmp_prepro),
                                                                                                              m_PredictorDataDir);
                            if(!predictorPreprocess)
                            {
                                return false;
                            }

                            asLogMessage(_("Creating maximum area."));

                            asGeo geo(predictorPreprocess->GetCoordSys());
                            double Vmax = params.GetPreloadVmin(tmp_step, tmp_ptor)+params.GetPredictorVstep(tmp_step, tmp_ptor)*(double)(params.GetPreloadVptsnb(tmp_step, tmp_ptor)-1);
                            if (Vmax > geo.GetAxisVmax())
                            {
                                double diff = Vmax-geo.GetAxisVmax();
                                int removePts = asTools::Round(diff/params.GetPredictorVstep(tmp_step, tmp_ptor));
                                params.SetPreloadVptsnb(tmp_step, tmp_ptor, params.GetPreloadVptsnb(tmp_step, tmp_ptor)-removePts);
                                asLogMessage(wxString::Format(_("Adapt V axis extent according to the maximum allowed (from %.2f to %.2f)."), Vmax, Vmax-diff));
                            }

                            // Area object instantiation
                            asGeoAreaCompositeGrid* area = asGeoAreaCompositeGrid::GetInstance(predictorPreprocess->GetCoordSys(),
                                                           params.GetPredictorGridType(tmp_step, tmp_ptor),
                                                           params.GetPreloadUmin(tmp_step, tmp_ptor),
                                                           params.GetPreloadUptsnb(tmp_step, tmp_ptor),
                                                           params.GetPredictorUstep(tmp_step, tmp_ptor),
                                                           params.GetPreloadVmin(tmp_step, tmp_ptor),
                                                           params.GetPreloadVptsnb(tmp_step, tmp_ptor),
                                                           params.GetPredictorVstep(tmp_step, tmp_ptor),
                                                           params.GetPreprocessLevel(tmp_step, tmp_ptor, tmp_prepro),
                                                           asNONE,
                                                           params.GetPredictorFlatAllowed(tmp_step, tmp_ptor));
                            wxASSERT(area);

                            asLogMessage(_("Area created."));

                            // Check the starting dates coherence
                            if (predictorPreprocess->GetOriginalProviderStart()>timeStartData)
                            {
                                asLogError(wxString::Format(_("The first year defined in the parameters (%s) is prior to the start date of the data (%s) (in asMethodCalibrator::PreloadData, preprocessing)."),
                                                            asTime::GetStringTime(timeStartData), asTime::GetStringTime(predictorPreprocess->GetOriginalProviderStart())));
                                wxDELETE(area);
                                wxDELETE(predictorPreprocess);
                                return false;
                            }

                            // Data loading
                            asLogMessage(wxString::Format(_("Loading %s data for level %d, %d h."),
                                                          params.GetPreprocessDataId(tmp_step, tmp_ptor, tmp_prepro).c_str(),
                                                          (int)params.GetPreprocessLevel(tmp_step, tmp_ptor, tmp_prepro),
                                                          (int)params.GetPreprocessTimeHours(tmp_step, tmp_ptor, tmp_prepro)));
                            if(!predictorPreprocess->Load(area, timeArrayData))
                            {
                                asLogError(_("The data could not be loaded."));
                                wxDELETE(area);
                                wxDELETE(predictorPreprocess);
                                return false;
                            }
                            wxDELETE(area);
                            m_StoragePredictorsPreprocess.push_back(predictorPreprocess);
                        }

                        // Fix the criteria if S1
                        if(params.GetPredictorCriteria(tmp_step, tmp_ptor).IsSameAs("S1"))
                        {
                            params.SetPredictorCriteria(tmp_step, tmp_ptor, "S1grads");
                        }

                        asLogMessage(_("Preprocessing data."));
                        asDataPredictorArchive* predictor = new asDataPredictorArchive(*m_StoragePredictorsPreprocess[0]);
                        try
                        {
                            if(!asPreprocessor::Preprocess(m_StoragePredictorsPreprocess, params.GetPreprocessMethod(tmp_step, tmp_ptor), predictor))
                            {
                                asLogError(_("Data preprocessing failed."));
                                wxDELETE(predictor);
                                return false;
                            }
                            m_PreloadedArchive[tmp_step][tmp_ptor][0][0]=predictor;
                        }
                        catch(bad_alloc& ba)
                        {
                            m_PreloadedArchive[tmp_step][tmp_ptor][0][0]=NULL;
                            wxString msg(ba.what(), wxConvUTF8);
                            asLogError(wxString::Format(_("Bad allocation caught in the data preprocessing: %s"), msg.c_str()));
                            wxDELETE(predictor);
                            return false;
                        }
                        catch (exception& e)
                        {
                            m_PreloadedArchive[tmp_step][tmp_ptor][0][0]=NULL;
                            wxString msg(e.what(), wxConvUTF8);
                            asLogError(wxString::Format(_("Exception in the data preprocessing: %s"), msg.c_str()));
                            wxDELETE(predictor);
                            return false;
                        }
                        DeletePreprocessData();
                        asLogMessage(_("Preprocessing over."));
                    }
                }
                else
                {
                    VectorFloat preloadLevels = params.GetPreloadLevels(tmp_step, tmp_ptor);
                    VectorDouble preloadTimeHours = params.GetPreloadTimeHours(tmp_step, tmp_ptor);

                    int preloadLevelsSize = wxMax(preloadLevels.size(),1);
                    int preloadTimeHoursSize = wxMax(preloadTimeHours.size(),1);

                    m_PreloadedArchive[tmp_step][tmp_ptor].resize(preloadLevelsSize);

                    // Load data for every level and every hour
                    for (int tmp_level=0; tmp_level<preloadLevelsSize; tmp_level++)
                    {
                        m_PreloadedArchive[tmp_step][tmp_ptor][tmp_level].resize(preloadTimeHoursSize);

                        for (int tmp_hour=0; tmp_hour<preloadTimeHoursSize; tmp_hour++)
                        {
                            m_PreloadedArchive[tmp_step][tmp_ptor][tmp_level][tmp_hour]=NULL;
                        }
                    }
                }
            }
        }
    }

    return true;
}

void asMethodCalibrator::Cleanup()
{
    DeletePreprocessData();

    if (m_StoragePredictors.size()>0)
    {
        for (unsigned int i=0; i<m_StoragePredictors.size(); i++)
        {
            wxDELETE(m_StoragePredictors[i]);
        }
        m_StoragePredictors.resize(0);
    }

    if (m_StorageCriteria.size()>0)
    {
        for (unsigned int i=0; i<m_StorageCriteria.size(); i++)
        {
            wxDELETE(m_StorageCriteria[i]);
        }
        m_StorageCriteria.resize(0);
    }

    // Do not delete preloaded data here !
}

void asMethodCalibrator::DeletePreprocessData()
{
    for (unsigned int i=0; i<m_StoragePredictorsPreprocess.size(); i++)
    {
        wxDELETE(m_StoragePredictorsPreprocess[i]);
    }
    m_StoragePredictorsPreprocess.resize(0);
}

void asMethodCalibrator::DeletePreloadedData()
{
    if (!m_Preloaded) return;

    for (unsigned int i=0; i<m_PreloadedArchive.size(); i++)
    {
        for (unsigned int j=0; j<m_PreloadedArchive[i].size(); j++)
        {
            if(!m_PreloadedArchivePointerCopy[i][j])
            {
                for (unsigned int k=0; k<m_PreloadedArchive[i][j].size(); k++)
                {
                    for (unsigned int l=0; l<m_PreloadedArchive[i][j][k].size(); l++)
                    {
                        wxDELETE(m_PreloadedArchive[i][j][k][l]);
                    }
                }
            }
        }
    }

    m_Preloaded = false;
}

bool asMethodCalibrator::GetAnalogsDates(asResultsAnalogsDates &results, asParametersScoring &params, int i_step, bool &containsNaNs)
{
    // Get the linear algebra method
    ThreadsManager().CritSectionConfig().Enter();
    int linAlgebraMethod = (int)(wxFileConfig::Get()->Read("/ProcessingOptions/ProcessingLinAlgebra", (long)asCOEFF_NOVAR));
    ThreadsManager().CritSectionConfig().Leave();

    // Initialize the result object
    results.SetCurrentStep(i_step);
    results.Init(params);

    // If result file already exists, load it
    if (results.Load())
    {
        return true;
    }

    // Get first year
    int yearStart = wxMin(params.GetArchiveYearStart(), params.GetCalibrationYearStart());
    int validationStart = params.GetValidationYearStart();
    if (!asTools::IsNaN(validationStart)) {
        yearStart = wxMin(yearStart, validationStart);
    }

    // Get last year
    int yearEnd = wxMax(params.GetArchiveYearEnd(), params.GetCalibrationYearEnd());
    int validationEnd = params.GetValidationYearEnd();
    if (!asTools::IsNaN(validationEnd)) {
        yearEnd = wxMax(yearEnd, validationEnd);
    }

    // Get full time array for the data
    double timeStartData = asTime::GetMJD(yearStart,1,1); // Always Jan 1st
    double timeEndData = asTime::GetMJD(yearEnd,12,31); // Always Dec 31
    asTimeArray timeArrayData(timeStartData, timeEndData,
                                params.GetTimeArrayAnalogsTimeStepHours(),
                                asTimeArray::Simple);
    timeArrayData.Init();

    // Archive date array
    double timeStartArchive = asTime::GetMJD(params.GetArchiveYearStart(),1,1); 
    double timeEndArchive = asTime::GetMJD(params.GetArchiveYearEnd(),12,31);
    timeStartArchive += params.GetTimeSplitStartDays(); // To avoid having dates before the start of the archive
    timeEndArchive += params.GetTimeSplitEndDays(); // Adjust so the predictors search won't overtake the array
    asTimeArray timeArrayArchive(timeStartArchive, timeEndArchive,
                                 params.GetTimeArrayAnalogsTimeStepHours(),
                                 asTimeArray::Simple);
    if(params.HasValidationPeriod()) // remove validation years
    {
        timeArrayArchive.SetForbiddenYears(params.GetValidationYearsVector());
    }
    timeArrayArchive.Init();

    // Target date array
    double timeStartCalibration = asTime::GetMJD(params.GetCalibrationYearStart(),1,1);
    double timeEndCalibration = asTime::GetMJD(params.GetCalibrationYearEnd(),12,31);
    timeStartCalibration += params.GetTimeSplitStartDays(); // To avoid having dates before the start of the archive
    timeEndCalibration += params.GetTimeSplitEndDays(); // Adjust so the predictors search won't overtake the array
    asTimeArray timeArrayTarget(timeStartCalibration, timeEndCalibration,
                                params.GetTimeArrayTargetTimeStepHours(),
                                params.GetTimeArrayTargetMode());
    if(!m_ValidationMode)
    {
        if(params.HasValidationPeriod()) // remove validation years
        {
            timeArrayTarget.SetForbiddenYears(params.GetValidationYearsVector());
        }
    }

    if(params.GetTimeArrayTargetMode().IsSameAs("PredictandThresholds"))
    {
        if(!timeArrayTarget.Init(*m_PredictandDB, params.GetTimeArrayTargetPredictandSerieName(), params.GetPredictandStationId(), params.GetTimeArrayTargetPredictandMinThreshold(), params.GetTimeArrayTargetPredictandMaxThreshold()))
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
    if(m_ValidationMode)
    {
        timeArrayTarget.KeepOnlyYears(params.GetValidationYearsVector());
    }

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
        neededMem += (params.GetPredictorUptsnb(i_step, i_ptor))
                     * (params.GetPredictorVptsnb(i_step, i_ptor));
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

            if(!params.NeedsPreprocessing(i_step, i_ptor))
            {
                VectorFloat preloadLevels = params.GetPreloadLevels(i_step, i_ptor);
                VectorDouble preloadTimeHours = params.GetPreloadTimeHours(i_step, i_ptor);
                wxASSERT(preloadLevels.size()>0);
                wxASSERT(preloadTimeHours.size()>0);

                // Get level and hour indices
                int i_level = asTools::SortedArraySearch(&preloadLevels[0], &preloadLevels[preloadLevels.size()-1], params.GetPredictorLevel(i_step, i_ptor));
                int i_hour = asTools::SortedArraySearch(&preloadTimeHours[0], &preloadTimeHours[preloadTimeHours.size()-1], params.GetPredictorTimeHours(i_step, i_ptor));
                if (i_level==asNOT_FOUND || i_level==asOUT_OF_RANGE)
                {
                    asLogError(_("The level could not be found in the preloaded data."));
                    return false;
                }
                if (i_hour==asNOT_FOUND || i_hour==asOUT_OF_RANGE)
                {
                    asLogError(_("The hour could not be found in the preloaded data."));
                    return false;
                }

                // Get data on the desired domain
                wxASSERT_MSG((unsigned)i_step<m_PreloadedArchive.size(), wxString::Format("i_step=%d, m_PreloadedArchive.size()=%d", i_step, (int)m_PreloadedArchive.size()));
                wxASSERT_MSG((unsigned)i_ptor<m_PreloadedArchive[i_step].size(), wxString::Format("i_ptor=%d, m_PreloadedArchive[i_step].size()=%d", i_ptor, (int)m_PreloadedArchive[i_step].size()));
                wxASSERT_MSG((unsigned)i_level<m_PreloadedArchive[i_step][i_ptor].size(), wxString::Format("i_level=%d, m_PreloadedArchive[i_step][i_ptor].size()=%d", i_level, (int)m_PreloadedArchive[i_step][i_ptor].size()));
                wxASSERT_MSG((unsigned)i_hour<m_PreloadedArchive[i_step][i_ptor][i_level].size(), wxString::Format("i_hour=%d, m_PreloadedArchive[i_step][i_ptor][i_level].size()=%d", i_hour, (int)m_PreloadedArchive[i_step][i_ptor][i_level].size()));
                ThreadsManager().CritSectionPreloadedData().Enter();
                if (m_PreloadedArchive[i_step][i_ptor][i_level][i_hour]==NULL)
                {
                    asLogError(_("The pointer to preloaded data is null."));
                    return false;
                }
                // Copy the data
                wxASSERT(m_PreloadedArchive[i_step][i_ptor][i_level][i_hour]);
                asDataPredictorArchive* desiredPredictor = new asDataPredictorArchive(*m_PreloadedArchive[i_step][i_ptor][i_level][i_hour]);
                ThreadsManager().CritSectionPreloadedData().Leave();

                // Area object instantiation
                asGeoAreaCompositeGrid* desiredArea = asGeoAreaCompositeGrid::GetInstance(desiredPredictor->GetCoordSys(),
                                               params.GetPredictorGridType(i_step, i_ptor),
                                               params.GetPredictorUmin(i_step, i_ptor),
                                               params.GetPredictorUptsnb(i_step, i_ptor),
                                               params.GetPredictorUstep(i_step, i_ptor),
                                               params.GetPredictorVmin(i_step, i_ptor),
                                               params.GetPredictorVptsnb(i_step, i_ptor),
                                               params.GetPredictorVstep(i_step, i_ptor),
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

                wxASSERT(desiredPredictor->GetSizeTime()>0);
                m_StoragePredictors.push_back(desiredPredictor);
            }
            else
            {
                // Get data on the desired domain
                wxASSERT_MSG((unsigned)i_step<m_PreloadedArchive.size(), wxString::Format("i_step=%d, m_PreloadedArchive.size()=%d", i_step, (int)m_PreloadedArchive.size()));
                wxASSERT_MSG((unsigned)i_ptor<m_PreloadedArchive[i_step].size(), wxString::Format("i_ptor=%d, m_PreloadedArchive[i_step].size()=%d", i_ptor, (int)m_PreloadedArchive[i_step].size()));
                wxASSERT_MSG((unsigned)m_PreloadedArchive[i_step][i_ptor].size()==1, wxString::Format("m_PreloadedArchive[i_step][i_ptor].size()=%d!=1", (int)m_PreloadedArchive[i_step][i_ptor].size()));
                wxASSERT_MSG((unsigned)m_PreloadedArchive[i_step][i_ptor][0].size()==1, wxString::Format("m_PreloadedArchive[i_step][i_ptor][0].size()=%d!=1", (int)m_PreloadedArchive[i_step][i_ptor][0].size()));
                ThreadsManager().CritSectionPreloadedData().Enter();
                if (m_PreloadedArchive[i_step][i_ptor][0][0]==NULL)
                {
                    asLogError(_("The pointer to preloaded data is null."));
                    return false;
                }
                wxASSERT(m_PreloadedArchive[i_step][i_ptor][0][0]);
                asDataPredictorArchive* desiredPredictor = new asDataPredictorArchive(*m_PreloadedArchive[i_step][i_ptor][0][0]);
                ThreadsManager().CritSectionPreloadedData().Leave();

                // Area object instantiation
                asGeoAreaCompositeGrid* desiredArea = asGeoAreaCompositeGrid::GetInstance(desiredPredictor->GetCoordSys(),
                                               params.GetPredictorGridType(i_step, i_ptor),
                                               params.GetPredictorUmin(i_step, i_ptor),
                                               params.GetPredictorUptsnb(i_step, i_ptor),
                                               params.GetPredictorUstep(i_step, i_ptor),
                                               params.GetPredictorVmin(i_step, i_ptor),
                                               params.GetPredictorVptsnb(i_step, i_ptor),
                                               params.GetPredictorVstep(i_step, i_ptor),
                                               params.GetPreprocessLevel(i_step, i_ptor, 0),
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

                wxASSERT(desiredPredictor->GetSizeTime()>0);
                m_StoragePredictors.push_back(desiredPredictor);
            }

        }
        else
        {
            asLogMessage(_("Loading data."));

            if(!params.NeedsPreprocessing(i_step, i_ptor))
            {
                // Loading the datasets information
                asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance(params.GetPredictorDatasetId(i_step, i_ptor), params.GetPredictorDataId(i_step, i_ptor), m_PredictorDataDir);
                if(!predictor)
                {
                    return false;
                }

                // Area object instantiation
                asGeoAreaCompositeGrid* area = asGeoAreaCompositeGrid::GetInstance(predictor->GetCoordSys(),
                                               params.GetPredictorGridType(i_step, i_ptor),
                                               params.GetPredictorUmin(i_step, i_ptor),
                                               params.GetPredictorUptsnb(i_step, i_ptor),
                                               params.GetPredictorUstep(i_step, i_ptor),
                                               params.GetPredictorVmin(i_step, i_ptor),
                                               params.GetPredictorVptsnb(i_step, i_ptor),
                                               params.GetPredictorVstep(i_step, i_ptor),
                                               params.GetPredictorLevel(i_step, i_ptor),
                                               asNONE,
                                               params.GetPredictorFlatAllowed(i_step, i_ptor));
                wxASSERT(area);

                // Check the starting dates coherence
                if (predictor->GetOriginalProviderStart()>timeStartData)
                {
                    asLogError(wxString::Format(_("The first year defined in the parameters (%s) is prior to the start date of the data (%s) (in asMethodCalibrator::GetAnalogsDates, no preprocessing)."),
                                                asTime::GetStringTime(timeStartData), asTime::GetStringTime(predictor->GetOriginalProviderStart())));
                    wxDELETE(area);
                    wxDELETE(predictor);
                    return false;
                }

                // Data loading
                if(!predictor->Load(area, timeArrayData))
                {
                    asLogError(_("The data could not be loaded."));
                    wxDELETE(area);
                    wxDELETE(predictor);
                    return false;
                }
                wxDELETE(area);
                m_StoragePredictors.push_back(predictor);
            }
            else
            {
                int preprocessSize = params.GetPreprocessSize(i_step, i_ptor);

                asLogMessage(wxString::Format(_("Preprocessing data (%d predictor(s)) while loading."),
                                              preprocessSize));

                for (int i_prepro=0; i_prepro<preprocessSize; i_prepro++)
                {
                    // Loading the datasets information
                    asDataPredictorArchive* predictorPreprocess = asDataPredictorArchive::GetInstance(params.GetPreprocessDatasetId(i_step, i_ptor, i_prepro),
                                                                                                             params.GetPreprocessDataId(i_step, i_ptor, i_prepro),
                                                                                                             m_PredictorDataDir);
                    if(!predictorPreprocess)
                    {
                        return false;
                    }

                    // Area object instantiation
                    asGeoAreaCompositeGrid* area = asGeoAreaCompositeGrid::GetInstance(predictorPreprocess->GetCoordSys(),
                                                                                       params.GetPredictorGridType(i_step, i_ptor),
                                                                                       params.GetPredictorUmin(i_step, i_ptor),
                                                                                       params.GetPredictorUptsnb(i_step, i_ptor),
                                                                                       params.GetPredictorUstep(i_step, i_ptor),
                                                                                       params.GetPredictorVmin(i_step, i_ptor),
                                                                                       params.GetPredictorVptsnb(i_step, i_ptor),
                                                                                       params.GetPredictorVstep(i_step, i_ptor),
                                                                                       params.GetPreprocessLevel(i_step, i_ptor, i_prepro),
                                                                                       asNONE,
                                                                                       params.GetPredictorFlatAllowed(i_step, i_ptor));
                    wxASSERT(area);

                    // Check the starting dates coherence
                    if (predictorPreprocess->GetOriginalProviderStart()>timeStartData)
                    {
                        asLogError(wxString::Format(_("The first year defined in the parameters (%s) is prior to the start date of the data (%s) (in asMethodCalibrator::GetAnalogsDates, preprocessing)."),
                                                    asTime::GetStringTime(timeStartData), asTime::GetStringTime(predictorPreprocess->GetOriginalProviderStart())));
                        wxDELETE(area);
                        wxDELETE(predictorPreprocess);
                        return false;
                    }

                    // Data loading
                    if(!predictorPreprocess->Load(area, timeArrayData))
                    {
                        asLogError(_("The data could not be loaded."));
                        wxDELETE(area);
                        wxDELETE(predictorPreprocess);
                        return false;
                    }
                    wxDELETE(area);
                    m_StoragePredictorsPreprocess.push_back(predictorPreprocess);
                }

                // Fix the criteria if S1
                if(params.GetPredictorCriteria(i_step, i_ptor).IsSameAs("S1"))
                {
                    params.SetPredictorCriteria(i_step, i_ptor, "S1grads");
                }

                asDataPredictorArchive* predictor = new asDataPredictorArchive(*m_StoragePredictorsPreprocess[0]);
                if(!asPreprocessor::Preprocess(m_StoragePredictorsPreprocess, params.GetPreprocessMethod(i_step, i_ptor), predictor))
                {
                   asLogError(_("Data preprocessing failed."));
                   return false;
                }

                DeletePreprocessData();
                m_StoragePredictors.push_back(predictor);
            }

            asLogMessage(_("Data loaded"));
        }

        // Instantiate a score object
        asLogMessage(_("Creating a criterion object."));
        asPredictorCriteria* criterion = asPredictorCriteria::GetInstance(params.GetPredictorCriteria(i_step, i_ptor),
                                                                          linAlgebraMethod);
        m_StorageCriteria.push_back(criterion);
        asLogMessage(_("Criterion object created."));

    }

    // Check time sizes
    #ifdef _DEBUG
        int prevTimeSize = 0;

        for (unsigned int i=0; i<m_StoragePredictors.size(); i++)
        {
            if (i>0)
            {
                wxASSERT(m_StoragePredictors[i]->GetSizeTime()==prevTimeSize);
            }
            prevTimeSize = m_StoragePredictors[i]->GetSizeTime();
        }
    #endif // _DEBUG

    // Send data and criteria to processor
    asLogMessage(_("Start processing the comparison."));
    //asDataPredictor predictors = predictorsArchive;

    if(!asProcessor::GetAnalogsDates(m_StoragePredictors, m_StoragePredictors,
                                     timeArrayData, timeArrayArchive, timeArrayData, timeArrayTarget,
                                     m_StorageCriteria, params, i_step, results, containsNaNs))
    {
        asLogError(_("Failed processing the analogs dates."));
        return false;
    }
    asLogMessage(_("The processing is over."));

    // Saving intermediate results
    results.Save();

    Cleanup();

    return true;
}

bool asMethodCalibrator::GetAnalogsSubDates(asResultsAnalogsDates &results, asParametersScoring &params, asResultsAnalogsDates &anaDates, int i_step, bool &containsNaNs)
{
    // Get the linear algebra method
    ThreadsManager().CritSectionConfig().Enter();
    int linAlgebraMethod = (int)(wxFileConfig::Get()->Read("/ProcessingOptions/ProcessingLinAlgebra", (long)asCOEFF_NOVAR));
    ThreadsManager().CritSectionConfig().Leave();

    // Initialize the result object
    results.SetCurrentStep(i_step);
    results.Init(params);

    // If result file already exists, load it
    if (results.Load()) return true;

    // Get first year
    int yearStart = wxMin(params.GetArchiveYearStart(), params.GetCalibrationYearStart());
    int validationStart = params.GetValidationYearStart();
    if (!asTools::IsNaN(validationStart)) {
        yearStart = wxMin(yearStart, validationStart);
    }

    // Get last year
    int yearEnd = wxMax(params.GetArchiveYearEnd(), params.GetCalibrationYearEnd());
    int validationEnd = params.GetValidationYearEnd();
    if (!asTools::IsNaN(validationEnd)) {
        yearEnd = wxMax(yearEnd, validationEnd);
    }

    // Get full time array for the data
    double timeStartData = asTime::GetMJD(yearStart,1,1); // Always Jan 1st
    double timeEndData = asTime::GetMJD(yearEnd,12,31); // Always Dec 31
    asTimeArray timeArrayData(timeStartData, timeEndData,
                              params.GetTimeArrayAnalogsTimeStepHours(),
                              asTimeArray::Simple);
    timeArrayData.Init();

    // Loop through every predictor
    for(int i_ptor=0; i_ptor<params.GetPredictorsNb(i_step); i_ptor++)
    {
        if(params.NeedsPreloading(i_step, i_ptor))
        {
            asLogMessage(_("Using preloaded data."));

            if(!params.NeedsPreprocessing(i_step, i_ptor))
            {
                VectorFloat preloadLevels = params.GetPreloadLevels(i_step, i_ptor);
                VectorDouble preloadTimeHours = params.GetPreloadTimeHours(i_step, i_ptor);
                wxASSERT(preloadLevels.size()>0);
                wxASSERT(preloadTimeHours.size()>0);

                // Get level and hour indices
                int i_level = asTools::SortedArraySearch(&preloadLevels[0], &preloadLevels[preloadLevels.size()-1], params.GetPredictorLevel(i_step, i_ptor));
                int i_hour = asTools::SortedArraySearch(&preloadTimeHours[0], &preloadTimeHours[preloadTimeHours.size()-1], params.GetPredictorTimeHours(i_step, i_ptor));
                if (i_level==asNOT_FOUND || i_level==asOUT_OF_RANGE)
                {
                    asLogError(_("The level could not be found in the preloaded data."));
                    return false;
                }
                if (i_hour==asNOT_FOUND || i_hour==asOUT_OF_RANGE)
                {
                    asLogError(_("The hour could not be found in the preloaded data."));
                    return false;
                }

                // Get data on the desired domain
                wxASSERT_MSG((unsigned)i_step<m_PreloadedArchive.size(), wxString::Format("i_step=%d, m_PreloadedArchive.size()=%d", i_step, (int)m_PreloadedArchive.size()));
                wxASSERT_MSG((unsigned)i_ptor<m_PreloadedArchive[i_step].size(), wxString::Format("i_ptor=%d, m_PreloadedArchive[i_step].size()=%d", i_ptor, (int)m_PreloadedArchive[i_step].size()));
                wxASSERT_MSG((unsigned)i_level<m_PreloadedArchive[i_step][i_ptor].size(), wxString::Format("i_level=%d, m_PreloadedArchive[i_step][i_ptor].size()=%d", i_level, (int)m_PreloadedArchive[i_step][i_ptor].size()));
                wxASSERT_MSG((unsigned)i_hour<m_PreloadedArchive[i_step][i_ptor][i_level].size(), wxString::Format("i_hour=%d, m_PreloadedArchive[i_step][i_ptor][i_level].size()=%d", i_hour, (int)m_PreloadedArchive[i_step][i_ptor][i_level].size()));
                ThreadsManager().CritSectionPreloadedData().Enter();
                if (m_PreloadedArchive[i_step][i_ptor][i_level][i_hour]==NULL)
                {
                    asLogError(_("The pointer to preloaded data is null."));
                    return false;
                }
                wxASSERT(m_PreloadedArchive[i_step][i_ptor][i_level][i_hour]);
                asDataPredictorArchive* desiredPredictor = new asDataPredictorArchive(*m_PreloadedArchive[i_step][i_ptor][i_level][i_hour]);
                ThreadsManager().CritSectionPreloadedData().Leave();

                // Area object instantiation
                asGeoAreaCompositeGrid* desiredArea = asGeoAreaCompositeGrid::GetInstance(desiredPredictor->GetCoordSys(),
                                               params.GetPredictorGridType(i_step, i_ptor),
                                               params.GetPredictorUmin(i_step, i_ptor),
                                               params.GetPredictorUptsnb(i_step, i_ptor),
                                               params.GetPredictorUstep(i_step, i_ptor),
                                               params.GetPredictorVmin(i_step, i_ptor),
                                               params.GetPredictorVptsnb(i_step, i_ptor),
                                               params.GetPredictorVstep(i_step, i_ptor),
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
                m_StoragePredictors.push_back(desiredPredictor);
            }
            else
            {
                // Get data on the desired domain
                wxASSERT_MSG((unsigned)i_step<m_PreloadedArchive.size(), wxString::Format("i_step=%d, m_PreloadedArchive.size()=%d", i_step, (int)m_PreloadedArchive.size()));
                wxASSERT_MSG((unsigned)i_ptor<m_PreloadedArchive[i_step].size(), wxString::Format("i_ptor=%d, m_PreloadedArchive[i_step].size()=%d", i_ptor, (int)m_PreloadedArchive[i_step].size()));
                wxASSERT_MSG((unsigned)m_PreloadedArchive[i_step][i_ptor].size()>0, wxString::Format("m_PreloadedArchive[i_step][i_ptor].size()=%d", (int)m_PreloadedArchive[i_step][i_ptor].size()));
                wxASSERT_MSG((unsigned)m_PreloadedArchive[i_step][i_ptor][0].size()>0, wxString::Format("m_PreloadedArchive[i_step][i_ptor][0].size()=%d", (int)m_PreloadedArchive[i_step][i_ptor][0].size()));
                ThreadsManager().CritSectionPreloadedData().Enter();
                if (m_PreloadedArchive[i_step][i_ptor][0][0]==NULL)
                {
                    asLogError(_("The pointer to preloaded data is null."));
                    return false;
                }
                wxASSERT(m_PreloadedArchive[i_step][i_ptor][0][0]);
                asDataPredictorArchive* desiredPredictor = new asDataPredictorArchive(*m_PreloadedArchive[i_step][i_ptor][0][0]);
                ThreadsManager().CritSectionPreloadedData().Leave();

                // Area object instantiation
                asGeoAreaCompositeGrid* desiredArea = asGeoAreaCompositeGrid::GetInstance(desiredPredictor->GetCoordSys(),
                                                                                        params.GetPredictorGridType(i_step, i_ptor),
                                                                                        params.GetPredictorUmin(i_step, i_ptor),
                                                                                        params.GetPredictorUptsnb(i_step, i_ptor),
                                                                                        params.GetPredictorUstep(i_step, i_ptor),
                                                                                        params.GetPredictorVmin(i_step, i_ptor),
                                                                                        params.GetPredictorVptsnb(i_step, i_ptor),
                                                                                        params.GetPredictorVstep(i_step, i_ptor),
                                                                                        params.GetPreprocessLevel(i_step, i_ptor, 0),
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
                m_StoragePredictors.push_back(desiredPredictor);
            }
        }
        else
        {
            asLogMessage(_("Loading data."));

            if(!params.NeedsPreprocessing(i_step, i_ptor))
            {
                // Loading the datasets information
                asDataPredictorArchive* predictor = asDataPredictorArchive::GetInstance(params.GetPredictorDatasetId(i_step, i_ptor), 
                                                                                        params.GetPredictorDataId(i_step, i_ptor), 
                                                                                        m_PredictorDataDir);
                if(!predictor)
                {
                    return false;
                }

                // Area object instantiation
                asGeoAreaCompositeGrid* area = asGeoAreaCompositeGrid::GetInstance(predictor->GetCoordSys(), 
                                                                                   params.GetPredictorGridType(i_step, i_ptor), 
                                                                                   params.GetPredictorUmin(i_step, i_ptor), 
                                                                                   params.GetPredictorUptsnb(i_step, i_ptor), 
                                                                                   params.GetPredictorUstep(i_step, i_ptor), 
                                                                                   params.GetPredictorVmin(i_step, i_ptor), 
                                                                                   params.GetPredictorVptsnb(i_step, i_ptor), 
                                                                                   params.GetPredictorVstep(i_step, i_ptor), 
                                                                                   params.GetPredictorLevel(i_step, i_ptor), 
                                                                                   asNONE, 
                                                                                   params.GetPredictorFlatAllowed(i_step, i_ptor));
                wxASSERT(area);

                // Check the starting dates coherence
                if (predictor->GetOriginalProviderStart()>timeStartData)
                {
                    asLogError(wxString::Format(_("The first year defined in the parameters (%s) is prior to the start date of the data (%s) (in asMethodCalibrator::GetAnalogsSubDates, no preprocessing)."), asTime::GetStringTime(timeStartData), asTime::GetStringTime(predictor->GetOriginalProviderStart())));
                    wxDELETE(area);
                    wxDELETE(predictor);
                    return false;
                }

                // Data loading
                if(!predictor->Load(area, timeArrayData))
                {
                    asLogError(_("The data could not be loaded."));
                    wxDELETE(area);
                    wxDELETE(predictor);
                    return false;
                }
                wxDELETE(area);
                m_StoragePredictors.push_back(predictor);
            }
            else
            {
                int preprocessSize = params.GetPreprocessSize(i_step, i_ptor);

                for (int i_prepro=0; i_prepro<preprocessSize; i_prepro++)
                {
                    // Loading the datasets information
                    asDataPredictorArchive* predictorPreprocess = asDataPredictorArchive::GetInstance(params.GetPreprocessDatasetId(i_step, i_ptor, i_prepro), 
                                                                                                      params.GetPreprocessDataId(i_step, i_ptor, i_prepro), 
                                                                                                      m_PredictorDataDir);
                    if(!predictorPreprocess)
                    {
                        return false;
                    }

                    // Area object instantiation
                    asGeoAreaCompositeGrid* area = asGeoAreaCompositeGrid::GetInstance(predictorPreprocess->GetCoordSys(), 
                                                                                       params.GetPredictorGridType(i_step, i_ptor), 
                                                                                       params.GetPredictorUmin(i_step, i_ptor), 
                                                                                       params.GetPredictorUptsnb(i_step, i_ptor), 
                                                                                       params.GetPredictorUstep(i_step, i_ptor), 
                                                                                       params.GetPredictorVmin(i_step, i_ptor), 
                                                                                       params.GetPredictorVptsnb(i_step, i_ptor), 
                                                                                       params.GetPredictorVstep(i_step, i_ptor), 
                                                                                       params.GetPreprocessLevel(i_step, i_ptor, i_prepro), 
                                                                                       asNONE, 
                                                                                       params.GetPredictorFlatAllowed(i_step, i_ptor));
                    wxASSERT(area);

                    // Check the starting dates coherence
                    if (predictorPreprocess->GetOriginalProviderStart()>timeStartData)
                    {
                        asLogError(wxString::Format(_("The first year defined in the parameters (%s) is prior to the start date of the data (%s) (in asMethodCalibrator::GetAnalogsSubDates, preprocessing)."), asTime::GetStringTime(timeStartData), asTime::GetStringTime(predictorPreprocess->GetOriginalProviderStart())));
                        wxDELETE(area);
                        wxDELETE(predictorPreprocess);
                        return false;
                    }

                    // Data loading
                    if(!predictorPreprocess->Load(area, timeArrayData))
                    {
                        asLogError(_("The data could not be loaded."));
                        wxDELETE(area);
                        wxDELETE(predictorPreprocess);
                        return false;
                    }
                    wxDELETE(area);
                    m_StoragePredictorsPreprocess.push_back(predictorPreprocess);
                }

                // Fix the criteria if S1
                if(params.GetPredictorCriteria(i_step, i_ptor).IsSameAs("S1"))
                {
                    params.SetPredictorCriteria(i_step, i_ptor, "S1grads");
                }

                asDataPredictorArchive* predictor = new asDataPredictorArchive(*m_StoragePredictorsPreprocess[0]);
                if(!asPreprocessor::Preprocess(m_StoragePredictorsPreprocess, params.GetPreprocessMethod(i_step, i_ptor), predictor))
                {
                   asLogError(_("Data preprocessing failed."));
                   return false;
                }

                m_StoragePredictors.push_back(predictor);

                DeletePreprocessData();
            }

            asLogMessage(_("Data loaded"));
        }

        // Instantiate a score object
        asLogMessage(_("Creating a criterion object."));
        asPredictorCriteria* criterion = asPredictorCriteria::GetInstance(params.GetPredictorCriteria(i_step, i_ptor), linAlgebraMethod);
        m_StorageCriteria.push_back(criterion);
        asLogMessage(_("Criterion object created."));

    }

    // Send data and criteria to processor
    asLogMessage(_("Start processing the comparison."));
    if(!asProcessor::GetAnalogsSubDates(m_StoragePredictors, m_StoragePredictors, timeArrayData, timeArrayData, anaDates, m_StorageCriteria, params, i_step, results, containsNaNs))
    {
        asLogError(_("Failed processing the analogs dates."));
        return false;
    }
    asLogMessage(_("The processing is over."));

    // Saving intermediate results
    results.Save();

    Cleanup();

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
    wxASSERT(m_PredictandDB);
    asLogMessage(_("Start setting the predictand values to the corresponding analog dates."));
    if(!asProcessor::GetAnalogsValues(*m_PredictandDB, anaDates, params, results))
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
    forecastScore->SetPercentile(params.GetForecastScorePercentile());
    forecastScore->SetThreshold(params.GetForecastScoreThreshold());

    if (forecastScore->UsesClimatology())
    {
        if (m_ScoreClimatology==0)
        {
            asLogMessage(_("Processing the score of the climatology."));

            // Get the whole values for the station of interest
            Array1DFloat predictandDataNorm = m_PredictandDB->GetDataNormalizedStation(params.GetPredictandStationId());
            Array1DDouble predictandTime = m_PredictandDB->GetTime();

            // Archive date array
            double timeStartArchive = asTime::GetMJD(params.GetArchiveYearStart(),1,1); 
            double timeEndArchive = asTime::GetMJD(params.GetArchiveYearEnd(),12,31);
            timeStartArchive += params.GetTimeSplitStartDays(); // To avoid having dates before the start of the archive
            timeEndArchive += params.GetTimeSplitEndDays(); // Adjust so the predictors search won't overtake the array

            // Get start and end dates
            float predictandTimeDays = params.GetPredictandTimeHours()/24.0;
            double timeStart = wxMax(predictandTime[0],timeStartArchive);
            double timeEnd = wxMin(predictandTime[predictandTime.size()-1],timeEndArchive);

            // Check if data are effectively available for this period
            int indexPredictandTimeStart = asTools::SortedArraySearchCeil(&predictandTime[0],&predictandTime[predictandTime.size()-1],timeStart);
            int indexPredictandTimeEnd = asTools::SortedArraySearchFloor(&predictandTime[0],&predictandTime[predictandTime.size()-1],timeEnd);
            while (asTools::IsNaN(predictandDataNorm(indexPredictandTimeStart)))
            {
                indexPredictandTimeStart++;
            }
            while (asTools::IsNaN(predictandDataNorm(indexPredictandTimeEnd)))
            {
                indexPredictandTimeEnd--;
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

            // Set data
            Array1DFloat climatologyData(dataLength);
            int counter = 0;
            for (int i=indexPredictandTimeStart; i<=indexPredictandTimeEnd; i+=indexStep)
            {
                climatologyData[counter] = predictandDataNorm[i];
                counter++;
            }
            wxASSERT(dataLength==counter);

            forecastScore->ProcessScoreClimatology(anaValues.GetTargetValues(), climatologyData);
            m_ScoreClimatology = forecastScore->GetScoreClimatology();
        }
        else
        {
            asLogMessage(_("Reloading the score of the climatology."));
            forecastScore->SetScoreClimatology(m_ScoreClimatology);
        }
    }
    asLogMessage(_("Forecast score object instantiated."));

    // Pass data and score to processor
    asLogMessage(_("Start processing the forecast scoring."));

    // Code for testing another F(0)
    //if(!asProcessorForecastScore::GetAnalogsForecastScoresLoadF0(anaValues, forecastScore, params, results))

    if(!asProcessorForecastScore::GetAnalogsForecastScores(anaValues, forecastScore, params, results))
    {
        asLogError(_("Failed processing the forecast scoring."));
        wxDELETE(forecastScore);
        return false;
    }
    asLogMessage(_("Processing over."));

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
    double timeStart = asTime::GetMJD(params.GetCalibrationYearStart(),1,1);
    double timeEnd = asTime::GetMJD(params.GetCalibrationYearEnd()+1,1,1);
    while (timeEnd>asTime::GetMJD(params.GetCalibrationYearEnd(),12,31,23,59))
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

    if(asTools::IsNaN(results.GetForecastScore()))
    {
        asLogError(_("The forecast score is NaN."));
        return false;
    }

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

        for (int i_anb=0; i_anb<=rowEnd; i_anb++)
        {
            params.SetForecastScoreAnalogsNumber(analogsNbVect[i_anb]);

            // Fixes and checks
            params.FixAnalogsNb();

            if(!GetAnalogsForecastScores(anaScores, params, anaValues, i_step))
                return false;
            if(!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, i_step))
                return false;

            m_ParametersTemp.push_back(params);
            m_ScoresCalibTemp.push_back(anaScoreFinal.GetForecastScore());
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
    wxFileConfig::Get()->Read("/Calibration/SkipValidation", &skipValidation, false);

    if (skipValidation)
    {
        return true;
    }

    if (m_Parameters.size()==0)
    {
        asLogError("The parameters array is empty in the validation procedure.");
        return false;
    }
    else if (m_Parameters.size()<unsigned(bestscorerow+1))
    {
        asLogError("Trying to access parameters outside the array in the validation procedure.");
        return false;
    }

    if (!m_Parameters[bestscorerow].HasValidationPeriod())
    {
        asLogWarning("The parameters have no validation period !");
        return false;
    }

    m_ValidationMode = true;

    asResultsAnalogsDates anaDatesPrevious;
    asResultsAnalogsDates anaDates;
    asResultsAnalogsValues anaValues;
    asResultsAnalogsForecastScores anaScores;
    asResultsAnalogsForecastScoreFinal anaScoreFinal;

    // Process every step one after the other
    int stepsNb = m_Parameters[bestscorerow].GetStepsNb();
    for (int i_step=0; i_step<stepsNb; i_step++)
    {
        bool containsNaNs = false;
        if (i_step==0)
        {
            if(!GetAnalogsDates(anaDates, m_Parameters[bestscorerow], i_step, containsNaNs)) return false;
        }
        else
        {
            anaDatesPrevious = anaDates;
            if(!GetAnalogsSubDates(anaDates, m_Parameters[bestscorerow], anaDatesPrevious, i_step, containsNaNs)) return false;
        }
        if (containsNaNs)
        {
            asLogError(_("The dates selection contains NaNs"));
            return false;
        }
    }
    if(!GetAnalogsValues(anaValues, m_Parameters[bestscorerow], anaDates, stepsNb-1)) return false;
    if(!GetAnalogsForecastScores(anaScores, m_Parameters[bestscorerow], anaValues, stepsNb-1)) return false;
    if(!GetAnalogsForecastScoreFinal(anaScoreFinal, m_Parameters[bestscorerow], anaScores, stepsNb-1)) return false;

    m_ScoreValid = anaScoreFinal.GetForecastScore();

    m_ValidationMode = false;

    return true;
}
