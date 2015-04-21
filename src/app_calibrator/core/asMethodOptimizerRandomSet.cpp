#include "asMethodOptimizerRandomSet.h"

#include "asFileAscii.h"
#include <asThreadMethodOptimizerRandomSet.h>
#ifndef UNIT_TESTING
    #include <AtmoswingAppCalibrator.h>
#endif

asMethodOptimizerRandomSet::asMethodOptimizerRandomSet()
:
asMethodOptimizer()
{

}

asMethodOptimizerRandomSet::~asMethodOptimizerRandomSet()
{
    //dtor
}

bool asMethodOptimizerRandomSet::Manager()
{
    // Get the processing method
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    bool parallelEvaluations;
    pConfig->Read("/Calibration/ParallelEvaluations", &parallelEvaluations, false);
    ThreadsManager().CritSectionConfig().Leave();

    // Seeds the random generator
    asTools::InitRandom();

    // Load parameters
    asParametersOptimization params;
    if (!params.LoadFromFile(m_paramsFilePath)) return false;
    if (m_predictandStationIds.size()>0)
    {
        params.SetPredictandStationIds(m_predictandStationIds);
    }
    InitParameters(params);
    m_originalParams = params;

    // Reset the score of the climatology
    m_scoreClimatology.clear();

    // Create a result object to save the parameters sets
    VectorInt stationId = m_originalParams.GetPredictandStationIds();
    wxString time = asTime::GetStringTime(asTime::NowMJD(asLOCAL), concentrate);
    asResultsParametersArray results_all;
    results_all.Init(wxString::Format(_("station_%s_tested_parameters"), GetPredictandStationIdsList(stationId).c_str()));
    asResultsParametersArray results_best;
    results_best.Init(wxString::Format(_("station_%s_best_parameters"), GetPredictandStationIdsList(stationId).c_str()));
    wxString resultsXmlFilePath = wxFileConfig::Get()->Read("/Paths/CalibrationResultsDir", asConfig::GetDefaultUserWorkingDir());
    resultsXmlFilePath.Append(wxString::Format("/Calibration/%s_station_%s_best_parameters.xml", time.c_str(), GetPredictandStationIdsList(stationId).c_str()));

    // Preload data
    if (!PreloadData(params))
    {
        asLogError(_("Could not preload the data."));
        return false;
    }

    // Get a forecast score object to extract the score order
    asForecastScore* forecastScore = asForecastScore::GetInstance(params.GetForecastScoreName());
    Order scoreOrder = forecastScore->GetOrder();
    wxDELETE(forecastScore);
    SetScoreOrder(scoreOrder);

    // Load the Predictand DB
    asLogMessage(_("Loading the Predictand DB."));
    if(!LoadPredictandDB(m_predictandDBFilePath)) return false;
    asLogMessage(_("Predictand DB loaded."));

    // Watch
    wxStopWatch sw;

    // Optimizer
    while(!IsOver())
    {
        // Get a parameters set
        params = GetNextParameters();

        if(!SkipNext() && !IsOver())
        {
            if(parallelEvaluations)
            {
                #ifndef UNIT_TESTING
                    if (g_responsive) wxGetApp().Yield();
                #endif
                if (m_cancel) return false;

                bool enableMessageBox = false;
                if (Log().IsMessageBoxOnErrorEnabled()) enableMessageBox = true;
                Log().DisableMessageBoxOnError();

                VectorFloat scoreClim = m_scoreClimatology;

                // Push the first parameters set
                asThreadMethodOptimizerRandomSet* firstthread = new asThreadMethodOptimizerRandomSet(this, params, &m_scoresCalib[m_iterator], &m_scoreClimatology);
                int threadType = firstthread->GetType();
                ThreadsManager().AddThread(firstthread);

                // Wait until done to get the score of the climatology
                if (scoreClim.size()==0)
                {
                    ThreadsManager().Wait(threadType);

                    #ifndef UNIT_TESTING
                        if (g_responsive) wxGetApp().Yield();
                    #endif
                    if (m_cancel) return false;
                }

                // Increment iterator
                IncrementIterator();

                // Get available threads nb
                int threadsNb = ThreadsManager().GetAvailableThreadsNb();
                threadsNb = wxMin(threadsNb, m_paramsNb-m_iterator);

                // Fill up the thread array
                for (int i_thread=0; i_thread<threadsNb; i_thread++)
                {
                    // Get a parameters set
                    params = GetNextParameters();

                    // Add it to the threads
                    asThreadMethodOptimizerRandomSet* thread = new asThreadMethodOptimizerRandomSet(this, params, &m_scoresCalib[m_iterator], &m_scoreClimatology);
                    ThreadsManager().AddThread(thread);

                    wxASSERT(m_scoresCalib.size()<=(unsigned)m_paramsNb);

                    // Increment iterator
                    IncrementIterator();
                }

                // Continue adding when threads become available
                while (m_iterator<m_paramsNb)
                {
                    #ifndef UNIT_TESTING
                        if (g_responsive) wxGetApp().Yield();
                    #endif
                    if (m_cancel) return false;

                    ThreadsManager().WaitForFreeThread(threadType);

                    // Get a parameters set
                    params = GetNextParameters();

                    // Add it to the threads
                    asThreadMethodOptimizerRandomSet* thread = new asThreadMethodOptimizerRandomSet(this, params, &m_scoresCalib[m_iterator], &m_scoreClimatology);
                    ThreadsManager().AddThread(thread);

                    wxASSERT(m_scoresCalib.size()<=(unsigned)m_paramsNb);

                    // Increment iterator
                    IncrementIterator();
                }

                // Wait until all done
                ThreadsManager().Wait(threadType);

                // Check results
                bool checkOK = true;
                for (unsigned int i_check=0; i_check<m_scoresCalib.size(); i_check++)
                {
                    if (asTools::IsNaN(m_scoresCalib[i_check]))
                    {
                        asLogError(wxString::Format(_("NaN found in the scores (element %d on %d in m_scoresCalib)."), (int)i_check+1, (int)m_scoresCalib.size()));
                        checkOK = false;
                    }
                }

                if (!checkOK) return false;

                if (enableMessageBox) Log().EnableMessageBoxOnError();
            }
            else
            {
                #ifndef UNIT_TESTING
                    if (g_responsive) wxGetApp().Yield();
                #endif
                if (m_cancel) return false;

                // Create results objects
                asResultsAnalogsDates anaDates;
                asResultsAnalogsDates anaDatesPrevious;
                asResultsAnalogsValues anaValues;
                asResultsAnalogsForecastScores anaScores;
                asResultsAnalogsForecastScoreFinal anaScoreFinal;

                // Process every step one after the other
                int stepsNb = params.GetStepsNb();
                for (int i_step=0; i_step<stepsNb; i_step++)
                {
                    bool containsNaNs = false;
                    if (i_step==0)
                    {
                        if(!GetAnalogsDates(anaDates, params, i_step, containsNaNs)) return false;
                        anaDatesPrevious = anaDates;
                    }
                    else
                    {
                        if(!GetAnalogsSubDates(anaDates, params, anaDatesPrevious, i_step, containsNaNs)) return false;
                        anaDatesPrevious = anaDates;
                    }
                    if (containsNaNs)
                    {
                        asLogError(_("The dates selection contains NaNs"));
                        return false;
                    }
                }
                if(!GetAnalogsValues(anaValues, params, anaDates, stepsNb-1)) return false;
                if(!GetAnalogsForecastScores(anaScores, params, anaValues, stepsNb-1)) return false;
                if(!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, stepsNb-1)) return false;

                // Store the result
                if(((m_optimizerStage==asINITIALIZATION) | (m_optimizerStage==asREASSESSMENT)) && m_iterator<m_paramsNb)
                {
                    m_scoresCalib[m_iterator] = anaScoreFinal.GetForecastScore();
                }
                else
                {
                    m_scoresCalibTemp.push_back(anaScoreFinal.GetForecastScore());
                }
                wxASSERT(m_scoresCalib.size()<=(unsigned)m_paramsNb);

                // Save all tested parameters in a text file
                results_all.Add(params,anaScoreFinal.GetForecastScore());

                // Increment iterator
                IncrementIterator();
            }
        }
    }

    // Display processing time
    asLogMessageImportant(wxString::Format(_("The whole processing took %ldms to execute"), sw.Time()));
    asLogState(_("Optimization over."));

    // Print parameters in a text file
    SetBestParameters(results_best);
    if(!results_best.Print()) return false;
    if(!results_all.Print()) return false;

    // Generate xml file with the best parameters set
    if(!m_parameters[0].GenerateSimpleParametersFile(resultsXmlFilePath)) return false;

    // Delete preloaded data
    DeletePreloadedData();

    return true;
}

void asMethodOptimizerRandomSet::InitParameters(asParametersOptimization &params)
{
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Read("/Calibration/MonteCarlo/RandomNb", &m_paramsNb, 1000);
    ThreadsManager().CritSectionConfig().Leave();

    // Get the number of runs
    params.InitRandomValues();

    // Create the corresponding number of parameters
    m_scoresCalib.resize(m_paramsNb);
    for (int i_var=0; i_var<m_paramsNb; i_var++)
    {
        asParametersOptimization paramsCopy;
        paramsCopy = params;
        paramsCopy.InitRandomValues();
        m_parameters.push_back(paramsCopy);
    }
}

asParametersOptimization asMethodOptimizerRandomSet::GetNextParameters()
{
    asParametersOptimization params;
    m_skipNext = false;

    if(((m_optimizerStage==asINITIALIZATION) | (m_optimizerStage==asREASSESSMENT)) && m_iterator<m_paramsNb)
    {
        params = m_parameters[m_iterator];
    }
    else
    {
        if(!Optimize(params)) asLogError(_("The parameters could not be optimized"));
    }

    return params;
}

bool asMethodOptimizerRandomSet::Optimize(asParametersOptimization &params)
{
    m_isOver = true;
    asLogMessage(_("Random method over."));
    return true;
}
