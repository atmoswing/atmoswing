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

    // Reset the score of the climatology
    m_ScoreClimatology = 0;

    // Create a result object to save the parameters sets
    asResultsParametersArray results_all;
    results_all.Init("tested_parameters");
    asResultsParametersArray results_best;
    results_best.Init("best_parameters");

    // Load parameters
    asParametersOptimization params;
    if (!params.LoadFromFile(m_ParamsFilePath)) return false;
    InitParameters(params);
    m_OriginalParams = params;

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
    wxDELETE(m_PredictandDB);
    m_PredictandDB = asDataPredictand::GetInstance(params.GetPredictandDB());
    if (!m_PredictandDB) return false;
    m_PredictandDB->Load(m_PredictandDBFilePath);
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
                    if (g_Responsive) wxGetApp().Yield();
                #endif
                if (m_Cancel) return false;

                bool enableMessageBox = false;
                if (Log().IsMessageBoxOnErrorEnabled()) enableMessageBox = true;
                Log().DisableMessageBoxOnError();

                int threadType = -1;
                float scoreClim = m_ScoreClimatology;

                // Push the first parameters set
                asThreadMethodOptimizerRandomSet* firstthread = new asThreadMethodOptimizerRandomSet(this, params, &m_ScoresCalib[m_Iterator], &m_ScoreClimatology);
                threadType = firstthread->GetType();
                ThreadsManager().AddThread(firstthread);

                // Wait until done to get the score of the climatology
                if (scoreClim==0)
                {
                    ThreadsManager().Wait(threadType);

                    #ifndef UNIT_TESTING
                        if (g_Responsive) wxGetApp().Yield();
                    #endif
                    if (m_Cancel) return false;
                }

                // Increment iterator
                IncrementIterator();

                // Get available threads nb
                int threadsNb = ThreadsManager().GetAvailableThreadsNb();
                threadsNb = wxMin(threadsNb, m_ParamsNb-m_Iterator);

                // Fill up the thread array
                for (int i_thread=0; i_thread<threadsNb; i_thread++)
                {
                    // Get a parameters set
                    params = GetNextParameters();

                    // Add it to the threads
                    asThreadMethodOptimizerRandomSet* thread = new asThreadMethodOptimizerRandomSet(this, params, &m_ScoresCalib[m_Iterator], &m_ScoreClimatology);
                    ThreadsManager().AddThread(thread);

                    wxASSERT(m_ScoresCalib.size()<=(unsigned)m_ParamsNb);

                    // Increment iterator
                    IncrementIterator();
                }

                // Continue adding when threads become available
                while (m_Iterator<m_ParamsNb)
                {
                    #ifndef UNIT_TESTING
                        if (g_Responsive) wxGetApp().Yield();
                    #endif
                    if (m_Cancel) return false;

                    ThreadsManager().WaitForFreeThread(threadType);

                    // Get a parameters set
                    params = GetNextParameters();

                    // Add it to the threads
                    asThreadMethodOptimizerRandomSet* thread = new asThreadMethodOptimizerRandomSet(this, params, &m_ScoresCalib[m_Iterator], &m_ScoreClimatology);
                    ThreadsManager().AddThread(thread);

                    wxASSERT(m_ScoresCalib.size()<=(unsigned)m_ParamsNb);

                    // Increment iterator
                    IncrementIterator();
                }

                // Wait until all done
                ThreadsManager().Wait(threadType);

                // Check results
                bool checkOK = true;
                for (unsigned int i_check=0; i_check<m_ScoresCalib.size(); i_check++)
                {
                    if (asTools::IsNaN(m_ScoresCalib[i_check]))
                    {
                        asLogError(wxString::Format(_("NaN found in the scores (element %d on %d in m_ScoresCalib)."), (int)i_check+1, (int)m_ScoresCalib.size()));
                        checkOK = false;
                    }
                }

                if (!checkOK) return false;

                if (enableMessageBox) Log().EnableMessageBoxOnError();
            }
            else
            {
                #ifndef UNIT_TESTING
                    if (g_Responsive) wxGetApp().Yield();
                #endif
                if (m_Cancel) return false;

                // Print in a temp file
                //params.PrintAndSaveTemp();

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
                if(((m_OptimizerStage==asINITIALIZATION) | (m_OptimizerStage==asREASSESSMENT)) && m_Iterator<m_ParamsNb)
                {
                    m_ScoresCalib[m_Iterator] = anaScoreFinal.GetForecastScore();
                }
                else
                {
                    m_ScoresCalibTemp.push_back(anaScoreFinal.GetForecastScore());
                }
                wxASSERT(m_ScoresCalib.size()<=(unsigned)m_ParamsNb);

                // Print all tested parameters in a text file
                results_all.Add(params,anaScoreFinal.GetForecastScore());
                if(!results_all.Print()) return false;

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

    // Delete preloaded data
    DeletePreloadedData();

    return true;
}

void asMethodOptimizerRandomSet::InitParameters(asParametersOptimization &params)
{
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Read("/Calibration/MonteCarlo/RandomNb", &m_ParamsNb, 1000);
    ThreadsManager().CritSectionConfig().Leave();

    // Get the number of runs
    params.InitRandomValues();

    // Create the corresponding number of parameters
    m_ScoresCalib.resize(m_ParamsNb);
    for (int i_var=0; i_var<m_ParamsNb; i_var++)
    {
        asParametersOptimization paramsCopy;
        paramsCopy = params;
        paramsCopy.InitRandomValues();
        m_Parameters.push_back(paramsCopy);
    }
}

asParametersOptimization asMethodOptimizerRandomSet::GetNextParameters()
{
    asParametersOptimization params;
    m_SkipNext = false;

    if(((m_OptimizerStage==asINITIALIZATION) | (m_OptimizerStage==asREASSESSMENT)) && m_Iterator<m_ParamsNb)
    {
        params = m_Parameters[m_Iterator];
    }
    else
    {
        if(!Optimize(params)) asLogError(_("The parameters could not be optimized"));
    }

    return params;
}

bool asMethodOptimizerRandomSet::Optimize(asParametersOptimization &params)
{
    m_IsOver = true;
    asLogMessage(_("Random method over."));
    return true;
}
