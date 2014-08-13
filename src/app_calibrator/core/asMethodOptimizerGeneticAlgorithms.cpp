#include "asMethodOptimizerGeneticAlgorithms.h"

#include <wx/dir.h>
#include "asFileAscii.h"
#include <asThreadMethodOptimizerGeneticAlgorithms.h>
#ifndef UNIT_TESTING
    #include <AtmoswingAppCalibrator.h>
#endif

asMethodOptimizerGeneticAlgorithms::asMethodOptimizerGeneticAlgorithms()
:
asMethodOptimizer()
{
    m_GenerationNb = 0;
    m_AssessmentCounter = 0;
    m_PopSize = 0;
    m_NaturalSelectionType = 0;
    m_CouplesSelectionType = 0;
    m_CrossoverType = 0;
    m_MutationsModeType = 0;
    m_AllowElitismForTheBest = true;
}

asMethodOptimizerGeneticAlgorithms::~asMethodOptimizerGeneticAlgorithms()
{
    //dtor
}

void asMethodOptimizerGeneticAlgorithms::ClearAll()
{
    m_ParametersTemp.clear();
    m_ScoresCalibTemp.clear();
    m_Parameters.clear();
    m_ScoresCalib.clear();
    m_ScoreValid = NaNFloat;
    m_BestScores.clear();
    m_MeanScores.clear();
}

void asMethodOptimizerGeneticAlgorithms::ClearTemp()
{
    m_ParametersTemp.clear();
    m_ScoresCalibTemp.clear();
}

void asMethodOptimizerGeneticAlgorithms::SortScoresAndParameters()
{
    wxASSERT(m_ScoresCalib.size()==m_Parameters.size());
    wxASSERT(m_ScoresCalib.size()>=1);
    wxASSERT(m_Parameters.size()>=1);

    if (m_Parameters.size()==1) return;

    // Sort according to the score
    Array1DFloat vIndices = Array1DFloat::LinSpaced(Eigen::Sequential, m_ParamsNb, 0, m_ParamsNb-1);
    asTools::SortArrays(&m_ScoresCalib[0],&m_ScoresCalib[m_ParamsNb-1],&vIndices[0],&vIndices[m_ParamsNb-1],m_ScoreOrder);

    // Sort the parameters sets as the scores
    std::vector <asParametersOptimizationGAs> copyParameters;
    for (int i=0; i<m_ParamsNb; i++)
    {
        copyParameters.push_back(m_Parameters[i]);
    }
    for (int i=0; i<m_ParamsNb; i++)
    {
        int index = vIndices(i);
        m_Parameters[i] = copyParameters[index];
    }
}

void asMethodOptimizerGeneticAlgorithms::SortScoresAndParametersTemp()
{
    wxASSERT(m_ScoresCalibTemp.size()==m_ParametersTemp.size());
    wxASSERT(m_ScoresCalibTemp.size()>=1);
    wxASSERT(m_ParametersTemp.size()>=1);

    if (m_ParametersTemp.size()==1) return;

    // Sort according to the score
    Array1DFloat vIndices = Array1DFloat::LinSpaced(Eigen::Sequential, m_ScoresCalibTemp.size(), 0, m_ScoresCalibTemp.size()-1);
    asTools::SortArrays(&m_ScoresCalibTemp[0],&m_ScoresCalibTemp[m_ScoresCalibTemp.size()-1],&vIndices[0],&vIndices[m_ScoresCalibTemp.size()-1],m_ScoreOrder);

    // Sort the parameters sets as the scores
    std::vector <asParametersOptimizationGAs> copyParameters;
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

void asMethodOptimizerGeneticAlgorithms::SortParamsLevelsAndTime()
{
    for (int i=0; i<m_Parameters.size(); i++)
    {
        m_Parameters[i].SortLevelsAndTime();
    }
}

bool asMethodOptimizerGeneticAlgorithms::ResumePreviousRun(asResultsParametersArray &results_generations)
{
    // Reload previous results
    if (g_ResumePreviousRun)
    {
        VectorInt stationId = m_OriginalParams.GetPredictandStationIds();
        wxString resultsDir = wxFileConfig::Get()->Read("/StandardPaths/CalibrationResultsDir", asConfig::GetDefaultUserWorkingDir());
        resultsDir.Append("/Calibration");

        wxDir dir(resultsDir);
        if ( !dir.IsOpened() )
        {
            asLogWarning(wxString::Format(_("The directory %s could not be opened."), resultsDir.c_str()));
        }
        else
        {
            // Check if the resulting file is already present
            wxString finalFilePattern = wxString::Format("*_station_%s_best_individual.txt", GetPredictandStationIdsList(stationId).c_str());
            if (dir.HasFiles(finalFilePattern))
            {
                asLogMessageImportant(wxString::Format(_("The directory %s already contains the resulting file."), resultsDir.c_str()));
                return true;
            }

            // Look for intermediate results to load
            wxString generationsFilePattern = wxString::Format("*_station_%s_generations.txt", GetPredictandStationIdsList(stationId).c_str());
            if (dir.HasFiles(generationsFilePattern))
            {
                wxString generationsFileName;
                dir.GetFirst(&generationsFileName, generationsFilePattern, wxDIR_FILES);
                while (dir.GetNext(&generationsFileName)) {} // Select the last available.

                asLogWarning(_("Previous intermediate results were found and will be loaded."));
                printf(_("Previous intermediate results were found and will be loaded.\n"));
                wxString filePath = resultsDir;
                filePath.Append(wxString::Format("/%s", generationsFileName.c_str()));
                asFileAscii prevResults(filePath, asFile::ReadOnly);
                if (!prevResults.Open())
                {
                    asLogError(wxString::Format(_("Couldn't open the file %s."), filePath.c_str()));
                    return false;
                }
                prevResults.SkipLines(1);

                // Check that the content match the current parameters
                wxString fileLine = prevResults.GetLineContent();
                wxString firstLineCopy = fileLine;
                wxString currentParamsPrint = m_OriginalParams.Print();
                int indexInFile, indexInParams;

                // Compare number of steps
                while(true)
                {
                    indexInFile = firstLineCopy.Find("Step");
                    indexInParams = currentParamsPrint.Find("Step");
                    if (indexInFile == wxNOT_FOUND && indexInParams == wxNOT_FOUND)
                    {
                        break;
                    }
                    else if ((indexInFile != wxNOT_FOUND && indexInParams == wxNOT_FOUND) || (indexInFile == wxNOT_FOUND && indexInParams != wxNOT_FOUND))
                    {
                        asLogError(_("The number of steps do not correspond between the current and the previous parameters."));
                        return false;
                    }

                    firstLineCopy.Replace("Step", wxEmptyString, false);
                    currentParamsPrint.Replace("Step", wxEmptyString, false);
                }

                // Compare number of predictors
                while(true)
                {
                    indexInFile = firstLineCopy.Find("Ptor");
                    indexInParams = currentParamsPrint.Find("Ptor");
                    if (indexInFile == wxNOT_FOUND && indexInParams == wxNOT_FOUND)
                    {
                        break;
                    }
                    else if ((indexInFile != wxNOT_FOUND && indexInParams == wxNOT_FOUND) || (indexInFile == wxNOT_FOUND && indexInParams != wxNOT_FOUND))
                    {
                        asLogError(_("The number of predictors do not correspond between the current and the previous parameters."));
                        return false;
                    }

                    firstLineCopy.Replace("Ptor", wxEmptyString, false);
                    currentParamsPrint.Replace("Ptor", wxEmptyString, false);
                }

                // Compare number of levels
                while(true)
                {
                    indexInFile = firstLineCopy.Find("Level");
                    indexInParams = currentParamsPrint.Find("Level");
                    if (indexInFile == wxNOT_FOUND && indexInParams == wxNOT_FOUND)
                    {
                        break;
                    }
                    else if ((indexInFile != wxNOT_FOUND && indexInParams == wxNOT_FOUND) || (indexInFile == wxNOT_FOUND && indexInParams != wxNOT_FOUND))
                    {
                        asLogError(_("The number of atmospheric levels do not correspond between the current and the previous parameters."));
                        return false;
                    }

                    firstLineCopy.Replace("Level", wxEmptyString, false);
                    currentParamsPrint.Replace("Level", wxEmptyString, false);
                }

                // Compare number of S1 criteria on gradients
                while(true)
                {
                    indexInFile = firstLineCopy.Find("S1grads");
                    indexInParams = currentParamsPrint.Find("S1grads");
                    if (indexInFile == wxNOT_FOUND && indexInParams == wxNOT_FOUND)
                    {
                        break;
                    }
                    else if ((indexInFile != wxNOT_FOUND && indexInParams == wxNOT_FOUND) || (indexInFile == wxNOT_FOUND && indexInParams != wxNOT_FOUND))
                    {
                        asLogError(_("The number of S1 criteria on gradients do not correspond between the current and the previous parameters."));
                        return false;
                    }

                    firstLineCopy.Replace("S1grads", wxEmptyString, false);
                    currentParamsPrint.Replace("S1grads", wxEmptyString, false);
                }

                // Compare number of tabs
                while(true)
                {
                    indexInFile = firstLineCopy.Find("\t");
                    indexInParams = currentParamsPrint.Find("\t");
                    if (indexInFile == wxNOT_FOUND && indexInParams == wxNOT_FOUND)
                    {
                        break;
                    }
                    else if ((indexInFile != wxNOT_FOUND && indexInParams == wxNOT_FOUND) || (indexInFile == wxNOT_FOUND && indexInParams != wxNOT_FOUND))
                    {
                        // In the file, there should be 3 tabs more (for the scores)
                        bool isOK = true;
                        firstLineCopy.Replace("\t", " ", false);
                        indexInFile = firstLineCopy.Find("\t");
                        if (indexInFile == wxNOT_FOUND) isOK = false;
                        firstLineCopy.Replace("\t", " ", false);
                        indexInFile = firstLineCopy.Find("\t");
                        if (indexInFile == wxNOT_FOUND) isOK = false;
                        firstLineCopy.Replace("\t", " ", false);
                        indexInFile = firstLineCopy.Find("\t");
                        if (indexInFile != wxNOT_FOUND) isOK = false;

                        if (!isOK)
                        {
                            asLogError(_("The number of tabs do not correspond between the current and the previous parameters."));
                            return false;
                        }

                        break;
                    }

                    firstLineCopy.Replace("\t", " ", false);
                    currentParamsPrint.Replace("\t", " ", false);
                }

                // Parse the parameters data
                std::vector < asParametersOptimizationGAs > vectParams;
                std::vector < float > vectScores;
                do
                {
                    if (fileLine.IsEmpty()) break;

                    asParametersOptimizationGAs prevParams = m_Parameters[0]; // And not m_OriginalParams due to initialization.
                    if (!prevParams.GetValuesFromString(fileLine))
                    {
                        return false;
                    }

                    // Get the score
                    int indexScoreCalib = fileLine.Find("Calib");
                    int indexScoreValid = fileLine.Find("Valid");
                    wxString strScore = fileLine.SubString(indexScoreCalib+6, indexScoreValid-2);
                    double scoreVal;
                    strScore.ToDouble(&scoreVal);
                    float prevScoresCalib = float(scoreVal);

                    // Add to the new array
                    results_generations.Add(prevParams,prevScoresCalib);
                    vectParams.push_back(prevParams);
                    vectScores.push_back(prevScoresCalib);

                    // Get next line
                    fileLine = prevResults.GetLineContent();
                }
                while (!prevResults.EndOfFile());
                prevResults.Close();

                asLogMessageImportant(wxString::Format(_("%d former results have been reloaded."), results_generations.GetCount()));
                printf(wxString::Format(_("%d former results have been reloaded.\n"), results_generations.GetCount()));

                // Check that it is consistent with the population size
                if (vectParams.size() % m_PopSize != 0)
                {
                    asLogError(wxString::Format(_("The number of former results is not consistent with the population size (%d)."), m_PopSize));
                    return false;
                }

                // Restore the last generation
                int genNb = vectParams.size() / m_PopSize;
                for (int i_var=0; i_var<m_PopSize; i_var++)
                {
                    int i_last_gen = (genNb-1) * m_PopSize;

                    wxASSERT(vectParams.size()>i_last_gen);
                    wxASSERT(vectScores.size()>i_last_gen);
                    m_Parameters[i_var] = vectParams[i_last_gen];
                    m_ScoresCalib[i_var] = vectScores[i_last_gen];

                    i_last_gen++;
                }

                // Restore best and mean scores
                m_BestScores.resize(genNb);
                m_MeanScores.resize(genNb);
                for (int i_gen=0; i_gen<genNb; i_gen++)
                {
                    int i_best = i_gen * m_PopSize;
                    m_BestScores[i_gen] = vectScores[i_best];

                    float mean = 0;
                    for (int i_next=0; i_next<m_PopSize; i_next++)
                    {
                        mean += vectScores[i_next];
                    }

                    m_MeanScores[i_gen] = mean/float(m_PopSize);
                }

                m_OptimizerStage = asREASSESSMENT;
                m_Iterator = m_ParamsNb;
                m_GenerationNb = genNb;
            }
        }
    }

    return true;
}

bool asMethodOptimizerGeneticAlgorithms::SetBestParameters(asResultsParametersArray &results)
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
        Validate(&m_Parameters[bestscorerow]);
    }

    // Sort according to the level and the observation time
    asParametersScoring sortedParams = m_Parameters[bestscorerow];
    sortedParams.SortLevelsAndTime();

    results.Add(sortedParams, m_ScoresCalib[bestscorerow], m_ScoreValid);

    return true;
}

bool asMethodOptimizerGeneticAlgorithms::Manager()
{
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    int nbRuns = 0;
    pConfig->Read("/Calibration/GeneticAlgorithms/NbRuns", &nbRuns, 20);
    pConfig->Read("/Calibration/GeneticAlgorithms/PopulationSize", &m_PopSize, 50);
    m_ParamsNb = m_PopSize;
    pConfig->Read("/Calibration/GeneticAlgorithms/AllowElitismForTheBest", &m_AllowElitismForTheBest, true);
    m_NaturalSelectionType = (int)pConfig->Read("/Calibration/GeneticAlgorithms/NaturalSelectionOperator", 0l);
    m_CouplesSelectionType = (int)pConfig->Read("/Calibration/GeneticAlgorithms/CouplesSelectionOperator", 0l);
    m_CrossoverType = (int)pConfig->Read("/Calibration/GeneticAlgorithms/CrossoverOperator", 0l);
    m_MutationsModeType = (int)pConfig->Read("/Calibration/GeneticAlgorithms/MutationOperator", 0l);
    ThreadsManager().CritSectionConfig().Leave();

    // Reset the score of the climatology
    m_ScoreClimatology.clear();

    for (int i=0; i<nbRuns; i++)
    {
        try
        {
            m_IsOver = false;
            ClearAll();
            if (!ManageOneRun())
            {
                DeletePreloadedData();
                return false;
            }
        }
        catch(bad_alloc& ba)
        {
            wxString msg(ba.what(), wxConvUTF8);
            asLogError(wxString::Format(_("Bad allocation caught in GAs: %s"), msg.c_str()));
            DeletePreloadedData();
            return false;
        }
        catch (exception& e)
        {
            wxString msg(e.what(), wxConvUTF8);
            asLogError(wxString::Format(_("Exception in the GAs: %s"), msg.c_str()));
            DeletePreloadedData();
            return false;
        }
    }

    // Delete preloaded data
    DeletePreloadedData();

    return true;
}

bool asMethodOptimizerGeneticAlgorithms::ManageOneRun()
{
    // Get the processing method
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    bool parallelEvaluations;
    pConfig->Read("/Calibration/ParallelEvaluations", &parallelEvaluations, true);
    ThreadsManager().CritSectionConfig().Leave();

    // Parameter to print the results every x generation
    int printResultsEveryNbGenerations = 5;

    // Reset some data members
    m_Iterator = 0;
    m_AssessmentCounter = 0;
    m_OptimizerStage = asINITIALIZATION;
    m_SkipNext = false;
    m_IsOver = false;
    m_GenerationNb = 1;

    // Seeds the random generator
    asTools::InitRandom();

    // Load parameters
    asParametersOptimizationGAs params;
    if (!params.LoadFromFile(m_ParamsFilePath)) return false;
    if (m_PredictandStationIds.size()>0)
    {
        params.SetPredictandStationIds(m_PredictandStationIds);
    }
    InitParameters(params);
    SortParamsLevelsAndTime();
    m_OriginalParams = params;
    
    // Create a result object to save the parameters sets
    VectorInt stationId = params.GetPredictandStationIds();
    wxString time = asTime::GetStringTime(asTime::NowMJD(asLOCAL), concentrate);
    asResultsParametersArray results_final_population;
    results_final_population.Init(wxString::Format(_("station_%s_final_population"), GetPredictandStationIdsList(stationId).c_str()));
    asResultsParametersArray results_best_individual;
    results_best_individual.Init(wxString::Format(_("station_%s_best_individual"), GetPredictandStationIdsList(stationId).c_str()));
    asResultsParametersArray results_generations;
    results_generations.Init(wxString::Format(_("station_%s_generations"), GetPredictandStationIdsList(stationId).c_str()));
    wxString resultsXmlFilePath = wxFileConfig::Get()->Read("/StandardPaths/CalibrationResultsDir", asConfig::GetDefaultUserWorkingDir());
    resultsXmlFilePath.Append(wxString::Format("/Calibration/%s_station_%s_best_parameters.xml", time.c_str(), GetPredictandStationIdsList(stationId).c_str()));
    int counterPrint = 0;

    ResumePreviousRun(results_generations);

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
        asLogError(wxString::Format(_("Bad allocation caught in the data preloading (in GAs): %s"), msg.c_str()));
        DeletePreloadedData();
        return false;
    }
    catch (exception& e)
    {
        wxString msg(e.what(), wxConvUTF8);
        asLogError(wxString::Format(_("Exception in the data preloading (in GAs): %s"), msg.c_str()));
        DeletePreloadedData();
        return false;
    }

    // Get a forecast score object to extract the score order
    asForecastScore* forecastScore = asForecastScore::GetInstance(params.GetForecastScoreName());
    Order scoreOrder = forecastScore->GetOrder();
    wxDELETE(forecastScore);
    SetScoreOrder(scoreOrder);

    // Load the Predictand DB
    asLogMessage(_("Loading the Predictand DB."));
    if(!LoadPredictandDB(m_PredictandDBFilePath)) return false;
    asLogMessage(_("Predictand DB loaded."));

    // Watch
    wxStopWatch sw;

    // Optimizer
    while(!IsOver())
    {
        // Get a parameters set
        asParametersOptimizationGAs newParams = GetNextParameters();

        if(!SkipNext() && !IsOver())
        {
            // Check on the parameters set
            if (newParams.GetStepsNb()==0)
            {
                asLogError(_("The new parameters set is not correcty initialized."));
                return false;
            }

            if(parallelEvaluations)
            {
                #ifndef UNIT_TESTING
                    if (g_Responsive) wxGetApp().Yield();
                #endif
                if (m_Cancel) return false;

                bool enableMessageBox = false;
                if (Log().IsMessageBoxOnErrorEnabled()) enableMessageBox = true;
                Log().DisableMessageBoxOnError();

                VectorFloat scoreClim = m_ScoreClimatology;

                // Push the first parameters set
                asThreadMethodOptimizerGeneticAlgorithms* firstthread = new asThreadMethodOptimizerGeneticAlgorithms(this, newParams, &m_ScoresCalib[m_Iterator], &m_ScoreClimatology);
                int threadType = firstthread->GetType();
                ThreadsManager().AddThread(firstthread);

                // Wait until done to get the score of the climatology
                if (scoreClim.size()==0)
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
                    asParametersOptimizationGAs nextParams = GetNextParameters();
                    if (nextParams.GetStepsNb()==0)
                    {
                        asLogError(wxString::Format(_("The new parameters set is not correcty initialized in the thread array filling (iterator %d/%d)."), m_Iterator, m_ParamsNb));
                        return false;
                    }

                    // Add it to the threads
                    asThreadMethodOptimizerGeneticAlgorithms* thread = new asThreadMethodOptimizerGeneticAlgorithms(this, nextParams, &m_ScoresCalib[m_Iterator], &m_ScoreClimatology);
                    ThreadsManager().AddThread(thread);

                    wxASSERT(m_ScoresCalib.size()<=(unsigned)m_ParamsNb);

                    // Increment iterator
                    IncrementIterator();

                    if (m_Iterator==m_ParamsNb)
                    {
                        break;
                    }
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
                    asParametersOptimizationGAs nextParams = GetNextParameters();
                    if (nextParams.GetStepsNb()==0)
                    {
                        asLogError(wxString::Format(_("The new parameters set is not correcty initialized in the continuous adding (iterator %d/%d)."), m_Iterator, m_ParamsNb));
                        return false;
                    }

                    // Add it to the threads
                    asThreadMethodOptimizerGeneticAlgorithms* thread = new asThreadMethodOptimizerGeneticAlgorithms(this, nextParams, &m_ScoresCalib[m_Iterator], &m_ScoreClimatology);
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
                        wxString paramsContent = m_Parameters[i_check].Print();
                        asLogError(wxString::Format(_("Parameters #%d: %s"), (int)i_check+1, paramsContent.c_str()));
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

                // Create results objects
                asResultsAnalogsDates anaDates;
                asResultsAnalogsDates anaDatesPrevious;
                asResultsAnalogsValues anaValues;
                asResultsAnalogsForecastScores anaScores;
                asResultsAnalogsForecastScoreFinal anaScoreFinal;

                // Process every step one after the other
                int stepsNb = newParams.GetStepsNb();
                for (int i_step=0; i_step<stepsNb; i_step++)
                {
                    bool containsNaNs = false;
                    if (i_step==0)
                    {
                        if(!GetAnalogsDates(anaDates, newParams, i_step, containsNaNs)) return false;
                        anaDatesPrevious = anaDates;
                    }
                    else
                    {
                        if(!GetAnalogsSubDates(anaDates, newParams, anaDatesPrevious, i_step, containsNaNs)) return false;
                        anaDatesPrevious = anaDates;
                    }
                    if (containsNaNs)
                    {
                        asLogError(_("The dates selection contains NaNs"));
                        return false;
                    }
                }
                if(!GetAnalogsValues(anaValues, newParams, anaDates, stepsNb-1)) return false;
                if(!GetAnalogsForecastScores(anaScores, newParams, anaValues, stepsNb-1)) return false;
                if(!GetAnalogsForecastScoreFinal(anaScoreFinal, newParams, anaScores, stepsNb-1)) return false;

                // Store the result
                if(((m_OptimizerStage==asINITIALIZATION) | (m_OptimizerStage==asREASSESSMENT)) && m_Iterator<m_ParamsNb)
                {
                    m_ScoresCalib[m_Iterator] = anaScoreFinal.GetForecastScore();
                }
                else
                {
                    wxLogError(_("This should not happen (in ManageOneRun)..."));
                }
                wxASSERT(m_ScoresCalib.size()<=(unsigned)m_ParamsNb);

                // Increment iterator
                IncrementIterator();
            }

            if (m_Iterator==m_ParamsNb)
            {
                // Elitism after mutation must occur after evaluation
                ElitismAfterMutation();

                // Save the full generation
                for (unsigned int i=0; i<m_Parameters.size(); i++)
                {
                    results_generations.Add(m_Parameters[i],m_ScoresCalib[i]);
                }

                // Print results every x generation
                if (counterPrint>printResultsEveryNbGenerations-1)
                {
                    results_generations.Print();
                    counterPrint = 0;
                }
                counterPrint++;

                // Display stats
                float meanScore = asTools::Mean(&m_ScoresCalib[0], &m_ScoresCalib[m_ScoresCalib.size()-1]);
                float bestScore = 0;
                switch (m_ScoreOrder)
                {
                    case (Asc):
                    {
                        bestScore = asTools::MinArray(&m_ScoresCalib[0], &m_ScoresCalib[m_ScoresCalib.size()-1]);
                        break;
                    }
                    case (Desc):
                    {
                        bestScore = asTools::MaxArray(&m_ScoresCalib[0], &m_ScoresCalib[m_ScoresCalib.size()-1]);
                        break;
                    }
                    default:
                    {
                        asLogError(_("The given natural selection method couldn't be found."));
                        return false;
                    }
                }
                m_BestScores.push_back(bestScore);
                m_MeanScores.push_back(meanScore);

                asLogMessageImportant(wxString::Format(_("Mean %g, best %g"), meanScore, bestScore));
            }
        }

        if (IsOver())
        {
            for (unsigned int i=0; i<m_Parameters.size(); i++)
            {
                results_final_population.Add(m_Parameters[i],m_ScoresCalib[i]);
            }
        }
    }

    // Display processing time
    asLogMessageImportant(wxString::Format(_("The whole processing took %ldms to execute"), sw.Time()));
    asLogState(_("Optimization over."));

    // Validate
    Validate(&m_Parameters[0]);

    // Print parameters in a text file
    SetSelectedParameters(results_final_population);
    if(!results_final_population.Print()) return false;
    SetBestParameters(results_best_individual);
    if(!results_best_individual.Print()) return false;
    if(!results_generations.Print()) return false;

    // Generate xml file with the best parameters set
    if(!m_Parameters[0].GenerateSimpleParametersFile(resultsXmlFilePath)) return false;

    // Print stats
    ThreadsManager().CritSectionConfig().Enter();
    wxString statsFilePath = wxFileConfig::Get()->Read("/StandardPaths/CalibrationResultsDir", asConfig::GetDefaultUserWorkingDir());
    ThreadsManager().CritSectionConfig().Leave();
    statsFilePath.Append(wxString::Format("/Calibration/%s_stats.txt", time.c_str()));
    asFileAscii stats(statsFilePath, asFile::New);

    return true;
}

void asMethodOptimizerGeneticAlgorithms::InitParameters(asParametersOptimizationGAs &params)
{
    // Get a first parameters set to get the number of unknown variables
    params.InitRandomValues();
    asLogMessage(wxString::Format(_("The population is made of %d individuals."), m_PopSize));

    // Create the corresponding number of parameters
    m_ScoresCalib.resize(m_PopSize);
    m_Parameters.resize(m_PopSize);
    for (int i_var=0; i_var<m_PopSize; i_var++)
    {
        asParametersOptimizationGAs paramsCopy;
        paramsCopy = params;
        paramsCopy.InitRandomValues();
        paramsCopy.BuildChromosomes();

        // Create arrays for the self-adaptation methods
        switch(m_MutationsModeType)
        {
            case (SelfAdaptationRate):
            {
                paramsCopy.InitIndividualSelfAdaptationMutationRate();
                break;
            }

            case (SelfAdaptationRadius):
            {
                paramsCopy.InitIndividualSelfAdaptationMutationRate();
                paramsCopy.InitIndividualSelfAdaptationMutationRadius();
                break;
            }

            case (SelfAdaptationRateChromosome):
            {
                paramsCopy.InitChromosomeSelfAdaptationMutationRate();
                break;
            }

            case (SelfAdaptationRadiusChromosome):
            {
                paramsCopy.InitChromosomeSelfAdaptationMutationRate();
                paramsCopy.InitChromosomeSelfAdaptationMutationRadius();
                break;
            }

            default:
            {
                // No self-adaptation required.
            }
        }

        m_Parameters[i_var] = paramsCopy;
        m_ScoresCalib[i_var] = NaNFloat;
    }
    m_ScoreValid = NaNFloat;
}

asParametersOptimizationGAs asMethodOptimizerGeneticAlgorithms::GetNextParameters()
{
    asParametersOptimizationGAs params;
    m_SkipNext = false;

    if(((m_OptimizerStage==asINITIALIZATION) | (m_OptimizerStage==asREASSESSMENT)) && m_Iterator<m_ParamsNb)
    {
        if (asTools::IsNaN(m_ScoresCalib[m_Iterator]))
        {
            params = m_Parameters[m_Iterator];
            m_AssessmentCounter++;
        }
        else
        {
            while (!asTools::IsNaN(m_ScoresCalib[m_Iterator]))
            {
                m_Iterator++;
                if (m_Iterator==m_ParamsNb)
                {
                    m_OptimizerStage=asCHECK_CONVERGENCE;
                    if(!Optimize(params)) asLogError(_("The parameters could not be optimized"));
                    return params;
                }
            }
            params = m_Parameters[m_Iterator];
            m_AssessmentCounter++;
        }
    }
    else if(((m_OptimizerStage==asINITIALIZATION) | (m_OptimizerStage==asREASSESSMENT)) && m_Iterator==m_ParamsNb)
    {
        m_OptimizerStage=asCHECK_CONVERGENCE;
        if(!Optimize(params)) asLogError(_("The parameters could not be optimized"));
    }
    else
    {
        wxLogError(_("This should not happen (in GetNextParameters)..."));
    }

    return params;
}

bool asMethodOptimizerGeneticAlgorithms::Optimize(asParametersOptimizationGAs &params)
{
    if (m_OptimizerStage==asCHECK_CONVERGENCE)
    {
        // Different operators consider that the scores are sorted !
        SortScoresAndParameters();

        // Check if we should end
        bool stopiterations = true;
        if (!CheckConvergence(stopiterations)) return false;
        if(stopiterations)
        {
            m_IsOver = true;
            asLogMessage(_("Optimization process over."));
            return true;
        }

        // Proceed to a new generation
        if (!NaturalSelection()) return false;
        if (!Mating()) return false;
        if (!Mutatation()) return false;

        SortParamsLevelsAndTime();

        m_Iterator = 0;
        m_OptimizerStage = asREASSESSMENT;
        m_SkipNext = true;
        m_GenerationNb++;

        asLogMessageImportant(wxString::Format(_("Generation number %d"), m_GenerationNb));

        return true;
    }
    else
    {
        asLogError(_("Optimization stage undefined"));
    }

    return false;
}

bool asMethodOptimizerGeneticAlgorithms::CheckConvergence(bool &stop)
{
    // NB: The parameters and scores are already sorted !

    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    int convergenceStepsNb;
    pConfig->Read("/Calibration/GeneticAlgorithms/ConvergenceStepsNb", &convergenceStepsNb, 20);
    ThreadsManager().CritSectionConfig().Leave();

    stop = true;

    // Check if enough generations
    if (m_BestScores.size()<(unsigned)convergenceStepsNb)
    {
        stop = false;
        return true;
    }

    // Check the best convergenceStepsNb scores
    for (unsigned int i=m_BestScores.size()-1; i>m_BestScores.size()-convergenceStepsNb; i--) // Checked
    {
        if (m_BestScores[i]!=m_BestScores[i-1])
        {
            stop = false;
            return true;
        }
    }

    return true;
}

bool asMethodOptimizerGeneticAlgorithms::ElitismAfterMutation()
{
    // Different operators consider that the scores are sorted !
    SortScoresAndParameters();

    // Apply elitism: If the best has been degraded during previous mutations, replace a random individual by the previous best.
    if (m_AllowElitismForTheBest && m_ParametersTemp.size()>0)
    {
        switch (m_ScoreOrder)
        {
            case (Asc):
            {
                float actualBest = m_ScoresCalib[0];
                int prevBestIndex = asTools::MinArrayIndex(&m_ScoresCalibTemp[0], &m_ScoresCalibTemp[m_ScoresCalibTemp.size()-1]);
                if (m_ScoresCalibTemp[prevBestIndex]<actualBest)
                {
                    asLogMessageImportant(_("Application of elitism after mutation."));
                    // Randomly select a row to replace
                    int randomRow = asTools::Random(0, m_ScoresCalib.size()-1, 1);
                    m_Parameters[randomRow] = m_ParametersTemp[prevBestIndex];
                    m_ScoresCalib[randomRow] = m_ScoresCalibTemp[prevBestIndex];
                    SortScoresAndParameters();
                }
                break;
            }
            case (Desc):
            {
                float actualBest = m_ScoresCalib[0];
                int prevBestIndex = asTools::MaxArrayIndex(&m_ScoresCalibTemp[0], &m_ScoresCalibTemp[m_ScoresCalibTemp.size()-1]);
                if (m_ScoresCalibTemp[prevBestIndex]>actualBest)
                {
                    asLogMessageImportant(_("Application of elitism after mutation."));
                    // Randomly select a row to replace
                    int randomRow = asTools::Random(0, m_ScoresCalib.size()-1, 1);
                    m_Parameters[randomRow] = m_ParametersTemp[prevBestIndex];
                    m_ScoresCalib[randomRow] = m_ScoresCalibTemp[prevBestIndex];
                    SortScoresAndParameters();
                }
                break;
            }
            default:
            {
                asLogError(_("Score order not correctly defined."));
                return false;
            }
        }
    }

    ClearTemp();

    return true;
}

bool asMethodOptimizerGeneticAlgorithms::ElitismAfterSelection()
{
    // Apply elitism:If the best has not been selected, replace a random individual by the best.
    if (m_AllowElitismForTheBest)
    {
        switch (m_ScoreOrder)
        {
            case (Asc):
            {
                float prevBest = m_ScoresCalib[0];
                float actualBest = asTools::MinArray(&m_ScoresCalibTemp[0], &m_ScoresCalibTemp[m_ScoresCalibTemp.size()-1]);
                if (prevBest<actualBest)
                {
                    asLogMessageImportant(_("Application of elitism in the natural selection."));
                    // Randomly select a row to replace
                    int randomRow = asTools::Random(0, m_ScoresCalibTemp.size()-1, 1);
                    m_ParametersTemp[randomRow] = m_Parameters[0];
                    m_ScoresCalibTemp[randomRow] = m_ScoresCalib[0];
                }
                break;
            }
            case (Desc):
            {
                float prevBest = m_ScoresCalib[0];
                float actualBest = asTools::MaxArray(&m_ScoresCalibTemp[0], &m_ScoresCalibTemp[m_ScoresCalibTemp.size()-1]);
                if (prevBest>actualBest)
                {
                    asLogMessageImportant(_("Application of elitism in the natural selection."));
                    // Randomly select a row to replace
                    int randomRow = asTools::Random(0, m_ScoresCalibTemp.size()-1, 1);
                    m_ParametersTemp[randomRow] = m_Parameters[0];
                    m_ScoresCalibTemp[randomRow] = m_ScoresCalib[0];
                }
                break;
            }
            default:
            {
                asLogError(_("Score order not correctly defined."));
                return false;
            }
        }
    }

    return true;
}

bool asMethodOptimizerGeneticAlgorithms::NaturalSelection()
{
    // NB: The parameters and scores are already sorted !

    asLogMessage(_("Applying natural selection."));

    ClearTemp();

    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    double ratioIntermediateGeneration;
    pConfig->Read("/Calibration/GeneticAlgorithms/RatioIntermediateGeneration", &ratioIntermediateGeneration, 0.5);
    ThreadsManager().CritSectionConfig().Leave();

    // Get intermediate generation size
    int intermediateGenerationSize = ratioIntermediateGeneration*m_PopSize;

    switch (m_NaturalSelectionType)
    {
        case (RatioElitism):
        {
            asLogMessage(_("Natural selection: ratio elitism"));

            for (int i=0; i<intermediateGenerationSize; i++)
            {
                m_ParametersTemp.push_back(m_Parameters[i]);
                m_ScoresCalibTemp.push_back(m_ScoresCalib[i]);
            }
            break;
        }

        case (Tournament):
        {
            asLogMessage(_("Natural selection: tournament"));

            double tournamentSelectionProbability;
            pConfig->Read("/Calibration/GeneticAlgorithms/NaturalSelectionTournamentProbability", &tournamentSelectionProbability, 0.9);

            for (int i=0; i<intermediateGenerationSize; i++)
            {
                // Choose candidates
                int candidateFinal = 0;
                int candidate1 = asTools::Random(0, m_Parameters.size()-1, 1);
                int candidate2 = asTools::Random(0, m_Parameters.size()-1, 1);

                // Check they are not the same
                while (candidate1==candidate2)
                {
                    candidate2 = asTools::Random(0, m_Parameters.size()-1, 1);
                }

                // Check probability of selection of the best
                bool keepBest = (asTools::Random(0.0, 1.0)<=tournamentSelectionProbability);

                // Use indexes as scores are already sorted (smaller is better)
                if (keepBest)
                {
                    if (candidate1<candidate2)
                    {
                        candidateFinal = candidate1;
                    }
                    else
                    {
                        candidateFinal = candidate2;
                    }
                }
                else
                {
                    if (candidate1<candidate2)
                    {
                        candidateFinal = candidate2;
                    }
                    else
                    {
                        candidateFinal = candidate1;
                    }
                }

                // If both scores are equal, select randomly and overwrite previous selection
                if (m_ScoresCalib[candidate1]==m_ScoresCalib[candidate2])
                {
                    double randomIndex = asTools::Random(0.0,1.0);
                    if (randomIndex<=0.5)
                    {
                        candidateFinal = candidate1;
                    }
                    else
                    {
                        candidateFinal = candidate2;
                    }
                }

                m_ParametersTemp.push_back(m_Parameters[candidateFinal]);
                m_ScoresCalibTemp.push_back(m_ScoresCalib[candidateFinal]);
            }
            break;
        }

        default:
        {
            asLogError(_("The given natural selection method couldn't be found."));
            return false;
        }
    }

    ElitismAfterSelection();

    return true;
}

bool asMethodOptimizerGeneticAlgorithms::Mating()
{
    // Different operators consider that the scores are sorted !
    SortScoresAndParametersTemp();

    wxASSERT(m_ParametersTemp.size()==m_ScoresCalibTemp.size());

    asLogMessage(_("Applying mating."));

    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    ThreadsManager().CritSectionConfig().Leave();

    // Build chromosomes
    for (unsigned int i=0; i<m_ParametersTemp.size(); i++)
    {
        wxASSERT(m_ParametersTemp[i].GetChromosomeLength()>0);
    }

    int sizeParents = m_ParametersTemp.size();
    int counter = 0;
    int counterSame = 0;
    bool initialized = false;
    VectorDouble probabilities;

    while (m_ParametersTemp.size()<(unsigned)m_PopSize)
    {
        // Couples selection only in the parents pool
        asLogMessage(_("Selecting couples."));
        int partner1 = 0, partner2 = 0;
        switch (m_CouplesSelectionType)
        {
            case (RankPairing):
            {
                asLogMessage(_("Couples selection: rank pairing"));

                partner1 = counter*2; // pairs
                partner2 = counter*2+1; // impairs

                // Check that we don't overflow from the array
                if (partner2>=sizeParents)
                {
                    partner1 = asTools::Random(0, sizeParents-1, 1);
                    partner2 = asTools::Random(0, sizeParents-1, 1);
                }
                break;
            }

            case (Random):
            {
                asLogMessage(_("Couples selection: random"));

                partner1 = asTools::Random(0, sizeParents-1, 1);
                partner2 = asTools::Random(0, sizeParents-1, 1);
                break;
            }

            case (RouletteWheelRank):
            {
                asLogMessage(_("Couples selection: roulette wheel rank"));

                // If the first round, initialize the probabilities.
                if (!initialized)
                {
                    double sum = 0;
                    probabilities.push_back(0.0);
                    for (int i=0; i<sizeParents; i++)
                    {
                        sum += i+1;
                    }
                    for (int i=0; i<sizeParents; i++)
                    {
                        double currentRank = i+1.0;
                        double prob = (sizeParents-currentRank+1.0) / sum;
                        double probCumul = prob+probabilities[probabilities.size()-1];
                        probabilities.push_back(probCumul);
                    }
                    if(abs(probabilities[probabilities.size()-1]-1.0)>0.00001)
                    {
                        double diff = probabilities[probabilities.size()-1]-1.0;
                        asLogWarning(wxString::Format(_("probabilities[last]-1.0=%f"),diff));
                    }
                    initialized = true;
                }

                // Select mates
                double partner1prob = asTools::Random(0.0, 1.0);
                partner1 = asTools::SortedArraySearchFloor(&probabilities[0], &probabilities[probabilities.size()-1], partner1prob);
                double partner2prob = asTools::Random(0.0, 1.0);
                partner2 = asTools::SortedArraySearchFloor(&probabilities[0], &probabilities[probabilities.size()-1], partner2prob);

                break;
            }

            case (RouletteWheelScore):
            {
                asLogMessage(_("Couples selection: roulette wheel score"));

                // If the first round, initialize the probabilities.
                if (!initialized)
                {
                    double sum = 0;
                    probabilities.push_back(0.0);
                    for (int i=0; i<sizeParents; i++)
                    {
                        sum += m_ScoresCalibTemp[i]-m_ScoresCalibTemp[sizeParents-1]+0.001; // 0.001 to avoid null probs
                    }
                    for (int i=0; i<sizeParents; i++)
                    {
                        if (sum>0)
                        {
                            double currentScore = m_ScoresCalibTemp[i]-m_ScoresCalibTemp[sizeParents-1]+0.001;
                            double prob = currentScore / sum;
                            double probCumul = prob+probabilities[probabilities.size()-1];
                            probabilities.push_back(probCumul);
                        }
                        else
                        {
                            asLogError(_("The sum of the probabilities is null."));
                            return false;
                        }
                    }
                    if(abs(probabilities[probabilities.size()-1]-1.0)>0.00001)
                    {
                        double diff = probabilities[probabilities.size()-1]-1.0;
                        asLogWarning(wxString::Format(_("probabilities[last]-1.0=%f"),diff));
                    }
                    initialized = true;
                }

                wxASSERT(probabilities.size()>2);

                // Select mates
                double partner1prob = asTools::Random(0.0, 1.0);
                partner1 = asTools::SortedArraySearchFloor(&probabilities[0], &probabilities[probabilities.size()-1], partner1prob);
                double partner2prob = asTools::Random(0.0, 1.0);
                partner2 = asTools::SortedArraySearchFloor(&probabilities[0], &probabilities[probabilities.size()-1], partner2prob);

                if (partner1<0)
                {
                    asLogError(_("Could not find a value in the probability distribution."));
                    asLogError(wxString::Format("probabilities[0] = %g, &probabilities[%d] = %g, partner1prob = %g", probabilities[0], (int)probabilities.size()-1, probabilities[probabilities.size()-1], partner1prob));
                    return false;
                }
                if (partner2<0)
                {
                    asLogError(_("Could not find a value in the probability distribution."));
                    asLogError(wxString::Format("probabilities[0] = %g, &probabilities[%d] = %g, partner2prob = %g", probabilities[0], (int)probabilities.size()-1, probabilities[probabilities.size()-1], partner2prob));
                    return false;
                }

                break;
            }

            case (TournamentCompetition):
            {
                asLogMessage(_("Couples selection: tournament"));

                // Get nb of points
                int couplesSelectionTournamentNb;
                ThreadsManager().CritSectionConfig().Enter();
                pConfig->Read("/Calibration/GeneticAlgorithms/CouplesSelectionTournamentNb", &couplesSelectionTournamentNb, 3);
                ThreadsManager().CritSectionConfig().Leave();
                if (couplesSelectionTournamentNb<2)
                {
                    asLogWarning(_("The number of individuals for tournament selection is inferior to 2."));
                    asLogWarning(_("The number of individuals for tournament selection has been changed."));
                    couplesSelectionTournamentNb = 2;
                }
                if (couplesSelectionTournamentNb>sizeParents/2)
                {
                    asLogWarning(_("The number of individuals for tournament selection superior to the half of the intermediate population."));
                    asLogWarning(_("The number of individuals for tournament selection has been changed."));
                    couplesSelectionTournamentNb = sizeParents/2;
                }

                // Select partner 1
                partner1 = sizeParents;
                for (int i=0; i<couplesSelectionTournamentNb; i++)
                {
                    int candidate = asTools::Random(0, sizeParents-1);
                    if (candidate<partner1) // Smaller rank reflects better score
                    {
                        partner1 = candidate;
                    }
                }

                // Select partner 2
                partner2 = sizeParents;
                for (int i=0; i<couplesSelectionTournamentNb; i++)
                {
                    int candidate = asTools::Random(0, sizeParents-1);
                    if (candidate<partner2) // Smaller rank reflects better score
                    {
                        partner2 = candidate;
                    }
                }

                break;
            }

            default:
            {
                asLogError(_("The desired couples selection method is not yet implemented."));
            }

        }

        asLogMessage(wxString::Format("partner1 = %d, partner2 = %d", partner1, partner2));


        // Check that we don't have the same individual
        if (partner1==partner2)
        {
            counterSame++;
            if(counterSame>=100)
            {
                for (int i=0; i<sizeParents; i++)
                {
                    asLogWarning(wxString::Format(_("m_ScoresCalibTemp[%d] = %f"), i, m_ScoresCalibTemp[i]));
                }

                for (unsigned int i=0; i<probabilities.size(); i++)
                {
                    asLogWarning(wxString::Format(_("probabilities[%d] = %f"), i, probabilities[i]));
                }
                asLogError(_("The same two partners were chosen more than 100 times. Lost in a loop."));
                return false;
            }
            continue;
        }
        else
        {
            counterSame = 0;
        }

        // Chromosomes crossings
        asLogMessage(_("Crossing chromosomes."));
        switch (m_CrossoverType)
        {
            case (SinglePointCrossover):
            {
                asLogMessage(_("Crossing: single point crossover"));

                // Get nb of points
                int crossoverNbPoints = 1;

                // Get points
                int chromosomeLength = m_ParametersTemp[partner1].GetChromosomeLength();

                VectorInt crossingPoints;
                for (int i_cross=0; i_cross<crossoverNbPoints; i_cross++)
                {
                    int crossingPoint = asTools::Random(0,chromosomeLength-1,1);
                    crossingPoints.push_back(crossingPoint);
                }

                // Proceed to crossover
                wxASSERT(m_ParametersTemp.size()>(unsigned)partner1);
                wxASSERT(m_ParametersTemp.size()>(unsigned)partner2);
                asParametersOptimizationGAs param1;
                param1 = m_ParametersTemp[partner1];
                asParametersOptimizationGAs param2;
                param2 = m_ParametersTemp[partner2];
                param1.SimpleCrossover(param2, crossingPoints);

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                m_ParametersTemp.push_back(param1);
                m_ScoresCalibTemp.push_back(NaNFloat);
                if (m_PopSize-m_ParametersTemp.size()>(unsigned)0)
                {
                    param2.CheckRange();

                    m_ParametersTemp.push_back(param2);
                    m_ScoresCalibTemp.push_back(NaNFloat);
                }

                break;
            }

            case (DoublePointsCrossover):
            {
                asLogMessage(_("Crossing: double points crossover"));

                // Get nb of points
                int crossoverNbPoints = 2;

                // Get points
                int chromosomeLength = m_ParametersTemp[partner1].GetChromosomeLength();

                VectorInt crossingPoints;
                for (int i_cross=0; i_cross<crossoverNbPoints; i_cross++)
                {
                    int crossingPoint = asTools::Random(0,chromosomeLength-1,1);
                    if (crossingPoints.size()>0)
                    {
                        // Check that is not already stored
                        if (chromosomeLength>crossoverNbPoints)
                        {
                            for (unsigned int i_pts=0; i_pts<crossingPoints.size(); i_pts++)
                            {
                                if (crossingPoints[i_pts]==crossingPoint)
                                {
                                    crossingPoints.erase (crossingPoints.begin()+i_pts);
                                    asLogMessage(_("Crossing point already selected. Selection of a new one."));
                                    i_cross--;
                                    break;
                                }
                            }
                        }
                        else
                        {
                            asLogMessage(_("There are more crossing points than chromosomes."));
                        }
                    }
                    crossingPoints.push_back(crossingPoint);
                }

                // Proceed to crossover
                wxASSERT(m_ParametersTemp.size()>(unsigned)partner1);
                wxASSERT(m_ParametersTemp.size()>(unsigned)partner2);
                asParametersOptimizationGAs param1;
                param1 = m_ParametersTemp[partner1];
                asParametersOptimizationGAs param2;
                param2 = m_ParametersTemp[partner2];
                param1.SimpleCrossover(param2, crossingPoints);

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                m_ParametersTemp.push_back(param1);
                m_ScoresCalibTemp.push_back(NaNFloat);
                if (m_PopSize-m_ParametersTemp.size()>(unsigned)0)
                {
                    param2.CheckRange();

                    m_ParametersTemp.push_back(param2);
                    m_ScoresCalibTemp.push_back(NaNFloat);
                }
                break;
            }

            case (MultiplePointsCrossover):
            {
                asLogMessage(_("Crossing: multiple points crossover"));

                // Get nb of points
                int crossoverNbPoints;
                ThreadsManager().CritSectionConfig().Enter();
                pConfig->Read("/Calibration/GeneticAlgorithms/CrossoverMultiplePointsNb", &crossoverNbPoints, 3);
                ThreadsManager().CritSectionConfig().Leave();

                // Get points
                int chromosomeLength = m_ParametersTemp[partner1].GetChromosomeLength();

                if(crossoverNbPoints>=chromosomeLength)
                {
                    asLogError(_("The desired crossings number is superior or equal to the genes number."));
                    return false;
                }

                VectorInt crossingPoints;
                for (int i_cross=0; i_cross<crossoverNbPoints; i_cross++)
                {
                    int crossingPoint = asTools::Random(0,chromosomeLength-1,1);
                    if (crossingPoints.size()>0)
                    {
                        // Check that is not already stored
                        if (chromosomeLength>crossoverNbPoints)
                        {
                            for (unsigned int i_pts=0; i_pts<crossingPoints.size(); i_pts++)
                            {
                                if (crossingPoints[i_pts]==crossingPoint)
                                {
                                    crossingPoints.erase (crossingPoints.begin()+i_pts);
                                    asLogMessage(_("Crossing point already selected. Selection of a new one."));
                                    i_cross--;
                                    break;
                                }
                            }
                        }
                        else
                        {
                            asLogMessage(_("There are more crossing points than chromosomes."));
                        }
                    }
                    crossingPoints.push_back(crossingPoint);
                }

                // Proceed to crossover
                wxASSERT(m_ParametersTemp.size()>(unsigned)partner1);
                wxASSERT(m_ParametersTemp.size()>(unsigned)partner2);
                asParametersOptimizationGAs param1;
                param1 = m_ParametersTemp[partner1];
                asParametersOptimizationGAs param2;
                param2 = m_ParametersTemp[partner2];
                param1.SimpleCrossover(param2, crossingPoints);

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                m_ParametersTemp.push_back(param1);
                m_ScoresCalibTemp.push_back(NaNFloat);
                if (m_PopSize-m_ParametersTemp.size()>(unsigned)0)
                {
                    param2.CheckRange();

                    m_ParametersTemp.push_back(param2);
                    m_ScoresCalibTemp.push_back(NaNFloat);
                }
                break;
            }

            case (UniformCrossover):
            {
                asLogMessage(_("Crossing: uniform crossover"));

                // Get points
                int chromosomeLength = m_ParametersTemp[partner1].GetChromosomeLength();

                VectorInt crossingPoints;
                bool previouslyCrossed = false; // flag

                for (int i_gene=0; i_gene<chromosomeLength; i_gene++)
                {
                    double doCross = asTools::Random(0.0,1.0);

                    if (doCross>=0.5) // Yes
                    {
                        if (!previouslyCrossed) // If situation changes
                        {
                            crossingPoints.push_back(i_gene);
                        }
                        previouslyCrossed = true;
                    }
                    else // No
                    {
                        if (previouslyCrossed) // If situation changes
                        {
                            crossingPoints.push_back(i_gene);
                        }
                        previouslyCrossed = false;
                    }
                }

                // Proceed to crossover
                wxASSERT(m_ParametersTemp.size()>(unsigned)partner1);
                wxASSERT(m_ParametersTemp.size()>(unsigned)partner2);
                asParametersOptimizationGAs param1;
                param1 = m_ParametersTemp[partner1];
                asParametersOptimizationGAs param2;
                param2 = m_ParametersTemp[partner2];
                if (crossingPoints.size()>0)
                {
                    param1.SimpleCrossover(param2, crossingPoints);
                }

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                m_ParametersTemp.push_back(param1);
                m_ScoresCalibTemp.push_back(NaNFloat);
                if (m_PopSize-m_ParametersTemp.size()>(unsigned)0)
                {
                    param2.CheckRange();

                    m_ParametersTemp.push_back(param2);
                    m_ScoresCalibTemp.push_back(NaNFloat);
                }
                break;
            }

            case (LimitedBlending):
            {
                asLogMessage(_("Crossing: limited blending"));

                // Get nb of points
                int crossoverNbPoints;
                ThreadsManager().CritSectionConfig().Enter();
                pConfig->Read("/Calibration/GeneticAlgorithms/CrossoverBlendingPointsNb", &crossoverNbPoints, 2);

                // Get option to share beta or to generate a new one at every step
                bool shareBeta;
                pConfig->Read("/Calibration/GeneticAlgorithms/CrossoverBlendingShareBeta", &shareBeta, true);
                ThreadsManager().CritSectionConfig().Leave();

                // Get points
                int chromosomeLength = m_ParametersTemp[partner1].GetChromosomeLength();

                VectorInt crossingPoints;
                for (int i_cross=0; i_cross<crossoverNbPoints; i_cross++)
                {
                    int crossingPoint = asTools::Random(0,chromosomeLength-1,1);
                    if (crossingPoints.size()>0)
                    {
                        // Check that is not already stored
                        if (chromosomeLength>crossoverNbPoints)
                        {
                            for (unsigned int i_pts=0; i_pts<crossingPoints.size(); i_pts++)
                            {
                                if (crossingPoints[i_pts]==crossingPoint)
                                {
                                    crossingPoints.erase (crossingPoints.begin()+i_pts);
                                    asLogMessage(_("Crossing point already selected. Selection of a new one."));
                                    i_cross--;
                                    break;
                                }
                            }
                        }
                        else
                        {
                            asLogMessage(_("There are more crossing points than chromosomes."));
                        }
                    }
                    crossingPoints.push_back(crossingPoint);
                }

                // Proceed to crossover
                wxASSERT(m_ParametersTemp.size()>(unsigned)partner1);
                wxASSERT(m_ParametersTemp.size()>(unsigned)partner2);
                asParametersOptimizationGAs param1;
                param1 = m_ParametersTemp[partner1];
                asParametersOptimizationGAs param2;
                param2 = m_ParametersTemp[partner2];
                param1.BlendingCrossover(param2, crossingPoints, shareBeta);

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                m_ParametersTemp.push_back(param1);
                m_ScoresCalibTemp.push_back(NaNFloat);
                if (m_PopSize-m_ParametersTemp.size()>(unsigned)0)
                {
                    param2.CheckRange();

                    m_ParametersTemp.push_back(param2);
                    m_ScoresCalibTemp.push_back(NaNFloat);
                }
                break;
            }

            case (LinearCrossover):
            {
                asLogMessage(_("Crossing: linear crossover"));

                // Get nb of points
                int crossoverNbPoints;
                ThreadsManager().CritSectionConfig().Enter();
                pConfig->Read("/Calibration/GeneticAlgorithms/CrossoverLinearPointsNb", &crossoverNbPoints, 2);
                ThreadsManager().CritSectionConfig().Leave();

                // Get points
                int chromosomeLength = m_ParametersTemp[partner1].GetChromosomeLength();

                VectorInt crossingPoints;
                for (int i_cross=0; i_cross<crossoverNbPoints; i_cross++)
                {
                    int crossingPoint = asTools::Random(0,chromosomeLength-1,1);
                    if (crossingPoints.size()>0)
                    {
                        // Check that is not already stored
                        if (chromosomeLength>crossoverNbPoints)
                        {
                            for (unsigned int i_pts=0; i_pts<crossingPoints.size(); i_pts++)
                            {
                                if (crossingPoints[i_pts]==crossingPoint)
                                {
                                    crossingPoints.erase (crossingPoints.begin()+i_pts);
                                    asLogMessage(_("Crossing point already selected. Selection of a new one."));
                                    i_cross--;
                                    break;
                                }
                            }
                        }
                        else
                        {
                            asLogMessage(_("There are more crossing points than chromosomes."));
                        }
                    }
                    crossingPoints.push_back(crossingPoint);
                }

                // Proceed to crossover
                wxASSERT(m_ParametersTemp.size()>(unsigned)partner1);
                wxASSERT(m_ParametersTemp.size()>(unsigned)partner2);
                asParametersOptimizationGAs param1;
                param1 = m_ParametersTemp[partner1];
                asParametersOptimizationGAs param2;
                param2 = m_ParametersTemp[partner2];
                asParametersOptimizationGAs param3;
                param3 = m_ParametersTemp[partner2];
                param1.LinearCrossover(param2, param3, crossingPoints);

                if (param1.IsInRange())
                {
                    param1.CheckRange();

                    m_ParametersTemp.push_back(param1);
                    m_ScoresCalibTemp.push_back(NaNFloat);
                }

                // Add the other parameters if ther is enough room
                if (m_PopSize-m_ParametersTemp.size()>(unsigned)0)
                {
                    if (param2.IsInRange())
                    {
                        param2.CheckRange();

                        m_ParametersTemp.push_back(param2);
                        m_ScoresCalibTemp.push_back(NaNFloat);
                    }
                }
                if (m_PopSize-m_ParametersTemp.size()>(unsigned)0)
                {
                    if (param3.IsInRange())
                    {
                        param3.CheckRange();

                        m_ParametersTemp.push_back(param3);
                        m_ScoresCalibTemp.push_back(NaNFloat);
                    }
                }

                break;
            }

            case (HeuristicCrossover):
            {
                asLogMessage(_("Crossing: heuristic crossover"));

                // Get nb of points
                int crossoverNbPoints;
                ThreadsManager().CritSectionConfig().Enter();
                pConfig->Read("/Calibration/GeneticAlgorithms/CrossoverHeuristicPointsNb", &crossoverNbPoints, 2);

                // Get option to share beta or to generate a new one at every step
                bool shareBeta;
                pConfig->Read("/Calibration/GeneticAlgorithms/CrossoverHeuristicShareBeta", &shareBeta, true);
                ThreadsManager().CritSectionConfig().Leave();

                // Get points
                int chromosomeLength = m_ParametersTemp[partner1].GetChromosomeLength();

                VectorInt crossingPoints;
                for (int i_cross=0; i_cross<crossoverNbPoints; i_cross++)
                {
                    int crossingPoint = asTools::Random(0,chromosomeLength-1,1);
                    if (crossingPoints.size()>0)
                    {
                        // Check that is not already stored
                        if (chromosomeLength>crossoverNbPoints)
                        {
                            for (unsigned int i_pts=0; i_pts<crossingPoints.size(); i_pts++)
                            {
                                if (crossingPoints[i_pts]==crossingPoint)
                                {
                                    crossingPoints.erase (crossingPoints.begin()+i_pts);
                                    asLogMessage(_("Crossing point already selected. Selection of a new one."));
                                    i_cross--;
                                    break;
                                }
                            }
                        }
                        else
                        {
                            asLogMessage(_("There are more crossing points than chromosomes."));
                        }
                    }
                    crossingPoints.push_back(crossingPoint);
                }

                // Proceed to crossover
                wxASSERT(m_ParametersTemp.size()>(unsigned)partner1);
                wxASSERT(m_ParametersTemp.size()>(unsigned)partner2);
                asParametersOptimizationGAs param1;
                param1 = m_ParametersTemp[partner1];
                asParametersOptimizationGAs param2;
                param2 = m_ParametersTemp[partner2];
                param1.HeuristicCrossover(param2, crossingPoints, shareBeta);

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                m_ParametersTemp.push_back(param1);
                m_ScoresCalibTemp.push_back(NaNFloat);
                if (m_PopSize-m_ParametersTemp.size()>(unsigned)0)
                {
                    param2.CheckRange();

                    m_ParametersTemp.push_back(param2);
                    m_ScoresCalibTemp.push_back(NaNFloat);
                }
                break;
            }

            case (BinaryLikeCrossover):
            {
                asLogMessage(_("Crossing: binary-like crossover"));

                // Get nb of points
                ThreadsManager().CritSectionConfig().Enter();
                int crossoverNbPoints;
                pConfig->Read("/Calibration/GeneticAlgorithms/CrossoverBinaryLikePointsNb", &crossoverNbPoints, 2);

                // Get option to share beta or to generate a new one at every step
                bool shareBeta;
                pConfig->Read("/Calibration/GeneticAlgorithms/CrossoverBinaryLikeShareBeta", &shareBeta, true);
                ThreadsManager().CritSectionConfig().Leave();

                // Get points
                int chromosomeLength = m_ParametersTemp[partner1].GetChromosomeLength();

                VectorInt crossingPoints;
                for (int i_cross=0; i_cross<crossoverNbPoints; i_cross++)
                {
                    int crossingPoint = asTools::Random(0,chromosomeLength-1,1);
                    if (crossingPoints.size()>0)
                    {
                        // Check that is not already stored
                        if (chromosomeLength>crossoverNbPoints)
                        {
                            for (unsigned int i_pts=0; i_pts<crossingPoints.size(); i_pts++)
                            {
                                if (crossingPoints[i_pts]==crossingPoint)
                                {
                                    crossingPoints.erase (crossingPoints.begin()+i_pts);
                                    asLogMessage(_("Crossing point already selected. Selection of a new one."));
                                    i_cross--;
                                    break;
                                }
                            }
                        }
                        else
                        {
                            asLogMessage(_("There are more crossing points than chromosomes."));
                        }
                    }
                    crossingPoints.push_back(crossingPoint);
                }

                // Proceed to crossover
                wxASSERT(m_ParametersTemp.size()>(unsigned)partner1);
                wxASSERT(m_ParametersTemp.size()>(unsigned)partner2);
                asParametersOptimizationGAs param1;
                param1 = m_ParametersTemp[partner1];
                asParametersOptimizationGAs param2;
                param2 = m_ParametersTemp[partner2];
                param1.BinaryLikeCrossover(param2, crossingPoints, shareBeta);

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                m_ParametersTemp.push_back(param1);
                m_ScoresCalibTemp.push_back(NaNFloat);
                if (m_PopSize-m_ParametersTemp.size()>(unsigned)0)
                {
                    param2.CheckRange();

                    m_ParametersTemp.push_back(param2);
                    m_ScoresCalibTemp.push_back(NaNFloat);
                }
                break;
            }

            case (LinearInterpolation):
            {
                asLogMessage(_("Crossing: linear interpolation"));

                // Proceed to crossover
                wxASSERT(m_ParametersTemp.size()>(unsigned)partner1);
                wxASSERT(m_ParametersTemp.size()>(unsigned)partner2);
                asParametersOptimizationGAs param1;
                param1 = m_ParametersTemp[partner1];
                asParametersOptimizationGAs param2;
                param2 = m_ParametersTemp[partner2];
                param1.LinearInterpolation(param2, true);

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                m_ParametersTemp.push_back(param1);
                m_ScoresCalibTemp.push_back(NaNFloat);
                if (m_PopSize-m_ParametersTemp.size()>(unsigned)0)
                {
                    param2.CheckRange();

                    m_ParametersTemp.push_back(param2);
                    m_ScoresCalibTemp.push_back(NaNFloat);
                }
                break;
            }

            case (FreeInterpolation):
            {
                asLogMessage(_("Crossing: free interpolation"));

                // Proceed to crossover
                wxASSERT(m_ParametersTemp.size()>(unsigned)partner1);
                wxASSERT(m_ParametersTemp.size()>(unsigned)partner2);
                asParametersOptimizationGAs param1;
                param1 = m_ParametersTemp[partner1];
                asParametersOptimizationGAs param2;
                param2 = m_ParametersTemp[partner2];
                param1.LinearInterpolation(param2, false);

                param1.CheckRange();

                // Add the new parameters if ther is enough room
                m_ParametersTemp.push_back(param1);
                m_ScoresCalibTemp.push_back(NaNFloat);
                if (m_PopSize-m_ParametersTemp.size()>(unsigned)0)
                {
                    param2.CheckRange();

                    m_ParametersTemp.push_back(param2);
                    m_ScoresCalibTemp.push_back(NaNFloat);
                }
                break;
            }

            default:
            {
                asLogError(_("The desired chromosomes crossing is not yet implemented."));
            }
        }

        counter++;
    }

    wxASSERT_MSG(m_ParametersTemp.size()==m_PopSize, wxString::Format("m_ParametersTemp.size() = %d, m_PopSize = %d", (int)m_ParametersTemp.size(), m_PopSize));
    wxASSERT(m_ParametersTemp.size()==m_ScoresCalibTemp.size());

    return true;
}

bool asMethodOptimizerGeneticAlgorithms::Mutatation()
{
    // NB: The parameters and scores are already sorted !

    asLogMessage(_("Applying mutations."));

    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    ThreadsManager().CritSectionConfig().Leave();

    m_Parameters.clear();
    m_ScoresCalib.clear();
    m_Parameters = m_ParametersTemp;
    m_ScoresCalib = m_ScoresCalibTemp;

    switch (m_MutationsModeType)
    {
        case (RandomUniformConstant):
        {
            double mutationsProbability;
            ThreadsManager().CritSectionConfig().Enter();
            pConfig->Read("/Calibration/GeneticAlgorithms/MutationsUniformConstantProbability", &mutationsProbability, 0.2);
            ThreadsManager().CritSectionConfig().Leave();

            for (unsigned int i_ind=0; i_ind<m_Parameters.size(); i_ind++)
            {
                // Mutate
                bool hasMutated = false;
                m_Parameters[i_ind].MutateUniformDistribution(mutationsProbability, hasMutated);
                if(hasMutated) m_ScoresCalib[i_ind] = NaNFloat;

                m_Parameters[i_ind].FixWeights();
                m_Parameters[i_ind].FixCoordinates();
                m_Parameters[i_ind].CheckRange();
                m_Parameters[i_ind].FixAnalogsNb();
            }
            break;
        }

        case (RandomUniformVariable):
        {
            int nbGenMax;
            double probStart, probEnd;
            ThreadsManager().CritSectionConfig().Enter();
            pConfig->Read("/Calibration/GeneticAlgorithms/MutationsUniformVariableMaxGensNbVar", &nbGenMax, 50);
            pConfig->Read("/Calibration/GeneticAlgorithms/MutationsUniformVariableProbabilityStart", &probStart, 0.5);
            pConfig->Read("/Calibration/GeneticAlgorithms/MutationsUniformVariableProbabilityEnd", &probEnd, 0.01);
            ThreadsManager().CritSectionConfig().Leave();

            double probIncrease = (probStart-probEnd)/(double)nbGenMax;
            double mutationsProbability = probStart+probIncrease*wxMin(m_GenerationNb-1,nbGenMax);

            for (unsigned int i_ind=0; i_ind<m_Parameters.size(); i_ind++)
            {
                // Mutate
                bool hasMutated = false;
                m_Parameters[i_ind].MutateUniformDistribution(mutationsProbability, hasMutated);
                if(hasMutated) m_ScoresCalib[i_ind] = NaNFloat;

                m_Parameters[i_ind].FixWeights();
                m_Parameters[i_ind].FixCoordinates();
                m_Parameters[i_ind].CheckRange();
                m_Parameters[i_ind].FixAnalogsNb();
            }
            break;
        }

        case (RandomNormalConstant):
        {
            double mutationsProbability;
            double stdDevRatioRange;
            ThreadsManager().CritSectionConfig().Enter();
            pConfig->Read("/Calibration/GeneticAlgorithms/MutationsNormalConstantProbability", &mutationsProbability, 0.2);
            pConfig->Read("/Calibration/GeneticAlgorithms/MutationsNormalConstantStdDevRatioRange", &stdDevRatioRange, 0.10);
            ThreadsManager().CritSectionConfig().Leave();

            for (unsigned int i_ind=0; i_ind<m_Parameters.size(); i_ind++)
            {
                // Mutate
                bool hasMutated = false;
                m_Parameters[i_ind].MutateNormalDistribution(mutationsProbability, stdDevRatioRange, hasMutated);
                if(hasMutated) m_ScoresCalib[i_ind] = NaNFloat;

                m_Parameters[i_ind].FixWeights();
                m_Parameters[i_ind].FixCoordinates();
                m_Parameters[i_ind].CheckRange();
                m_Parameters[i_ind].FixAnalogsNb();
            }
            break;
        }

        case (RandomNormalVariable):
        {
            int nbGenMaxProb, nbGenMaxStdDev;
            double probStart, probEnd;
            double stdDevStart, stdDevEnd;
            ThreadsManager().CritSectionConfig().Enter();
            pConfig->Read("/Calibration/GeneticAlgorithms/MutationsNormalVariableMaxGensNbVarProb", &nbGenMaxProb, 50);
            pConfig->Read("/Calibration/GeneticAlgorithms/MutationsNormalVariableMaxGensNbVarStdDev", &nbGenMaxStdDev, 50);
            pConfig->Read("/Calibration/GeneticAlgorithms/MutationsNormalVariableProbabilityStart", &probStart, 0.5);
            pConfig->Read("/Calibration/GeneticAlgorithms/MutationsNormalVariableProbabilityEnd", &probEnd, 0.05);
            pConfig->Read("/Calibration/GeneticAlgorithms/MutationsNormalVariableStdDevStart", &stdDevStart, 0.5);
            pConfig->Read("/Calibration/GeneticAlgorithms/MutationsNormalVariableStdDevEnd", &stdDevEnd, 0.01);
            ThreadsManager().CritSectionConfig().Leave();

            double probIncrease = (probStart-probEnd)/(double)nbGenMaxProb;
            double mutationsProbability = probStart+probIncrease*wxMin(m_GenerationNb-1,nbGenMaxProb);

            double stdDevIncrease = (stdDevStart-stdDevEnd)/(double)nbGenMaxStdDev;
            double stdDevRatioRange = stdDevStart+stdDevIncrease*wxMin(m_GenerationNb-1,nbGenMaxStdDev);

            for (unsigned int i_ind=0; i_ind<m_Parameters.size(); i_ind++)
            {
                // Mutate
                bool hasMutated = false;
                m_Parameters[i_ind].MutateNormalDistribution(mutationsProbability, stdDevRatioRange, hasMutated);
                if(hasMutated) m_ScoresCalib[i_ind] = NaNFloat;

                m_Parameters[i_ind].FixWeights();
                m_Parameters[i_ind].FixCoordinates();
                m_Parameters[i_ind].CheckRange();
                m_Parameters[i_ind].FixAnalogsNb();
            }
            break;
        }

        case (NonUniform):
        {
            int nbGenMax;
            double mutationsProbability, minRate;
            ThreadsManager().CritSectionConfig().Enter();
            pConfig->Read("/Calibration/GeneticAlgorithms/MutationsNonUniformProbability", &mutationsProbability, 0.2);
            pConfig->Read("/Calibration/GeneticAlgorithms/MutationsNonUniformMaxGensNbVar", &nbGenMax, 50);
            pConfig->Read("/Calibration/GeneticAlgorithms/MutationsNonUniformMinRate", &minRate, 0.20);
            ThreadsManager().CritSectionConfig().Leave();

            for (unsigned int i_ind=0; i_ind<m_Parameters.size(); i_ind++)
            {
                // Mutate
                bool hasMutated = false;
                m_Parameters[i_ind].MutateNonUniform(mutationsProbability, m_GenerationNb, nbGenMax, minRate, hasMutated);
                if(hasMutated) m_ScoresCalib[i_ind] = NaNFloat;

                m_Parameters[i_ind].FixWeights();
                m_Parameters[i_ind].FixCoordinates();
                m_Parameters[i_ind].CheckRange();
                m_Parameters[i_ind].FixAnalogsNb();
            }
            break;
        }

        case (SelfAdaptationRate):
        {
            for (unsigned int i_ind=0; i_ind<m_Parameters.size(); i_ind++)
            {
                // Mutate
                bool hasMutated = false;
                m_Parameters[i_ind].MutateSelfAdaptationRate(hasMutated);
                if(hasMutated) m_ScoresCalib[i_ind] = NaNFloat;

                m_Parameters[i_ind].FixWeights();
                m_Parameters[i_ind].FixCoordinates();
                m_Parameters[i_ind].CheckRange();
                m_Parameters[i_ind].FixAnalogsNb();
            }
            break;
        }

        case (SelfAdaptationRadius):
        {
            for (unsigned int i_ind=0; i_ind<m_Parameters.size(); i_ind++)
            {
                // Mutate
                bool hasMutated = false;
                m_Parameters[i_ind].MutateSelfAdaptationRadius(hasMutated);
                if(hasMutated) m_ScoresCalib[i_ind] = NaNFloat;

                m_Parameters[i_ind].FixWeights();
                m_Parameters[i_ind].FixCoordinates();
                m_Parameters[i_ind].CheckRange();
                m_Parameters[i_ind].FixAnalogsNb();
            }
            break;
        }

        case (SelfAdaptationRateChromosome):
        {
            for (unsigned int i_ind=0; i_ind<m_Parameters.size(); i_ind++)
            {
                // Mutate
                bool hasMutated = false;
                m_Parameters[i_ind].MutateSelfAdaptationRateChromosome(hasMutated);
                if(hasMutated) m_ScoresCalib[i_ind] = NaNFloat;

                m_Parameters[i_ind].FixWeights();
                m_Parameters[i_ind].FixCoordinates();
                m_Parameters[i_ind].CheckRange();
                m_Parameters[i_ind].FixAnalogsNb();
            }
            break;
        }

        case (SelfAdaptationRadiusChromosome):
        {
            for (unsigned int i_ind=0; i_ind<m_Parameters.size(); i_ind++)
            {
                // Mutate
                bool hasMutated = false;
                m_Parameters[i_ind].MutateSelfAdaptationRadiusChromosome(hasMutated);
                if(hasMutated) m_ScoresCalib[i_ind] = NaNFloat;

                m_Parameters[i_ind].FixWeights();
                m_Parameters[i_ind].FixCoordinates();
                m_Parameters[i_ind].CheckRange();
                m_Parameters[i_ind].FixAnalogsNb();
            }
            break;
        }

        case (MultiScale):
        {
            double mutationsProbability;
            ThreadsManager().CritSectionConfig().Enter();
            pConfig->Read("/Calibration/GeneticAlgorithms/MutationsMultiScaleProbability", &mutationsProbability, 0.2);
            ThreadsManager().CritSectionConfig().Leave();

            for (unsigned int i_ind=0; i_ind<m_Parameters.size(); i_ind++)
            {
                // Mutate
                bool hasMutated = false;
                m_Parameters[i_ind].MutateMultiScale(mutationsProbability, hasMutated);
                if(hasMutated) m_ScoresCalib[i_ind] = NaNFloat;

                m_Parameters[i_ind].FixWeights();
                m_Parameters[i_ind].FixCoordinates();
                m_Parameters[i_ind].CheckRange();
                m_Parameters[i_ind].FixAnalogsNb();
            }
            break;
        }

        case (NoMutation):
        {
            // Nothing to do
            break;
        }

        default:
        {
            asLogError(_("The desired mutation method is not yet implemented."));
        }
    }

    wxASSERT(m_ParametersTemp.size()==m_ScoresCalibTemp.size());

    return true;
}
