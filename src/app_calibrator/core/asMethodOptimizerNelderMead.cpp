#include "asMethodOptimizerNelderMead.h"

#ifndef UNIT_TESTING
    #include <AtmoswingAppCalibrator.h>
#endif

asMethodOptimizerNelderMead::asMethodOptimizerNelderMead()
:
asMethodOptimizer()
{
    // Load preferences
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Read("/Calibration/NelderMead/Rho", &m_NelderMeadRho, float(1.0)); // reflection
    pConfig->Read("/Calibration/NelderMead/Chi", &m_NelderMeadChi, float(2.0)); // expansion
    pConfig->Read("/Calibration/NelderMead/Gamma", &m_NelderMeadGamma, float(0.5)); // contraction
    pConfig->Read("/Calibration/NelderMead/Sigma", &m_NelderMeadSigma, float(0.5)); // reduction
    ThreadsManager().CritSectionConfig().Leave();
}

asMethodOptimizerNelderMead::~asMethodOptimizerNelderMead()
{
    //dtor
}

void asMethodOptimizerNelderMead::ClearAll()
{
    m_ParametersTemp.clear();
    m_ScoresCalibTemp.clear();
    m_Parameters.clear();
    m_ScoresCalib.clear();
    m_ScoreValid = NaNFloat;
}

void asMethodOptimizerNelderMead::ClearTemp()
{
    m_ParametersTemp.clear();
    m_ScoresCalibTemp.clear();
}

void asMethodOptimizerNelderMead::SortScoresAndParameters()
{
    wxASSERT(m_ScoresCalib.size()==m_Parameters.size());
    wxASSERT(m_ScoresCalib.size()>=1);
    wxASSERT(m_Parameters.size()>=1);

    if (m_Parameters.size()==1) return;

    // Sort according to the score
    Array1DFloat vIndices = Array1DFloat::LinSpaced(Eigen::Sequential, m_ParamsNb, 0, m_ParamsNb-1);
    asTools::SortArrays(&m_ScoresCalib[0],&m_ScoresCalib[m_ParamsNb-1],&vIndices[0],&vIndices[m_ParamsNb-1],m_ScoreOrder);

    // Sort the parameters sets as the scores
    std::vector <asParametersOptimizationNelderMead> copyParameters;
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

void asMethodOptimizerNelderMead::SortScoresAndParametersTemp()
{
    wxASSERT(m_ScoresCalibTemp.size()==m_ParametersTemp.size());
    wxASSERT(m_ScoresCalibTemp.size()>=1);
    wxASSERT(m_ParametersTemp.size()>=1);

    if (m_ParametersTemp.size()==1) return;

    // Sort according to the score
    Array1DFloat vIndices = Array1DFloat::LinSpaced(Eigen::Sequential, m_ScoresCalibTemp.size(), 0, m_ScoresCalibTemp.size()-1);
    asTools::SortArrays(&m_ScoresCalibTemp[0],&m_ScoresCalibTemp[m_ScoresCalibTemp.size()-1],&vIndices[0],&vIndices[m_ScoresCalibTemp.size()-1],m_ScoreOrder);

    // Sort the parameters sets as the scores
    std::vector <asParametersOptimizationNelderMead> copyParameters;
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

bool asMethodOptimizerNelderMead::SetBestParameters(asResultsParametersArray &results)
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

    results.Add(m_Parameters[bestscorerow],m_ScoresCalib[bestscorerow], m_ScoreValid);

    return true;
}

bool asMethodOptimizerNelderMead::Manager()
{
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    int nbRuns = 0;
    pConfig->Read("/Calibration/NelderMead/NbRuns", &nbRuns, 20);
    ThreadsManager().CritSectionConfig().Leave();

    // Reset the score of the climatology
    m_ScoreClimatology = 0;

    for (int i=0; i<nbRuns; i++)
    {
        ClearAll();
        if (!ManageOneRun()) return false;
    }

    // Delete preloaded data
    DeletePreloadedData();

    return true;
}

bool asMethodOptimizerNelderMead::ManageOneRun()
{
    // Seeds the random generator
    asTools::InitRandom();

    // Load parameters
    asParametersOptimizationNelderMead params;
    if (!params.LoadFromFile(m_ParamsFilePath)) return false;
    if (m_PredictandStationId>0)
    {
        params.SetPredictandStationId(m_PredictandStationId);
    }
    InitParameters(params);
    m_OriginalParams = params;

    // Create a result object to save the parameters sets
    int stationId = m_OriginalParams.GetPredictandStationId();
    wxString time = asTime::GetStringTime(asTime::NowMJD(asLOCAL), concentrate);
    asResultsParametersArray results_all;
    results_all.Init(wxString::Format(_("station_%d_tested_parameters"), stationId));
    asResultsParametersArray results_slct;
    results_slct.Init(wxString::Format(_("station_%d_selected_parameters"), stationId));
    asResultsParametersArray results_best;
    results_best.Init(wxString::Format(_("station_%d_best_parameters"), stationId));
    asResultsParametersArray results_simplex;
    results_simplex.Init(wxString::Format(_("station_%d_simplex"), stationId));
    results_simplex.CreateFile();
    wxString resultsXmlFilePath = wxFileConfig::Get()->Read("/StandardPaths/CalibrationResultsDir", asConfig::GetDefaultUserWorkingDir());
    resultsXmlFilePath.Append(wxString::Format("/Calibration/%s_station_%d_best_parameters.xml", time.c_str(), stationId));

    // Reset some data members
    m_Iterator = 0;
    m_OptimizerStage = asINITIALIZATION;
    m_SkipNext = false;
    m_IsOver = false;

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
    if(!LoadPredictandDB(m_PredictandDBFilePath)) return false;
    asLogMessage(_("Predictand DB loaded."));

    // Watch
    wxStopWatch sw;

    // Create results objects
    asResultsAnalogsDates anaDates;
    asResultsAnalogsDates anaDatesPrevious;
    asResultsAnalogsValues anaValues;
    asResultsAnalogsForecastScores anaScores;
    asResultsAnalogsForecastScoreFinal anaScoreFinal;

    // Optimizer
    while(!IsOver())
    {
        // Get a parameters set
        params = GetNextParameters();

        #ifndef UNIT_TESTING
            if (g_Responsive) wxGetApp().Yield();
        #endif
        if (m_Cancel) return false;

        if(!SkipNext() && !IsOver())
        {
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
            if(((m_OptimizerStage==asINITIALIZATION) | (m_OptimizerStage==asREASSESSMENT) | (m_OptimizerStage==asFINAL_REASSESSMENT)) && m_Iterator<m_ParamsNb)
            {
                m_ScoresCalib[m_Iterator] = anaScoreFinal.GetForecastScore();
            }
            else
            {
                m_ScoresCalibTemp.push_back(anaScoreFinal.GetForecastScore());
            }
            wxASSERT(m_ScoresCalib.size()<=(unsigned)m_ParamsNb);

            // Save all tested parameters in a text file
            results_all.Add(params,anaScoreFinal.GetForecastScore());

            // Clear actual simplex result and recreate
            results_simplex.Clear();
            for (unsigned int i=0; i<m_Parameters.size(); i++)
            {
                results_simplex.Add(m_Parameters[i],m_ScoresCalib[i]);
            }
            if(!results_simplex.AppendContent()) return false;

            // Increment iterator
            IncrementIterator();
        }
    }

    // Display processing time
    asLogMessageImportant(wxString::Format(_("The whole processing took %ldms to execute"), sw.Time()));
    asLogState(_("Optimization over."));

    // Validate
    Validate(&m_Parameters[0]);

    // Print parameters in a text file
    SetSelectedParameters(results_slct);
    if(!results_slct.Print()) return false;
    SetBestParameters(results_best);
    if(!results_best.Print()) return false;
    if(!results_simplex.Print()) return false;
    if(!results_all.Print()) return false;

    // Generate xml file with the best parameters set
    if(!m_Parameters[0].GenerateSimpleParametersFile(resultsXmlFilePath)) return false;

    return true;
}

void asMethodOptimizerNelderMead::InitParameters(asParametersOptimizationNelderMead &params)
{
    // Get a first parameters set to get the number of unknown variables
    params.InitRandomValues();
    m_ParamsNb = params.GetVariablesNb()+1;
    asLogMessage(wxString::Format(_("The simplex is made of %d points."), m_ParamsNb));

    // Create the corresponding number of parameters
    m_ScoresCalib.resize(m_ParamsNb);
    for (int i_var=0; i_var<m_ParamsNb; i_var++)
    {
        asParametersOptimizationNelderMead paramsCopy;
        paramsCopy = params;
        paramsCopy.InitRandomValues();
        m_Parameters.push_back(paramsCopy);
        m_ScoresCalib[i_var] = NaNFloat;
    }
    m_ScoreValid = NaNFloat;
}

asParametersOptimizationNelderMead asMethodOptimizerNelderMead::GetNextParameters()
{
    asParametersOptimizationNelderMead params;
    m_SkipNext = false;

    if(((m_OptimizerStage==asINITIALIZATION) | (m_OptimizerStage==asREASSESSMENT) | (m_OptimizerStage==asFINAL_REASSESSMENT)) && m_Iterator<m_ParamsNb)
    {
        params = m_Parameters[m_Iterator];
    }
    else if(((m_OptimizerStage==asINITIALIZATION) | (m_OptimizerStage==asREASSESSMENT)) && m_Iterator==m_ParamsNb)
    {
        m_OptimizerStage=asSORTING_AND_REFLECTION;
        if(!Optimize(params)) asLogError(_("The parameters could not be optimized"));
    }
    else if((m_OptimizerStage==asFINAL_REASSESSMENT) && m_Iterator==m_ParamsNb)
    {
        SortScoresAndParameters();
        m_IsOver = true;
        m_SkipNext = true;
    }
    else
    {
        if(!Optimize(params)) asLogError(_("The parameters could not be optimized"));
    }

    return params;
}

bool asMethodOptimizerNelderMead::Optimize(asParametersOptimizationNelderMead &params)
{
    if (m_OptimizerStage==asSORTING_AND_REFLECTION)
    {
        asLogState(_("Optimization: sorting and reflection."));

        // Sort according to the score
        SortScoresAndParameters();
        ClearTemp();

        // Process the center of gravity of all points but the last one
        asParametersOptimizationNelderMead centerParams;
        centerParams = m_Parameters[m_ParamsNb-1];
        centerParams.SetMeans(m_Parameters, m_ParamsNb-1); // on m_ParamsNb-1 elements (not on the worst point)

        // Process the reflection of the last element by means of the center
        asParametersOptimizationNelderMead reflectedParams;
        reflectedParams = m_Parameters[m_ParamsNb-1];
        reflectedParams.GeometricTransform(centerParams, m_NelderMeadRho);

        // Set the reflected parameters set in the temporary array
        m_ParametersTemp.push_back(centerParams);
        m_ParametersTemp.push_back(reflectedParams);

        // Flag the state to continue to the correct stage
        m_OptimizerStage = asEVALUATE_REFLECTION;
        m_SkipNext = false;

        // Return the reflected parameters
        params = reflectedParams;
        return true;
    }
    else if (m_OptimizerStage==asEVALUATE_REFLECTION)
    {
        asLogState(_("Optimization: evaluation of the reflection, then do reflection or expansion or contraction."));

        // Compare the score of the reflected value with others
        float scoreReflection = m_ScoresCalibTemp[m_ScoresCalibTemp.size()-1];

        bool check1=false, check2=false, check3=false, check4=false;
        switch (m_ScoreOrder)
        {
            case Asc:
                check1 = (scoreReflection>=m_ScoresCalib[0] && scoreReflection<m_ScoresCalib[m_ParamsNb-2]); // If the reflected point is better than the second worst, but not better than the best
                check2 = (scoreReflection<m_ScoresCalib[0]); // If the reflected point is the best point so far
                check3 = (scoreReflection>=m_ScoresCalib[m_ParamsNb-2] && scoreReflection<m_ScoresCalib[m_ParamsNb-1]); // If the reflected point is not better than second worst but better than the worst
                check4 = (scoreReflection>=m_ScoresCalib[m_ParamsNb-1]); // If the reflected point is not better than the worst
                break;
            case Desc:
                check1 = (scoreReflection<=m_ScoresCalib[0] && scoreReflection>m_ScoresCalib[m_ParamsNb-2]); // If the reflected point is better than the second worst, but not better than the best
                check2 = (scoreReflection>m_ScoresCalib[0]); // If the reflected point is the best point so far
                check3 = (scoreReflection<=m_ScoresCalib[m_ParamsNb-2] && scoreReflection>m_ScoresCalib[m_ParamsNb-1]); // If the reflected point is not better than second worst but better than the worst
                check4 = (scoreReflection<=m_ScoresCalib[m_ParamsNb-1]); // If the reflected point is not better than the worst
                break;
            case NoOrder:
                asLogError(_("The score order was not correctly defined."));
                return false;
            default:
                asLogError(_("The score order was not correctly defined."));
                return false;
        }

        // Check consistency: there must be only one true
        int check_increment = 0;
        if (check1) check_increment++;
        if (check2) check_increment++;
        if (check3) check_increment++;
        if (check4) check_increment++;
        if (check_increment!=1)
        {
            asLogError("The results after reflexion are not coherents.");
            return false;
        }

        if (check1) // If the reflected point is better than the second worst, but not better than the best
        {
            m_Parameters[m_ParamsNb-1] = m_ParametersTemp[m_ParametersTemp.size()-1];
            m_ScoresCalib[m_ParamsNb-1] = scoreReflection;
            m_OptimizerStage = asSORTING_AND_REFLECTION;
            ClearTemp();
            m_SkipNext = true;
            return true;
        }
        else if (check2) // If the reflected point is the best point so far
        {
            // Process the expansion of the last element by means of the center
            asParametersOptimizationNelderMead expandedParams;
            expandedParams = m_Parameters[m_ParamsNb-1];
            expandedParams.GeometricTransform(m_ParametersTemp[0], m_NelderMeadRho*m_NelderMeadChi); // m_ParametersTemp[0] is the center of gravity

            // Set the reflected parameters set in the temporary array
            m_ParametersTemp.push_back(expandedParams);

            // Flag the state to continue to the correct stage
            m_OptimizerStage = asCOMPARE_EXPANSION_REFLECTION;

            // Return the reflected parameters
            params = expandedParams;
            return true;
        }
        else if (check3) // If the reflected point is not better than second worst but better than the worst
        {
            // Process the external contraction of the last element by means of the center
            asParametersOptimizationNelderMead contractedParams;
            contractedParams = m_Parameters[m_ParamsNb-1];
            contractedParams.GeometricTransform(m_ParametersTemp[0], m_NelderMeadRho*m_NelderMeadGamma);

            // Set the contracted parameters set in the temporary array
            m_ParametersTemp.push_back(contractedParams);

            // Flag the state to continue to the correct stage
            m_OptimizerStage = asEVALUATE_EXTERNAL_CONTRACTION;

            // Return the reflected parameters
            params = contractedParams;
            return true;
        }
        else if (check4) // If the reflected point is not better than the worst
        {
            // Process the external contraction of the last element by means of the center
            asParametersOptimizationNelderMead contractedParams;
            contractedParams = m_Parameters[m_ParamsNb-1];
            contractedParams.GeometricTransform(m_ParametersTemp[0], -m_NelderMeadGamma);

            // Set the contracted parameters set in the temporary array
            m_ParametersTemp.push_back(contractedParams);

            // Flag the state to continue to the correct stage
            m_OptimizerStage = asEVALUATE_INTERNAL_CONTRACTION;

            // Return the reflected parameters
            params = contractedParams;
            return true;
        }
        else
        {
            asThrowException(wxString::Format(_("Error in the Nelder-Mead optimization: actual score = %g, best score = %g, 2nd worst score = %g."), scoreReflection, m_ScoresCalib[0], m_ScoresCalib[m_ParamsNb-2]));
        }
    }
    else if (m_OptimizerStage==asCOMPARE_EXPANSION_REFLECTION)
    {
        asLogState(_("Optimization: compare expansion and reflection."));

        // Compare the score of the expanded value with others
        float scoreReflection = m_ScoresCalibTemp[m_ScoresCalibTemp.size()-2];
        float scoreExpansion = m_ScoresCalibTemp[m_ScoresCalibTemp.size()-1];

        bool check=false;
        switch (m_ScoreOrder)
        {
            case Asc:
                check = (scoreExpansion<scoreReflection); // If the expanded point is better than the reflected point
                break;
            case Desc:
                check = (scoreExpansion>scoreReflection); // If the expanded point is better than the reflected point
                break;
            case NoOrder:
                asLogError(_("The score order was not correctly defined."));
                return false;
            default:
                asLogError(_("The score order was not correctly defined."));
                return false;
        }

        if(check) // If the expanded point is better than the reflected point
        {
            m_Parameters[m_ParamsNb-1] = m_ParametersTemp[m_ParametersTemp.size()-1];
            m_ScoresCalib[m_ParamsNb-1] = scoreExpansion;
            m_OptimizerStage = asSORTING_AND_REFLECTION;
            ClearTemp();
            m_SkipNext = true;
            return true;
        }
        else
        {
            m_Parameters[m_ParamsNb-1] = m_ParametersTemp[m_ParametersTemp.size()-2];
            m_ScoresCalib[m_ParamsNb-1] = scoreReflection;
            m_OptimizerStage = asSORTING_AND_REFLECTION;
            ClearTemp();
            m_SkipNext = true;
            return true;
        }
    }
    else if (m_OptimizerStage==asEVALUATE_EXTERNAL_CONTRACTION)
    {
        asLogState(_("Optimization: evaluate external contraction"));

        // Compare the score of the contracted value with others
        float scoreContraction = m_ScoresCalibTemp[m_ScoresCalibTemp.size()-1];
        float scoreReflection = m_ScoresCalibTemp[m_ScoresCalibTemp.size()-2];

        bool check=false;
        switch (m_ScoreOrder)
        {
            case Asc:
                check = (scoreContraction<=scoreReflection);
                break;
            case Desc:
                check = (scoreContraction>=scoreReflection);
                break;
            case NoOrder:
                asLogError(_("The score order was not correctly defined."));
                return false;
            default:
                asLogError(_("The score order was not correctly defined."));
                return false;
        }

        if(check)
        {
            m_Parameters[m_ParamsNb-1] = m_ParametersTemp[m_ParametersTemp.size()-1];
            m_ScoresCalib[m_ParamsNb-1] = scoreContraction;
            m_OptimizerStage = asSORTING_AND_REFLECTION;
            ClearTemp();
            m_SkipNext = true;
            return true;
        }
        else
        {
            m_OptimizerStage = asPROCESS_REDUCTION;
            ClearTemp();
            m_SkipNext = true;
            return true;
        }
    }
    else if (m_OptimizerStage==asEVALUATE_INTERNAL_CONTRACTION)
    {
        asLogState(_("Optimization: evaluate internal contraction"));

        // Compare the score of the contracted value with others
        float scoreContraction = m_ScoresCalibTemp[m_ScoresCalibTemp.size()-1];

        bool check=false;
        switch (m_ScoreOrder)
        {
            case Asc:
                check = (scoreContraction<m_ScoresCalib[m_ParamsNb-1]);
                break;
            case Desc:
                check = (scoreContraction>m_ScoresCalib[m_ParamsNb-1]);
                break;
            case NoOrder:
                asLogError(_("The score order was not correctly defined."));
                return false;
            default:
                asLogError(_("The score order was not correctly defined."));
                return false;
        }

        if(check)
        {
            m_Parameters[m_ParamsNb-1] = m_ParametersTemp[m_ParametersTemp.size()-1];
            m_ScoresCalib[m_ParamsNb-1] = scoreContraction;
            m_OptimizerStage = asSORTING_AND_REFLECTION;
            ClearTemp();
            m_SkipNext = true;
            return true;
        }
        else
        {
            m_OptimizerStage = asPROCESS_REDUCTION;
            ClearTemp();
            m_SkipNext = true;
            return true;
        }
    }
    else if (m_OptimizerStage==asPROCESS_REDUCTION)
    {
        asLogState(_("Optimization: proceed to the reduction."));

        // Make a copy of the parameters vector to identify the end of the method
        std::vector <asParametersOptimizationNelderMead> ParametersBefore;
        for (int i=0; i<m_ParamsNb; i++)
        {
            ParametersBefore.push_back(m_Parameters[i]);
        }

        // Process the reduction of every element but the first
        for (int i_el=1; i_el<m_ParamsNb; i_el++)
        {
            m_Parameters[i_el].Reduction(m_Parameters[0], m_NelderMeadSigma);
        }

        asLogMessage(_("Check if we should end."));

        // Check if we should end
        bool stopiterations = true;
        for (int i_el=0; i_el<m_ParamsNb; i_el++)
        {
            if (!m_Parameters[i_el].IsCloseTo(ParametersBefore[i_el]))
            {
                stopiterations = false;
            }
        }

        if(stopiterations)
        {
            asLogMessage(_("Optimization process over. Proceed to final assessment."));
            m_OptimizerStage = asFINAL_REASSESSMENT;
            ClearScores();
            ClearTemp();
            m_SkipNext = true;
            m_Iterator = 0;
            return true;
        }

        asLogMessage(_("Optimization not over."));

        m_OptimizerStage = asREASSESSMENT; // Needs to reassess the simplex values
        ClearScores();
        ClearTemp();
        m_SkipNext = true;
        m_Iterator = 0;
        return true;
    }

    return false;
}
