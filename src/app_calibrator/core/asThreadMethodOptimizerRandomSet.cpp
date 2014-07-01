#include "asThreadMethodOptimizerRandomSet.h"

#include "asMethodCalibratorSingle.h"

asThreadMethodOptimizerRandomSet::asThreadMethodOptimizerRandomSet(const asMethodOptimizerRandomSet* optimizer, const asParametersOptimization &params, float *finalScoreCalib, VectorFloat *scoreClimatology)
:
asThread()
{
    m_Status = Initializing;

    m_Type = asThread::MethodOptimizerRandomSet;

    m_Optimizer = *optimizer; // copy
    m_Params = params;
    m_ScoreClimatology = scoreClimatology;
    m_FinalScoreCalib = finalScoreCalib;

    m_Status = Waiting;
}

asThreadMethodOptimizerRandomSet::~asThreadMethodOptimizerRandomSet()
{
    //dtor
}

wxThread::ExitCode asThreadMethodOptimizerRandomSet::Entry()
{
    m_Status = Working;

    // Create results objects. Needs to be in a critical section because of access to the config pointer.
    asResultsAnalogsDates anaDates;
    asResultsAnalogsDates anaDatesPrevious;
    asResultsAnalogsValues anaValues;
    asResultsAnalogsForecastScores anaScores;
    asResultsAnalogsForecastScoreFinal anaScoreFinal;
    // Set the climatology score value
    if (m_ScoreClimatology->size()!=0)
    {
        asLogMessage(_("Process score of the climatology"));
        m_Optimizer.SetScoreClimatology(*m_ScoreClimatology);
    }

    // Process every step one after the other
    int stepsNb = m_Params.GetStepsNb();
    for (int i_step=0; i_step<stepsNb; i_step++)
    {
        bool containsNaNs = false;
        if (i_step==0)
        {
            if(!m_Optimizer.GetAnalogsDates(anaDates, m_Params, i_step, containsNaNs))
            {
                asLogError(_("Failed processing the analogs dates"));
                return NULL;
            }
            anaDatesPrevious = anaDates;
        }
        else
        {
            if(!m_Optimizer.GetAnalogsSubDates(anaDates, m_Params, anaDatesPrevious, i_step, containsNaNs))
            {
                asLogError(_("Failed processing the analogs sub dates"));
                return NULL;
            }
            anaDatesPrevious = anaDates;
        }
        if (containsNaNs)
        {
            asLogError(_("The dates selection contains NaNs"));
            return NULL;
        }
    }
    if(!m_Optimizer.GetAnalogsValues(anaValues, m_Params, anaDates, stepsNb-1))
    {
        asLogError(_("Failed processing the analogs values"));
        return NULL;
    }
    if(!m_Optimizer.GetAnalogsForecastScores(anaScores, m_Params, anaValues, stepsNb-1))
    {
        asLogError(_("Failed processing the forecast scores"));
        return NULL;
    }
    if(!m_Optimizer.GetAnalogsForecastScoreFinal(anaScoreFinal, m_Params, anaScores, stepsNb-1))
    {
        asLogError(_("Failed processing the final score"));
        return NULL;
    }
    *m_FinalScoreCalib = anaScoreFinal.GetForecastScore();

    if (m_ScoreClimatology->size()==0)
    {
        *m_ScoreClimatology = m_Optimizer.GetScoreClimatology();
    }

    m_Status = Done;

    return 0;
}
