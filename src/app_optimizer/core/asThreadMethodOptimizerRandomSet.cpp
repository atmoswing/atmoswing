#include "asThreadMethodOptimizerRandomSet.h"

#include "asMethodCalibratorSingle.h"

asThreadMethodOptimizerRandomSet::asThreadMethodOptimizerRandomSet(const asMethodOptimizerRandomSet *optimizer,
                                                                   const asParametersOptimization &params,
                                                                   float *finalScoreCalib,
                                                                   VectorFloat *scoreClimatology)
        : asThread()
{
    m_status = Initializing;

    m_type = asThread::MethodOptimizerRandomSet;

    m_optimizer = *optimizer; // copy
    m_params = params;
    m_scoreClimatology = scoreClimatology;
    m_finalScoreCalib = finalScoreCalib;

    m_status = Waiting;
}

asThreadMethodOptimizerRandomSet::~asThreadMethodOptimizerRandomSet()
{
    //dtor
}

wxThread::ExitCode asThreadMethodOptimizerRandomSet::Entry()
{
    m_status = Working;

    // Create results objects. Needs to be in a critical section because of access to the config pointer.
    asResultsAnalogsDates anaDates;
    asResultsAnalogsDates anaDatesPrevious;
    asResultsAnalogsValues anaValues;
    asResultsAnalogsForecastScores anaScores;
    asResultsAnalogsForecastScoreFinal anaScoreFinal;

    // Set the climatology score value
    if (m_scoreClimatology->size() != 0) {
        asLogMessage(_("Process score of the climatology"));
        m_optimizer.SetScoreClimatology(*m_scoreClimatology);
    }

    // Process every step one after the other
    int stepsNb = m_params.GetStepsNb();
    for (int i_step = 0; i_step < stepsNb; i_step++) {
        bool containsNaNs = false;
        if (i_step == 0) {
            if (!m_optimizer.GetAnalogsDates(anaDates, m_params, i_step, containsNaNs)) {
                asLogError(_("Failed processing the analogs dates"));
                return NULL;
            }
            anaDatesPrevious = anaDates;
        } else {
            if (!m_optimizer.GetAnalogsSubDates(anaDates, m_params, anaDatesPrevious, i_step, containsNaNs)) {
                asLogError(_("Failed processing the analogs sub dates"));
                return NULL;
            }
            anaDatesPrevious = anaDates;
        }
        if (containsNaNs) {
            asLogError(_("The dates selection contains NaNs"));
            return NULL;
        }
    }
    if (!m_optimizer.GetAnalogsValues(anaValues, m_params, anaDates, stepsNb - 1)) {
        asLogError(_("Failed processing the analogs values"));
        return NULL;
    }
    if (!m_optimizer.GetAnalogsForecastScores(anaScores, m_params, anaValues, stepsNb - 1)) {
        asLogError(_("Failed processing the forecast scores"));
        return NULL;
    }
    if (!m_optimizer.GetAnalogsForecastScoreFinal(anaScoreFinal, m_params, anaScores, stepsNb - 1)) {
        asLogError(_("Failed processing the final score"));
        return NULL;
    }
    *m_finalScoreCalib = anaScoreFinal.GetForecastScore();

    if (m_scoreClimatology->size() == 0) {
        *m_scoreClimatology = m_optimizer.GetScoreClimatology();
    }

    m_status = Done;

    return 0;
}
