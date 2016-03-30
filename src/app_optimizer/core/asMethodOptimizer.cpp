#include "asMethodOptimizer.h"

#include <asResultsParametersArray.h>
#include <asResultsAnalogsDates.h>
#include <asResultsAnalogsValues.h>
#include <asResultsAnalogsForecastScores.h>
#include <asResultsAnalogsForecastScoreFinal.h>
#include <asForecastScore.h>
#include <asDataPredictand.h>

asMethodOptimizer::asMethodOptimizer()
:
asMethodCalibrator()
{
    m_iterator = 0;
    m_optimizerStage = asINITIALIZATION;
    m_skipNext = false;
    m_isOver = false;
    m_paramsNb = 0;

    // Seeds the random generator
    asTools::InitRandom();
}

asMethodOptimizer::~asMethodOptimizer()
{
    //dtor
}

bool asMethodOptimizer::Validate(asParametersOptimization* params)
{
    if (!params->HasValidationPeriod())
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
    int stepsNb = params->GetStepsNb();
    for (int i_step=0; i_step<stepsNb; i_step++)
    {
        bool containsNaNs = false;
        if (i_step==0)
        {
            if(!GetAnalogsDates(anaDates, *params, i_step, containsNaNs)) return false;
        }
        else
        {
            anaDatesPrevious = anaDates;
            if(!GetAnalogsSubDates(anaDates, *params, anaDatesPrevious, i_step, containsNaNs)) return false;
        }
        if (containsNaNs)
        {
            asLogError(_("The dates selection contains NaNs"));
            return false;
        }
    }
    if(!GetAnalogsValues(anaValues, *params, anaDates, stepsNb-1)) return false;
    if(!GetAnalogsForecastScores(anaScores, *params, anaValues, stepsNb-1)) return false;
    if(!GetAnalogsForecastScoreFinal(anaScoreFinal, *params, anaScores, stepsNb-1)) return false;

    m_scoreValid = anaScoreFinal.GetForecastScore();

    m_validationMode = false;

    return true;
}
