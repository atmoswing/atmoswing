#include "asMethodOptimizer.h"

asMethodOptimizer::asMethodOptimizer()
        : asMethodCalibrator()
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

bool asMethodOptimizer::SaveDetails(asParametersOptimization &params)
{
    asResultsAnalogsDates anaDatesPrevious;
    asResultsAnalogsDates anaDates;
    asResultsAnalogsValues anaValues;
    asResultsAnalogsForecastScores anaScores;
    asResultsAnalogsForecastScoreFinal anaScoreFinal;

    // Process every step one after the other
    int stepsNb = params.GetStepsNb();
    for (int iStep = 0; iStep < stepsNb; iStep++) {
        bool containsNaNs = false;
        if (iStep == 0) {
            if (!GetAnalogsDates(anaDates, params, iStep, containsNaNs))
                return false;
        } else {
            anaDatesPrevious = anaDates;
            if (!GetAnalogsSubDates(anaDates, params, anaDatesPrevious, iStep, containsNaNs))
                return false;
        }
        if (containsNaNs) {
            wxLogError(_("The dates selection contains NaNs"));
            return false;
        }
    }
    if (!GetAnalogsValues(anaValues, params, anaDates, stepsNb - 1))
        return false;
    if (!GetAnalogsForecastScores(anaScores, params, anaValues, stepsNb - 1))
        return false;
    if (!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, stepsNb - 1))
        return false;

    anaDates.SetSubFolder("calibration");
    anaDates.Save();
    anaValues.SetSubFolder("calibration");
    anaValues.Save();
    anaScores.SetSubFolder("calibration");
    anaScores.Save();

    return true;
}

bool asMethodOptimizer::Validate(asParametersOptimization &params)
{
    if (!params.HasValidationPeriod()) {
        wxLogWarning("The parameters have no validation period !");
        return false;
    }

    m_validationMode = true;

    asResultsAnalogsDates anaDatesPrevious;
    asResultsAnalogsDates anaDates;
    asResultsAnalogsValues anaValues;
    asResultsAnalogsForecastScores anaScores;
    asResultsAnalogsForecastScoreFinal anaScoreFinal;

    // Process every step one after the other
    int stepsNb = params.GetStepsNb();
    for (int iStep = 0; iStep < stepsNb; iStep++) {
        bool containsNaNs = false;
        if (iStep == 0) {
            if (!GetAnalogsDates(anaDates, params, iStep, containsNaNs))
                return false;
        } else {
            anaDatesPrevious = anaDates;
            if (!GetAnalogsSubDates(anaDates, params, anaDatesPrevious, iStep, containsNaNs))
                return false;
        }
        if (containsNaNs) {
            wxLogError(_("The dates selection contains NaNs"));
            return false;
        }
    }
    if (!GetAnalogsValues(anaValues, params, anaDates, stepsNb - 1))
        return false;
    if (!GetAnalogsForecastScores(anaScores, params, anaValues, stepsNb - 1))
        return false;
    if (!GetAnalogsForecastScoreFinal(anaScoreFinal, params, anaScores, stepsNb - 1))
        return false;

    anaDates.SetSubFolder("validation");
    anaDates.Save();
    anaValues.SetSubFolder("validation");
    anaValues.Save();
    anaScores.SetSubFolder("validation");
    anaScores.Save();

    m_scoreValid = anaScoreFinal.GetForecastScore();

    m_validationMode = false;

    return true;
}
