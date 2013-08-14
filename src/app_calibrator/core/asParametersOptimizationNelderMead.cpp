#include "asParametersOptimizationNelderMead.h"

#include <asFileParametersOptimization.h>


asParametersOptimizationNelderMead::asParametersOptimizationNelderMead()
:
asParametersOptimization()
{

}

asParametersOptimizationNelderMead::~asParametersOptimizationNelderMead()
{
    //dtor
}

void asParametersOptimizationNelderMead::SetMeans(std::vector <asParametersOptimizationNelderMead> &vParameters, int elementsNb)
{
    // Set zeros
    if(!m_TimeArrayAnalogsIntervalDaysLocks) m_TimeArrayAnalogsIntervalDays = 0;

    for (int i_step=0; i_step<GetStepsNb(); i_step++)
    {
        if(!m_StepsLocks[i_step].AnalogsNumber) SetAnalogsNumber(i_step, 0);

        for (int i_ptor=0; i_ptor<GetPredictorsNb(i_step); i_ptor++)
        {
            if(!m_StepsLocks[i_step].Predictors[i_ptor].Umin) SetPredictorUmin(i_step, i_ptor, 0);
            if(!m_StepsLocks[i_step].Predictors[i_ptor].Uptsnb) SetPredictorUptsnb(i_step, i_ptor, 0);
            if(!m_StepsLocks[i_step].Predictors[i_ptor].Vmin) SetPredictorVmin(i_step, i_ptor, 0);
            if(!m_StepsLocks[i_step].Predictors[i_ptor].Vptsnb) SetPredictorVptsnb(i_step, i_ptor, 0);
            if(!m_StepsLocks[i_step].Predictors[i_ptor].DTimeHours) SetPredictorDTimeHours(i_step, i_ptor, 0);
            if(!m_StepsLocks[i_step].Predictors[i_ptor].Weight) SetPredictorWeight(i_step, i_ptor, 0);
        }
    }
    if(!m_ForecastScoreLocks.AnalogsNumber) SetForecastScoreAnalogsNumber(0);

    // Proceed the sum of all parameters
    for (int i_set=0; i_set<elementsNb; i_set++)
    {
        if(!m_TimeArrayAnalogsIntervalDaysLocks) m_TimeArrayAnalogsIntervalDays += vParameters[i_set].GetTimeArrayAnalogsIntervalDays();

        for (int i_step=0; i_step<GetStepsNb(); i_step++)
        {
            if(!m_StepsLocks[i_step].AnalogsNumber) SetAnalogsNumber(i_step, GetAnalogsNumber(i_step) + vParameters[i_set].GetAnalogsNumber(i_step));

            for (int i_ptor=0; i_ptor<GetPredictorsNb(i_step); i_ptor++)
            {
                if(!m_StepsLocks[i_step].Predictors[i_ptor].Umin) SetPredictorUmin(i_step, i_ptor, GetPredictorUmin(i_step, i_ptor) + vParameters[i_set].GetPredictorUmin(i_step, i_ptor));
                if(!m_StepsLocks[i_step].Predictors[i_ptor].Uptsnb) SetPredictorUptsnb(i_step, i_ptor, GetPredictorUptsnb(i_step, i_ptor) + vParameters[i_set].GetPredictorUptsnb(i_step, i_ptor));
                if(!m_StepsLocks[i_step].Predictors[i_ptor].Vmin) SetPredictorVmin(i_step, i_ptor, GetPredictorVmin(i_step, i_ptor) + vParameters[i_set].GetPredictorVmin(i_step, i_ptor));
                if(!m_StepsLocks[i_step].Predictors[i_ptor].Vptsnb) SetPredictorVptsnb(i_step, i_ptor, GetPredictorVptsnb(i_step, i_ptor) + vParameters[i_set].GetPredictorVptsnb(i_step, i_ptor));
                if(!m_StepsLocks[i_step].Predictors[i_ptor].DTimeHours) SetPredictorDTimeHours(i_step, i_ptor, GetPredictorDTimeHours(i_step, i_ptor) + vParameters[i_set].GetPredictorDTimeHours(i_step, i_ptor));
                if(!m_StepsLocks[i_step].Predictors[i_ptor].Weight) SetPredictorWeight(i_step, i_ptor, GetPredictorWeight(i_step, i_ptor) + vParameters[i_set].GetPredictorWeight(i_step, i_ptor));
            }
        }
        if(!m_ForecastScoreLocks.AnalogsNumber) SetForecastScoreAnalogsNumber(GetForecastScoreAnalogsNumber() + vParameters[i_set].GetForecastScoreAnalogsNumber());
    }

    // Divide by the number of elements to get the mean
    if(!m_TimeArrayAnalogsIntervalDaysLocks) m_TimeArrayAnalogsIntervalDays /= (double)elementsNb;

    for (int i_step=0; i_step<GetStepsNb(); i_step++)
    {
        if(!m_StepsLocks[i_step].AnalogsNumber) SetAnalogsNumber(i_step, GetAnalogsNumber(i_step) / (double)elementsNb);

        for (int i_ptor=0; i_ptor<GetPredictorsNb(i_step); i_ptor++)
        {
            if(!m_StepsLocks[i_step].Predictors[i_ptor].Umin) SetPredictorUmin(i_step, i_ptor, GetPredictorUmin(i_step, i_ptor) / (double)elementsNb);
            if(!m_StepsLocks[i_step].Predictors[i_ptor].Uptsnb) SetPredictorUptsnb(i_step, i_ptor, (int)((double)GetPredictorUptsnb(i_step, i_ptor) / (double)elementsNb));
            if(!m_StepsLocks[i_step].Predictors[i_ptor].Vmin) SetPredictorVmin(i_step, i_ptor, GetPredictorVmin(i_step, i_ptor) / (double)elementsNb);
            if(!m_StepsLocks[i_step].Predictors[i_ptor].Vptsnb) SetPredictorVptsnb(i_step, i_ptor, (int)((double)GetPredictorVptsnb(i_step, i_ptor) / (double)elementsNb));
            if(!m_StepsLocks[i_step].Predictors[i_ptor].DTimeHours) SetPredictorDTimeHours(i_step, i_ptor, GetPredictorDTimeHours(i_step, i_ptor) / (double)elementsNb);
            if(!m_StepsLocks[i_step].Predictors[i_ptor].Weight) SetPredictorWeight(i_step, i_ptor, GetPredictorWeight(i_step, i_ptor) / (double)elementsNb);
        }
    }
    if(!m_ForecastScoreLocks.AnalogsNumber) SetForecastScoreAnalogsNumber(GetForecastScoreAnalogsNumber() / (double)elementsNb);

    // No fix as it is not a parameters set that will be evaluated.
}

void asParametersOptimizationNelderMead::GeometricTransform(asParametersOptimizationNelderMead &refParams, float coefficient)
{
    // Set reflection
    if(!m_TimeArrayAnalogsIntervalDaysLocks) m_TimeArrayAnalogsIntervalDays = refParams.GetTimeArrayAnalogsIntervalDays()+coefficient*(refParams.GetTimeArrayAnalogsIntervalDays()-m_TimeArrayAnalogsIntervalDays);

    for (int i_step=0; i_step<GetStepsNb(); i_step++)
    {
        if(!m_StepsLocks[i_step].AnalogsNumber) SetAnalogsNumber(i_step, refParams.GetAnalogsNumber(i_step)+coefficient*(refParams.GetAnalogsNumber(i_step)-GetAnalogsNumber(i_step)));

        for (int i_ptor=0; i_ptor<GetPredictorsNb(i_step); i_ptor++)
        {
            if(!m_StepsLocks[i_step].Predictors[i_ptor].Umin) SetPredictorUmin(i_step, i_ptor, refParams.GetPredictorUmin(i_step, i_ptor)+coefficient*(refParams.GetPredictorUmin(i_step, i_ptor)-GetPredictorUmin(i_step,i_ptor)));
            if(!m_StepsLocks[i_step].Predictors[i_ptor].Uptsnb) SetPredictorUptsnb(i_step, i_ptor, refParams.GetPredictorUptsnb(i_step, i_ptor)+coefficient*(refParams.GetPredictorUptsnb(i_step, i_ptor)-GetPredictorUptsnb(i_step,i_ptor)));
            if(!m_StepsLocks[i_step].Predictors[i_ptor].Vmin) SetPredictorVmin(i_step, i_ptor, refParams.GetPredictorVmin(i_step, i_ptor)+coefficient*(refParams.GetPredictorVmin(i_step, i_ptor)-GetPredictorVmin(i_step,i_ptor)));
            if(!m_StepsLocks[i_step].Predictors[i_ptor].Vptsnb) SetPredictorVptsnb(i_step, i_ptor, refParams.GetPredictorVptsnb(i_step, i_ptor)+coefficient*(refParams.GetPredictorVptsnb(i_step, i_ptor)-GetPredictorVptsnb(i_step,i_ptor)));
            if(!m_StepsLocks[i_step].Predictors[i_ptor].DTimeHours) SetPredictorDTimeHours(i_step, i_ptor, refParams.GetPredictorDTimeHours(i_step, i_ptor)+coefficient*(refParams.GetPredictorDTimeHours(i_step, i_ptor)-GetPredictorDTimeHours(i_step,i_ptor)));
            if(!m_StepsLocks[i_step].Predictors[i_ptor].Weight) SetPredictorWeight(i_step, i_ptor, refParams.GetPredictorWeight(i_step, i_ptor)+coefficient*(refParams.GetPredictorWeight(i_step, i_ptor)-GetPredictorWeight(i_step,i_ptor)));
        }
    }
    if(!m_ForecastScoreLocks.AnalogsNumber) SetForecastScoreAnalogsNumber(refParams.GetForecastScoreAnalogsNumber()+coefficient*(refParams.GetForecastScoreAnalogsNumber()-GetForecastScoreAnalogsNumber()));

    FixTimeShift();
    FixWeights();
    FixCoordinates();
    CheckRange();
    FixAnalogsNb();
}

void asParametersOptimizationNelderMead::Reduction(asParametersOptimizationNelderMead &refParams, float sigma)
{
    // Set reduction
    if(!m_TimeArrayAnalogsIntervalDaysLocks) m_TimeArrayAnalogsIntervalDays = refParams.GetTimeArrayAnalogsIntervalDays()+sigma*(m_TimeArrayAnalogsIntervalDays-refParams.GetTimeArrayAnalogsIntervalDays());

    for (int i_step=0; i_step<GetStepsNb(); i_step++)
    {
        if(!m_StepsLocks[i_step].AnalogsNumber) SetAnalogsNumber(i_step, refParams.GetAnalogsNumber(i_step)+sigma*(GetAnalogsNumber(i_step)-refParams.GetAnalogsNumber(i_step)));

        for (int i_ptor=0; i_ptor<GetPredictorsNb(i_step); i_ptor++)
        {
            if(!m_StepsLocks[i_step].Predictors[i_ptor].Umin) SetPredictorUmin(i_step, i_ptor, refParams.GetPredictorUmin(i_step, i_ptor)+sigma*(GetPredictorUmin(i_step, i_ptor)-refParams.GetPredictorUmin(i_step, i_ptor)));
            if(!m_StepsLocks[i_step].Predictors[i_ptor].Uptsnb) SetPredictorUptsnb(i_step, i_ptor, refParams.GetPredictorUptsnb(i_step, i_ptor)+sigma*(GetPredictorUptsnb(i_step, i_ptor)-refParams.GetPredictorUptsnb(i_step, i_ptor)));
            if(!m_StepsLocks[i_step].Predictors[i_ptor].Vmin) SetPredictorVmin(i_step, i_ptor, refParams.GetPredictorVmin(i_step, i_ptor)+sigma*(GetPredictorVmin(i_step, i_ptor)-refParams.GetPredictorVmin(i_step, i_ptor)));
            if(!m_StepsLocks[i_step].Predictors[i_ptor].Vptsnb) SetPredictorVptsnb(i_step, i_ptor, refParams.GetPredictorVptsnb(i_step, i_ptor)+sigma*(GetPredictorVptsnb(i_step, i_ptor)-refParams.GetPredictorVptsnb(i_step, i_ptor)));
            if(!m_StepsLocks[i_step].Predictors[i_ptor].DTimeHours) SetPredictorDTimeHours(i_step, i_ptor, refParams.GetPredictorDTimeHours(i_step, i_ptor)+sigma*(GetPredictorDTimeHours(i_step, i_ptor)-refParams.GetPredictorDTimeHours(i_step, i_ptor)));
            if(!m_StepsLocks[i_step].Predictors[i_ptor].Weight) SetPredictorWeight(i_step, i_ptor, refParams.GetPredictorWeight(i_step, i_ptor)+sigma*(GetPredictorWeight(i_step, i_ptor)-refParams.GetPredictorWeight(i_step, i_ptor)));
        }
    }
    if(!m_ForecastScoreLocks.AnalogsNumber) SetForecastScoreAnalogsNumber(refParams.GetForecastScoreAnalogsNumber()+sigma*(GetForecastScoreAnalogsNumber()-refParams.GetForecastScoreAnalogsNumber()));

    FixTimeShift();
    FixWeights();
    FixCoordinates();
    CheckRange();
    FixAnalogsNb();
}