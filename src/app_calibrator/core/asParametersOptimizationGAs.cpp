#include "asParametersOptimizationGAs.h"

#include <asFileParametersOptimization.h>


asParametersOptimizationGAs::asParametersOptimizationGAs()
:
asParametersOptimization()
{
    m_IndividualSelfAdaptationMutationRate = 0;
    m_IndividualSelfAdaptationMutationRadius = 0;
    m_HasChromosomeSelfAdaptationMutationRate = false;
    m_HasChromosomeSelfAdaptationMutationRadius = false;
    m_AllParametersCount = 0;
}

asParametersOptimizationGAs::~asParametersOptimizationGAs()
{
    //dtor
}

void asParametersOptimizationGAs::BuildChromosomes()
{
    int counter = 0;
    VectorInt indices;

    if(!m_TimeArrayAnalogsIntervalDaysLocks)
    {
        indices.push_back(counter);
    }
    counter++;

    for (int i=0; i<GetStepsNb(); i++)
    {
        if(!m_StepsLocks[i].AnalogsNumber)
        {
            indices.push_back(counter);
        }
        counter++;

        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            if (NeedsPreprocessing(i,j))
            {
                for (int k=0; k<GetPreprocessSize(i,j); k++)
                {
                    if(!m_StepsLocks[i].Predictors[j].PreprocessDataId[k])
                    {
                        indices.push_back(counter);
                    }
                    counter++;

                    if(!m_StepsLocks[i].Predictors[j].PreprocessLevels[k])
                    {
                        indices.push_back(counter);
                    }
                    counter++;

                    if(!m_StepsLocks[i].Predictors[j].PreprocessTimeHours[k])
                    {
                        indices.push_back(counter);
                    }
                    counter++;
                }
            }
            else
            {
                if(!m_StepsLocks[i].Predictors[j].DataId)
                {
                    indices.push_back(counter);
                }
                counter++;

                if(!m_StepsLocks[i].Predictors[j].Level)
                {
                    indices.push_back(counter);
                }
                counter++;

                if(!m_StepsLocks[i].Predictors[j].TimeHours)
                {
                    indices.push_back(counter);
                }
                counter++;
            }

            if(!m_StepsLocks[i].Predictors[j].Umin)
            {
                indices.push_back(counter);
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Uptsnb)
            {
                indices.push_back(counter);
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Vmin)
            {
                indices.push_back(counter);
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Vptsnb)
            {
                indices.push_back(counter);
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Weight)
            {
                indices.push_back(counter);
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Criteria)
            {
                indices.push_back(counter);
            }
            counter++;
        }
    }

    if(!m_ForecastScoreLocks.AnalogsNumber)
    {
        indices.push_back(counter);
    }
    counter++;

    m_ChromosomeIndices = indices;
    m_AllParametersCount = counter;
}

void asParametersOptimizationGAs::InitIndividualSelfAdaptationMutationRate()
{
    m_IndividualSelfAdaptationMutationRate = asTools::Random(0.0,1.0);
}

void asParametersOptimizationGAs::InitIndividualSelfAdaptationMutationRadius()
{
    m_IndividualSelfAdaptationMutationRadius = asTools::Random(0.0,1.0);
}

void asParametersOptimizationGAs::InitChromosomeSelfAdaptationMutationRate()
{
    int length = GetChromosomeLength();
    m_ChromosomeSelfAdaptationMutationRate.resize(length);

    for (int i=0; i<length; i++)
    {
        m_ChromosomeSelfAdaptationMutationRate[i] = asTools::Random(0.0,1.0);
    }

    m_HasChromosomeSelfAdaptationMutationRate = true;
}

void asParametersOptimizationGAs::InitChromosomeSelfAdaptationMutationRadius()
{
    int length = GetChromosomeLength();
    m_ChromosomeSelfAdaptationMutationRadius.resize(length);

    for (int i=0; i<length; i++)
    {
        m_ChromosomeSelfAdaptationMutationRadius[i] = asTools::Random(0.0,1.0);
    }

    m_HasChromosomeSelfAdaptationMutationRadius = true;
}

bool asParametersOptimizationGAs::IsParamLocked(int index)
{
    int counter = 0;
    m_ParametersListOver = false;

    if(!m_TimeArrayAnalogsIntervalDaysLocks)
    {
        if (counter==index) return false;
    }
    else
    {
        if (counter==index) return true;
    }
    counter++;

    for (int i=0; i<GetStepsNb(); i++)
    {

        if(!m_StepsLocks[i].AnalogsNumber)
        {
            if (counter==index) return false;
        }
        else
        {
            if (counter==index) return true;
        }
        counter++;

        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            if (NeedsPreprocessing(i,j))
            {
                for (int k=0; k<GetPreprocessSize(i,j); k++)
                {
                    if(!m_StepsLocks[i].Predictors[j].PreprocessDataId[k])
                    {
                        if (counter==index) return false;
                    }
                    else
                    {
                        if (counter==index) return true;
                    }
                    counter++;

                    if(!m_StepsLocks[i].Predictors[j].PreprocessLevels[k])
                    {
                        if (counter==index) return false;
                    }
                    else
                    {
                        if (counter==index) return true;
                    }
                    counter++;

                    if(!m_StepsLocks[i].Predictors[j].PreprocessTimeHours[k])
                    {
                        if (counter==index) return false;
                    }
                    else
                    {
                        if (counter==index) return true;
                    }
                    counter++;
                }
            }
            else
            {
                if(!m_StepsLocks[i].Predictors[j].DataId)
                {
                    if (counter==index) return false;
                }
                else
                {
                    if (counter==index) return true;
                }
                counter++;

                if(!m_StepsLocks[i].Predictors[j].Level)
                {
                    if (counter==index) return false;
                }
                else
                {
                    if (counter==index) return true;
                }
                counter++;

                if(!m_StepsLocks[i].Predictors[j].TimeHours)
                {
                    if (counter==index) return false;
                }
                else
                {
                    if (counter==index) return true;
                }
                counter++;
            }

            if(!m_StepsLocks[i].Predictors[j].Umin)
            {
                if (counter==index) return false;
            }
            else
            {
                if (counter==index) return true;
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Uptsnb)
            {
                if (counter==index) return false;
            }
            else
            {
                if (counter==index) return true;
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Vmin)
            {
                if (counter==index) return false;
            }
            else
            {
                if (counter==index) return true;
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Vptsnb)
            {
                if (counter==index) return false;
            }
            else
            {
                if (counter==index) return true;
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Weight)
            {
                if (counter==index) return false;
            }
            else
            {
                if (counter==index) return true;
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Criteria)
            {
                if (counter==index) return false;
            }
            else
            {
                if (counter==index) return true;
            }
            counter++;
        }
    }

    if(!m_ForecastScoreLocks.AnalogsNumber)
    {
        if (counter==index) return false;
    }
    else
    {
        if (counter==index) return true;
    }
    counter++;

    wxASSERT_MSG(counter==m_AllParametersCount, wxString::Format("The counter (%d) did not match the number of parameters (%d).", counter, m_AllParametersCount));
    wxASSERT_MSG(counter<=index, "Couldn't access the desired index in the parameters chromosome.");

    m_ParametersListOver = true;

    return true;
}

bool asParametersOptimizationGAs::IsParamList(int index)
{
    int counter = 0;

    // AnalogsIntervalDays
    if (counter==index)
    {
        return false;
    }
    counter++;

    for (int i=0; i<GetStepsNb(); i++)
    {

        // AnalogsNumber
        if (counter==index)
        {
            return false;
        }
        counter++;

        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            if (NeedsPreprocessing(i,j))
            {
                for (int k=0; k<GetPreprocessSize(i,j); k++)
                {
                    // PreprocessDataId
                    if (counter==index)
                    {
                        return true;
                    }
                    counter++;

                    // PreprocessLevel
                    if (counter==index)
                    {
                        return true;
                    }
                    counter++;

                    // PreprocessTimeHours
                    if (counter==index)
                    {
                        return false;
                    }
                    counter++;
                }
            }
            else
            {
                // DataId
                if (counter==index)
                {
                    return true;
                }
                counter++;

                // Level
                if (counter==index)
                {
                    return true;
                }
                counter++;

                // TimeHours
                if (counter==index)
                {
                    return false;
                }
                counter++;
            }

            // Umin
            if (counter==index)
            {
                return false;
            }
            counter++;

            // Uptsnb
            if (counter==index)
            {
                return false;
            }
            counter++;

            // Vmin
            if (counter==index)
            {
                return false;
            }
            counter++;

            // Vptsnb
            if (counter==index)
            {
                return false;
            }
            counter++;

            // Weight
            if (counter==index)
            {
                return false;
            }
            counter++;

            // Criteria
            if (counter==index)
            {
                return true;
            }
            counter++;
        }
    }

    //AnalogsNumber
    if (counter==index)
    {
        return false;
    }
    counter++;

    wxASSERT_MSG(counter==m_AllParametersCount, wxString::Format("The counter (%d) did not match the number of parameters (%d).", counter, m_AllParametersCount));
    wxASSERT_MSG(counter<=index, "Couldn't access the desired index in the parameters chromosome.");

    asThrowException(_("We should never reach that point..."));

}

double asParametersOptimizationGAs::GetParameterValue(int index)
{
    int counter = 0;

    if(!m_TimeArrayAnalogsIntervalDaysLocks)
    {
        if (counter==index)
        {
            return (double)GetTimeArrayAnalogsIntervalDays();
        }
    }
    counter++;

    for (int i=0; i<GetStepsNb(); i++)
    {

        if(!m_StepsLocks[i].AnalogsNumber)
        {
            if (counter==index)
            {
                return (double)GetAnalogsNumber(i);
            }
        }
        counter++;

        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            if (NeedsPreprocessing(i,j))
            {
                for (int k=0; k<GetPreprocessSize(i,j); k++)
                {
                    if(!m_StepsLocks[i].Predictors[j].PreprocessDataId[k])
                    {
                        if (counter==index)
                        {
                            wxString dat = GetPreprocessDataId(i,j,k);
                            VectorString vect = GetPreprocessDataIdVector(i,j,k);
                            int i_dat = -1;
                            for (unsigned int r=0; r<vect.size(); r++)
                            {
                                if (vect[r].IsSameAs(dat, false)) i_dat = r;
                            }
                            wxASSERT(i_dat>=0);

                            return (double)i_dat;
                        }
                    }
                    counter++;

                    if(!m_StepsLocks[i].Predictors[j].PreprocessLevels[k])
                    {
                        if (counter==index)
                        {
                            float dat = GetPreprocessLevel(i,j,k);
                            VectorFloat vect = GetPreprocessLevelVector(i,j,k);
                            int i_dat = -1;
                            for (unsigned int r=0; r<vect.size(); r++)
                            {
                                if (vect[r] == dat) i_dat = r;
                            }
                            wxASSERT(i_dat>=0);

                            return (double)i_dat;
                        }
                    }
                    counter++;

                    if(!m_StepsLocks[i].Predictors[j].PreprocessTimeHours[k])
                    {
                        if (counter==index)
                        {
                            return (double)GetPreprocessTimeHours(i,j,k);
                        }
                    }
                    counter++;
                }
            }
            else
            {
                if(!m_StepsLocks[i].Predictors[j].DataId)
                {
                    if (counter==index)
                    {
                        wxString dat = GetPredictorDataId(i,j);
                        VectorString vect = GetPredictorDataIdVector(i,j);
                        int i_dat = -1;
                        for (unsigned int r=0; r<vect.size(); r++)
                        {
                            if (vect[r].IsSameAs(dat, false)) i_dat = r;
                        }
                        wxASSERT(i_dat>=0);

                        return (double)i_dat;
                    }
                }
                counter++;

                if(!m_StepsLocks[i].Predictors[j].Level)
                {
                    if (counter==index)
                    {
                        float dat = GetPredictorLevel(i,j);
                        VectorFloat vect = GetPredictorLevelVector(i,j);
                        int i_dat = -1;
                        for (unsigned int r=0; r<vect.size(); r++)
                        {
                            if (vect[r] == dat) i_dat = r;
                        }
                        wxASSERT(i_dat>=0);

                        return (double)i_dat;
                    }
                }
                counter++;

                if(!m_StepsLocks[i].Predictors[j].TimeHours)
                {
                    if (counter==index)
                    {
                        return (double)GetPredictorTimeHours(i,j);
                    }
                }
                counter++;
            }

            if(!m_StepsLocks[i].Predictors[j].Umin)
            {
                if (counter==index)
                {
                    return (double)GetPredictorUmin(i,j);
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Uptsnb)
            {
                if (counter==index)
                {
                    return (double)GetPredictorUptsnb(i,j);
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Vmin)
            {
                if (counter==index)
                {
                    return (double)GetPredictorVmin(i,j);
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Vptsnb)
            {
                if (counter==index)
                {
                    return (double)GetPredictorVptsnb(i,j);
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Weight)
            {
                if (counter==index)
                {
                    return (double)GetPredictorWeight(i,j);
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Criteria)
            {
                if (counter==index)
                {
                    wxString dat = GetPredictorCriteria(i,j);
                    VectorString vect = GetPredictorCriteriaVector(i,j);
                    int i_dat = -1;
                    for (unsigned int r=0; r<vect.size(); r++)
                    {
                        if (vect[r].IsSameAs(dat, false)) i_dat = r;
                    }
                    wxASSERT(i_dat>=0);

                    return (double)i_dat;
                }
            }
            counter++;
        }
    }

    if(!m_ForecastScoreLocks.AnalogsNumber)
    {
        if (counter==index)
        {
            return (double)GetForecastScoreAnalogsNumber();
        }
    }
    counter++;

    wxASSERT_MSG(counter==m_AllParametersCount, wxString::Format("The counter (%d) did not match the number of parameters (%d).", counter, m_AllParametersCount));
    wxASSERT_MSG(counter<=index, "Couldn't access the desired index in the parameters chromosome.");

    return NaNDouble;
}

double asParametersOptimizationGAs::GetParameterUpperLimit(int index)
{
    int counter = 0;

    if(!m_TimeArrayAnalogsIntervalDaysLocks)
    {
        if (counter==index)
        {
            return (double)GetTimeArrayAnalogsIntervalDaysUpperLimit();
        }
    }
    counter++;

    for (int i=0; i<GetStepsNb(); i++)
    {

        if(!m_StepsLocks[i].AnalogsNumber)
        {
            if (counter==index)
            {
                return (double)GetAnalogsNumberUpperLimit(i);
            }
        }
        counter++;

        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            if (NeedsPreprocessing(i,j))
            {
                for (int k=0; k<GetPreprocessSize(i,j); k++)
                {
                    if(!m_StepsLocks[i].Predictors[j].PreprocessDataId[k])
                    {
                        if (counter==index)
                        {
                            VectorString vect = GetPreprocessDataIdVector(i,j,k);
                            return (double)(vect.size()-1);
                        }
                    }
                    counter++;

                    if(!m_StepsLocks[i].Predictors[j].PreprocessLevels[k])
                    {
                        if (counter==index)
                        {
                            VectorFloat vect = GetPreprocessLevelVector(i,j,k);
                            return (double)(vect.size()-1);
                        }
                    }
                    counter++;

                    if(!m_StepsLocks[i].Predictors[j].PreprocessTimeHours[k])
                    {
                        if (counter==index)
                        {
                            return (double)GetPreprocessTimeHoursUpperLimit(i,j,k);
                        }
                    }
                    counter++;
                }
            }
            else
            {
                if(!m_StepsLocks[i].Predictors[j].DataId)
                {
                    if (counter==index)
                    {
                        VectorString vect = GetPredictorDataIdVector(i,j);
                        return (double)(vect.size()-1);
                    }
                }
                counter++;

                if(!m_StepsLocks[i].Predictors[j].Level)
                {
                    if (counter==index)
                    {
                        VectorFloat vect = GetPredictorLevelVector(i,j);
                        return (double)(vect.size()-1);
                    }
                }
                counter++;

                if(!m_StepsLocks[i].Predictors[j].TimeHours)
                {
                    if (counter==index)
                    {
                        return (double)GetPredictorTimeHoursUpperLimit(i,j);
                    }
                }
                counter++;
            }

            if(!m_StepsLocks[i].Predictors[j].Umin)
            {
                if (counter==index)
                {
                    return (double)GetPredictorUminUpperLimit(i,j);
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Uptsnb)
            {
                if (counter==index)
                {
                    return (double)GetPredictorUptsnbUpperLimit(i,j);
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Vmin)
            {
                if (counter==index)
                {
                    return (double)GetPredictorVminUpperLimit(i,j);
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Vptsnb)
            {
                if (counter==index)
                {
                    return (double)GetPredictorVptsnbUpperLimit(i,j);
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Weight)
            {
                if (counter==index)
                {
                    return (double)GetPredictorWeightUpperLimit(i,j);
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Criteria)
            {
                if (counter==index)
                {
                    VectorString vect = GetPredictorCriteriaVector(i,j);
                    return (double)(vect.size()-1);
                }
            }
            counter++;
        }
    }

    if(!m_ForecastScoreLocks.AnalogsNumber)
    {
        if (counter==index)
        {
            return (double)GetForecastScoreAnalogsNumberUpperLimit();
        }
    }
    counter++;

    wxASSERT_MSG(counter==m_AllParametersCount, wxString::Format("The counter (%d) did not match the number of parameters (%d).", counter, m_AllParametersCount));
    wxASSERT_MSG(counter<=index, "Couldn't access the desired index in the parameters chromosome.");

    return NaNDouble;
}

double asParametersOptimizationGAs::GetParameterLowerLimit(int index)
{
    int counter = 0;

    if(!m_TimeArrayAnalogsIntervalDaysLocks)
    {
        if (counter==index)
        {
            return (double)GetTimeArrayAnalogsIntervalDaysLowerLimit();
        }
    }
    counter++;

    for (int i=0; i<GetStepsNb(); i++)
    {

        if(!m_StepsLocks[i].AnalogsNumber)
        {
            if (counter==index)
            {
                return (double)GetAnalogsNumberLowerLimit(i);
            }
        }
        counter++;

        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            if (NeedsPreprocessing(i,j))
            {
                for (int k=0; k<GetPreprocessSize(i,j); k++)
                {
                    if(!m_StepsLocks[i].Predictors[j].PreprocessDataId[k])
                    {
                        if (counter==index)
                        {
                            return 0.0;
                        }
                    }
                    counter++;

                    if(!m_StepsLocks[i].Predictors[j].PreprocessLevels[k])
                    {
                        if (counter==index)
                        {
                            return 0.0;
                        }
                    }
                    counter++;

                    if(!m_StepsLocks[i].Predictors[j].PreprocessTimeHours[k])
                    {
                        if (counter==index)
                        {
                            return (double)GetPreprocessTimeHoursLowerLimit(i,j,k);
                        }
                    }
                    counter++;
                }
            }
            else
            {
                if(!m_StepsLocks[i].Predictors[j].DataId)
                {
                    if (counter==index)
                    {
                        return 0.0;
                    }
                }
                counter++;

                if(!m_StepsLocks[i].Predictors[j].Level)
                {
                    if (counter==index)
                    {
                        return 0.0;
                    }
                }
                counter++;

                if(!m_StepsLocks[i].Predictors[j].TimeHours)
                {
                    if (counter==index)
                    {
                        return (double)GetPredictorTimeHoursLowerLimit(i,j);
                    }
                }
                counter++;
            }

            if(!m_StepsLocks[i].Predictors[j].Umin)
            {
                if (counter==index)
                {
                    return (double)GetPredictorUminLowerLimit(i,j);
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Uptsnb)
            {
                if (counter==index)
                {
                    return (double)GetPredictorUptsnbLowerLimit(i,j);
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Vmin)
            {
                if (counter==index)
                {
                    return (double)GetPredictorVminLowerLimit(i,j);
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Vptsnb)
            {
                if (counter==index)
                {
                    return (double)GetPredictorVptsnbLowerLimit(i,j);
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Weight)
            {
                if (counter==index)
                {
                    return (double)GetPredictorWeightLowerLimit(i,j);
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Criteria)
            {
                if (counter==index)
                {
                    return (double)0.0;
                }
            }
            counter++;
        }
    }

    if(!m_ForecastScoreLocks.AnalogsNumber)
    {
        if (counter==index)
        {
            return (double)GetForecastScoreAnalogsNumberLowerLimit();
        }
    }
    counter++;

    wxASSERT_MSG(counter==m_AllParametersCount, wxString::Format("The counter (%d) did not match the number of parameters (%d).", counter, m_AllParametersCount));
    wxASSERT_MSG(counter<=index, "Couldn't access the desired index in the parameters chromosome.");

    return NaNDouble;
}

double asParametersOptimizationGAs::GetParameterIteration(int index)
{
    int counter = 0;

    if(!m_TimeArrayAnalogsIntervalDaysLocks)
    {
        if (counter==index)
        {
            return (double)GetTimeArrayAnalogsIntervalDaysIteration();
        }
    }
    counter++;

    for (int i=0; i<GetStepsNb(); i++)
    {

        if(!m_StepsLocks[i].AnalogsNumber)
        {
            if (counter==index)
            {
                return (double)GetAnalogsNumberIteration(i);
            }
        }
        counter++;

        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            if (NeedsPreprocessing(i,j))
            {
                for (int k=0; k<GetPreprocessSize(i,j); k++)
                {
                    if(!m_StepsLocks[i].Predictors[j].PreprocessDataId[k])
                    {
                        if (counter==index)
                        {
                            return 1.0;
                        }
                    }
                    counter++;

                    if(!m_StepsLocks[i].Predictors[j].PreprocessLevels[k])
                    {
                        if (counter==index)
                        {
                            return 1.0;
                        }
                    }
                    counter++;

                    if(!m_StepsLocks[i].Predictors[j].PreprocessTimeHours[k])
                    {
                        if (counter==index)
                        {
                            return (double)GetPreprocessTimeHoursIteration(i,j,k);
                        }
                    }
                    counter++;
                }
            }
            else
            {
                if(!m_StepsLocks[i].Predictors[j].DataId)
                {
                    if (counter==index)
                    {
                        return 1.0;
                    }
                }
                counter++;

                if(!m_StepsLocks[i].Predictors[j].Level)
                {
                    if (counter==index)
                    {
                        return 1.0;
                    }
                }
                counter++;

                if(!m_StepsLocks[i].Predictors[j].TimeHours)
                {
                    if (counter==index)
                    {
                        return (double)GetPredictorTimeHoursIteration(i,j);
                    }
                }
                counter++;
            }

            if(!m_StepsLocks[i].Predictors[j].Umin)
            {
                if (counter==index)
                {
                    return (double)GetPredictorUminIteration(i,j);
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Uptsnb)
            {
                if (counter==index)
                {
                    return (double)GetPredictorUptsnbIteration(i,j);
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Vmin)
            {
                if (counter==index)
                {
                    return (double)GetPredictorVminIteration(i,j);
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Vptsnb)
            {
                if (counter==index)
                {
                    return (double)GetPredictorVptsnbIteration(i,j);
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Weight)
            {
                if (counter==index)
                {
                    return (double)GetPredictorWeightIteration(i,j);
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Criteria)
            {
                if (counter==index)
                {
                    return (double)1.0;
                }
            }
            counter++;
        }
    }

    if(!m_ForecastScoreLocks.AnalogsNumber)
    {
        if (counter==index)
        {
            return (double)GetForecastScoreAnalogsNumberIteration();
        }
    }
    counter++;

    wxASSERT_MSG(counter==m_AllParametersCount, wxString::Format("The counter (%d) did not match the number of parameters (%d).", counter, m_AllParametersCount));
    wxASSERT_MSG(counter<=index, "Couldn't access the desired index in the parameters chromosome.");

    return NaNDouble;
}

void asParametersOptimizationGAs::SetParameterValue(int index, double newVal)
{
    int counter = 0;

    if(!m_TimeArrayAnalogsIntervalDaysLocks)
    {
        if (counter==index)
        {
            int val = asTools::Round(newVal);
            SetTimeArrayAnalogsExcludeDays(val);
            return;
        }
    }
    counter++;

    for (int i=0; i<GetStepsNb(); i++)
    {
        if(!m_StepsLocks[i].AnalogsNumber)
        {
            if (counter==index)
            {
                int val = asTools::Round(newVal);
                SetAnalogsNumber(i, val);
                return;
            }
        }
        counter++;

        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            if (NeedsPreprocessing(i,j))
            {
                for (int k=0; k<GetPreprocessSize(i,j); k++)
                {
                    if(!m_StepsLocks[i].Predictors[j].PreprocessDataId[k])
                    {
                        if (counter==index)
                        {
                            int val = asTools::Round(newVal);

                            VectorString vect = GetPreprocessDataIdVector(i,j,k);
                            if (val<0) val=0;
                            if ((unsigned)val>=vect.size()) val=vect.size()-1;

                            SetPreprocessDataId(i, j, k, vect[val]);

                            return;
                        }
                    }
                    counter++;

                    if(!m_StepsLocks[i].Predictors[j].PreprocessLevels[k])
                    {
                        if (counter==index)
                        {
                            int val = asTools::Round(newVal);

                            VectorFloat vect = GetPreprocessLevelVector(i,j,k);
                            if (val<0) val=0;
                            if ((unsigned)val>=vect.size()) val=vect.size()-1;

                            SetPreprocessLevel(i, j, k, vect[val]);

                            return;
                        }
                    }
                    counter++;

                    if(!m_StepsLocks[i].Predictors[j].PreprocessTimeHours[k])
                    {
                        if (counter==index)
                        {
                            int val = asTools::Round(newVal);
                            SetPreprocessTimeHours(i, j, k, val);
                            return;
                        }
                    }
                    counter++;
                }
            }
            else
            {
                if(!m_StepsLocks[i].Predictors[j].DataId)
                {
                    if (counter==index)
                    {
                        int val = asTools::Round(newVal);

                        VectorString vect = GetPredictorDataIdVector(i,j);
                        if (val<0) val=0;
                        if ((unsigned)val>=vect.size()) val=vect.size()-1;

                        SetPredictorDataId(i, j, vect[val]);

                        return;
                    }
                }
                counter++;

                if(!m_StepsLocks[i].Predictors[j].Level)
                {
                    if (counter==index)
                    {
                        int val = asTools::Round(newVal);

                        VectorFloat vect = GetPredictorLevelVector(i,j);
                        if (val<0) val=0;
                        if ((unsigned)val>=vect.size()) val=vect.size()-1;

                        SetPredictorLevel(i, j, vect[val]);

                        return;
                    }
                }
                counter++;

                if(!m_StepsLocks[i].Predictors[j].TimeHours)
                {
                    if (counter==index)
                    {
                        int val = asTools::Round(newVal);
                        SetPredictorTimeHours(i, j, val);
                        return;
                    }
                }
                counter++;
            }

            if(!m_StepsLocks[i].Predictors[j].Umin)
            {
                if (counter==index)
                {
                    SetPredictorUmin(i, j, newVal);
                    return;
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Uptsnb)
            {
                if (counter==index)
                {
                    int val = asTools::Round(newVal);
                    SetPredictorUptsnb(i, j, val);
                    return;
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Vmin)
            {
                if (counter==index)
                {
                    SetPredictorVmin(i, j, newVal);
                    return;
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Vptsnb)
            {
                if (counter==index)
                {
                    int val = asTools::Round(newVal);
                    SetPredictorVptsnb(i, j, val);
                    return;
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Weight)
            {
                if (counter==index)
                {
                    float val = (float)newVal;
                    SetPredictorWeight(i, j, val);
                    return;
                }
            }
            counter++;

            if(!m_StepsLocks[i].Predictors[j].Criteria)
            {
                if (counter==index)
                {
                    int val = asTools::Round(newVal);

                    VectorString vect = GetPredictorCriteriaVector(i,j);
                    if (val<0) val=0;
                    if ((unsigned)val>=vect.size()) val=vect.size()-1;

                    SetPredictorCriteria(i, j, vect[val]);

                    return;
                }
            }
            counter++;
        }
    }

    if(!m_ForecastScoreLocks.AnalogsNumber)
    {
        if (counter==index)
        {
            int val = asTools::Round(newVal);
            SetForecastScoreAnalogsNumber(val);
            return;
        }
    }
    counter++;

    wxASSERT_MSG(counter==m_AllParametersCount, wxString::Format("The counter (%d) did not match the number of parameters (%d).", counter, m_AllParametersCount));
    wxASSERT_MSG(counter<=index, "Couldn't access the desired index in the parameters chromosome.");

    return;
}

void asParametersOptimizationGAs::SimpleCrossover(asParametersOptimizationGAs &otherParam, VectorInt &crossingPoints)
{
    wxASSERT(crossingPoints.size()>0);

    // Sort the crossing points vector
    asTools::SortArray(&crossingPoints[0], &crossingPoints[crossingPoints.size()-1], Asc);

    unsigned int nextpointIndex = 0;
    int nextpoint = m_ChromosomeIndices[crossingPoints[nextpointIndex]];
    int counter = 0;
    int counterSelfAdapt = 0;
    bool doCrossing = false;

    do
    {
        if (!IsParamLocked(counter))
        {
            if (counter==nextpoint)
            {
                doCrossing = !doCrossing;
                if (nextpointIndex<crossingPoints.size()-1)
                {
                    nextpointIndex++;
                    nextpoint = m_ChromosomeIndices[crossingPoints[nextpointIndex]];
                }
                else
                {
                    nextpoint = -1;
                }
            }

            if (doCrossing)
            {
                double val1 = GetParameterValue(counter);
                double val2 = otherParam.GetParameterValue(counter);
                double newval1 = val2;
                double newval2 = val1;
                SetParameterValue(counter, newval1);
                otherParam.SetParameterValue(counter, newval2);

                // Apply to self-adaptation
                if (m_HasChromosomeSelfAdaptationMutationRate)
                {
                    float mutRate = GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                    SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, otherParam.GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt));
                    otherParam.SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, mutRate);
                }
                if (m_HasChromosomeSelfAdaptationMutationRadius)
                {
                    float mutRadius = GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                    SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, otherParam.GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt));
                    otherParam.SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, mutRadius);
                }
            }
            counterSelfAdapt++;
        }
        counter++;
    }
    while(!m_ParametersListOver);
}

void asParametersOptimizationGAs::BlendingCrossover(asParametersOptimizationGAs &otherParam, VectorInt &crossingPoints, bool shareBeta, double betaMin, double betaMax)
{
    wxASSERT(crossingPoints.size()>0);

    // Sort the crossing points vector
    asTools::SortArray(&crossingPoints[0], &crossingPoints[crossingPoints.size()-1], Asc);

    unsigned int nextpointIndex = 0;
    int nextpoint = m_ChromosomeIndices[crossingPoints[nextpointIndex]];
    int counter = 0;
    int counterSelfAdapt = 0;
    bool doCrossing = false;
    double beta = asTools::Random(betaMin, betaMax);

    do
    {
        if (!IsParamLocked(counter))
        {
            if (counter==nextpoint)
            {
                doCrossing = !doCrossing;
                if (nextpointIndex<crossingPoints.size()-1)
                {
                    nextpointIndex++;
                    nextpoint = m_ChromosomeIndices[crossingPoints[nextpointIndex]];
                }
                else
                {
                    nextpoint = -1;
                }
            }

            if (doCrossing)
            {
                if (!shareBeta)
                {
                    beta = asTools::Random(betaMin, betaMax);
                }

                double val1 = GetParameterValue(counter);
                double val2 = otherParam.GetParameterValue(counter);
                double newval1 = beta*val1 + (1.0-beta)*val2;
                double newval2 = (1.0-beta)*val1 + beta*val2;
                SetParameterValue(counter, newval1);
                otherParam.SetParameterValue(counter, newval2);

                // Apply to self-adaptation
                if (m_HasChromosomeSelfAdaptationMutationRate)
                {
                    float mutRate1 = GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                    float mutRate2 = otherParam.GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                    float newMutRate1 = beta*mutRate1 + (1.0-beta)*mutRate2;
                    float newMutRate2 = (1.0-beta)*mutRate1 + beta*mutRate2;
                    SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, newMutRate1);
                    otherParam.SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, newMutRate2);
                }
                if (m_HasChromosomeSelfAdaptationMutationRadius)
                {
                    float mutRadius1 = GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                    float mutRadius2 = otherParam.GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                    float newMutRadius1 = beta*mutRadius1 + (1.0-beta)*mutRadius2;
                    float newMutRadius2 = (1.0-beta)*mutRadius1 + beta*mutRadius2;
                    SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, newMutRadius1);
                    otherParam.SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, newMutRadius2);
                }
            }
            counterSelfAdapt++;
        }
        counter++;
    }
    while(!m_ParametersListOver);
}

void asParametersOptimizationGAs::HeuristicCrossover(asParametersOptimizationGAs &otherParam, VectorInt &crossingPoints, bool shareBeta, double betaMin, double betaMax)
{
    wxASSERT(crossingPoints.size()>0);

    // Sort the crossing points vector
    asTools::SortArray(&crossingPoints[0], &crossingPoints[crossingPoints.size()-1], Asc);

    unsigned int nextpointIndex = 0;
    int nextpoint = m_ChromosomeIndices[crossingPoints[nextpointIndex]];
    int counter = 0;
    int counterSelfAdapt = 0;
    bool doCrossing = false;
    double beta = asTools::Random(betaMin, betaMax);

    do
    {
        if (!IsParamLocked(counter))
        {
            if (counter==nextpoint)
            {
                doCrossing = !doCrossing;
                if (nextpointIndex<crossingPoints.size()-1)
                {
                    nextpointIndex++;
                    nextpoint = m_ChromosomeIndices[crossingPoints[nextpointIndex]];
                }
                else
                {
                    nextpoint = -1;
                }
            }

            if (doCrossing)
            {
                if (!shareBeta)
                {
                    beta = asTools::Random(betaMin, betaMax);
                }

                double val1 = GetParameterValue(counter);
                double val2 = otherParam.GetParameterValue(counter);
                double newval1 = beta*(val1 - val2) + val1;
                double newval2 = beta*(val2 - val1) + val2;
                SetParameterValue(counter, newval1);
                otherParam.SetParameterValue(counter, newval2);

                // Apply to self-adaptation
                if (m_HasChromosomeSelfAdaptationMutationRate)
                {
                    float mutRate1 = GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                    float mutRate2 = otherParam.GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                    float newMutRate1 = beta*(mutRate1 - mutRate2) + mutRate1;
                    float newMutRate2 = beta*(mutRate2 - mutRate1) + mutRate2;
                    SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, newMutRate1);
                    otherParam.SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, newMutRate2);
                }
                if (m_HasChromosomeSelfAdaptationMutationRadius)
                {
                    float mutRadius1 = GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                    float mutRadius2 = otherParam.GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                    float newMutRadius1 = beta*(mutRadius1 - mutRadius2) + mutRadius1;
                    float newMutRadius2 = beta*(mutRadius2 - mutRadius1) + mutRadius2;
                    SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, newMutRadius1);
                    otherParam.SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, newMutRadius2);
                }
            }
            counterSelfAdapt++;
        }
        counter++;
    }
    while(!m_ParametersListOver);

}

void asParametersOptimizationGAs::BinaryLikeCrossover(asParametersOptimizationGAs &otherParam, VectorInt &crossingPoints, bool shareBeta, double betaMin, double betaMax)
{
    wxASSERT(crossingPoints.size()>0);

    // Sort the crossing points vector
    asTools::SortArray(&crossingPoints[0], &crossingPoints[crossingPoints.size()-1], Asc);

    unsigned int nextpointIndex = 0;
    int nextpoint = m_ChromosomeIndices[crossingPoints[nextpointIndex]];
    int counter = 0;
    int counterSelfAdapt = 0;
    bool doCrossing = false;
    double beta = asTools::Random(betaMin, betaMax);

    do
    {
        if (!IsParamLocked(counter))
        {
            if (counter==nextpoint)
            {
                if (!shareBeta)
                {
                    beta = asTools::Random(betaMin, betaMax);
                }

                double val1 = GetParameterValue(counter);
                double val2 = otherParam.GetParameterValue(counter);
                double newval1 = val1 - beta*(val1 - val2);
                double newval2 = val2 + beta*(val1 - val2);
                SetParameterValue(counter, newval1);
                otherParam.SetParameterValue(counter, newval2);

                // Apply to self-adaptation
                if (m_HasChromosomeSelfAdaptationMutationRate)
                {
                    float mutRate1 = GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                    float mutRate2 = otherParam.GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                    float newMutRate1 = mutRate1 - beta*(mutRate1 - mutRate2);
                    float newMutRate2 = mutRate2 + beta*(mutRate1 - mutRate2);
                    SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, newMutRate1);
                    otherParam.SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, newMutRate2);
                }
                if (m_HasChromosomeSelfAdaptationMutationRadius)
                {
                    float mutRadius1 = GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                    float mutRadius2 = otherParam.GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                    float newMutRadius1 = mutRadius1 - beta*(mutRadius1 - mutRadius2);
                    float newMutRadius2 = mutRadius2 + beta*(mutRadius1 - mutRadius2);
                    SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, newMutRadius1);
                    otherParam.SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, newMutRadius2);
                }

                doCrossing = !doCrossing;
                if (nextpointIndex<crossingPoints.size()-1)
                {
                    nextpointIndex++;
                    nextpoint = m_ChromosomeIndices[crossingPoints[nextpointIndex]];
                }
                else
                {
                    nextpoint = -1;
                }
            }
            else
            {
                if (doCrossing)
                {
                    double val1 = GetParameterValue(counter);
                    double val2 = otherParam.GetParameterValue(counter);
                    double newval1 = val2;
                    double newval2 = val1;
                    SetParameterValue(counter, newval1);
                    otherParam.SetParameterValue(counter, newval2);

                    // Apply to self-adaptation
                    if (m_HasChromosomeSelfAdaptationMutationRate)
                    {
                        float mutRate = GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                        SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, otherParam.GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt));
                        otherParam.SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, mutRate);
                    }
                    if (m_HasChromosomeSelfAdaptationMutationRadius)
                    {
                        float mutRadius = GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                        SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, otherParam.GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt));
                        otherParam.SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, mutRadius);
                    }
                }
            }
            counterSelfAdapt++;
        }
        counter++;
    }
    while(!m_ParametersListOver);

}

void asParametersOptimizationGAs::LinearCrossover(asParametersOptimizationGAs &otherParam, asParametersOptimizationGAs &thirdParam, VectorInt &crossingPoints)
{
    wxASSERT(crossingPoints.size()>0);

    // Sort the crossing points vector
    asTools::SortArray(&crossingPoints[0], &crossingPoints[crossingPoints.size()-1], Asc);

    unsigned int nextpointIndex = 0;
    int nextpoint = m_ChromosomeIndices[crossingPoints[nextpointIndex]];
    int counter = 0;
    int counterSelfAdapt = 0;
    bool doCrossing = false;

    do
    {
        if (!IsParamLocked(counter))
        {
            if (counter==nextpoint)
            {
                doCrossing = !doCrossing;
                if (nextpointIndex<crossingPoints.size()-1)
                {
                    nextpointIndex++;
                    nextpoint = m_ChromosomeIndices[crossingPoints[nextpointIndex]];
                }
                else
                {
                    nextpoint = -1;
                }
            }

            if (doCrossing)
            {
                double val1 = GetParameterValue(counter);
                double val2 = otherParam.GetParameterValue(counter);
                double newval1 = 0.5*val1 + 0.5*val2;
                double newval2 = 1.5*val1 - 0.5*val2;
                double newval3 = -0.5*val1 + 1.5*val2;
                SetParameterValue(counter, newval1);
                otherParam.SetParameterValue(counter, newval2);
                thirdParam.SetParameterValue(counter, newval3);

                // Apply to self-adaptation
                if (m_HasChromosomeSelfAdaptationMutationRate)
                {
                    float mutRate1 = GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                    float mutRate2 = otherParam.GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                    float newMutRate1 = 0.5*mutRate1 + 0.5*mutRate2;
                    float newMutRate2 = 1.5*mutRate1 - 0.5*mutRate2;
                    float newMutRate3 = -0.5*mutRate1 + 1.5*mutRate2;
                    SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, newMutRate1);
                    otherParam.SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, newMutRate2);
                    thirdParam.SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, newMutRate3);
                }
                if (m_HasChromosomeSelfAdaptationMutationRadius)
                {
                    float mutRadius1 = GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                    float mutRadius2 = otherParam.GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                    float newMutRadius1 = 0.5*mutRadius1 + 0.5*mutRadius2;
                    float newMutRadius2 = 1.5*mutRadius1 - 0.5*mutRadius2;
                    float newMutRadius3 = -0.5*mutRadius1 + 1.5*mutRadius2;
                    SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, newMutRadius1);
                    otherParam.SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, newMutRadius2);
                    thirdParam.SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, newMutRadius3);
                }
            }
            counterSelfAdapt++;
        }
        counter++;
    }
    while(!m_ParametersListOver);

}

void asParametersOptimizationGAs::LinearInterpolation(asParametersOptimizationGAs &otherParam, bool shareBeta)
{
    double beta = asTools::Random(0.0, 1.0);
    int counterSelfAdapt = 0;
    int counter = 0;

    do
    {
        if (!IsParamLocked(counter))
        {
            if (!shareBeta)
            {
                beta = asTools::Random(0.0, 1.0);
            }

            double val1 = GetParameterValue(counter);
            double val2 = otherParam.GetParameterValue(counter);
            double newval1 = val1 - beta*(val1-val2);
            double newval2 = val2 + beta*(val1-val2);
            SetParameterValue(counter, newval1);
            otherParam.SetParameterValue(counter, newval2);

            // Apply to self-adaptation
            if (m_HasChromosomeSelfAdaptationMutationRate)
            {
                float mutRate1 = GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                float mutRate2 = otherParam.GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                float newMutRate1 = mutRate1 - beta*(mutRate1-mutRate2);
                float newMutRate2 = mutRate2 + beta*(mutRate1-mutRate2);
                SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, newMutRate1);
                otherParam.SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, newMutRate2);
            }
            if (m_HasChromosomeSelfAdaptationMutationRadius)
            {
                float mutRadius1 = GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                float mutRadius2 = otherParam.GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                float newMutRadius1 = mutRadius1 - beta*(mutRadius1-mutRadius2);
                float newMutRadius2 = mutRadius2 + beta*(mutRadius1-mutRadius2);
                SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, newMutRadius1);
                otherParam.SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, newMutRadius2);
            }
            counterSelfAdapt++;
        }
        counter++;
    }
    while(!m_ParametersListOver);
}

void asParametersOptimizationGAs::MutateUniformDistribution(double probability, bool &hasMutated)
{
    int counter = 0;

    do
    {
        if (!IsParamLocked(counter))
        {
            if (asTools::Random(0.0, 1.0)<probability)
            {
                double newVal = asTools::Random(GetParameterLowerLimit(counter),
                                                GetParameterUpperLimit(counter),
                                                GetParameterIteration(counter));
                SetParameterValue(counter, newVal);

                hasMutated = true;
            }
        }
        counter++;
    }
    while(!m_ParametersListOver);
}

void asParametersOptimizationGAs::MutateNormalDistribution(double probability, double stdDevRatioRange, bool &hasMutated)
{
    int counter = 0;

    do
    {
        if (!IsParamLocked(counter))
        {
            if (asTools::Random(0.0, 1.0)<probability)
            {
                if (!IsParamList(counter))
                {
                    double mean = GetParameterValue(counter);
                    double stdDev = stdDevRatioRange*(GetParameterUpperLimit(counter)-GetParameterLowerLimit(counter));
                    double step = GetParameterIteration(counter);
                    double newVal = asTools::RandomNormalDistribution(mean, stdDev, step);
                    SetParameterValue(counter, newVal);
                }
                else
                {
                    // Uniform distribution in the case of parameters as a list
                    double newVal = asTools::Random(GetParameterLowerLimit(counter),
                                                    GetParameterUpperLimit(counter),
                                                    GetParameterIteration(counter));
                    SetParameterValue(counter, newVal);
                }

                hasMutated = true;
            }
        }
        counter++;
    }
    while(!m_ParametersListOver);
}

void asParametersOptimizationGAs::MutateNonUniform(double probability, int nbGen, int nbGenMax, double minRate, bool &hasMutated)
{
    double ratioGens = (double)nbGen/(double)nbGenMax;
    double cstFactor = (1.0-wxMin(ratioGens, 1.0)*(1.0-minRate))*(1.0-wxMin(ratioGens, 1.0)*(1.0-minRate));

    int counter = 0;

    do
    {
        if (!IsParamLocked(counter))
        {
            if (asTools::Random(0.0, 1.0)<probability)
            {
                if (!IsParamList(counter))
                {
                    double r1 = asTools::Random(0.0, 1.0);
                    double r2 = asTools::Random(0.0, 1.0);
                    double actVal = GetParameterValue(counter);
                    double lowerLimit = GetParameterLowerLimit(counter);
                    double upperLimit = GetParameterUpperLimit(counter);
                    double newVal;

                    if (r1<0.5)
                    {
                        newVal = actVal + (upperLimit-actVal)*r2*cstFactor;
                    }
                    else
                    {
                        newVal = actVal - (actVal-lowerLimit)*r2*cstFactor;
                    }

                    SetParameterValue(counter, newVal);
                }
                else
                {
                    // Uniform distribution in the case of parameters as a list
                    double newVal = asTools::Random(GetParameterLowerLimit(counter),
                                                    GetParameterUpperLimit(counter),
                                                    GetParameterIteration(counter));
                    SetParameterValue(counter, newVal);
                }

                hasMutated = true;
            }
        }
        counter++;
    }
    while(!m_ParametersListOver);
}

void asParametersOptimizationGAs::MutateSelfAdaptationRate(bool &hasMutated)
{
    // Mutate mutation probability
    if (asTools::Random(0.0, 1.0)<m_IndividualSelfAdaptationMutationRate)
    {
        m_IndividualSelfAdaptationMutationRate = asTools::Random(0.0, 1.0);
    }

    // Mutate data
    MutateUniformDistribution(m_IndividualSelfAdaptationMutationRate, hasMutated);

}

void asParametersOptimizationGAs::MutateSelfAdaptationRadius(bool &hasMutated)
{
    // Mutate mutation probability
    if (asTools::Random(0.0, 1.0)<m_IndividualSelfAdaptationMutationRate)
    {
        m_IndividualSelfAdaptationMutationRate = asTools::Random(0.0, 1.0);
    }

    // Mutate mutation radius. Use the radius here as a probability !!
    if (asTools::Random(0.0, 1.0)<m_IndividualSelfAdaptationMutationRadius)
    {
        m_IndividualSelfAdaptationMutationRadius = asTools::Random(0.0, 1.0);
    }

    // Mutate data
    int counter = 0;
    do
    {
        if (!IsParamLocked(counter))
        {
            if (asTools::Random(0.0, 1.0)<m_IndividualSelfAdaptationMutationRate)
            {
                if (!IsParamList(counter))
                {
                    double r1 = asTools::Random(0.0, 1.0);
                    double r2 = asTools::Random(0.0, 1.0);
                    double actVal = GetParameterValue(counter);
                    double lowerLimit = GetParameterLowerLimit(counter);
                    double upperLimit = GetParameterUpperLimit(counter);
                    double newVal;

                    if (r1<0.5)
                    {
                        newVal = actVal + (upperLimit-actVal)*r2*m_IndividualSelfAdaptationMutationRadius;
                    }
                    else
                    {
                        newVal = actVal - (actVal-lowerLimit)*r2*m_IndividualSelfAdaptationMutationRadius;
                    }

                    SetParameterValue(counter, newVal);
                }
                else
                {
                    // Uniform distribution in the case of parameters as a list
                    double newVal = asTools::Random(GetParameterLowerLimit(counter),
                                                    GetParameterUpperLimit(counter),
                                                    GetParameterIteration(counter));
                    SetParameterValue(counter, newVal);
                }

                hasMutated = true;
            }
        }
        counter++;
    }
    while(!m_ParametersListOver);
}

void asParametersOptimizationGAs::MutateSelfAdaptationRateChromosome(bool &hasMutated)
{
    wxASSERT(m_ChromosomeSelfAdaptationMutationRate.size()>0);
    wxASSERT(m_ChromosomeSelfAdaptationMutationRate.size()==GetChromosomeLength());
    int chromosomeLength = m_ChromosomeSelfAdaptationMutationRate.size();

    // Mutate mutation probability
    for (unsigned int i=0; i<m_ChromosomeSelfAdaptationMutationRate.size(); i++)
    {
        if (asTools::Random(0.0, 1.0)<m_ChromosomeSelfAdaptationMutationRate[i])
        {
            m_ChromosomeSelfAdaptationMutationRate[i] = asTools::Random(0.0, 1.0);
        }

    }

    // Mutate data
    int counter = 0;
    int counterSelfAdapt = 0;
    do
    {
        if (!IsParamLocked(counter))
        {
            wxASSERT(counterSelfAdapt<chromosomeLength);

            if (asTools::Random(0.0, 1.0)<m_ChromosomeSelfAdaptationMutationRate[counterSelfAdapt])
            {
                // Uniform distribution in the case of parameters as a list
                double newVal = asTools::Random(GetParameterLowerLimit(counter),
                                                GetParameterUpperLimit(counter),
                                                GetParameterIteration(counter));
                SetParameterValue(counter, newVal);

                hasMutated = true;
            }
            counterSelfAdapt++;
        }
        counter++;
    }
    while(!m_ParametersListOver);
}

void asParametersOptimizationGAs::MutateSelfAdaptationRadiusChromosome(bool &hasMutated)
{
    wxASSERT(m_ChromosomeSelfAdaptationMutationRate.size()>0);
    wxASSERT(m_ChromosomeSelfAdaptationMutationRadius.size()>0);
    wxASSERT(m_ChromosomeSelfAdaptationMutationRate.size()==m_ChromosomeSelfAdaptationMutationRadius.size());
    wxASSERT(m_ChromosomeSelfAdaptationMutationRate.size()==GetChromosomeLength());
    int chromosomeLength = m_ChromosomeSelfAdaptationMutationRate.size();

    // Mutate mutation probability
    for (unsigned int i=0; i<m_ChromosomeSelfAdaptationMutationRate.size(); i++)
    {
        if (asTools::Random(0.0, 1.0)<m_ChromosomeSelfAdaptationMutationRate[i])
        {
            m_ChromosomeSelfAdaptationMutationRate[i] = asTools::Random(0.0, 1.0);
        }

    }

    // Mutate mutation radius. Use the radius here as a probability !!
    for (unsigned int i=0; i<m_ChromosomeSelfAdaptationMutationRadius.size(); i++)
    {
        if (asTools::Random(0.0, 1.0)<m_ChromosomeSelfAdaptationMutationRadius[i])
        {
            m_ChromosomeSelfAdaptationMutationRadius[i] = asTools::Random(0.0, 1.0);
        }

    }

    // Mutate data
    int counter = 0;
    int counterSelfAdapt = 0;
    do
    {
        if (!IsParamLocked(counter))
        {
            wxASSERT(counterSelfAdapt<chromosomeLength);

            if (asTools::Random(0.0, 1.0)<m_ChromosomeSelfAdaptationMutationRate[counterSelfAdapt])
            {
                if (!IsParamList(counter))
                {
                    double r1 = asTools::Random(0.0, 1.0);
                    double r2 = asTools::Random(0.0, 1.0);
                    double actVal = GetParameterValue(counter);
                    double lowerLimit = GetParameterLowerLimit(counter);
                    double upperLimit = GetParameterUpperLimit(counter);
                    double newVal;

                    if (r1<0.5)
                    {
                        newVal = actVal + (upperLimit-actVal)*r2*m_ChromosomeSelfAdaptationMutationRadius[counterSelfAdapt];
                    }
                    else
                    {
                        newVal = actVal - (actVal-lowerLimit)*r2*m_ChromosomeSelfAdaptationMutationRadius[counterSelfAdapt];
                    }

                    SetParameterValue(counter, newVal);
                }
                else
                {
                    // Uniform distribution in the case of parameters as a list
                    double newVal = asTools::Random(GetParameterLowerLimit(counter),
                                                    GetParameterUpperLimit(counter),
                                                    GetParameterIteration(counter));
                    SetParameterValue(counter, newVal);
                }

                hasMutated = true;
            }
            counterSelfAdapt++;
        }
        counter++;
    }
    while(!m_ParametersListOver);
}

void asParametersOptimizationGAs::MutateMultiScale(double probability, bool &hasMutated)
{
    // Choose the radius
    double radiusChoice = asTools::Random(0.0, 1.0);
    double radius = 0;

    if (radiusChoice<0.25)
    {
        radius = 1;
    }
    else if (radiusChoice>=0.25 && radiusChoice<0.50)
    {
        radius = 0.5;
    }
    else if (radiusChoice>=0.50 && radiusChoice<0.75)
    {
        radius = 0.1;
    }
    else
    {
        radius = 0.02;
    }

    // Mutate data
    int counter = 0;
    do
    {
        if (!IsParamLocked(counter))
        {
            if (asTools::Random(0.0, 1.0)<probability)
            {
                if (!IsParamList(counter))
                {
                    double r1 = asTools::Random(0.0, 1.0);
                    double r2 = asTools::Random(0.0, 1.0);
                    double actVal = GetParameterValue(counter);
                    double lowerLimit = GetParameterLowerLimit(counter);
                    double upperLimit = GetParameterUpperLimit(counter);
                    double newVal;

                    if (r1<0.5)
                    {
                        newVal = actVal + (upperLimit-actVal)*r2*radius;
                    }
                    else
                    {
                        newVal = actVal - (actVal-lowerLimit)*r2*radius;
                    }

                    SetParameterValue(counter, newVal);
                }
                else
                {
                    // Uniform distribution in the case of parameters as a list
                    double newVal = asTools::Random(GetParameterLowerLimit(counter),
                                                    GetParameterUpperLimit(counter),
                                                    GetParameterIteration(counter));
                    SetParameterValue(counter, newVal);
                }

                hasMutated = true;
            }
        }
        counter++;
    }
    while(!m_ParametersListOver);
}
