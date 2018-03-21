/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * When distributing Covered Code, include this CDDL Header Notice in
 * each file and include the License file (licence.txt). If applicable,
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 *
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#include "asParametersOptimizationGAs.h"

#include <asFileParametersOptimization.h>


asParametersOptimizationGAs::asParametersOptimizationGAs()
        : asParametersOptimization(),
          m_individualSelfAdaptationMutationRate(0),
          m_individualSelfAdaptationMutationRadius(0),
          m_hasChromosomeSelfAdaptationMutationRate(false),
          m_hasChromosomeSelfAdaptationMutationRadius(false),
          m_timeArrayAnalogsIntervalDaysIteration(0),
          m_timeArrayAnalogsIntervalDaysUpperLimit(0),
          m_timeArrayAnalogsIntervalDaysLowerLimit(0),
          m_timeArrayAnalogsIntervalDaysLocks(true),
          m_allParametersCount(0),
          m_parametersListOver(false)
{

}

asParametersOptimizationGAs::~asParametersOptimizationGAs()
{
    //dtor
}

void asParametersOptimizationGAs::BuildChromosomes()
{
    int counter = 0;
    vi indices;

    if (!m_timeArrayAnalogsIntervalDaysLocks) {
        indices.push_back(counter);
    }
    counter++;

    for (int i = 0; i < GetStepsNb(); i++) {
        if (!m_stepsLocks[i].analogsNumber) {
            indices.push_back(counter);
        }
        counter++;

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (!m_stepsLocks[i].predictors[j].preprocessDataId[k]) {
                        indices.push_back(counter);
                    }
                    counter++;

                    if (!m_stepsLocks[i].predictors[j].preprocessLevels[k]) {
                        indices.push_back(counter);
                    }
                    counter++;

                    if (!m_stepsLocks[i].predictors[j].preprocessTimeHours[k]) {
                        indices.push_back(counter);
                    }
                    counter++;
                }
            } else {
                if (!m_stepsLocks[i].predictors[j].dataId) {
                    indices.push_back(counter);
                }
                counter++;

                if (!m_stepsLocks[i].predictors[j].level) {
                    indices.push_back(counter);
                }
                counter++;

                if (!m_stepsLocks[i].predictors[j].timeHours) {
                    indices.push_back(counter);
                }
                counter++;
            }

            if (!m_stepsLocks[i].predictors[j].xMin) {
                indices.push_back(counter);
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].xPtsNb) {
                indices.push_back(counter);
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].yMin) {
                indices.push_back(counter);
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].yPtsNb) {
                indices.push_back(counter);
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].weight) {
                indices.push_back(counter);
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].criteria) {
                indices.push_back(counter);
            }
            counter++;
        }
    }

    m_chromosomeIndices = indices;
    m_allParametersCount = counter;
}

void asParametersOptimizationGAs::InitIndividualSelfAdaptationMutationRate()
{
    m_individualSelfAdaptationMutationRate = asRandom(0.0, 1.0);
}

void asParametersOptimizationGAs::InitIndividualSelfAdaptationMutationRadius()
{
    m_individualSelfAdaptationMutationRadius = asRandom(0.0, 1.0);
}

void asParametersOptimizationGAs::InitChromosomeSelfAdaptationMutationRate()
{
    int length = GetChromosomeLength();
    m_chromosomeSelfAdaptationMutationRate.resize(length);

    for (int i = 0; i < length; i++) {
        m_chromosomeSelfAdaptationMutationRate[i] = asRandom(0.0, 1.0);
    }

    m_hasChromosomeSelfAdaptationMutationRate = true;
}

void asParametersOptimizationGAs::InitChromosomeSelfAdaptationMutationRadius()
{
    int length = GetChromosomeLength();
    m_chromosomeSelfAdaptationMutationRadius.resize(length);

    for (int i = 0; i < length; i++) {
        m_chromosomeSelfAdaptationMutationRadius[i] = asRandom(0.0, 1.0);
    }

    m_hasChromosomeSelfAdaptationMutationRadius = true;
}

bool asParametersOptimizationGAs::IsParamLocked(int index)
{
    int counter = 0;
    m_parametersListOver = false;

    if (!m_timeArrayAnalogsIntervalDaysLocks) {
        if (counter == index)
            return false;
    } else {
        if (counter == index)
            return true;
    }
    counter++;

    for (int i = 0; i < GetStepsNb(); i++) {

        if (!m_stepsLocks[i].analogsNumber) {
            if (counter == index)
                return false;
        } else {
            if (counter == index)
                return true;
        }
        counter++;

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (!m_stepsLocks[i].predictors[j].preprocessDataId[k]) {
                        if (counter == index)
                            return false;
                    } else {
                        if (counter == index)
                            return true;
                    }
                    counter++;

                    if (!m_stepsLocks[i].predictors[j].preprocessLevels[k]) {
                        if (counter == index)
                            return false;
                    } else {
                        if (counter == index)
                            return true;
                    }
                    counter++;

                    if (!m_stepsLocks[i].predictors[j].preprocessTimeHours[k]) {
                        if (counter == index)
                            return false;
                    } else {
                        if (counter == index)
                            return true;
                    }
                    counter++;
                }
            } else {
                if (!m_stepsLocks[i].predictors[j].dataId) {
                    if (counter == index)
                        return false;
                } else {
                    if (counter == index)
                        return true;
                }
                counter++;

                if (!m_stepsLocks[i].predictors[j].level) {
                    if (counter == index)
                        return false;
                } else {
                    if (counter == index)
                        return true;
                }
                counter++;

                if (!m_stepsLocks[i].predictors[j].timeHours) {
                    if (counter == index)
                        return false;
                } else {
                    if (counter == index)
                        return true;
                }
                counter++;
            }

            if (!m_stepsLocks[i].predictors[j].xMin) {
                if (counter == index)
                    return false;
            } else {
                if (counter == index)
                    return true;
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].xPtsNb) {
                if (counter == index)
                    return false;
            } else {
                if (counter == index)
                    return true;
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].yMin) {
                if (counter == index)
                    return false;
            } else {
                if (counter == index)
                    return true;
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].yPtsNb) {
                if (counter == index)
                    return false;
            } else {
                if (counter == index)
                    return true;
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].weight) {
                if (counter == index)
                    return false;
            } else {
                if (counter == index)
                    return true;
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].criteria) {
                if (counter == index)
                    return false;
            } else {
                if (counter == index)
                    return true;
            }
            counter++;
        }
    }

    wxASSERT_MSG(counter == m_allParametersCount,
                 wxString::Format("The counter (%d) did not match the number of parameters (%d).", counter,
                                  m_allParametersCount));
    wxASSERT_MSG(counter <= index, "Couldn't access the desired index in the parameters chromosome.");

    m_parametersListOver = true;

    return true;
}

int asParametersOptimizationGAs::GetParamType(int index)
{
    /*
     * return: 
     * - 1 for value 
     * - 2 for advanced list (notion of proximity) 
     * - 3 for simple list (no proximity between elements)
     */

    int counter = 0;

    // AnalogsIntervalDays
    if (counter == index) {
        return 1;
    }
    counter++;

    for (int i = 0; i < GetStepsNb(); i++) {

        // AnalogsNumber
        if (counter == index) {
            return 1;
        }
        counter++;

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    // PreprocessDataId
                    if (counter == index) {
                        return 3;
                    }
                    counter++;

                    // PreprocessLevel
                    if (counter == index) {
                        return 3;
                    }
                    counter++;

                    // PreprocessTimeHours
                    if (counter == index) {
                        return 1;
                    }
                    counter++;
                }
            } else {
                // DataId
                if (counter == index) {
                    return 3;
                }
                counter++;

                // Level
                if (counter == index) {
                    return 3;
                }
                counter++;

                // TimeHours
                if (counter == index) {
                    return 1;
                }
                counter++;
            }

            // xMin
            if (counter == index) {
                return 1;
            }
            counter++;

            // xPtsNb
            if (counter == index) {
                return 1;
            }
            counter++;

            // yMin
            if (counter == index) {
                return 1;
            }
            counter++;

            // yPtsNb
            if (counter == index) {
                return 1;
            }
            counter++;

            // Weight
            if (counter == index) {
                return 1;
            }
            counter++;

            // Criteria
            if (counter == index) {
                return 3;
            }
            counter++;
        }
    }

    wxASSERT_MSG(counter == m_allParametersCount,
                 wxString::Format("The counter (%d) did not match the number of parameters (%d).", counter,
                                  m_allParametersCount));
    wxASSERT_MSG(counter <= index, "Couldn't access the desired index in the parameters chromosome.");

    asThrowException(_("We should never reach that point..."));

}

double asParametersOptimizationGAs::GetParameterValue(int index)
{
    int counter = 0;

    if (!m_timeArrayAnalogsIntervalDaysLocks) {
        if (counter == index) {
            return (double) GetTimeArrayAnalogsIntervalDays();
        }
    }
    counter++;

    for (int i = 0; i < GetStepsNb(); i++) {

        if (!m_stepsLocks[i].analogsNumber) {
            if (counter == index) {
                return (double) GetAnalogsNumber(i);
            }
        }
        counter++;

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (!m_stepsLocks[i].predictors[j].preprocessDataId[k]) {
                        if (counter == index) {
                            wxString dat = GetPreprocessDataId(i, j, k);
                            vwxs vect = GetPreprocessDataIdVector(i, j, k);
                            int iPre = -1;
                            for (unsigned int r = 0; r < vect.size(); r++) {
                                if (vect[r].IsSameAs(dat, false))
                                    iPre = r;
                            }
                            wxASSERT(iPre >= 0);

                            return (double) iPre;
                        }
                    }
                    counter++;

                    if (!m_stepsLocks[i].predictors[j].preprocessLevels[k]) {
                        if (counter == index) {
                            float dat = GetPreprocessLevel(i, j, k);
                            vf vect = GetPreprocessLevelVector(i, j, k);
                            int iPre = -1;
                            for (unsigned int r = 0; r < vect.size(); r++) {
                                if (vect[r] == dat)
                                    iPre = r;
                            }
                            wxASSERT(iPre >= 0);

                            return (double) iPre;
                        }
                    }
                    counter++;

                    if (!m_stepsLocks[i].predictors[j].preprocessTimeHours[k]) {
                        if (counter == index) {
                            return (double) GetPreprocessTimeHours(i, j, k);
                        }
                    }
                    counter++;
                }
            } else {
                if (!m_stepsLocks[i].predictors[j].dataId) {
                    if (counter == index) {
                        wxString dat = GetPredictorDataId(i, j);
                        vwxs vect = GetPredictorDataIdVector(i, j);
                        int iPre = -1;
                        for (unsigned int r = 0; r < vect.size(); r++) {
                            if (vect[r].IsSameAs(dat, false))
                                iPre = r;
                        }
                        wxASSERT(iPre >= 0);

                        return (double) iPre;
                    }
                }
                counter++;

                if (!m_stepsLocks[i].predictors[j].level) {
                    if (counter == index) {
                        float dat = GetPredictorLevel(i, j);
                        vf vect = GetPredictorLevelVector(i, j);
                        int iPre = -1;
                        for (unsigned int r = 0; r < vect.size(); r++) {
                            if (vect[r] == dat)
                                iPre = r;
                        }
                        wxASSERT(iPre >= 0);

                        return (double) iPre;
                    }
                }
                counter++;

                if (!m_stepsLocks[i].predictors[j].timeHours) {
                    if (counter == index) {
                        return (double) GetPredictorTimeHours(i, j);
                    }
                }
                counter++;
            }

            if (!m_stepsLocks[i].predictors[j].xMin) {
                if (counter == index) {
                    return (double) GetPredictorXmin(i, j);
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].xPtsNb) {
                if (counter == index) {
                    return (double) GetPredictorXptsnb(i, j);
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].yMin) {
                if (counter == index) {
                    return (double) GetPredictorYmin(i, j);
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].yPtsNb) {
                if (counter == index) {
                    return (double) GetPredictorYptsnb(i, j);
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].weight) {
                if (counter == index) {
                    return (double) GetPredictorWeight(i, j);
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].criteria) {
                if (counter == index) {
                    wxString dat = GetPredictorCriteria(i, j);
                    vwxs vect = GetPredictorCriteriaVector(i, j);
                    int iPre = -1;
                    for (unsigned int r = 0; r < vect.size(); r++) {
                        if (vect[r].IsSameAs(dat, false))
                            iPre = r;
                    }
                    wxASSERT(iPre >= 0);

                    return (double) iPre;
                }
            }
            counter++;
        }
    }

    wxASSERT_MSG(counter == m_allParametersCount,
                 wxString::Format("The counter (%d) did not match the number of parameters (%d).", counter,
                                  m_allParametersCount));
    wxASSERT_MSG(counter <= index, "Couldn't access the desired index in the parameters chromosome.");

    return NaNd;
}

double asParametersOptimizationGAs::GetParameterUpperLimit(int index)
{
    int counter = 0;

    if (!m_timeArrayAnalogsIntervalDaysLocks) {
        if (counter == index) {
            return (double) GetTimeArrayAnalogsIntervalDaysUpperLimit();
        }
    }
    counter++;

    for (int i = 0; i < GetStepsNb(); i++) {

        if (!m_stepsLocks[i].analogsNumber) {
            if (counter == index) {
                return (double) GetAnalogsNumberUpperLimit(i);
            }
        }
        counter++;

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (!m_stepsLocks[i].predictors[j].preprocessDataId[k]) {
                        if (counter == index) {
                            vwxs vect = GetPreprocessDataIdVector(i, j, k);
                            return (double) (vect.size() - 1);
                        }
                    }
                    counter++;

                    if (!m_stepsLocks[i].predictors[j].preprocessLevels[k]) {
                        if (counter == index) {
                            vf vect = GetPreprocessLevelVector(i, j, k);
                            return (double) (vect.size() - 1);
                        }
                    }
                    counter++;

                    if (!m_stepsLocks[i].predictors[j].preprocessTimeHours[k]) {
                        if (counter == index) {
                            return (double) GetPreprocessTimeHoursUpperLimit(i, j, k);
                        }
                    }
                    counter++;
                }
            } else {
                if (!m_stepsLocks[i].predictors[j].dataId) {
                    if (counter == index) {
                        vwxs vect = GetPredictorDataIdVector(i, j);
                        return (double) (vect.size() - 1);
                    }
                }
                counter++;

                if (!m_stepsLocks[i].predictors[j].level) {
                    if (counter == index) {
                        vf vect = GetPredictorLevelVector(i, j);
                        return (double) (vect.size() - 1);
                    }
                }
                counter++;

                if (!m_stepsLocks[i].predictors[j].timeHours) {
                    if (counter == index) {
                        return (double) GetPredictorTimeHoursUpperLimit(i, j);
                    }
                }
                counter++;
            }

            if (!m_stepsLocks[i].predictors[j].xMin) {
                if (counter == index) {
                    return (double) GetPredictorXminUpperLimit(i, j);
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].xPtsNb) {
                if (counter == index) {
                    return (double) GetPredictorXptsnbUpperLimit(i, j);
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].yMin) {
                if (counter == index) {
                    return (double) GetPredictorYminUpperLimit(i, j);
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].yPtsNb) {
                if (counter == index) {
                    return (double) GetPredictorYptsnbUpperLimit(i, j);
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].weight) {
                if (counter == index) {
                    return (double) GetPredictorWeightUpperLimit(i, j);
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].criteria) {
                if (counter == index) {
                    vwxs vect = GetPredictorCriteriaVector(i, j);
                    return (double) (vect.size() - 1);
                }
            }
            counter++;
        }
    }

    wxASSERT_MSG(counter == m_allParametersCount,
                 wxString::Format("The counter (%d) did not match the number of parameters (%d).", counter,
                                  m_allParametersCount));
    wxASSERT_MSG(counter <= index, "Couldn't access the desired index in the parameters chromosome.");

    return NaNd;
}

double asParametersOptimizationGAs::GetParameterLowerLimit(int index)
{
    int counter = 0;

    if (!m_timeArrayAnalogsIntervalDaysLocks) {
        if (counter == index) {
            return (double) GetTimeArrayAnalogsIntervalDaysLowerLimit();
        }
    }
    counter++;

    for (int i = 0; i < GetStepsNb(); i++) {

        if (!m_stepsLocks[i].analogsNumber) {
            if (counter == index) {
                return (double) GetAnalogsNumberLowerLimit(i);
            }
        }
        counter++;

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (!m_stepsLocks[i].predictors[j].preprocessDataId[k]) {
                        if (counter == index) {
                            return 0.0;
                        }
                    }
                    counter++;

                    if (!m_stepsLocks[i].predictors[j].preprocessLevels[k]) {
                        if (counter == index) {
                            return 0.0;
                        }
                    }
                    counter++;

                    if (!m_stepsLocks[i].predictors[j].preprocessTimeHours[k]) {
                        if (counter == index) {
                            return (double) GetPreprocessTimeHoursLowerLimit(i, j, k);
                        }
                    }
                    counter++;
                }
            } else {
                if (!m_stepsLocks[i].predictors[j].dataId) {
                    if (counter == index) {
                        return 0.0;
                    }
                }
                counter++;

                if (!m_stepsLocks[i].predictors[j].level) {
                    if (counter == index) {
                        return 0.0;
                    }
                }
                counter++;

                if (!m_stepsLocks[i].predictors[j].timeHours) {
                    if (counter == index) {
                        return (double) GetPredictorTimeHoursLowerLimit(i, j);
                    }
                }
                counter++;
            }

            if (!m_stepsLocks[i].predictors[j].xMin) {
                if (counter == index) {
                    return (double) GetPredictorXminLowerLimit(i, j);
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].xPtsNb) {
                if (counter == index) {
                    return (double) GetPredictorXptsnbLowerLimit(i, j);
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].yMin) {
                if (counter == index) {
                    return (double) GetPredictorYminLowerLimit(i, j);
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].yPtsNb) {
                if (counter == index) {
                    return (double) GetPredictorYptsnbLowerLimit(i, j);
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].weight) {
                if (counter == index) {
                    return (double) GetPredictorWeightLowerLimit(i, j);
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].criteria) {
                if (counter == index) {
                    return (double) 0.0;
                }
            }
            counter++;
        }
    }

    wxASSERT_MSG(counter == m_allParametersCount,
                 wxString::Format("The counter (%d) did not match the number of parameters (%d).", counter,
                                  m_allParametersCount));
    wxASSERT_MSG(counter <= index, "Couldn't access the desired index in the parameters chromosome.");

    return NaNd;
}

double asParametersOptimizationGAs::GetParameterIteration(int index)
{
    int counter = 0;

    if (!m_timeArrayAnalogsIntervalDaysLocks) {
        if (counter == index) {
            return (double) GetTimeArrayAnalogsIntervalDaysIteration();
        }
    }
    counter++;

    for (int i = 0; i < GetStepsNb(); i++) {

        if (!m_stepsLocks[i].analogsNumber) {
            if (counter == index) {
                return (double) GetAnalogsNumberIteration(i);
            }
        }
        counter++;

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (!m_stepsLocks[i].predictors[j].preprocessDataId[k]) {
                        if (counter == index) {
                            return 1.0;
                        }
                    }
                    counter++;

                    if (!m_stepsLocks[i].predictors[j].preprocessLevels[k]) {
                        if (counter == index) {
                            return 1.0;
                        }
                    }
                    counter++;

                    if (!m_stepsLocks[i].predictors[j].preprocessTimeHours[k]) {
                        if (counter == index) {
                            return (double) GetPreprocessTimeHoursIteration(i, j, k);
                        }
                    }
                    counter++;
                }
            } else {
                if (!m_stepsLocks[i].predictors[j].dataId) {
                    if (counter == index) {
                        return 1.0;
                    }
                }
                counter++;

                if (!m_stepsLocks[i].predictors[j].level) {
                    if (counter == index) {
                        return 1.0;
                    }
                }
                counter++;

                if (!m_stepsLocks[i].predictors[j].timeHours) {
                    if (counter == index) {
                        return (double) GetPredictorTimeHoursIteration(i, j);
                    }
                }
                counter++;
            }

            if (!m_stepsLocks[i].predictors[j].xMin) {
                if (counter == index) {
                    return (double) GetPredictorXminIteration(i, j);
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].xPtsNb) {
                if (counter == index) {
                    return (double) GetPredictorXptsnbIteration(i, j);
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].yMin) {
                if (counter == index) {
                    return (double) GetPredictorYminIteration(i, j);
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].yPtsNb) {
                if (counter == index) {
                    return (double) GetPredictorYptsnbIteration(i, j);
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].weight) {
                if (counter == index) {
                    return (double) GetPredictorWeightIteration(i, j);
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].criteria) {
                if (counter == index) {
                    return (double) 1.0;
                }
            }
            counter++;
        }
    }

    wxASSERT_MSG(counter == m_allParametersCount,
                 wxString::Format("The counter (%d) did not match the number of parameters (%d).", counter,
                                  m_allParametersCount));
    wxASSERT_MSG(counter <= index, "Couldn't access the desired index in the parameters chromosome.");

    return NaNd;
}

void asParametersOptimizationGAs::SetParameterValue(int index, double newVal)
{
    int counter = 0;

    if (!m_timeArrayAnalogsIntervalDaysLocks) {
        if (counter == index) {
            int val = asRound(newVal);
            if (!SetTimeArrayAnalogsIntervalDays(val)) {
                asThrowException(_("Error in SetTimeArrayAnalogsIntervalDays from GAs."));
            }
            return;
        }
    }
    counter++;

    for (int i = 0; i < GetStepsNb(); i++) {
        if (!m_stepsLocks[i].analogsNumber) {
            if (counter == index) {
                int val = asRound(newVal);
                if (!SetAnalogsNumber(i, val)) {
                    asThrowException(_("Error in SetAnalogsNumber from GAs."));
                }
                return;
            }
        }
        counter++;

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (!m_stepsLocks[i].predictors[j].preprocessDataId[k]) {
                        if (counter == index) {
                            int val = asRound(newVal);

                            vwxs vect = GetPreprocessDataIdVector(i, j, k);
                            if (val < 0)
                                val = 0;
                            if ((unsigned) val >= vect.size())
                                val = vect.size() - 1;

                            if (!SetPreprocessDataId(i, j, k, vect[val])) {
                                asThrowException(_("Error in SetPreprocessDataId from GAs."));
                            }
                            return;
                        }
                    }
                    counter++;

                    if (!m_stepsLocks[i].predictors[j].preprocessLevels[k]) {
                        if (counter == index) {
                            int val = asRound(newVal);

                            vf vect = GetPreprocessLevelVector(i, j, k);
                            if (val < 0)
                                val = 0;
                            if ((unsigned) val >= vect.size())
                                val = vect.size() - 1;

                            if (!SetPreprocessLevel(i, j, k, vect[val])) {
                                asThrowException(_("Error in SetPreprocessLevel from GAs."));
                            }
                            return;
                        }
                    }
                    counter++;

                    if (!m_stepsLocks[i].predictors[j].preprocessTimeHours[k]) {
                        if (counter == index) {
                            int val = asRound(newVal);
                            if (!SetPreprocessTimeHours(i, j, k, val)) {
                                asThrowException(_("Error in SetPreprocessTimeHours from GAs."));
                            }
                            return;
                        }
                    }
                    counter++;
                }
            } else {
                if (!m_stepsLocks[i].predictors[j].dataId) {
                    if (counter == index) {
                        int val = asRound(newVal);

                        vwxs vect = GetPredictorDataIdVector(i, j);
                        if (val < 0)
                            val = 0;
                        if ((unsigned) val >= vect.size())
                            val = vect.size() - 1;

                        if (!SetPredictorDataId(i, j, vect[val])) {
                            asThrowException(_("Error in SetPredictorDataId from GAs."));
                        }
                        return;
                    }
                }
                counter++;

                if (!m_stepsLocks[i].predictors[j].level) {
                    if (counter == index) {
                        int val = asRound(newVal);

                        vf vect = GetPredictorLevelVector(i, j);
                        if (val < 0)
                            val = 0;
                        if ((unsigned) val >= vect.size())
                            val = vect.size() - 1;

                        if (!SetPredictorLevel(i, j, vect[val])) {
                            asThrowException(_("Error in SetPredictorLevel from GAs."));
                        }
                        return;
                    }
                }
                counter++;

                if (!m_stepsLocks[i].predictors[j].timeHours) {
                    if (counter == index) {
                        int val = asRound(newVal);
                        if (!SetPredictorTimeHours(i, j, val)) {
                            asThrowException(_("Error in SetTimeArrayAnalogsIntervalDays from GAs."));
                        }
                        return;
                    }
                }
                counter++;
            }

            if (!m_stepsLocks[i].predictors[j].xMin) {
                if (counter == index) {
                    if (!SetPredictorXmin(i, j, newVal)) {
                        asThrowException(_("Error in SetPredictorXmin from GAs."));
                    }
                    return;
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].xPtsNb) {
                if (counter == index) {
                    int val = asRound(newVal);
                    if (!SetPredictorXptsnb(i, j, val)) {
                        asThrowException(_("Error in SetPredictorXptsnb from GAs."));
                    }
                    return;
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].yMin) {
                if (counter == index) {
                    if (!SetPredictorYmin(i, j, newVal)) {
                        asThrowException(_("Error in SetPredictorYmin from GAs."));
                    }
                    return;
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].yPtsNb) {
                if (counter == index) {
                    int val = asRound(newVal);
                    if (!SetPredictorYptsnb(i, j, val)) {
                        asThrowException(_("Error in SetPredictorYptsnb from GAs."));
                    }
                    return;
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].weight) {
                if (counter == index) {
                    float val = (float) newVal;
                    if (!SetPredictorWeight(i, j, val)) {
                        asThrowException(_("Error in SetPredictorWeight from GAs."));
                    }
                    return;
                }
            }
            counter++;

            if (!m_stepsLocks[i].predictors[j].criteria) {
                if (counter == index) {
                    int val = asRound(newVal);

                    vwxs vect = GetPredictorCriteriaVector(i, j);
                    if (val < 0)
                        val = 0;
                    if ((unsigned) val >= vect.size())
                        val = vect.size() - 1;

                    if (!SetPredictorCriteria(i, j, vect[val])) {
                        asThrowException(_("Error in SetPredictorCriteria from GAs."));
                    }
                    return;
                }
            }
            counter++;
        }
    }

    wxASSERT_MSG(counter == m_allParametersCount,
                 wxString::Format("The counter (%d) did not match the number of parameters (%d).", counter,
                                  m_allParametersCount));
    wxASSERT_MSG(counter <= index, "Couldn't access the desired index in the parameters chromosome.");

    return;
}

void asParametersOptimizationGAs::SimpleCrossover(asParametersOptimizationGAs &otherParam, vi &crossingPoints)
{
    wxASSERT(!crossingPoints.empty());

    // Sort the crossing points vector
    asSortArray(&crossingPoints[0], &crossingPoints[crossingPoints.size() - 1], Asc);

    unsigned int nextpointIndex = 0;
    int nextpoint = m_chromosomeIndices[crossingPoints[nextpointIndex]];
    int counter = 0;
    int counterSelfAdapt = 0;
    bool doCrossing = false;

    do {
        if (!IsParamLocked(counter)) {
            if (counter == nextpoint) {
                doCrossing = !doCrossing;
                if (nextpointIndex < crossingPoints.size() - 1) {
                    nextpointIndex++;
                    nextpoint = m_chromosomeIndices[crossingPoints[nextpointIndex]];
                } else {
                    nextpoint = -1;
                }
            }

            if (doCrossing) {
                double val1 = GetParameterValue(counter);
                double val2 = otherParam.GetParameterValue(counter);
                double newval1 = val2;
                double newval2 = val1;
                SetParameterValue(counter, newval1);
                otherParam.SetParameterValue(counter, newval2);

                // Apply to self-adaptation
                if (m_hasChromosomeSelfAdaptationMutationRate) {
                    float mutRate = GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                    SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt,
                                                                otherParam.GetSelfAdaptationMutationRateFromChromosome(
                                                                        counterSelfAdapt));
                    otherParam.SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, mutRate);
                }
                if (m_hasChromosomeSelfAdaptationMutationRadius) {
                    float mutRadius = GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                    SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt,
                                                                  otherParam.GetSelfAdaptationMutationRadiusFromChromosome(
                                                                          counterSelfAdapt));
                    otherParam.SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, mutRadius);
                }
            }
            counterSelfAdapt++;
        }
        counter++;
    } while (!m_parametersListOver);
}

void asParametersOptimizationGAs::BlendingCrossover(asParametersOptimizationGAs &otherParam, vi &crossingPoints,
                                                    bool shareBeta, double betaMin, double betaMax)
{
    wxASSERT(!crossingPoints.empty());

    // Sort the crossing points vector
    asSortArray(&crossingPoints[0], &crossingPoints[crossingPoints.size() - 1], Asc);

    unsigned int nextpointIndex = 0;
    int nextpoint = m_chromosomeIndices[crossingPoints[nextpointIndex]];
    int counter = 0;
    int counterSelfAdapt = 0;
    bool doCrossing = false;
    double beta = asRandom(betaMin, betaMax);

    do {
        if (!IsParamLocked(counter)) {
            if (counter == nextpoint) {
                doCrossing = !doCrossing;
                if (nextpointIndex < crossingPoints.size() - 1) {
                    nextpointIndex++;
                    nextpoint = m_chromosomeIndices[crossingPoints[nextpointIndex]];
                } else {
                    nextpoint = -1;
                }
            }

            if (doCrossing) {
                if (!shareBeta) {
                    beta = asRandom(betaMin, betaMax);
                }

                double val1 = GetParameterValue(counter);
                double val2 = otherParam.GetParameterValue(counter);
                double newval1 = beta * val1 + (1.0 - beta) * val2;
                double newval2 = (1.0 - beta) * val1 + beta * val2;
                SetParameterValue(counter, newval1);
                otherParam.SetParameterValue(counter, newval2);

                // Apply to self-adaptation
                if (m_hasChromosomeSelfAdaptationMutationRate) {
                    float mutRate1 = GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                    float mutRate2 = otherParam.GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                    float newMutRate1 = beta * mutRate1 + (1.0 - beta) * mutRate2;
                    float newMutRate2 = (1.0 - beta) * mutRate1 + beta * mutRate2;
                    SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, newMutRate1);
                    otherParam.SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, newMutRate2);
                }
                if (m_hasChromosomeSelfAdaptationMutationRadius) {
                    float mutRadius1 = GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                    float mutRadius2 = otherParam.GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                    float newMutRadius1 = beta * mutRadius1 + (1.0 - beta) * mutRadius2;
                    float newMutRadius2 = (1.0 - beta) * mutRadius1 + beta * mutRadius2;
                    SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, newMutRadius1);
                    otherParam.SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, newMutRadius2);
                }
            }
            counterSelfAdapt++;
        }
        counter++;
    } while (!m_parametersListOver);
}

void asParametersOptimizationGAs::HeuristicCrossover(asParametersOptimizationGAs &otherParam, vi &crossingPoints,
                                                     bool shareBeta, double betaMin, double betaMax)
{
    wxASSERT(!crossingPoints.empty());

    // Sort the crossing points vector
    asSortArray(&crossingPoints[0], &crossingPoints[crossingPoints.size() - 1], Asc);

    unsigned int nextpointIndex = 0;
    int nextpoint = m_chromosomeIndices[crossingPoints[nextpointIndex]];
    int counter = 0;
    int counterSelfAdapt = 0;
    bool doCrossing = false;
    double beta = asRandom(betaMin, betaMax);

    do {
        if (!IsParamLocked(counter)) {
            if (counter == nextpoint) {
                doCrossing = !doCrossing;
                if (nextpointIndex < crossingPoints.size() - 1) {
                    nextpointIndex++;
                    nextpoint = m_chromosomeIndices[crossingPoints[nextpointIndex]];
                } else {
                    nextpoint = -1;
                }
            }

            if (doCrossing) {
                if (!shareBeta) {
                    beta = asRandom(betaMin, betaMax);
                }

                double val1 = GetParameterValue(counter);
                double val2 = otherParam.GetParameterValue(counter);
                double newval1 = beta * (val1 - val2) + val1;
                double newval2 = beta * (val2 - val1) + val2;
                SetParameterValue(counter, newval1);
                otherParam.SetParameterValue(counter, newval2);

                // Apply to self-adaptation
                if (m_hasChromosomeSelfAdaptationMutationRate) {
                    float mutRate1 = GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                    float mutRate2 = otherParam.GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                    float newMutRate1 = beta * (mutRate1 - mutRate2) + mutRate1;
                    float newMutRate2 = beta * (mutRate2 - mutRate1) + mutRate2;
                    SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, newMutRate1);
                    otherParam.SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, newMutRate2);
                }
                if (m_hasChromosomeSelfAdaptationMutationRadius) {
                    float mutRadius1 = GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                    float mutRadius2 = otherParam.GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                    float newMutRadius1 = beta * (mutRadius1 - mutRadius2) + mutRadius1;
                    float newMutRadius2 = beta * (mutRadius2 - mutRadius1) + mutRadius2;
                    SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, newMutRadius1);
                    otherParam.SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, newMutRadius2);
                }
            }
            counterSelfAdapt++;
        }
        counter++;
    } while (!m_parametersListOver);

}

void asParametersOptimizationGAs::BinaryLikeCrossover(asParametersOptimizationGAs &otherParam, vi &crossingPoints,
                                                      bool shareBeta, double betaMin, double betaMax)
{
    wxASSERT(!crossingPoints.empty());

    // Sort the crossing points vector
    asSortArray(&crossingPoints[0], &crossingPoints[crossingPoints.size() - 1], Asc);

    unsigned int nextpointIndex = 0;
    int nextpoint = m_chromosomeIndices[crossingPoints[nextpointIndex]];
    int counter = 0;
    int counterSelfAdapt = 0;
    bool doCrossing = false;
    double beta = asRandom(betaMin, betaMax);

    do {
        if (!IsParamLocked(counter)) {
            if (counter == nextpoint) {
                if (!shareBeta) {
                    beta = asRandom(betaMin, betaMax);
                }

                double val1 = GetParameterValue(counter);
                double val2 = otherParam.GetParameterValue(counter);
                double newval1 = val1 - beta * (val1 - val2);
                double newval2 = val2 + beta * (val1 - val2);
                SetParameterValue(counter, newval1);
                otherParam.SetParameterValue(counter, newval2);

                // Apply to self-adaptation
                if (m_hasChromosomeSelfAdaptationMutationRate) {
                    float mutRate1 = GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                    float mutRate2 = otherParam.GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                    float newMutRate1 = mutRate1 - beta * (mutRate1 - mutRate2);
                    float newMutRate2 = mutRate2 + beta * (mutRate1 - mutRate2);
                    SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, newMutRate1);
                    otherParam.SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, newMutRate2);
                }
                if (m_hasChromosomeSelfAdaptationMutationRadius) {
                    float mutRadius1 = GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                    float mutRadius2 = otherParam.GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                    float newMutRadius1 = mutRadius1 - beta * (mutRadius1 - mutRadius2);
                    float newMutRadius2 = mutRadius2 + beta * (mutRadius1 - mutRadius2);
                    SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, newMutRadius1);
                    otherParam.SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, newMutRadius2);
                }

                doCrossing = !doCrossing;
                if (nextpointIndex < crossingPoints.size() - 1) {
                    nextpointIndex++;
                    nextpoint = m_chromosomeIndices[crossingPoints[nextpointIndex]];
                } else {
                    nextpoint = -1;
                }
            } else {
                if (doCrossing) {
                    double val1 = GetParameterValue(counter);
                    double val2 = otherParam.GetParameterValue(counter);
                    double newval1 = val2;
                    double newval2 = val1;
                    SetParameterValue(counter, newval1);
                    otherParam.SetParameterValue(counter, newval2);

                    // Apply to self-adaptation
                    if (m_hasChromosomeSelfAdaptationMutationRate) {
                        float mutRate = GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                        SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt,
                                                                    otherParam.GetSelfAdaptationMutationRateFromChromosome(
                                                                            counterSelfAdapt));
                        otherParam.SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, mutRate);
                    }
                    if (m_hasChromosomeSelfAdaptationMutationRadius) {
                        float mutRadius = GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                        SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt,
                                                                      otherParam.GetSelfAdaptationMutationRadiusFromChromosome(
                                                                              counterSelfAdapt));
                        otherParam.SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, mutRadius);
                    }
                }
            }
            counterSelfAdapt++;
        }
        counter++;
    } while (!m_parametersListOver);

}

void asParametersOptimizationGAs::LinearCrossover(asParametersOptimizationGAs &otherParam,
                                                  asParametersOptimizationGAs &thirdParam, vi &crossingPoints)
{
    wxASSERT(!crossingPoints.empty());

    // Sort the crossing points vector
    asSortArray(&crossingPoints[0], &crossingPoints[crossingPoints.size() - 1], Asc);

    unsigned int nextpointIndex = 0;
    int nextpoint = m_chromosomeIndices[crossingPoints[nextpointIndex]];
    int counter = 0;
    int counterSelfAdapt = 0;
    bool doCrossing = false;

    do {
        if (!IsParamLocked(counter)) {
            if (counter == nextpoint) {
                doCrossing = !doCrossing;
                if (nextpointIndex < crossingPoints.size() - 1) {
                    nextpointIndex++;
                    nextpoint = m_chromosomeIndices[crossingPoints[nextpointIndex]];
                } else {
                    nextpoint = -1;
                }
            }

            if (doCrossing) {
                double val1 = GetParameterValue(counter);
                double val2 = otherParam.GetParameterValue(counter);
                double newval1 = 0.5 * val1 + 0.5 * val2;
                double newval2 = 1.5 * val1 - 0.5 * val2;
                double newval3 = -0.5 * val1 + 1.5 * val2;
                SetParameterValue(counter, newval1);
                otherParam.SetParameterValue(counter, newval2);
                thirdParam.SetParameterValue(counter, newval3);

                // Apply to self-adaptation
                if (m_hasChromosomeSelfAdaptationMutationRate) {
                    float mutRate1 = GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                    float mutRate2 = otherParam.GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                    float newMutRate1 = 0.5 * mutRate1 + 0.5 * mutRate2;
                    float newMutRate2 = 1.5 * mutRate1 - 0.5 * mutRate2;
                    float newMutRate3 = -0.5 * mutRate1 + 1.5 * mutRate2;
                    SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, newMutRate1);
                    otherParam.SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, newMutRate2);
                    thirdParam.SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, newMutRate3);
                }
                if (m_hasChromosomeSelfAdaptationMutationRadius) {
                    float mutRadius1 = GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                    float mutRadius2 = otherParam.GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                    float newMutRadius1 = 0.5 * mutRadius1 + 0.5 * mutRadius2;
                    float newMutRadius2 = 1.5 * mutRadius1 - 0.5 * mutRadius2;
                    float newMutRadius3 = -0.5 * mutRadius1 + 1.5 * mutRadius2;
                    SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, newMutRadius1);
                    otherParam.SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, newMutRadius2);
                    thirdParam.SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, newMutRadius3);
                }
            }
            counterSelfAdapt++;
        }
        counter++;
    } while (!m_parametersListOver);

}

void asParametersOptimizationGAs::LinearInterpolation(asParametersOptimizationGAs &otherParam, bool shareBeta)
{
    double beta = asRandom(0.0, 1.0);
    int counterSelfAdapt = 0;
    int counter = 0;

    do {
        if (!IsParamLocked(counter)) {
            if (!shareBeta) {
                beta = asRandom(0.0, 1.0);
            }

            double val1 = GetParameterValue(counter);
            double val2 = otherParam.GetParameterValue(counter);
            double newval1 = val1 - beta * (val1 - val2);
            double newval2 = val2 + beta * (val1 - val2);
            SetParameterValue(counter, newval1);
            otherParam.SetParameterValue(counter, newval2);

            // Apply to self-adaptation
            if (m_hasChromosomeSelfAdaptationMutationRate) {
                float mutRate1 = GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                float mutRate2 = otherParam.GetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt);
                float newMutRate1 = mutRate1 - beta * (mutRate1 - mutRate2);
                float newMutRate2 = mutRate2 + beta * (mutRate1 - mutRate2);
                SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, newMutRate1);
                otherParam.SetSelfAdaptationMutationRateFromChromosome(counterSelfAdapt, newMutRate2);
            }
            if (m_hasChromosomeSelfAdaptationMutationRadius) {
                float mutRadius1 = GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                float mutRadius2 = otherParam.GetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt);
                float newMutRadius1 = mutRadius1 - beta * (mutRadius1 - mutRadius2);
                float newMutRadius2 = mutRadius2 + beta * (mutRadius1 - mutRadius2);
                SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, newMutRadius1);
                otherParam.SetSelfAdaptationMutationRadiusFromChromosome(counterSelfAdapt, newMutRadius2);
            }
            counterSelfAdapt++;
        }
        counter++;
    } while (!m_parametersListOver);
}

void asParametersOptimizationGAs::MutateUniformDistribution(double probability, bool &hasMutated)
{
    int counter = 0;

    do {
        if (!IsParamLocked(counter)) {
            if (asRandom(0.0, 1.0) < probability) {
                double newVal = asRandom(GetParameterLowerLimit(counter), GetParameterUpperLimit(counter),
                                                GetParameterIteration(counter));
                SetParameterValue(counter, newVal);

                hasMutated = true;
            }
        }
        counter++;
    } while (!m_parametersListOver);
}

void asParametersOptimizationGAs::MutateNormalDistribution(double probability, double stdDevRatioRange,
                                                           bool &hasMutated)
{
    int counter = 0;

    do {
        if (!IsParamLocked(counter)) {
            if (asRandom(0.0, 1.0) < probability) {
                if (GetParamType(counter) < 3) {
                    double mean = GetParameterValue(counter);
                    double stdDev =
                            stdDevRatioRange * (GetParameterUpperLimit(counter) - GetParameterLowerLimit(counter));
                    double step = GetParameterIteration(counter);
                    double newVal = asRandomNormal(mean, stdDev, step);
                    SetParameterValue(counter, newVal);
                } else {
                    // Uniform distribution in the case of parameters as a list
                    double newVal = asRandom(GetParameterLowerLimit(counter), GetParameterUpperLimit(counter),
                                                    GetParameterIteration(counter));
                    SetParameterValue(counter, newVal);
                }

                hasMutated = true;
            }
        }
        counter++;
    } while (!m_parametersListOver);
}

void asParametersOptimizationGAs::MutateNonUniform(double probability, int nbGen, int nbGenMax, double minRate,
                                                   bool &hasMutated)
{
    double ratioGens = (double) nbGen / (double) nbGenMax;
    double cstFactor =
            (1.0 - wxMin(ratioGens, 1.0) * (1.0 - minRate)) * (1.0 - wxMin(ratioGens, 1.0) * (1.0 - minRate));

    int counter = 0;

    do {
        if (!IsParamLocked(counter)) {
            if (asRandom(0.0, 1.0) < probability) {
                if (GetParamType(counter) < 3) {
                    double r1 = asRandom(0.0, 1.0);
                    double r2 = asRandom(0.0, 1.0);
                    double actVal = GetParameterValue(counter);
                    double lowerLimit = GetParameterLowerLimit(counter);
                    double upperLimit = GetParameterUpperLimit(counter);
                    double newVal;

                    if (r1 < 0.5) {
                        newVal = actVal + (upperLimit - actVal) * r2 * cstFactor;
                    } else {
                        newVal = actVal - (actVal - lowerLimit) * r2 * cstFactor;
                    }

                    SetParameterValue(counter, newVal);
                } else {
                    // Uniform distribution in the case of parameters as a list
                    double newVal = asRandom(GetParameterLowerLimit(counter), GetParameterUpperLimit(counter),
                                                    GetParameterIteration(counter));
                    SetParameterValue(counter, newVal);
                }

                hasMutated = true;
            }
        }
        counter++;
    } while (!m_parametersListOver);
}

void asParametersOptimizationGAs::MutateSelfAdaptationRate(bool &hasMutated)
{
    // Mutate mutation probability
    if (asRandom(0.0, 1.0) < m_individualSelfAdaptationMutationRate) {
        m_individualSelfAdaptationMutationRate = asRandom(0.0, 1.0);
    }

    // Mutate data
    MutateUniformDistribution(m_individualSelfAdaptationMutationRate, hasMutated);

}

void asParametersOptimizationGAs::MutateSelfAdaptationRadius(bool &hasMutated)
{
    // Mutate mutation probability
    if (asRandom(0.0, 1.0) < m_individualSelfAdaptationMutationRate) {
        m_individualSelfAdaptationMutationRate = asRandom(0.0, 1.0);
    }

    // Mutate mutation radius. Use the radius here as a probability !!
    if (asRandom(0.0, 1.0) < m_individualSelfAdaptationMutationRadius) {
        m_individualSelfAdaptationMutationRadius = asRandom(0.0, 1.0);
    }

    // Mutate data
    int counter = 0;
    do {
        if (!IsParamLocked(counter)) {
            if (asRandom(0.0, 1.0) < m_individualSelfAdaptationMutationRate) {
                if (GetParamType(counter) < 3) {
                    double r1 = asRandom(0.0, 1.0);
                    double r2 = asRandom(0.0, 1.0);
                    double actVal = GetParameterValue(counter);
                    double lowerLimit = GetParameterLowerLimit(counter);
                    double upperLimit = GetParameterUpperLimit(counter);
                    double newVal;

                    if (r1 < 0.5) {
                        newVal = actVal + (upperLimit - actVal) * r2 * m_individualSelfAdaptationMutationRadius;
                    } else {
                        newVal = actVal - (actVal - lowerLimit) * r2 * m_individualSelfAdaptationMutationRadius;
                    }

                    SetParameterValue(counter, newVal);
                } else {
                    // Uniform distribution in the case of parameters as a list
                    double newVal = asRandom(GetParameterLowerLimit(counter), GetParameterUpperLimit(counter),
                                                    GetParameterIteration(counter));
                    SetParameterValue(counter, newVal);
                }

                hasMutated = true;
            }
        }
        counter++;
    } while (!m_parametersListOver);
}

void asParametersOptimizationGAs::MutateSelfAdaptationRateChromosome(bool &hasMutated)
{
    wxASSERT(!m_chromosomeSelfAdaptationMutationRate.empty());
    wxASSERT(m_chromosomeSelfAdaptationMutationRate.size() == GetChromosomeLength());

    // Mutate mutation probability
    for (unsigned int i = 0; i < m_chromosomeSelfAdaptationMutationRate.size(); i++) {
        if (asRandom(0.0, 1.0) < m_chromosomeSelfAdaptationMutationRate[i]) {
            m_chromosomeSelfAdaptationMutationRate[i] = asRandom(0.0, 1.0);
        }

    }

    // Mutate data
    int counter = 0;
    int counterSelfAdapt = 0;
    do {
        if (!IsParamLocked(counter)) {
            wxASSERT(counterSelfAdapt < m_chromosomeSelfAdaptationMutationRate.size());

            if (asRandom(0.0, 1.0) < m_chromosomeSelfAdaptationMutationRate[counterSelfAdapt]) {
                // Uniform distribution in the case of parameters as a list
                double newVal = asRandom(GetParameterLowerLimit(counter), GetParameterUpperLimit(counter),
                                                GetParameterIteration(counter));
                SetParameterValue(counter, newVal);

                hasMutated = true;
            }
            counterSelfAdapt++;
        }
        counter++;
    } while (!m_parametersListOver);
}

void asParametersOptimizationGAs::MutateSelfAdaptationRadiusChromosome(bool &hasMutated)
{
    wxASSERT(!m_chromosomeSelfAdaptationMutationRate.empty());
    wxASSERT(!m_chromosomeSelfAdaptationMutationRadius.empty());
    wxASSERT(m_chromosomeSelfAdaptationMutationRate.size() == m_chromosomeSelfAdaptationMutationRadius.size());
    wxASSERT(m_chromosomeSelfAdaptationMutationRate.size() == GetChromosomeLength());

    // Mutate mutation probability
    for (unsigned int i = 0; i < m_chromosomeSelfAdaptationMutationRate.size(); i++) {
        if (asRandom(0.0, 1.0) < m_chromosomeSelfAdaptationMutationRate[i]) {
            m_chromosomeSelfAdaptationMutationRate[i] = asRandom(0.0, 1.0);
        }

    }

    // Mutate mutation radius. Use the radius here as a probability !!
    for (unsigned int i = 0; i < m_chromosomeSelfAdaptationMutationRadius.size(); i++) {
        if (asRandom(0.0, 1.0) < m_chromosomeSelfAdaptationMutationRadius[i]) {
            m_chromosomeSelfAdaptationMutationRadius[i] = asRandom(0.0, 1.0);
        }

    }

    // Mutate data
    int counter = 0;
    int counterSelfAdapt = 0;
    do {
        if (!IsParamLocked(counter)) {
            wxASSERT(counterSelfAdapt < m_chromosomeSelfAdaptationMutationRate.size());

            if (asRandom(0.0, 1.0) < m_chromosomeSelfAdaptationMutationRate[counterSelfAdapt]) {
                if (GetParamType(counter) < 3) {
                    double r1 = asRandom(0.0, 1.0);
                    double r2 = asRandom(0.0, 1.0);
                    double actVal = GetParameterValue(counter);
                    double lowerLimit = GetParameterLowerLimit(counter);
                    double upperLimit = GetParameterUpperLimit(counter);
                    double newVal;

                    if (r1 < 0.5) {
                        newVal = actVal + (upperLimit - actVal) * r2 *
                                          m_chromosomeSelfAdaptationMutationRadius[counterSelfAdapt];
                    } else {
                        newVal = actVal - (actVal - lowerLimit) * r2 *
                                          m_chromosomeSelfAdaptationMutationRadius[counterSelfAdapt];
                    }

                    SetParameterValue(counter, newVal);
                } else {
                    // Uniform distribution in the case of parameters as a list
                    double newVal = asRandom(GetParameterLowerLimit(counter), GetParameterUpperLimit(counter),
                                                    GetParameterIteration(counter));
                    SetParameterValue(counter, newVal);
                }

                hasMutated = true;
            }
            counterSelfAdapt++;
        }
        counter++;
    } while (!m_parametersListOver);
}

void asParametersOptimizationGAs::MutateMultiScale(double probability, bool &hasMutated)
{
    // Choose the radius
    double radiusChoice = asRandom(0.0, 1.0);
    double radius = 0;

    if (radiusChoice < 0.25) {
        radius = 1;
    } else if (radiusChoice >= 0.25 && radiusChoice < 0.50) {
        radius = 0.5;
    } else if (radiusChoice >= 0.50 && radiusChoice < 0.75) {
        radius = 0.1;
    } else {
        radius = 0.02;
    }

    // Mutate data
    int counter = 0;
    do {
        if (!IsParamLocked(counter)) {
            if (asRandom(0.0, 1.0) < probability) {
                if (GetParamType(counter) < 3) {
                    double r1 = asRandom(0.0, 1.0);
                    double r2 = asRandom(0.0, 1.0);
                    double actVal = GetParameterValue(counter);
                    double lowerLimit = GetParameterLowerLimit(counter);
                    double upperLimit = GetParameterUpperLimit(counter);
                    double newVal;

                    if (r1 < 0.5) {
                        newVal = actVal + (upperLimit - actVal) * r2 * radius;
                    } else {
                        newVal = actVal - (actVal - lowerLimit) * r2 * radius;
                    }

                    SetParameterValue(counter, newVal);
                } else {
                    // Uniform distribution in the case of parameters as a list
                    double newVal = asRandom(GetParameterLowerLimit(counter), GetParameterUpperLimit(counter),
                                                    GetParameterIteration(counter));
                    SetParameterValue(counter, newVal);
                }

                hasMutated = true;
            }
        }
        counter++;
    } while (!m_parametersListOver);
}
