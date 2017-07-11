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

#ifndef ASPARAMETERSOPTIMIZATIONGAS_H
#define ASPARAMETERSOPTIMIZATIONGAS_H

#include "asIncludes.h"
#include <asParametersOptimization.h>

class asFileParametersOptimization;


class asParametersOptimizationGAs
        : public asParametersOptimization
{
public:
    asParametersOptimizationGAs();

    virtual ~asParametersOptimizationGAs();

    void BuildChromosomes();

    void InitIndividualSelfAdaptationMutationRate();

    void InitIndividualSelfAdaptationMutationRadius();

    void InitChromosomeSelfAdaptationMutationRate();

    void InitChromosomeSelfAdaptationMutationRadius();

    void SimpleCrossover(asParametersOptimizationGAs &otherParam, vi &crossingPoints);

    void BlendingCrossover(asParametersOptimizationGAs &otherParam, vi &crossingPoints, bool shareBeta,
                           double betaMin = 0.0, double betaMax = 1.0);

    void HeuristicCrossover(asParametersOptimizationGAs &otherParam, vi &crossingPoints, bool shareBeta,
                            double betaMin = 0.0, double betaMax = 1.0);

    void BinaryLikeCrossover(asParametersOptimizationGAs &otherParam, vi &crossingPoints, bool shareBeta,
                             double betaMin = 0.0, double betaMax = 1.0);

    void LinearCrossover(asParametersOptimizationGAs &otherParam, asParametersOptimizationGAs &thirdParam,
                         vi &crossingPoints);

    void LinearInterpolation(asParametersOptimizationGAs &otherParam, bool shareBeta);

    void MutateUniformDistribution(double probability, bool &hasMutated);

    void MutateNormalDistribution(double probability, double stdDevRatioRange, bool &hasMutated);

    void MutateNonUniform(double probability, int nbGen, int nbGenMax, double minRate, bool &hasMutated);

    void MutateSelfAdaptationRate(bool &hasMutated);

    void MutateSelfAdaptationRadius(bool &hasMutated);

    void MutateSelfAdaptationRateChromosome(bool &hasMutated);

    void MutateSelfAdaptationRadiusChromosome(bool &hasMutated);

    void MutateMultiScale(double probability, bool &hasMutated);

    int GetChromosomeLength()
    {
        return (int) m_chromosomeIndices.size();
    }

protected:

private:
    float m_individualSelfAdaptationMutationRate;
    float m_individualSelfAdaptationMutationRadius;
    vi m_chromosomeIndices;
    vf m_chromosomeSelfAdaptationMutationRate;
    vf m_chromosomeSelfAdaptationMutationRadius;
    bool m_hasChromosomeSelfAdaptationMutationRate;
    bool m_hasChromosomeSelfAdaptationMutationRadius;
    int m_timeArrayAnalogsIntervalDaysIteration;
    int m_timeArrayAnalogsIntervalDaysUpperLimit;
    int m_timeArrayAnalogsIntervalDaysLowerLimit;
    bool m_timeArrayAnalogsIntervalDaysLocks;
    int m_allParametersCount;
    bool m_parametersListOver;

    bool IsParamLocked(int index);

    /** Get the parameter type (list of value vs value)
     * \param index The index in the chromosome
     * \return A code: 1 for value - 2 for advanced list (notion of proximity) - 3 for simple list (no proximity between elements)
     */
    int GetParamType(int index);

    double GetParameterValue(int index);

    void SetParameterValue(int index, double newVal);

    double GetParameterLowerLimit(int index);

    double GetParameterUpperLimit(int index);

    double GetParameterIteration(int index);

    float GetSelfAdaptationMutationRateFromChromosome(int index)
    {
        wxASSERT(m_chromosomeSelfAdaptationMutationRate.size() > (unsigned) index);
        return m_chromosomeSelfAdaptationMutationRate[index];
    }

    void SetSelfAdaptationMutationRateFromChromosome(int index, float val)
    {
        wxASSERT(m_chromosomeSelfAdaptationMutationRate.size() > (unsigned) index);
        m_chromosomeSelfAdaptationMutationRate[index] = val;
    }

    float GetSelfAdaptationMutationRadiusFromChromosome(int index)
    {
        wxASSERT(m_chromosomeSelfAdaptationMutationRadius.size() > (unsigned) index);
        return m_chromosomeSelfAdaptationMutationRadius[index];
    }

    void SetSelfAdaptationMutationRadiusFromChromosome(int index, float val)
    {
        wxASSERT(m_chromosomeSelfAdaptationMutationRadius.size() > (unsigned) index);
        m_chromosomeSelfAdaptationMutationRadius[index] = val;
    }
};

#endif // asParametersOptimizationGAS_H
