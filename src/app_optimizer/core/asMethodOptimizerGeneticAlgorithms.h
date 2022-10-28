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

#ifndef AS_METHOD_OPTIMIZER_GENETIC_ALGORITHMS_H
#define AS_METHOD_OPTIMIZER_GENETIC_ALGORITHMS_H

#include "asMethodOptimizer.h"
#include "asParametersOptimizationGAs.h"

class asMethodOptimizerGeneticAlgorithms : public asMethodOptimizer {
  public:
    enum NaturalSelectionType {
        RatioElitism,
        Tournament
    };
    enum CouplesSelectionType {
        RankPairing,
        Random,
        RouletteWheelRank,
        RouletteWheelScore,
        TournamentCompetition
    };
    enum CrossoverType {
        SinglePointCrossover,
        DoublePointsCrossover,
        MultiplePointsCrossover,
        UniformCrossover,
        LimitedBlending,
        LinearCrossover,
        HeuristicCrossover,
        BinaryLikeCrossover,
        LinearInterpolation,
        FreeInterpolation
    };
    enum MutationsModeType {
        RandomUniformConstant,
        RandomUniformVariable,
        RandomNormalConstant,
        RandomNormalVariable,
        NonUniform,
        SelfAdaptationRate,
        SelfAdaptationRadius,
        SelfAdaptationRateChromosome,
        SelfAdaptationRadiusChromosome,
        MultiScale,
        NoMutation
    };

    asMethodOptimizerGeneticAlgorithms();

    ~asMethodOptimizerGeneticAlgorithms() override;

    bool Manager() override;

    bool ManageOneRun();

  protected:

  private:
    std::vector<asParametersOptimizationGAs> m_parameters;
    asParametersOptimizationGAs m_parameterBest;
    float m_scoreCalibBest;
    asResultsParametersArray m_resGenerations;
    int m_generationNb;
    int m_assessmentCounter;
    int m_popSize;
    int m_naturalSelectionType;
    int m_couplesSelectionType;
    int m_crossoverType;
    int m_mutationsModeType;
    bool m_allowElitismForTheBest;
    int m_epochMax;
    vf m_bestScores;
    vf m_meanScores;

    void ClearAll() override;

    void SortScoresAndParameters();

    bool SetBestParameters(asResultsParametersArray& results) override;

    float ComputeScoreFullPeriod(asParametersOptimizationGAs& param);

    bool ResumePreviousRun(asParametersOptimizationGAs& params, const wxString& operatorsFilePath);

    bool HasPreviousRunConverged(asParametersOptimizationGAs& params);

    bool SaveOperators(const wxString& filePath);

    void InitParameters(asParametersOptimizationGAs& params);

    asParametersOptimizationGAs* GetNextParameters();

    bool Optimize();

    bool HasConverged();

    bool ElitismAfterSelection();

    bool ElitismAfterMutation();

    bool NaturalSelection();

    bool Mating();

    bool Mutation();
};

#endif
