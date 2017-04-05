#ifndef ASMETHODOPTIMIZERGeneticAlgorithms_H
#define ASMETHODOPTIMIZERGeneticAlgorithms_H

#include <asMethodOptimizer.h>
#include <asParametersOptimizationGAs.h>


class asMethodOptimizerGeneticAlgorithms
        : public asMethodOptimizer
{
public:
    enum NaturalSelectionType
    {
        RatioElitism, Tournament
    };
    enum CouplesSelectionType
    {
        RankPairing, Random, RouletteWheelRank, RouletteWheelScore, TournamentCompetition
    };
    enum CrossoverType
    {
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
    enum MutationsModeType
    {
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

    virtual ~asMethodOptimizerGeneticAlgorithms();

    bool Manager();

    bool ManageOneRun();

protected:

private:
    std::vector<asParametersOptimizationGAs> m_parameters;
    std::vector<asParametersOptimizationGAs> m_parametersTemp;
    asParametersOptimizationGAs m_originalParams;
    int m_generationNb;
    int m_assessmentCounter;
    int m_popSize;
    int m_naturalSelectionType;
    int m_couplesSelectionType;
    int m_crossoverType;
    int m_mutationsModeType;
    bool m_allowElitismForTheBest;
    vf m_bestScores;
    vf m_meanScores;

    void ClearAll();

    void ClearTemp();

    void SortScoresAndParameters();

    void SortScoresAndParametersTemp();

    bool SetBestParameters(asResultsParametersArray &results);

    bool ResumePreviousRun(asParametersOptimizationGAs &params, asResultsParametersArray &results_generations);

    void InitParameters(asParametersOptimizationGAs &params);

    asParametersOptimizationGAs *GetNextParameters();

    bool Optimize();

    bool CheckConvergence(bool &stop);

    bool ElitismAfterSelection();

    bool ElitismAfterMutation();

    bool NaturalSelection();

    bool Mating();

    bool Mutation();

};

#endif // ASMETHODOPTIMIZERGeneticAlgorithms_H
