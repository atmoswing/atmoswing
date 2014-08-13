#ifndef ASMETHODOPTIMIZERGeneticAlgorithms_H
#define ASMETHODOPTIMIZERGeneticAlgorithms_H

#include <asMethodOptimizer.h>
#include <asParametersOptimizationGAs.h>


class asMethodOptimizerGeneticAlgorithms: public asMethodOptimizer
{
public:
    enum NaturalSelectionType //!< Enumaration of natural selection options
    {
        RatioElitism,
        Tournament
    };
    enum CouplesSelectionType //!< Enumaration of couples selection options
    {
        RankPairing,
        Random,
        RouletteWheelRank,
        RouletteWheelScore,
        TournamentCompetition
    };
    enum CrossoverType //!< Enumaration of natural selection options
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
    enum MutationsModeType //!< Enumaration of natural selection options
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
    std::vector <asParametersOptimizationGAs> m_Parameters;
    std::vector <asParametersOptimizationGAs> m_ParametersTemp;
    asParametersOptimizationGAs m_OriginalParams;
    int m_GenerationNb;
    int m_AssessmentCounter;
    int m_PopSize;
    int m_NaturalSelectionType;
    int m_CouplesSelectionType;
    int m_CrossoverType;
    int m_MutationsModeType;
    bool m_AllowElitismForTheBest;
    VectorFloat m_BestScores;
    VectorFloat m_MeanScores;

    void ClearAll();
    void ClearTemp();
    void SortScoresAndParameters();
    void SortScoresAndParametersTemp();
    void SortParamsLevelsAndTime();
    bool ResumePreviousRun(asResultsParametersArray &results_generations);
    bool SetBestParameters(asResultsParametersArray &results);
    void InitParameters(asParametersOptimizationGAs &params);
	asParametersOptimizationGAs GetNextParameters();
	bool Optimize(asParametersOptimizationGAs &params);
	bool CheckConvergence(bool &stop);
	bool ElitismAfterSelection();
	bool ElitismAfterMutation();
	bool NaturalSelection();
    bool Mating();
    bool Mutatation();

};

#endif // ASMETHODOPTIMIZERGeneticAlgorithms_H
