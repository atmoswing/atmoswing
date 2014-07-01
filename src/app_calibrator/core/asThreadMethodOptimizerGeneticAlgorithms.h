#ifndef ASTHREADMETHODOPTIMIZERGENETICALGORITHMS_H
#define ASTHREADMETHODOPTIMIZERGENETICALGORITHMS_H

#include <asThread.h>
#include <asParametersOptimization.h>
#include <asMethodOptimizerGeneticAlgorithms.h>
#include <asIncludes.h>

class asThreadMethodOptimizerGeneticAlgorithms: public asThread
{
public:
    /** Default constructor */
    asThreadMethodOptimizerGeneticAlgorithms(asMethodOptimizerGeneticAlgorithms* optimizer, const asParametersOptimization &params, float *finalScoreCalib, VectorFloat *scoreClimatology);
    /** Default destructor */
    virtual ~asThreadMethodOptimizerGeneticAlgorithms();

    ExitCode Entry();

protected:
private:
    asMethodOptimizerGeneticAlgorithms* m_Optimizer;
    asParametersOptimization m_Params;
    float* m_FinalScoreCalib;
    VectorFloat* m_ScoreClimatology;
};

#endif // ASTHREADMETHODOPTIMIZERGENETICALGORITHMS_H
