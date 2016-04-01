#ifndef ASTHREADMETHODOPTIMIZERGENETICALGORITHMS_H
#define ASTHREADMETHODOPTIMIZERGENETICALGORITHMS_H

#include <asThread.h>
#include <asParametersOptimization.h>
#include <asMethodOptimizerGeneticAlgorithms.h>
#include <asIncludes.h>

class asThreadMethodOptimizerGeneticAlgorithms
        : public asThread
{
public:
    asThreadMethodOptimizerGeneticAlgorithms(asMethodOptimizerGeneticAlgorithms *optimizer,
                                             asParametersOptimization *params, float *finalScoreCalib,
                                             VectorFloat *scoreClimatology);

    virtual ~asThreadMethodOptimizerGeneticAlgorithms();

    ExitCode Entry();

protected:
private:
    asMethodOptimizerGeneticAlgorithms *m_optimizer;
    asParametersOptimization *m_params;
    float *m_finalScoreCalib;
    VectorFloat *m_scoreClimatology;
};

#endif // ASTHREADMETHODOPTIMIZERGENETICALGORITHMS_H
