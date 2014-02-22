#ifndef ASTHREADMETHODOPTIMIZERRANDOMSET_H
#define ASTHREADMETHODOPTIMIZERRANDOMSET_H

#include <asThread.h>
#include <asParametersOptimization.h>
#include <asMethodOptimizerRandomSet.h>
#include <asIncludes.h>

class asThreadMethodOptimizerRandomSet: public asThread
{
public:
    /** Default constructor */
    asThreadMethodOptimizerRandomSet(const asMethodOptimizerRandomSet* optimizer, const asParametersOptimization &params, float *finalScoreCalib, float *scoreClimatology = 0);
    /** Default destructor */
    virtual ~asThreadMethodOptimizerRandomSet();

    ExitCode Entry();

protected:
private:
    asMethodOptimizerRandomSet m_Optimizer;
    asParametersOptimization m_Params;
    float* m_FinalScoreCalib;
    float* m_ScoreClimatology;
};

#endif // ASTHREADMETHODOPTIMIZERRANDOMSET_H
