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
    asThreadMethodOptimizerRandomSet(const asMethodOptimizerRandomSet* optimizer, const asParametersOptimization &params, float *finalScoreCalib, VectorFloat *scoreClimatology);
    /** Default destructor */
    virtual ~asThreadMethodOptimizerRandomSet();

    ExitCode Entry();

protected:
private:
    asMethodOptimizerRandomSet m_optimizer;
    asParametersOptimization m_params;
    float* m_finalScoreCalib;
    VectorFloat* m_scoreClimatology;
};

#endif // ASTHREADMETHODOPTIMIZERRANDOMSET_H
