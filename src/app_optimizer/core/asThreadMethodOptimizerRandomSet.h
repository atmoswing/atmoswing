#ifndef ASTHREADMETHODOPTIMIZERRANDOMSET_H
#define ASTHREADMETHODOPTIMIZERRANDOMSET_H

#include <asThread.h>
#include <asParametersOptimization.h>
#include <asMethodOptimizerRandomSet.h>
#include <asIncludes.h>

class asThreadMethodOptimizerRandomSet
        : public asThread
{
public:
    asThreadMethodOptimizerRandomSet(asMethodOptimizerRandomSet *optimizer,
                                     asParametersOptimization *params, float *finalScoreCalib,
                                     vf *scoreClimatology);

    virtual ~asThreadMethodOptimizerRandomSet();

    ExitCode Entry();

protected:
private:
    asMethodOptimizerRandomSet *m_optimizer;
    asParametersOptimization *m_params;
    float *m_finalScoreCalib;
    vf *m_scoreClimatology;
};

#endif // ASTHREADMETHODOPTIMIZERRANDOMSET_H
