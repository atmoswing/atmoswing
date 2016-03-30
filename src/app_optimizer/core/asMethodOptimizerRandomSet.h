#ifndef ASMETHODOPTIMIZERRANDOMSET_H
#define ASMETHODOPTIMIZERRANDOMSET_H

#include <asMethodOptimizer.h>


class asMethodOptimizerRandomSet
        : public asMethodOptimizer
{
public:
    asMethodOptimizerRandomSet();

    virtual ~asMethodOptimizerRandomSet();

    bool Manager();

protected:
    virtual void InitParameters(asParametersOptimization &params);

    virtual asParametersOptimization GetNextParameters();

    virtual bool Optimize(asParametersOptimization &params);

private:
    std::vector<asParametersOptimization> m_parameters;
    asParametersOptimization m_originalParams;

};

#endif // ASMETHODOPTIMIZERRANDOMSET_H
