#ifndef ASPARAMETERSOPTIMIZATIONNELDERMEAD_H
#define ASPARAMETERSOPTIMIZATIONNELDERMEAD_H

#include "asIncludes.h"
#include <asParametersOptimization.h>

class asFileParametersOptimization;


class asParametersOptimizationNelderMead : public asParametersOptimization
{
public:

    asParametersOptimizationNelderMead();
    virtual ~asParametersOptimizationNelderMead();

    void GeometricTransform(asParametersOptimizationNelderMead &refParams, float coefficient);

    void Reduction(asParametersOptimizationNelderMead &refParams, float sigma);

    void SetMeans(std::vector <asParametersOptimizationNelderMead> &vParameters, int elementsNb);

protected:

private:

};

#endif // ASPARAMETERSOPTIMIZATIONNELDERMEAD_H
