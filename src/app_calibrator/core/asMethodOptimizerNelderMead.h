#ifndef ASMETHODOPTIMIZERNELDERMEAD_H
#define ASMETHODOPTIMIZERNELDERMEAD_H

#include <asMethodOptimizer.h>
#include <asParametersOptimizationNelderMead.h>


class asMethodOptimizerNelderMead: public asMethodOptimizer
{
public:
    asMethodOptimizerNelderMead();
    virtual ~asMethodOptimizerNelderMead();
    bool Manager();
    bool ManageOneRun();

protected:

private:
    std::vector <asParametersOptimizationNelderMead> m_Parameters;
    std::vector <asParametersOptimizationNelderMead> m_ParametersTemp;
    asParametersOptimizationNelderMead m_OriginalParams;
    float m_NelderMeadRho;
    float m_NelderMeadChi;
    float m_NelderMeadGamma;
    float m_NelderMeadSigma;

    void ClearAll();
    void ClearTemp();
    void SortScoresAndParameters();
    void SortScoresAndParametersTemp();
    bool SetBestParameters(asResultsParametersArray &results);
    void InitParameters(asParametersOptimizationNelderMead &params);
    asParametersOptimizationNelderMead GetNextParameters();
    bool Optimize(asParametersOptimizationNelderMead &params);

};

#endif // ASMETHODOPTIMIZERNELDERMEAD_H
