#ifndef ASTHREADPRELOADDATA_H
#define ASTHREADPRELOADDATA_H

#include <asThread.h>
#include <asIncludes.h>
#include "asMethodCalibrator.h"

class asThreadPreloadData
        : public asThread
{
public:
    asThreadPreloadData(asMethodCalibrator *optimizer, asParametersScoring &params, int iStep, int iPtor, int iPre);

    virtual ~asThreadPreloadData();

    ExitCode Entry();

protected:
private:
    asMethodCalibrator *m_optimizer;
    asParametersScoring m_params;
    int m_iStep;
    int m_iProt;
    int m_iDat;
};

#endif // ASTHREADPRELOADDATA_H
