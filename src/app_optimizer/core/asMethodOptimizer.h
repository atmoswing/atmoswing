#ifndef ASMETHODOPTIMIZER_H
#define ASMETHODOPTIMIZER_H

#include "asIncludes.h"
#include <asMethodCalibrator.h>
#include <asParametersOptimization.h>


class asMethodOptimizer
        : public asMethodCalibrator
{
public:
    asMethodOptimizer();

    virtual ~asMethodOptimizer();

    virtual bool Manager() = 0;

protected:
    bool m_isOver;
    bool m_skipNext;
    int m_optimizerStage;
    int m_paramsNb;
    int m_iterator;

    virtual bool Calibrate(asParametersCalibration &params)
    {
        asLogError(_("asMethodOptimizer do optimize, not calibrate..."));
        return false;
    }

    bool SaveDetails(asParametersOptimization &params);

    bool Validate(asParametersOptimization &params);

    void IncrementIterator()
    {
        m_iterator++;
    }

    bool IsOver() const
    {
        return m_isOver;
    }

    bool SkipNext() const
    {
        return m_skipNext;
    }

private:

};

#endif // ASMETHODOPTIMIZER_H
