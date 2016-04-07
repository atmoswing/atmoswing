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

    void SetPredictandStationIds(VectorInt val)
    {
        m_predictandStationIds = val;
    }

protected:
    VectorInt m_predictandStationIds;
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

    bool Validate(asParametersOptimization *params);

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