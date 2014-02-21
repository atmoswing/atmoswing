#ifndef ASMETHODOPTIMIZER_H
#define ASMETHODOPTIMIZER_H

#include "asIncludes.h"
#include <asMethodCalibrator.h>
#include <asParametersOptimization.h>


class asMethodOptimizer: public asMethodCalibrator
{
public:
    asMethodOptimizer();
    virtual ~asMethodOptimizer();

    bool Manager() = 0;

    void SetPredictandStationId(int val)
    {
        m_PredictandStationId = val;
    }


protected:
    int m_PredictandStationId;
    bool m_IsOver;
    bool m_SkipNext;
    int m_OptimizerStage;
    int m_ParamsNb;
    int m_Iterator;


//	virtual void InitParameters(asParametersOptimization &params) = 0;
//	virtual asParametersOptimization GetNextParameters() = 0;
//	virtual bool Optimize(asParametersOptimization &params) = 0;
	virtual bool Calibrate(asParametersCalibration &params)
	{
	    asLogError(_("asMethodOptimizer do optimize, not calibrate..."));
	    return false;
	}

	bool Validate(asParametersOptimization* params);

	void IncrementIterator()
    {
        m_Iterator++;
    }

	bool IsOver()
    {
        return m_IsOver;
    }

    bool SkipNext()
    {
        return m_SkipNext;
    }

private:

};

#endif // ASMETHODOPTIMIZER_H
