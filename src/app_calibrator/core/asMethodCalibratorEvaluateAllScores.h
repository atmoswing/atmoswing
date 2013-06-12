#ifndef ASMETHODCALIBRATOREVALUATEALLSCORES_H
#define ASMETHODCALIBRATOREVALUATEALLSCORES_H

#include <asMethodCalibrator.h>


class asMethodCalibratorEvaluateAllScores: public asMethodCalibrator
{
public:
    asMethodCalibratorEvaluateAllScores();
    virtual ~asMethodCalibratorEvaluateAllScores();

protected:
	virtual bool Calibrate(asParametersCalibration &params);

private:

};

#endif // ASMETHODCALIBRATOREVALUATEALLSCORES_H
