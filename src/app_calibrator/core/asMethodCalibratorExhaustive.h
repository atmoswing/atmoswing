#ifndef ASMETHODCALIBRATOREXHAUSTIVE_H
#define ASMETHODCALIBRATOREXHAUSTIVE_H

#include <asMethodCalibrator.h>


class asMethodCalibratorExhaustive: public asMethodCalibrator
{
public:
    asMethodCalibratorExhaustive();
    virtual ~asMethodCalibratorExhaustive();

protected:
	virtual bool Calibrate(asParametersCalibration &params);

private:

};

#endif // ASMETHODCALIBRATOREXHAUSTIVE_H
