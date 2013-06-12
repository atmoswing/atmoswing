#ifndef ASMETHODCALIBRATORSINGLE_H
#define ASMETHODCALIBRATORSINGLE_H

#include <asMethodCalibrator.h>


class asMethodCalibratorSingle: public asMethodCalibrator
{
public:
    asMethodCalibratorSingle();
    virtual ~asMethodCalibratorSingle();

protected:
	virtual bool Calibrate(asParametersCalibration &params);

private:

};

#endif // ASMETHODCALIBRATORSINGLE_H
