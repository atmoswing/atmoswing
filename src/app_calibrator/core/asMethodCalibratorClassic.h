#ifndef ASMETHODCALIBRATORCLASSIC_H
#define ASMETHODCALIBRATORCLASSIC_H

#include <asMethodCalibrator.h>


class asMethodCalibratorClassic: public asMethodCalibrator
{
public:
    asMethodCalibratorClassic();
    virtual ~asMethodCalibratorClassic();

protected:
	virtual bool Calibrate(asParametersCalibration &params);

private:

};

#endif // ASMETHODCALIBRATORCLASSIC_H
