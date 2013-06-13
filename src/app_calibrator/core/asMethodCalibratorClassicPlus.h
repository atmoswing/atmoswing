#ifndef ASMETHODCALIBRATORCLASSICPLUS_H
#define ASMETHODCALIBRATORCLASSICPLUS_H

#include <asMethodCalibrator.h>


class asMethodCalibratorClassicPlus: public asMethodCalibrator
{
public:
    asMethodCalibratorClassicPlus();
    virtual ~asMethodCalibratorClassicPlus();

protected:
	virtual bool Calibrate(asParametersCalibration &params);

private:

};

#endif // ASMETHODCALIBRATORCLASSICPLUS_H
