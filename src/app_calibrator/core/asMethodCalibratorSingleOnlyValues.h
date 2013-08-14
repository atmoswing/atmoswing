#ifndef ASMETHODCALIBRATORSINGLEONLYVALUES_H
#define ASMETHODCALIBRATORSINGLEONLYVALUES_H

#include <asMethodCalibrator.h>


class asMethodCalibratorSingleOnlyValues: public asMethodCalibrator
{
public:
    asMethodCalibratorSingleOnlyValues();
    virtual ~asMethodCalibratorSingleOnlyValues();

protected:
	virtual bool Calibrate(asParametersCalibration &params);

private:

};

#endif // ASMETHODCALIBRATORSINGLEONLYVALUES_H
