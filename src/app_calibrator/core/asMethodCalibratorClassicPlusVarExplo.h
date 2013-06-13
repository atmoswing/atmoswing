#ifndef ASMETHODCALIBRATORCLASSICPLUSVAREXPLO_H
#define ASMETHODCALIBRATORCLASSICPLUSVAREXPLO_H

#include <asMethodCalibratorClassicPlus.h>


class asMethodCalibratorClassicPlusVarExplo: public asMethodCalibratorClassicPlus
{
public:
    asMethodCalibratorClassicPlusVarExplo();
    virtual ~asMethodCalibratorClassicPlusVarExplo();

protected:
	virtual bool Calibrate(asParametersCalibration &params);

private:

};

#endif // ASMETHODCALIBRATORCLASSICPLUSVAREXPLO_H
