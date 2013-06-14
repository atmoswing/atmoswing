#ifndef ASPREDICTORSMANAGER_H
#define ASPREDICTORSMANAGER_H

#include <asIncludes.h>


class asPredictorsManager
{
public:
    asPredictorsManager();
    virtual ~asPredictorsManager();
    bool GetArchiveData(double date, int type, Array2DFloat *data, Array1DFloat *Uaxis, Array1DFloat *Vaxis);

protected:
private:
    Array2DFloat m_DataArchive;
    Array1DFloat m_UaxisDataArchive;
    Array1DFloat m_VaxisDataArchive;
    Array2DFloat m_DataTarget;
    Array1DFloat m_UaxisDataTarget;
    Array1DFloat m_VaxisDataTarget;
};

#endif // ASPREDICTORSMANAGER_H
