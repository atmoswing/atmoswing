#ifndef VRLAYERRASTERPRESSURE_H
#define VRLAYERRASTERPRESSURE_H

#include <asIncludes.h>
#include <wx/image.h>
#include "vrlayerraster.h"

class vrRealRect;
class vrRender;
class vrLabel;

class vrLayerRasterPressure : public vrLayerRasterGDAL
{
public:
    /** Default constructor */
    vrLayerRasterPressure();
    /** Default destructor */
    virtual ~vrLayerRasterPressure();

    virtual bool CreateInMemory(Array2DFloat *data, Array1DFloat *Uaxis, Array1DFloat *Vaxis);

protected:
    virtual bool _GetRasterData(unsigned char ** imgdata,
								const wxSize & outimgpxsize,
								const wxRect & readimgpxinfo,
								const vrRender * render);
private:
    bool _Close();
};

#endif // VRLAYERRASTERPRESSURE_H
