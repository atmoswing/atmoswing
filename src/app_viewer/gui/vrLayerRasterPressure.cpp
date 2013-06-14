#include "vrLayerRasterPressure.h"

#include "vrrealrect.h"
#include "vrlabel.h"

vrLayerRasterPressure::vrLayerRasterPressure()
{
    //ctor
}

vrLayerRasterPressure::~vrLayerRasterPressure()
{
    //dtor
}

bool vrLayerRasterPressure::_Close()
{
	if (m_Dataset==NULL){
		return false;
	}

	GDALClose(m_Dataset);
	m_Dataset = NULL;
	return true;
}

bool vrLayerRasterPressure::CreateInMemory(Array2DFloat *data, Array1DFloat *Uaxis, Array1DFloat *Vaxis)
{
    wxASSERT(Uaxis->size()>1);
    wxASSERT(Vaxis->size()>1);
    wxASSERT(data->size()>1);

    // Try to close
	_Close();
	wxASSERT(m_Dataset == NULL);

	// Init filename and type
	m_FileName = wxFileName("memory");
	vrDrivers myDriver;
	m_DriverType = vrDRIVER_RASTER_TIFF;

    // Create
    GDALDriver *poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if( poDriver == NULL )
	{
	    asLogError("Failed to open the GDAL driver.");
	    return false;
	}
	char **papszOptions = NULL;
	m_Dataset = poDriver->Create( "tmp", 512, 512, 1, GDT_Byte,
                                papszOptions );


	if(m_Dataset == NULL){
		wxLogError("Unable to create %s, maybe driver not registered -  GDALAllRegister()\nGDAL Error: '%s'",
				   m_FileName.GetFullName(),
				   wxString(CPLGetLastErrorMsg()));

		return false;
	}


// GDALContoursGenerate




/*
http://www.gdal.org/frmt_mem.html
http://www.gdal.org/gdal_8h.html#22e22ce0a55036a96f652765793fb7a4
http://www.gdal.org/classGDALDataset.html#e077c53268d2272eebed10b891a05743
http://osgeo-org.1803224.n2.nabble.com/I-need-some-help-on-using-the-SetMercator-method-td2024475.html
http://www.gdal.org/gdal_tutorial.html
http://www.gdal.org/gdal_8h.html#22e22ce0a55036a96f652765793fb7a4
http://www.gdal.org/gdal_8h.html#dfd09c07260442427a225e2a536ead72
http://www.gdal.org/classGDALDriverManager.html
http://www.gdal.org/classGDALDriver.html
http://www.gdal.org/gdal_tutorial.html
http://www.gdal.org/classGDALRasterBand.html#f3e0fb84d29756a67214cff571a3e114
http://www.gdal.org/classGDALRasterBand.html#5497e8d29e743ee9177202cb3f61c3c7
http://www.gdal.org/classGDALRasterBand.html
http://www.gdal.org/gdal__alg_8h.html

http://ww2010.atmos.uiuc.edu/(Gh)/guides/maps/sfcobs/wnd.rxml
http://www.zygrib.org/index.php?page=download



*/

    double lonStart = (*Uaxis)[0];
    double latStart = (*Vaxis)[0];
    double lonStep = (*Uaxis)[1]-(*Uaxis)[0];
    double latStep = (*Vaxis)[1]-(*Vaxis)[0];

    double adfGeoTransform[6] = { lonStart, lonStep, 0, latStart, 0, latStep }; // {UOrigin, UCellSize, Rotation, VOrigin, Rotation, VCellSize}
    OGRSpatialReference oSRS;
    char *pszSRS_WKT = NULL;
    GDALRasterBand *poBand;
    float* abyRaster = new float[Uaxis->size()*Vaxis->size()];


srand ( time(NULL) );
    for (int i=0; i<Uaxis->size()*Vaxis->size(); i++)
    {
        abyRaster[i] = rand();
    }



    m_Dataset->SetGeoTransform( adfGeoTransform );

    oSRS.SetUTM( 11, TRUE );
    oSRS.SetWellKnownGeogCS( "WGS84" );
    oSRS.exportToWkt( &pszSRS_WKT );
    m_Dataset->SetProjection( pszSRS_WKT );
    CPLFree( pszSRS_WKT );

    poBand = m_Dataset->GetRasterBand(1);
    poBand->RasterIO( GF_Write, 0, 0, 512, 512,
                      abyRaster, 512, 512, GDT_Float32, 0, 0 );



	return true;





/*
types:

GDT_Unknown 	 Unknown or unspecified type
GDT_Byte 	 Eight bit unsigned integer
GDT_UInt16 	 Sixteen bit unsigned integer
GDT_Int16 	 Sixteen bit signed integer
GDT_UInt32 	 Thirty two bit unsigned integer
GDT_Int32 	 Thirty two bit signed integer
GDT_Float32 	 Thirty two bit floating point
GDT_Float64 	 Sixty four bit floating point
GDT_CInt16 	 Complex Int16
GDT_CInt32 	 Complex Int32
GDT_CFloat32 	 Complex Float32
GDT_CFloat64 	 Complex Float64

*/















}

bool vrLayerRasterPressure::_GetRasterData(unsigned char ** imgdata, const wxSize & outimgpxsize,
									  const wxRect & readimgpxinfo, const vrRender * render) {
	/*
	wxASSERT(m_Dataset);
	m_Dataset->FlushCache();
    int myRasterCount  = m_Dataset->GetRasterCount();

	if (myRasterCount != 3) {
		wxLogError("Corrupted C2D file, contain %d band instead of 3", myRasterCount);
		return false;
	}

	// create array for image data (RGBRGBRGB...)
	unsigned int myimgRGBLen = outimgpxsize.GetWidth() * outimgpxsize.GetHeight() * 3;
	*imgdata = (unsigned char *) malloc(myimgRGBLen);
	if (*imgdata == NULL) {
		wxLogError("Image creation failed, out of memory");
		return false;
	}


	// read band 2 (slope)
	GDALRasterBand *band = m_Dataset->GetRasterBand(2);
	int myDataSize = GDALGetDataTypeSize(GDT_Float32) / 8;
	void * mySlopeData = CPLMalloc(myDataSize * outimgpxsize.GetWidth() * outimgpxsize.GetHeight());
	if (band->RasterIO(GF_Read,
					   readimgpxinfo.GetX(), readimgpxinfo.GetY(),
					   readimgpxinfo.GetWidth(), readimgpxinfo.GetHeight(),
					   mySlopeData,
					   outimgpxsize.GetWidth(), outimgpxsize.GetHeight(),
					   GDT_Float32, 0, 0) != CE_None) {
		wxLogError("Error gettign C2D slope, maybe out of memory");
		if (mySlopeData != NULL) {
			CPLFree (mySlopeData);
			mySlopeData = NULL;
		}
		return false;
	}


	// read band 3 (aspect)
	band = m_Dataset->GetRasterBand(3);
	void * myAspectData = CPLMalloc(myDataSize * outimgpxsize.GetWidth() * outimgpxsize.GetHeight());
	if (band->RasterIO(GF_Read,
					   readimgpxinfo.GetX(), readimgpxinfo.GetY(),
					   readimgpxinfo.GetWidth(), readimgpxinfo.GetHeight(),
					   myAspectData,
					   outimgpxsize.GetWidth(), outimgpxsize.GetHeight(),
					   GDT_Float32, 0, 0) != CE_None) {
		wxLogError("Error gettign C2D aspect, maybe out of memory");
		if (myAspectData != NULL) {
			CPLFree (myAspectData);
			myAspectData = NULL;
		}
		return false;
	}

	vrRenderRasterColtop * myColtopRender = (vrRenderRasterColtop*) render;
	wxASSERT(myColtopRender);
	wxASSERT(myColtopRender->GetType() == vrRENDER_RASTER_C2D);

	// convert to RGB
	for (unsigned int i = 0; i< myimgRGBLen; i += 3) {
		double mySlpDouble = _ReadGDALValueToDouble(mySlopeData, GDT_Float32, i/3);
		double myAspDouble = _ReadGDALValueToDouble(myAspectData, GDT_Float32, i/3);

		wxImage::RGBValue myRGBValue = myColtopRender->GetColorFromDipDir(mySlpDouble, myAspDouble);

		// fill buffer
		*(*imgdata + i)	= myRGBValue.red;
		*(*imgdata + i + 1) = myRGBValue.green;
		*(*imgdata + i + 2) = myRGBValue.blue;
	}

	CPLFree(myAspectData);
	myAspectData = NULL;
	CPLFree(mySlopeData);
	mySlopeData = NULL;
	return true;*/
	return false;
}
