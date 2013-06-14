#include "asPredictorsManager.h"

#include "asPredictorsViewer.h"
#include "asCatalogPredictorsArchive.h"
#include "asDataPredictorArchive.h"


asPredictorsManager::asPredictorsManager()
{
    //ctor
}

asPredictorsManager::~asPredictorsManager()
{
    //dtor
}

bool asPredictorsManager::GetArchiveData(double date, int type, Array2DFloat *data, Array1DFloat *Uaxis, Array1DFloat *Vaxis)
{
    switch (type)
    {
        case (asPredictorsViewer::Hgt1000):
        {
//            asCatalogPredictorsArchive catalog;
//            if(!catalog.Load("NCEP_R-1_subset","hgt")) return false;
//            asDataPredictorArchive dataArchive(catalog);
//            if(!dataArchive.LoadFullArea(date, (float)1000.0)) return false;
//            VArray2DFloat arrayData = dataArchive.GetData();
//            wxASSERT(arrayData.size()==1);
//            m_DataArchive = arrayData[0];
//            m_UaxisDataArchive = dataArchive.GetAxisLon();
//            m_VaxisDataArchive = dataArchive.GetAxisLat();
//            break;
        }
        default:
        {
            asLogError(_("This archive predictor type is not handled for display."));
            return false;
        }
    }

    data = &m_DataArchive;
    Uaxis = &m_UaxisDataArchive;
    Vaxis = &m_VaxisDataArchive;
    return true;
}
