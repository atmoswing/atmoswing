/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
#ifndef ASDATAPREDICTORREALTIME_H
#define ASDATAPREDICTORREALTIME_H

#include <asIncludes.h>
#include <asDataPredictor.h>
#include <asCatalogPredictorsRealtime.h>


class asDataPredictorRealtime: public asDataPredictor
{
public:

    /** Default constructor
     * \param catalog The predictor catalog
     */
    asDataPredictorRealtime(asCatalogPredictorsRealtime &catalog);

    /** Default destructor */
    virtual ~asDataPredictorRealtime();

    int Download(asCatalogPredictorsRealtime &catalog);

    bool LoadFullArea(double date, float level, const VectorString &AlternatePredictorDataPath);

    virtual bool Load(asGeoAreaCompositeGrid &desiredArea, asTimeArray &timeArray, const VectorString &AlternatePredictorDataPath = VectorString(0));

    bool Load(asGeoAreaCompositeGrid *desiredArea, asTimeArray &timeArray, const VectorString &AlternatePredictorDataPath = VectorString(0));
    bool Load(asTimeArray &timeArray, const VectorString &AlternatePredictorDataPath = VectorString(0));

    /** Access m_Catalog
     * \return The current value of m_Catalog
     */
    asCatalogPredictorsRealtime& GetCatalog()
    {
        return m_Catalog;
    }


protected:
    asCatalogPredictorsRealtime m_Catalog; //!< Member variable "m_Catalog"

private:

};

#endif // ASDATAPREDICTORREALTIME_H
