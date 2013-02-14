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
 
#ifndef ASDATAPREDICTORARCHIVE_H
#define ASDATAPREDICTORARCHIVE_H

#include <asIncludes.h>
#include <asDataPredictor.h>
#include <asCatalogPredictorsArchive.h>

class asGeoArea;

class asDataPredictorArchive: public asDataPredictor
{
public:

    /** Default constructor
     * \param catalog The predictor catalog
     */
    asDataPredictorArchive(asCatalogPredictorsArchive &catalog);

    /** Default destructor */
    virtual ~asDataPredictorArchive();

    bool LoadFullArea(double date, float level, const wxString &AlternatePredictorDataPath = wxEmptyString);

    virtual bool Load(asGeoAreaCompositeGrid &desiredArea, asTimeArray &timeArray, const VectorString &AlternatePredictorDataPath = VectorString(0));

    bool Load(asGeoAreaCompositeGrid &desiredArea, double date, const wxString &AlternatePredictorDataPath = wxEmptyString);
    bool Load(asGeoAreaCompositeGrid *desiredArea, double date, const wxString &AlternatePredictorDataPath = wxEmptyString);
    bool Load(asGeoAreaCompositeGrid &desiredArea, asTimeArray &timeArray, const wxString &AlternatePredictorDataPath);


    /** Method to load a tensor of data for a given area and a given time array
     * \param desiredArea The desired area
     * \param timeArray The desired time array
     */
    bool Load(asGeoAreaCompositeGrid *desiredArea, asTimeArray &timeArray, const wxString &AlternatePredictorDataPath = wxEmptyString);

    bool ClipToArea(asGeoAreaCompositeGrid *desiredArea);

    /** Access m_Catalog
     * \return The current value of m_Catalog
     */
    asCatalogPredictorsArchive& GetCatalog()
    {
        return m_Catalog;
    }


protected:
    asCatalogPredictorsArchive m_Catalog; //!< Member variable "m_Catalog"

    /** Method to check the time array compatibility with the catalog
     * \param timeArray The time array to check
     * \return True if compatible with the catalog
     */
    bool CheckTimeArray(asTimeArray &timeArray);

private:

};

#endif // ASDATAPREDICTORARCHIVE_H
