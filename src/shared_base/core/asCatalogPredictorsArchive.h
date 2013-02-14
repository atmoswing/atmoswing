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
 
#ifndef ASCATALOGPREDICTORARCHIVES_H
#define ASCATALOGPREDICTORARCHIVES_H

#include <asIncludes.h>
#include <asCatalogPredictors.h>

class asCatalogPredictorsArchive: public asCatalogPredictors
{
public:

    /** Default constructor
     * \param DataSetId The dataset ID
     * \param DataId The data ID. If not set, load the whole database information
     */
    asCatalogPredictorsArchive(const wxString &alternateFilePath = wxEmptyString);

    /** Default destructor */
    virtual ~asCatalogPredictorsArchive();

    bool Load(const wxString &DataSetId, const wxString &DataId = wxEmptyString);

    /** Method that get dataset information from file
     * \param DataSetId The data Set ID
     */
    bool LoadDatasetProp(const wxString &DataSetId);

    /** Method that get data information from file
     * \param DataSetId The data Set ID
     * \param DataId The data ID
     */
    bool LoadDataProp(const wxString &DataSetId, const wxString &DataId);


protected:

private:

};

#endif // ASCATALOGPREDICTORARCHIVES_H
