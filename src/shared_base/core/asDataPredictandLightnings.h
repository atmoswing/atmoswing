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
 
#ifndef ASDATAPREDICTANDLIGHTNINGS_H
#define ASDATAPREDICTANDLIGHTNINGS_H

#include <asIncludes.h>
#include <asDataPredictand.h>


class asDataPredictandLightnings: public asDataPredictand
{
public:
    asDataPredictandLightnings(PredictandDB predictandDB);
    virtual ~asDataPredictandLightnings();

    virtual bool Load(const wxString &AlternateFilePath = wxEmptyString);

    virtual bool Save(const wxString &AlternateDestinationDir = wxEmptyString);

    virtual bool BuildPredictandDB(const wxString &AlternateCatalogFilePath = wxEmptyString, const wxString &AlternateDataDir = wxEmptyString, const wxString &AlternatePatternDir = wxEmptyString, const wxString &AlternateDestinationDir = wxEmptyString);


protected:

private:
	
    /** Initialize the containers
     * \return True on success
     */
    bool InitContainers();
};

#endif // ASDATAPREDICTANDLIGHTNINGS_H
