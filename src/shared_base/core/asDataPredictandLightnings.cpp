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
 
#include "asDataPredictandLightnings.h"

#include "wx/fileconf.h"

#include <asFileNetcdf.h>
#include <asTimeArray.h>
#include <asCatalog.h>
#include <asCatalogPredictands.h>


asDataPredictandLightnings::asDataPredictandLightnings(PredictandDB predictandDB)
:
asDataPredictand(predictandDB)
{
    //ctor
}

asDataPredictandLightnings::~asDataPredictandLightnings()
{
    //dtor
}

bool asDataPredictandLightnings::Load(const wxString &AlternateFilePath)
{
    return false;
}

bool asDataPredictandLightnings::Save(const wxString &AlternateDestinationDir)
{
    return false;
}

bool asDataPredictandLightnings::BuildPredictandDB(const wxString &AlternateDestinationDir)
{
    return false;
}
