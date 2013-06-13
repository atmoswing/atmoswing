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
 
#include "asDataPredictandTemperature.h"

#include "wx/fileconf.h"

#include <asFileNetcdf.h>
#include <asTimeArray.h>
#include <asCatalog.h>
#include <asCatalogPredictands.h>


asDataPredictandTemperature::asDataPredictandTemperature(PredictandDB predictandDB)
:
asDataPredictand(predictandDB)
{
    //ctor
}

asDataPredictandTemperature::~asDataPredictandTemperature()
{
    //dtor
}

bool asDataPredictandTemperature::Load(const wxString &AlternateFilePath)
{
    return false;
}

bool asDataPredictandTemperature::Save(const wxString &AlternateDestinationDir)
{
    return false;
}

bool asDataPredictandTemperature::BuildPredictandDB(const wxString &AlternateDestinationDir)
{
    return false;
}
