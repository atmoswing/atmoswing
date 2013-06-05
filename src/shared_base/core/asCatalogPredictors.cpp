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
 
#include "asCatalogPredictors.h"

#include "wx/fileconf.h"

#include <asFileXml.h>


asCatalogPredictors::asCatalogPredictors(const wxString &alternateFilePath)
:
asCatalog(alternateFilePath)
{
    // Initiate some data
    m_Website = wxEmptyString;
    m_Ftp = wxEmptyString;
    m_Data.Id = wxEmptyString;
    m_Data.Name = wxEmptyString;
    m_Data.FileLength = Total;
    m_Data.FileName = wxEmptyString;
    m_Data.FileVarName = wxEmptyString;
}

asCatalogPredictors::~asCatalogPredictors()
{
    //dtor
}

bool asCatalogPredictors::Load(const wxString &DataSetId, const wxString &DataId)
{
    return false;
}

bool asCatalogPredictors::LoadDatasetProp(const wxString &DataSetId)
{
    return false;
}

bool asCatalogPredictors::LoadDataProp(const wxString &DataSetId, const wxString &DataId)
{
    return false;
}
