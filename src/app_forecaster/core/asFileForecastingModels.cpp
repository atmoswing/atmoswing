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
 
#include "asFileForecastingModels.h"

asFileForecastingModels::asFileForecastingModels(const wxString &FileName, const ListFileMode &FileMode)
:
asFileXml(FileName, FileMode)
{
    // FindAndOpen() processed by asFileXml
}

asFileForecastingModels::~asFileForecastingModels()
{
    //dtor
}

bool asFileForecastingModels::InsertRootElement()
{
    if(!GoToFirstNodeWithPath("AtmoswingFile")) return false;
    if(!InsertElement(wxEmptyString, "ForecastingModels")) return false;
    if(!GoToFirstNodeWithPath("ForecastingModels")) return false;
    return true;
}

bool asFileForecastingModels::GoToRootElement()
{
    if(!GoToFirstNodeWithPath("AtmoswingFile.ForecastingModels")) return false;
    return true;
}
