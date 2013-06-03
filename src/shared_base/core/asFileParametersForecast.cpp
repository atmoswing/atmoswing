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
 
#include "asFileParametersForecast.h"

asFileParametersForecast::asFileParametersForecast(const wxString &FileName, const ListFileMode &FileMode)
:
asFileParameters(FileName, FileMode)
{
    // FindAndOpen() processed by asFileXml
}

asFileParametersForecast::~asFileParametersForecast()
{
    //dtor
}

bool asFileParametersForecast::InsertRootElement()
{
    if(!GoToFirstNodeWithPath("AtmoswingFile")) return false;
    if(!InsertElement(wxEmptyString, "ForecastSet")) return false;
    if(!GoToFirstNodeWithPath("ForecastSet")) return false;
    return true;
}

bool asFileParametersForecast::GoToRootElement()
{
    if(!GoToFirstNodeWithPath("AtmoswingFile.ForecastSet"))
    {
        asLogError(wxString::Format(_("The file %s is not an Atmoswing forecast parameters file."), m_FileName.GetFullName()));
        return false;
    }
    return true;
}
