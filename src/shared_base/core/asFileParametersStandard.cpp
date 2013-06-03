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
 
#include "asFileParametersStandard.h"

asFileParametersStandard::asFileParametersStandard(const wxString &FileName, const ListFileMode &FileMode)
:
asFileParameters(FileName, FileMode)
{
    // FindAndOpen() processed by asFileXml
}

asFileParametersStandard::~asFileParametersStandard()
{
    //dtor
}

bool asFileParametersStandard::InsertRootElement()
{
    if(!GoToFirstNodeWithPath("AtmoswingFile")) return false;
    if(!InsertElement(wxEmptyString, "StandardSet")) return false;
    if(!GoToFirstNodeWithPath("StandardSet")) return false;
    return true;
}

bool asFileParametersStandard::GoToRootElement()
{
    if(!GoToFirstNodeWithPath("AtmoswingFile.StandardSet"))
    {
        asLogError(wxString::Format(_("The file %s is not an Atmoswing parameters file."), m_FileName.GetFullName()));
        return false;
    }
    return true;
}
