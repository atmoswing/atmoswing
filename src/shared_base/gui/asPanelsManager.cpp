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
 
#include "asPanelsManager.h"

asPanelsManager::asPanelsManager()
{
    //ctor
}

asPanelsManager::~asPanelsManager()
{

}

void asPanelsManager::LayoutFrame(wxWindow* element)
{
     // Get parent frame for layout
    wxWindow* Parent = element;
    wxWindow* SearchParent = Parent;
    while (SearchParent)
    {
        Parent = SearchParent;
        SearchParent = Parent->GetParent();
    }
    Parent->Layout();
}

wxWindow* asPanelsManager::GetTopFrame(wxWindow* element)
{
     // Get parent frame for layout
    wxWindow* Parent = element;
    wxWindow* SearchParent = Parent;
    while (SearchParent)
    {
        Parent = SearchParent;
        SearchParent = Parent->GetParent();
    }

    return Parent;
}
