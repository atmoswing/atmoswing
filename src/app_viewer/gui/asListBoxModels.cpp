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
 
#include "asListBoxModels.h"

#include "asIncludes.h"
#include <asForecastViewer.h>


BEGIN_EVENT_TABLE(asListBoxModels, wxListBox)
	EVT_LISTBOX(wxID_ANY, asListBoxModels::OnModelSlctChange)
END_EVENT_TABLE()

wxDEFINE_EVENT(asEVT_ACTION_FORECAST_MODEL_SELECTION_CHANGED, wxCommandEvent);


asListBoxModels::asListBoxModels(wxWindow *parent, wxWindowID id, const wxPoint &pos, const wxSize &size, int n, const wxString choices[], long style)
:
wxListBox(parent, id, pos, size, n, choices, style)
{
    //ctor
}

asListBoxModels::~asListBoxModels()
{
    //dtor
}

bool asListBoxModels::Add(const wxString &modelName, const wxString &leadTimeOriginStr)
{
    wxString newOption = wxString::Format("%d. ", (int)GetStrings().GetCount()+1) + modelName + " (" + leadTimeOriginStr + ") ";

    Append(newOption);

    SetSelection(GetCount()-1);

    return true;
}

void asListBoxModels::OnModelSlctChange( wxCommandEvent & event )
{
    wxCommandEvent eventSlct (asEVT_ACTION_FORECAST_MODEL_SELECTION_CHANGED);
    eventSlct.SetInt(event.GetInt());
    GetParent()->ProcessWindowEvent(eventSlct);
}
