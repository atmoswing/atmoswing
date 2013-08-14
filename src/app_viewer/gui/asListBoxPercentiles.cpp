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
 
#include "asListBoxPercentiles.h"

#include "asIncludes.h"
#include <asForecastViewer.h>


BEGIN_EVENT_TABLE(asListBoxPercentiles, wxListBox)
	EVT_LISTBOX(wxID_ANY, asListBoxPercentiles::OnPercentileSlctChange)
END_EVENT_TABLE()

wxDEFINE_EVENT(asEVT_ACTION_FORECAST_PERCENTILE_SELECTION_CHANGED, wxCommandEvent);


asListBoxPercentiles::asListBoxPercentiles(wxWindow *parent, wxWindowID id, const wxPoint &pos, const wxSize &size, int n, const wxString choices[], long style)
:
wxListBox(parent, id, pos, size, n, choices, style)
{
    //ctor
}

asListBoxPercentiles::~asListBoxPercentiles()
{
    //dtor
}

void asListBoxPercentiles::OnPercentileSlctChange( wxCommandEvent & event )
{
	wxCommandEvent eventSlct (asEVT_ACTION_FORECAST_PERCENTILE_SELECTION_CHANGED);
    eventSlct.SetInt(event.GetInt());
    GetParent()->ProcessWindowEvent(eventSlct);
}
