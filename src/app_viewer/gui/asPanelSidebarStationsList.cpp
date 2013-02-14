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
 
#include "asPanelSidebarStationsList.h"

wxDEFINE_EVENT(asEVT_ACTION_STATION_SELECTION_CHANGED, wxCommandEvent);


asPanelSidebarStationsList::asPanelSidebarStationsList( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style )
:
asPanelSidebar( parent, id, pos, size, style )
{
    m_Header->SetLabelText(_("Station selection"));

    wxArrayString stationSelectionChoices;
	m_ChoiceStationSelection = new wxChoice( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, stationSelectionChoices, 0 );
	m_ChoiceStationSelection->SetSelection( 0 );
	m_SizerContent->Add( m_ChoiceStationSelection, 0, wxALL|wxEXPAND, 5 );

	m_ChoiceStationSelection->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asPanelSidebarStationsList::OnStationSelection ), NULL, this );

    Layout();
	m_SizerContent->Fit( this );
}

asPanelSidebarStationsList::~asPanelSidebarStationsList()
{
    m_ChoiceStationSelection->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asPanelSidebarStationsList::OnStationSelection ), NULL, this );
}

void asPanelSidebarStationsList::OnStationSelection( wxCommandEvent& event )
{
    // Send event
    wxCommandEvent eventParent (asEVT_ACTION_STATION_SELECTION_CHANGED);
    eventParent.SetInt(event.GetInt());

    GetParent()->ProcessWindowEvent(eventParent);
}

void asPanelSidebarStationsList::SetChoices(wxArrayString &arrayStation)
{
    m_ChoiceStationSelection->Set(arrayStation);
}
