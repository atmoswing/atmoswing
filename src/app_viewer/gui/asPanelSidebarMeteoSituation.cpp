/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 * 
 * When distributing Covered Code, include this CDDL Header Notice in 
 * each file and include the License file (licence.txt). If applicable, 
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 * 
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2014 Pascal Horton, Terr@num.
 */
 
#include "asPanelSidebarMeteoSituation.h"

wxDEFINE_EVENT(asEVT_ACTION_PRESET_SELECTION_CHANGED, wxCommandEvent);


asPanelSidebarMeteoSituation::asPanelSidebarMeteoSituation( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style )
:
asPanelSidebar( parent, id, pos, size, style )
{
    m_Header->SetLabelText(_("Preset selection"));

    wxArrayString selectionChoices;
    m_ChoicePreset = new wxChoice( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, selectionChoices, 0 );
    m_ChoicePreset->SetSelection( 0 );
    m_SizerContent->Add( m_ChoicePreset, 0, wxALL|wxEXPAND, 5 );

    m_ChoicePreset->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asPanelSidebarMeteoSituation::OnPresetSelection ), NULL, this );

    Layout();
    m_SizerContent->Fit( this );
}

asPanelSidebarMeteoSituation::~asPanelSidebarMeteoSituation()
{
    m_ChoicePreset->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asPanelSidebarMeteoSituation::OnPresetSelection ), NULL, this );
}

void asPanelSidebarMeteoSituation::OnPresetSelection( wxCommandEvent& event )
{
    // Send event
    wxCommandEvent eventParent (asEVT_ACTION_PRESET_SELECTION_CHANGED);
    eventParent.SetInt(event.GetInt());

    GetParent()->ProcessWindowEvent(eventParent);
}

void asPanelSidebarMeteoSituation::SetChoices(wxArrayString &val)
{
    m_ChoicePreset->Set(val);
}
