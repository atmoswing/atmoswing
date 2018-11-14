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
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 */

#include "asPanelSidebarStationsList.h"

wxDEFINE_EVENT(asEVT_ACTION_STATION_SELECTION_CHANGED, wxCommandEvent);


asPanelSidebarStationsList::asPanelSidebarStationsList(wxWindow *parent, wxWindowID id, const wxPoint &pos,
                                                       const wxSize &size, long style)
        : asPanelSidebar(parent, id, pos, size, style)
{
    m_header->SetLabelText(_("Station selection"));

    wxArrayString stationSelectionChoices;
    m_choiceStationSelection = new wxChoice(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, stationSelectionChoices,
                                            0);
    m_choiceStationSelection->SetSelection(0);
    m_sizerContent->Add(m_choiceStationSelection, 0, wxALL | wxEXPAND, 5);

    m_choiceStationSelection->Connect(wxEVT_COMMAND_CHOICE_SELECTED,
                                      wxCommandEventHandler(asPanelSidebarStationsList::OnStationSelection), nullptr,
                                      this);

    Layout();
    m_sizerContent->Fit(this);
}

asPanelSidebarStationsList::~asPanelSidebarStationsList()
{
    m_choiceStationSelection->Disconnect(wxEVT_COMMAND_CHOICE_SELECTED,
                                         wxCommandEventHandler(asPanelSidebarStationsList::OnStationSelection), nullptr,
                                         this);
}

void asPanelSidebarStationsList::OnStationSelection(wxCommandEvent &event)
{
    // Send event
    wxCommandEvent eventParent(asEVT_ACTION_STATION_SELECTION_CHANGED);
    eventParent.SetInt(event.GetInt());

    GetParent()->ProcessWindowEvent(eventParent);
}

void asPanelSidebarStationsList::SetChoices(wxArrayString &arrayStation)
{
    m_choiceStationSelection->Set(arrayStation);
}
