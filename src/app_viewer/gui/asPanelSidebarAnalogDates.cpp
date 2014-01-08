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
 * Portions Copyright 2008-2013 University of Lausanne.
 */
 
#include "asPanelSidebarAnalogDates.h"

wxDEFINE_EVENT(asEVT_ACTION_ANALOG_DATE_SELECTION_CHANGED, wxCommandEvent);


asPanelSidebarAnalogDates::asPanelSidebarAnalogDates( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style )
:
asPanelSidebar( parent, id, pos, size, style )
{
    m_Header->SetLabelText(_("Analog dates"));

    wxSize listSize = wxSize();
    listSize.SetHeight(120);
    m_ListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, listSize, wxLC_REPORT|wxNO_BORDER|wxLC_SINGLE_SEL );
    m_ListCtrl->InsertColumn( 0l, _("Analog"), wxLIST_FORMAT_RIGHT, 50);
    m_ListCtrl->InsertColumn( 1l, _("Date"), wxLIST_FORMAT_LEFT, 100);
    m_ListCtrl->InsertColumn( 2l, _("Criteria"), wxLIST_FORMAT_LEFT, 80);
    m_ListCtrl->Layout();

    m_SizerContent->Add( m_ListCtrl, 0, wxEXPAND, 0 );

    m_ListCtrl->Connect( wxEVT_COMMAND_LIST_ITEM_SELECTED, wxListEventHandler( asPanelSidebarAnalogDates::OnDateSelection ), NULL, this );

    Layout();
    m_SizerContent->Fit( this );
}

asPanelSidebarAnalogDates::~asPanelSidebarAnalogDates()
{
    m_ListCtrl->Disconnect( wxEVT_COMMAND_LIST_ITEM_SELECTED, wxListEventHandler( asPanelSidebarAnalogDates::OnDateSelection ), NULL, this );
}

void asPanelSidebarAnalogDates::OnDateSelection( wxListEvent& event )
{
    // Send event
    wxCommandEvent eventParent (asEVT_ACTION_ANALOG_DATE_SELECTION_CHANGED);
    eventParent.SetInt(event.GetInt());

    GetParent()->ProcessWindowEvent(eventParent);
}

void asPanelSidebarAnalogDates::SetChoices(Array1DFloat &arrayDate, Array1DFloat &arrayCriteria)
{
    // To speed up inserting we hide the control temporarily
    m_ListCtrl->Freeze();

    m_ListCtrl->DeleteAllItems();

    for ( int i=0; i<arrayDate.size(); i++ )
    {
        wxString buf;
        buf.Printf("%d", i+1);
        long tmp = m_ListCtrl->InsertItem(i, buf, 0);
        m_ListCtrl->SetItemData(tmp, i);

        buf.Printf("%s", asTime::GetStringTime(arrayDate[i], "DD.MM.YYYY").c_str());
        m_ListCtrl->SetItem(tmp, 1, buf);

        buf.Printf("%g", arrayCriteria[i]);
        m_ListCtrl->SetItem(tmp, 2, buf);
    }

    m_ListCtrl->Thaw();
}
