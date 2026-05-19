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

#include "asPanelSidebarAnalogDates.h"

wxDEFINE_EVENT(asEVT_ACTION_ANALOG_DATE_SELECTION_CHANGED, wxCommandEvent);

asPanelSidebarAnalogDates::asPanelSidebarAnalogDates(wxWindow* parent, wxWindowID id, const wxPoint& pos,
                                                     const wxSize& size, long style)
    : asPanelSidebar(parent, id, pos, size, style) {
    _header->SetLabelText(_("Analog dates"));

    wxSize listSize = wxSize();
    listSize.SetHeight(120);
    _listCtrl = new wxListCtrl(this, wxID_ANY, wxDefaultPosition, listSize,
                               wxLC_REPORT | wxNO_BORDER | wxLC_SINGLE_SEL);
    _listCtrl->InsertColumn(0l, _("Analog"), wxLIST_FORMAT_RIGHT, 50);
    _listCtrl->InsertColumn(1l, _("Date"), wxLIST_FORMAT_LEFT, 100);
    _listCtrl->InsertColumn(2l, _("Criteria"), wxLIST_FORMAT_LEFT, 80);
    _listCtrl->Layout();

    _sizerContent->Add(_listCtrl, 0, wxEXPAND, 0);

    _listCtrl->Connect(wxEVT_COMMAND_LIST_ITEM_SELECTED, wxListEventHandler(asPanelSidebarAnalogDates::OnDateSelection),
                       nullptr, this);

    Layout();
    _sizerContent->Fit(this);
}

asPanelSidebarAnalogDates::~asPanelSidebarAnalogDates() {
    _listCtrl->Disconnect(wxEVT_COMMAND_LIST_ITEM_SELECTED,
                          wxListEventHandler(asPanelSidebarAnalogDates::OnDateSelection), nullptr, this);
}

void asPanelSidebarAnalogDates::OnDateSelection(wxListEvent& event) {
    // Send event
    wxCommandEvent eventParent(asEVT_ACTION_ANALOG_DATE_SELECTION_CHANGED);
    eventParent.SetInt(event.GetInt());

    GetParent()->ProcessWindowEvent(eventParent);
}

void asPanelSidebarAnalogDates::SetChoices(a1f& arrayDate, a1f& arrayCriteria, const wxString& dateFormat) {
    _listCtrl->Freeze();
    _listCtrl->DeleteAllItems();

    for (int i = 0; i < arrayDate.size(); i++) {
        wxString buf;
        buf.Printf("%d", i + 1);
        long tmp = _listCtrl->InsertItem(i, buf, 0);
        _listCtrl->SetItemData(tmp, i);

        buf.Printf("%s", asTime::GetStringTime(arrayDate[i], dateFormat));
        _listCtrl->SetItem(tmp, 1, buf);

        buf.Printf("%g", arrayCriteria[i]);
        _listCtrl->SetItem(tmp, 2, buf);
    }

    _listCtrl->Thaw();
}
