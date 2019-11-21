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
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#include "asPanelSidebarCalendar.h"

#include "images.h"

asPanelSidebarCalendar::asPanelSidebarCalendar(wxWindow *parent, wxWindowID id, const wxPoint &pos, const wxSize &size,
                                               long style)
    : asPanelSidebar(parent, id, pos, size, style) {
    m_header->SetLabelText(_("Day of the forecast"));

    m_calendarForecastDate =
        new wxCalendarCtrl(this, wxID_ANY, wxDefaultDateTime, wxDefaultPosition, wxDefaultSize,
                           wxCAL_MONDAY_FIRST | wxCAL_SHOW_HOLIDAYS | wxCAL_SHOW_SURROUNDING_WEEKS);
    m_sizerContent->Add(m_calendarForecastDate, 0, wxALL | wxALIGN_CENTER_HORIZONTAL, 5);

    wxBoxSizer *bSizer35;
    bSizer35 = new wxBoxSizer(wxHORIZONTAL);

    m_staticTextForecastHour = new wxStaticText(this, wxID_ANY, wxT("Hour (UTM)"), wxDefaultPosition, wxDefaultSize, 0);
    m_staticTextForecastHour->Wrap(-1);
    bSizer35->Add(m_staticTextForecastHour, 0, wxTOP | wxBOTTOM | wxLEFT, 5);

    m_textCtrlForecastHour = new wxTextCtrl(this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize(40, -1), 0);
    m_textCtrlForecastHour->SetMaxLength(2);
    bSizer35->Add(m_textCtrlForecastHour, 0, wxALL, 5);

    m_bpButtonNow = new wxBitmapButton(this, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize(-1, -1),
                                       wxBU_AUTODRAW | wxNO_BORDER);
    m_bpButtonNow->SetToolTip(wxT("Set current date."));
    m_bpButtonNow->SetBitmapLabel(*_img_update);

    bSizer35->Add(m_bpButtonNow, 0, wxTOP | wxBOTTOM, 5);

    m_sizerContent->Add(bSizer35, 0, wxALIGN_CENTER_HORIZONTAL, 5);

    m_bpButtonNow->Connect(wxEVT_COMMAND_BUTTON_CLICKED,
                           wxCommandEventHandler(asPanelSidebarCalendar::OnSetPresentDate), nullptr, this);
}

asPanelSidebarCalendar::~asPanelSidebarCalendar() {
    m_bpButtonNow->Disconnect(wxEVT_COMMAND_BUTTON_CLICKED,
                              wxCommandEventHandler(asPanelSidebarCalendar::OnSetPresentDate), nullptr, this);
}

void asPanelSidebarCalendar::OnSetPresentDate(wxCommandEvent &event) {
    SetPresentDate();
}

void asPanelSidebarCalendar::SetPresentDate() {
    // Set the present date in the calendar and the hour field
    wxDateTime nowWx = asTime::NowWxDateTime(asUTM);
    Time nowStruct = asTime::NowTimeStruct(asUTM);
    wxString hourStr = wxString::Format("%d", nowStruct.hour);
    m_calendarForecastDate->SetDate(nowWx);
    m_textCtrlForecastHour->SetValue(hourStr);
}
