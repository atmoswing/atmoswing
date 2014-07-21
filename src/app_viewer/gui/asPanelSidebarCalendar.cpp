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
 
#include "asPanelSidebarCalendar.h"

#include "img_bullets.h"
#include <wx/calctrl.h>

asPanelSidebarCalendar::asPanelSidebarCalendar( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style )
:
asPanelSidebar( parent, id, pos, size, style )
{
    m_Header->SetLabelText(_("Day of the forecast"));

    m_CalendarForecastDate = new wxCalendarCtrl( this, wxID_ANY, wxDefaultDateTime, wxDefaultPosition, wxDefaultSize, wxCAL_MONDAY_FIRST|wxCAL_SHOW_HOLIDAYS|wxCAL_SHOW_SURROUNDING_WEEKS );
    m_SizerContent->Add( m_CalendarForecastDate, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );

    wxBoxSizer* bSizer35;
    bSizer35 = new wxBoxSizer( wxHORIZONTAL );

    m_StaticTextForecastHour = new wxStaticText( this, wxID_ANY, wxT("Hour (UTM)"), wxDefaultPosition, wxDefaultSize, 0 );
    m_StaticTextForecastHour->Wrap( -1 );
    bSizer35->Add( m_StaticTextForecastHour, 0, wxTOP|wxBOTTOM|wxLEFT, 5 );

    m_TextCtrlForecastHour = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 40,-1 ), 0 );
    m_TextCtrlForecastHour->SetMaxLength( 2 );
    bSizer35->Add( m_TextCtrlForecastHour, 0, wxALL, 5 );

    m_BpButtonNow = new wxBitmapButton( this, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( -1,-1 ), wxBU_AUTODRAW|wxNO_BORDER );
    m_BpButtonNow->SetToolTip( wxT("Set current date.") );
    m_BpButtonNow->SetBitmapLabel(img_clock_now);

    bSizer35->Add( m_BpButtonNow, 0, wxTOP|wxBOTTOM, 5 );

    m_SizerContent->Add( bSizer35, 0, wxALIGN_CENTER_HORIZONTAL, 5 );

    m_BpButtonNow->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asPanelSidebarCalendar::OnSetPresentDate ), NULL, this );
}

asPanelSidebarCalendar::~asPanelSidebarCalendar()
{
    m_BpButtonNow->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asPanelSidebarCalendar::OnSetPresentDate ), NULL, this );
}

void asPanelSidebarCalendar::OnSetPresentDate( wxCommandEvent& event )
{
    SetPresentDate();
}

void asPanelSidebarCalendar::SetPresentDate( )
{
    // Set the present date in the calendar and the hour field
    wxDateTime nowWx = asTime::NowWxDateTime(asUTM);
    TimeStruct nowStruct = asTime::NowTimeStruct(asUTM);
    wxString hourStr = wxString::Format("%d", nowStruct.hour);
    m_CalendarForecastDate->SetDate(nowWx);
    m_TextCtrlForecastHour->SetValue(hourStr);
}

