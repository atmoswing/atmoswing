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
 
#include "asPanelSidebarForecasts.h"

#include <wx/statline.h>

asPanelSidebarForecasts::asPanelSidebarForecasts( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style )
:
asPanelSidebar( parent, id, pos, size, style )
{
    m_Header->SetLabelText(_("Forecasts"));

    // Forecasts controls
    wxSize modelsSize = wxSize();
    modelsSize.SetHeight(80);
    m_ModelsCtrl = new asListBoxModels( this, wxID_ANY, wxDefaultPosition, modelsSize, 0, NULL, wxNO_BORDER );
    m_SizerContent->Add( m_ModelsCtrl, 1, wxEXPAND, 5 );

    wxSize lineSize = wxSize();
    lineSize.SetHeight(10);
    wxStaticLine *staticline = new wxStaticLine(this, wxID_ANY, wxDefaultPosition, lineSize);
    m_SizerContent->Add( staticline, 0, 0, 0 );

    wxBoxSizer* subSizer;
    subSizer = new wxBoxSizer( wxHORIZONTAL );

    wxSize displaySize = wxSize();
    displaySize.SetHeight(100);
    m_ForecastDisplayCtrl = new asListBoxForecastDisplay( this, wxID_ANY, wxDefaultPosition, displaySize, 0, NULL, wxNO_BORDER );
    subSizer->Add( m_ForecastDisplayCtrl, 1, wxEXPAND, 5 );

    m_PercentilesCtrl = new asListBoxPercentiles( this, wxID_ANY, wxDefaultPosition, displaySize, 0, NULL, wxNO_BORDER );
    subSizer->Add( m_PercentilesCtrl, 1, wxEXPAND, 5 );
    subSizer->Fit( this );
    m_SizerContent->Add( subSizer, 0, wxEXPAND, 5 );

    Layout();
    m_SizerContent->Fit( this );
}

asPanelSidebarForecasts::~asPanelSidebarForecasts()
{

}

void asPanelSidebarForecasts::ClearForecasts()
{
    m_ModelsCtrl->Clear();
}

void asPanelSidebarForecasts::AddForecast(const wxString &methodId, const wxString &methodIdDisplay, const wxString &specificTag, const wxString &specificTagDisplay, DataParameter dataParameter, DataTemporalResolution dataTemporalResolution)
{
    m_ModelsCtrl->Add(methodId, methodIdDisplay, specificTag, specificTagDisplay, dataParameter, dataTemporalResolution);
}
