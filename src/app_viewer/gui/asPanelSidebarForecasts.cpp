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

void asPanelSidebarForecasts::AddForecast(const wxString &modelName, const wxString &leadTimeOriginStr, DataParameter dataParameter, DataTemporalResolution dataTemporalResolution)
{
    m_ModelsCtrl->Add(modelName, leadTimeOriginStr, dataParameter, dataTemporalResolution);
}
