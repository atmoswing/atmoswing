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
 
#include "asPanelSidebarGisLayers.h"

asPanelSidebarGisLayers::asPanelSidebarGisLayers( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style )
:
asPanelSidebar( parent, id, pos, size, style )
{
    m_Header->SetLabelText(_("GIS layers"));

   // m_TocCtrl = new vrViewerTOCTree( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0, NULL, wxNO_BORDER);
    m_TocCtrl = new vrViewerTOCList( this, wxID_ANY);
	m_SizerContent->Add( m_TocCtrl->GetControl(), 1, wxEXPAND, 5 );

    Layout();
	m_SizerContent->Fit( this );
}

asPanelSidebarGisLayers::~asPanelSidebarGisLayers()
{

}
/*
void asPanelSidebarGisLayers::OnPaint( wxCommandEvent& event )
{
    event.Skip();
}
*/
