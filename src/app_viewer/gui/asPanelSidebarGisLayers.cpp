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

#include "asPanelSidebarGisLayers.h"

asPanelSidebarGisLayers::asPanelSidebarGisLayers(wxWindow *parent, wxWindowID id, const wxPoint &pos,
                                                 const wxSize &size, long style)
        : asPanelSidebar(parent, id, pos, size, style)
{
    m_header->SetLabelText(_("GIS layers"));

    // m_tocCtrl = new vrViewerTOCTree( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0, NULL, wxNO_BORDER);
    m_tocCtrl = new vrViewerTOCList(this, wxID_ANY);
    m_sizerContent->Add(m_tocCtrl->GetControl(), 1, wxEXPAND, 5);

    Layout();
    m_sizerContent->Fit(this);
}

asPanelSidebarGisLayers::~asPanelSidebarGisLayers()
{

}
