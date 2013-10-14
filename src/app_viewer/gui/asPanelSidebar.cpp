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
 
#include "asPanelSidebar.h"

#include "img_bullets.h"

asPanelSidebar::asPanelSidebar( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style )
:
asPanelSidebarVirtual( parent, id, pos, size, style )
{
    m_BpButtonReduce->SetBitmapLabel(img_shown);
}

wxWindow* asPanelSidebar::GetTopFrame(wxWindow* element)
{
     // Get parent frame for layout
    wxWindow* Parent = element;
    wxWindow* SearchParent = Parent;
    while (SearchParent)
    {
        Parent = SearchParent;
        SearchParent = Parent->GetParent();
    }

    return Parent;
}

void asPanelSidebar::OnReducePanel( wxCommandEvent& event )
{
    wxWindow* topFrame = GetTopFrame(this);

    topFrame->Freeze();

    if(m_SizerMain->IsShown(m_SizerContent))
    {
        m_SizerMain->Hide(m_SizerContent, true);
        m_BpButtonReduce->SetBitmapLabel(img_hidden);
    }
    else
    {
        m_SizerMain->Show(m_SizerContent, true);
        m_BpButtonReduce->SetBitmapLabel(img_shown);
    }

    // Refresh elements
    m_SizerMain->Layout();
    Layout();
    GetSizer()->Fit(GetParent());
    topFrame->Layout();
    topFrame->Refresh();

    topFrame->Thaw();
}

void asPanelSidebar::ReducePanel()
{
    if(m_SizerMain->IsShown(m_SizerContent))
    {
        m_SizerMain->Hide(m_SizerContent, true);
        m_BpButtonReduce->SetBitmapLabel(img_hidden);
    }
    else
    {
        m_SizerMain->Show(m_SizerContent, true);
        m_BpButtonReduce->SetBitmapLabel(img_shown);
    }

}

void asPanelSidebar::OnPaint( wxCommandEvent& event )
{
    event.Skip();
}
