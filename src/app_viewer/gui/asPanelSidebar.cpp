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
