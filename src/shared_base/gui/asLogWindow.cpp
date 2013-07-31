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
 
#include "asLogWindow.h"

asLogWindow::asLogWindow(wxFrame *parent, const wxString& title, bool show, bool passToOld)
:
wxLogWindow(parent, title, show, passToOld)
{
    // Reduce the font size
    wxFrame* pFrame = this->GetFrame();
    wxFont font = pFrame->GetFont();
    font.SetPointSize(8);
    wxWindow* pLogTxt = pFrame->GetChildren()[0];
    pLogTxt->SetFont(font);

}

asLogWindow::~asLogWindow()
{
    //wxDELETE(m_LogWindow);
}

void asLogWindow::DoShow(bool bShow)
{
    Show(bShow);

    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Standard/DisplayLogWindow", bShow);
    ThreadsManager().CritSectionConfig().Leave();
}

bool asLogWindow::OnFrameClose (wxFrame *frame)
{
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/Standard/DisplayLogWindow", false);
    ThreadsManager().CritSectionConfig().Leave();

    return true;
}
