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

#include "asLogWindow.h"

asLogWindow::asLogWindow(wxFrame *parent, const wxString &title, bool show, bool passToOld)
    : wxLogWindow(parent, title, show, passToOld) {
  // Reduce the font size
  wxFrame *pFrame = this->GetFrame();
  wxFont font = pFrame->GetFont();
  font.SetPointSize(8);
  wxWindow *pLogTxt = pFrame->GetChildren()[0];
  pLogTxt->SetFont(font);
}

void asLogWindow::DoShow(bool bShow) {
  Show(bShow);

  ThreadsManager().CritSectionConfig().Enter();
  wxConfigBase *pConfig = wxFileConfig::Get();
  pConfig->Write("/General/DisplayLogWindow", bShow);
  ThreadsManager().CritSectionConfig().Leave();
}

bool asLogWindow::OnFrameClose(wxFrame *frame) {
  ThreadsManager().CritSectionConfig().Enter();
  wxConfigBase *pConfig = wxFileConfig::Get();
  pConfig->Write("/General/DisplayLogWindow", false);
  ThreadsManager().CritSectionConfig().Leave();

  return true;
}
