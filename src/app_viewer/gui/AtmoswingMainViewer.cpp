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

#ifdef WX_PRECOMP
#include "wx_pch.h"
#endif

#ifdef __BORLANDC__
#pragma hdrstop
#endif //__BORLANDC__

#include "AtmoswingMainViewer.h"

#include "asGeo.h"
#include "asGeoArea.h"
#include "asGeoAreaRegularGrid.h"
#include "asGeoAreaComposite.h"
#include "asGeoAreaCompositeGrid.h"
#include "asGeoPoint.h"
#include "asTime.h"
#include "asTimeArray.h"
#include "asFileNetcdf.h"
#include "asFileXml.h"
#include "asFileAscii.h"
#include "asFileDat.h"
#include "asConfig.h"


AtmoswingFrameViewer::AtmoswingFrameViewer(wxFrame *frame)
    : asFrameForecast(frame)
{
#if wxUSE_STATUSBAR
    wxLogStatus(_("Welcome to AtmoSwing %s."), asVersion::GetFullString().c_str());
#endif

    // Config file
    wxConfigBase *pConfig = wxFileConfig::Get();

    // Create log window and file
    bool displayLogWindow;
    pConfig->Read("/General/DisplayLogWindow", &displayLogWindow, false);
    m_logWindow = new asLogWindow(this, _("AtmoSwing log window"), displayLogWindow);
    Log().CreateFile("AtmoSwingViewer.log");
}

AtmoswingFrameViewer::~AtmoswingFrameViewer()
{
    //wxDELETE(m_logWindow);
}

void AtmoswingFrameViewer::OnClose(wxCloseEvent &event)
{
    Close(true);
}

void AtmoswingFrameViewer::OnQuit(wxCommandEvent &event)
{
    Close(true);
}

void AtmoswingFrameViewer::OnShowLog( wxCommandEvent& event )
{
    wxASSERT(m_logWindow);
    m_logWindow->Show();
}

