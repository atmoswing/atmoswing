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

#ifdef WX_PRECOMP
#include "wx_pch.h"
#endif

#ifdef __BORLANDC__
#pragma hdrstop
#endif  //__BORLANDC__

#include "AtmoswingMainForecaster.h"

AtmoswingFrameForecaster::AtmoswingFrameForecaster(wxFrame* frame)
    : asFrameForecaster(frame) {
#if wxUSE_STATUSBAR
    wxLogStatus(_("Welcome to AtmoSwing %s."), asVersion::GetFullString());
#endif

    // Config file
    wxConfigBase* pConfig = wxFileConfig::Get();

    // Set default options
    SetDefaultOptions();

    // Create log window and file
    delete wxLog::SetActiveTarget(new asLogGui());
    m_logWindow = new asLogWindow(this, _("AtmoSwing log window"),
                                  pConfig->ReadBool("/General/DisplayLogWindow", true));
    Log()->CreateFile("AtmoSwingForecaster.log");
    Log()->SetLevel(wxFileConfig::Get()->ReadLong("/General/LogLevel", 2l));

    // Restore frame position and size
    int minHeight = 600, minWidth = 500;
    int x = (int)pConfig->ReadLong("/MainFrame/x", 50), y = (int)pConfig->ReadLong("/MainFrame/y", 50),
        w = (int)pConfig->ReadLong("/MainFrame/w", minWidth), h = (int)pConfig->ReadLong("/MainFrame/h", minHeight);
    wxRect screen = wxGetClientDisplayRect();
    if (x < screen.x - 10) x = screen.x;
    if (x > screen.width) x = screen.x;
    if (y < screen.y - 10) y = screen.y;
    if (y > screen.height) y = screen.y;
    if (w + x > screen.width) w = screen.width - x;
    if (w < minWidth) w = minWidth;
    if (w + x > screen.width) x = screen.width - w;
    if (h + y > screen.height) h = screen.height - y;
    if (h < minHeight) h = minHeight;
    if (h + y > screen.height) y = screen.height - h;
    Move(x, y);
    SetClientSize(w, h);
    Fit();

    // Get the GUI mode -> silent or not
    long guiOptions = pConfig->ReadLong("/General/GuiOptions", 0l);
    if (guiOptions == 0l) {
        g_silentMode = true;
    } else {
        g_silentMode = false;
        g_verboseMode = false;
        if (guiOptions == 2l) {
            g_verboseMode = true;
        }
    }
}

void AtmoswingFrameForecaster::SetDefaultOptions() {
    wxConfigBase* pConfig = wxFileConfig::Get();

    // General
    pConfig->Write("/General/GuiOptions", pConfig->ReadLong("/General/GuiOptions", 1l));
    pConfig->Write("/General/Responsive", pConfig->ReadBool("/General/Responsive", true));
    pConfig->Write("/General/LogLevel", pConfig->Read("/General/LogLevel", 1));
    pConfig->Write("/General/DisplayLogWindow", pConfig->ReadBool("/General/DisplayLogWindow", false));

    // Internet
    pConfig->Write("/Internet/MaxPreviousStepsNb", pConfig->Read("/Internet/MaxPreviousStepsNb", "5"));
    pConfig->Write("/Internet/ParallelRequestsNb", pConfig->Read("/Internet/ParallelRequestsNb", "5"));
    pConfig->Write("/Internet/RestrictDownloads", pConfig->ReadBool("/Internet/RestrictDownloads", true));
    pConfig->Write("/Internet/UsesProxy", pConfig->ReadBool("/Internet/UsesProxy", false));

    // Processing
    bool allowMultithreading = pConfig->ReadBool("/Processing/AllowMultithreading", true);
    pConfig->Write("/Processing/AllowMultithreading", allowMultithreading);
    int maxThreads = wxThread::GetCPUCount();
    if (maxThreads == -1) maxThreads = 2;
    pConfig->Write("/Processing/ThreadsNb", pConfig->Read("/Processing/ThreadsNb", asStrF("%d", maxThreads)));
    long processingMethod = pConfig->Read("/Processing/Method", (long)asMULTITHREADS);
    if (!allowMultithreading) {
        processingMethod = (long)asMULTITHREADS;
    }
    pConfig->Write("/Processing/Method", processingMethod);

    pConfig->Flush();
}

AtmoswingFrameForecaster::~AtmoswingFrameForecaster() {
    // Config file
    wxConfigBase* pConfig = wxFileConfig::Get();
    if (!pConfig) return;

    // Save the frame position
    int x, y, w, h;
    GetClientSize(&w, &h);
    GetPosition(&x, &y);
    pConfig->Write("/MainFrame/x", (long)x);
    pConfig->Write("/MainFrame/y", (long)y);
    pConfig->Write("/MainFrame/w", (long)w);
    pConfig->Write("/MainFrame/h", (long)h);

    // wxDELETE(m_logWindow);
}

void AtmoswingFrameForecaster::OnClose(wxCloseEvent& event) {
    Close(true);
}

void AtmoswingFrameForecaster::OnQuit(wxCommandEvent& event) {
    Close(true);
}
