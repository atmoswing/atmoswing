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

#ifndef AS_PANEL_SIDEBAR_H
#define AS_PANEL_SIDEBAR_H

#include "AtmoSwingViewerGui.h"
#include "asIncludes.h"

class asPanelSidebar : public asPanelSidebarVirtual {
  public:
    explicit asPanelSidebar(wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition,
                            const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);

    void ReducePanel();

    bool IsReduced() const {
        return !m_sizerMain->IsShown(m_sizerContent);
    }

  protected:
    void OnReducePanel(wxMouseEvent& event) override;

    void OnPaint(wxCommandEvent& event);
};

#endif
