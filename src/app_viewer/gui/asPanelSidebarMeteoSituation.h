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
 * Portions Copyright 2014 Pascal Horton, Terr@num.
 */
 
#ifndef __asPanelSidebarMeteoSituation__
#define __asPanelSidebarMeteoSituation__

#include "asPanelSidebar.h"

#include "asIncludes.h"

/** Implementing asPanelSidebarMeteoSituation */
class asPanelSidebarMeteoSituation : public asPanelSidebar
{
public:
    /** Constructor */
    asPanelSidebarMeteoSituation( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~asPanelSidebarMeteoSituation();

    void SetChoices(wxArrayString &val);

    wxChoice* GetChoiceCtrl()
    {
        return m_ChoicePreset;
    }

private:
    wxChoice* m_ChoicePreset;

    void OnPresetSelection( wxCommandEvent& event );

};

#endif // __asPanelSidebarMeteoSituation__
