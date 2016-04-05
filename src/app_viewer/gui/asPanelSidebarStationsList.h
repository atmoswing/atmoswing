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

#ifndef __asPanelSidebarStationsList__
#define __asPanelSidebarStationsList__

#include "asPanelSidebar.h"

#include "asIncludes.h"

class asPanelSidebarStationsList
        : public asPanelSidebar
{
public:
    asPanelSidebarStationsList(wxWindow *parent, wxWindowID id = wxID_ANY, const wxPoint &pos = wxDefaultPosition,
                               const wxSize &size = wxDefaultSize, long style = wxTAB_TRAVERSAL);

    ~asPanelSidebarStationsList();

    void SetChoices(wxArrayString &arrayStation);

    wxChoice *GetChoiceCtrl() const
    {
        return m_choiceStationSelection;
    }

private:
    wxChoice *m_choiceStationSelection;

    void OnStationSelection(wxCommandEvent &event);

};

#endif // __asPanelSidebarStationsList__
