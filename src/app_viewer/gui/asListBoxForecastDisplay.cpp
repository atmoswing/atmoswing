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

#include "asListBoxForecastDisplay.h"

#include "asIncludes.h"
#include <asForecastViewer.h>


BEGIN_EVENT_TABLE(asListBoxForecastDisplay, wxListBox)
    EVT_LISTBOX(wxID_ANY, asListBoxForecastDisplay::OnForecastDisplaySlctChange)
END_EVENT_TABLE()

wxDEFINE_EVENT(asEVT_ACTION_FORECAST_RATIO_SELECTION_CHANGED, wxCommandEvent);


asListBoxForecastDisplay::asListBoxForecastDisplay(wxWindow *parent, wxWindowID id, const wxPoint &pos,
                                                   const wxSize &size, int n, const wxString choices[], long style)
        : wxListBox(parent, id, pos, size, n, choices, style)
{
    //ctor
}

void asListBoxForecastDisplay::OnForecastDisplaySlctChange(wxCommandEvent &event)
{
    wxCommandEvent eventSlct(asEVT_ACTION_FORECAST_RATIO_SELECTION_CHANGED);
    eventSlct.SetInt(event.GetInt());
    GetParent()->ProcessWindowEvent(eventSlct);
}
