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
 
#include "asListBoxModels.h"

#include <asForecastViewer.h>


BEGIN_EVENT_TABLE(asListBoxModels, wxListBox)
    EVT_LISTBOX(wxID_ANY, asListBoxModels::OnModelSlctChange)
END_EVENT_TABLE()

wxDEFINE_EVENT(asEVT_ACTION_FORECAST_MODEL_SELECTION_CHANGED, wxCommandEvent);


asListBoxModels::asListBoxModels(wxWindow *parent, wxWindowID id, const wxPoint &pos, const wxSize &size, int n, const wxString choices[], long style)
:
wxListBox(parent, id, pos, size, n, choices, style)
{
    //ctor
}

asListBoxModels::~asListBoxModels()
{
    //dtor
}

bool asListBoxModels::Add(const wxString &modelName, const wxString &leadTimeOriginStr, DataParameter dataParameter, DataTemporalResolution dataTemporalResolution)
{
    wxString newOption = wxString::Format("%d. ", (int)GetStrings().GetCount()+1) + modelName + " (" + leadTimeOriginStr + ") ";
    /*
    switch (dataParameter)
    {
        case (Precipitation):
            break;
        case (AirTemperature):
            break;
        case (Wind):
            break;
        case (Lightnings):
            break;
        default:

    }*/

    Append(newOption);

    return true;
}

void asListBoxModels::OnModelSlctChange( wxCommandEvent & event )
{
    wxCommandEvent eventSlct (asEVT_ACTION_FORECAST_MODEL_SELECTION_CHANGED);
    eventSlct.SetInt(event.GetInt());
    GetParent()->ProcessWindowEvent(eventSlct);
}
