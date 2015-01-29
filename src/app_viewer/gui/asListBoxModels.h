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
 
#ifndef ASLISTBOXMODELS_H
#define ASLISTBOXMODELS_H

#include <wx/listbox.h>

#include "asIncludes.h"

class asForecastViewer;

class asListBoxModels : public wxListBox
{
public:
    asListBoxModels(wxWindow *parent, wxWindowID id, const wxPoint &pos = wxDefaultPosition, const wxSize &size = wxDefaultSize, int n = 0, const wxString choices[] = NULL, long style = 0);
    virtual ~asListBoxModels();
    bool Add(const wxString &methodId, const wxString &methodIdDisplay, const wxString &specificTag, const wxString &specificTagDisplay, DataParameter dataParameter, DataTemporalResolution dataTemporalResolution);

protected:

private:
    void OnModelSlctChange( wxCommandEvent & event );

    DECLARE_EVENT_TABLE();
};

#endif // ASLISTBOXMODELS_H
