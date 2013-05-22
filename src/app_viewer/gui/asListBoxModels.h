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
 
#ifndef ASLISTBOXMODELS_H
#define ASLISTBOXMODELS_H

#include <wx/listbox.h>

class asForecastViewer;

class asListBoxModels : public wxListBox
{
public:
    asListBoxModels(wxWindow *parent, wxWindowID id, const wxPoint &pos = wxDefaultPosition, const wxSize &size = wxDefaultSize, int n = 0, const wxString choices[] = NULL, long style = 0);
    virtual ~asListBoxModels();
    bool Add(const wxString &modelName, const wxString &leadTimeOriginStr);

protected:

private:
    void OnModelSlctChange( wxCommandEvent & event );

    DECLARE_EVENT_TABLE();
};

#endif // ASLISTBOXMODELS_H
