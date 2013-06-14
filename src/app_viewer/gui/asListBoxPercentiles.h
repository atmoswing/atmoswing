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
 
#ifndef ASLISTBOXPERCENTILES_H
#define ASLISTBOXPERCENTILES_H

#include <wx/listbox.h>

class asForecastViewer;

class asListBoxPercentiles : public wxListBox
{
public:
    asListBoxPercentiles(wxWindow *parent, wxWindowID id, const wxPoint &pos = wxDefaultPosition, const wxSize &size = wxDefaultSize, int n = 0, const wxString choices[] = NULL, long style = 0);
    virtual ~asListBoxPercentiles();

    void SetStringArray(wxArrayString options)
    {
        Set(options);
        SetSelection(1);
    }

protected:

private:
    void OnPercentileSlctChange( wxCommandEvent & event );

    DECLARE_EVENT_TABLE();
};

#endif // ASLISTBOXPERCENTILES_H
