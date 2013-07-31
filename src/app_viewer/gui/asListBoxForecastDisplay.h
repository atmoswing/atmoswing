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
 
#ifndef ASLISTBOXFORECASTDISPLAY_H
#define ASLISTBOXFORECASTDISPLAY_H

#include <wx/listbox.h>

class asForecastViewer;

class asListBoxForecastDisplay : public wxListBox
{
public:
    asListBoxForecastDisplay(wxWindow *parent, wxWindowID id, const wxPoint &pos = wxDefaultPosition, const wxSize &size = wxDefaultSize, int n = 0, const wxString choices[] = NULL, long style = 0);
    virtual ~asListBoxForecastDisplay();

    void SetStringArray(wxArrayString options)
    {
        Set(options);
        SetSelection(3);
    }

protected:

private:
    void OnForecastDisplaySlctChange( wxCommandEvent & event );

    DECLARE_EVENT_TABLE();
};

#endif // ASLISTBOXFORECASTDISPLAY_H
