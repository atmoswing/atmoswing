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
 
#ifndef __asPanelSidebarAnalogDates__
#define __asPanelSidebarAnalogDates__

#include "asPanelSidebar.h"

#include "asIncludes.h"
#include <wx/listctrl.h>

/** Implementing asPanelSidebarAnalogDates */
class asPanelSidebarAnalogDates : public asPanelSidebar
{
public:
    /** Constructor */
    asPanelSidebarAnalogDates( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~asPanelSidebarAnalogDates();

    void SetChoices(Array1DFloat &arrayDate, Array1DFloat &arrayCriteria);

private:
    wxListCtrl* m_ListCtrl;

    void OnDateSelection( wxListEvent& event );

};

#endif // __asPanelSidebarAnalogDates__
