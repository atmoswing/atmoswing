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
 
#ifndef __asPanelSidebarCalendar__
#define __asPanelSidebarCalendar__

#include "asPanelSidebar.h"

#include "asIncludes.h"
#include <wx/calctrl.h>

/** Implementing asPanelSidebarCalendar */
class asPanelSidebarCalendar : public asPanelSidebar
{
public:
    /** Constructor */
    asPanelSidebarCalendar( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~asPanelSidebarCalendar();

    void OnSetPresentDate( wxCommandEvent& event );
    void SetPresentDate();

    wxCalendarCtrl* GetCalendar()
    {
        return m_CalendarForecastDate;
    }

    wxDateTime GetDate()
    {
        return m_CalendarForecastDate->GetDate();
    }

    double GetHour()
    {
        wxString forecastHourStr = m_TextCtrlForecastHour->GetValue();
        double forecastHour = 0;
        forecastHourStr.ToDouble(&forecastHour);
        return forecastHour;
    }

private:
    wxCalendarCtrl* m_CalendarForecastDate;
    wxStaticText* m_StaticTextForecastHour;
    wxTextCtrl* m_TextCtrlForecastHour;
    wxBitmapButton* m_BpButtonNow;
};

#endif // __asPanelSidebar__
