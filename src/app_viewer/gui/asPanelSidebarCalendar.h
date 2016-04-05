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
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#ifndef __asPanelSidebarCalendar__
#define __asPanelSidebarCalendar__

#include "asPanelSidebar.h"

#include "asIncludes.h"
#include <wx/calctrl.h>

class asPanelSidebarCalendar
        : public asPanelSidebar
{
public:
    asPanelSidebarCalendar(wxWindow *parent, wxWindowID id = wxID_ANY, const wxPoint &pos = wxDefaultPosition,
                           const wxSize &size = wxDefaultSize, long style = wxTAB_TRAVERSAL);

    ~asPanelSidebarCalendar();

    void OnSetPresentDate(wxCommandEvent &event);

    void SetPresentDate();

    wxCalendarCtrl *GetCalendar() const
    {
        return m_calendarForecastDate;
    }

    wxDateTime GetDate() const
    {
        return m_calendarForecastDate->GetDate();
    }

    double GetHour() const
    {
        wxString forecastHourStr = m_textCtrlForecastHour->GetValue();
        double forecastHour = 0;
        forecastHourStr.ToDouble(&forecastHour);
        return forecastHour;
    }

private:
    wxCalendarCtrl *m_calendarForecastDate;
    wxStaticText *m_staticTextForecastHour;
    wxTextCtrl *m_textCtrlForecastHour;
    wxBitmapButton *m_bpButtonNow;
};

#endif // __asPanelSidebar__
