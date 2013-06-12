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
 
#ifndef __asFrameForecastDots__
#define __asFrameForecastDots__

#include "asFrameForecast.h"
#include "asPanelSidebarAnalogDates.h"

class asFrameForecastRings;

/** Implementing asFrameForecastDots */
class asFrameForecastDots : public asFrameForecast
{
public:
    /** Constructor */
    asFrameForecastDots( wxWindow* parent, asFrameForecastRings* frameRings, wxWindowID id=asWINDOW_VIEWER_DOTS );
    ~asFrameForecastDots();

    void OnInit();
    void OnQuit( wxCommandEvent& event );


protected:
    asPanelSidebarCaptionForecastDots *m_PanelSidebarCaptionForecastDots;
    asPanelSidebarAnalogDates *m_PanelSidebarAnalogDates;

private:
    wxSlider* m_SliderLeadTime;
    wxStaticText* m_StaticTextLeadTime;
    asFrameForecastRings* m_FrameRings;

    void OnForecastNewAdded( wxCommandEvent& event );
    void OnForecastRatioSelectionChange( wxCommandEvent& event );
    void OnForecastModelSelectionChange( wxCommandEvent& event );
    void OnForecastPercentileSelectionChange( wxCommandEvent& event );
    void OnLeadtimeChange(wxScrollEvent &event);
    void OnForecastSelectionChange( wxCommandEvent& event );
    void UpdateHeaderTexts();
    void UpdateLeadTimeSlider();
    void UpdatePanelCaption();
    void UpdatePanelAnalogDates();
    void UpdatePanelStationsList();

    DECLARE_EVENT_TABLE()

};

#endif // __asFrameForecastDots__
