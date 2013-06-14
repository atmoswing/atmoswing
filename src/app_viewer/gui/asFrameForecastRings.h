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
 
#ifndef __asFrameForecastRings__
#define __asFrameForecastRings__

#include "asFrameForecast.h"

class asFrameForecastDots;

/** Implementing asFrameForecastRings */
class asFrameForecastRings : public asFrameForecast
{
public:
    /** Constructor */
    asFrameForecastRings( wxWindow* parent, wxWindowID id=asWINDOW_VIEWER_RINGS );
    ~asFrameForecastRings();

    void OnInit();
    void OnQuit( wxCommandEvent& event );
    void NullFrameDotsPointer();


protected:
    asPanelSidebarCaptionForecastRing *m_PanelSidebarCaptionForecastRing;
    asPanelSidebarAlarms *m_PanelSidebarAlarms;


private:
    asFrameForecastDots* m_FrameDots;

    void OpenFrameDots( wxCommandEvent& event );
    void OnForecastNewAdded( wxCommandEvent& event );
    void OnForecastRatioSelectionChange( wxCommandEvent& event );
    void OnForecastModelSelectionChange( wxCommandEvent& event );
    void OnForecastPercentileSelectionChange( wxCommandEvent& event );
    void OnForecastSelectionChange( wxCommandEvent& event );
    void UpdateHeaderTexts();
    void UpdatePanelCaptionAll();
    void UpdatePanelCaptionColorbar();
    void UpdatePanelStationsList();
    void UpdatePanelAlarms();

    DECLARE_EVENT_TABLE()

};

#endif // __asFrameForecastRings__
