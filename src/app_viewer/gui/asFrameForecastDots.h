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
