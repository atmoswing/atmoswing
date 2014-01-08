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
